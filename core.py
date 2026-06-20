"""
PulseGrid - core engine
Event-impact forecasting + deployment recommendation for Bengaluru Traffic Police (ASTraM data).

All ML runs on scikit-learn only (HistGradientBoosting) so the app deploys with zero
heavy dependencies. Models train in a few seconds on ~8k rows and are cached by the app.
"""
import os
import re
import glob
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import (
    mean_absolute_error, r2_score, roc_auc_score, f1_score, average_precision_score,
    brier_score_loss,
)

HERE = os.path.dirname(os.path.abspath(__file__))

_DATA_CANDIDATES = [
    os.path.join(HERE, "data", "astram_events.csv"),
    os.path.join(HERE, "astram_events.csv"),
    os.path.join(os.getcwd(), "data", "astram_events.csv"),
    os.path.join(os.getcwd(), "astram_events.csv"),
]


def _find_data():
    for c in _DATA_CANDIDATES:
        if os.path.exists(c):
            return c
    # fallback: locate any CSV in the repo (handles renamed/original-named exports)
    for d in (HERE, os.path.join(HERE, "data"), os.getcwd(),
              os.path.join(os.getcwd(), "data")):
        if not os.path.isdir(d):
            continue
        hits = sorted(glob.glob(os.path.join(d, "*.csv")))
        pref = [h for h in hits
                if re.search(r"astram|event", os.path.basename(h), re.I)]
        if pref:
            return pref[0]
        if hits:
            return hits[0]
    raise FileNotFoundError(
        "No incident CSV found in the repo. Upload the ASTraM export (any .csv "
        "name works) to the repo root or a data/ folder. Looked in: "
        + " | ".join(_DATA_CANDIDATES))


DATA_PATH = _DATA_CANDIDATES[0]

CAT_COLS = ["event_type", "event_cause", "veh_type", "corridor", "police_station"]
NUM_COLS = ["lat", "lon", "hour", "dow", "month", "is_weekend", "is_night"]

# semantic severity weight used in the allocation optimiser
SEV_WEIGHT = {"High": 2.0, "Low": 1.0}


# ----------------------------------------------------------------------------- data
def load_and_prepare(path: str = None) -> pd.DataFrame:
    df = pd.read_csv(path or _find_data())
    sd = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True)
    rd = pd.to_datetime(df["resolved_datetime"], errors="coerce", utc=True)
    cd = pd.to_datetime(df["closed_datetime"], errors="coerce", utc=True)
    end = rd.fillna(cd)

    df["start_ts"] = sd.dt.tz_convert(None)
    df["duration_min"] = (end - sd).dt.total_seconds() / 60.0
    df["hour"] = sd.dt.hour
    df["dow"] = sd.dt.dayofweek
    df["month"] = sd.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)

    for c in CAT_COLS + ["priority"]:
        df[c] = df[c].astype("string").fillna("unknown")
    df["road_closure"] = df["requires_road_closure"].astype(int)
    df["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
    # keep plausible Bengaluru coords for the map
    df["geo_ok"] = df["lat"].between(12.6, 13.3) & df["lon"].between(77.3, 77.9)
    return df


def _pipe(estimator):
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
         ("num", "passthrough", NUM_COLS)]
    )
    return Pipeline([("pre", pre), ("model", estimator)])


def _frame(d: pd.DataFrame) -> pd.DataFrame:
    return d[CAT_COLS + NUM_COLS].copy()


def _haversine_km(a_lat, a_lon, b_lat, b_lon):
    R = 6371.0
    p1, p2 = np.radians(a_lat), np.radians(b_lat)
    dphi, dl = np.radians(b_lat - a_lat), np.radians(b_lon - a_lon)
    h = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(h)))


def nearest_alt(corridor, centroids, k=1):
    """Nearest parallel corridor(s) by geography — a concrete diversion route."""
    if centroids is None or corridor not in centroids.index:
        return None
    here = centroids.loc[corridor]
    bad = {corridor, "Non-corridor", "unknown", "Others", "others"}
    others = centroids.drop(index=[i for i in centroids.index if i in bad])
    if not len(others):
        return None
    d = others.apply(lambda r: _haversine_km(here["lat"], here["lon"], r["lat"], r["lon"]), axis=1)
    nb = d.nsmallest(k).index.tolist()
    return nb[0] if nb else None


# --------------------------------------------------------------------------- models
def train_models(df: pd.DataFrame, with_cv: bool = True) -> dict:
    out = {"metrics": {}}

    # 1) clearance-time — calibrated quantile band (P50 + P90), intake features only
    d = df[df["duration_min"].between(1, 60 * 24 * 7)].copy()
    Xd = _frame(d)
    yd = d["duration_min"].to_numpy()
    yd_log = np.log1p(yd)
    qkw = dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
               l2_regularization=1.0, random_state=0)
    dur50 = _pipe(HistGradientBoostingRegressor(loss="quantile", quantile=0.5, **qkw))
    dur90 = _pipe(HistGradientBoostingRegressor(loss="quantile", quantile=0.9, **qkw))
    if with_cv:
        kf = KFold(5, shuffle=True, random_state=0)
        p50 = np.clip(np.expm1(cross_val_predict(dur50, Xd, yd_log, cv=kf)), 0, None)
        p90 = np.clip(np.expm1(cross_val_predict(dur90, Xd, yd_log, cv=kf)), 0, None)
        # conformal scaling so the band has ~90% empirical coverage
        cf = float(np.quantile(yd / np.maximum(p90, 1.0), 0.90))
        out["duration_cf"] = cf
        out["metrics"]["duration"] = {
            "n": len(d), "MAE_min": float(mean_absolute_error(yd, p50)),
            "median_AE_min": float(np.median(np.abs(yd - p50))),
            "p90_coverage_raw": float(np.mean(yd <= np.maximum(p90, p50))),
            "p90_coverage_cal": float(np.mean(yd <= np.maximum(cf * p90, p50)))}
    else:
        out["duration_cf"] = 1.0
    dur50.fit(Xd, yd_log)
    dur90.fit(Xd, yd_log)
    out["duration"] = dur50
    out["duration_p90"] = dur90

    # 2) priority classifier (High vs Low) -- intake features only
    p_ = df[df["priority"].isin(["High", "Low"])].copy()
    Xp, yp = _frame(p_), (p_["priority"] == "High").astype(int).values
    pri = _pipe(HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31, random_state=0))
    if with_cv:
        pr = cross_val_predict(pri, Xp, yp, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                               method="predict_proba")[:, 1]
        out["metrics"]["priority"] = {
            "n": len(p_), "AUC": float(roc_auc_score(yp, pr)),
            "F1": float(f1_score(yp, (pr > 0.5).astype(int))),
            "base_high": float(yp.mean())}
    pri.fit(Xp, yp)
    out["priority"] = pri

    # 3) road-closure classifier -- baseline features, with calibration + operating points
    yc = df["road_closure"].to_numpy()
    Xc = _frame(df)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    clo = _pipe(HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                max_leaf_nodes=31, class_weight="balanced", random_state=0))
    if with_cv:
        cr = cross_val_predict(clo, Xc, yc, cv=skf, method="predict_proba")[:, 1]
        ops = {}
        for thr in (0.20, 0.30, 0.40, 0.50):
            fl = cr >= thr
            ops[f"{thr:.2f}"] = {
                "recall": float(yc[fl].sum() / max(1, yc.sum())),
                "precision": float(yc[fl].sum() / max(1, fl.sum())),
                "flag_rate": float(fl.mean())}
        out["metrics"]["closure"] = {
            "n": len(df), "AUC": float(roc_auc_score(yc, cr)),
            "PR_AUC": float(average_precision_score(yc, cr)),
            "base_rate": float(yc.mean()),
            "brier": float(brier_score_loss(yc, cr)),
            "recall_at_30": ops["0.30"]["recall"],
            "flag_rate_at_30": ops["0.30"]["flag_rate"],
            "ops": ops}
    clo.fit(Xc, yc)
    out["closure"] = clo

    # 4) daily-volume forecaster
    daily = (df.dropna(subset=["start_ts"]).assign(date=lambda x: x["start_ts"].dt.normalize())
             .groupby("date").size().rename("n").reset_index().sort_values("date"))
    daily["dow"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["lag7"] = daily["n"].shift(7)
    daily["roll7"] = daily["n"].shift(1).rolling(7).mean()
    dd = daily.dropna()
    vol = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
    feats = ["dow", "month", "lag7", "roll7"]
    if with_cv and len(dd) > 30:
        vp = cross_val_predict(vol, dd[feats], dd["n"].values, cv=KFold(5, shuffle=True, random_state=0))
        out["metrics"]["volume"] = {
            "days": len(dd), "MAE": float(mean_absolute_error(dd["n"], vp)),
            "naive_MAE": float(mean_absolute_error(dd["n"], dd["lag7"])),
            "mean_per_day": float(dd["n"].mean())}
    vol.fit(dd[feats], dd["n"].values)
    out["volume"] = vol
    out["daily"] = daily
    out["centroids"] = (df[df["geo_ok"]].groupby("corridor")[["lat", "lon"]].median()
                        if "geo_ok" in df.columns else None)
    return out


# ------------------------------------------------------------------ live prediction
def predict_event(models: dict, ev: dict) -> dict:
    row = {c: ev.get(c, "unknown") for c in CAT_COLS}
    for c in NUM_COLS:
        row[c] = ev.get(c, 0)
    row["junction"] = ev.get("junction", "unknown")
    base = pd.DataFrame([row])
    X = base[CAT_COLS + NUM_COLS]
    p50 = float(np.clip(np.expm1(models["duration"].predict(X)[0]), 1, None))
    p90 = float(np.clip(np.expm1(models["duration_p90"].predict(X)[0]), p50, None))
    p90 *= models.get("duration_cf", 1.0)
    p90 = max(p90, p50)
    pri_p = float(models["priority"].predict_proba(X)[0, 1])
    clo_p = float(models["closure"].predict_proba(X)[0, 1])
    return {
        "duration_min": p50,
        "duration_p90": p90,
        "clearance_interval": f"{p50:.0f}–{p90:.0f} min",
        "clearance_band": ("Quick (<30 min)" if p50 < 30 else
                           "Moderate (30 min–2 hr)" if p50 < 120 else
                           "Extended (>2 hr)"),
        "priority": "High" if pri_p >= 0.5 else "Low",
        "priority_proba": pri_p,
        "closure_proba": clo_p,
    }


def recommend_deployment(pred: dict, corridor: str = "", centroids=None) -> dict:
    dur, clo, is_high = pred["duration_min"], pred["closure_proba"], pred["priority"] == "High"
    if clo >= 0.50:
        barricade = "Deploy barricades"
    elif clo >= 0.25:
        barricade = "Barricades on standby"
    else:
        barricade = "Not required"
    units = 1 + int(is_high) + int(dur > 120) + int(clo >= 0.50)
    units = min(units, 4)
    if clo >= 0.50 and corridor and corridor not in ("Non-corridor", "unknown"):
        alt = nearest_alt(corridor, centroids)
        diversion = (f"Divert via {alt} (nearest parallel corridor)" if alt
                     else f"Set up diversion around {corridor}")
    else:
        diversion = "—"
    score = 2 * int(is_high) + int(dur > 120) + int(dur > 360) + 2 * int(clo >= 0.50)
    level = ("Critical" if score >= 5 else "High" if score >= 3
             else "Moderate" if score >= 1 else "Low")
    return {"response_level": level, "officers": units,
            "barricade": barricade, "diversion": diversion}


# --------------------------------------------------------------------- forecasting
def forecast_daily(models: dict, horizon: int = 7) -> pd.DataFrame:
    daily = models["daily"].copy()
    hist_n = list(daily["n"].values)
    last = daily["date"].max()
    feats = ["dow", "month", "lag7", "roll7"]
    rows = []
    for i in range(1, horizon + 1):
        date = last + pd.Timedelta(days=i)
        lag7 = hist_n[-7] if len(hist_n) >= 7 else np.mean(hist_n)
        roll7 = np.mean(hist_n[-7:])
        x = pd.DataFrame([{"dow": date.dayofweek, "month": date.month, "lag7": lag7, "roll7": roll7}])[feats]
        yhat = float(max(0, models["volume"].predict(x)[0]))
        rows.append({"date": date, "forecast": round(yhat)})
        hist_n.append(yhat)
    return pd.DataFrame(rows)


def corridor_load(df: pd.DataFrame, dow: int) -> pd.DataFrame:
    """Expected per-corridor load for a given day-of-week (count x mean duration x severity)."""
    d = df[df["dow"] == dow].copy()
    d = d[d["corridor"] != "unknown"]
    n_weeks = max(1, df["start_ts"].dt.isocalendar().week.nunique())
    g = d.groupby("corridor").agg(
        events=("id", "count"),
        avg_dur=("duration_min", lambda s: np.nanmedian(s) if s.notna().any() else 45.0),
        high_share=("priority", lambda s: (s == "High").mean()),
    ).reset_index()
    g["avg_dur"] = g["avg_dur"].clip(15, 600)   # tame multi-day outliers
    g["events_per_day"] = g["events"] / n_weeks
    g["sev"] = 1.0 + g["high_share"]            # 1.0–2.0
    g["load"] = g["events_per_day"] * g["avg_dur"] * g["sev"]
    return g.sort_values("load", ascending=False)


def allocate_officers(load_df: pd.DataFrame, total_officers: int) -> pd.DataFrame:
    g = load_df[load_df["load"] > 0].copy()
    if g.empty:
        return g
    g = g.head(min(len(g), 15, max(1, total_officers))).copy()
    share = g["load"] / g["load"].sum()
    raw = share * total_officers
    g["officers"] = np.maximum(1, np.round(raw)).astype(int)
    # trim/pad to hit the exact total
    diff = total_officers - int(g["officers"].sum())
    order = g["load"].rank(ascending=False).astype(int).values
    idx = np.argsort(-g["load"].values)
    k = 0
    while diff != 0 and k < 1000:
        j = idx[k % len(idx)]
        if diff > 0:
            g.iloc[j, g.columns.get_loc("officers")] += 1; diff -= 1
        elif g.iloc[j]["officers"] > 1:
            g.iloc[j, g.columns.get_loc("officers")] -= 1; diff += 1
        k += 1
    return g[["corridor", "events_per_day", "avg_dur", "high_share", "officers"]]


def hotspots(df: pd.DataFrame, top: int = 12) -> pd.DataFrame:
    d = df[df["junction"].notna() & (df["junction"].astype(str) != "nan")].copy()
    g = d.groupby("junction").agg(
        events=("id", "count"),
        lat=("lat", "median"), lon=("lon", "median"),
        high_share=("priority", lambda s: (s == "High").mean()),
        closure_rate=("road_closure", "mean"),
    ).reset_index().sort_values("events", ascending=False)
    return g.head(top)


PLANNED_CATEGORIES = {
    "Festival / public event": "public_event", "Political rally": "procession",
    "Sports event": "public_event", "VIP movement": "vip_movement",
    "Procession": "procession", "Construction / roadwork": "construction"}
CROWD_LEVELS = {"Small (<5k)": (1.0, 4), "Medium (5k-25k)": (1.6, 8),
                "Large (25k-75k)": (2.3, 16), "Massive (>75k)": (3.0, 28)}


def plan_event(models, category, corridor, crowd, hour, dow, centroid):
    """Forecast impact + pre-deployment plan for a PLANNED event."""
    cause = PLANNED_CATEGORIES.get(category, "public_event")
    lat, lon = centroid
    ev = {"event_type": "planned", "event_cause": cause, "veh_type": "unknown",
          "corridor": corridor, "police_station": "unknown", "lat": lat, "lon": lon,
          "hour": hour, "dow": dow, "month": 3, "is_weekend": int(dow >= 5),
          "is_night": int(hour in [0, 1, 2, 3, 4, 5, 22, 23])}
    pred = predict_event(models, ev)
    mult, base_off = CROWD_LEVELS[crowd]
    duration = pred["duration_min"] * mult
    officers = int(round(base_off * (1.3 if pred["priority"] == "High" else 1.0)))
    closure_risk = float(min(0.95, pred["closure_proba"] * mult))
    barricade = closure_risk >= 0.40 or crowd.startswith(("Large", "Massive"))
    return {
        "category": category, "corridor": corridor, "crowd": crowd,
        "expected_duration_min": duration, "priority": pred["priority"],
        "closure_risk": closure_risk, "officers": officers, "barricade": barricade,
        "advisory_window": f"deploy T-60min -> T+{duration/60:.1f}h",
        "diversion": (f"Pre-stage diversion around {corridor}"
                      if barricade and corridor not in ("Non-corridor", "unknown") else "-")}


def risk_factors(df, ev):
    """Plain-language, data-driven reasons behind a prediction (explainability)."""
    out, base = [], df["road_closure"].mean()
    s = df[df["event_cause"] == ev["event_cause"]]
    if len(s) > 20:
        out.append(f"Cause '{ev['event_cause']}': {s['road_closure'].mean():.0%} historically "
                   f"need closure, ~{np.nanmedian(s['duration_min']):.0f} min median clearance")
    s2 = df[df["corridor"] == ev["corridor"]]
    if len(s2) > 20:
        out.append(f"Corridor '{ev['corridor']}': "
                   f"{(s2['priority'] == 'High').mean():.0%} of events run High priority")
    s3 = df[df["hour"] == ev["hour"]]
    if len(s3) > 50 and s3["road_closure"].mean() > base * 1.25:
        out.append(f"Time {ev['hour']:02d}:00 sits above the city's average closure risk")
    return out[:3]


def corridor_hourly(df, corridor):
    s = df[df["corridor"] == corridor]
    return (s.groupby("hour").size().reindex(range(24), fill_value=0)
            .rename("events").reset_index())


def surge_days(models, fc):
    mu, sd = models["daily"]["n"].mean(), models["daily"]["n"].std()
    fc = fc.copy()
    fc["surge"] = fc["forecast"] > (mu + sd)
    return fc, float(mu), float(sd)


if __name__ == "__main__":
    import json
    df = load_and_prepare()
    print("rows:", len(df), "| geo_ok:", int(df["geo_ok"].sum()))
    m = train_models(df, with_cv=True)
    print("\nMETRICS:"); print(json.dumps(m["metrics"], indent=2))

    ev = {"event_type": "unplanned", "event_cause": "accident", "veh_type": "heavy_vehicle",
          "corridor": "Hosur Road", "police_station": "HSR Layout",
          "lat": 12.92, "lon": 77.64, "hour": 18, "dow": 1, "month": 3,
          "is_weekend": 0, "is_night": 0}
    pred = predict_event(m, ev)
    rec = recommend_deployment(pred, ev["corridor"])
    print("\nSAMPLE EVENT -> PREDICTION:"); print(json.dumps(pred, indent=2))
    print("SAMPLE EVENT -> RECOMMENDATION:"); print(json.dumps(rec, indent=2))

    print("\nFORECAST (next 7 days):")
    print(forecast_daily(m, 7).to_string(index=False))
    print("\nSTAFFING (Tue, 40 officers):")
    load = corridor_load(df, dow=1)
    print(allocate_officers(load, 40).round(1).to_string(index=False))
    print("\nHOTSPOTS:")
    print(hotspots(df).round(3).to_string(index=False))
