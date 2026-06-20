"""
PulseGrid live-operations engine.

A lightweight discrete-event simulation that replays Bengaluru traffic in
accelerated time. New incidents are sampled from the REAL ASTraM arrival
patterns (per-hour Poisson rates + empirical cause -> corridor -> vehicle
distributions); every incident is scored by the trained ML models for
clearance time, priority and road-closure risk; officers are committed from a
finite pool and released as events clear. Pure-Python state, fully headless
testable — Streamlit just renders whatever `step` returns.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import core

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# --------------------------------------------------------------------------- #
#  Profile: empirical distributions learned once from the real data
# --------------------------------------------------------------------------- #
def build_profile(df: pd.DataFrame) -> dict:
    g = df[df["geo_ok"]]
    centroids = g.groupby("corridor")[["lat", "lon"]].median()

    # arrivals per clock-hour, averaged over the number of distinct days logged
    n_days = max(1, df["start_ts"].dt.normalize().nunique())
    by_hour = df.groupby("hour").size().reindex(range(24), fill_value=0) / n_days
    hourly_lambda = by_hour.to_dict()

    causes = df["event_cause"].value_counts(normalize=True)
    # per-cause corridor + vehicle tables (fall back to global if sparse)
    corr_by_cause, veh_by_cause = {}, {}
    glob_corr = df["corridor"].value_counts(normalize=True)
    glob_veh = df[df["veh_type"] != "unknown"]["veh_type"].value_counts(normalize=True)
    for c in causes.index:
        sub = df[df["event_cause"] == c]
        cc = sub["corridor"].value_counts(normalize=True)
        corr_by_cause[c] = cc if len(cc) else glob_corr
        vv = sub[sub["veh_type"] != "unknown"]["veh_type"].value_counts(normalize=True)
        veh_by_cause[c] = vv if len(vv) else glob_veh

    # station lookup: most common station per corridor
    stn = (df.groupby("corridor")["police_station"]
             .agg(lambda s: s.mode().iat[0] if len(s.mode()) else "unknown").to_dict())

    return {
        "centroids": centroids, "hourly_lambda": hourly_lambda,
        "causes": causes, "corr_by_cause": corr_by_cause, "veh_by_cause": veh_by_cause,
        "station": stn, "global_corr": glob_corr,
        "officers_total": 120, "pressure_cap": _calibrate_cap(df),
    }


def _calibrate_cap(df):
    """Pressure cap = 90th-percentile of real concurrent severity-weighted load."""
    d = df.dropna(subset=["start_ts"]).copy()
    d = d[d["duration_min"].between(1, 60 * 24)]
    if not len(d):
        return 16.0
    w = 1.0 + 2.0 * d["road_closure"].to_numpy() + (d["priority"] == "High").to_numpy().astype(float)
    end = d["start_ts"] + pd.to_timedelta(d["duration_min"].to_numpy(), unit="m")
    ev = pd.concat([
        pd.DataFrame({"t": d["start_ts"].to_numpy(), "delta": w}),
        pd.DataFrame({"t": end.to_numpy(), "delta": -w})]).sort_values("t")
    load = ev["delta"].cumsum().to_numpy()
    return float(max(8.0, np.percentile(load, 90)))


def arrival_fidelity(profile, days=21, seed=3):
    """Does the sampled stream reproduce the real hour-of-day incidence? (faithfulness check)"""
    rng = np.random.default_rng(seed)
    counts = np.zeros(24)
    cur = 6 * 60
    for _ in range(0, days * 1440, 5):
        hour = (cur // 60) % 24
        lam = profile["hourly_lambda"].get(int(hour), 1.0) * (5 / 60.0)
        counts[hour] += rng.poisson(max(lam, 0.01))
        cur = (cur + 5) % 1440
    sim_h = counts / max(1, counts.sum())
    real = np.array([profile["hourly_lambda"].get(h, 0.0) for h in range(24)])
    real = real / max(1e-9, real.sum())
    corr = float(np.corrcoef(sim_h, real)[0, 1])
    mae = float(np.mean(np.abs(sim_h - real)))
    return {"corr": corr, "mae": mae}


def _pick(series: pd.Series, rng: np.random.Generator):
    return rng.choice(series.index.to_numpy(), p=series.to_numpy())


def sample_event(profile: dict, rng: np.random.Generator, hour: int, dow: int) -> dict:
    cause = _pick(profile["causes"], rng)
    corridor = _pick(profile["corr_by_cause"][cause], rng)
    veh_tbl = profile["veh_by_cause"].get(cause)
    veh = _pick(veh_tbl, rng) if veh_tbl is not None and len(veh_tbl) else "unknown"
    if corridor in profile["centroids"].index:
        lat = float(profile["centroids"].loc[corridor, "lat"]) + rng.normal(0, 0.006)
        lon = float(profile["centroids"].loc[corridor, "lon"]) + rng.normal(0, 0.006)
    else:
        lat, lon = 12.97, 77.59
    return {
        "event_type": "unplanned", "event_cause": cause, "veh_type": veh,
        "corridor": corridor, "police_station": profile["station"].get(corridor, "unknown"),
        "lat": lat, "lon": lon, "hour": int(hour), "dow": int(dow), "month": 3,
        "is_weekend": int(dow >= 5), "is_night": int(hour in [0, 1, 2, 3, 4, 5, 22, 23]),
    }


# --------------------------------------------------------------------------- #
#  Severity helpers
# --------------------------------------------------------------------------- #
def officers_for(pred: dict, veh: str) -> int:
    if pred["closure_proba"] >= 0.40:
        n = 6
    elif pred["priority"] == "High":
        n = 3
    else:
        n = 2
    if veh in ("heavy_vehicle", "lcv"):
        n += 1
    return int(min(8, n))


def _sev_weight(ev: dict) -> float:
    return 1.0 + (2.0 if ev["closure"] else 0.0) + (1.0 if ev["priority"] == "High" else 0.0)


def _haversine(a_lat, a_lon, b_lat, b_lon) -> float:
    R = 6371.0
    p1, p2 = np.radians(a_lat), np.radians(b_lat)
    dphi = np.radians(b_lat - a_lat)
    dlmb = np.radians(b_lon - a_lon)
    h = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(h)))


# --------------------------------------------------------------------------- #
#  Simulation state machine
# --------------------------------------------------------------------------- #
def init_state(profile: dict, start_hour: int = 7, dow: int = 1,
               officers: int | None = None, seed: int = 7) -> dict:
    return {
        "clock": start_hour * 60, "day": 0, "dow": dow,
        "officers_total": officers or profile["officers_total"],
        "active": [], "backlog": [], "log": [],
        "spawned": 0, "cleared": 0, "queued_peak": 0, "uid": 0,
        "pressure": 0.0, "pressure_hist": [], "seed": seed,
        "_rng": np.random.default_rng(seed),
    }


def _deployed(state: dict) -> int:
    return sum(e["officers"] for e in state["active"])


def _try_dispatch(state: dict, profile: dict):
    """Move backlog -> active while officers are free (highest severity first)."""
    state["backlog"].sort(key=lambda e: (-_sev_weight(e), e["start"]))
    avail = state["officers_total"] - _deployed(state)
    still = []
    for e in state["backlog"]:
        if e["officers"] <= avail:
            e["status"] = "active"
            state["active"].append(e)
            avail -= e["officers"]
            state["log"].append((_clock_str(state), "DISPATCH",
                                 f"{e['cause']} · {e['corridor']} · {e['officers']} units"))
        else:
            still.append(e)
    state["backlog"] = still


def step(state: dict, profile: dict, models: dict, dt_min: int = 15) -> dict:
    rng = state["_rng"]
    # 1) advance clock
    state["clock"] += dt_min
    while state["clock"] >= 1440:
        state["clock"] -= 1440
        state["day"] += 1
        state["dow"] = (state["dow"] + 1) % 7
    abs_now = state["day"] * 1440 + state["clock"]
    hour = state["clock"] // 60

    # 2) clear finished events -> release officers
    done = [e for e in state["active"] if e["end_abs"] <= abs_now]
    for e in done:
        state["cleared"] += 1
        state["log"].append((_clock_str(state), "CLEARED",
                             f"{e['cause']} · {e['corridor']} · held {e['dur']:.0f} min"))
    state["active"] = [e for e in state["active"] if e["end_abs"] > abs_now]

    # 3) spawn new arrivals (Poisson on the real per-hour rate)
    lam = profile["hourly_lambda"].get(int(hour), 1.0) * (dt_min / 60.0)
    for _ in range(int(rng.poisson(max(lam, 0.01)))):
        ev = sample_event(profile, rng, hour, state["dow"])
        pred = core.predict_event(models, ev)
        state["uid"] += 1
        rec = {
            "id": state["uid"], "cause": ev["event_cause"], "corridor": ev["corridor"],
            "veh": ev["veh_type"], "lat": ev["lat"], "lon": ev["lon"],
            "priority": pred["priority"], "closure": pred["closure_proba"] >= 0.40,
            "closure_p": pred["closure_proba"], "dur": pred["duration_min"],
            "officers": officers_for(pred, ev["veh_type"]),
            "start": abs_now, "end_abs": abs_now + pred["duration_min"], "status": "queued",
        }
        state["spawned"] += 1
        state["backlog"].append(rec)

    # 4) dispatch from pool, recompute pressure + ripple
    _try_dispatch(state, profile)
    state["queued_peak"] = max(state["queued_peak"], len(state["backlog"]))
    load = sum(_sev_weight(e) for e in state["active"]) + 0.5 * len(state["backlog"])
    state["pressure"] = float(np.clip(load / profile["pressure_cap"] * 100, 0, 100))
    state["pressure_hist"].append(state["pressure"])
    state["pressure_hist"] = state["pressure_hist"][-60:]
    return state


def inject(state: dict, profile: dict, models: dict, cause: str, corridor: str) -> dict:
    """Manually drop a specific incident in right now (what-if button)."""
    rng = state["_rng"]
    if corridor in profile["centroids"].index:
        lat = float(profile["centroids"].loc[corridor, "lat"])
        lon = float(profile["centroids"].loc[corridor, "lon"])
    else:
        lat, lon = 12.97, 77.59
    hour = state["clock"] // 60
    ev = {"event_type": "unplanned", "event_cause": cause, "veh_type": "heavy_vehicle",
          "corridor": corridor, "police_station": profile["station"].get(corridor, "unknown"),
          "lat": lat, "lon": lon, "hour": int(hour), "dow": int(state["dow"]), "month": 3,
          "is_weekend": int(state["dow"] >= 5), "is_night": int(hour in [0,1,2,3,4,5,22,23])}
    pred = core.predict_event(models, ev)
    abs_now = state["day"] * 1440 + state["clock"]
    state["uid"] += 1
    state["spawned"] += 1
    state["backlog"].append({
        "id": state["uid"], "cause": cause, "corridor": corridor, "veh": "heavy_vehicle",
        "lat": lat, "lon": lon, "priority": pred["priority"],
        "closure": pred["closure_proba"] >= 0.40, "closure_p": pred["closure_proba"],
        "dur": pred["duration_min"], "officers": officers_for(pred, "heavy_vehicle"),
        "start": abs_now, "end_abs": abs_now + pred["duration_min"], "status": "queued"})
    state["log"].append((_clock_str(state), "INJECT", f"{cause} · {corridor} (manual)"))
    _try_dispatch(state, profile)
    return state


def ripple(state: dict, profile: dict, radius_km: float = 3.5) -> pd.DataFrame:
    """Corridors under congestion pressure radiating from active high-impact events."""
    cent = profile["centroids"]
    stress = {}
    for e in state["active"]:
        if not (e["closure"] or e["priority"] == "High"):
            continue
        w = _sev_weight(e)
        for corr, row in cent.iterrows():
            d = _haversine(e["lat"], e["lon"], row["lat"], row["lon"])
            if d <= radius_km:
                stress[corr] = stress.get(corr, 0.0) + w * (1 - d / radius_km)
    if not stress:
        return pd.DataFrame(columns=["corridor", "stress", "level"])
    out = (pd.DataFrame({"corridor": list(stress), "stress": list(stress.values())})
           .sort_values("stress", ascending=False).reset_index(drop=True))
    out["level"] = pd.cut(out["stress"], [-1, 1.5, 3.5, 1e9],
                          labels=["watch", "elevated", "critical"]).astype(str)
    return out


def _clock_str(state: dict) -> str:
    h, m = divmod(state["clock"], 60)
    return f"D{state['day']+1} {DOW_NAMES[state['dow']]} {h:02d}:{m:02d}"


def clock_label(state: dict) -> str:
    return _clock_str(state)


def active_frame(state: dict) -> pd.DataFrame:
    if not state["active"]:
        return pd.DataFrame(columns=["lat", "lon", "sev", "cause", "corridor"])
    rows = [{"lat": e["lat"], "lon": e["lon"],
             "sev": 3 if e["closure"] else 2 if e["priority"] == "High" else 1,
             "cause": e["cause"], "corridor": e["corridor"], "officers": e["officers"]}
            for e in state["active"]]
    return pd.DataFrame(rows)


def _generate_stream(profile, models, days, start_hour, dow, seed, dt=5, surge_mult=1.0):
    rng = np.random.default_rng(seed)
    events, horizon = [], days * 1440
    cur_clock, cur_dow = start_hour * 60, dow
    for t in range(0, horizon, dt):
        hour = (cur_clock // 60) % 24
        # event surges concentrate in daytime/peak hours, like a real festival or rally
        peak = 1.0 + (surge_mult - 1.0) * (1.0 if 9 <= hour <= 21 else 0.2)
        lam = profile["hourly_lambda"].get(int(hour), 1.0) * (dt / 60.0) * peak
        for _ in range(int(rng.poisson(max(lam, 0.01)))):
            ev = sample_event(profile, rng, hour, cur_dow)
            pred = core.predict_event(models, ev)
            events.append({"arr": t, "dur": pred["duration_min"],
                           "off": officers_for(pred, ev["veh_type"]),
                           "high": pred["priority"] == "High",
                           "closure": pred["closure_proba"] >= 0.40})
        cur_clock += dt
        if cur_clock >= 1440:
            cur_clock -= 1440
            cur_dow = (cur_dow + 1) % 7
    return events, horizon


def _run_policy(events, officers, policy, horizon, dt=5, reserve_frac=0.20):
    import math
    reserve = math.ceil(reserve_frac * officers) if policy == "pulsegrid" else 0
    events = sorted(events, key=lambda e: e["arr"])
    queue, active, ei = [], [], 0
    waits, crit_waits, immediate, total, maxq = [], [], 0, 0, 0
    for t in range(0, horizon, dt):
        active = [e for e in active if e["end"] > t]
        deployed = sum(e["off"] for e in active)
        while ei < len(events) and events[ei]["arr"] < t + dt:
            queue.append(dict(events[ei]))
            ei += 1
        if policy == "pulsegrid":
            crit_waiting = any(e["high"] or e["closure"] for e in queue)
            queue.sort(key=lambda e: (-(2 if e["closure"] else 1 if e["high"] else 0), e["arr"]))
            avail = officers - deployed - (0 if crit_waiting else reserve)
        else:
            queue.sort(key=lambda e: e["arr"])
            avail = officers - deployed
        avail = max(0, avail)
        rest = []
        for e in queue:
            if e["off"] <= avail:
                avail -= e["off"]
                e["end"] = t + e["dur"]
                active.append(e)
                w = max(0, t - e["arr"])
                waits.append(w)
                total += 1
                if e["high"] or e["closure"]:
                    crit_waits.append(w)
                if w == 0:
                    immediate += 1
            else:
                rest.append(e)
        queue = rest
        maxq = max(maxq, len(queue))
    return {
        "avg_wait": float(np.mean(waits)) if waits else 0.0,
        "avg_crit_wait": float(np.mean(crit_waits)) if crit_waits else 0.0,
        "p90_crit_wait": float(np.percentile(crit_waits, 90)) if crit_waits else 0.0,
        "immediate_pct": float(immediate / max(1, total)),
        "served": total, "max_queue": maxq}


def compare_policies(profile, models, days=2, officers=18,
                     start_hour=6, dow=0, seed=11, surge_mult=2.6):
    """Replay one identical event-surge incident stream under reactive vs PulseGrid."""
    events, horizon = _generate_stream(profile, models, days, start_hour, dow, seed,
                                       surge_mult=surge_mult)
    base = _run_policy(events, officers, "reactive", horizon)
    smart = _run_policy(events, officers, "pulsegrid", horizon)

    def _red(a, b):
        return 0.0 if a <= 0 else max(0.0, (a - b) / a)

    return {"events": len(events), "officers": officers, "days": days,
            "reactive": base, "pulsegrid": smart,
            "crit_wait_reduction": _red(base["avg_crit_wait"], smart["avg_crit_wait"]),
            "p90_reduction": _red(base["p90_crit_wait"], smart["p90_crit_wait"]),
            "immediate_gain": smart["immediate_pct"] - base["immediate_pct"]}


if __name__ == "__main__":
    df = core.load_and_prepare()
    M = core.train_models(df, with_cv=False)
    prof = build_profile(df)
    st = init_state(prof, start_hour=7, dow=1)
    for _ in range(96):  # a full 24h day at 15-min ticks
        step(st, prof, M, 15)
        assert _deployed(st) <= st["officers_total"], "officer overcommit!"
        assert 0 <= st["pressure"] <= 100
    print(f"clock {clock_label(st)} | spawned {st['spawned']} cleared {st['cleared']} "
          f"active {len(st['active'])} backlog {len(st['backlog'])} "
          f"peak-queue {st['queued_peak']} deployed {_deployed(st)}/{st['officers_total']} "
          f"pressure {st['pressure']:.0f}")
    print("ripple corridors:\n", ripple(st, prof).head(6).to_string(index=False))
