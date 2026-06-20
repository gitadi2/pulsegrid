"""
PulseGrid 'Ask the Console' — an offline natural-language co-pilot.

No external API: a scoped intent + entity parser maps a controller's plain-English
question onto the trained models and returns a grounded answer. Handles event
staffing/planning, 7-day forecasts, hotspot lookups and corridor risk profiles,
with fuzzy corridor matching and sensible defaults so it degrades gracefully.
"""
from __future__ import annotations
import re
import difflib
import numpy as np
import core

_DAYS = {"monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "wednesday": 2, "wed": 2,
         "thursday": 3, "thu": 3, "friday": 4, "fri": 4, "saturday": 5, "sat": 5,
         "sunday": 6, "sun": 6}
_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_CAT_KEYS = [("festival", "Festival / public event"), ("concert", "Festival / public event"),
             ("public event", "Festival / public event"), ("rally", "Political rally"),
             ("protest", "Political rally"), ("sport", "Sports event"), ("match", "Sports event"),
             ("stadium", "Sports event"), ("vip", "VIP movement"), ("convoy", "VIP movement"),
             ("procession", "Procession"), ("march", "Procession"),
             ("construction", "Construction / roadwork"), ("roadwork", "Construction / roadwork")]
_CAUSE_SYN = {"crash": "accident", "collision": "accident", "accident": "accident",
              "breakdown": "vehicle_breakdown", "stalled": "vehicle_breakdown",
              "flood": "water_logging", "waterlog": "water_logging", "rain": "water_logging",
              "pothole": "pot_holes", "tree": "tree_fall", "vip": "vip_movement"}


def _corridor(text, corridors):
    tl = text.lower()
    for c in sorted(corridors, key=len, reverse=True):
        if c.lower() in tl:
            return c
    words = [w for w in re.findall(r"[a-z]+", tl) if len(w) > 2]
    best, score = None, 0
    for c in corridors:
        cl = c.lower()
        s = sum(1 for w in words if w in cl)
        if s > score:
            best, score = c, s
    if best:
        return best
    m = difflib.get_close_matches(tl, [c.lower() for c in corridors], n=1, cutoff=0.45)
    if m:
        return next(c for c in corridors if c.lower() == m[0])
    return None


def _cause(text, causes):
    tl = text.lower()
    for c in causes:
        if c.replace("_", " ") in tl or c in tl:
            return c
    for k, v in _CAUSE_SYN.items():
        if k in tl and v in causes:
            return v
    return None


def _category(text):
    tl = text.lower()
    for k, v in _CAT_KEYS:
        if k in tl:
            return v
    return None


def _crowd(text):
    tl = text.lower()
    if any(k in tl for k in ["massive", "huge", "lakh", "100k", ">75", "enormous"]):
        return "Massive (>75k)"
    if any(k in tl for k in ["large", "big", "major", "25k", "50k"]):
        return "Large (25k-75k)"
    if any(k in tl for k in ["small", "minor", "5k"]):
        return "Small (<5k)"
    return "Medium (5k-25k)"


def _hour(text):
    tl = text.lower()
    m = re.search(r"(\d{1,2})\s*(?::|\.)?\s*(\d{2})?\s*(am|pm)", tl)
    if m:
        h = int(m.group(1)) % 12
        if m.group(3) == "pm":
            h += 12
        return h
    for k, h in [("midnight", 0), ("dawn", 6), ("morning", 9), ("noon", 12),
                 ("afternoon", 15), ("evening", 18), ("night", 21)]:
        if k in tl:
            return h
    return None


def _day(text):
    tl = text.lower()
    if "weekend" in tl:
        return 5
    if "weekday" in tl:
        return 1
    for k, v in _DAYS.items():
        if re.search(rf"\b{k}\b", tl):
            return v
    return None


def _help():
    return {"kind": "help", "answer": (
        "I'm the PulseGrid co-pilot. Ask me things like:\n\n"
        "- *How many officers for a festival on Hosur Road Saturday evening?*\n"
        "- *What's the closure risk on Mysore Road?*\n"
        "- *Show me the worst hotspots this week*\n"
        "- *What's the 7-day incident forecast?*")}


def answer_query(text, df, models, profile, corridors, causes):
    tl = (text or "").lower().strip()
    if not tl:
        return _help()

    corr = _corridor(tl, corridors)
    cause = _cause(tl, causes)
    cent = profile["centroids"]

    plan_intent = (_category(tl) is not None
                   or any(k in tl for k in ["plan", "crowd", "festival", "event", "rally"])
                   or ("officer" in tl and corr) or ("how many" in tl and corr))
    fcast_intent = any(k in tl for k in ["forecast", "next", "coming", "week", "expect",
                                         "days ahead", "upcoming", "predict volume"])
    hotspot_intent = any(k in tl for k in ["hotspot", "chokepoint", "worst", "busiest",
                                           "most incidents", "blackspot", "where are"])

    # ---- 1) event staffing / planning ----
    if plan_intent:
        cat = _category(tl) or "Festival / public event"
        cor = corr or "Hosur Road"
        crowd = _crowd(tl)
        hour = _hour(tl) or 18
        dow = _day(tl)
        dow = 5 if dow is None else dow
        c = ((float(cent.loc[cor, "lat"]), float(cent.loc[cor, "lon"]))
             if cor in cent.index else (12.97, 77.59))
        p = core.plan_event(models, cat, cor, crowd, hour, dow, c)
        bar = "pre-position barricades" if p["barricade"] else "keep barricades on standby"
        return {"kind": "plan", "answer": (
            f"**{cat}** · {crowd.split(' (')[0]} crowd · **{cor}** · {_DAY_NAMES[dow]} {hour:02d}:00\n\n"
            f"- Pre-stage **{p['officers']} officers**\n"
            f"- Expected disruption **~{p['expected_duration_min']:.0f} min**, "
            f"predicted **{p['priority']}** priority\n"
            f"- Road-closure risk **{p['closure_risk']*100:.0f}%** → {bar}\n"
            f"- Advisory window: **{p['advisory_window']}**\n"
            f"- {p['diversion']}")}

    # ---- 2) hotspots (checked before forecast so 'hotspots this week' wins) ----
    if hotspot_intent:
        hs = core.hotspots(df, 5)
        rows = "\n".join(
            f"- **{r.junction}** — {int(r.events)} events · "
            f"{r.high_share*100:.0f}% High · {r.closure_rate*100:.0f}% closures"
            for r in hs.itertuples())
        return {"kind": "hotspot", "answer": f"**Top incident chokepoints**\n\n{rows}"}

    # ---- 3) forecast ----
    if fcast_intent:
        fc = core.forecast_daily(models, 7)
        fc, mu, sd = core.surge_days(models, fc)
        avg = fc["forecast"].mean()
        surge = fc[fc["surge"]]["date"].dt.strftime("%a %d %b").tolist()
        line = (f"Surge days flagged: **{', '.join(surge)}** (above the {mu:.0f}±{sd:.0f}/day norm)."
                if surge else "No surge days above the norm in this window.")
        return {"kind": "forecast", "answer": (
            f"**7-day incident forecast** — averaging **{avg:.0f} events/day**.\n\n{line}")}

    # ---- 4) corridor risk profile (default when a corridor is named) ----
    if corr:
        s = df[df["corridor"] == corr]
        if len(s):
            cr = s["road_closure"].mean()
            hp = (s["priority"] == "High").mean()
            md = np.nanmedian(s["duration_min"])
            busiest = int(s.groupby("hour").size().idxmax())
            extra = ""
            if cause:
                sc = s[s["event_cause"] == cause]
                if len(sc) > 5:
                    extra = (f"\n- For **{cause}** here: {sc['road_closure'].mean()*100:.0f}% "
                             f"closure, ~{np.nanmedian(sc['duration_min']):.0f} min clearance")
            return {"kind": "risk", "answer": (
                f"**{corr}** risk profile ({len(s)} logged events)\n\n"
                f"- **{cr*100:.0f}%** require road closure\n"
                f"- **{hp*100:.0f}%** run High priority\n"
                f"- Median clearance **~{md:.0f} min**\n"
                f"- Busiest hour around **{busiest:02d}:00**{extra}")}

    return _help()
