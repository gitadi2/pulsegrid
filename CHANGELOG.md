# PulseGrid — Changelog

## Already built (shipped & deployed)

**Models (`core.py`)** — four scikit-learn HistGradientBoosting models on real ASTraM data:
road-closure / barricade classifier (AUC 0.79; surfaces ~57% of closures while flagging only ~16%
of incidents), priority classifier (recovers the ASTraM rule), clearance-time regressor (median
error ~35 min on a heavy-tailed target), and a daily volume forecaster (beats the naïve baseline).
Plus the recommender, planned-event planner, explainability (`risk_factors`), corridor hour-profile,
and surge detector.

**Console (`app.py`)**
- 🛰 **Live Operations** — real-time discrete-event engine: incidents arrive on Bengaluru's real
  per-hour rates, are scored live by the models, and consume a finite officer pool. Moving City
  Pressure gauge, live map, force-utilisation, congestion-ripple board, dispatch log, 💥 Inject.
- ⚡ **Live Triage** — instant dispatch ticket + "Why this call" explainability.
- 📋 **Plan Event** — pre-deployment plan for planned events, scaled by crowd size.
- 🗺 **Hotspot Map** — geospatial chokepoint view.
- 📈 **Forecast & Staffing** — 7-day forecast with surge alerts + officer allocation.
- 🧪 **Model Card** — transparent metrics + proactive-advantage stat.

**Simulation (`sim.py`)** — arrival/severity sampling, officer-pool state machine, congestion ripple.

**Infra** — deployed live on Streamlit Cloud; resilient data loader finds the CSV by pattern
regardless of filename/location; no heavy dependencies.

---

## New in this update (two flagship features)

### 1. 💬 Ask the Console — natural-language co-pilot  *(new file: `assistant.py`)*
A scoped intent + entity parser turns plain-English questions into model-backed answers:
- *"Officers for a festival on Hosur Road Saturday evening?"* → full pre-deployment plan
- *"Closure risk on Mysore Road?"* → corridor risk profile
- *"Worst hotspots this week"* → ranked chokepoints
- *"7-day incident forecast"* → volume forecast + surge days

Fuzzy corridor matching, cause synonyms, day/time/crowd extraction, and a graceful help fallback.
**Runs fully offline — no external API or keys** — so it deploys clean and works in the live demo.

### 2. 🧠 Policy Lab — counterfactual impact engine  *(added to `sim.py`)*
Quantifies what PulseGrid's dispatch logic is *worth*. The same event-surge incident stream is
replayed under a **reactive control room** (FIFO, no reserve) vs **PulseGrid** (severity-first
dispatch + forecast-aware reserve) on identical officers. Headline result:

> **~50% faster to critical incidents** — average wait-for-units on road-closure & High-priority
> incidents drops from ~51 min to ~24 min on a 16-officer force (stable across random seeds).

Sliders vary force size and surge intensity; the advantage holds throughout.

---

## Files changed in this update
| File | Change |
|------|--------|
| `assistant.py` | **new** — offline NL co-pilot |
| `sim.py` | **added** `compare_policies` + stream/policy simulators |
| `app.py` | **added** two tabs (💬 Ask the Console, 🧠 Policy Lab); now 8 tabs |
| `APPROACH.md` | capabilities list updated to 8 features |
| `CHANGELOG.md` | **new** — this file |
| `core.py` | unchanged this update (resilient loader from the prior fix) |
