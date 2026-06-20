# PulseGrid — Bengaluru Event-Impact Command Console

**Gridlock Hackathon 2.0 · Round 2 · Theme: Event-Driven Congestion (Planned & Unplanned)**

PulseGrid turns Bengaluru Traffic Police's raw incident stream into **decisions**. The moment an
event is logged, it forecasts the disruption and recommends a deployment plan — how long it will
take to clear, whether it needs barricading/diversion, and how many officers to send — then plans
city-wide staffing from a volume forecast.

Built on **8,173 real, anonymised ASTraM incidents** (Nov 2023 – Apr 2024).

---

## Why it matters (mapped to the brief)

| Pain point (from the theme)                  | What PulseGrid does                                                            |
| -------------------------------------------- | ----------------------------------------------------------------------------- |
| Event impact is **not quantified in advance**| Clearance-time + priority models predict impact at intake                     |
| Resource deployment is **experience-driven** | Road-closure model + load-balancing optimiser produce a data-driven plan      |
| **No post-event learning** system            | Every model retrains on resolved events — the resolution log is the label     |

## The four models (5-fold CV on real data)

| Model | Task | Result |
|---|---|---|
| **Road closure** | Will this event need a barricade/diversion? | **AUC 0.787**, PR-AUC 0.42 (8.3% base rate) |
| **Priority** | High vs Low triage | **AUC 0.9997** — recovers ASTraM's rule instantly |
| **Clearance time** | How long until cleared? | **median error ≈ 35 min** (long-tailed; reported robustly) |
| **Daily volume** | City-wide event load | **MAE 18/day**, beats naïve baseline (21) |

All ML is scikit-learn `HistGradientBoosting` — trains in seconds, deploys with no heavy deps.

## The console (4 views)

1. **Live Triage** — log an event → a colour-coded dispatch ticket with clearance band, priority,
   road-closure probability, officer count, barricade + diversion call.
2. **Hotspot Map** — every incident on a dark Bengaluru map, severity-coloured, filterable; surfaces
   the real chokepoints (Silk Board, Mekhri Circle, Yeshwanthpura…).
3. **Forecast & Staffing** — 7-day volume forecast + a per-corridor officer allocation you can size.
4. **Model Card** — the validated metrics, methodology, and an honest limitations note.

---

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dataset lives at `data/astram_events.csv`. Models train once on first load and are cached.

## Deploy (free, public link for the judges)

1. Push this folder to a public GitHub repo.
2. Go to https://share.streamlit.io → **New app** → pick the repo → main file `app.py`.
3. Done — you get a shareable URL to demo live.

## Structure

```
pulsegrid/
├── app.py              # Streamlit command console (UI)
├── core.py             # data prep, 4 models, recommender, forecast, optimiser
├── data/astram_events.csv
├── requirements.txt
└── .streamlit/config.toml
```

## Honesty note
Clearance time has a heavy tail (a few events span days), so we report **median** error and present a
clearance *band* rather than overclaiming a point estimate. Priority is near-deterministic from cause
in this dataset — the model recovers that logic at ~100%, which is a feature (consistent instant
triage), not leakage.
