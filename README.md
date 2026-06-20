# PulseGrid — Bengaluru Event-Impact Command Console

**Gridlock Hackathon 2.0 · Round 2 · Theme: Event-Driven Congestion (Planned & Unplanned)**

PulseGrid turns Bengaluru Traffic Police's raw incident stream into **decisions**. The moment an
event is logged, it forecasts the disruption and recommends a deployment plan — how long it will
take to clear, whether it needs barricading or a diversion, and how many officers to send — then
plans city-wide staffing from a volume forecast and lets a controller **run the whole thing live**.

Built on **8,173 real, anonymised ASTraM incidents** (Nov 2023 – Apr 2024).
Live demo: deployed on Streamlit Community Cloud · all ML in scikit-learn, no heavy dependencies.

---

## What makes it stand out

- 🛰 **A live command center, not a static dashboard.** A discrete-event simulation replays Bengaluru
  in accelerated time — incidents arrive on the city's *real* per-hour rates, get scored by the models
  on the spot, and consume a **finite officer pool**. Watch the City Pressure Index move, the force
  saturate, and incidents queue as *awaiting units*.
- 💬 **You can just ask it.** A natural-language co-pilot answers *"how many officers for a festival on
  Hosur Road Saturday evening?"* using the same models — **fully offline, no API key**.
- 🧠 **It proves its own worth.** A counterfactual engine shows PulseGrid's dispatch logic gets units to
  critical incidents **~50% faster** than a reactive control room, on identical officers.

---

## Why it matters (mapped to the brief)

| Pain point (from the theme)                   | What PulseGrid does                                                          |
| --------------------------------------------- | --------------------------------------------------------------------------- |
| Event impact is **not quantified in advance** | Clearance-time + priority + closure models predict impact at intake         |
| Resource deployment is **experience-driven**  | Closure model + a load-balancing optimiser produce a data-driven plan       |
| **Planned** events aren't pre-planned for     | The Plan Event view scales officers/barricades/advisory window by crowd size |
| Triage under a surge is **first-come**        | Severity-aware dispatch + a forecast-aware reserve (proven in Policy Lab)    |
| **No post-event learning** loop               | Every model trains on resolved events — the resolution log *is* the label    |

---

## The models (5-fold CV on real data)

| Model | Task | Result |
|---|---|---|
| **Road closure** | Will this event need a barricade / diversion? | **AUC 0.787**, PR-AUC 0.42 (8.3% base rate) |
| **Priority** | High vs Low triage | **AUC 0.9997** — recovers ASTraM's rule instantly |
| **Clearance time** | How long until cleared? | **median error ≈ 35 min** (long-tailed; reported robustly) |
| **Daily volume** | City-wide event load | **MAE 18/day**, beats the naïve baseline (21) |

**Proactive advantage:** at its alert threshold the closure model surfaces **~57% of all road-closure
events while flagging only ~16% of incidents** — a **3.6× lift** over reviewing cases at random.

All ML is scikit-learn `HistGradientBoosting` — trains in seconds, deploys with no heavy deps.

---

## The console (8 views)

1. 🛰 **Live Ops** — real-time engine: incidents stream in on real arrival patterns, are scored live,
   and draw from a finite officer pool. City Pressure gauge, live map, force-utilisation,
   **congestion-ripple** board, dispatch log, and a 💥 **Inject** button for live what-if drama.
2. 💬 **Ask the Console** — plain-English co-pilot over the whole system (event staffing, corridor
   risk, hotspots, forecast). Offline intent parser, no external API.
3. ⚡ **Live Triage** — log an event → a colour-coded dispatch ticket (clearance band, priority,
   closure probability, officer count, barricade + diversion) **with a "Why this call" explainer**.
4. 📋 **Plan Event** — pre-deployment plan for *planned* events (festival, rally, VIP, procession,
   roadwork), scaled by expected crowd, with the venue's historical busy-hour profile.
5. 🗺 **Hotspot Map** — every incident on a dark Bengaluru map, severity-coloured and filterable;
   surfaces the real chokepoints (Silk Board, Mekhri Circle, Yeshwanthpura…).
6. 📈 **Forecast & Staffing** — 7-day volume forecast with **surge alerts** + a per-corridor officer
   allocation you can size to any headcount.
7. 🧠 **Policy Lab** — replays one event-surge stream under a reactive control room vs PulseGrid's
   severity-aware dispatch: **~50% faster to critical incidents** (≈51 → 24 min on a 16-officer force),
   with sliders to vary force size and surge intensity.
8. 🧪 **Model Card** — validated metrics, methodology, and an honest limitations note.

---

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dataset lives at `data/astram_events.csv` (the loader also finds it at the repo root or under any
ASTraM-style filename). Models train once on first load and are cached.

## Deploy (free, public link for the judges)

1. Push this folder to a **public** GitHub repo (include the data CSV).
2. Go to https://share.streamlit.io → **Create app** → pick the repo → main file `app.py`.
3. Done — you get a shareable URL to demo live.

## Structure

```
pulsegrid/
├── app.py              # Streamlit command console — 8 tabs (UI)
├── core.py             # data prep, 4 models, recommender, planner, forecast, optimiser
├── sim.py              # real-time simulation engine + Policy Lab counterfactual
├── assistant.py        # offline natural-language co-pilot
├── data/astram_events.csv
├── requirements.txt
├── .streamlit/config.toml
├── APPROACH.md         # methodology & design write-up
└── CHANGELOG.md        # what shipped, and what's new
```

---

## Honesty note

Clearance time has a heavy tail (a few events span days), so we report **median** error and present a
clearance *band* rather than overclaiming a point estimate. Priority is near-deterministic from cause
in this dataset — the model recovers that logic at ~100%, which is a feature (consistent instant
triage), not leakage. The Policy Lab compares two dispatch policies on an identical simulated stream;
it is a controlled what-if, clearly labelled as such, not a claim about historical outcomes.
