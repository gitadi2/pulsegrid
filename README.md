# PulseGrid — Bengaluru Event-Impact Command Console

**Gridlock Hackathon 2.0 · Round 2 · Theme: Event-Driven Congestion (Planned & Unplanned)**

<p align="center">
  <a href="https://pulsegrid-diecjwaeyxc8supyngnd4y.streamlit.app/" target="_blank"><img src="https://img.shields.io/badge/%E2%96%B6%20Live%20Demo-Open%20the%20Console-36CFC9?style=for-the-badge" alt="Live Demo"/></a>
  <img src="https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
</p>

PulseGrid turns Bengaluru Traffic Police's raw incident stream into **decisions**. The moment an
event is logged, it forecasts the disruption and recommends a deployment plan — how long it will
take to clear, whether it needs barricading or a diversion, and how many officers to send — then
plans city-wide staffing from a volume forecast and lets a controller **run the whole thing live**.

Built on **8,173 real, anonymised ASTraM incidents** (Nov 2023 – Apr 2024).
**🔴 Live demo → https://pulsegrid-diecjwaeyxc8supyngnd4y.streamlit.app/** · all ML in scikit-learn, no heavy dependencies.

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
| **Road closure** | Will this event need a barricade / diversion? | **AUC 0.787**, PR-AUC 0.42, **Brier 0.08** (well-calibrated) |
| **Priority** | High vs Low triage | **AUC 0.9997** — recovers ASTraM's rule instantly |
| **Clearance time** | How long until cleared? | **P50 median err ≈35 min + conformally-calibrated 90% band** (verified coverage) |
| **Daily volume** | City-wide event load | **MAE 18/day**, beats the naïve baseline (21) |

**Proactive advantage:** at its alert threshold the closure model surfaces **~57% of all road-closure
events while flagging only ~16% of incidents** — a **3.6× lift** over reviewing cases at random.

Every claim is validated: the clearance band is conformally calibrated to true 90% coverage, closure
probabilities are well-calibrated (Brier 0.08) with a published operating-point table, the volume
forecast beats a seasonal-naive baseline by ~15%, and the Live-Ops simulator reproduces the real
hour-of-day incidence at r≈0.98. All ML is scikit-learn `HistGradientBoosting` — no heavy deps.

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

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dataset lives at `data/astram_events.csv` (the loader also finds it at the repo root or under any
ASTraM-style filename). Models train once on first load (~1 min) and are cached.

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

---

### Built by ADITYA SATAPATHY

<p align="center">
  <a href="https://github.com/gitadi2" target="_blank"><img src="https://cdn.simpleicons.org/github/ffffff" alt="GitHub @gitadi2" height="40" style="margin: 0 15px;"/></a>
  <img width="12" />
  <a href="https://www.linkedin.com/in/adisatapathy" target="_blank"><img src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="LinkedIn @adisatapathy" height="40" style="margin: 0 15px;"/></a>
  <img width="12" />
  <a href="mailto:satgriezeleo1007@gmail.com" target="_blank"><img src="https://cdn.simpleicons.org/gmail/EA4335" alt="satgriezeleo1007@gmail.com" height="40" style="margin: 0 15px;"/></a>
</p>

<p align="center">
  <a href="https://github.com/gitadi2"><code>@gitadi2</code></a> &middot;
  <a href="https://www.linkedin.com/in/adisatapathy"><code>@adisatapathy</code></a> &middot;
  <a href="mailto:satgriezeleo1007@gmail.com"><code>satgriezeleo1007@gmail.com</code></a>
</p>

<p align="center"><em>Built for <b>Flipkart Gridlock 2.0</b>, 2026</em></p>

<p align="center">
  <a href="https://www.flipkart.com/" target="_blank"><img src="https://github.com/Flipkart.png" alt="Flipkart" height="44"/></a>
</p>
