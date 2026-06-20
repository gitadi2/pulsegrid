# PulseGrid — Solution Approach

**Gridlock Hackathon 2.0 · Round 2 · Theme 2: Event-Driven Congestion (Planned & Unplanned)**
**Participant:** Aditya Satapathy

---

## 1. The problem, restated as a decision

Political rallies, festivals, accidents, breakdowns and construction create localised traffic
breakdowns across Bengaluru. Today the response is reactive: impact isn't quantified in advance,
deployment leans on individual officers' experience, and nothing systematically learns from past
events. PulseGrid reframes this as a **decision problem** — for every event, and for the city as a
whole, *what is the impact, and where should resources go?* — and answers it with models trained on
the Bengaluru Traffic Police's own incident log.

## 2. Data

8,173 real, anonymised ASTraM incidents (Nov 2023 – Apr 2024): location (lat/lon, corridor, junction,
police station), cause (17 types), planned/unplanned flag, priority, road-closure flag, vehicle type,
and full lifecycle timestamps (created → resolved/closed). The resolution timestamps let us derive a
**ground-truth disruption duration** for ~2,800 events — the supervision signal most teams won't have
spotted.

## 3. Method

Four models, each predicting from **intake-known signals only** (so they fire the instant an event is
logged), all scikit-learn gradient-boosted trees:

1. **Clearance-time regressor** — log-duration from cause, vehicle, corridor, station, location, time.
2. **Priority classifier** — High vs Low triage.
3. **Road-closure classifier** — does this event need barricading/diversion? (class-balanced; the
   operationally critical, rare-but-costly call.)
4. **Daily-volume forecaster** — calendar + lag/rolling features for city-wide staffing.

A **deployment layer** converts predictions into action: officer count scales with predicted
severity and duration; barricade/diversion fire off the road-closure probability; and a
**load-balancing optimiser** allocates a fixed officer budget across corridors in proportion to
expected disruption load (volume × median clearance × severity).

## 4. Results (5-fold cross-validation, real data)

| Model | Metric | Value |
|---|---|---|
| Road closure | ROC-AUC / PR-AUC | **0.787** / 0.42 (base rate 8.3%) |
| Priority | ROC-AUC | **0.9997** |
| Clearance time | median abs. error | **≈ 35 min** |
| Daily volume | MAE (vs naïve 21) | **18 events/day** |

Predicting a rare road-closure event at 0.79 AUC, and city volume better than a strong seasonal
baseline, is the substance behind the dashboard — not a mock-up.

## 5. How it answers the three stated pain points

- **Impact not quantified in advance** → clearance-time + priority models quantify it at intake.
- **Deployment is experience-driven** → road-closure prediction + the allocation optimiser make it
  data-driven and explainable.
- **No post-event learning** → models retrain on resolved events; accuracy compounds as incidents
  close. This is built into the architecture, not bolted on.

## 6. Innovation, scalability, viability

- **Innovation:** most solutions stop at a forecast. PulseGrid closes the loop to a *deployment plan*
  — the decision layer and the self-improving learning loop are the differentiators.
- **Scalability:** sub-second inference, seconds to retrain on 8k rows; the same pipeline scales to
  streaming ingestion and any city with an incident log.
- **Real-world viability:** built on the actual ASTraM schema and surfaces real chokepoints
  (Silk Board, Mekhri Circle), so it drops into the existing control-room workflow.

## 7. Honesty & limitations

Clearance time is heavy-tailed (a few multi-day events), so we report median error and a clearance
*band* rather than a misleadingly precise point estimate. Priority is near-deterministic from cause in
this dataset — the model recovers that rule at ~100%, which is useful (consistent instant triage)
rather than a leak. `zone` is sparsely populated, so geography leans on corridor + station + lat/lon.

## 7b. Rigor & calibration (what makes the numbers trustworthy)

- **Clearance band, not a point guess.** Two quantile regressors (P50, P90) give an interval, then a
  conformal step calibrates it to a **verified 90% coverage** (81% → 90%). Honest about a hard,
  long-tailed target.
- **Calibrated closure probabilities.** Brier 0.08 with a published recall/precision/flag-rate table
  across thresholds — so a controller can pick an operating point, not trust a single number.
- **Forecast beats a real baseline.** Day-of-week + lag features beat a seasonal-naive forecast by ~15%.
- **Grounded heuristics.** The City Pressure cap is the 90th-percentile of *real concurrent
  severity-weighted load* (not a guessed constant), and the Live-Ops simulator's sampled stream
  reproduces the real hour-of-day incidence at **r≈0.98**.
- **Concrete diversion.** Closures recommend the **nearest parallel corridor** by geography, not a
  vague instruction.

## 8. Console capabilities (what a controller actually does)

1. **Live Operations (real-time engine).** A discrete-event simulation replays Bengaluru in
   accelerated time: incidents arrive on the city's *real* per-hour Poisson rates and empirical
   cause→corridor→vehicle distributions, every incident is scored live by the ML models, and officers
   are committed from a **finite pool** and released as events clear. The console shows a moving
   **City Pressure Index** gauge, a live incident map, force-utilisation, a **congestion-ripple** board
   (corridors absorbing spillover from active closures), and a scrolling dispatch log. A **💥 Inject**
   button and an officer-pool slider let a controller stress-test "what if three corridors close at
   rush hour?" and watch the force saturate and incidents queue as *awaiting units* — the proactive
   decision-support story, live.
2. **Ask the Console (NL co-pilot).** A controller can type a plain-English question —
   *"how many officers for a festival on Hosur Road Saturday evening?"*, *"closure risk on Mysore
   Road?"*, *"worst hotspots this week"* — and a scoped intent+entity parser routes it onto the same
   trained models and answers in seconds. Runs **fully offline** (no external API or keys), so it
   deploys clean and works on the live cloud demo.
3. **Policy Lab (counterfactual impact).** The decisive "is this worth it?" answer. The same event-surge
   incident stream is replayed under a **reactive control room** vs **PulseGrid's severity-aware dispatch
   with a forecast-aware reserve**, on identical officers. Result: PulseGrid gets units to road-closure
   and High-priority incidents **~50% faster** (e.g. 51 → 24 min average wait on a 16-officer force) —
   same people, same incidents, smarter triage. Sliders let judges vary the force size and surge
   intensity and watch the advantage hold.
4. **Live Triage** — log an incoming incident; get an instant color-coded dispatch ticket
   (clearance band, priority, closure risk, officers, barricade/diversion) **plus a "Why this call"
   explainability panel** citing the historical closure rate and priority profile behind each number.
5. **Plan Event** — for *planned* congestion (festivals, rallies, VIP movement, processions, roadwork):
   pick the event, venue corridor, expected crowd size and time → a pre-deployment plan that scales
   officers / barricades / advisory window by crowd, alongside the venue's historical hour-by-hour
   risk profile.
6. **Hotspot Map** — geospatial view of the real chokepoints to position reserves.
7. **Forecast & Staffing** — 7-day volume forecast with **automatic surge alerts** (days running above
   the mean+1σ norm) and a corridor-level officer allocation for any chosen headcount.
8. **Model Card** — transparent metrics and a **proactive-advantage** statement: the barricade model
   surfaces ~57% of all road-closure events while flagging only ~16% of incidents — a 3.6× lift.

## 9. Next steps
Real-time API ingestion, route-graph diversion routing (vs. corridor-level today), an LP for the
officer allocation under shift constraints, and an NLP pass over the free-text (incl. Kannada)
incident descriptions.
