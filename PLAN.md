# NEM Price Spike Forecaster — build plan

> **Rev 2 — 17 Aug 2026.** Critical pass over rev 1, per the standing rule that a plan is a living document. Seven flaws found; the four that changed the design are recorded here so the reasoning survives.
>
> 1. **Resolution was never decided.** Rev 1 backfilled 30-minute history to 2015 on Day 2 while defining every feature on 5-minute lags. Those cannot train one model. **Decided: the model runs at 5-minute resolution on the ~13-month archive. The long history is repurposed, not discarded** — see §2.3.
> 2. **The positive count breaks the fold design.** ~114,000 intervals and roughly 100–200 distinct spike episodes against 1-week test folds means most folds contain zero events and per-fold PR-AUC is undefined. Evaluation redesigned in §2.4.
> 3. **The weather source in rev 1's §8 was the reanalysis API** — observed conditions reconstructed after the fact, which is exactly the leakage §4.1 exists to prevent. Corrected to the Historical Forecast API throughout.
> 4. **The stated edge was never built.** Rev 1 argued AEMO under-predicts spikes caused by rebidding, then used no bid data. `Bidmove_Complete` and `Dispatch_SCADA` are both live in the NEMWeb archive (verified 17 Aug 2026). Bid-stack features are now Phase 2's headline, structured as a clean ablation — see §3.3.
>
> Also fixed: a leak in the battery charging rule (§2.2) and an optimistic Airflow estimate (§5).

**Status:** Phase 1 complete (3 Sep 2026); Phases 2 to 6 not started. This is the plan as written before the build, kept so its reasoning survives; where the build proved it wrong, the README says so. Rev 1 written 3 Aug 2026, rev 2 17 Aug 2026.
**Repo name:** `nem-price-forecaster`
**Budget:** 2 weeks intense. A defensible, demonstrable system exists at end of Day 7; Days 8–14 add the bid-stack ablation, operations and the economic backtest.

---

## 1. Why this dataset

- **Free and unambiguously licensed.** AEMO's Copyright Permissions Notice grants "general permission for anyone to use AEMO Material for any purpose" with attribution, and states "You do not need to obtain specific permission". No key, no registration, no non-commercial clause.
- **It genuinely arrives.** `nemweb.com.au/Reports/Current/DispatchIS_Reports/` publishes a new file every 5 minutes — verified from directory timestamps, not documentation. Drift and retraining are real problems here, not staged ones.
- **There is an incumbent forecast to benchmark against.** AEMO publishes its own 5-minute pre-dispatch (`P5_Reports`) and 30-minute pre-dispatch. Having a published production baseline to compare against is rare, and it is worth more than the model itself.
- **A real decision hangs off it.** Five-minute spot price sets battery and VPP dispatch, demand-response triggering, and hedging. Prices range from −$1,000 to $17,500/MWh. The output converts directly to dollars.
- **The mechanics are explainable.** Prices spike for physical reasons: a unit trips, an interconnector binds, the evening ramp meets low wind. Features can be built from those causes rather than from price history alone.

---

## 2. The problem, stated precisely

**Prediction target.** Given all information available at time `t`, predict `P(RRP > $300/MWh at any dispatch interval in (t, t+30min])` for a single NEM region (start with NSW1; extend to VIC1/SA1/QLD1/TAS1 if time allows).

**Why classification rather than price regression.** Regression on RRP is dominated by a heavy right tail — the metric is driven almost entirely by a handful of intervals, and RMSE improvements are hard to interpret. A spike-probability classifier has a clear metric, a genuinely imbalanced positive class (real ML problem, not a toy), and matches the actual decision: dispatch or don't.

**The $300 threshold** is a starting point, chosen because it sits well above typical operating cost and is where battery arbitrage starts mattering. Sensitivity to the threshold ($100 / $300 / $1,000) is part of the analysis, not a fixed assumption.

### 2.1 Baselines to beat

Three, in increasing order of difficulty:

1. **Climatology** — historical spike rate by time-of-day × day-of-week × season. Trivially beatable; establishes the floor.
2. **Persistence** — spike in the previous 30 min implies spike now.
3. **AEMO P5MIN pre-dispatch** — "AEMO's own forecast RRP for the target interval exceeds $300". This is the one that matters. AEMO's pre-dispatch assumes bids stay as submitted, so it is known to under-predict spikes caused by rebidding; that is the exploitable gap. It may still be hard to beat.

**Honest position on the result.** Beating AEMO may not happen. Design the writeup so a null result is still a good project: "matched AEMO on PR-AUC but achieved 2.4× precision at the top 50 alerts, worth $X in the dispatch backtest" is a legitimate and interesting finding. A fabricated win is not. Report whatever the backtest produces.

### 2.2 Resolution: 5-minute primary, long history repurposed `[decided rev 2]`

**The model trains and predicts at 5-minute dispatch resolution on the ~13-month archive.** Three reasons, in order of weight:

1. **The benchmark only exists at 5 minutes.** AEMO P5MIN is the incumbent forecast worth beating. A 30-minute model would have to be compared against 30-minute pre-dispatch, which forecasts further ahead and is an easier target — a weaker claim, and an obviously weaker one to anyone who knows the market.
2. **The decision is a 5-minute decision.** Spot price is set every 5 minutes and battery dispatch responds at that cadence. A 30-minute forecast answers a question nobody asks.
3. **Half the features do not exist at 30 minutes.** Interconnector binding, ramp rate, and available-generation headroom are dispatch-interval quantities. Aggregating them to 30 minutes averages away the thing that causes the spike.

**The cost of this choice, stated up front:** 13 months contains exactly one summer, and spikes concentrate in summer. The model is therefore fitted to one season's worth of extreme events. This is a real limitation and belongs in the README rather than in a footnote.

**The 30-minute history to 2015 is still worth ingesting, for three things it is genuinely better at:**

- **The climatology baseline.** Spike rate by time-of-day × day-of-week × season, estimated over ten years rather than one. A baseline built on more data is a *harder* baseline, which strengthens the comparison rather than flattering it.
- **Threshold sensitivity.** How the $100 / $300 / $1,000 spike rate has moved year on year. Cheap to compute, and it justifies the $300 choice with evidence instead of assertion.
- **Regime-change analysis.** Whether the spike process in the 13-month training window resembles the preceding decade. If it does not, that is a finding about the model's shelf life and a natural lead-in to the drift monitoring in Phase 3.

Day 2 therefore still backfills the 30-minute series. What changed is that it feeds baselines and analysis, never the feature table.

### 2.3 Evaluation design `[rewritten rev 2]`

Rev 1 specified walk-forward with 1-week test folds. Do the arithmetic first: ~114,000 5-minute intervals over 13 months, a spike rate in the low fractions of a percent, and spikes that arrive in clusters rather than independently. That is on the order of 100–200 distinct episodes. Split into ~56 weekly folds, most folds contain **zero** episodes, PR-AUC is undefined on them, and the mean across folds is decided by a handful of weeks.

Rules, fixed before any model is trained:

- **Fold by event, not by calendar.** Expanding-window walk-forward where each test fold is sized to contain a minimum number of spike episodes rather than a fixed number of days.
- **Pool predictions, then score once.** Concatenate out-of-fold predictions across the whole backtest and compute one PR-AUC on the pooled set. Averaging per-fold metrics over folds with wildly different positive counts is the wrong estimator.
- **Report the denominator everywhere.** Every metric in the README carries the number of positive episodes behind it. "PR-AUC 0.41 over 137 spike episodes" is a defensible sentence; "PR-AUC 0.41" alone is not.
- **Confidence intervals by blocked bootstrap.** Resample contiguous blocks, not individual intervals, because neighbouring intervals are not independent. If the interval around the model overlaps the interval around AEMO, say so plainly rather than claiming a win.
- **Purge gap between train and test** to stop rolling-window features straddling the boundary.

**Primary metric: precision@k on the pooled predictions**, where k is the number of alerts an operator would actually action in the period. PR-AUC is reported alongside it. Precision@k matches the decision, and it stays interpretable when the positive class is this rare.

### 2.4 Economic evaluation

Metrics alone stop short. Convert the probability into a dispatch policy and backtest it:

- Simple battery: 1 MW / 2 MWh, 88% round-trip efficiency, one cycle per day cap.
- Policy: discharge when P(spike) > τ, charge when price falls below a **trailing** threshold — the 10th percentile of the preceding 7 days, recomputed daily. `[fixed rev 2: rev 1 said "bottom decile" without a window, which if computed over the day or the test period is a policy that trades on information it would not have had. A backtest that leaks is worth less than no backtest, because it produces a revenue number that cannot be defended.]`
- Report annualised revenue for: perfect foresight (ceiling), your model, AEMO P5MIN policy, and always-charge-overnight-discharge-evening (naive).
- Sweep τ and show the revenue/false-alarm trade-off.

---

## 3. Architecture

Local-first, Docker Compose, no cloud bill — same pattern as `edge-reliability-lab`, so anyone can clone and run it.

```
NEMWeb (5-min ZIP)  ─┐
Open-Meteo (weather) ─┼─→ [ingest]  → Postgres raw
AEMO monthly CSVs   ─┘                    │
                                          ↓
                                    [dbt] staging → intermediate → marts
                                          │
                                          ↓
                              feature table (point-in-time correct)
                                          │
                      ┌───────────────────┼───────────────────┐
                      ↓                   ↓                   ↓
              [train + backtest]    [FastAPI /predict]   [drift monitor]
                      │                   │                   │
                      ↓                   ↓                   ↓
                  MLflow registry    Prometheus ←──────── Grafana
                      │
                      └── weekly retrain DAG with promotion gate
                                   (Airflow)
```

### 3.1 Components

| Layer | Choice | Why this one |
|---|---|---|
| Ingestion | Python + `httpx`, own MMS parser | MMS files are nested multi-table flat files whose columns change between versions; owning the parser means owning the schema checks |
| Backfill | `nemosis` package for long history | Legitimate use of a maintained library; do not hand-roll 27 years of parsing |
| Storage | Postgres 16 | Handles a concurrent writer (the poller) and readers (dbt, the API); dbt-postgres is mature |
| Transform | **dbt-core** | SQL transformations with tests and lineage built in |
| Orchestration | **Airflow 3** (Docker Compose) | Scheduling with retries and backfills; heavier than Prefect |
| Model | **LightGBM** | Strong on tabular time series and trains in seconds; a transformer would add cost with no reason to expect a gain here |
| Tracking/registry | **MLflow** | Experiment tracking plus a model registry for the promotion gate |
| Serving | FastAPI + Docker | Consistent with existing work; measure and report p99 |
| Monitoring | Prometheus + Grafana | Same stack as `edge-reliability-lab`, so the alerting patterns carry over |
| CI | GitHub Actions | ruff + pytest + `dbt build` against a test schema + docker build |

### 3.2 Features

Available strictly as-of prediction time:

- **Price:** RRP lags (5, 10, 15, 30, 60 min), rolling mean/max/std over 30/60/240 min, time since last spike, count of spikes in trailing 24h.
- **Demand:** TOTALDEMAND level, 30-min ramp rate, demand vs same-time-last-week.
- **Supply side:** available generation vs demand headroom, interconnector flows and whether limits are binding, rooftop PV actuals (`ROOFTOP_PV`), semi-scheduled (wind/solar) availability.
- **AEMO's own forecast:** P5MIN forecast RRP and forecast demand for the target interval, as a feature *and* as the benchmark.
- **Weather:** temperature, apparent temperature, wind speed and cloud cover at region-load-centre coordinates, from Open-Meteo forecast API (what was forecast at prediction time, not what was later observed).
- **Calendar:** half-hour of day, day of week, month, public holidays (source: `holidays` package, `AU`/`NSW` subdivision — pin the version, since holiday sets change), summer/winter peak flags.

### 3.3 Bid-stack features — the actual hypothesis `[added rev 2]`

Rev 1 argued that AEMO's pre-dispatch under-predicts spikes **because it assumes bids stay as submitted, and generators rebid**. It then built no bid features. The model was left to infer rebidding from price lags, which is the weakest available proxy for it.

Both required sources are live in the NEMWeb archive, verified 17 Aug 2026:

- **`Bidmove_Complete`** — energy offers by unit and price band, including rebids and their stated reasons.
- **`Dispatch_SCADA`** — actual unit output at 5-minute resolution, which gives unit trips and unexplained output drops.

Candidate features, all strictly as-of prediction time:

- **Headroom below the threshold:** total capacity offered at or below $300 for the region, minus current demand. This is the direct physical statement of "how close are we to running out of cheap supply".
- **Stack shape:** capacity offered in each price band ($0–50, $50–300, $300–5,000, above $5,000), as levels and as a share of available capacity.
- **Stack movement:** change in each of the above since the last pre-dispatch run — the rebidding signal itself, which is precisely what P5MIN cannot see.
- **Concentration:** share of the marginal band held by the largest participant. Thin, concentrated margins are where rebidding pays.
- **Supply shocks:** units whose SCADA output dropped sharply against their offered availability in the last few intervals.

**Structure this as an ablation, not as a feature dump.** Train the Phase 2 model without bid features first, record the pooled precision@k and PR-AUC, then add them and record again. Either outcome is publishable and interesting:

- *They help* — the README states how much, and the project has demonstrated a hypothesis about market mechanics, expressed it as features, and measured it. That is a stronger claim than any single metric.
- *They do not help* — that is a genuine finding about the limits of the rebidding explanation, and it is more interesting than a model that got lucky. Report it.

**Cost:** `Bidmove_Complete` is another MMS format and another parser, and it is the largest of the feeds. Budget two days, not one. This is why it is Phase 2 rather than Phase 1 — the system must already work end to end before this is attempted, so that a parsing rabbit hole cannot sink the whole project.

---

## 4. The things most likely to sink this

### 4.1 Leakage — the number one risk

Every feature must be restricted to the vintage available at prediction time. Two specific traps:

- **Pre-dispatch vintages.** A P5MIN run published at 14:05 forecasts 14:10–15:05. If you join "AEMO's forecast for 14:30" without checking which run it came from, you can silently pull in a run published at 14:25 — after your prediction time. Every pre-dispatch join must be an explicit as-of join on `(run_datetime <= t, target_interval = t+30min)`, taking the latest qualifying run.
- **Weather.** Use Open-Meteo's **Historical Forecast API** (`historical-forecast-api`), which archives what past forecast runs actually said. Do **not** use the Historical Weather API: that is ERA5 reanalysis, meaning observed conditions reconstructed afterwards, and joining it is a direct leak. Observed temperature at 14:30 was not knowable at 14:00. `[rev 1 linked the reanalysis product in its Sources section while §3.2 called for forecasts — corrected rev 2.]`

**Mitigation:** a dedicated `tests/test_no_leakage.py` that, for a sample of prediction timestamps, asserts every source row feeding the feature vector has `publish_time <= t`. Write this test on Day 4, before any model is trained. Also a dbt test on the as-of join.

### 4.2 The MMS flat-file format

NEMWeb ZIPs contain AEMO MMS flat files — a nested `C`/`I`/`D` record layout with multiple tables per file and column sets that change between report versions. This is not a `pd.read_csv`. Budget a full day. Verify the internal layout against AEMO's MMS Data Model documentation on Day 1 before committing to a schema; the plan above assumes the standard layout but that has not been confirmed by unzipping a file.

### 4.3 Other known constraints

- **Archive retention is ~13 months.** `Reports/Archive/DispatchIS_Reports/` currently holds daily rollups from Jul 2025 to Jul 2026 only. So: use the **30-minute Aggregated Price and Demand CSVs (Dec 1998 →, `aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_YYYYMM_NSW1.csv`)** for long history, and 5-minute data for the recent window. Be explicit in the README that the high-resolution training window is ~13 months.
- **Data volume.** ~5.5 MB/day compressed for the DISPATCHIS archive; 13 months ≈ 2 GB. Fine locally, but do not commit it — `.gitignore` the data directory and ship a `make backfill` instead.
- **Open-Meteo free tier is non-commercial** and capped at 10,000 calls/day, 300,000/month. A non-commercial project sits inside those terms. Do not use BOM: their terms explicitly prohibit scraping and redistribution.
- **Attribution.** README must credit AEMO (data © AEMO) and Open-Meteo (CC BY 4.0).

---

## 5. Schedule `[restructured rev 2]`

Rev 1 was a fourteen-day calendar. Two of its estimates were wrong — Airflow in one day, and bid parsing not costed at all — and a calendar that is wrong in two places stops being useful the moment it slips, because there is no rule for what to drop.

**Phases replace days.** Each phase has an exit condition. The rule that survives from rev 1 and matters more than any date: **there is a working system end to end before anything is made sophisticated.**

| Phase | What it delivers | Exit condition | Estimate |
|---|---|---|---|
| **1. Working pipeline** | Days 1–7 below, unchanged | `curl` returns a live prediction from real data, leakage test green | 7 days |
| **2. The edge** | Bid-stack features (§3.3) as a measured ablation | Two pooled precision@k numbers, with and without bids | 2–3 days |
| **3. Operations** | MLflow registry with a promotion gate, drift monitoring, alerts | A challenger is rejected and the rejection is logged | 2 days |
| **4. The economics** | Battery backtest, τ sweep, calibration, error analysis | Revenue table with a trailing-threshold policy | 2 days |
| **5. Orchestration** | Airflow DAGs | DAGs green, screenshot | 2 days |
| **6. Ship** | CI, `make demo`, README, case study, public push | Clean clone runs | 1–2 days |

**Phase 5 is last on purpose, and this is a change from rev 1.** Airflow demonstrates tooling, while phases 2 to 4 are the ones that demonstrate judgement. If the fortnight compresses, a project with a measured bid-feature ablation and an honest economic backtest orchestrated by a Makefile beats a project with three green DAGs and neither. Run the pipeline from `make` until Airflow is earned.

### Phase 1 detail — a working pipeline

| Day | Deliverable | Done when |
|---|---|---|
| 1 | Repo scaffold, Docker Compose (Postgres + Adminer), MMS format confirmed by unzipping a real file | One DISPATCHIS file parsed into a dataframe with correct columns |
| 2 | Ingestion: live 5-min poller + `nemosis` backfill of 30-min history to 2015 + Open-Meteo forecast archive pull | `raw.*` tables populated; poller runs on a loop without dropping intervals |
| 3 | dbt project: staging → intermediate → marts; dbt tests for uniqueness, not-null, freshness | `dbt build` green |
| 4 | Feature table with point-in-time as-of joins; **leakage test written and passing** | `test_no_leakage.py` green on 200 sampled timestamps |
| 5 | Baselines implemented and measured: climatology (estimated on the 10-year 30-min series, §2.2), persistence, AEMO P5MIN | Three pooled precision@k and PR-AUC numbers written down, each with its positive-episode count |
| 6 | LightGBM trained; walk-forward backtest harness (expanding window, **event-sized folds**, purge gap, pooled scoring, blocked-bootstrap CIs — §2.3) | Model beats climatology by more than the bootstrap interval; MLflow logging experiments |
| 7 | FastAPI `/predict` + `/forecast/latest`, containerised; README v1 with architecture diagram and current numbers | `curl` returns a live prediction from real data |

**Checkpoint at the end of Phase 1.** If behind, cut multi-region and keep running from `make`. A single-region system with honest baselines and a passing leakage test is already a better project than most.

### Phases 2–6 detail

| Phase | Work | Done when |
|---|---|---|
| 2 | Parse `Bidmove_Complete`; build the bid-stack features in §3.3; retrain; record the ablation both ways | Two pooled precision@k figures and a paragraph saying which way it went and why |
| 3 | MLflow registry; champion/challenger gate — retrain weekly, promote only on a held-out recent window. Drift: PSI on key features, prediction-distribution drift, data-freshness SLA; Prometheus + Grafana + alert rules | A challenger is rejected and logged; a stale-feed alert fires when the poller is stopped |
| 4 | Battery backtest with the trailing-threshold policy (§2.4), τ sweep, revenue against perfect foresight / AEMO / naive. Calibration curve, precision@k, error analysis, SHAP | A revenue table, a trade-off chart, and an honest "when this model fails" section |
| 5 | Airflow: ingest DAG (5-min), transform DAG, train DAG | DAGs green in the UI, screenshot for the README |
| 6 | GitHub Actions CI; `make demo`; `/predict` latency measured (p50/p99); README, CASE_STUDY.md, screenshots; push public | CI green on a clean clone; README readable by a non-specialist |

---

## 6. Open decisions

- **Region scope.** Start NSW1 only. Extend to all five regions only if Week 2 has slack — cross-region features (interconnector coupling) are interesting but are a Week 3 problem.
- **Threshold.** $300 to start; report sensitivity.
- **Horizon.** 30 minutes to start. 60 minutes is more useful operationally and harder; consider as a second target once the 30-min pipeline works.

---

## 7. Sources

- AEMO NEMWeb current reports — https://nemweb.com.au/Reports/Current
- AEMO NEMWeb archive — https://nemweb.com.au/Reports/Archive
- AEMO aggregated price and demand — https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data
- AEMO copyright permissions — https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions
- Open-Meteo terms and rate limits — https://open-meteo.com/en/terms
- Open-Meteo **historical forecast** API (archived past forecast runs — this is the one to use) — https://open-meteo.com/en/docs/historical-forecast-api
- Open-Meteo historical weather API (ERA5 reanalysis — **do not join this to features**, see §4.1) — https://open-meteo.com/en/docs/historical-weather-api
- AEMO MMS Data Model report documentation (confirm the `Bidmove_Complete` and `DispatchIS` layouts before writing parsers) — https://nemweb.com.au/Reports/Current
