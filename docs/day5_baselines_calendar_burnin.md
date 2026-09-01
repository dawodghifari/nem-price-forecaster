# Day 5: baselines

Generated 01 Sep 2026 05:27 UTC by `scripts/baselines.py`. Not hand-edited.

## The comparison set

- Rows scored: 82,424 (of 108,272 in the feature table)
- Positive rows: 477
- Spike episodes: 72 across 24 days
- Rows AEMO published a forecast for: 82,424 (100.0%)
- Positive rows with no AEMO forecast: 0
- Rows with a full 30-minute rolling window: 82,418

Rows before the first test fold are training data for every fold and are
excluded. Rows a baseline has no value for keep their place and rank last:
abstaining is not rewarded.

## Results

Every figure is pooled across the out-of-fold predictions and scored once,
never averaged over folds. `precision@k` is the share of the top k alerts
followed by a spike; `episode recall@k` is the share of distinct episodes
with at least one alert inside it. The two answer different questions and
the second is the one an operator would ask.

| Baseline | PR-AUC | 95% CI | precision@50 | lift@50 | episode recall@50 | precision@100 | lift@100 | episode recall@100 | precision@200 | lift@200 | episode recall@200 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Climatology | 0.018 | 0.008 to 0.046 | 0.000 | 0.0x | 0.000 | 0.160 | 27.6x | 0.042 | 0.080 | 13.8x | 0.042 |
| Persistence | 0.481 | 0.328 to 0.608 | 0.620 | 107.1x | 0.083 | 0.620 | 107.1x | 0.222 | 0.665 | 114.9x | 0.347 |
| AEMO P5MIN | 0.417 | 0.212 to 0.558 | 0.780 | 134.8x | 0.125 | 0.670 | 115.8x | 0.236 | 0.645 | 111.5x | 0.375 |
| Random (floor) | 0.006 | 0.003 to 0.009 | 0.000 | 0.0x | 0.000 | 0.000 | 0.0x | 0.000 | 0.005 | 0.9x | 0.014 |
| Perfect foresight (ceiling) | 1.000 | 1.000 to 1.000 | 1.000 | 172.8x | 0.111 | 1.000 | 172.8x | 0.264 | 1.000 | 172.8x | 0.431 |

Base rate 0.0058. Lift is precision@k over that: 1.0x is no
better than guessing.

### Where the tie-break carries the result

`precision@k` breaks ties chronologically. The column below is the same
figure averaged over every possible tie ordering. A gap means the score is
too coarse to rank at that k, which is a fact about the baseline.

| Baseline | k | chronological | expected under random ties |
|---|---|---|---|
| Climatology | 50 | 0.000 | 0.148 |
| Climatology | 100 | 0.160 | 0.148 |
| Climatology | 200 | 0.080 | 0.093 |
| Persistence | 50 | 0.620 | 0.670 |
| Persistence | 100 | 0.620 | 0.624 |
| Persistence | 200 | 0.665 | 0.665 |
| AEMO P5MIN | 50 | 0.780 | 0.693 |
| AEMO P5MIN | 100 | 0.670 | 0.667 |
| AEMO P5MIN | 200 | 0.645 | 0.645 |
| Random (floor) | 50 | 0.000 | 0.000 |
| Random (floor) | 100 | 0.000 | 0.000 |
| Random (floor) | 200 | 0.005 | 0.005 |
| Perfect foresight (ceiling) | 50 | 1.000 | 1.000 |
| Perfect foresight (ceiling) | 100 | 1.000 | 1.000 |
| Perfect foresight (ceiling) | 200 | 1.000 | 1.000 |

### The boolean operating point at $300

PLAN §2.1 defines the AEMO and persistence baselines as booleans. Scored
that way they cannot be ranked, so the tables above use the underlying
magnitude. This is the decision the boolean rule actually makes.

| Baseline | alerts | precision | recall |
|---|---|---|---|
| AEMO P5MIN | 605 | 0.473 | 0.600 |
| Persistence | 514 | 0.564 | 0.608 |

## Fold structure

Expanding window, folds sized by spiking days rather than by calendar or
by episode count, with a 24-hour purge gap between train and test.

| fold | train_rows | test_from | test_to | test_rows | positive_rows | spiking_days |
|---|---|---|---|---|---|---|
| 0 | 25,560 | 2025-11-10 18:07 | 2025-11-21 18:02 | 3,168 | 53 | 5 |
| 1 | 28,728 | 2025-11-21 18:07 | 2025-12-09 18:02 | 5,184 | 75 | 5 |
| 2 | 33,913 | 2025-12-09 18:07 | 2026-01-10 18:02 | 9,216 | 148 | 5 |
| 3 | 43,128 | 2026-01-10 18:07 | 2026-08-24 23:21 | 64,856 | 201 | 9 |

