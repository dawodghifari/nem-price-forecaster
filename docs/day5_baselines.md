# Day 5: baselines

Generated 01 Sep 2026 05:27 UTC by `scripts/baselines.py`. Not hand-edited.

## The comparison set

- Rows scored: 99,704 (of 108,272 in the feature table)
- Positive rows: 683
- Spike episodes: 116 across 39 days
- Rows AEMO published a forecast for: 99,704 (100.0%)
- Positive rows with no AEMO forecast: 0
- Rows with a full 30-minute rolling window: 99,698

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
| Climatology | 0.025 | 0.014 to 0.044 | 0.000 | 0.0x | 0.000 | 0.000 | 0.0x | 0.000 | 0.000 | 0.0x | 0.000 |
| Persistence | 0.399 | 0.274 to 0.524 | 0.640 | 93.4x | 0.060 | 0.580 | 84.7x | 0.129 | 0.575 | 83.9x | 0.216 |
| AEMO P5MIN | 0.356 | 0.219 to 0.483 | 0.800 | 116.8x | 0.086 | 0.630 | 92.0x | 0.129 | 0.605 | 88.3x | 0.267 |
| Random (floor) | 0.007 | 0.004 to 0.010 | 0.000 | 0.0x | 0.000 | 0.000 | 0.0x | 0.000 | 0.010 | 1.5x | 0.017 |
| Perfect foresight (ceiling) | 1.000 | 1.000 to 1.000 | 1.000 | 146.0x | 0.078 | 1.000 | 146.0x | 0.164 | 1.000 | 146.0x | 0.293 |

Base rate 0.0069. Lift is precision@k over that: 1.0x is no
better than guessing.

### Where the tie-break carries the result

`precision@k` breaks ties chronologically. The column below is the same
figure averaged over every possible tie ordering. A gap means the score is
too coarse to rank at that k, which is a fact about the baseline.

| Baseline | k | chronological | expected under random ties |
|---|---|---|---|
| Climatology | 50 | 0.000 | 0.000 |
| Climatology | 100 | 0.000 | 0.000 |
| Climatology | 200 | 0.000 | 0.000 |
| Persistence | 50 | 0.640 | 0.655 |
| Persistence | 100 | 0.580 | 0.574 |
| Persistence | 200 | 0.575 | 0.575 |
| AEMO P5MIN | 50 | 0.800 | 0.690 |
| AEMO P5MIN | 100 | 0.630 | 0.640 |
| AEMO P5MIN | 200 | 0.605 | 0.605 |
| Random (floor) | 50 | 0.000 | 0.000 |
| Random (floor) | 100 | 0.000 | 0.000 |
| Random (floor) | 200 | 0.010 | 0.010 |
| Perfect foresight (ceiling) | 50 | 1.000 | 1.000 |
| Perfect foresight (ceiling) | 100 | 1.000 | 1.000 |
| Perfect foresight (ceiling) | 200 | 1.000 | 1.000 |

### The boolean operating point at $300

PLAN §2.1 defines the AEMO and persistence baselines as booleans. Scored
that way they cannot be ranked, so the tables above use the underlying
magnitude. This is the decision the boolean rule actually makes.

| Baseline | alerts | precision | recall |
|---|---|---|---|
| AEMO P5MIN | 825 | 0.432 | 0.521 |
| Persistence | 741 | 0.514 | 0.558 |

## Fold structure

Expanding window, folds sized by spiking days rather than by calendar or
by episode count, with a 24-hour purge gap between train and test.

| fold | train_rows | test_from | test_to | test_rows | positive_rows | spiking_days |
|---|---|---|---|---|---|---|
| 0 | 8,281 | 2025-09-11 18:07 | 2025-10-11 18:02 | 8,640 | 70 | 5 |
| 1 | 16,921 | 2025-10-11 18:07 | 2025-10-30 18:02 | 5,472 | 71 | 5 |
| 2 | 22,392 | 2025-10-30 18:07 | 2025-11-08 18:02 | 2,592 | 65 | 5 |
| 3 | 24,985 | 2025-11-08 18:07 | 2025-11-21 18:02 | 3,744 | 53 | 5 |
| 4 | 28,728 | 2025-11-21 18:07 | 2025-12-09 18:02 | 5,184 | 75 | 5 |
| 5 | 33,913 | 2025-12-09 18:07 | 2026-01-10 18:02 | 9,216 | 148 | 5 |
| 6 | 43,128 | 2026-01-10 18:07 | 2026-08-24 23:21 | 64,856 | 201 | 9 |

