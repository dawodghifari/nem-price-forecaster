# Day 6: model and backtest

Generated 01 Sep 2026 10:03 UTC by `scripts/train.py`. Not hand-edited.

## Setup

LightGBM, expanding-window walk-forward, folds sized by spiking days,
24-hour purge between train and test, early stopping on a purged tail of
each fold's own training window. No hyperparameter search: against this
many events a search tunes to the fold structure and the reported score
stops being out-of-sample in the sense it claims. Training is unweighted,
so the output probabilities stay calibrated.

```
bagging_fraction: 0.7
bagging_freq: 1
deterministic: True
feature_fraction: 0.7
force_row_wise: True
lambda_l2: 10.0
learning_rate: 0.05
max_depth: 5
metric: average_precision
min_data_in_leaf: 200
num_leaves: 15
objective: binary
seed: 0
verbosity: -1
```

## The full evaluation set

99,704 rows, 683 positive, 116 episodes across 39 days.

| Score | PR-AUC | 95% CI | p@50 | lift@50 | ep.recall@50 | p@100 | lift@100 | ep.recall@100 | p@200 | lift@200 | ep.recall@200 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Climatology | 0.025 | 0.014 to 0.044 | 0.000 | 0.0x | 0.000 | 0.000 | 0.0x | 0.000 | 0.000 | 0.0x | 0.000 |
| Persistence | 0.399 | 0.274 to 0.514 | 0.640 | 93.4x | 0.060 | 0.580 | 84.7x | 0.129 | 0.575 | 83.9x | 0.216 |
| AEMO P5MIN | 0.356 | 0.222 to 0.484 | 0.800 | 116.8x | 0.086 | 0.630 | 92.0x | 0.129 | 0.605 | 88.3x | 0.267 |
| Random (floor) | 0.007 | 0.004 to 0.010 | 0.000 | 0.0x | 0.000 | 0.000 | 0.0x | 0.000 | 0.010 | 1.5x | 0.017 |
| Perfect foresight (ceiling) | 1.000 | 1.000 to 1.000 | 1.000 | 146.0x | 0.078 | 1.000 | 146.0x | 0.164 | 1.000 | 146.0x | 0.293 |
| Model (no AEMO feature) | 0.470 | 0.324 to 0.606 | 0.620 | 90.5x | 0.147 | 0.760 | 110.9x | 0.216 | 0.775 | 113.1x | 0.353 |
| Model (+ AEMO feature) | 0.498 | 0.342 to 0.629 | 0.720 | 105.1x | 0.129 | 0.730 | 106.6x | 0.250 | 0.790 | 115.3x | 0.431 |
| Shuffled labels (leak control) | 0.007 | 0.004 to 0.013 | 0.000 | 0.0x | 0.000 | 0.000 | 0.0x | 0.000 | 0.000 | 0.0x | 0.000 |

## The onset slice

Rows where nothing in the trailing thirty minutes had already exceeded the
threshold, so persistence is silent by construction. Day 5 measured
persistence at 84x lift because most positives are continuations of an
episode already in progress; this slice keeps only the forecasts that are
genuinely forecasts. It was defined before this script was written.

98,957 rows, 302 positive, 55 episodes across 39 days.

| Score | PR-AUC | 95% CI | p@50 | lift@50 | ep.recall@50 | p@100 | lift@100 | ep.recall@100 | p@200 | lift@200 | ep.recall@200 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Climatology | 0.014 | 0.007 to 0.026 | 0.000 | 0.0x | 0.000 | 0.000 | 0.0x | 0.000 | 0.000 | 0.0x | 0.000 |
| Persistence | 0.057 | 0.033 to 0.098 | 0.000 | 0.0x | 0.000 | 0.080 | 26.2x | 0.036 | 0.075 | 24.6x | 0.073 |
| AEMO P5MIN | 0.090 | 0.048 to 0.166 | 0.380 | 124.5x | 0.127 | 0.250 | 81.9x | 0.145 | 0.200 | 65.5x | 0.291 |
| Random (floor) | 0.003 | 0.002 to 0.005 | 0.000 | 0.0x | 0.000 | 0.000 | 0.0x | 0.000 | 0.005 | 1.6x | 0.018 |
| Perfect foresight (ceiling) | 1.000 | 1.000 to 1.000 | 1.000 | 327.7x | 0.182 | 1.000 | 327.7x | 0.382 | 1.000 | 327.7x | 0.727 |
| Model (no AEMO feature) | 0.100 | 0.061 to 0.171 | 0.300 | 98.3x | 0.091 | 0.180 | 59.0x | 0.127 | 0.205 | 67.2x | 0.273 |
| Model (+ AEMO feature) | 0.108 | 0.065 to 0.185 | 0.300 | 98.3x | 0.127 | 0.280 | 91.7x | 0.218 | 0.205 | 67.2x | 0.291 |
| Shuffled labels (leak control) | 0.004 | 0.002 to 0.006 | 0.000 | 0.0x | 0.000 | 0.000 | 0.0x | 0.000 | 0.000 | 0.0x | 0.000 |

## Feature importance

Mean gain across folds, as a share of the total. One fold's ranking is
close to arbitrary at this event count; only the average is worth reading.

Both models are shown. The one without AEMO's forecast is the more
informative of the two: it has to find the shortage some other way, and
what it reaches for is the answer to what pre-dispatch is actually
contributing.

### Without the AEMO feature

| feature | mean_gain | share |
|---|---|---|
| rrp_now | 7,936.9040 | 0.4158 |
| rolling_max_rrp_30m | 3,130.3360 | 0.1640 |
| rrp_lag_5m | 2,497.8704 | 0.1308 |
| rolling_max_rrp_60m | 1,151.2111 | 0.0603 |
| total_demand | 615.3243 | 0.0322 |
| nem_hour | 614.4615 | 0.0322 |
| demand_ramp_30m | 436.6565 | 0.0229 |
| temperature_c | 324.8239 | 0.0170 |
| spikes_trailing_24h | 280.6160 | 0.0147 |
| rolling_std_rrp_60m | 258.6219 | 0.0135 |
| demand_vs_same_time_last_week | 258.0723 | 0.0135 |
| cumulative_price | 188.6954 | 0.0099 |
| apparent_temperature_c | 177.5005 | 0.0093 |
| solar_forecast | 160.4280 | 0.0084 |
| battery_max_available | 139.7109 | 0.0073 |

### With the AEMO feature

| feature | mean_gain | share |
|---|---|---|
| rrp_now | 6,852.2169 | 0.3617 |
| rolling_max_rrp_30m | 3,409.6227 | 0.1800 |
| benchmark_max_forecast_rrp | 1,583.6849 | 0.0836 |
| rrp_lag_5m | 1,517.9865 | 0.0801 |
| rolling_max_rrp_60m | 1,139.8004 | 0.0602 |
| total_demand | 503.3382 | 0.0266 |
| nem_hour | 482.8555 | 0.0255 |
| benchmark_predicts_spike | 441.0121 | 0.0233 |
| demand_ramp_30m | 387.3482 | 0.0204 |
| demand_vs_same_time_last_week | 326.2298 | 0.0172 |
| rolling_std_rrp_60m | 302.6347 | 0.0160 |
| temperature_c | 284.6459 | 0.0150 |
| spikes_trailing_24h | 264.5402 | 0.0140 |
| apparent_temperature_c | 173.0431 | 0.0091 |
| solar_forecast | 151.7017 | 0.0080 |

## Fold structure

| fold | train_rows | test_from | test_to | test_rows | positive_rows | spiking_days |
|---|---|---|---|---|---|---|
| 0 | 8,281 | 2025-09-11 18:07 | 2025-10-11 18:02 | 8,640 | 70 | 5 |
| 1 | 16,921 | 2025-10-11 18:07 | 2025-10-30 18:02 | 5,472 | 71 | 5 |
| 2 | 22,392 | 2025-10-30 18:07 | 2025-11-08 18:02 | 2,592 | 65 | 5 |
| 3 | 24,985 | 2025-11-08 18:07 | 2025-11-21 18:02 | 3,744 | 53 | 5 |
| 4 | 28,728 | 2025-11-21 18:07 | 2025-12-09 18:02 | 5,184 | 75 | 5 |
| 5 | 33,913 | 2025-12-09 18:07 | 2026-01-10 18:02 | 9,216 | 148 | 5 |
| 6 | 43,128 | 2026-01-10 18:07 | 2026-08-24 23:21 | 64,856 | 201 | 9 |

## Per-fold training detail

`early_stopped` false means the validation tail held too few positives to
stop on, and the fold ran the full round count rather than stopping on noise.

| index | n_train | n_valid | n_valid_positives | n_test | best_iteration | early_stopped |
|---|---|---|---|---|---|---|
| 0 | 8,281 | 0 | 0 | 8,640 | 500 | 0 |
| 1 | 8,568 | 8,065 | 64 | 5,472 | 30 | 1 |
| 2 | 14,039 | 8,065 | 121 | 2,592 | 74 | 1 |
| 3 | 16,633 | 8,064 | 131 | 3,744 | 139 | 1 |
| 4 | 20,376 | 8,064 | 141 | 5,184 | 70 | 1 |
| 5 | 25,560 | 8,065 | 120 | 9,216 | 20 | 1 |
| 6 | 34,776 | 8,065 | 96 | 64,856 | 22 | 1 |

