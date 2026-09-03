"""LightGBM over the walk-forward folds, with the knobs deliberately not turned.

Three choices here are stricter than PLAN §2.3 requires, and each one costs
something.

**No hyperparameter search.** The plan is silent and the instinct is a sweep.
Against 116 episodes a search tunes to the fold structure, and the reported
score stops being out-of-sample in the sense it claims. The parameters below
were fixed before the first run: shallow trees, large leaves, heavy L2. If the
model wins, it wins without anyone having hunted for the configuration where it
does. The cost is that a better model probably exists; that is the right trade
for a number nobody can attack.

**Early stopping validates on the tail of the training window, never the test
fold.** With its own purge gap, so the stopping round cannot be chosen on data
the fold is about to be scored on. Where the validation tail holds too few
positives to mean anything, the harness says so and falls back to a fixed round
count rather than stopping on noise.

**Training is unweighted.** `scale_pos_weight` on a 0.7% class is standard
advice, and it distorts the output probabilities. The deliverable is an endpoint
returning a probability an operator acts on, and every metric here is rank-based
and cannot see the reweighting anyway. If it turns out to matter, that is a
finding to report, not a default to apply quietly.

The benchmark is both a feature and the thing being beaten, which PLAN §2.1
never resolves, so `ModelSpec.use_benchmark` runs it both ways: without it, the
question is whether pre-dispatch can be replaced; with it, whether it can be
improved on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from nemfc.folds import DEFAULT_PURGE, Fold, split

#: Columns AEMO's own forecast contributes. Held separately so the ablation is
#: a change of one flag rather than an edit to a list someone has to trust.
BENCHMARK_FEATURES = ("benchmark_max_forecast_rrp", "benchmark_predicts_spike")

#: Everything else the mart offers that was knowable at prediction time.
BASE_FEATURES = (
    "rrp_now",
    "rrp_lag_5m",
    "rrp_lag_10m",
    "rrp_lag_15m",
    "rrp_lag_30m",
    "rrp_lag_60m",
    "rolling_max_rrp_30m",
    "rolling_mean_rrp_30m",
    "rolling_max_rrp_60m",
    "rolling_mean_rrp_60m",
    "rolling_std_rrp_60m",
    "rolling_max_rrp_240m",
    "rolling_mean_rrp_240m",
    "spikes_trailing_24h",
    "total_demand",
    "demand_ramp_30m",
    "demand_vs_same_time_last_week",
    "available_generation",
    "local_generation_balance",
    "net_interchange",
    "solar_forecast",
    "wind_forecast",
    "battery_max_available",
    "binding_interconnectors",
    "max_abs_marginal_value",
    "apc_flag",
    "market_suspended_flag",
    "cumulative_price",
    "temperature_c",
    "apparent_temperature_c",
    "wind_speed_100m",
    "cloud_cover_pct",
    "nem_hour",
    "nem_day_of_week",
    "nem_month",
)

CATEGORICAL_FEATURES = ("nem_hour", "nem_day_of_week", "nem_month")

#: Columns that are the answer, or are derived from it. A feature list is a
#: thing people edit, and the one edit that destroys the project is adding one
#: of these. Checked rather than trusted.
FORBIDDEN_FEATURES = frozenset(
    {
        "is_spike",
        "target_max_rrp",
        "target_window_start",
        "target_window_end",
        "min_target_known_at",
        "episode_id",
    }
)

#: Fixed before the first run. Small trees and a large minimum leaf because
#: there are 116 events, not 116,000.
DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 5,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "verbosity": -1,
    "seed": 0,
    "deterministic": True,
    "force_row_wise": True,
}

NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 50

#: Days of the training tail reserved for early stopping.
VALIDATION_DAYS = 28

#: Below this many positives the validation curve is noise, and stopping on it
#: is worse than not stopping at all.
MIN_VALIDATION_POSITIVES = 15


@dataclass(frozen=True)
class ModelSpec:
    """One configuration, named so its results can be told apart."""

    name: str
    use_benchmark: bool = True
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    num_boost_round: int = NUM_BOOST_ROUND

    def features(self) -> list[str]:
        columns = list(BASE_FEATURES)
        if self.use_benchmark:
            columns += list(BENCHMARK_FEATURES)
        return columns


@dataclass
class FoldReport:
    """What happened in one fold, so a strange pooled score can be traced."""

    index: int
    n_train: int
    n_valid: int
    n_valid_positives: int
    n_test: int
    best_iteration: int
    early_stopped: bool
    #: Gain per feature for this fold's model. Carried here so importance costs
    #: no extra training run; with this few events one fold's ranking is close
    #: to arbitrary and only the average across folds is worth reading.
    importance: dict[str, float] = field(default_factory=dict)


def check_features(columns: list[str]) -> None:
    """Refuse to train on anything derived from the answer."""
    offending = sorted(FORBIDDEN_FEATURES.intersection(columns))
    if offending:
        raise ValueError(f"target-derived columns in the feature list: {offending}")


def prepare(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Feature matrix with booleans as integers and categoricals as categories.

    Missing values are left as NaN. LightGBM routes them at each split, and the
    nulls in this table are meaningful -- a null rolling statistic means the
    window was short, which is information rather than an inconvenience.
    """
    check_features(features)

    # Named before pandas gets a chance to raise. A bare KeyError here arrives
    # thirty frames deep in `.loc`, mid-backtest, and says nothing about which
    # side is wrong -- the query, the mart, or the list.
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise KeyError(
            f"{len(missing)} feature columns are not in the frame: {missing}. "
            "The loader and the feature list disagree; check that the query "
            "returns them and that dbt still builds them."
        )

    matrix = frame.loc[:, features].copy()

    for column in matrix.columns:
        if matrix[column].dtype == bool:
            matrix[column] = matrix[column].astype("int8")
        elif matrix[column].dtype == object:
            matrix[column] = pd.to_numeric(matrix[column], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        if column in matrix.columns:
            matrix[column] = matrix[column].astype("float64").astype("category")

    return matrix


def inner_split(
    train: pd.DataFrame,
    *,
    time_column: str = "prediction_at",
    label_column: str = "is_spike",
    validation_days: int = VALIDATION_DAYS,
    purge: timedelta = DEFAULT_PURGE,
    min_positives: int = MIN_VALIDATION_POSITIVES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the training window into an inner train and a validation tail.

    The tail widens until it holds enough positives to stop on, because a fixed
    28 days lands in a month with no events often enough to matter -- 40 of the
    50 spiking days in this archive fall in five months. Returns an empty
    validation frame when even the whole window cannot supply them, which the
    caller reads as "do not early stop".
    """
    if train.empty:
        return train, train.iloc[:0]

    times = train[time_column]
    end = times.max()
    span_days = max((end - times.min()).days, 1)

    for days in range(validation_days, span_days + 1, 7):
        cutoff = end - pd.Timedelta(days=days)
        valid = train[times >= cutoff]
        inner = train[times < cutoff - purge]
        if inner.empty:
            break
        if int(valid[label_column].astype(bool).sum()) >= min_positives:
            return inner, valid

    return train, train.iloc[:0]


def fit_fold(
    train: pd.DataFrame,
    spec: ModelSpec,
    *,
    label_column: str = "is_spike",
    time_column: str = "prediction_at",
):
    """Train one fold's model. Returns the booster and what it took."""
    import lightgbm as lgb

    features = spec.features()
    inner, valid = inner_split(train, time_column=time_column, label_column=label_column)

    train_set = lgb.Dataset(
        prepare(inner, features),
        label=inner[label_column].astype(int),
        free_raw_data=False,
    )

    callbacks = [lgb.log_evaluation(period=0)]
    valid_sets = []
    early_stopped = False
    if not valid.empty:
        valid_sets = [
            lgb.Dataset(
                prepare(valid, features),
                label=valid[label_column].astype(int),
                reference=train_set,
                free_raw_data=False,
            )
        ]
        callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False))
        early_stopped = True

    booster = lgb.train(
        spec.params,
        train_set,
        num_boost_round=spec.num_boost_round,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )

    report = FoldReport(
        index=-1,
        n_train=len(inner),
        n_valid=len(valid),
        n_valid_positives=int(valid[label_column].astype(bool).sum()) if not valid.empty else 0,
        n_test=0,
        best_iteration=booster.best_iteration or spec.num_boost_round,
        early_stopped=early_stopped,
        importance=dict(
            zip(
                booster.feature_name(),
                (float(g) for g in booster.feature_importance(importance_type="gain")),
                strict=True,
            )
        ),
    )
    return booster, report


def predict_out_of_fold(
    frame: pd.DataFrame,
    folds: list[Fold],
    spec: ModelSpec,
    *,
    time_column: str = "prediction_at",
    label_column: str = "is_spike",
) -> tuple[pd.Series, list[FoldReport]]:
    """Train once per fold and predict that fold's test rows.

    Rows before the first fold stay NaN. They are training data for every fold
    and belong in no pooled metric, which is the same rule climatology follows,
    so the two are scored on identical rows.
    """
    scores = pd.Series(np.nan, index=frame.index, dtype=float)
    reports: list[FoldReport] = []

    for fold in folds:
        train, test = split(frame, fold, time_column=time_column)
        if train.empty or test.empty:
            continue

        # The invariant the whole project rests on, asserted per fold rather
        # than assumed from the fold construction.
        if not train[time_column].max() < test[time_column].min():
            raise AssertionError(f"fold {fold.index}: a training row is not before the test rows")

        booster, report = fit_fold(train, spec, label_column=label_column, time_column=time_column)
        report.index = fold.index
        report.n_test = len(test)
        reports.append(report)

        scores.loc[test.index] = booster.predict(
            prepare(test, spec.features()), num_iteration=booster.best_iteration or None
        )

    return scores, reports


def aggregate_importance(reports: list[FoldReport], *, top: int = 15) -> pd.DataFrame:
    """Mean gain per feature across folds.

    Averaged rather than read off one model: against 116 events a single
    fold's ranking is close to arbitrary. Reported as a share of total gain so
    the numbers mean something without knowing LightGBM's units.
    """
    totals: dict[str, float] = {}
    for report in reports:
        for name, gain in report.importance.items():
            totals[name] = totals.get(name, 0.0) + gain

    if not totals or not reports:
        return pd.DataFrame(columns=["feature", "mean_gain", "share"])

    grand = sum(totals.values()) or 1.0
    table = pd.DataFrame(
        [
            {
                "feature": name,
                "mean_gain": total / len(reports),
                "share": total / grand,
            }
            for name, total in totals.items()
        ]
    ).sort_values("mean_gain", ascending=False)
    return table.head(top).reset_index(drop=True)
