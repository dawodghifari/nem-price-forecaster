"""Tests for the training harness.

The point of these is not that LightGBM works. It is that the harness around it
cannot leak, cannot train on the future, and cannot quietly be handed the
answer. Two of them are controls with known outcomes: labels shuffled must
score at the base rate, and the target passed as a feature must score near 1.0.
A harness that fails either is not measuring what it says it measures.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from nemfc.evaluate import average_precision, onset_mask
from nemfc.folds import make_folds, split
from nemfc.model import (
    BASE_FEATURES,
    BENCHMARK_FEATURES,
    ModelSpec,
    check_features,
    fit_fold,
    inner_split,
    predict_out_of_fold,
    prepare,
)

BASE = datetime(2025, 9, 1, tzinfo=timezone.utc)
THRESHOLD = 300.0


def _frame(days: int = 160, seed: int = 3) -> pd.DataFrame:
    """A market with a learnable signal: spikes follow tight supply in the evening."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(days):
        day = BASE + timedelta(days=d)
        tight = rng.random() < 0.25  # a tight day
        for i in range(48):
            stamp = day + timedelta(minutes=30 * i)
            hour = (stamp + timedelta(hours=10)).hour
            evening = hour in (17, 18, 19)
            spiking = tight and evening and rng.random() < 0.35
            rrp = float(rng.uniform(400, 15000)) if spiking else float(rng.normal(70, 20))
            rows.append(
                {
                    "prediction_at": stamp,
                    "anchor_settlement_end": stamp,
                    "nem_trading_date": date(day.year, day.month, day.day),
                    "rrp_now": rrp,
                    "nem_hour": hour,
                    "nem_day_of_week": (stamp.weekday() + 1) % 7,
                    "nem_month": stamp.month,
                    "local_generation_balance": -800.0 if tight else 2500.0,
                    "total_demand": 9000.0 + (1800.0 if evening else 0.0),
                    "benchmark_max_forecast_rrp": rrp * 0.5 if spiking else rrp,
                }
            )
    frame = pd.DataFrame(rows)

    future = frame["rrp_now"].shift(-1).rolling(2).max().shift(-1)
    frame["target_max_rrp"] = future
    frame["is_spike"] = (future > THRESHOLD).fillna(False)
    frame["rolling_max_rrp_30m"] = frame["rrp_now"].rolling(2, min_periods=2).max()

    for column in BASE_FEATURES:
        if column not in frame.columns:
            frame[column] = rng.normal(size=len(frame))
    for column in BENCHMARK_FEATURES:
        if column not in frame.columns:
            frame[column] = frame["benchmark_max_forecast_rrp"] > THRESHOLD

    return frame.dropna(subset=["target_max_rrp"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def data():
    frame = _frame()
    folds = make_folds(frame, min_train_days=45, min_train_spiking_days=5, min_spiking_days=4)
    return frame, folds


# --------------------------------------------------------------------------
# The feature list
# --------------------------------------------------------------------------


def test_a_target_derived_column_is_refused():
    """The one edit to a feature list that destroys the project."""
    with pytest.raises(ValueError, match="target-derived"):
        check_features([*BASE_FEATURES, "target_max_rrp"])

    with pytest.raises(ValueError, match="target-derived"):
        check_features([*BASE_FEATURES, "is_spike"])


def test_the_benchmark_is_included_only_when_asked():
    assert "benchmark_max_forecast_rrp" in ModelSpec("with", use_benchmark=True).features()
    assert "benchmark_max_forecast_rrp" not in ModelSpec("without", use_benchmark=False).features()


def test_a_missing_feature_column_says_which_one():
    """What the first Day 6 run should have printed instead of a stack trace."""
    frame = _frame(days=10).drop(columns=["wind_speed_100m", "cloud_cover_pct"])
    with pytest.raises(KeyError, match="wind_speed_100m"):
        prepare(frame, list(BASE_FEATURES))


def test_nulls_survive_preparation():
    """A null rolling statistic means a short window, which is information."""
    frame = _frame(days=20)
    frame.loc[0, "rolling_max_rrp_30m"] = np.nan
    assert np.isnan(prepare(frame, list(BASE_FEATURES)).loc[0, "rolling_max_rrp_30m"])


# --------------------------------------------------------------------------
# The inner validation split
# --------------------------------------------------------------------------


def test_the_validation_tail_is_purged_from_the_inner_training_rows(data):
    frame, folds = data
    train, _ = split(frame, folds[-1])
    inner, valid = inner_split(train)

    assert not valid.empty
    gap = valid["prediction_at"].min() - inner["prediction_at"].max()
    assert gap >= timedelta(hours=24)


def test_the_validation_tail_widens_until_it_holds_events(data):
    """A fixed 28 days lands in an empty month often enough to matter."""
    frame, folds = data
    train, _ = split(frame, folds[-1])
    _, valid = inner_split(train, min_positives=40)
    if not valid.empty:
        assert int(valid["is_spike"].sum()) >= 40


def test_no_validation_set_when_the_events_are_not_there(data):
    """Better to skip early stopping than to stop on noise."""
    frame, folds = data
    train, _ = split(frame, folds[0])
    _, valid = inner_split(train, min_positives=10**6)
    assert valid.empty


def test_a_fold_with_no_validation_still_trains(data):
    frame, folds = data
    train, _ = split(frame, folds[0])
    train = train.assign(is_spike=False)
    train.loc[train.index[:5], "is_spike"] = True

    booster, report = fit_fold(train, ModelSpec("tiny"))
    assert booster is not None
    assert not report.early_stopped


# --------------------------------------------------------------------------
# The controls
# --------------------------------------------------------------------------


def test_the_model_learns_something_real(data):
    frame, folds = data
    scores, reports = predict_out_of_fold(frame, folds, ModelSpec("with benchmark"))

    scored = frame[scores.notna()]
    value = average_precision(scored["is_spike"].to_numpy(), scores[scores.notna()].to_numpy())
    assert value > 3 * scored["is_spike"].mean()
    assert len(reports) == len(folds)


def test_shuffled_labels_score_at_the_base_rate(data):
    """If this beats chance, the harness is leaking and nothing else counts."""
    frame, folds = data
    rng = np.random.default_rng(0)
    shuffled = frame.assign(is_spike=rng.permutation(frame["is_spike"].to_numpy()))

    scores, _ = predict_out_of_fold(shuffled, folds, ModelSpec("shuffled"))
    scored_mask = scores.notna()
    value = average_precision(
        shuffled.loc[scored_mask, "is_spike"].to_numpy(), scores[scored_mask].to_numpy()
    )
    base = float(shuffled.loc[scored_mask, "is_spike"].mean())
    assert value < base * 3, f"shuffled labels scored {value:.4f} against a base rate of {base:.4f}"


def test_handing_the_model_the_answer_scores_near_one(data):
    """The other end: the control must be capable of detecting a leak."""
    frame, folds = data
    leaky = frame.assign(cumulative_price=frame["target_max_rrp"])

    scores, _ = predict_out_of_fold(leaky, folds, ModelSpec("leaky"))
    scored_mask = scores.notna()
    value = average_precision(
        leaky.loc[scored_mask, "is_spike"].to_numpy(), scores[scored_mask].to_numpy()
    )
    assert value > 0.9, f"a planted leak only scored {value:.3f}: the control cannot see leaks"


def test_no_training_row_is_later_than_a_test_row(data):
    """Asserted inside the harness too; pinned here so the assert stays."""
    frame, folds = data
    for fold in folds:
        train, test = split(frame, fold)
        assert train["prediction_at"].max() < test["prediction_at"].min()


def test_rows_before_the_first_fold_are_left_unscored(data):
    """Same rule climatology follows, so the two are scored on identical rows."""
    frame, folds = data
    scores, _ = predict_out_of_fold(frame, folds, ModelSpec("m"))

    before = frame["prediction_at"] < folds[0].test_start
    assert scores[before].isna().all()
    assert scores[~before].notna().all()


def test_predictions_are_reproducible(data):
    frame, folds = data
    first, _ = predict_out_of_fold(frame, folds, ModelSpec("m"))
    second, _ = predict_out_of_fold(frame, folds, ModelSpec("m"))
    pd.testing.assert_series_equal(first, second)


# --------------------------------------------------------------------------
# The onset slice
# --------------------------------------------------------------------------


def test_the_onset_slice_excludes_rows_persistence_can_already_see():
    frame = _frame(days=40)
    mask = onset_mask(frame, THRESHOLD)
    assert (frame.loc[mask, "rolling_max_rrp_30m"] <= THRESHOLD).all()


def test_a_row_with_no_trailing_window_is_not_called_an_onset():
    """An unverifiable onset is not an onset."""
    frame = _frame(days=20)
    frame.loc[0, "rolling_max_rrp_30m"] = np.nan
    assert not onset_mask(frame, THRESHOLD).iloc[0]


def test_the_onset_slice_is_the_harder_half():
    """Fewer positives per row than the full set, by construction."""
    frame = _frame(days=120)
    mask = onset_mask(frame, THRESHOLD)
    assert frame.loc[mask, "is_spike"].mean() < frame["is_spike"].mean()
    assert int(frame.loc[mask, "is_spike"].sum()) > 0
