"""Tests for the three baselines.

The ones that matter here are about fairness rather than correctness: that
climatology cannot see its own test fold, that the long history sets shape and
not level, and that the continuous scoring of AEMO and persistence agrees with
the boolean rule it replaces at the same threshold.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from nemfc.baselines import (
    aemo_score,
    apply_climatology,
    climatology_out_of_fold,
    day_type,
    fit_climatology,
    long_history_shape,
    operating_point,
    persistence_score,
)
from nemfc.folds import make_folds

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _features(days: int = 200, spike_hour: int = 18) -> pd.DataFrame:
    """A market where spikes happen in the evening, every fourth day."""
    rows = []
    for d in range(days):
        day = BASE + timedelta(days=d)
        for hour in range(24):
            stamp = day + timedelta(hours=hour)
            spike = (d % 4 == 0) and hour == spike_hour
            rows.append(
                {
                    "prediction_at": stamp,
                    "nem_trading_date": date(day.year, day.month, day.day),
                    "nem_hour": hour,
                    "nem_day_of_week": (stamp.weekday() + 1) % 7,
                    "is_spike": spike,
                    "rolling_max_rrp_30m": 900.0 if spike else 60.0,
                    "benchmark_max_forecast_rrp": 800.0 if spike else 55.0,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------


def test_day_type_uses_the_postgres_day_numbering():
    """Sunday is 0 and Saturday is 6 in `extract(dow ...)`, not Monday 0."""
    dow = pd.Series([0, 1, 5, 6])
    assert day_type(dow).tolist() == ["weekend", "weekday", "weekday", "weekend"]


# --------------------------------------------------------------------------
# The long history sets shape, not level
# --------------------------------------------------------------------------


NEM_OFFSET_HOURS = 10


def _history(risky_local_hours: set[int], spacing: int = 30, days: int = 60) -> pd.DataFrame:
    """A long series whose spikes sit at named *NEM-local* hours.

    Timestamps are stored in UTC, as they are in the warehouse, so the fixture
    subtracts the offset the function will add back. Getting this backwards is
    exactly the mistake the offset test below exists to catch.
    """
    rows = []
    for d in range(days):
        for local_hour in range(24):
            stamp = BASE - timedelta(days=days - d) + timedelta(hours=local_hour - NEM_OFFSET_HOURS)
            rows.append(
                {
                    "settlement_end": stamp,
                    "spacing_minutes": spacing,
                    "is_spike": local_hour in risky_local_hours and d % 2 == 0,
                }
            )
    return pd.DataFrame(rows)


def test_the_shape_is_normalised_to_a_mean_of_one():
    """Whatever level the long series implies, it is discarded."""
    shape = long_history_shape(_history({18, 19}))
    assert shape.mean() == pytest.approx(1.0)


def test_the_shape_is_keyed_on_nem_local_hours_not_utc():
    """A ten-hour error here would put the evening peak at breakfast."""
    shape = long_history_shape(_history({18}))
    assert shape[(18, "weekday")] > 0
    assert shape[(8, "weekday")] == pytest.approx(0.0)


def test_the_shape_ranks_the_risky_hours_highest():
    shape = long_history_shape(_history({18, 19}))
    evening = shape[[(18, "weekday"), (18, "weekend")]].mean()
    quiet = shape[[(3, "weekday"), (3, "weekend")]].mean()
    assert evening > quiet


def test_half_hourly_rows_are_weighted_by_the_time_they_cover():
    """Otherwise the post-2021 five-minute years drown the decade before them.

    Same underlying pattern at both resolutions must give the same shape.
    """
    coarse = long_history_shape(_history({18}, spacing=30))
    fine = long_history_shape(_history({18}, spacing=5))
    pd.testing.assert_series_equal(coarse, fine)


# --------------------------------------------------------------------------
# Climatology
# --------------------------------------------------------------------------


def test_climatology_shrinks_a_thin_cell_toward_the_prior():
    """Five positives in a cell is noise, and must be treated as such."""
    frame = _features(days=40)
    rates = fit_climatology(frame, alpha=200.0)
    raw_rate = frame.groupby([frame["nem_hour"]])["is_spike"].mean()

    fitted_evening = rates[(18, "weekday")]
    assert fitted_evening < raw_rate[18]
    assert fitted_evening > frame["is_spike"].mean()


def test_no_smoothing_recovers_the_raw_cell_rate():
    frame = _features(days=40)
    rates = fit_climatology(frame, alpha=0.0)
    subset = frame[(frame["nem_hour"] == 18) & (day_type(frame["nem_day_of_week"]) == "weekday")]
    assert rates[(18, "weekday")] == pytest.approx(subset["is_spike"].mean())


def test_climatology_still_works_without_a_long_history():
    """Weaker, but it must not fail closed."""
    frame = _features(days=40)
    rates = fit_climatology(frame, shape=None, alpha=200.0)
    assert rates.notna().all()
    assert (rates > 0).all()


def test_an_unseen_cell_falls_back_to_the_base_rate_not_to_zero():
    train = _features(days=40)
    rates = fit_climatology(train, alpha=10.0)
    unseen = train.head(1).copy()
    unseen["nem_hour"] = 99  # a cell that cannot be in training

    scored = apply_climatology(rates, unseen, fallback=0.02)
    assert scored[0] == pytest.approx(0.02)


def test_climatology_never_sees_its_own_test_fold():
    """The leak PLAN §2.2 does not guard against, checked directly.

    Spikes are moved to a different hour partway through. If the fitted rates
    had seen the test period, the later folds would score the new hour highly.
    They must not: out of fold, the model only knows the old pattern.
    """
    early = _features(days=120, spike_hour=18)
    late = _features(days=120, spike_hour=6)
    late["prediction_at"] = late["prediction_at"] + timedelta(days=120)
    late["nem_trading_date"] = late["prediction_at"].dt.date
    frame = pd.concat([early, late], ignore_index=True)

    folds = make_folds(frame, min_train_days=90, min_spiking_days=5)
    scores = climatology_out_of_fold(frame, folds)
    frame = frame.assign(clim=scores)

    first_fold_of_the_new_regime = frame[
        (frame["prediction_at"] >= folds[0].test_start)
        & (frame["prediction_at"] < BASE + timedelta(days=125))
    ]
    scored = first_fold_of_the_new_regime.dropna(subset=["clim"])
    if not scored.empty:
        assert scored[scored["nem_hour"] == 18]["clim"].mean() > (
            scored[scored["nem_hour"] == 6]["clim"].mean()
        )


def test_rows_before_the_first_fold_are_left_unscored():
    """They are training data for every fold and belong in no pooled metric."""
    frame = _features(days=200)
    folds = make_folds(frame, min_train_days=90, min_spiking_days=5)
    scores = climatology_out_of_fold(frame, folds)

    before = frame["prediction_at"] < folds[0].test_start
    assert scores[before].isna().all()
    assert scores[~before].notna().all()


# --------------------------------------------------------------------------
# Persistence and AEMO
# --------------------------------------------------------------------------


def test_persistence_and_aemo_are_scored_continuously():
    """More than two distinct values, or precision@k has nothing to rank."""
    frame = _features(days=20)
    frame.loc[3, "rolling_max_rrp_30m"] = 310.0
    frame.loc[4, "benchmark_max_forecast_rrp"] = 450.0

    assert persistence_score(frame).nunique() > 2
    assert aemo_score(frame).nunique() > 2


def test_the_continuous_score_agrees_with_the_boolean_rule_at_its_threshold():
    """The continuous form must not quietly be a different baseline.

    Thresholding it at $300 has to reproduce PLAN §2.1's rule exactly.
    """
    frame = _features(days=20)
    point = operating_point(frame, aemo_score(frame), 300.0)

    boolean_rule = frame["benchmark_max_forecast_rrp"] > 300.0
    hits = int((boolean_rule & frame["is_spike"]).sum())

    assert point["n_alerts"] == int(boolean_rule.sum())
    assert point["precision"] == pytest.approx(hits / boolean_rule.sum())


def test_an_operating_point_with_no_alerts_reports_nan_not_zero():
    """Zero precision would read as a baseline that was wrong, not silent."""
    frame = _features(days=5)
    frame["benchmark_max_forecast_rrp"] = 10.0
    point = operating_point(frame, aemo_score(frame), 300.0)

    assert point["n_alerts"] == 0
    assert np.isnan(point["precision"])
