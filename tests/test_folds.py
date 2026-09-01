"""Tests for the walk-forward split.

The properties that matter are the ones a bug would not announce: no training
row after a test row, a real gap between them, no row silently dropped, and no
fold with zero positives.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from nemfc.folds import DEFAULT_PURGE, describe, make_folds, split

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _frame(days: int = 200, spike_every: int = 4, rows_per_day: int = 12) -> pd.DataFrame:
    rows = []
    for d in range(days):
        day = BASE + timedelta(days=d)
        for i in range(rows_per_day):
            rows.append(
                {
                    "prediction_at": day + timedelta(hours=2 * i),
                    "nem_trading_date": date(day.year, day.month, day.day),
                    "is_spike": (d % spike_every == 0) and i in (5, 6),
                }
            )
    return pd.DataFrame(rows)


def test_every_fold_contains_at_least_the_requested_events():
    folds = make_folds(_frame(), min_spiking_days=5, min_train_days=90)
    assert folds
    for fold in folds:
        assert fold.n_test_spiking_days >= 5
        assert not fold.is_empty


def test_no_fold_is_empty_of_positives():
    """The failure weekly folds would have produced on this data."""
    folds = make_folds(_frame(), min_spiking_days=5, min_train_days=90)
    assert all(f.n_test_positive_rows > 0 for f in folds)


def test_training_never_reaches_past_the_purge_gap():
    frame = _frame()
    for fold in make_folds(frame, min_train_days=90):
        train, test = split(frame, fold)
        assert train["prediction_at"].max() < test["prediction_at"].min()
        gap = test["prediction_at"].min() - train["prediction_at"].max()
        assert gap >= DEFAULT_PURGE


def test_the_purge_is_a_full_day_not_a_window_width():
    """`spikes_trailing_24h` looks back a day, so a shorter gap leaks."""
    assert DEFAULT_PURGE.total_seconds() == 24 * 60 * 60


def test_the_window_expands():
    frame = _frame()
    folds = make_folds(frame, min_train_days=90)
    sizes = [f.n_train_rows for f in folds]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_test_periods_are_contiguous_and_do_not_overlap():
    folds = make_folds(_frame(), min_train_days=90)
    for earlier, later in pairwise(folds):
        assert earlier.test_end < later.test_start


def test_no_day_after_the_first_fold_is_dropped():
    """A silently discarded tail is how a pooled metric goes wrong unnoticed."""
    frame = _frame()
    folds = make_folds(frame, min_train_days=90)

    covered = pd.concat([split(frame, f)[1] for f in folds])
    first_test_start = folds[0].test_start
    expected = frame[frame["prediction_at"] >= first_test_start]

    assert len(covered) == len(expected)
    assert covered["prediction_at"].nunique() == expected["prediction_at"].nunique()


def test_a_short_tail_is_merged_rather_than_left_as_a_thin_fold():
    """Two spiking days left over must not become a fold of their own."""
    frame = _frame(days=160, spike_every=10)
    folds = make_folds(frame, min_spiking_days=5, min_train_days=90)
    assert all(f.n_test_spiking_days >= 5 for f in folds)


def test_the_burn_in_waits_for_events_not_just_for_days():
    """The rule the front-loaded archive forced.

    Spikes here land only in the first stretch, exactly as they do in the real
    data. A calendar-only burn-in would hand the whole spike-rich period to
    training and leave nothing to evaluate on.
    """
    frame = _frame(days=200, spike_every=4)
    # Silence every spike after day 60, leaving a front-loaded distribution.
    late = frame["prediction_at"] >= BASE + timedelta(days=60)
    frame.loc[late, "is_spike"] = False

    folds = make_folds(frame, min_train_days=10, min_train_spiking_days=5, min_spiking_days=2)
    assert folds
    # The burn-in must have stopped early, inside the spiking stretch, rather
    # than running to the calendar floor and beyond.
    assert folds[0].test_start < BASE + timedelta(days=60)
    assert sum(f.n_test_spiking_days for f in folds) > 0


def test_the_calendar_floor_still_applies_when_events_are_dense():
    """Events alone must not let the first model train on a fortnight."""
    frame = _frame(days=200, spike_every=1)
    folds = make_folds(frame, min_train_days=40, min_train_spiking_days=3)
    assert folds[0].test_start >= BASE + timedelta(days=39)


def test_too_few_events_to_burn_in_is_an_error():
    frame = _frame(days=200, spike_every=100)  # two spiking days in total
    with pytest.raises(ValueError, match="no fold could be cut"):
        make_folds(frame, min_train_days=30, min_train_spiking_days=10)


def test_too_little_history_is_an_error_not_a_degenerate_fold():
    with pytest.raises(ValueError, match="no fold could be cut"):
        make_folds(_frame(days=30), min_train_days=90)


def test_an_empty_frame_gives_no_folds():
    empty = _frame(days=0)
    assert make_folds(empty) == []


def test_describe_reports_the_denominator_for_every_fold():
    folds = make_folds(_frame(), min_train_days=90)
    table = describe(folds)
    assert list(table.columns) == [
        "fold",
        "train_rows",
        "test_from",
        "test_to",
        "test_rows",
        "positive_rows",
        "spiking_days",
    ]
    assert (table["positive_rows"] > 0).all()


def test_splitting_is_disjoint():
    frame = _frame()
    for fold in make_folds(frame, min_train_days=90):
        train, test = split(frame, fold)
        overlap = set(train["prediction_at"]) & set(test["prediction_at"])
        assert not overlap


def test_folds_are_deterministic():
    frame = _frame()
    assert make_folds(frame, min_train_days=90) == make_folds(frame, min_train_days=90)


def test_rarer_events_give_fewer_wider_folds():
    """The fold count follows the events, which is the point of the design."""
    dense = make_folds(_frame(spike_every=4), min_train_days=90)
    sparse = make_folds(_frame(spike_every=12), min_train_days=90)
    assert len(sparse) < len(dense)
    assert np.mean([f.n_test_rows for f in sparse]) > np.mean([f.n_test_rows for f in dense])
