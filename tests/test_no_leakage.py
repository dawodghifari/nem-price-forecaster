"""The leakage test, written before the feature table it guards.

PLAN §4.1 says to write this on Day 4, before any model is trained. Doing so is
only worth something if the check is known to fire, so every invariant here is
proved twice: once against data that satisfies it, and once against data with a
leak planted on purpose. A test that has only ever passed is not evidence.

Two invariants, because Day 1 turned up a second one the plan does not mention:

    features   nothing readable only after the prediction was made
    target     nothing already readable when the prediction was made

The second exists because AEMO publishes a dispatch price about two minutes
before the interval it belongs to ends. Predicting an interval whose price is
already public is not forecasting, and no feature-side check would object.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from nemfc.leakage import (
    check_features_not_from_the_future,
    check_no_incomplete_days,
    check_target_not_already_known,
    raise_on_any_leak,
)

BASE = datetime(2026, 3, 1, 4, 0, tzinfo=timezone.utc)
FIVE_MIN = timedelta(minutes=5)


def _rows(n: int = 20, **overrides) -> pd.DataFrame:
    """A clean feature table: features known by t, target known strictly after.

    Modelled on the real shape. Prediction time is the moment the anchor
    interval's price was published; the target is the following six intervals,
    the first of which becomes readable five minutes later.
    """
    prediction_at = [BASE + i * FIVE_MIN for i in range(n)]
    frame = pd.DataFrame(
        {
            "prediction_at": prediction_at,
            # The anchor's own publication defines t, so equality is expected
            # and must be allowed.
            "max_feature_known_at": prediction_at,
            "min_target_known_at": [t + FIVE_MIN for t in prediction_at],
            "day_is_complete": True,
        }
    )
    for column, value in overrides.items():
        frame[column] = value
    return frame


def test_a_clean_table_passes_every_check():
    raise_on_any_leak(_rows())


def test_a_feature_published_after_the_prediction_is_caught():
    frame = _rows()
    # One row reads a value that appeared a single second too late.
    frame.loc[7, "max_feature_known_at"] += timedelta(seconds=1)

    report = check_features_not_from_the_future(frame)
    assert not report.ok
    assert len(report.violations) == 1
    assert "features not from the future" in report.describe()


def test_a_feature_known_exactly_at_prediction_time_is_allowed():
    """The anchor interval's publication IS the prediction moment."""
    assert check_features_not_from_the_future(_rows()).ok


def test_a_target_already_published_is_caught():
    """The failure mode the plan does not describe.

    A target window starting at the prediction moment contains an interval
    whose price is already public. The features are spotless; the label is not
    a forecast.
    """
    frame = _rows()
    frame.loc[3, "min_target_known_at"] = frame.loc[3, "prediction_at"]

    report = check_target_not_already_known(frame)
    assert not report.ok
    assert len(report.violations) == 1


def test_a_target_published_before_the_prediction_is_caught():
    frame = _rows()
    frame.loc[3, "min_target_known_at"] = frame.loc[3, "prediction_at"] - FIVE_MIN
    assert not check_target_not_already_known(frame).ok


def test_the_naive_thirty_minute_window_fails_this_check():
    """Why the target is anchored to publication rather than to the clock.

    PLAN §2 defines the target over (t, t+30min]. Prices are published about
    two minutes before their interval ends, so the interval ending at t+5 was
    readable at roughly t+3 -- before the prediction. Building the window that
    way puts an already-known interval in the label on every single row.
    """
    naive = _rows()
    naive["min_target_known_at"] = naive["prediction_at"] - timedelta(minutes=2)

    report = check_target_not_already_known(naive)
    assert not report.ok
    assert len(report.violations) == len(naive), "every row should be leaking"


def test_the_archive_to_live_seam_is_caught():
    """The real violation the first build found: four rows in 108,276.

    Where backfilled data meets polled data, the two sides measure vintage
    differently. A live row carries its observed NEMWeb listing time, about two
    minutes before its interval ends. An archive row has no listing time, so it
    gets generation time plus the observed publication lag -- about seven
    minutes after. Both are honest; they are not comparable.

    The consequence is an archive-sourced anchor whose prediction time lands
    nine minutes later than a live one would, making the live intervals that
    follow it look already published. Measured on 16 Aug 2026: anchor interval
    ending 13:55, prediction time 14:02:45, and a target already readable at
    14:02:00.
    """
    anchor_end = datetime(2026, 8, 16, 13, 55, tzinfo=timezone.utc)
    seam = pd.DataFrame(
        {
            # Archive vintage: interval end plus roughly seven minutes.
            "prediction_at": [anchor_end + timedelta(minutes=7, seconds=45)],
            "max_feature_known_at": [anchor_end + timedelta(minutes=7, seconds=45)],
            # Live vintage: whole-minute listing time, two minutes before the
            # interval it belongs to ends.
            "min_target_known_at": [anchor_end + timedelta(minutes=7)],
            "day_is_complete": [True],
        }
    )

    assert check_features_not_from_the_future(seam).ok, "features are fine; the label is not"
    assert not check_target_not_already_known(seam).ok


def test_rows_from_incomplete_days_are_caught():
    frame = _rows()
    frame.loc[[4, 5], "day_is_complete"] = False

    report = check_no_incomplete_days(frame)
    assert not report.ok
    assert len(report.violations) == 2


def test_incomplete_days_are_only_checked_when_the_column_exists():
    """Absence of the column is a missing check, not a silent pass."""
    frame = _rows().drop(columns=["day_is_complete"])
    raise_on_any_leak(frame)  # the other two invariants still run

    with pytest.raises(KeyError, match="day_is_complete"):
        check_no_incomplete_days(frame)


def test_missing_provenance_columns_fail_loudly():
    """A feature table that cannot prove its provenance is not usable.

    Silently skipping the check would be the worst outcome: the pipeline would
    look green while asserting nothing at all.
    """
    frame = _rows().drop(columns=["max_feature_known_at"])
    with pytest.raises(KeyError, match="max_feature_known_at"):
        check_features_not_from_the_future(frame)


def test_raise_on_any_leak_reports_every_failure_not_just_the_first():
    frame = _rows()
    frame.loc[2, "max_feature_known_at"] += FIVE_MIN
    frame.loc[9, "min_target_known_at"] -= FIVE_MIN
    frame.loc[11, "day_is_complete"] = False

    with pytest.raises(AssertionError) as caught:
        raise_on_any_leak(frame)

    message = str(caught.value)
    assert "features not from the future" in message
    assert "target not already known" in message
    assert "incomplete days" in message


def test_an_empty_table_is_reported_as_clean_not_as_an_error():
    """Vacuously true, and worth pinning: it must not raise.

    An empty feature table is a build problem for someone else to catch; this
    check has nothing to say about it.
    """
    empty = _rows(0)
    assert check_features_not_from_the_future(empty).ok
    assert check_target_not_already_known(empty).ok
