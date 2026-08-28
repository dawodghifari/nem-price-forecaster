"""Point-in-time correctness, as assertions rather than prose.

PLAN §4.1 names leakage as the project's largest risk and describes one half of
it: a feature must not carry information published after the prediction was
made. Day 1 turned up the other half.

**Dispatch prices are set before their interval ends.** AEMO solves dispatch at
the start of a five-minute interval and publishes the result about two minutes
before the interval's own timestamp. So at any given moment the price of the
interval ending shortly *ahead* is already public. A target defined as "any
spike in the next thirty minutes" therefore contains at least one interval
whose answer is already sitting in the database, and no feature-side check
would notice: nothing leaked into the features, the label simply was not a
prediction.

Hence two invariants, not one:

    features   every contributing row was readable at or before prediction time
    target     every interval being predicted was NOT readable at that moment

Both are expressed here as functions over a dataframe so that the same code
can be pointed at a synthetic fixture, where a leak is planted deliberately,
and at the real feature table. A check that has only ever passed is not
evidence that it works.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Columns the feature table carries so that it can prove its own correctness.
# Written by the mart, verified here, and re-derived independently from raw by
# tests/test_no_leakage.py -- a table asserting its own integrity is only worth
# as much as the independent check behind it.
PREDICTION_AT = "prediction_at"
MAX_FEATURE_KNOWN_AT = "max_feature_known_at"
MIN_TARGET_KNOWN_AT = "min_target_known_at"


@dataclass
class LeakageReport:
    """What a check found. Empty `violations` means the invariant held."""

    invariant: str
    checked: int
    violations: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def ok(self) -> bool:
        return len(self.violations) == 0

    def describe(self) -> str:
        if self.ok:
            return f"{self.invariant}: clean over {self.checked} rows"
        worst = self.violations.head(3).to_dict("records")
        return (
            f"{self.invariant}: {len(self.violations)} of {self.checked} rows "
            f"violate it. First few: {worst}"
        )

    def raise_if_violated(self) -> None:
        if not self.ok:
            raise AssertionError(self.describe())


def check_features_not_from_the_future(frame: pd.DataFrame) -> LeakageReport:
    """No feature may come from a row published after the prediction was made.

    Equality is allowed: a row that became readable at exactly the prediction
    moment was readable, and the anchor interval is precisely such a row -- its
    publication is what defines the prediction time.
    """
    _require(frame, [PREDICTION_AT, MAX_FEATURE_KNOWN_AT])
    bad = frame[frame[MAX_FEATURE_KNOWN_AT] > frame[PREDICTION_AT]]
    return LeakageReport("features not from the future", len(frame), bad)


def check_target_not_already_known(frame: pd.DataFrame) -> LeakageReport:
    """Every interval being predicted must still have been unpublished.

    Strict inequality: an interval readable at exactly the prediction moment is
    already known, so predicting it is not forecasting.
    """
    _require(frame, [PREDICTION_AT, MIN_TARGET_KNOWN_AT])
    bad = frame[frame[MIN_TARGET_KNOWN_AT] <= frame[PREDICTION_AT]]
    return LeakageReport("target not already known", len(frame), bad)


def check_no_incomplete_days(frame: pd.DataFrame, column: str = "day_is_complete") -> LeakageReport:
    """Rows from days AEMO did not fully publish must not be present.

    Not leakage in the strict sense, but the same class of silent error: a
    rolling feature spanning an outage is computed over fewer intervals than it
    believes, and reports a number rather than a gap.
    """
    _require(frame, [column])
    bad = frame[~frame[column].astype(bool)]
    return LeakageReport("no rows from incomplete days", len(frame), bad)


def check_all(frame: pd.DataFrame) -> list[LeakageReport]:
    reports = [
        check_features_not_from_the_future(frame),
        check_target_not_already_known(frame),
    ]
    if "day_is_complete" in frame.columns:
        reports.append(check_no_incomplete_days(frame))
    return reports


def raise_on_any_leak(frame: pd.DataFrame) -> None:
    failures = [r for r in check_all(frame) if not r.ok]
    if failures:
        raise AssertionError("\n".join(r.describe() for r in failures))


def _require(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(
            f"cannot check point-in-time correctness without {missing}. "
            "The feature table must carry its own provenance columns."
        )
