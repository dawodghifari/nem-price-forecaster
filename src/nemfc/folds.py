"""Walk-forward folds sized by events rather than by the calendar.

PLAN §2.3 rejected weekly folds after the arithmetic: 13 months of five-minute
intervals split into ~56 weeks, against roughly 100-200 spike episodes, leaves
most weeks with zero positives, PR-AUC undefined on them, and the mean across
folds decided by a handful of weeks.

The measured numbers came in at 144 episodes over 50 days, so folds here are
sized by **spiking days**, not by episodes and not by weeks. Episodes cluster --
2.9 per spiking day -- and two episodes on one evening share their weather,
their demand and the shortfall that caused them. A fold holding ten episodes
drawn from three days is far weaker evidence than one holding ten from ten days,
and sizing by episode count hides the difference.

Two rules the plan states and one it does not:

**Expanding window.** Training data is everything before the fold, never after.
A fold's model has seen only what an operator would have had.

**Purge gap.** PLAN §2.3 asks for a gap "to stop rolling-window features
straddling the boundary", which suggests one window's width. That is not
enough. `spikes_trailing_24h` looks back a full day, so a test row's features
can be computed from intervals inside the training period; and two rows thirty
minutes apart share five of their six target intervals, so the *label* leaks
across the boundary too. The gap therefore defaults to 24 hours -- the longest
feature lookback -- which subsumes the target overlap.

**No silent discards.** Days after the last complete fold are merged into it
rather than dropped. A tail quietly falling out of the evaluation is the kind
of thing that makes a pooled metric wrong in a way nobody notices.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

#: The longest feature lookback in the mart, and so the shortest defensible gap.
DEFAULT_PURGE = timedelta(hours=24)

#: Spiking days per test fold. Five gives ten folds against the measured 50.
DEFAULT_MIN_SPIKING_DAYS = 5

#: Calendar floor on the first training window. A model fitted on a fortnight
#: is not worth evaluating however many events that fortnight happened to hold.
DEFAULT_MIN_TRAIN_DAYS = 30

#: Events the first training window must contain.
#:
#: This started as a flat 90-day burn-in, and the data showed that to be the
#: wrong rule. Spikes in this archive are heavily front-loaded -- 40 of the 50
#: spiking days fall in Aug-Dec 2025, against 10 across the following eight
#: months -- so a calendar burn-in discarded 26 spiking days, more than half the
#: evidence, before evaluation began. The rest of this module already sizes by
#: events rather than by calendar; the burn-in was simply the one place that
#: rule was not applied.
DEFAULT_MIN_TRAIN_SPIKING_DAYS = 10


@dataclass(frozen=True)
class Fold:
    """One expanding-window split, with the counts needed to judge it."""

    index: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train_rows: int
    n_test_rows: int
    n_test_positive_rows: int
    n_test_spiking_days: int

    @property
    def is_empty(self) -> bool:
        return self.n_test_positive_rows == 0


def make_folds(
    frame: pd.DataFrame,
    *,
    time_column: str = "prediction_at",
    day_column: str = "nem_trading_date",
    label_column: str = "is_spike",
    min_spiking_days: int = DEFAULT_MIN_SPIKING_DAYS,
    min_train_days: int = DEFAULT_MIN_TRAIN_DAYS,
    min_train_spiking_days: int = DEFAULT_MIN_TRAIN_SPIKING_DAYS,
    purge: timedelta = DEFAULT_PURGE,
) -> list[Fold]:
    """Cut the frame into expanding-window folds of roughly equal event weight.

    The first training window has to satisfy both a calendar floor and an event
    quota, and ends as soon as it satisfies the later of the two. Everything
    after it is evaluated.
    """
    if frame.empty:
        return []

    frame = frame.sort_values(time_column)
    days = np.sort(frame[day_column].unique())
    spiking_days = set(frame.loc[frame[label_column].astype(bool), day_column].unique())

    if len(days) <= min_train_days:
        raise ValueError(
            f"{len(days)} days available, {min_train_days} reserved for the first "
            "training window: no fold could be cut"
        )

    burn_in = min_train_days
    seen = sum(1 for day in days[:burn_in] if day in spiking_days)
    while burn_in < len(days) and seen < min_train_spiking_days:
        if days[burn_in] in spiking_days:
            seen += 1
        burn_in += 1

    if burn_in >= len(days):
        raise ValueError(
            f"only {seen} spiking days in the whole period, "
            f"{min_train_spiking_days} needed before the first fold: "
            "no fold could be cut"
        )

    times = frame[time_column]
    day_of_row = frame[day_column]

    folds: list[Fold] = []
    cursor = burn_in
    while cursor < len(days):
        collected = 0
        stop = cursor
        while stop < len(days):
            if days[stop] in spiking_days:
                collected += 1
            stop += 1
            if collected >= min_spiking_days:
                break

        remaining_after = set(days[stop:]) & spiking_days
        if collected < min_spiking_days or len(remaining_after) < min_spiking_days:
            # What is left cannot stand as a fold of its own, so it joins this
            # one. Note the guard is not conditional on a fold already existing:
            # when the whole evaluation period holds barely more than one fold's
            # worth of events, the answer is one wide fold, not a full fold plus
            # a thin one whose metrics would be noise.
            stop = len(days)

        window_days = set(days[cursor:stop])
        in_test = day_of_row.isin(window_days)
        test_rows = frame[in_test]
        if test_rows.empty:
            break

        test_start = test_rows[time_column].min()
        train_end = test_start - purge
        n_train = int((times < train_end).sum())

        if folds or n_train > 0:
            folds.append(
                Fold(
                    index=len(folds),
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_rows[time_column].max(),
                    n_train_rows=n_train,
                    n_test_rows=len(test_rows),
                    n_test_positive_rows=int(test_rows[label_column].astype(bool).sum()),
                    n_test_spiking_days=len(window_days & spiking_days),
                )
            )

        cursor = stop

    return folds


def split(
    frame: pd.DataFrame,
    fold: Fold,
    *,
    time_column: str = "prediction_at",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The training and test rows for one fold.

    Training ends at `train_end`, which already has the purge subtracted, so
    the gap is enforced here rather than trusted to the caller.
    """
    times = frame[time_column]
    train = frame[times < fold.train_end]
    test = frame[(times >= fold.test_start) & (times <= fold.test_end)]
    return train, test


def describe(folds: list[Fold]) -> pd.DataFrame:
    """A table of the fold structure, for the writeup."""
    return pd.DataFrame(
        [
            {
                "fold": f.index,
                "train_rows": f.n_train_rows,
                "test_from": f.test_start,
                "test_to": f.test_end,
                "test_rows": f.n_test_rows,
                "positive_rows": f.n_test_positive_rows,
                "spiking_days": f.n_test_spiking_days,
            }
            for f in folds
        ]
    )
