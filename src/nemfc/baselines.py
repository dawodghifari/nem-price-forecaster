"""The three baselines from PLAN §2.1, with two deviations from how it scores them.

**Both of the simple baselines are scored continuously, not as booleans.**
PLAN §2.1 defines AEMO's baseline as "forecast RRP exceeds $300" and persistence
as "spiked in the previous 30 minutes". As booleans they produce two distinct
scores, which makes precision@k undefined -- there is no way to rank alerts --
and reduces the PR curve to a single point. Scoring the eventual model's
continuous probability against that would be a win handed over by the metric's
shape rather than earned by forecasting. So AEMO is scored by its forecast
*magnitude* and persistence by the trailing maximum price, with the $300
decision reported separately as the operating point AEMO's own users see.

**Climatology takes its level from the training window, not the long history.**
PLAN §2.2 has it estimated on the ten-year 30-minute series, on the reasoning
that a baseline built on more data is a harder one. That reasoning is right and
the method inverts it. The target is the *maximum* over six five-minute
intervals; the aggregated series carries a half-hourly *mean*. One interval at
$500 among five at $50 averages to $125 -- a spike by our definition and not by
the long series'. A rate fitted that way is biased low, which makes the floor
easier to clear rather than harder.

So the long history sets the *shape* -- how spike risk varies across the hours
of the day and between weekdays and weekends, a pattern that needs a decade to
estimate stably -- and the training window sets the *level*, using the same
target definition everything else is scored on. The long series keeps the job it
is genuinely better at and loses the one it is wrong for.

Climatology is also fitted out-of-fold, which the plan does not ask for. A rate
computed over all thirteen months and then scored on those same months has read
its own test set. The leak is small next to the ones Day 4 chased, and this
project's claim is that it has none.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nemfc.folds import Fold, split

#: Weight on the prior, in pseudo-observations. Around 540 rows land in each
#: (hour, day type) cell per 90 days of training, of which roughly five are
#: positive, so an unsmoothed cell rate is mostly noise. 200 pulls a cell most
#: of the way to the decade-long shape unless it has real evidence of its own.
DEFAULT_ALPHA = 200.0

WEEKEND_DAYS = {0, 6}  # Postgres dow: Sunday 0, Saturday 6


def day_type(day_of_week: pd.Series) -> pd.Series:
    """Weekday against weekend.

    Coarser than PLAN §2.1's "day of week". Seven levels crossed with 24 hours
    is 168 cells over 50 spiking days, which estimates noise. Two levels is the
    distinction the demand profile actually turns on.
    """
    return day_of_week.astype(int).isin(WEEKEND_DAYS).map({True: "weekend", False: "weekday"})


def _cells(frame: pd.DataFrame) -> pd.Series:
    return list(
        zip(
            frame["nem_hour"].astype(int),
            day_type(frame["nem_day_of_week"]),
            strict=True,
        )
    )


def long_history_shape(history: pd.DataFrame, *, nem_offset_hours: int = 10) -> pd.Series:
    """Relative spike risk by hour and day type, from the decade-long series.

    Normalised to a mean of one, so it carries only the shape. The level it
    would imply is discarded, for the mean-against-maximum reason above.

    `spacing_minutes` weights each row: half-hourly rows before October 2021
    cover six times the time a five-minute row does, and pooling them unweighted
    would let the recent years dominate a pattern the older years are there to
    stabilise.
    """
    frame = history.copy()
    local = pd.to_datetime(frame["settlement_end"], utc=True) + pd.Timedelta(hours=nem_offset_hours)
    frame["nem_hour"] = local.dt.hour
    frame["nem_day_of_week"] = (local.dt.dayofweek + 1) % 7  # pandas Mon=0 -> Postgres Sun=0
    frame["cell"] = _cells(frame)
    frame["weight"] = frame["spacing_minutes"].astype(float)

    grouped = frame.groupby("cell")
    weighted_rate = grouped.apply(
        lambda g: float((g["is_spike"].astype(float) * g["weight"]).sum() / g["weight"].sum()),
        include_groups=False,
    )

    mean = weighted_rate.mean()
    if not mean or np.isnan(mean):
        return pd.Series(1.0, index=weighted_rate.index)
    return weighted_rate / mean


def fit_climatology(
    train: pd.DataFrame,
    shape: pd.Series | None = None,
    *,
    alpha: float = DEFAULT_ALPHA,
    label_column: str = "is_spike",
) -> pd.Series:
    """Smoothed spike rate per (hour, day type), fitted on training rows only.

    Each cell is pulled toward `global_rate * shape[cell]` in proportion to how
    little evidence it has of its own. With no shape supplied it falls back to
    the flat global rate, so the baseline still works when the long history is
    unavailable -- it is simply weaker, which is the honest consequence.
    """
    if train.empty:
        return pd.Series(dtype=float)

    frame = train.copy()
    frame["cell"] = _cells(frame)
    labels = frame[label_column].astype(float)

    global_rate = float(labels.mean())
    counts = frame.groupby("cell").size()
    positives = labels.groupby(frame["cell"]).sum()

    if shape is None:
        prior = pd.Series(global_rate, index=counts.index)
    else:
        prior = (shape.reindex(counts.index).fillna(1.0) * global_rate).clip(0.0, 1.0)

    return (positives + alpha * prior) / (counts + alpha)


def apply_climatology(rates: pd.Series, frame: pd.DataFrame, *, fallback: float) -> np.ndarray:
    """Score rows from fitted cell rates.

    A cell absent from training gets `fallback`, the training-window base rate.
    Silently returning zero would rank those rows last on a technicality rather
    than on evidence.
    """
    cells = pd.Series(_cells(frame), index=frame.index)
    return cells.map(rates).astype(float).fillna(fallback).to_numpy()


def climatology_out_of_fold(
    frame: pd.DataFrame,
    folds: list[Fold],
    shape: pd.Series | None = None,
    *,
    alpha: float = DEFAULT_ALPHA,
    label_column: str = "is_spike",
) -> pd.Series:
    """Climatology scored on each fold from that fold's training rows alone.

    Rows before the first fold get no score. They are training data for every
    fold and belong in no pooled metric, so they are left as NaN and dropped by
    the caller rather than filled with something plausible.
    """
    scores = pd.Series(np.nan, index=frame.index, dtype=float)

    for fold in folds:
        train, test = split(frame, fold)
        if train.empty or test.empty:
            continue
        rates = fit_climatology(train, shape, alpha=alpha, label_column=label_column)
        fallback = float(train[label_column].astype(float).mean())
        scores.loc[test.index] = apply_climatology(rates, test, fallback=fallback)

    return scores


def persistence_score(frame: pd.DataFrame) -> pd.Series:
    """The trailing 30-minute maximum price.

    PLAN §2.1's persistence rule -- "spiked in the previous 30 minutes" -- is
    this score thresholded at $300. Keeping the magnitude makes it rankable and
    strictly more informative, so the comparison is against the stronger form.

    Null where the rolling window was short. Those rows rank last, which is the
    correct treatment: a baseline with no history to lean on has no opinion.
    """
    return frame["rolling_max_rrp_30m"].astype(float)


def aemo_score(frame: pd.DataFrame) -> pd.Series:
    """AEMO's own pre-dispatch: the highest price it forecasts in the window."""
    return frame["benchmark_max_forecast_rrp"].astype(float)


def operating_point(
    frame: pd.DataFrame,
    score: pd.Series,
    threshold: float,
    *,
    label_column: str = "is_spike",
) -> dict[str, float]:
    """Precision and recall at a fixed cut, for the boolean form of a baseline.

    Reported beside the continuous curve because it is the decision an operator
    reading AEMO's forecast actually faces: alert when the number crosses $300.
    """
    alerted = score.to_numpy(dtype=float) > threshold
    truth = frame[label_column].to_numpy(dtype=bool)

    n_alerts = int(alerted.sum())
    hits = int((alerted & truth).sum())
    positives = int(truth.sum())

    return {
        "threshold": threshold,
        "n_alerts": n_alerts,
        "precision": hits / n_alerts if n_alerts else float("nan"),
        "recall": hits / positives if positives else float("nan"),
        "n_positive_rows": positives,
    }
