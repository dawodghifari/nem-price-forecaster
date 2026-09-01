"""Scoring for a rare, clustered, heavily overlapping positive class.

Three properties of this problem break the usual scoring code, and each one is
handled explicitly here rather than inherited from a library default.

**Positives overlap six to one.** A prediction's target is the next six
intervals, so consecutive predictions five minutes apart share five of them. One
spiking interval therefore produces up to six positive rows. Counting rows
overstates the evidence roughly sixfold: the feature table's 989 positive rows
come from 482 spiking intervals in 144 episodes across 50 days. Every number
this module reports carries its denominator, and `episode_recall_at_k` answers
the question rows cannot -- how many distinct events were caught.

**Scores tie, and the ties are not incidental.** AEMO's pre-dispatch forecast
piles up at the administered cap, and persistence takes the same rolling maximum
across a whole flat stretch. A tie group straddling the k boundary can therefore
decide precision@k on its own, and `numpy.argsort` would settle it by array
order -- an arbitrary rule that is not visibly arbitrary. So ties break
chronologically, on the grounds that an operator works an alert queue in the
order it arrives, and `expected_precision_at_k` reports what random tie-breaking
would have given. When those two disagree, the score is too coarse to rank at
that k, which is a finding about the baseline rather than a detail of the
implementation.

**Neighbouring rows are not independent.** Confidence intervals come from
resampling whole NEM trading days, never rows or episodes. Two episodes on the
same evening share weather, demand and the same underlying shortfall; treating
them as independent draws would produce intervals far too narrow, and narrow
intervals are exactly how a null result gets mistaken for a win.

No scikit-learn. Average precision is forty lines and the modelling stack does
not arrive until Day 6; more to the point, a metric written out and tested
against cases whose answers are known by inspection is easier to defend than an
import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

DISPATCH_INTERVAL = timedelta(minutes=5)
HORIZON = timedelta(minutes=30)

#: Alert budgets, fixed here before any baseline has been measured. Pinning k
#: in code rather than choosing it beside the results is the difference between
#: a metric and a knob.
DEFAULT_K = (50, 100, 200)

NO_EPISODE = -1


# --------------------------------------------------------------------------
# Episodes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Episode:
    """A maximal run of consecutive spiking intervals."""

    episode_id: int
    start: pd.Timestamp
    end: pd.Timestamp
    n_intervals: int


def find_episodes(
    spiking_ends: pd.Series | pd.DatetimeIndex,
    *,
    max_gap: timedelta = DISPATCH_INTERVAL,
) -> list[Episode]:
    """Group spiking intervals into episodes.

    A run breaks when consecutive spiking intervals are further apart than
    `max_gap`. The default is one dispatch interval, so only strictly
    contiguous intervals join -- a single calm interval splits an episode in
    two. Widening `max_gap` merges across short lulls; the count is sensitive
    to it, which is why the value used is reported alongside the number.
    """
    stamps = pd.DatetimeIndex(pd.Series(spiking_ends).dropna().unique()).sort_values()
    if len(stamps) == 0:
        return []

    breaks = np.diff(stamps.to_numpy()) > np.timedelta64(max_gap)
    run_index = np.concatenate([[0], np.cumsum(breaks)])

    episodes = []
    for episode_id in range(int(run_index[-1]) + 1):
        members = stamps[run_index == episode_id]
        episodes.append(
            Episode(
                episode_id=episode_id,
                start=members[0],
                end=members[-1],
                n_intervals=len(members),
            )
        )
    return episodes


def assign_episodes(
    anchor_ends: pd.Series,
    spiking_ends: pd.Series | pd.DatetimeIndex,
    episodes: list[Episode],
    *,
    horizon: timedelta = HORIZON,
) -> np.ndarray:
    """Map each prediction to the episode its target window opens onto.

    A positive row is one whose window `(anchor, anchor + horizon]` contains a
    spiking interval. Where it contains several, the row is attributed to the
    episode of the *first* one, so a row belongs to exactly one episode and
    episode recall counts each event once.

    Returns `NO_EPISODE` for rows whose window contains no spiking interval.
    """
    stamps = pd.DatetimeIndex(pd.Series(spiking_ends).dropna().unique()).sort_values()
    anchors = pd.DatetimeIndex(anchor_ends)

    if len(stamps) == 0:
        return np.full(len(anchors), NO_EPISODE, dtype=int)

    # Which episode owns each spiking interval.
    owner = np.empty(len(stamps), dtype=int)
    for episode in episodes:
        inside = (stamps >= episode.start) & (stamps <= episode.end)
        owner[inside] = episode.episode_id

    # First spiking interval strictly after the anchor.
    nxt = stamps.searchsorted(anchors, side="right")
    out = np.full(len(anchors), NO_EPISODE, dtype=int)

    found = nxt < len(stamps)
    if found.any():
        candidate = stamps[nxt[found]]
        within = candidate <= (anchors[found] + horizon)
        rows = np.flatnonzero(found)[within]
        out[rows] = owner[nxt[found][within]]

    return out


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------


def _ranked_score(score: np.ndarray) -> np.ndarray:
    """Scores with missing values placed below every real one.

    Deliberately a finite sentinel rather than `-inf`. Tie groups are found by
    differencing adjacent sorted scores, and `-inf - -inf` is NaN, which
    compares unequal to zero -- so two unscored rows would be read as two
    distinct thresholds instead of one tie. That inflates average precision on
    exactly the rows a baseline had no opinion about.
    """
    score = np.asarray(score, dtype=float)
    missing = np.isnan(score)
    if not missing.any():
        return score
    if missing.all():
        return np.zeros_like(score)
    return np.where(missing, np.nanmin(score) - 1.0, score)


def rank_order(score: np.ndarray, tiebreak: np.ndarray) -> np.ndarray:
    """Indices sorted by score descending, ties broken by `tiebreak` ascending.

    `tiebreak` is prediction time. Nulls sort last: a baseline with no opinion
    about a row must never be credited for it.
    """
    return np.lexsort((np.asarray(tiebreak), -_ranked_score(score)))


def precision_at_k(y_true: np.ndarray, score: np.ndarray, tiebreak: np.ndarray, k: int) -> float:
    """Share of the top k alerts that were followed by a spike."""
    if k <= 0:
        raise ValueError("k must be positive")
    order = rank_order(score, tiebreak)[:k]
    return float(np.asarray(y_true, dtype=bool)[order].mean())


def expected_precision_at_k(y_true: np.ndarray, score: np.ndarray, k: int) -> float:
    """Precision@k averaged over every way of breaking the ties.

    Rows above the cut count in full. The tie group straddling the boundary
    contributes its positive rate times the number of its members that fit,
    which is the expectation under a uniformly random ordering within the
    group. Compared against `precision_at_k`, it shows whether the tie-break
    rule is carrying the result.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    y_true = np.asarray(y_true, dtype=bool)
    score = np.asarray(score, dtype=float)
    filled = _ranked_score(score)

    order = np.argsort(-filled, kind="stable")
    ranked_score = filled[order]
    ranked_true = y_true[order]

    taken = 0
    positives = 0.0
    cursor = 0
    for size in _group_sizes(ranked_score):
        start, stop = cursor, cursor + size
        cursor = stop
        group_positives = int(ranked_true[start:stop].sum())
        if taken + size <= k:
            positives += group_positives
            taken += size
        else:
            room = k - taken
            positives += group_positives * room / size
            taken = k
            break

    if taken < k:
        raise ValueError(f"only {taken} rows available, k={k}")
    return float(positives / k)


def _group_sizes(sorted_desc: np.ndarray) -> list[int]:
    """Sizes of the runs of equal values in an already-sorted array."""
    if len(sorted_desc) == 0:
        return []
    change = np.flatnonzero(np.diff(sorted_desc) != 0) + 1
    edges = np.concatenate([[0], change, [len(sorted_desc)]])
    return np.diff(edges).tolist()


def episode_recall_at_k(
    episode_id: np.ndarray,
    score: np.ndarray,
    tiebreak: np.ndarray,
    k: int,
    n_episodes: int,
) -> float:
    """Share of distinct episodes with at least one alert in the top k.

    The number precision@k cannot give. Fifty alerts landing on eight evenings
    score 1.0 on precision and 0.06 on this.
    """
    if n_episodes == 0:
        return float("nan")
    order = rank_order(score, tiebreak)[:k]
    caught = set(np.asarray(episode_id)[order].tolist()) - {NO_EPISODE}
    return len(caught) / n_episodes


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    """Area under the precision-recall curve, by the step-wise sum.

    Thresholds are the distinct scores, so tied rows are resolved together
    rather than in an arbitrary order -- the interpolated trapezoid form is
    optimistic on a coarse score and is not used. Rows with no score are
    ranked last rather than dropped, because a baseline that abstains on a row
    still has to live with it.
    """
    y_true = np.asarray(y_true, dtype=bool)
    score = np.asarray(score, dtype=float)
    if len(y_true) == 0:
        return float("nan")

    total_positive = int(y_true.sum())
    if total_positive == 0:
        return float("nan")

    filled = _ranked_score(score)
    order = np.argsort(-filled, kind="stable")
    ranked_score = filled[order]
    ranked_true = y_true[order]

    cum_tp = np.cumsum(ranked_true)
    cum_n = np.arange(1, len(ranked_true) + 1)

    # One point per distinct score: the last index of each tie group.
    last_of_group = np.flatnonzero(np.diff(ranked_score) != 0)
    last_of_group = np.concatenate([last_of_group, [len(ranked_score) - 1]])

    precision = cum_tp[last_of_group] / cum_n[last_of_group]
    recall = cum_tp[last_of_group] / total_positive
    recall_gain = np.diff(np.concatenate([[0.0], recall]))

    return float((precision * recall_gain).sum())


# --------------------------------------------------------------------------
# Scorecards
# --------------------------------------------------------------------------


@dataclass
class ScoreCard:
    """One baseline's result, carrying every denominator behind it."""

    name: str
    n_rows: int
    n_positive_rows: int
    n_episodes: int
    n_days_with_episodes: int
    base_rate: float
    pr_auc: float
    precision_at_k: dict[int, float] = field(default_factory=dict)
    expected_precision_at_k: dict[int, float] = field(default_factory=dict)
    episode_recall_at_k: dict[int, float] = field(default_factory=dict)
    pr_auc_ci: tuple[float, float] | None = None
    n_scored: int = 0

    def lift_at_k(self, k: int) -> float:
        """Precision@k over the base rate. 1.0 is no better than guessing."""
        return self.precision_at_k[k] / self.base_rate if self.base_rate else float("nan")

    def headline(self, k: int = 100) -> str:
        return (
            f"{self.name}: precision@{k} {self.precision_at_k[k]:.3f} "
            f"({self.lift_at_k(k):.1f}x base rate), "
            f"episode recall@{k} {self.episode_recall_at_k[k]:.3f}, "
            f"PR-AUC {self.pr_auc:.3f} "
            f"over {self.n_episodes} episodes on {self.n_days_with_episodes} days"
        )


def score(
    frame: pd.DataFrame,
    score_column: str,
    name: str,
    *,
    label_column: str = "is_spike",
    time_column: str = "prediction_at",
    episode_column: str = "episode_id",
    day_column: str = "nem_trading_date",
    ks: tuple[int, ...] = DEFAULT_K,
) -> ScoreCard:
    """Score one baseline over a frame that already carries episode labels."""
    y_true = frame[label_column].to_numpy(dtype=bool)
    raw = frame[score_column].to_numpy(dtype=float)
    tiebreak = frame[time_column].to_numpy()
    episodes = frame[episode_column].to_numpy()

    positive_episodes = set(episodes[y_true].tolist()) - {NO_EPISODE}
    days = frame.loc[y_true, day_column].nunique()

    card = ScoreCard(
        name=name,
        n_rows=len(frame),
        n_positive_rows=int(y_true.sum()),
        n_episodes=len(positive_episodes),
        n_days_with_episodes=int(days),
        base_rate=float(y_true.mean()) if len(frame) else float("nan"),
        pr_auc=average_precision(y_true, raw),
        n_scored=int(np.isfinite(raw).sum()),
    )
    for k in ks:
        if k > len(frame):
            continue
        card.precision_at_k[k] = precision_at_k(y_true, raw, tiebreak, k)
        card.expected_precision_at_k[k] = expected_precision_at_k(y_true, raw, k)
        card.episode_recall_at_k[k] = episode_recall_at_k(
            episodes, raw, tiebreak, k, card.n_episodes
        )
    return card


def blocked_bootstrap_pr_auc(
    frame: pd.DataFrame,
    score_column: str,
    *,
    label_column: str = "is_spike",
    day_column: str = "nem_trading_date",
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval for PR-AUC, resampling whole trading days.

    Days, not rows and not episodes. Rows within a day are near-duplicates of
    each other and episodes within an evening share their cause; resampling
    either would treat repeated observations of one event as independent
    evidence and shrink the interval until any difference looked significant.
    """
    rng = np.random.default_rng(seed)

    # Days held as plain arrays rather than frames. `pd.concat` of several
    # hundred small frames dominates the runtime; `np.concatenate` of the two
    # columns actually needed turns minutes into seconds.
    groups = [
        (
            g[label_column].to_numpy(dtype=bool),
            g[score_column].to_numpy(dtype=float),
        )
        for _, g in frame.groupby(day_column, sort=True)
    ]
    if not groups:
        return (float("nan"), float("nan"))

    draws = []
    for _ in range(n_resamples):
        picks = rng.integers(0, len(groups), size=len(groups))
        labels = np.concatenate([groups[i][0] for i in picks])
        scores = np.concatenate([groups[i][1] for i in picks])
        value = average_precision(labels, scores)
        if not np.isnan(value):
            draws.append(value)

    if not draws:
        return (float("nan"), float("nan"))

    tail = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(draws, tail)),
        float(np.quantile(draws, 1.0 - tail)),
    )
