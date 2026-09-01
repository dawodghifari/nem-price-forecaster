"""Tests for the scoring module, against answers known by inspection.

Every metric here is hand-written rather than imported, so each one is pinned
to a case whose value can be worked out on paper. Two of them exist to catch
the failure this project keeps finding: a number that looks reasonable, is
wrong, and has nothing to disagree with it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from nemfc.evaluate import (
    NO_EPISODE,
    assign_episodes,
    average_precision,
    blocked_bootstrap_pr_auc,
    episode_recall_at_k,
    expected_precision_at_k,
    find_episodes,
    precision_at_k,
    rank_order,
    score,
)

BASE = datetime(2026, 3, 1, 8, 0, tzinfo=timezone.utc)
FIVE = timedelta(minutes=5)


def at(minutes: int) -> pd.Timestamp:
    return pd.Timestamp(BASE + timedelta(minutes=minutes))


# --------------------------------------------------------------------------
# Episodes
# --------------------------------------------------------------------------


def test_contiguous_intervals_form_one_episode():
    episodes = find_episodes(pd.Series([at(0), at(5), at(10)]))
    assert len(episodes) == 1
    assert episodes[0].n_intervals == 3
    assert episodes[0].start == at(0)
    assert episodes[0].end == at(10)


def test_a_single_calm_interval_splits_an_episode():
    """The strict definition, and the one behind the count of 144."""
    episodes = find_episodes(pd.Series([at(0), at(5), at(15), at(20)]))
    assert [e.n_intervals for e in episodes] == [2, 2]


def test_widening_the_gap_merges_across_a_lull():
    """The count is sensitive to `max_gap`, so the value must be reported."""
    stamps = pd.Series([at(0), at(5), at(15), at(20)])
    assert len(find_episodes(stamps, max_gap=timedelta(minutes=10))) == 1
    assert len(find_episodes(stamps, max_gap=timedelta(minutes=5))) == 2


def test_no_spikes_gives_no_episodes():
    assert find_episodes(pd.Series([], dtype="datetime64[ns, UTC]")) == []


def test_a_row_is_attributed_to_the_first_episode_its_window_reaches():
    """One row, one episode, so episode recall counts each event once."""
    spiking = pd.Series([at(0), at(5), at(20), at(25)])
    episodes = find_episodes(spiking)

    anchors = pd.Series([at(-5), at(12), at(30)])
    assigned = assign_episodes(anchors, spiking, episodes)

    # Window (-5, 25] reaches both episodes; the earlier one owns the row.
    assert assigned[0] == 0
    # Window (12, 42] reaches only the second.
    assert assigned[1] == 1
    # Window (30, 60] reaches neither.
    assert assigned[2] == NO_EPISODE


def test_an_anchor_that_is_itself_spiking_does_not_claim_its_own_interval():
    """The window is half-open: strictly after the anchor.

    An anchor's own price is a feature, not a target. Including it would make
    every spiking interval predict itself.
    """
    spiking = pd.Series([at(0)])
    episodes = find_episodes(spiking)
    assert assign_episodes(pd.Series([at(0)]), spiking, episodes)[0] == NO_EPISODE


def test_a_spike_beyond_the_horizon_is_not_claimed():
    spiking = pd.Series([at(35)])
    episodes = find_episodes(spiking)
    assigned = assign_episodes(pd.Series([at(0), at(5)]), spiking, episodes)
    assert assigned[0] == NO_EPISODE  # 35 minutes away, horizon is 30
    assert assigned[1] == 0  # 30 minutes away, inclusive


# --------------------------------------------------------------------------
# Average precision
# --------------------------------------------------------------------------


def test_average_precision_on_a_hand_computed_case():
    """y = 1,0,1,0 ranked in that order.

    rank 1   P = 1/1   R = 1/2   gain 1/2   -> 0.5
    rank 2   P = 1/2   R = 1/2   gain 0     -> 0
    rank 3   P = 2/3   R = 1     gain 1/2   -> 1/3
    rank 4   P = 2/4   R = 1     gain 0     -> 0
    """
    y = np.array([True, False, True, False])
    s = np.array([4.0, 3.0, 2.0, 1.0])
    assert average_precision(y, s) == pytest.approx(0.5 + 1 / 3)


def test_a_perfect_ranking_scores_one():
    y = np.array([True, True, False, False, False])
    s = np.array([9.0, 8.0, 3.0, 2.0, 1.0])
    assert average_precision(y, s) == pytest.approx(1.0)


def test_a_constant_score_scores_the_base_rate():
    """The floor any baseline must clear, and the check on the tie handling.

    With one tie group there is a single threshold: precision is the base rate
    and recall is 1. A trapezoid-interpolated implementation would return
    something higher, which is why this module does the step-wise sum.
    """
    y = np.array([True] + [False] * 9)
    s = np.ones(10)
    assert average_precision(y, s) == pytest.approx(0.1)


def test_the_worst_possible_ranking_scores_far_below_the_base_rate():
    y = np.array([False, False, False, True])
    s = np.array([4.0, 3.0, 2.0, 1.0])
    assert average_precision(y, s) == pytest.approx(0.25)


def test_unscored_rows_rank_last_rather_than_vanishing():
    """A baseline that abstains on a row still has to live with it.

    Dropping unscored rows would let a baseline improve its own score by
    declining to answer on the hard ones.
    """
    y = np.array([True, False, False])
    kept = average_precision(y, np.array([1.0, np.nan, np.nan]))
    assert kept == pytest.approx(1.0)

    missed = average_precision(np.array([False, False, True]), np.array([1.0, 2.0, np.nan]))
    assert missed == pytest.approx(1 / 3)


def test_two_unscored_rows_are_one_tie_group_not_two_thresholds():
    """The bug a `-inf` sentinel introduces, pinned so it cannot come back.

    One scored negative, then two unscored rows of which one is positive. The
    unscored pair is a single threshold: recall reaches 1 at a precision of
    1/3, and the score is 1/3.

    Splitting them into two thresholds would have credited the positive one for
    sitting earlier in the array, giving it a precision of 1/2 at full recall
    and a score of 0.5 -- a fifty percent overstatement, on rows the baseline
    explicitly had no opinion about.
    """
    y = np.array([False, True, False])
    s = np.array([5.0, np.nan, np.nan])
    assert average_precision(y, s) == pytest.approx(1 / 3)


def test_average_precision_is_undefined_without_positives():
    assert np.isnan(average_precision(np.array([False, False]), np.array([1.0, 2.0])))


# --------------------------------------------------------------------------
# precision@k and ties
# --------------------------------------------------------------------------


def test_precision_at_k_counts_the_top_k_only():
    y = np.array([True, True, False, False])
    s = np.array([4.0, 3.0, 2.0, 1.0])
    t = np.arange(4)
    assert precision_at_k(y, s, t, 2) == pytest.approx(1.0)
    assert precision_at_k(y, s, t, 4) == pytest.approx(0.5)


def test_ties_are_broken_chronologically_not_by_array_order():
    """An operator works the queue in the order it arrives."""
    y = np.array([False, True])
    s = np.array([1.0, 1.0])
    later_first = np.array([2, 1])  # row 1 arrived earlier despite its position
    assert rank_order(s, later_first).tolist() == [1, 0]
    assert precision_at_k(y, s, later_first, 1) == pytest.approx(1.0)


def test_the_tie_break_is_reported_against_random_ordering():
    """When these disagree, the score is too coarse to rank at that k.

    One row scores 5 and is positive; four tie at 1, of which one is positive.
    Taking the top 2 chronologically lands on the positive tied row and reports
    1.0. Under random tie-breaking the second pick is positive a quarter of the
    time, so the honest expectation is (1 + 0.25) / 2.
    """
    y = np.array([True, True, False, False, False])
    s = np.array([5.0, 1.0, 1.0, 1.0, 1.0])
    t = np.arange(5)

    assert precision_at_k(y, s, t, 2) == pytest.approx(1.0)
    assert expected_precision_at_k(y, s, 2) == pytest.approx(0.625)


def test_expected_precision_treats_unscored_rows_as_one_tie_group():
    """The same `-inf` bug as above, in the other function that ranks.

    It was fixed in `average_precision` and missed here, and the miss announced
    itself only as a numpy RuntimeWarning during the first real run. Two
    unscored rows must form one group; splitting them would count the positive
    one in full instead of at its quarter share.
    """
    y = np.array([True, True, False, False, False])
    s = np.array([5.0, np.nan, np.nan, np.nan, np.nan])

    with np.errstate(invalid="raise"):
        assert expected_precision_at_k(y, s, 2) == pytest.approx(0.625)


def test_expected_precision_matches_the_deterministic_value_without_ties():
    y = np.array([True, False, True])
    s = np.array([3.0, 2.0, 1.0])
    t = np.arange(3)
    assert expected_precision_at_k(y, s, 2) == pytest.approx(precision_at_k(y, s, t, 2))


def test_asking_for_more_alerts_than_rows_is_an_error_not_a_shrug():
    y = np.array([True, False])
    with pytest.raises(ValueError):
        expected_precision_at_k(y, np.array([1.0, 2.0]), 5)


# --------------------------------------------------------------------------
# Episode recall
# --------------------------------------------------------------------------


def test_episode_recall_sees_what_precision_cannot():
    """Six alerts, all correct, all pointed at the same evening.

    Precision says the alerts were perfect. Episode recall says one event out
    of three was caught. Both are true, and only the second describes what the
    operator got.
    """
    episodes = np.array([0, 0, 0, 0, 0, 0, 1, 2])
    s = np.array([9, 9, 9, 9, 9, 9, 1, 1], dtype=float)
    y = np.array([True] * 6 + [True, True])
    t = np.arange(8)

    assert precision_at_k(y, s, t, 6) == pytest.approx(1.0)
    assert episode_recall_at_k(episodes, s, t, 6, n_episodes=3) == pytest.approx(1 / 3)


def test_unassigned_rows_never_count_as_a_caught_episode():
    episodes = np.array([NO_EPISODE, NO_EPISODE, 0])
    s = np.array([9.0, 8.0, 1.0])
    assert episode_recall_at_k(episodes, s, np.arange(3), 2, n_episodes=1) == 0.0


# --------------------------------------------------------------------------
# Scorecard
# --------------------------------------------------------------------------


def _frame(n: int = 40) -> pd.DataFrame:
    stamps = [at(5 * i) for i in range(n)]
    rrp = np.full(n, 50.0)
    rrp[[10, 11, 30]] = 900.0  # two episodes: one of two intervals, one of one
    frame = pd.DataFrame(
        {
            "prediction_at": stamps,
            "anchor_settlement_end": stamps,
            "rrp_now": rrp,
            "nem_trading_date": [s.date() for s in stamps],
        }
    )
    spiking = frame.loc[frame["rrp_now"] > 300, "anchor_settlement_end"]
    episodes = find_episodes(spiking)
    frame["episode_id"] = assign_episodes(frame["anchor_settlement_end"], spiking, episodes)
    frame["is_spike"] = frame["episode_id"] != NO_EPISODE
    # A score that is right about the first episode and silent about the second.
    frame["oracle_ish"] = np.where(frame["episode_id"] == 0, 10.0, 0.0)
    return frame


def test_a_scorecard_carries_every_denominator():
    """Three spiking intervals, two episodes, thirteen positive rows.

    The 6.5-to-1 ratio between rows and intervals is the whole reason this
    module exists: reporting 13 as the evidence for 2 events would overstate
    it by the width of the target window.
    """
    card = score(_frame(), "oracle_ish", "partial oracle", ks=(5,))
    assert card.n_rows == 40
    assert card.n_positive_rows == 13
    assert card.n_episodes == 2
    assert card.n_days_with_episodes == 1
    assert card.base_rate == pytest.approx(13 / 40)


def test_a_score_blind_to_one_episode_cannot_recall_it():
    card = score(_frame(), "oracle_ish", "partial oracle", ks=(5,))
    assert card.precision_at_k[5] == pytest.approx(1.0)
    assert card.episode_recall_at_k[5] == pytest.approx(0.5)


def test_lift_reports_the_multiple_over_guessing():
    card = score(_frame(), "oracle_ish", "partial oracle", ks=(5,))
    assert card.lift_at_k(5) == pytest.approx(1.0 / card.base_rate)


def test_a_constant_score_gets_no_lift():
    frame = _frame()
    frame["flat"] = 1.0
    card = score(frame, "flat", "constant", ks=(5,))
    assert card.pr_auc == pytest.approx(card.base_rate)


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------


def _multi_day_frame(days: int = 12) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(7)
    for day in range(days):
        for i in range(30):
            spike = day % 3 == 0 and i in (20, 21)
            rows.append(
                {
                    "nem_trading_date": day,
                    "is_spike": spike,
                    "signal": rng.normal(3.0 if spike else 0.0),
                    "episode_id": day if spike else NO_EPISODE,
                    "prediction_at": at(day * 1000 + i),
                }
            )
    return pd.DataFrame(rows)


def test_the_bootstrap_interval_brackets_the_point_estimate():
    frame = _multi_day_frame()
    point = average_precision(frame["is_spike"].to_numpy(), frame["signal"].to_numpy())
    low, high = blocked_bootstrap_pr_auc(frame, "signal", n_resamples=200, seed=1)
    assert low <= point <= high


def test_the_bootstrap_is_reproducible():
    frame = _multi_day_frame()
    first = blocked_bootstrap_pr_auc(frame, "signal", n_resamples=100, seed=3)
    second = blocked_bootstrap_pr_auc(frame, "signal", n_resamples=100, seed=3)
    assert first == second


def _by_row(frame: pd.DataFrame, **kwargs) -> tuple[float, float]:
    """The wrong bootstrap: every row treated as an independent draw."""
    frame = frame.copy()
    frame["row_id"] = np.arange(len(frame))
    return blocked_bootstrap_pr_auc(frame, "signal", day_column="row_id", **kwargs)


def test_the_choice_of_block_changes_the_answer():
    """Whichever way it goes, the block is not a cosmetic detail."""
    frame = _multi_day_frame()
    by_day = blocked_bootstrap_pr_auc(frame, "signal", n_resamples=300, seed=5)
    by_row = _by_row(frame, n_resamples=300, seed=5)

    width_day = by_day[1] - by_day[0]
    width_row = by_row[1] - by_row[0]
    assert abs(width_day - width_row) > 0.1


def _good_and_bad_days(days: int = 20, seed: int = 11) -> pd.DataFrame:
    """Days that differ in how predictable they are.

    On half of them the signal separates spikes cleanly; on the other half it
    is noise. Which days a resample happens to draw therefore dominates the
    score, and that day-to-day variation is the uncertainty a confidence
    interval is supposed to express.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(days):
        predictable = day % 2 == 0
        for i in range(60):
            spike = i >= 40
            # A clean separation on predictable days, pure noise on the rest.
            signal = rng.normal(6.0 if spike else 0.0, 0.5) if predictable else rng.normal(0.0, 1.0)
            rows.append(
                {
                    "nem_trading_date": day,
                    "is_spike": spike,
                    "signal": signal,
                    "episode_id": day if spike else NO_EPISODE,
                    "prediction_at": at(day * 1000 + i),
                }
            )
    return pd.DataFrame(rows)


def test_row_resampling_understates_uncertainty_when_rows_repeat_an_event():
    """The failure mode the blocked bootstrap exists to prevent.

    With enough positives that resampling noise in the positive count is not
    the dominant term -- 400 here, against 989 in the real feature table --
    pooling rows averages the good days and the bad ones together in every
    resample, and the interval collapses around that average. Resampling days
    keeps a resample's luck intact and reports an interval several times wider.

    The direction is not universal. In an earlier version of this test, with
    only eight positives in the whole frame, the row bootstrap came out
    *wider*, because destroying the block also destroyed the positive count and
    that noise swamped everything else. The general claim is only that the row
    bootstrap assumes an effective sample size it does not have; which way the
    error runs depends on the regime. This project's regime is this one.
    """
    frame = _good_and_bad_days()
    assert int(frame["is_spike"].sum()) == 400  # 20 days x 20 spiking rows

    by_day = blocked_bootstrap_pr_auc(frame, "signal", n_resamples=300, seed=5)
    by_row = _by_row(frame, n_resamples=300, seed=5)

    assert (by_day[1] - by_day[0]) > 3 * (by_row[1] - by_row[0])
