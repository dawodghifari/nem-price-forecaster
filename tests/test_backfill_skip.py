"""Skipping rollups already loaded, and bind-parameter safety.

Both exist to protect a multi-hour backfill: one from wasting time re-fetching
what it has, the other from dying part-way through on a limit that a fixed
chunk size would eventually cross.
"""

from __future__ import annotations

from collections import Counter
from datetime import date

from nemfc.db import MAX_BIND_PARAMS, safe_chunksize
from nemfc.ingest import OWN_INTERVALS, _day_complete, member_day


def test_member_day_reads_the_interval_label():
    assert member_day("PUBLIC_P5MIN_202607250005_20260725000033.CSV") == "20260725"
    assert member_day("PUBLIC_DISPATCHIS_202608160130_0000000532838629.CSV") == "20260816"


def test_member_day_returns_none_for_anything_unexpected():
    assert member_day("nonsense.CSV") is None
    assert member_day("PUBLIC_X_short_1.CSV") is None


def _counter(**days: int) -> Counter:
    return Counter(dict(days))


def test_a_fully_loaded_day_is_skipped():
    per_day = _counter(**{"20260725": OWN_INTERVALS, "20260726": 1})
    assert _day_complete(per_day, date(2026, 7, 25))


def test_a_day_that_reached_288_by_two_routes_is_skipped():
    """16 Aug arrived as 228 archive members plus 60 from the live poller."""
    per_day = _counter(**{"20260816": 288, "20260817": 287})
    assert _day_complete(per_day, date(2026, 8, 16))


def test_a_partially_loaded_day_is_not_skipped():
    per_day = _counter(**{"20260725": 100, "20260726": 1})
    assert not _day_complete(per_day, date(2026, 7, 25))


def test_a_day_is_not_skipped_when_its_midnight_interval_is_missing():
    """The rollup for day D carries the interval stamped D+1 00:00.

    Skipping on the day's own count alone would drop that one interval, leaving
    a single-interval hole that nothing downstream would notice.
    """
    per_day = _counter(**{"20260725": OWN_INTERVALS})
    assert not _day_complete(per_day, date(2026, 7, 25))


def test_an_untouched_day_is_not_skipped():
    assert not _day_complete(Counter(), date(2026, 7, 25))


def test_chunksize_stays_under_the_postgres_bind_limit():
    for columns in (7, 30, 118, 124, 126, 132, 500):
        assert safe_chunksize(columns) * columns <= MAX_BIND_PARAMS


def test_the_widest_real_tables_would_have_broken_a_fixed_chunk_of_1000():
    """P5MIN.REGIONSOLUTION is 118 columns, DISPATCH.REGIONSUM is 126."""
    for columns in (118 + 6, 126 + 6):
        assert 1000 * columns > 65_535
        assert safe_chunksize(columns) < 1000


def test_chunksize_never_reaches_zero():
    assert safe_chunksize(100_000) == 1
    assert safe_chunksize(0) == MAX_BIND_PARAMS
