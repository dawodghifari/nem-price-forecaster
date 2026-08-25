"""Which MMS tables get loaded, and the arithmetic behind the choice.

Measured from live files and the NEMWeb archive index on 18 Aug 2026. The
archive holds 6 Aug 2025 to 15 Aug 2026 — 375 daily rollups, so about 12.3
months rather than the 13 the plan assumed.

Compressed download, whole archive:

    DISPATCHIS   ~5.5 MB/day  x 375  =  ~2.1 GB
    P5MIN         ~55 MB/day  x 375  = ~20.6 GB

The plan costed 2 GB, which was DISPATCHIS only. P5MIN is ten times larger and
was never priced.

Rows per file, and what a full backfill would produce at 288 files a day for
375 days:

    | table                     | rows/file |    total rows |
    |---------------------------|-----------|---------------|
    | DISPATCH.CASE_SOLUTION    |         1 |       108,000 |
    | DISPATCH.PRICE            |         5 |       540,000 |
    | DISPATCH.REGIONSUM        |         5 |       540,000 |
    | DISPATCH.INTERCONNECTORRES|         6 |       648,000 |
    | DISPATCH.LOCAL_PRICE      |        ~9 |       972,000 |
    | DISPATCH.CONSTRAINT       |       959 |   103,600,000 |
    | P5MIN.CASESOLUTION        |         1 |       108,000 |
    | P5MIN.REGIONSOLUTION      |        60 |     6,480,000 |
    | P5MIN.INTERCONNECTORSOLN  |        72 |     7,776,000 |
    | P5MIN.LOCAL_PRICE         |       118 |    12,744,000 |
    | P5MIN.CONSTRAINTSOLUTION  |    11,508 | 1,242,864,000 |

The two constraint tables are 97% of DISPATCHIS rows and 98% of P5MIN rows, and
together they are 1.35 billion rows. That is not a Postgres-on-a-laptop
proposition, and none of it feeds a Phase 1 feature: interconnector flows and
their limits come from `DISPATCH.INTERCONNECTORRES`, which carries the flow and
both limits directly, so binding can be derived without the constraint set.

Loading the allowlist below produces about 8.3 million rows.

If Phase 2 wants constraint data — marginal value on a binding constraint is a
strong spike signal — the move is a row filter at load time rather than the
whole table, since the large majority of the 959 rows per interval are
non-binding with a marginal value of zero.
"""

from __future__ import annotations

# Enough for the Phase 1 feature table and both benchmarks.
PHASE1_TABLES: frozenset[str] = frozenset(
    {
        # DISPATCHIS
        "PRICE",  # RRP, the label, plus intervention and price-cap flags
        "REGIONSUM",  # demand, available generation, semi-scheduled output
        "INTERCONNECTORRES",  # flows and limits, for binding
        "CASE_SOLUTION",  # run metadata, intervention and solution status
        # P5MIN
        "REGIONSOLUTION",  # the benchmark forecast, and its own demand outlook
        "CASESOLUTION",
    }
)

# Adds AEMO's interconnector forecast. Roughly 7.8M more rows.
WITH_INTERCONNECTOR_FORECAST: frozenset[str] = PHASE1_TABLES | {"INTERCONNECTORSOLN"}

# Everything. Do not use against the full archive without reading the note above.
ALL_TABLES = None

REPORTS = {
    "DISPATCHIS": "DispatchIS_Reports",
    "P5MIN": "P5_Reports",
}
