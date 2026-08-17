# Conventions

Rules that are cheap to follow and expensive to retrofit. Each one exists
because getting it wrong produces a silent error rather than a crash.

## Time

**Every timestamp in the database is UTC.** Postgres columns are `timestamptz`,
the container runs with `TZ=UTC`, and conversion happens at ingest.

**NEM local time is AEST, a fixed +10:00 offset all year.** The National
Electricity Market does not observe daylight saving, so a fixed offset is
correct rather than an approximation. Do not use a named zone such as
`Australia/Sydney`, which would shift half the year by an hour and shift the
weather join with it.

**NEMWeb directory index times are assumed to be NEM local.** This is an
assumption until a listing time is checked against a known publication.
`scripts/verify_formats.py` records it as an assumption rather than a
measurement, and it is safe in the conservative direction: if the index is
actually UTC, the recorded availability time is later than the truth, which
delays a feature rather than leaking one.

**Interval labelling is settled by reconciliation, not by documentation.** The
NEM convention is that `SETTLEMENTDATE` labels the *end* of its interval, but
NEMWeb lists some DISPATCHIS files before the timestamp they carry. Rather than
pick a reading, `verify_formats.py` averages six parsed five-minute prices and
compares them against AEMO's separately published aggregated series; the
alignment that reconciles is the one recorded, and every lag is defined against
it.

Read the wrong way, a five-minute label shifts every price lag by one interval,
in the direction that makes the model look better than it is.

## Vintage — when a row became knowable

Three timestamps travel with every raw row, and they are not interchangeable.

| Column | Meaning |
|---|---|
| `_generated_at` | What AEMO stamped inside the file or its filename |
| `_listed_at` | When the file appeared in the NEMWeb directory index |
| `_known_at` | `max` of the two. **The only one a feature may join on.** |
| `_ingested_at` | When this pipeline wrote the row. Operational, never a feature. |

The gap between the first two is real and variable. A P5MIN run generated at
23:35:33 did not appear on the site until 23:48. Joining on `_generated_at`
would hand the model AEMO's forecast twelve minutes before anyone could have
read it, on the single feature the project's headline claim depends on.

## Raw is append only

AEMO revises prices after the fact. The archive holds final values; the live
poller sees first-published ones. Overwriting would mean training on numbers
that were not visible at prediction time, and would make the Phase 3 drift
monitor report a market regime change whose real cause was this pipeline.

So nothing is updated in place. A revision arrives as a new file, gets a new
`_source_file`, and lands as new rows. The primary key is
`(_source_file, _row_ordinal)`, which makes re-running a file a no-op.

## Raw tables are generated from the file

Columns come from the `I` record, and the table name carries the report
version: `DISPATCH.PRICE` version 5 becomes `raw.dispatch_price_v5`. When AEMO
changes a column set it bumps that version, so the pipeline gets a new table
instead of a shifted column, and the dbt staging model is the single place that
has to notice.

Every raw column is `text`. Typing happens in staging, so a value AEMO writes
in an unexpected shape fails a visible test rather than being dropped at
ingest.

## Intervention runs

When AEMO intervenes in the market — directing a unit to run, deploying
reserves — it solves dispatch twice: once for the physical outcome and once for
pricing. Both solutions can appear for the same interval, distinguished by an
`INTERVENTION` flag, and `DISPATCH.PRICE` also carries `RRP` alongside `ROP`,
the price the solve originally produced before adjustment.

The parser keeps every row. Staging picks one run and one price column, and the
choice is recorded there rather than assumed here.

## Data is not committed

`data/` is gitignored. Roughly 2 GB of archive is reproduced by `make backfill`
rather than stored in git.

## Attribution

Market data © AEMO, used under AEMO's copyright permissions notice. Weather
from Open-Meteo, CC BY 4.0.
