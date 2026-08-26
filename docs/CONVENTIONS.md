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

**`SETTLEMENTDATE` labels the end of its interval.** Confirmed 18 Aug 2026, not
assumed: a DISPATCHIS row stamped `00:50:00` came from a solve whose C header
was written at `00:45:13`. The solve ran at the interval's start and the row
carries its end.

Read the wrong way, a five-minute label shifts every price lag by one interval,
in the direction that makes the model look better than it is.

## A dispatch price is set before its interval, not measured after it

This is the piece of market mechanics that shapes the target variable, and it
looks like a bug until it clicks.

AEMO does not observe five minutes of trading and then compute a price. It runs
a dispatch solve at the *start* of each interval, choosing which generators run
at what output, and the price that solve produces is the price the interval
settles at. So for the interval
ending 00:50 — covering 00:45 to 00:50 — the solve ran at 00:45:13, the file
appeared on NEMWeb at 00:48, and the price applies to five minutes that are
still running. The file is published two minutes before the timestamp it
carries, which is correct rather than anomalous.

**The consequence is a constraint on the label, and it is not covered by PLAN
§4.1, which deals only with features.** At prediction time `t`, the price of
the interval ending around `t+2` has already been published. A target defined
as *any spike in `(t, t+30min]`* therefore contains one interval whose answer
is already sitting in the raw tables. Nothing in a feature-side leakage test
would catch it, because the leak is in the target.

So: the target window covers only intervals whose DISPATCHIS was **not**
published at or before `t` — in practice, intervals ending in
`(t+5min, t+35min]`, filtered on `_known_at > t` rather than on
`SETTLEMENTDATE`. `tests/test_no_leakage.py` on Day 4 asserts this alongside
the feature-side condition.

## Interval resolution changed under us

AEMO's aggregated price and demand series is **five-minute**, not thirty, for
any month after five-minute settlement began in October 2021 — the August 2026
file holds 4,896 rows for seventeen days, which is 288 per day. Earlier months
are half-hourly.

Two things follow. The long history is richer than PLAN §2.2 assumed, since
price and demand are available at five minutes for roughly five years rather
than thirteen months; only the full MMS feeds are limited to the archive
window. And any code reading these files must **detect** the spacing per file
instead of assuming it, because a backfill spanning October 2021 crosses the
change.

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

## A backfilled vintage is weaker than a live one, and says so

A live download knows when a file appeared in the NEMWeb index, to the minute.
The archive publishes one zip per day the following morning, so per-interval
listing times are gone and only the generation timestamp inside each member
survives.

Since P5MIN publication ran 7.5 minutes behind generation at the median and 23
minutes at the tail, treating a backfilled row as knowable at its generation
time would make the training set optimistic by that much — and the model would
then meet a live feed where the same feature arrives later than it did in
training.

So archive-sourced rows get generation time plus the observed 90th-percentile
lag, and carry `_vintage_source = 'archive'` so the distinction is auditable
rather than assumed. Live rows carry `'live'` and their measured listing time.

## Weather: name the model, or get observations

There are two Open-Meteo products, and PLAN §4.1 correctly says to use the
Historical Forecast API rather than the Historical Weather API, which is ERA5
reanalysis — conditions reconstructed after the fact, and a direct leak.

Choosing the right endpoint turns out not to be enough. Measured 18 Aug 2026
over 26–28 July, mean absolute difference against ERA5 for the same 72 hours:

| `models=` | vs ERA5 |
|---|---|
| unspecified | **0.000 °C** |
| `ecmwf_ifs025` | 0.586 °C |
| `icon_seamless` | 0.954 °C |
| `gfs_seamless` | 1.018 °C |

The named models disagree with each other by up to 1.519 °C, which is what
distinct numerical weather models do. With no model named, the *forecast*
endpoint returns reanalysis exactly — and says nothing about it in the
response.

So the leak sat one level deeper than the plan described. Someone who followed
the plan to the letter, used the historical forecast API as instructed, and
never passed `models=`, would have trained on observations with no way of
knowing. `WeatherRequest.model` therefore defaults to `ecmwf_ifs025`.

That the chosen model is also the closest to ERA5 is shared lineage, not
leakage: ERA5 is ECMWF's reanalysis and IFS is ECMWF's forecast model, so they
agree more with each other than either does with GFS. The evidence that it is a
forecast is that its figure is not zero while the unspecified one is.

## Two market behaviours that look like bugs

Both were found by staging tests failing, and both are the market rather than
the parser. Recording them so nobody later "fixes" them.

**Prices pile up exactly at the administered cap.** NSW1 and SA1 top out at
`20300.00` to the cent during FY2025-26; TAS1 reaches `23200.00` on 13 July
2026, days into the new financial year, because the market price cap escalates
annually. The floor behaves the same way: QLD1 bottoms at exactly `-1000.00`.
So the extremes are not a long tail, they are a wall — 37 intervals sitting on
it. Those are the events the model exists to predict, and clipping them would
remove the target.

Range tests are set well clear of the cap rather than equal to it, because a
bound that tracks the cap fails every July.

**South Australian demand goes negative.** 738 intervals, to −311 MW, when
rooftop solar exceeds the entire state's load. Increasingly common as rooftop
PV grows, and a distinct regime rather than an error: a region generating more
than it consumes prices very differently from one that is short.

## Known data gaps

`make gaps` lists days where AEMO published fewer than 288 dispatch intervals.
It excludes the first and last calendar day of coverage, which are not gaps: an
interval stamped 00:00 covers 23:55 to midnight of the evening before and ships
in the previous day's rollup, so the oldest day is always short by exactly that
one and the newest holds only it.

One real gap in the archive as at 24 Aug 2026:

**10 March 2026 — DISPATCHIS lost 21 of 288 intervals between 18:35 and 20:25,
in two blocks of 13 and 8 with a brief recovery between them. P5MIN lost 24
runs the same day.** Both feeds short means an AEMO publication outage, not a
parser fault.

**The two counts differ, and that is the point.** A gap in the label does not
imply a gap in the benchmark, or the reverse, so the feature layer excludes the
**union** of both windows. Checking only DISPATCHIS would leave rows whose
AEMO forecast is absent — on the single comparison the project is built around.

Exclude whole windows rather than the individual missing intervals. The few
intervals that did publish inside the outage have no usable history either
side, so any lag or rolling feature computed on them reaches back into absent
data.

And say where it sits: 18:35 to 20:25 is the evening demand ramp, which is when
NSW prices spike. One evening in 375 barely moves the totals, but the loss is
not uniformly distributed and the README should not imply the training window
is.

**Joint coverage starts 13 August 2025**, not the 12th: the P5MIN archive
begins a day after DISPATCHIS, and the usable window is bounded by the later of
the two.

A missing interval is not a value to impute past. It is a period where the
market's state is unknown, so the feature layer refuses to emit a row whose
window is not fully populated rather than quietly returning a number built on
less data than it claims.

## Pace, and never argue with a refusal

NEMWeb is a public web server, not a metered API. AEMO grants permission to use
the data; it does not offer a quota.

Measured 18 Aug 2026: a first poller pass fetched 163 files in 52 seconds, about
three a second, and the 164th came back 403. Two rules came out of it, and both
are enforced in `NemwebClient`.

**One request a second by default.** A full 375-day archive backfill is one
request per day, so the pacing costs about six minutes across the whole thing.

**A 4xx is never retried.** Retrying a 403 is how a rate limit turns into a
block. A refusal raises `RateLimited`, the poller backs off for at least a
minute and doubles up to fifteen, and `Retry-After` is honoured when sent.

Most of those 163 requests were also avoidable — see the next section.

## One dispatch interval, one key

The same interval arrives by two routes with two different paths: live as its
own zip, backfilled inside a daily rollup. Keyed on the full path they look
like different files, so a poller run after a backfill re-fetched two days of
intervals it already held.

The innermost CSV name is common to both, so that is the key, and it is stored
on `raw.ingest_log.member_key` with a unique index. It is also predictable from
a directory listing, which is what lets a pass skip a file *before* spending a
request on it.

## Not every table gets loaded

Two constraint tables — `DISPATCH.CONSTRAINT` and `P5MIN.CONSTRAINTSOLUTION` —
are 97% and 98% of their files' rows, and across the archive they come to about
1.35 billion rows between them. Neither feeds a Phase 1 feature: interconnector
flows and their limits come from `DISPATCH.INTERCONNECTORRES`, which carries
the flow and both limits directly.

`src/nemfc/tables.py` holds the allowlist and the arithmetic behind it. Loading
it produces about 8.3 million rows. The parser skips unwanted data rows before
they reach the CSV machinery, which is the difference between minutes and hours
across a full backfill.

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
