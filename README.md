# NEM Price Spike Forecaster

Forecasting the probability that the NSW spot electricity price exceeds
$300/MWh in the next 30 minutes, from AEMO's live five-minute dispatch feed,
measured against AEMO's own published pre-dispatch forecast.

**Status: Day 2 of 7 in Phase 1. No model exists yet and no results are
claimed.** The build plan, including the reasoning behind each design decision,
is in [PLAN.md](PLAN.md).

## Why this problem

The National Electricity Market sets a spot price for each region every five
minutes. It normally sits somewhere between $30 and $150/MWh, but it can reach
$17,500 when supply gets tight — a unit trips, an interconnector hits its
limit, or the evening demand ramp arrives on a low-wind day. Battery and
demand-response operators decide whether to dispatch against that price, so a
probability of a spike in the next half hour is a decision, not a statistic.

AEMO already publishes its own five-minute pre-dispatch forecast, which makes
this one of the few forecasting problems where a production incumbent is
available to benchmark against. That comparison, rather than the model, is the
point of the project.

## What exists today

| | |
|---|---|
| MMS flat-file parser | Reads the column set from the file's own `I` records rather than an assumed schema, keeps intervention rows, records malformed rows instead of dropping them |
| NEMWeb client | Directory listings, retries, and both publication timestamps captured per file |
| Raw schema | Postgres, append only, three vintage timestamps per row |
| Format verification | `make verify` downloads live files and reconciles the parsed prices against AEMO's separately published series |
| Live poller | Ingests every file in `Reports/Current` not already loaded, so a missed pass recovers itself |
| Archive backfill | Resumable, table-filtered, with a weaker vintage recorded for backfilled rows |
| Long history | AEMO's aggregated series with per-file resolution detection |
| Weather | Open-Meteo archived forecasts, strict vintage by default |
| Tests | 54, offline |

## Running it

```bash
cp .env.example .env
make up        # Postgres 16 + Adminer on :8080
make test      # offline test suite
make verify    # downloads live NEMWeb files, writes docs/day1_format_findings.md
```

`make verify` is Day 1's exit condition. It answers, from real files rather
than documentation: what tables DISPATCHIS and P5MIN actually contain, whether
`SETTLEMENTDATE` labels the start or the end of its interval, how long after
generation a pre-dispatch run actually appears on the site, and whether the
parsed five-minute prices reconcile against AEMO's aggregated series.

Findings are written to `docs/day1_format_findings.md`, which is generated and
not hand-edited.

## Design notes worth reading first

[docs/CONVENTIONS.md](docs/CONVENTIONS.md) covers the decisions that are cheap
now and expensive to retrofit: UTC storage against a fixed AEST offset, the
three vintage timestamps and why only one of them may be joined on, append-only
raw tables, and how intervention pricing is handled.

The short version of the one that shaped the schema: a pre-dispatch file's
internal timestamp is not when it became readable. One run generated at
23:35:33 did not appear on NEMWeb until 23:48. Since AEMO's forecast is both a
model feature and the benchmark, joining on the internal timestamp would let
the model read a forecast twelve minutes before anyone could have.

## Data

Market data © AEMO, used under AEMO's copyright permissions notice. Weather
from Open-Meteo, CC BY 4.0.

`data/` is not committed. It is reproduced with `make backfill`.
