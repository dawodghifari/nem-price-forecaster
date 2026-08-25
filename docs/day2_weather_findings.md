# Day 2 — Open-Meteo archive findings

Generated 23 Aug 2026 11:00 UTC for NSW1 at (-33.87, 151.21).

Forecast archive against reanalysis, over two windows. ERA5 reanalysis runs about
5 days behind real time, so a comparison inside that lag tests nothing.

| window | dates | hours | mean abs diff | max diff | verdict |
|---|---|---|---|---|---|
| recent | 2026-08-18 to 2026-08-20 | 72 | 0.000 °C | 0.000 °C | identical |
| settled | 2026-07-31 to 2026-08-02 | 72 | 0.000 °C | 0.000 °C | identical |

- **recent** — inside the ERA5 lag; expected to match, which is the warning
- **settled** — well outside the ERA5 lag; this is the real test

## Named models, over the settled window

| model | hours | mean abs diff vs ERA5 |
|---|---|---|
| unspecified | 72 | 0.000 °C |
| ecmwf_ifs025 | 72 | 0.499 °C |
| gfs_seamless | 72 | 0.857 °C |
| icon_seamless | 72 | 1.317 °C |

Widest disagreement between two named models: **1.843 °C**.

The named models disagree with each other, so they are genuine model output.
Pin one in `WeatherRequest.model` rather than leaving the choice to the API.


## What this means

**The archive holds real forecasts, but only if a model is named.**

Three named models disagree with each other by up to 1.843 °C, which is what
distinct numerical weather models do. Left unspecified, though, the endpoint returned
values matching ERA5 reanalysis to 0.000 °C — that is, exactly.

So the leak PLAN §4.1 warned about was real and was hiding one level deeper than the
plan described. Choosing the historical forecast endpoint over the reanalysis one is
not sufficient; the forecast endpoint hands back reanalysis unless a model is named,
and nothing in the response says so.

`WeatherRequest.model` is therefore set, and defaults to `ecmwf_ifs025`.

Worth stating plainly, because it looks wrong at a glance: the chosen model is
also the closest to reanalysis, at 0.499 °C. That is shared lineage,
not leakage — ERA5 is ECMWF's reanalysis and IFS is ECMWF's forecast model, so
they agree more with each other than either does with GFS. The evidence that it
is a forecast is that the figure is not zero while the unspecified one is.

**Both endpoints also agreed inside the ERA5 lag**, where reanalysis does not yet
exist and the archive endpoint fills from forecast output without saying so. A
separate reason to keep the reanalysis endpoint out of the feature table entirely.

## The vintage rule

`src/nemfc/weather.py` treats a value valid at `V` as knowable only from `V`, because
the response does not say which run produced it. At a thirty-minute horizon that costs
little: temperature and wind move slowly enough that conditions at prediction time carry
nearly all the signal that conditions at the target interval would.

Loosen `WeatherRequest.lead` only with evidence about how old the archived runs are.
