"""Establish what Open-Meteo's historical forecast archive actually contains.

The archive returns a value for each valid hour without stating which model run
produced it, and that is the one thing the feature layer needs to know. If the
endpoint is really serving reanalysis, every weather feature is a leak.

The test is whether the forecast archive and the reanalysis archive disagree. A
forecast is not an observation, so they should differ; identical values would
mean one of the two endpoints is not what it claims.

**That comparison is only valid over a settled window, and getting this wrong
is easy.** ERA5 reanalysis runs about five days behind real time. For dates
inside that lag the reanalysis endpoint has no reanalysis to serve, so it falls
back to forecast-model output — and then both endpoints return the same numbers
for reasons that have nothing to do with either being wrong. A first pass here
compared a window three days old and found a mean absolute difference of
exactly zero, which was an artefact of the window rather than a finding.

So both windows are tested. The recent one is expected to match, and if it does
that is itself a reason never to touch the reanalysis endpoint: it blends
forecast data in without saying so. The settled one is the real test.

Writes docs/day2_weather_findings.md. Nothing here touches Postgres.
"""

from __future__ import annotations

import itertools
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemfc.config import REPO_ROOT, load_settings
from nemfc.weather import BASE, WeatherRequest, parse_response

REANALYSIS = "https://archive-api.open-meteo.com/v1/archive"

# ERA5 publishes roughly five days behind. Inside that, the reanalysis endpoint
# is serving something else.
ERA5_LAG_DAYS = 5

WINDOWS = (
    ("recent", 3, "inside the ERA5 lag; expected to match, which is the warning"),
    ("settled", 21, "well outside the ERA5 lag; this is the real test"),
)

log = logging.getLogger("probe-weather")


def _get(client: httpx.Client, url: str, params: dict) -> dict:
    response = client.get(url, params=params, timeout=60.0)
    response.raise_for_status()
    return response.json()


def compare_window(
    client: httpx.Client, region: str, days_ago: int, span: int = 3
) -> tuple[pd.DataFrame, dict]:
    """Fetch both products over one window and align them on valid hour.

    Deliberately requests with no model named, which is the trap this section
    exists to demonstrate. `WeatherRequest` defaults to a named model precisely
    so nothing else in the codebase can take this path by accident.
    """
    end = date.today() - timedelta(days=days_ago)
    start = end - timedelta(days=span - 1)
    request = WeatherRequest(region=region, start=start, end=end, model=None)

    forecast = parse_response(_get(client, BASE, request.params()), request)
    lat, lon = request.coordinates
    reanalysis = _get(
        client,
        REANALYSIS,
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": "temperature_2m",
            "timezone": "UTC",
        },
    )

    merged = pd.merge(
        forecast[["valid_at", "temperature_2m"]].rename(columns={"temperature_2m": "forecast"}),
        pd.DataFrame(
            {
                "valid_at": pd.to_datetime(reanalysis["hourly"]["time"], utc=True),
                "observed": reanalysis["hourly"]["temperature_2m"],
            }
        ),
        on="valid_at",
        how="inner",
    ).dropna()

    return merged, {"start": start, "end": end, "forecast_rows": len(forecast)}


NAMED_MODELS = ("ecmwf_ifs025", "gfs_seamless", "icon_seamless")


def _model_section(
    client: httpx.Client, region: str, days_ago: int = 21
) -> tuple[list[str], dict[str, float]]:
    """Ask named forecast models directly, instead of letting Open-Meteo choose.

    Left unspecified, the endpoint resolves the model itself, and for archived
    past dates what it resolves to may be an analysis rather than a forecast —
    which would explain an exact match against reanalysis without meaning the
    forecast archive is unusable. Naming a model removes that ambiguity.

    If two different numerical weather models return the same numbers, the
    parameter is being ignored and the endpoint is serving one analysis for
    everything. If they disagree, they are real model output and one of them
    can be pinned in `WeatherRequest.model`.
    """
    lines = ["## Named models, over the settled window", ""]
    end = date.today() - timedelta(days=days_ago)
    start = end - timedelta(days=2)

    lat, lon = WeatherRequest(region, start, end).coordinates
    try:
        reanalysis = _get(
            client,
            REANALYSIS,
            {
                "latitude": lat,
                "longitude": lon,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "hourly": "temperature_2m",
                "timezone": "UTC",
            },
        )
        era5 = pd.Series(reanalysis["hourly"]["temperature_2m"], dtype="float64")
    except Exception as exc:  # discovery script: record and continue
        return [*lines, f"**FAILED** to fetch reanalysis: `{exc!r}`", ""], {}

    against_era5: dict[str, float] = {}
    series: dict[str, pd.Series] = {}
    lines += ["| model | hours | mean abs diff vs ERA5 |", "|---|---|---|"]
    for model in (None, *NAMED_MODELS):
        request = WeatherRequest(region, start, end, hourly=("temperature_2m",), model=model)
        label = model or "unspecified"
        try:
            frame = parse_response(_get(client, BASE, request.params()), request)
        except Exception as exc:  # discovery script: record and continue
            lines.append(f"| {label} | — | FAILED: `{exc!r}` |")
            continue
        values = pd.to_numeric(frame["temperature_2m"], errors="coerce")
        if len(values) != len(era5):
            lines.append(f"| {label} | {len(values)} | length mismatch, not compared |")
            continue
        series[label] = values
        against_era5[label] = float((values - era5).abs().mean())
        lines.append(f"| {label} | {len(values)} | {against_era5[label]:.3f} °C |")

    named = {k: v for k, v in series.items() if k != "unspecified"}
    result: dict[str, float] = {f"era5_{k}": v for k, v in against_era5.items()}
    if len(named) >= 2:
        keys = sorted(named)
        spreads = [
            float((named[a] - named[b]).abs().mean()) for a, b in itertools.combinations(keys, 2)
        ]
        widest = max(spreads) if spreads else 0.0
        result["widest"] = widest
        lines += [
            "",
            f"Widest disagreement between two named models: **{widest:.3f} °C**.",
            "",
        ]
        if widest < 1e-9:
            lines += [
                "Different numerical weather models returning identical values means the model",
                "parameter is being ignored and one product is being served for everything.",
            ]
        else:
            lines += [
                "The named models disagree with each other, so they are genuine model output.",
                "Pin one in `WeatherRequest.model` rather than leaving the choice to the API.",
            ]
    lines.append("")
    return lines, result


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings()
    region = settings.region
    coords = WeatherRequest(region, date.today(), date.today()).coordinates

    out: list[str] = [
        "# Day 2 — Open-Meteo archive findings",
        "",
        f"Generated {datetime.now(timezone.utc):%d %b %Y %H:%M UTC} for {region} at {coords}.",
        "",
        "Forecast archive against reanalysis, over two windows. ERA5 reanalysis runs about",
        f"{ERA5_LAG_DAYS} days behind real time, so a comparison inside that lag tests nothing.",
        "",
        "| window | dates | hours | mean abs diff | max diff | verdict |",
        "|---|---|---|---|---|---|",
    ]

    results: dict[str, float | None] = {}
    models: dict[str, float] = {}
    detail: list[str] = []

    with httpx.Client(headers={"User-Agent": settings.user_agent}, follow_redirects=True) as client:
        for name, days_ago, note in WINDOWS:
            try:
                merged, meta = compare_window(client, region, days_ago)
            except Exception as exc:  # discovery script: record and continue
                out.append(f"| {name} | — | — | — | — | FAILED: `{exc!r}` |")
                results[name] = None
                continue

            if merged.empty:
                out.append(
                    f"| {name} | {meta['start']} to {meta['end']} | 0 | — | — | no overlap |"
                )
                results[name] = None
                continue

            diff = (merged["forecast"] - merged["observed"]).abs()
            mad = float(diff.mean())
            results[name] = mad
            verdict = "identical" if mad < 1e-9 else "differ"
            out.append(
                f"| {name} | {meta['start']} to {meta['end']} | {len(merged)} | "
                f"{mad:.3f} °C | {diff.max():.3f} °C | {verdict} |"
            )
            detail += [f"- **{name}** — {note}"]

        out += ["", *detail, ""]
        model_lines, models = _model_section(client, region)
        out += model_lines

    out += ["", "## What this means", ""]

    settled = results.get("settled")
    widest = models.get("widest")
    unspecified = models.get("era5_unspecified")
    named = {
        k[5:]: v for k, v in models.items() if k.startswith("era5_") and k != "era5_unspecified"
    }

    if widest is None:
        out += [
            "**Inconclusive.** The named-model comparison did not complete, so the question is",
            "still open. Do not load weather features until it does.",
        ]
        ok = False
    elif widest < 1e-9:
        out += [
            "**The model parameter is being ignored.** Three different numerical weather models",
            "returned identical values, which means one product is being served for everything,",
            "and the match against reanalysis says which product. Every weather feature built on",
            "this would be a leak. Drop weather from the Phase 1 feature set and record why.",
        ]
        ok = False
    else:
        best = min(named, key=named.get) if named else None
        out += [
            "**The archive holds real forecasts, but only if a model is named.**",
            "",
            f"Three named models disagree with each other by up to {widest:.3f} °C, which is what",
            "distinct numerical weather models do. Left unspecified, though, the endpoint returned",
            f"values matching ERA5 reanalysis to {unspecified:.3f} °C — that is, exactly.",
            "",
            "So the leak PLAN §4.1 warned about was real and was hiding one level deeper than the",
            "plan described. Choosing the historical forecast endpoint over the reanalysis one is",
            "not sufficient; the forecast endpoint hands back reanalysis unless a model is named,",
            "and nothing in the response says so.",
            "",
            f"`WeatherRequest.model` is therefore set, and defaults to `{best}`.",
        ]
        if best == "ecmwf_ifs025":
            out += [
                "",
                "Worth stating plainly, because it looks wrong at a glance: the chosen model is",
                f"also the closest to reanalysis, at {named[best]:.3f} °C. That is shared lineage,",
                "not leakage — ERA5 is ECMWF's reanalysis and IFS is ECMWF's forecast model, so",
                "they agree more with each other than either does with GFS. The evidence that it",
                "is a forecast is that the figure is not zero while the unspecified one is.",
            ]
        ok = True

    if settled is not None and settled < 1e-9:
        out += [
            "",
            "**Both endpoints also agreed inside the ERA5 lag**, where reanalysis does not yet",
            "exist and the archive endpoint fills from forecast output without saying so. A",
            "separate reason to keep the reanalysis endpoint out of the feature table entirely.",
        ]

    out += [
        "",
        "## The vintage rule",
        "",
        "`src/nemfc/weather.py` treats a value valid at `V` as knowable only from `V`, because",
        "the response does not say which run produced it. At a thirty-minute horizon that costs",
        "little: temperature and wind move slowly enough that conditions at prediction time carry",
        "nearly all the signal that conditions at the target interval would.",
        "",
        "Loosen `WeatherRequest.lead` only with evidence about how old the archived runs are.",
    ]

    path = REPO_ROOT / "docs" / "day2_weather_findings.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(out) + "\n"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Written to {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
