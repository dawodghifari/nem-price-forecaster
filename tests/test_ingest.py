"""Table filtering, archive vintage, and the long-history spacing check."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from nemfc.aggregated import (
    FIVE_MINUTE_SETTLEMENT_START,
    MonthlySeries,
    _measure_spacing,
    expected_rows,
    months_between,
    trading_day,
)
from nemfc.db import FileVintage
from nemfc.ingest import ARCHIVE_PUBLISH_LAG, _day_from
from nemfc.mms import parse_mms_text
from nemfc.tables import PHASE1_TABLES
from nemfc.weather import DEFAULT_HOURLY, WeatherRequest, parse_response

# Two tables, one of which stands in for the constraint sets that are 98% of a
# real P5MIN file.
SAMPLE = """C,NEMP.WORLD,P5MIN,AEMO,PUBLIC,2026/08/17,20:08:00,1,,
I,P5MIN,REGIONSOLUTION,10,RUN_DATETIME,INTERVAL_DATETIME,REGIONID,RRP
D,P5MIN,REGIONSOLUTION,10,"2026/08/17 20:10:00","2026/08/17 20:10:00",NSW1,89.63
D,P5MIN,REGIONSOLUTION,10,"2026/08/17 20:10:00","2026/08/17 20:15:00",NSW1,91.10
I,P5MIN,CONSTRAINTSOLUTION,6,RUN_DATETIME,CONSTRAINTID,MARGINALVALUE
D,P5MIN,CONSTRAINTSOLUTION,6,"2026/08/17 20:10:00",X_1,0
D,P5MIN,CONSTRAINTSOLUTION,6,"2026/08/17 20:10:00",X_2,0
D,P5MIN,CONSTRAINTSOLUTION,6,"2026/08/17 20:10:00",X_3,0
C,"END OF REPORT",9
"""


def test_filter_keeps_only_allowlisted_data_rows():
    parsed = parse_mms_text(SAMPLE, tables={"REGIONSOLUTION"})
    assert len(parsed.get("REGIONSOLUTION")) == 2
    assert parsed.get("CONSTRAINTSOLUTION").empty


def test_filtered_tables_still_appear_in_the_inventory():
    """Headers are cheap, so the file's full shape stays visible in logs."""
    inventory = parse_mms_text(SAMPLE, tables={"REGIONSOLUTION"}).inventory()
    assert set(inventory["table"]) == {"REGIONSOLUTION", "CONSTRAINTSOLUTION"}


def test_filtering_does_not_break_the_footer_check():
    """Skipped rows are counted, so a filtered parse still verifies integrity."""
    full = parse_mms_text(SAMPLE)
    filtered = parse_mms_text(SAMPLE, tables={"REGIONSOLUTION"})
    assert full.observed_record_count == filtered.observed_record_count == 9
    assert filtered.record_count_matches is True
    assert filtered.observed_row_count == 2


def test_filter_is_case_insensitive():
    assert len(parse_mms_text(SAMPLE, tables={"regionsolution"}).get("REGIONSOLUTION")) == 2


def test_phase1_allowlist_covers_the_label_and_the_benchmark():
    assert "PRICE" in PHASE1_TABLES  # the label
    assert "REGIONSOLUTION" in PHASE1_TABLES  # AEMO's forecast, the benchmark
    assert "CONSTRAINTSOLUTION" not in PHASE1_TABLES  # 1.24 billion rows
    assert "CONSTRAINT" not in PHASE1_TABLES  # 104 million rows


def test_archive_vintage_is_pushed_out_by_the_publication_lag():
    """A rollup loses per-interval listing times, so generated + lag is used."""
    generated = datetime(2026, 8, 17, 10, 10, tzinfo=timezone.utc)
    live = FileVintage(source_file="a", generated_at=generated, listed_at=generated)
    archived = FileVintage(
        source_file="b",
        generated_at=generated,
        listed_at=generated + ARCHIVE_PUBLISH_LAG,
        vintage_source="archive",
    )
    assert archived.known_at > live.known_at
    assert archived.known_at - live.known_at == timedelta(minutes=12, seconds=30)


def test_known_at_never_precedes_generation():
    generated = datetime(2026, 8, 17, 10, 10, tzinfo=timezone.utc)
    vintage = FileVintage(source_file="a", generated_at=generated, listed_at=None)
    assert vintage.known_at == generated


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("PUBLIC_DISPATCHIS_20260815.zip", date(2026, 8, 15)),
        ("PUBLIC_P5MIN_20250806.zip", date(2025, 8, 6)),
        ("nonsense.zip", None),
    ],
)
def test_archive_filenames_resolve_to_days(filename, expected):
    assert _day_from(filename) == expected


def _series(month: date, minutes: int) -> MonthlySeries:
    stamps = pd.date_range("2026-08-01", periods=10, freq=f"{minutes}min")
    return MonthlySeries("NSW1", month, pd.DataFrame({"SETTLEMENTDATE": stamps}), minutes, "u")


def test_spacing_is_measured_not_assumed():
    frame = pd.DataFrame({"SETTLEMENTDATE": pd.date_range("2026-08-01", periods=6, freq="5min")})
    assert _measure_spacing(frame) == 5


def test_recent_months_are_expected_to_be_five_minute():
    assert _series(date(2026, 8, 1), 5).matches_expected_spacing
    assert not _series(date(2026, 8, 1), 30).matches_expected_spacing


def test_months_before_five_minute_settlement_are_expected_to_be_half_hourly():
    before = FIVE_MINUTE_SETTLEMENT_START - timedelta(days=1)
    assert _series(date(before.year, before.month, 1), 30).matches_expected_spacing


def test_expected_rows_counts_whole_past_months():
    today = date(2026, 8, 23)
    assert expected_rows(date(2026, 7, 1), 5, today) == 31 * 288
    assert expected_rows(date(2026, 2, 1), 30, today) == 28 * 48


def test_the_month_in_progress_is_only_counted_to_yesterday():
    """6,336 rows on 23 August is 22 complete days, not a gap."""
    assert expected_rows(date(2026, 8, 1), 5, date(2026, 8, 23)) == 22 * 288 == 6336


def test_months_between_is_inclusive_and_ordered():
    got = list(months_between(date(2025, 11, 1), date(2026, 2, 15)))
    assert got == [date(2025, 11, 1), date(2025, 12, 1), date(2026, 1, 1), date(2026, 2, 1)]


def test_trading_day_starts_at_four_am():
    """01:00 on 18 August belongs to trading day 17 August, as DISPATCHIS said."""
    from nemfc.nemweb import NEM_TZ

    one_am = datetime(2026, 8, 18, 1, 0, tzinfo=NEM_TZ)
    five_am = datetime(2026, 8, 18, 5, 0, tzinfo=NEM_TZ)
    assert trading_day(one_am) == date(2026, 8, 17)
    assert trading_day(five_am) == date(2026, 8, 18)


def test_weather_defaults_to_the_strict_vintage():
    """With zero lead, a value is usable exactly when it becomes valid."""
    request = WeatherRequest("NSW1", date(2026, 8, 1), date(2026, 8, 1))
    frame = parse_response(
        {
            "hourly": {
                "time": ["2026-08-01T00:00", "2026-08-01T01:00"],
                "temperature_2m": [9.0, 8.5],
            }
        },
        request,
    )
    assert (frame["known_at"] == frame["valid_at"]).all()


def test_weather_lead_moves_the_vintage_earlier():
    request = WeatherRequest("NSW1", date(2026, 8, 1), date(2026, 8, 1), lead=timedelta(hours=6))
    frame = parse_response(
        {"hourly": {"time": ["2026-08-01T12:00"], "temperature_2m": [14.0]}}, request
    )
    assert frame["known_at"].iloc[0] == frame["valid_at"].iloc[0] - timedelta(hours=6)
    assert frame["lead_seconds"].iloc[0] == 21600


def test_weather_request_uses_the_forecast_archive_not_reanalysis():
    from nemfc.weather import BASE

    assert "historical-forecast-api" in BASE
    assert "archive-api" not in BASE


def test_a_model_is_always_named():
    """Unnamed, the forecast endpoint returns reanalysis. Measured 18 Aug 2026."""
    request = WeatherRequest("NSW1", date(2026, 8, 1), date(2026, 8, 1))
    assert request.model == "ecmwf_ifs025"
    assert request.params()["models"] == "ecmwf_ifs025"


def test_the_model_can_be_overridden_but_not_silently_dropped():
    named = WeatherRequest("NSW1", date(2026, 8, 1), date(2026, 8, 1), model="gfs_seamless")
    assert named.params()["models"] == "gfs_seamless"

    # Passing None is the leaky path, so it must be an explicit choice that is
    # visible in the request rather than a default anyone could fall into.
    unnamed = WeatherRequest("NSW1", date(2026, 8, 1), date(2026, 8, 1), model=None)
    assert "models" not in unnamed.params()


def test_every_region_has_a_load_centre():
    for region in ("NSW1", "VIC1", "QLD1", "SA1", "TAS1"):
        lat, lon = WeatherRequest(region, date(2026, 8, 1), date(2026, 8, 1)).coordinates
        assert -45 < lat < -10 and 110 < lon < 155


def test_unknown_region_fails_loudly():
    request = WeatherRequest("XXX1", date(2026, 8, 1), date(2026, 8, 1))
    with pytest.raises(KeyError, match="no load centre"):
        _ = request.coordinates


def test_default_hourly_variables_cover_demand_and_supply_drivers():
    assert "temperature_2m" in DEFAULT_HOURLY  # demand
    assert "wind_speed_100m" in DEFAULT_HOURLY  # wind generation at hub height
    assert "shortwave_radiation" in DEFAULT_HOURLY  # solar
