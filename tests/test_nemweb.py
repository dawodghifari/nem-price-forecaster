"""Filename and directory-index parsing. These carry the vintage timestamps."""

from __future__ import annotations

from datetime import datetime, timezone

from nemfc.nemweb import parse_directory_listing, parse_report_filename

LISTING = """<html><head><title>nemweb.com.au - /Reports/CURRENT/P5_Reports/</title></head>
<body><H1>nemweb.com.au - /Reports/CURRENT/P5_Reports/</H1><hr><pre>
<A HREF="/Reports/Current/">[To Parent Directory]</A><br><br>
 Monday, August 17, 2026 11:35 PM       190132 <A HREF="/Reports/Current/P5_Reports/PUBLIC_P5MIN_202608172330_20260817232532.zip">PUBLIC_P5MIN_202608172330_20260817232532.zip</A><br>
 Monday, August 17, 2026 11:48 PM       190175 <A HREF="/Reports/Current/P5_Reports/PUBLIC_P5MIN_202608172340_20260817233533.zip">PUBLIC_P5MIN_202608172340_20260817233533.zip</A><br>
Tuesday, August 18, 2026 12:24 AM       189731 <A HREF="/Reports/Current/P5_Reports/PUBLIC_P5MIN_202608180025_20260818002033.zip">PUBLIC_P5MIN_202608180025_20260818002033.zip</A><br>
</pre><hr></body></html>
"""


def test_listing_yields_one_entry_per_zip():
    entries = parse_directory_listing(LISTING)
    assert [e.filename for e in entries] == [
        "PUBLIC_P5MIN_202608172330_20260817232532.zip",
        "PUBLIC_P5MIN_202608172340_20260817233533.zip",
        "PUBLIC_P5MIN_202608180025_20260818002033.zip",
    ]
    assert entries[0].size_bytes == 190132
    assert entries[0].url.startswith("https://nemweb.com.au/")


def test_the_parent_link_is_not_treated_as_a_file():
    assert all(e.filename.endswith(".zip") for e in parse_directory_listing(LISTING))


def test_publish_lag_is_measured_from_generation_to_listing():
    """The gap the point-in-time join has to respect."""
    entries = {e.filename: e for e in parse_directory_listing(LISTING)}
    slow = entries["PUBLIC_P5MIN_202608172340_20260817233533.zip"]
    assert slow.generated_at_utc == datetime(2026, 8, 17, 13, 35, 33, tzinfo=timezone.utc)
    assert slow.listed_at_utc == datetime(2026, 8, 17, 13, 48, tzinfo=timezone.utc)
    assert slow.publish_lag.total_seconds() / 60 == 12.45

    quick = entries["PUBLIC_P5MIN_202608180025_20260818002033.zip"]
    assert round(quick.publish_lag.total_seconds() / 60, 2) == 3.45


def test_known_at_takes_the_later_timestamp():
    entry = parse_directory_listing(LISTING)[1]
    assert entry.known_at_utc == entry.listed_at_utc
    assert entry.known_at_utc > entry.generated_at_utc


def test_p5min_tail_is_a_generation_time():
    meta = parse_report_filename("PUBLIC_P5MIN_202608180025_20260818002033.zip")
    assert meta["report"] == "P5MIN"
    assert meta["generated_at_utc"] == datetime(2026, 8, 17, 14, 20, 33, tzinfo=timezone.utc)
    assert meta["serial"] is None


def test_dispatchis_tail_is_a_sequence_number_not_a_time():
    meta = parse_report_filename("PUBLIC_DISPATCHIS_202608172010_0000000533127148.zip")
    assert meta["report"] == "DISPATCHIS"
    assert meta["generated_at_utc"] is None
    assert meta["serial"] == "0000000533127148"
    assert meta["interval_label_utc"] == datetime(2026, 8, 17, 10, 10, tzinfo=timezone.utc)


def test_unrecognised_filename_returns_empty_metadata_rather_than_raising():
    meta = parse_report_filename("something_else.zip")
    assert meta == {
        "report": None,
        "interval_label_utc": None,
        "generated_at_utc": None,
        "serial": None,
    }
