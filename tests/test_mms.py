"""Parser tests. No network, so CI can run them on a clean clone."""

from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime, timezone

import pytest

from nemfc.db import raw_table_name
from nemfc.mms import TableKey, parse_mms_text, read_mms_zip_single

SAMPLE = """C,NEMP.WORLD,DISPATCHIS,AEMO,PUBLIC,2026/08/17,20:08:00,0000000533127148,DISPATCHIS,0000000533127148
I,DISPATCH,CASE_SOLUTION,2,SETTLEMENTDATE,RUNNO,INTERVENTION
D,DISPATCH,CASE_SOLUTION,2,"2026/08/17 20:10:00",1,0
I,DISPATCH,PRICE,5,SETTLEMENTDATE,RUNNO,REGIONID,INTERVENTION,RRP,ROP
D,DISPATCH,PRICE,5,"2026/08/17 20:10:00",1,NSW1,0,89.63,89.63
D,DISPATCH,PRICE,5,"2026/08/17 20:10:00",1,VIC1,0,74.21,74.21
D,DISPATCH,PRICE,5,"2026/08/17 20:10:00",1,QLD1,1,15100.00,88.10
C,"END OF REPORT",4
"""


def test_tables_and_columns_come_from_the_i_record():
    parsed = parse_mms_text(SAMPLE, source_name="sample")
    assert set(parsed.tables) == {
        TableKey("DISPATCH", "CASE_SOLUTION", "2"),
        TableKey("DISPATCH", "PRICE", "5"),
    }
    price = parsed.get("PRICE")
    assert list(price.columns) == [
        "SETTLEMENTDATE",
        "RUNNO",
        "REGIONID",
        "INTERVENTION",
        "RRP",
        "ROP",
    ]
    assert len(price) == 3


def test_header_and_footer():
    parsed = parse_mms_text(SAMPLE, source_name="sample")
    assert parsed.header.report_id == "DISPATCHIS"
    assert parsed.header.generated_at_local == datetime(2026, 8, 17, 20, 8, 0)
    assert parsed.header.serial == "0000000533127148"
    assert parsed.observed_row_count == 4


def test_footer_counts_every_record_not_just_data_rows():
    """4 D rows + 2 I records + 2 C records = 8, which is what the footer says.

    Measured on live files 18 Aug 2026: a DISPATCHIS footer claiming 997 held
    988 D rows across 7 tables (988 + 7 + 2 = 997), and a P5MIN footer claiming
    11766 held 11759 across 5 (11759 + 5 + 2 = 11766).
    """
    parsed = parse_mms_text(SAMPLE.replace('C,"END OF REPORT",4', 'C,"END OF REPORT",8'))
    assert parsed.observed_row_count == 4
    assert parsed.observed_record_count == 8
    assert parsed.declared_record_count == 8
    assert parsed.record_count_matches is True


def test_intervention_rows_are_kept_not_merged():
    """Both solutions survive parsing; filtering is the staging layer's job."""
    price = parse_mms_text(SAMPLE).get("PRICE")
    qld = price[price["REGIONID"] == "QLD1"].iloc[0]
    assert qld["INTERVENTION"] == "1"
    assert qld["RRP"] != qld["ROP"]


def test_quoted_commas_survive():
    text = (
        "C,NEMP.WORLD,BIDMOVE,AEMO,PUBLIC,2026/08/17,20:08:00,1,,\n"
        "I,BIDS,BIDDAYOFFER,2,DUID,REASON\n"
        'D,BIDS,BIDDAYOFFER,2,BW01,"unit trip, returning to service"\n'
        'C,"END OF REPORT",1\n'
    )
    df = parse_mms_text(text).get("BIDDAYOFFER")
    assert df.iloc[0]["REASON"] == "unit trip, returning to service"


def test_same_table_two_versions_stay_separate():
    text = (
        "C,NEMP.WORLD,X,AEMO,PUBLIC,2026/08/17,20:08:00,1,,\n"
        "I,DISPATCH,PRICE,5,SETTLEMENTDATE,RRP\n"
        'D,DISPATCH,PRICE,5,"2026/08/17 20:10:00",10\n'
        "I,DISPATCH,PRICE,6,SETTLEMENTDATE,RRP,NEWCOL\n"
        'D,DISPATCH,PRICE,6,"2026/08/17 20:10:00",10,x\n'
        'C,"END OF REPORT",2\n'
    )
    parsed = parse_mms_text(text)
    assert len(parsed.tables) == 2
    with pytest.raises(KeyError, match="ambiguous"):
        parsed.get("PRICE")


def test_short_rows_are_padded_and_recorded():
    text = (
        "C,NEMP.WORLD,X,AEMO,PUBLIC,2026/08/17,20:08:00,1,,\n"
        "I,DISPATCH,PRICE,5,SETTLEMENTDATE,RRP,ROP\n"
        'D,DISPATCH,PRICE,5,"2026/08/17 20:10:00",10\n'
        'C,"END OF REPORT",1\n'
    )
    parsed = parse_mms_text(text)
    assert parsed.get("PRICE").iloc[0]["ROP"] is None
    assert [a.kind for a in parsed.anomalies] == ["short_row"]


def test_long_rows_are_truncated_and_recorded():
    text = (
        "C,NEMP.WORLD,X,AEMO,PUBLIC,2026/08/17,20:08:00,1,,\n"
        "I,DISPATCH,PRICE,5,SETTLEMENTDATE,RRP\n"
        'D,DISPATCH,PRICE,5,"2026/08/17 20:10:00",10,extra\n'
        'C,"END OF REPORT",1\n'
    )
    parsed = parse_mms_text(text)
    assert list(parsed.get("PRICE").iloc[0]) == ["2026/08/17 20:10:00", "10"]
    assert [a.kind for a in parsed.anomalies] == ["long_row"]


def test_data_without_a_header_is_flagged_not_silently_dropped():
    text = (
        "C,NEMP.WORLD,X,AEMO,PUBLIC,2026/08/17,20:08:00,1,,\n"
        'D,DISPATCH,PRICE,5,"2026/08/17 20:10:00",10\n'
        'C,"END OF REPORT",1\n'
    )
    parsed = parse_mms_text(text)
    assert [a.kind for a in parsed.anomalies] == ["orphan_data"]
    assert parsed.observed_row_count == 0


def test_footer_mismatch_is_detected():
    text = SAMPLE.replace('C,"END OF REPORT",4', 'C,"END OF REPORT",99')
    assert parse_mms_text(text).record_count_matches is False


def test_missing_table_error_lists_what_is_present():
    with pytest.raises(KeyError, match=re.escape("DISPATCH.PRICE.v5")):
        parse_mms_text(SAMPLE, source_name="s").get("REGIONSUM")


def _zip_bytes(members: dict[str, str]) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    buf.seek(0)
    return buf


def test_reads_a_single_member_zip():
    parsed = read_mms_zip_single(_zip_bytes({"PUBLIC_DISPATCHIS_202608172010.CSV": SAMPLE}))
    assert len(parsed.get("PRICE")) == 3


def test_reads_a_nested_archive_rollup():
    from nemfc.mms import read_mms_zip

    inner = _zip_bytes({"a.CSV": SAMPLE}).getvalue()
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as zf:
        zf.writestr("PUBLIC_DISPATCHIS_20260817.zip", inner)
        zf.writestr("PUBLIC_DISPATCHIS_20260818.zip", inner)
    outer.seek(0)
    assert len(read_mms_zip(outer)) == 2


def test_raw_table_name_carries_the_version():
    assert raw_table_name(TableKey("DISPATCH", "PRICE", "5")) == "dispatch_price_v5"


def test_utc_conversion_is_fixed_offset_all_year():
    """AEST has no daylight saving, so January and July convert identically."""
    from nemfc.nemweb import parse_report_filename

    jan = parse_report_filename("PUBLIC_P5MIN_202601150025_20260115002033.zip")
    jul = parse_report_filename("PUBLIC_P5MIN_202607150025_20260715002033.zip")
    assert jan["interval_label_utc"] == datetime(2026, 1, 14, 14, 25, tzinfo=timezone.utc)
    assert jul["interval_label_utc"] == datetime(2026, 7, 14, 14, 25, tzinfo=timezone.utc)
