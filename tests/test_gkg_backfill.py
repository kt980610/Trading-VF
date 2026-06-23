"""Tests for the read-only GKG historical feature backfill tool.

These exercise the pure aggregation / selection / subsampling logic only; no
network access happens (we hand-build tiny in-memory GKG zips).
"""

from __future__ import annotations

import os
import sys
import zipfile
from datetime import datetime, timezone

import pytest

# The backfill script lives in scripts/ and imports its sibling probe via a
# sys.path insert; mirror that here so the import resolves.
_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import gkg_historical_feature_backfill as bf  # noqa: E402
import gkg_coverage_probe as probe  # noqa: E402


def _row(url="", title="", themes_v1="", themes_v2="", tone0=0.0, domain="example.com",
         date="20240501000000"):
    """Build one valid 27-field GKG v2.1 row."""
    fields = [""] * probe.GKG_FIELD_COUNT
    fields[probe.F_DATE] = date
    fields[probe.F_DOMAIN] = domain
    fields[probe.F_URL] = url
    fields[probe.F_THEMES_V1] = themes_v1
    fields[probe.F_THEMES_V2] = themes_v2
    fields[probe.F_TONE] = f"{tone0},2.0,1.0,3.0,4.0,5.0,100"
    if title:
        fields[probe.F_EXTRAS_XML] = f"<PAGE_TITLE>{title}</PAGE_TITLE>"
    return "\t".join(fields)


def _make_zip(tmp_path, ts, lines):
    csv = "\n".join(lines)
    zp = tmp_path / f"{ts}.gkg.csv.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr(f"{ts}.gkg.csv", csv)
    return zp


# --------------------------------------------------------------------------- #
# Selection vocab helpers
# --------------------------------------------------------------------------- #
def test_macro_categories_match_theme_tokens():
    assert bf._macro_categories("ARMEDCONFLICT;TAX_FNCACT") == {"conflict"}
    assert bf._macro_categories("ECON_INTEREST_RATE") == {"rates"}
    assert bf._macro_categories("ELECTION;GENERAL_GOVERNMENT") == {"politics"}
    assert bf._macro_categories("GOLD") == {"gold"}
    assert bf._macro_categories("ECON_CURRENCY;FOREX") == {"fx"}
    assert bf._macro_categories("SPORT;ENTERTAINMENT") == set()


def test_btc_vs_altcoins_regex():
    assert bf._BTC_RE.search("bitcoin price surges")
    assert bf._BTC_RE.search("btc rallies")
    assert not bf._BTC_RE.search("the method works")  # 'btc'/'eth' boundary guard
    assert bf._ALT_RE.search("ethereum upgrade")
    assert bf._ALT_RE.search("solana defi")
    assert not bf._ALT_RE.search("canada exports")  # 'ada' inside word must not hit
    assert not bf._ALT_RE.search("click this dot here")  # bare 'dot' excluded


def test_tone_stats():
    mean, vol = bf._tone_stats(0.0, 0.0, 0)
    assert mean is None and vol is None
    mean, vol = bf._tone_stats(4.0, 16.0, 2)  # values 0 and 4 -> mean 2, std 2
    assert mean == pytest.approx(2.0)
    assert vol == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# aggregate_slot
# --------------------------------------------------------------------------- #
def test_aggregate_slot_crypto_and_macro(tmp_path):
    ts = "20240501000000"
    lines = [
        _row(title="Bitcoin rallies hard", tone0=2.0),
        _row(url="https://news.x/ethereum-merge", tone0=-1.0),
        _row(themes_v1="ARMEDCONFLICT;TAX_FNCACT", tone0=3.0),
        _row(themes_v2="ECON_INTEREST_RATE", tone0=1.0),
        _row(url="https://x/sports", title="Football final", themes_v1="SPORT"),
    ]
    zp = _make_zip(tmp_path, ts, lines)
    feat, errors = bf.aggregate_slot(ts, zp, lag=15, downloaded=123)

    assert errors == []
    assert feat["status"] == "ok"
    assert feat["rows"] == 5
    assert feat["schema_errors"] == 0
    assert feat["source_coverage"] == 1.0
    assert feat["downloaded_bytes"] == 123

    # crypto: bitcoin row + ethereum row
    assert feat["crypto_candidates"] == 2
    assert feat["btc_count"] == 1
    assert feat["altcoins_count"] == 1
    assert feat["btc_tone_mean"] == pytest.approx(2.0)
    assert feat["altcoins_tone_mean"] == pytest.approx(-1.0)
    assert feat["crypto_tone_mean"] == pytest.approx(0.5)

    # macro
    assert feat["macro_conflict_count"] == 1
    assert feat["macro_rates_count"] == 1
    assert feat["macro_politics_count"] == 0
    assert feat["macro_conflict_tone_mean"] == pytest.approx(3.0)
    assert feat["macro_rates_tone_mean"] == pytest.approx(1.0)


def test_aggregate_slot_url_dedup_within_group(tmp_path):
    ts = "20240501000000"
    url = "https://news.x/ethereum-merge"
    lines = [
        _row(url=url, tone0=-1.0),
        _row(url=url, tone0=5.0),  # duplicate URL -> ignored in altcoins/crypto
    ]
    zp = _make_zip(tmp_path, ts, lines)
    feat, _ = bf.aggregate_slot(ts, zp, lag=15, downloaded=1)
    assert feat["altcoins_count"] == 1
    assert feat["crypto_candidates"] == 1
    assert feat["altcoins_tone_mean"] == pytest.approx(-1.0)


def test_aggregate_slot_row_in_btc_and_altcoins(tmp_path):
    ts = "20240501000000"
    lines = [_row(title="Bitcoin and ethereum both rally", tone0=4.0)]
    zp = _make_zip(tmp_path, ts, lines)
    feat, _ = bf.aggregate_slot(ts, zp, lag=15, downloaded=1)
    assert feat["btc_count"] == 1
    assert feat["altcoins_count"] == 1
    assert feat["crypto_candidates"] == 1  # one unique URL overall


def test_aggregate_slot_schema_error_fails_closed(tmp_path):
    ts = "20240501000000"
    bad = "\t".join(["x"] * (probe.GKG_FIELD_COUNT - 1))  # wrong field count
    zp = _make_zip(tmp_path, ts, [bad])
    feat, errors = bf.aggregate_slot(ts, zp, lag=15, downloaded=1)
    assert feat["status"] == "failed_schema"
    assert feat["schema_errors"] == 1
    assert feat["source_coverage"] == 0.0
    assert errors and errors[0]["error_type"] == "schema_error"


def test_aggregate_slot_bad_tone_is_schema_error(tmp_path):
    ts = "20240501000000"
    fields = [""] * probe.GKG_FIELD_COUNT
    fields[probe.F_URL] = "https://news.x/ethereum"
    fields[probe.F_TONE] = "not,a,number"  # not a 7-tuple
    zp = _make_zip(tmp_path, ts, ["\t".join(fields)])
    feat, _ = bf.aggregate_slot(ts, zp, lag=15, downloaded=1)
    assert feat["status"] == "failed_schema"
    assert feat["schema_errors"] == 1


def test_empty_feature_has_all_columns():
    feat = bf._empty_feature("20240501000000", lag=15, status="ok", downloaded=0)
    assert set(feat) == set(bf._FEATURE_COLUMNS)
    for col in bf._FEATURE_COLUMNS:
        if col.endswith("_tone_mean") or col.endswith("_tone_vol"):
            assert feat[col] is None


# --------------------------------------------------------------------------- #
# slot subsampling + arg validation
# --------------------------------------------------------------------------- #
def test_generate_slots_and_stride():
    start = datetime(2024, 5, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 5, 1, 23, 45, tzinfo=timezone.utc)
    slots = bf.generate_slots(start, end)
    assert len(slots) == 96  # 24h * 4
    two_hourly = slots[::8]
    assert len(two_hourly) == 12
    assert two_hourly[0] == "20240501000000"
    assert two_hourly[1] == "20240501020000"
    assert two_hourly[-1] == "20240501220000"


def test_validate_args_rejects_bad_stride():
    args = bf._build_parser().parse_args(
        [
            "--start", "20240501000000",
            "--end", "20240501234500",
            "--output-db", "x.sqlite",
            "--max-download-mb", "10",
            "--slot-stride", "0",
        ]
    )
    assert bf._validate_args(args) == 2


def test_validate_args_accepts_valid():
    args = bf._build_parser().parse_args(
        [
            "--start", "20240501000000",
            "--end", "20240501234500",
            "--output-db", "x.sqlite",
            "--max-download-mb", "10",
            "--slot-stride", "8",
        ]
    )
    assert bf._validate_args(args) is None


# --------------------------------------------------------------------------- #
# SQLite schema round-trip
# --------------------------------------------------------------------------- #
def test_init_db_and_write_slot_roundtrip(tmp_path):
    db = tmp_path / "feat.sqlite"
    conn = bf.init_db(db)
    try:
        feat = bf._empty_feature("20240501000000", lag=15, status="ok", downloaded=10)
        feat["btc_count"] = 3
        feat["btc_tone_mean"] = 1.25
        bf.write_slot(conn, feat, [])
        cur = conn.execute(
            "SELECT btc_count, btc_tone_mean FROM gkg_slot_features WHERE observed_utc = ?",
            (feat["observed_utc"],),
        )
        row = cur.fetchone()
        assert row == (3, 1.25)
    finally:
        conn.close()
