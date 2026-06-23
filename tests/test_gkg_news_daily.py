"""Tests for the GKG SQLite -> daily news features producer (src.gkg_news_daily)."""

from __future__ import annotations

import json
import sqlite3

import pytest

from src import gkg_news_daily as gnd
from src import rf_dataset as rfd

import pandas as pd

# All columns aggregate_daily selects from gkg_slot_features.
_GROUP_COLS = [
    col for g in gnd._GROUP_SOURCE.values() for col in g
]
_ALL_COLS = ["observed_utc", "status"] + _GROUP_COLS


def _make_db(path, rows):
    conn = sqlite3.connect(str(path))
    decls = []
    for c in _ALL_COLS:
        if c in ("observed_utc", "status"):
            decls.append(f"{c} TEXT")
        elif c.endswith("_tone_mean") or c.endswith("_tone_vol"):
            decls.append(f"{c} REAL")
        else:
            decls.append(f"{c} INTEGER")
    conn.execute(f"CREATE TABLE gkg_slot_features ({', '.join(decls)})")
    for r in rows:
        vals = [r.get(c) for c in _ALL_COLS]
        ph = ", ".join("?" for _ in _ALL_COLS)
        conn.execute(
            f"INSERT INTO gkg_slot_features ({', '.join(_ALL_COLS)}) VALUES ({ph})", vals
        )
    conn.commit()
    return conn


def _slot(observed, status="ok", **kw):
    r = {c: 0 for c in _ALL_COLS}
    r["observed_utc"] = observed
    r["status"] = status
    for c in _GROUP_COLS:
        if c.endswith("_tone_mean") or c.endswith("_tone_vol"):
            r[c] = None
    r.update(kw)
    return r


def test_pooled_aggregation_and_coverage(tmp_path):
    rows = [
        _slot("2024-05-01T00:00:00Z", btc_count=2, btc_tone_mean=1.0, btc_tone_vol=0.0),
        _slot("2024-05-01T02:00:00Z", btc_count=2, btc_tone_mean=3.0, btc_tone_vol=0.0),
        _slot("2024-05-01T04:00:00Z", status="failed"),  # counts only in coverage denom
    ]
    conn = _make_db(tmp_path / "g.sqlite", rows)
    try:
        daily = gnd.aggregate_daily(conn)
    finally:
        conn.close()

    assert len(daily) == 1
    d = daily[0]
    assert d["date"] == "2024-05-01"
    # Pooled: N=4, mean=2.0, var = (2*1 + 2*9)/4 - 4 = 1 -> vol 1.0
    assert d["gkg_btc_count"] == 4
    assert d["gkg_btc_tone_mean"] == pytest.approx(2.0)
    assert d["gkg_btc_tone_vol"] == pytest.approx(1.0)
    # 2 ok of 3 present slots.
    assert d["gkg_coverage"] == pytest.approx(2 / 3)


def test_groups_and_dates(tmp_path):
    rows = [
        _slot("2024-05-01T00:00:00Z", altcoins_count=1, altcoins_tone_mean=-2.0,
              macro_conflict_count=5, macro_conflict_tone_mean=0.5),
        _slot("2024-05-02T00:00:00Z", macro_rates_count=3, macro_rates_tone_mean=1.5),
    ]
    conn = _make_db(tmp_path / "g.sqlite", rows)
    try:
        daily = gnd.aggregate_daily(conn)
    finally:
        conn.close()
    by_date = {d["date"]: d for d in daily}
    assert set(by_date) == {"2024-05-01", "2024-05-02"}
    assert by_date["2024-05-01"]["gkg_altcoins_count"] == 1
    assert by_date["2024-05-01"]["gkg_altcoins_tone_mean"] == pytest.approx(-2.0)
    assert by_date["2024-05-01"]["gkg_macro_conflict_count"] == 5
    assert by_date["2024-05-02"]["gkg_macro_rates_count"] == 3
    # Empty group on a day -> zero count, zero tone (not None).
    assert by_date["2024-05-02"]["gkg_btc_count"] == 0
    assert by_date["2024-05-02"]["gkg_btc_tone_mean"] == 0.0


def test_run_writes_jsonl_without_symbol(tmp_path):
    db = tmp_path / "g.sqlite"
    out = tmp_path / "news_features_daily.jsonl"
    conn = _make_db(db, [_slot("2024-05-01T00:00:00Z", btc_count=1, btc_tone_mean=0.3)])
    conn.close()

    gnd.run(config_path="config/distribution_config.yaml",
            gkg_db=str(db), output=str(out), min_coverage=0.0)

    lines = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    assert len(lines) == 1
    rec = lines[0]
    assert rec["date"] == "2024-05-01"
    assert "symbol" not in rec  # market-wide row; training accepts symbol=None
    assert rec["gkg_btc_count"] == 1
    assert any(k.startswith("gkg_") for k in rec)


def test_emitted_columns_are_discovered_as_news_features():
    # The producer's gkg_ columns must be picked up by the df-driven discovery so
    # they enter the RF feature schema (training + Rust live consume the same set).
    df = pd.DataFrame(
        [{"gkg_btc_count": 1, "gkg_macro_conflict_tone_mean": 0.2, "gkg_coverage": 1.0,
          "LongEdge_Return": 0.0}]
    )
    news_cols = set(rfd.news_like_columns(df))
    assert {"gkg_btc_count", "gkg_macro_conflict_tone_mean", "gkg_coverage"} <= news_cols
    assert "LongEdge_Return" not in news_cols


def test_min_coverage_filters_days(tmp_path):
    db = tmp_path / "g.sqlite"
    out = tmp_path / "out.jsonl"
    rows = [
        _slot("2024-05-01T00:00:00Z", btc_count=1, btc_tone_mean=0.0),  # coverage 1.0
        _slot("2024-05-02T00:00:00Z", status="failed"),  # coverage 0.0
    ]
    conn = _make_db(db, rows)
    conn.close()
    gnd.run("config/distribution_config.yaml", str(db), str(out), min_coverage=0.5)
    days = {json.loads(x)["date"] for x in out.read_text().splitlines() if x.strip()}
    assert days == {"2024-05-01"}
