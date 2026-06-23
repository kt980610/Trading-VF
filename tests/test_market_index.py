"""Unit tests for the local cap-weighted market index builder (no network)."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import build_market_index as bmi  # noqa: E402


def _series(pairs):
    idx = pd.to_datetime([d for d, _ in pairs])
    return pd.Series([v for _, v in pairs], index=idx)


def test_build_index_sums_overlapping_dates():
    a = _series([("2022-06-01", 100.0), ("2022-06-02", 110.0)])
    b = _series([("2022-06-01", 10.0), ("2022-06-02", 20.0)])
    rows = bmi.build_index({"A": a, "B": b})
    assert rows == [
        ("2022-06-01", 110.0, 2),
        ("2022-06-02", 130.0, 2),
    ]


def test_build_index_counts_only_present_constituents():
    a = _series([("2018-08-01", 50.0), ("2018-08-02", 60.0)])
    b = _series([("2018-08-02", 5.0)])  # missing the first day
    rows = bmi.build_index({"A": a, "B": b})
    assert rows[0] == ("2018-08-01", 50.0, 1)
    assert rows[1] == ("2018-08-02", 65.0, 2)


def test_build_index_skips_nonpositive_and_empty():
    assert bmi.build_index({}) == []
    z = _series([("2026-04-21", 0.0)])
    assert bmi.build_index({"Z": z}) == []


def test_parse_symbol_ids_roundtrip():
    out = bmi._parse_symbol_ids("NEOUSDT=neo, LINKUSDT=chainlink")
    assert out == {"NEOUSDT": "neo", "LINKUSDT": "chainlink"}
    assert bmi._parse_symbol_ids("") == {}


def test_parse_symbol_ids_rejects_bad_entry():
    with pytest.raises(SystemExit):
        bmi._parse_symbol_ids("BADENTRY")
