from datetime import date, timedelta

import pandas as pd

from src.portfolio import hybrid_matrix as hm


def _frame(rows):
    return pd.DataFrame(rows, columns=["date", "symbol", "return", "source"])


def test_realized_overrides_simulated():
    d = date(2024, 9, 28)
    sim = _frame([[d, "BTCUSDT", 0.10, "simulated"]])
    real = _frame([[d, "BTCUSDT", 0.50, "live_realized"]])
    hybrid = hm.build_hybrid_returns(sim, real)
    row = hybrid[(hybrid["date"] == d) & (hybrid["symbol"] == "BTCUSDT")]
    assert len(row) == 1
    assert row["return"].iloc[0] == 0.50
    assert row["source"].iloc[0] == "live_realized"


def test_simulated_used_when_no_realized():
    d = date(2024, 9, 28)
    sim = _frame([[d, "ETHUSDT", 0.20, "simulated"]])
    real = _frame([])
    hybrid = hm.build_hybrid_returns(sim, real)
    assert hybrid["return"].iloc[0] == 0.20


def _series(symbol, start_offset, n_days, as_of, value=0.01):
    rows = []
    for i in range(n_days):
        rows.append([as_of - timedelta(days=start_offset + i), symbol, value, "simulated"])
    return rows


def test_min_required_days_marks_symbol_invalid():
    as_of = date(2024, 9, 29)
    rows = _series("BTCUSDT", 1, 26, as_of) + _series("ETHUSDT", 1, 10, as_of)
    hybrid = _frame(rows)
    rm = hm.build_return_matrix(hybrid, ["BTCUSDT", "ETHUSDT"], as_of, lookback_days=30, min_required_days=25)
    assert "BTCUSDT" in rm.valid_symbols
    assert "ETHUSDT" not in rm.valid_symbols
    assert rm.invalid["ETHUSDT"] == hm.REASON_NOT_ENOUGH_DAYS


def test_window_excludes_as_of_and_older_than_lookback():
    as_of = date(2024, 9, 29)
    rows = [
        [as_of, "BTCUSDT", 0.5, "simulated"],            # as_of day -> excluded
        [as_of - timedelta(days=60), "BTCUSDT", 0.9, "simulated"],  # too old -> excluded
    ] + _series("BTCUSDT", 1, 25, as_of)                 # 25 valid in-window days
    hybrid = _frame(rows)
    rm = hm.build_return_matrix(hybrid, ["BTCUSDT"], as_of, lookback_days=30, min_required_days=25)
    assert rm.counts["BTCUSDT"] == 25
    assert rm.window_end == as_of - timedelta(days=1)


def test_missing_symbol_is_invalid():
    as_of = date(2024, 9, 29)
    hybrid = _frame(_series("BTCUSDT", 1, 26, as_of))
    rm = hm.build_return_matrix(hybrid, ["BTCUSDT", "SOLUSDT"], as_of, lookback_days=30, min_required_days=25)
    assert rm.counts["SOLUSDT"] == 0
    assert rm.invalid["SOLUSDT"] == hm.REASON_NOT_ENOUGH_DAYS
