"""Tests for the overlapping (one-entry-per-day) episode generator."""

import pandas as pd

from src import simulator as sim
from src.integral_cache import build_cache_for_symbol
from tests.synthetic import make_minute_df, make_symbol_entry


def _cache():
    return build_cache_for_symbol("BTCUSDT", make_symbol_entry(seed=1), window=30)


def _flat_df(n, start="2021-01-01"):
    # Flat prices -> no liquidation, so each episode runs to its boundary.
    return make_minute_df([(100.0, 100.2, 99.8)] * n, start=start)


def test_overlapping_yields_one_group_per_entry():
    cache = _cache()
    mdf = _flat_df(120)
    account = sim.AccountConfig()
    out = sim.generate_overlapping_trade_dataset(
        mdf, cache, "BTCUSDT", account, [0, 30, 60]
    )
    assert not out.empty
    assert out["trade_id"].nunique() == 3
    assert set(out["symbol"]) == {"BTCUSDT"}


def test_overlapping_far_more_groups_than_sequential():
    cache = _cache()
    mdf = _flat_df(240)
    account = sim.AccountConfig()
    seq = sim.generate_trade_dataset(mdf, cache, "BTCUSDT", account)
    entries = list(range(0, 240, 10))  # 24 entries
    over = sim.generate_overlapping_trade_dataset(
        mdf, cache, "BTCUSDT", account, entries
    )
    # Sequential holds one hedge open the whole flat series -> a single group;
    # overlapping yields one group per entry.
    assert seq["trade_id"].nunique() <= 2
    assert over["trade_id"].nunique() == len(entries)


def test_overlapping_never_crosses_data_gap():
    cache = _cache()
    seg1 = pd.date_range("2021-01-01", periods=100, freq="1min", tz="UTC")
    seg2 = pd.date_range("2021-02-01", periods=100, freq="1min", tz="UTC")  # ~31d gap
    rows = [
        {"timestamp": t, "open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0, "volume": 1.0}
        for t in list(seg1) + list(seg2)
    ]
    mdf = pd.DataFrame(rows)
    account = sim.AccountConfig()
    out = sim.generate_overlapping_trade_dataset(mdf, cache, "BTCUSDT", account, [0])
    # The episode entered in segment 1 must not reach segment 2 timestamps.
    assert out["timestamp"].max() < pd.Timestamp("2021-01-02", tz="UTC")


def test_overlapping_respects_max_end_index():
    cache = _cache()
    mdf = _flat_df(200)
    account = sim.AccountConfig()
    out = sim.generate_overlapping_trade_dataset(
        mdf, cache, "BTCUSDT", account, [0], max_end_index=50
    )
    assert out["trade_id"].nunique() == 1
    assert len(out) <= 50


def test_overlapping_respects_max_hold_minutes():
    cache = _cache()
    mdf = _flat_df(2000)  # flat -> no liquidation, would otherwise run to the end
    account = sim.AccountConfig()
    out = sim.generate_overlapping_trade_dataset(
        mdf, cache, "BTCUSDT", account, [0], max_hold_minutes=60
    )
    assert out["trade_id"].nunique() == 1
    # Entry row excluded; episode holds at most max_hold_minutes rows.
    assert len(out) <= 60


def test_spill_matches_in_memory(tmp_path):
    cache = _cache()
    mdf = _flat_df(300)
    account = sim.AccountConfig()
    entries = [0, 50, 100, 150]
    in_mem = sim.generate_overlapping_trade_dataset(
        mdf, cache, "BTCUSDT", account, entries
    )
    spilled = sim.generate_overlapping_trade_dataset(
        mdf, cache, "BTCUSDT", account, entries,
        spill_path=str(tmp_path / "btc_w0"), flush_rows=37,  # tiny -> many flushes
    )
    # Same shape and trade groups; spilled is float32 but values match.
    assert len(spilled) == len(in_mem)
    assert spilled["trade_id"].nunique() == in_mem["trade_id"].nunique()
    common = ["trade_id", "close_label", "net_close_pnl_now"]
    import numpy as np
    for c in common:
        assert np.allclose(
            spilled[c].to_numpy(dtype="float64"),
            in_mem[c].to_numpy(dtype="float64"),
            equal_nan=True,
        )
    # No leftover part files (read_back cleans them up).
    assert not list(tmp_path.glob("*.part*"))


def test_segment_bounds_single_block():
    mdf = _flat_df(50)
    assert sim._contiguous_segment_bounds(mdf) == [0, 50]
