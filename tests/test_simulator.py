import numpy as np
import pandas as pd

from src import rf_model as rfm
from src import simulator as sim
from src.integral_cache import build_cache_for_symbol
from src.parallel import parallel_map
from tests.synthetic import make_minute_df, make_symbol_entry


def _cache():
    return build_cache_for_symbol("BTCUSDT", make_symbol_entry(seed=42), window=30)


def test_intraday_liquidation_counted_even_when_close_is_safe():
    cache = _cache()
    # Open at 100 -> long liq price = 90 (10x leverage). Candle low pierces 90
    # while the close (100) looks safe; this must still count as a liquidation.
    prices = [100.0, (100.0, 101.0, 89.0), 100.0, 100.0, 100.0]
    minute_df = make_minute_df(prices)
    account = sim.AccountConfig(initial_balance=10000.0, leverage=10.0, notional_per_leg=1000.0)

    result = sim.simulate_symbol(minute_df, cache, rfm.BaselineModel(), "BTCUSDT", account)
    assert not result.empty
    assert int(result["liquidation_count"].sum()) >= 1


def test_simulation_is_deterministic():
    cache = _cache()
    prices = [100 + 0.5 * np.sin(i / 5.0) for i in range(60)]
    minute_df = make_minute_df([(p, p + 0.3, p - 0.3) for p in prices])
    account = sim.AccountConfig()

    r1 = sim.simulate_symbol(minute_df, cache, rfm.BaselineModel(), "BTCUSDT", account)
    r2 = sim.simulate_symbol(minute_df, cache, rfm.BaselineModel(), "BTCUSDT", account)
    assert r1.equals(r2)


def test_parallel_map_matches_sequential():
    items = list(range(20))
    seq = [x * x for x in items]
    par = parallel_map(lambda x: x * x, items, max_workers=4)
    assert par == seq


def test_no_lookahead_future_candles_do_not_change_features():
    cache = _cache()
    base_prices = [100 + 0.2 * np.sin(i / 4.0) for i in range(14)]
    extra_prices = [100 + 0.2 * np.sin(i / 4.0) for i in range(14, 24)]

    df_short = make_minute_df([(p, p + 0.3, p - 0.3) for p in base_prices])
    df_long = make_minute_df([(p, p + 0.3, p - 0.3) for p in base_prices + extra_prices])

    account = sim.AccountConfig()
    rows_short = sim.generate_training_rows(df_short, cache, "BTCUSDT", account)
    rows_long = sim.generate_training_rows(df_long, cache, "BTCUSDT", account)

    exclude = {"symbol", "timestamp", "pnl_if_continue", "pnl_if_close", "target_policy_improvement", "mode", "side"}

    def by_ts(rows):
        return {str(r["timestamp"]): r for r in rows}

    short_map = by_ts(rows_short)
    long_map = by_ts(rows_long)
    shared = set(short_map) & set(long_map)
    assert shared  # there must be overlapping decision points

    for ts in shared:
        a, b = short_map[ts], long_map[ts]
        for key in a:
            if key in exclude:
                continue
            assert np.isclose(float(a[key]), float(b[key])), f"feature {key} changed with future data"


def test_daily_results_have_expected_columns():
    cache = _cache()
    minute_df = make_minute_df([(100.0, 100.3, 99.7)] * 30)
    account = sim.AccountConfig()
    result = sim.simulate_symbol(minute_df, cache, rfm.BaselineModel(), "BTCUSDT", account)
    for col in ["date", "symbol", "daily_pnl", "liquidation_count", "close_count", "continue_count"]:
        assert col in result.columns
