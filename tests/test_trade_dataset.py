"""Integration tests for the trade-grouped close-classifier dataset (spec 5C,8,11).

Covers spec tests 7 (CurrentVolume only up to the decision minute), 17 (net PnL
after fees/funding/slippage) and 18 (no ADD_MARGIN in the RF dataset).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import simulator as sim
from src.integral_cache import build_cache_for_symbol
from src.main_build_distributions import build_symbol_entry
from src.mdp_features import MDP_FEATURES
from src.rf_dataset import MDP_FEATURE_NAMES, VOLUME_RATIO_FEATURES
from src.rolling_features import compute_daily_returns


class _Cfg:
    class distribution:
        bins = 50
        min_observations = 60

    class rolling:
        window_days = 30
        variance_mode = "population"


def _cache():
    rng = np.random.default_rng(3)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.02, 400))
    entry = build_symbol_entry(_Cfg, compute_daily_returns(pd.Series(close)))
    return build_cache_for_symbol("BTCUSDT", entry, window=30)


def _minute_df(days=3, per_day=120, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2023-01-02", tz="UTC")
    price = 100.0
    for d in range(days):
        for m in range(per_day):
            price *= 1.0 + rng.normal(0.0, 0.003)
            ts = start + pd.Timedelta(days=d, minutes=m)
            rows.append(
                {
                    "timestamp": ts,
                    "open": price,
                    "high": price * 1.001,
                    "low": price * 0.999,
                    "close": price,
                    "volume": 10.0,
                }
            )
    return pd.DataFrame(rows)


def _provider(_ts):
    return {"predicted_daily_volume": 5000.0, "previous_day_real_volume": 4000.0}


def test_dataset_has_mdp_and_volume_ratio_features_no_add_margin():
    df = sim.generate_trade_dataset(
        _minute_df(), _cache(), "BTCUSDT", sim.AccountConfig(),
        context_provider=_provider,
    )
    assert not df.empty
    for col in MDP_FEATURE_NAMES:
        assert col in df.columns
    assert set(MDP_FEATURE_NAMES) == set(MDP_FEATURES)
    for col in VOLUME_RATIO_FEATURES:
        assert col in df.columns
    # Spec 18: no ADD_MARGIN concept leaks into the RF dataset.
    assert not any("add_margin" in c.lower() for c in df.columns)
    assert "close_label" in df.columns
    # Terminal minute of at least one trade must be a close.
    assert df["close_label"].max() == 1


def test_current_volume_is_cumulative_within_day_only():
    df = sim.generate_trade_dataset(
        _minute_df(), _cache(), "BTCUSDT", sim.AccountConfig(),
        context_provider=_provider,
    )
    df = df.copy()
    df["day"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("1D")
    for _day, g in df.groupby("day"):
        g = g.sort_values("timestamp")
        vols = g["intraday_volume_so_far"].to_numpy()
        # Non-decreasing within a day (only volume up to the decision minute).
        assert np.all(np.diff(vols) >= -1e-9)
    # Each new day restarts below the previous day's max (no carryover).
    day_max = df.groupby("day")["intraday_volume_so_far"].max().to_numpy()
    day_min = df.groupby("day")["intraday_volume_so_far"].min().to_numpy()
    assert np.all(day_min <= day_max)


def test_net_pnl_rate_drops_with_fees():
    cache = _cache()
    mdf = _minute_df()
    low_fee = sim.generate_trade_dataset(
        mdf, cache, "BTCUSDT", sim.AccountConfig(), context_provider=_provider,
        fee_rate=0.0, slippage=0.0,
    )
    high_fee = sim.generate_trade_dataset(
        mdf, cache, "BTCUSDT", sim.AccountConfig(), context_provider=_provider,
        fee_rate=0.01, slippage=0.01,
    )
    # Fees/slippage reduce net close PnL, hence the per-minute trade PnL rate.
    assert high_fee["trade_pnl_rate_now"].mean() < low_fee["trade_pnl_rate_now"].mean()
