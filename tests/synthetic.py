"""Synthetic fixtures shared across tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.distributions import build_histogram_pdf
from src.rolling_features import (
    compute_rolling_state,
    rolling_mean,
    rolling_mean_of_mean,
    rolling_var_of_mean,
    rolling_variance,
)


def make_symbol_entry(seed: int = 0, n: int = 600, bins: int = 100, scale: float = 0.02) -> dict:
    """Build a valid snapshot symbol entry with all five distributions."""
    rng = np.random.default_rng(seed)
    returns = pd.Series(rng.normal(0.0, scale, n))
    window = 30

    rmean = rolling_mean(returns, window)
    rvar = rolling_variance(returns, window, "population")
    rmom = rolling_mean_of_mean(rmean, window)
    rvom = rolling_var_of_mean(rmean, window, "population")

    return {
        "valid": True,
        "n_daily_returns": int(len(returns)),
        "return_distribution": build_histogram_pdf(returns.to_numpy(), bins),
        "mean_distribution": build_histogram_pdf(rmean.dropna().to_numpy(), bins),
        "variance_distribution": build_histogram_pdf(rvar.dropna().to_numpy(), bins),
        "mean_of_mean_distribution": build_histogram_pdf(rmom.dropna().to_numpy(), bins),
        "var_of_mean_distribution": build_histogram_pdf(rvom.dropna().to_numpy(), bins),
        "rolling_state": compute_rolling_state(returns, rmean, window, "population"),
    }


def make_minute_df(prices, start="2021-01-01", freq="1min") -> pd.DataFrame:
    """OHLCV frame from a list of (close, high, low) or scalar closes."""
    rows = []
    ts = pd.date_range(start=start, periods=len(prices), freq=freq, tz="UTC")
    for t, p in zip(ts, prices):
        if isinstance(p, (tuple, list)):
            close, high, low = p
        else:
            close = high = low = float(p)
        rows.append(
            {"timestamp": t, "open": close, "high": high, "low": low, "close": close, "volume": 1.0}
        )
    return pd.DataFrame(rows)
