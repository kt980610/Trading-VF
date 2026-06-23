"""Rolling-window features over daily returns."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _ddof_for_mode(mode: str) -> int:
    return 1 if str(mode).lower() == "sample" else 0


def compute_daily_returns(close: pd.Series) -> pd.Series:
    close = pd.Series(close, dtype="float64")
    returns = close / close.shift(1) - 1.0
    return returns.iloc[1:]


def rolling_mean(returns: pd.Series, window: int) -> pd.Series:
    returns = pd.Series(returns, dtype="float64")
    return returns.rolling(window=window, min_periods=window).mean()


def rolling_variance(returns: pd.Series, window: int, mode: str = "population") -> pd.Series:
    returns = pd.Series(returns, dtype="float64")
    ddof = _ddof_for_mode(mode)
    return returns.rolling(window=window, min_periods=window).var(ddof=ddof)


def rolling_mean_of_mean(rolling_mean_series: pd.Series, window: int) -> pd.Series:
    series = pd.Series(rolling_mean_series, dtype="float64").dropna()
    return series.rolling(window=window, min_periods=window).mean()


def rolling_var_of_mean(rolling_mean_series: pd.Series, window: int, mode: str = "population") -> pd.Series:
    series = pd.Series(rolling_mean_series, dtype="float64").dropna()
    ddof = _ddof_for_mode(mode)
    return series.rolling(window=window, min_periods=window).var(ddof=ddof)


def compute_rolling_state(
    returns: pd.Series,
    rolling_mean_series: pd.Series,
    window: int,
    mode: str = "population",
) -> Optional[dict]:
    returns = pd.Series(returns, dtype="float64").dropna()
    means = pd.Series(rolling_mean_series, dtype="float64").dropna()

    if len(returns) < window or len(means) < window:
        return None

    ddof = _ddof_for_mode(mode)
    last_returns = returns.to_numpy()[-window:]
    last_means = means.to_numpy()[-window:]

    return {
        "Last30DaysMean": float(np.mean(last_returns)),
        "Last30DaysVar": float(np.var(last_returns, ddof=ddof)),
        "Last30DaysMeanOfMean": float(np.mean(last_means)),
        "Last30DaysVarOfMean": float(np.var(last_means, ddof=ddof)),
        "ReturnIn30DaysBefore": float(last_returns[0]),
        "MeanIn30DaysBefore": float(last_means[0]),
    }
