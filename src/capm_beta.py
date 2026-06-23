"""CAPM-style rolling Beta for each coin, computed per day (leakage-safe).

Definition (per the project spec):

* ``R_coin(t)``   = coin value 30-day change      = ``P(t) / P(t-L) - 1``
* ``R_market(t)`` = crypto market value 30-day change = ``M(t) / M(t-L) - 1``
* ``beta(t)``     = Cov(R_coin, R_market) / Var(R_market)

where the covariance/variance are taken over the trailing ``window`` days of the
daily 30-day-return series (default ``L = 30`` day return lag, ``window = 90``).

``beta(t)`` only uses information available up to and including day ``t``; the
training/live join always reads the *previous completed day* (D-1), so there is
no same-day look-ahead.

This module is intentionally source-agnostic: it operates on aligned daily value
series and does not care whether the market value comes from a paid global
market-cap feed or a free top-N basket proxy.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_RETURN_LAG_DAYS = 30
DEFAULT_BETA_WINDOW_DAYS = 90
# Minimum lead-in of daily values required before the first computable beta.
# Needs ``window`` daily returns, each of which needs ``lag`` prior days.
def required_leadin_days(
    return_lag: int = DEFAULT_RETURN_LAG_DAYS, window: int = DEFAULT_BETA_WINDOW_DAYS
) -> int:
    return int(return_lag) + int(window)


def lagged_returns(values: Sequence[float], lag: int) -> List[Optional[float]]:
    """``v[t] / v[t-lag] - 1`` aligned to ``values`` (first ``lag`` entries None)."""
    n = len(values)
    out: List[Optional[float]] = [None] * n
    for t in range(lag, n):
        prev = values[t - lag]
        cur = values[t]
        if prev is None or cur is None:
            continue
        try:
            prev_f = float(prev)
            cur_f = float(cur)
        except (TypeError, ValueError):
            continue
        if prev_f == 0.0 or not math.isfinite(prev_f) or not math.isfinite(cur_f):
            continue
        out[t] = cur_f / prev_f - 1.0
    return out


def _beta_from_pairs(
    rc: Sequence[Optional[float]], rm: Sequence[Optional[float]]
) -> Optional[float]:
    """Cov(rc, rm) / Var(rm) over the paired, finite samples; None if undefined."""
    xs: List[float] = []
    ys: List[float] = []
    for a, b in zip(rc, rm):
        if a is None or b is None:
            continue
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        xs.append(float(a))
        ys.append(float(b))
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / n
    var_y = sum((y - mean_y) ** 2 for y in ys) / n
    if var_y <= 0.0 or not math.isfinite(var_y):
        return None
    beta = cov / var_y
    if not math.isfinite(beta):
        return None
    return beta


def rolling_beta_series(
    coin_values: Sequence[float],
    market_values: Sequence[float],
    return_lag: int = DEFAULT_RETURN_LAG_DAYS,
    window: int = DEFAULT_BETA_WINDOW_DAYS,
) -> List[Optional[float]]:
    """Per-index rolling beta aligned to the input series (None where undefined).

    ``coin_values`` and ``market_values`` must be the SAME length and aligned by
    calendar day (index ``t`` = same day in both).
    """
    if len(coin_values) != len(market_values):
        raise ValueError("coin_values and market_values must be equal length")
    rc = lagged_returns(coin_values, return_lag)
    rm = lagged_returns(market_values, return_lag)
    n = len(coin_values)
    out: List[Optional[float]] = [None] * n
    for t in range(n):
        lo = t - window + 1
        if lo < 0:
            continue
        out[t] = _beta_from_pairs(rc[lo : t + 1], rm[lo : t + 1])
    return out


def rolling_beta_by_date(
    dates: Sequence[str],
    coin_values: Sequence[float],
    market_values: Sequence[float],
    return_lag: int = DEFAULT_RETURN_LAG_DAYS,
    window: int = DEFAULT_BETA_WINDOW_DAYS,
) -> List[Tuple[str, float]]:
    """``[(date, beta), ...]`` for every date where beta is defined.

    ``dates`` must be ascending, one entry per calendar day, aligned with the two
    value series.
    """
    if not (len(dates) == len(coin_values) == len(market_values)):
        raise ValueError("dates, coin_values, market_values must be equal length")
    betas = rolling_beta_series(coin_values, market_values, return_lag, window)
    return [(dates[t], float(b)) for t, b in enumerate(betas) if b is not None]


__all__ = [
    "DEFAULT_RETURN_LAG_DAYS",
    "DEFAULT_BETA_WINDOW_DAYS",
    "required_leadin_days",
    "lagged_returns",
    "rolling_beta_series",
    "rolling_beta_by_date",
]
