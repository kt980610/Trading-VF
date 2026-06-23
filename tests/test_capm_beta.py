"""Tests for the source-agnostic CAPM rolling-beta module."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import capm_beta as cb  # noqa: E402


def test_lagged_returns_basic():
    vals = [10.0, 11.0, 12.0, 13.2]
    out = cb.lagged_returns(vals, lag=1)
    assert out[0] is None
    assert abs(out[1] - 0.1) < 1e-12
    assert abs(out[2] - (12.0 / 11.0 - 1.0)) < 1e-12
    assert abs(out[3] - 0.10) < 1e-12


def test_lagged_returns_guards_zero_and_none():
    vals = [0.0, 5.0, None, 8.0]
    out = cb.lagged_returns(vals, lag=1)
    # index1: prev=0 -> undefined; index2: cur None -> undefined; index3: prev None
    assert out == [None, None, None, None]


def test_beta_of_market_against_itself_is_one():
    # If coin == market, beta must be exactly 1.
    market = [100.0 * (1.03 ** i) for i in range(200)]
    betas = cb.rolling_beta_series(market, market, return_lag=30, window=90)
    last = betas[-1]
    assert last is not None
    assert abs(last - 1.0) < 1e-9


def test_beta_scales_with_amplitude():
    # coin returns = 2x market returns -> beta ~ 2.
    import math

    market = []
    coin = []
    m = 100.0
    c = 100.0
    for i in range(220):
        # market daily multiplicative wiggle; coin moves twice as hard in log space
        r = 0.01 * math.sin(i / 3.0)
        market.append(m)
        coin.append(c)
        m *= (1.0 + r)
        c *= (1.0 + 2.0 * r)
    betas = cb.rolling_beta_series(coin, market, return_lag=30, window=90)
    last = betas[-1]
    assert last is not None
    # 30-day compounded returns are ~ linear in r for small r, so beta ~ 2.
    assert 1.7 < last < 2.3


def test_insufficient_history_returns_none():
    vals = [float(i + 1) for i in range(50)]
    betas = cb.rolling_beta_series(vals, vals, return_lag=30, window=90)
    # Need lag(30)+window(90)-1 = 119 index before first beta; 50 points -> all None.
    assert all(b is None for b in betas)


def test_zero_variance_market_is_undefined():
    coin = [float(i + 1) for i in range(200)]
    flat = [100.0] * 200  # market never moves -> Var(R_market)=0 -> beta undefined
    betas = cb.rolling_beta_series(coin, flat, return_lag=30, window=90)
    assert all(b is None for b in betas)


def test_required_leadin():
    assert cb.required_leadin_days(30, 90) == 120
    assert cb.required_leadin_days(30, 60) == 90


def test_rolling_beta_by_date_alignment():
    dates = [f"2020-{m:02d}-{d:02d}" for m in range(1, 9) for d in range(1, 29)]
    n = len(dates)
    market = [100.0 * (1.01 ** i) for i in range(n)]
    coin = [100.0 * (1.01 ** i) for i in range(n)]
    pairs = cb.rolling_beta_by_date(dates, coin, market, return_lag=30, window=90)
    assert pairs, "expected some computable betas"
    # Each returned date must be one of the inputs and betas ~ 1 (coin==market).
    in_dates = set(dates)
    for d, b in pairs:
        assert d in in_dates
        assert abs(b - 1.0) < 1e-9


if __name__ == "__main__":  # pragma: no cover
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
