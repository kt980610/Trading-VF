import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rolling_features import (
    compute_daily_returns,
    compute_rolling_state,
    rolling_mean,
    rolling_mean_of_mean,
    rolling_var_of_mean,
    rolling_variance,
)


def test_daily_returns_known_series():
    close = pd.Series([100.0, 110.0, 99.0])
    returns = compute_daily_returns(close).to_numpy()
    expected = np.array([0.1, 99.0 / 110.0 - 1.0])
    assert np.allclose(returns, expected)
    assert len(returns) == 2


def test_rolling_mean_window_3():
    returns = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    rm = rolling_mean(returns, 3).to_numpy()
    assert np.isnan(rm[0]) and np.isnan(rm[1])
    assert rm[2] == pytest.approx(2.0)
    assert rm[3] == pytest.approx(3.0)
    assert rm[4] == pytest.approx(4.0)


def test_rolling_variance_population():
    returns = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    rv = rolling_variance(returns, 3, "population").to_numpy()
    window = np.array([3.0, 4.0, 5.0])
    expected = np.mean((window - window.mean()) ** 2)
    assert rv[4] == pytest.approx(expected)
    assert rv[4] == pytest.approx(2.0 / 3.0)


def test_rolling_variance_sample():
    returns = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    rv = rolling_variance(returns, 3, "sample").to_numpy()
    window = np.array([3.0, 4.0, 5.0])
    expected = np.var(window, ddof=1)
    assert rv[4] == pytest.approx(expected)


def test_rolling_mean_of_mean():
    returns = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    rm = rolling_mean(returns, 3)
    rmom = rolling_mean_of_mean(rm, 3).dropna().to_numpy()
    means = rm.dropna().to_numpy()
    expected_first = np.mean(means[0:3])
    assert rmom[0] == pytest.approx(expected_first)


def test_rolling_var_of_mean():
    returns = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    rm = rolling_mean(returns, 3)
    rvom = rolling_var_of_mean(rm, 3, "population").dropna().to_numpy()
    means = rm.dropna().to_numpy()
    expected_first = np.var(means[0:3], ddof=0)
    assert rvom[0] == pytest.approx(expected_first)


def test_return_in_30_days_before_selects_oldest():
    returns = pd.Series([0.01 * i for i in range(1, 11)])
    window = 3
    rm = rolling_mean(returns, window)
    state = compute_rolling_state(returns, rm, window, "population")
    expected_oldest = returns.to_numpy()[-window]
    assert state["ReturnIn30DaysBefore"] == pytest.approx(expected_oldest)


def test_mean_in_30_days_before_selects_oldest():
    returns = pd.Series([0.01 * i for i in range(1, 11)])
    window = 3
    rm = rolling_mean(returns, window)
    means = rm.dropna().to_numpy()
    state = compute_rolling_state(returns, rm, window, "population")
    expected_oldest_mean = means[-window]
    assert state["MeanIn30DaysBefore"] == pytest.approx(expected_oldest_mean)


def test_rolling_state_insufficient_data():
    returns = pd.Series([0.1, 0.2])
    rm = rolling_mean(returns, 3)
    assert compute_rolling_state(returns, rm, 3, "population") is None
