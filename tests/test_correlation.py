import numpy as np
import pandas as pd

from src import correlation as corr


def _matrix():
    idx = pd.to_datetime([f"2024-01-{d:02d}" for d in range(1, 10)], utc=True)
    # A and B are identical before 01-05 and opposite from 01-05 onwards.
    a = [0.01, 0.02, -0.01, 0.03, 0.02, -0.02, 0.01, -0.03, 0.02]
    b = [0.01, 0.02, -0.01, 0.03, -0.02, 0.02, -0.01, 0.03, -0.02]
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def test_self_correlation_is_one():
    m = _matrix()
    c = corr.correlation_for_date(
        m, pd.Timestamp("2024-01-05", tz="UTC"), ["A", "B"],
        lookback_days=100, min_required_days=3,
    )
    assert c["A"]["A"] == 1.0
    assert c["B"]["B"] == 1.0


def test_no_leakage_uses_only_prior_days():
    m = _matrix()
    early = corr.correlation_for_date(
        m, pd.Timestamp("2024-01-05", tz="UTC"), ["A", "B"],
        lookback_days=100, min_required_days=3,
    )
    later = corr.correlation_for_date(
        m, pd.Timestamp("2024-01-09", tz="UTC"), ["A", "B"],
        lookback_days=100, min_required_days=3,
    )
    # Before 01-05 the series are identical -> ~+1; including the opposite days
    # after 01-05 must lower the correlation.
    assert early["A"]["B"] > 0.9
    assert early["A"]["B"] > later["A"]["B"]


def test_min_required_days_falls_back():
    m = _matrix()
    c = corr.correlation_for_date(
        m, pd.Timestamp("2024-01-03", tz="UTC"), ["A", "B"],
        lookback_days=100, min_required_days=10, fallback_cross_corr=0.0,
    )
    # Only 01-01/01-02 precede 01-03 -> below min_required -> fallback.
    assert c["A"]["B"] == 0.0
    assert c["A"]["A"] == 1.0


def test_daily_returns_matrix_aligns_dates():
    d1 = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True), "close": [100.0, 110.0]}
    )
    d2 = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True), "close": [50.0, 55.0]}
    )
    m = corr.daily_returns_matrix({"A": d1, "B": d2})
    assert np.isclose(m.iloc[1]["A"], 0.1)
    assert np.isclose(m.iloc[1]["B"], 0.1)


def test_provider_weight_fallbacks():
    p = corr.CorrelationProvider.from_rows(
        [{"date": "2024-01-02", "corr": {"ETHUSDT": {"ETHUSDT": 1.0, "BTCUSDT": 0.5}}}]
    )
    assert p.weight("2024-01-02", "ETHUSDT", "BTCUSDT") == 0.5
    assert p.weight("2024-01-02", "ETHUSDT", "ETHUSDT") == 1.0
    # Unknown source -> cross fallback (0.0 by default).
    assert p.weight("2024-01-02", "ETHUSDT", "SOLUSDT") == 0.0
