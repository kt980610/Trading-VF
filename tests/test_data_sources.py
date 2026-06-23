"""Production-safe data-flow tests (network-free).

Covers the seven required cases:
1. daily loader uses the daily file even when a 1m file also exists;
2. daily close [100,110,99] -> returns [0.10,-0.10];
3. distribution artifact stores a return_decimal grid (not price levels);
4. missing daily source -> missing_daily_source (no fallback to minute);
5. missing minute source -> missing_minute_source;
6. minute RF feature join uses only the previous completed day's artifact;
7. scenario price mapping uses entry_price * (1 + return_decimal).
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.artifact_join import build_previous_day_provider
from src.config import load_config
from src.data_loader import (
    DataSourceError,
    load_daily_ohlcv,
    load_minute_ohlcv,
    minute_returns,
)
from src.pricing import scenario_price
from src.rolling_features import compute_daily_returns


def _write_csv(path, timestamps, closes):
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1.0] * len(closes),
        }
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _config(tmp_path, **data_overrides):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "symbols:",
        "  - BTCUSDT",
        "data:",
        f"  raw_dir: {raw_dir.as_posix()}",
    ]
    for k, v in data_overrides.items():
        lines.append(f"  {k}: {v}")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return load_config(str(cfg_path))


# 1. Daily loader must pick the daily file even when a 1m file also exists.
def test_daily_loader_ignores_minute_file(tmp_path):
    raw = tmp_path / "data" / "raw"
    days = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    _write_csv(str(raw / "BTCUSDT_daily.csv"), days, [100.0, 110.0, 99.0])
    minutes = pd.date_range("2024-01-01", periods=3, freq="min", tz="UTC")
    _write_csv(str(raw / "BTCUSDT_1m.csv"), minutes, [1.0, 2.0, 3.0])

    cfg = _config(tmp_path)
    daily = load_daily_ohlcv(cfg, "BTCUSDT")
    assert list(daily["close"]) == [100.0, 110.0, 99.0]  # from the daily file


# 2. Daily returns from [100,110,99] are [0.10, -0.10].
def test_daily_returns_are_percent_change(tmp_path):
    raw = tmp_path / "data" / "raw"
    days = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    _write_csv(str(raw / "BTCUSDT_daily.csv"), days, [100.0, 110.0, 99.0])
    cfg = _config(tmp_path)
    daily = load_daily_ohlcv(cfg, "BTCUSDT")
    rets = compute_daily_returns(daily["close"]).to_numpy()
    assert np.allclose(rets, [0.10, -0.10])


# 3. Distribution artifact stores a return_decimal grid (small values), not price.
def test_distribution_grid_is_return_decimal(tmp_path, monkeypatch):
    from src.main_build_distributions import run

    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "data" / "raw"
    days = pd.date_range("2020-01-01", periods=400, freq="D", tz="UTC")
    rng = np.random.default_rng(7)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.02, 400))
    _write_csv(str(raw / "BTCUSDT_daily.csv"), days, list(close))

    cfg_text = (
        "symbols:\n  - BTCUSDT\n"
        "data:\n  raw_dir: data/raw\n  output_path: out.json\n"
        "distribution:\n  method: histogram\n  bins: 100\n  min_observations: 60\n"
        "rolling:\n  window_days: 30\n  variance_mode: population\n"
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    run(str(cfg_path))

    loaded = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert loaded["distribution_unit"] == "return_decimal"
    grid = loaded["symbols"]["BTCUSDT"]["return_distribution"]["grid"]
    # Return grid lives near zero; price levels (~100) would violate this.
    assert max(abs(g) for g in grid) < 1.0


# 4. Missing daily source -> missing_daily_source, even if minute file exists.
def test_missing_daily_source_no_fallback(tmp_path):
    raw = tmp_path / "data" / "raw"
    minutes = pd.date_range("2024-01-01", periods=5, freq="min", tz="UTC")
    _write_csv(str(raw / "BTCUSDT_1m.csv"), minutes, [1.0, 2.0, 3.0, 4.0, 5.0])
    cfg = _config(tmp_path)
    with pytest.raises(DataSourceError) as exc:
        load_daily_ohlcv(cfg, "BTCUSDT")
    assert exc.value.reason == "missing_daily_source"


# 5. Missing minute source -> missing_minute_source.
def test_missing_minute_source(tmp_path):
    raw = tmp_path / "data" / "raw"
    days = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    _write_csv(str(raw / "BTCUSDT_daily.csv"), days, [100.0, 110.0, 99.0])
    cfg = _config(tmp_path)
    with pytest.raises(DataSourceError) as exc:
        load_minute_ohlcv(cfg, "BTCUSDT")
    assert exc.value.reason == "missing_minute_source"


# 5b. require_*_source: false -> returns None instead of raising.
def test_optional_source_returns_none(tmp_path):
    cfg = _config(tmp_path, require_daily_source="false", require_minute_source="false")
    assert load_daily_ohlcv(cfg, "BTCUSDT") is None
    assert load_minute_ohlcv(cfg, "BTCUSDT") is None


# 6. Minute RF feature join uses only the previous completed day's artifact.
def test_previous_day_provider_no_same_day_lookahead():
    ctx = {
        "2024-01-01": {"feat": 1.0},
        "2024-01-02": {"feat": 2.0},
        "2024-01-03": {"feat": 3.0},
    }
    provider = build_previous_day_provider(ctx)
    # A minute on 2024-01-03 must see 2024-01-02's artifact, not the same day's.
    assert provider(pd.Timestamp("2024-01-03 09:30", tz="UTC")) == {"feat": 2.0}
    assert provider(pd.Timestamp("2024-01-02 00:00", tz="UTC")) == {"feat": 1.0}
    # No prior day available -> empty.
    assert provider(pd.Timestamp("2024-01-01 12:00", tz="UTC")) == {}


# 9. News join never uses same-day or future news (only previous completed day).
def test_news_join_excludes_future_and_same_day():
    news_by_date = {
        "2024-01-01": {"macro_news_sentiment": 0.1},
        "2024-01-02": {"macro_news_sentiment": 0.2},
        "2024-01-03": {"macro_news_sentiment": 0.9},  # same-day must be excluded
    }
    provider = build_previous_day_provider(news_by_date)
    feats = provider(pd.Timestamp("2024-01-03 14:00", tz="UTC"))
    assert feats == {"macro_news_sentiment": 0.2}  # previous completed day only


# 7. Scenario price uses entry_price * (1 + return_decimal).
def test_scenario_price_mapping():
    assert scenario_price(100.0, 0.10) == pytest.approx(110.0)
    assert scenario_price(250.0, -0.025) == pytest.approx(243.75)


def test_minute_returns_helper():
    rets = minute_returns(pd.Series([100.0, 110.0, 99.0])).to_numpy()
    assert np.allclose(rets, [0.10, -0.10])
