import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.main_build_distributions import build_symbol_entry, run
from src.rolling_features import compute_daily_returns
from src.snapshot_writer import build_snapshot, write_snapshot


class _Cfg:
    class distribution:
        bins = 50
        min_observations = 60
        method = "histogram"

    class rolling:
        window_days = 5
        variance_mode = "population"


def _synthetic_close(n, seed, drift):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.02, n)
    close = 100.0 * np.cumprod(1.0 + rets)
    return pd.Series(close)


def test_insufficient_data_invalid():
    close = _synthetic_close(20, 1, 0.001)
    returns = compute_daily_returns(close)
    entry = build_symbol_entry(_Cfg, returns)
    assert entry["valid"] is False
    assert entry["reason"] == "not_enough_observations"


def test_snapshot_schema_keys(tmp_path):
    close = _synthetic_close(400, 2, 0.001)
    returns = compute_daily_returns(close)
    entry = build_symbol_entry(_Cfg, returns)
    assert entry["valid"] is True

    snapshot = build_snapshot("histogram", 50, {"BTCUSDT": entry})
    out = tmp_path / "snap.json"
    write_snapshot(snapshot, str(out))

    with open(out, "r", encoding="utf-8") as fh:
        loaded = json.load(fh)

    assert "created_at" in loaded
    assert loaded["method"] == "histogram"
    assert loaded["bins"] == 50
    assert "symbols" in loaded

    sym = loaded["symbols"]["BTCUSDT"]
    for key in [
        "return_distribution",
        "mean_distribution",
        "variance_distribution",
        "mean_of_mean_distribution",
        "var_of_mean_distribution",
        "rolling_state",
        "n_daily_returns",
    ]:
        assert key in sym
    for key in [
        "Last30DaysMean",
        "Last30DaysVar",
        "Last30DaysMeanOfMean",
        "Last30DaysVarOfMean",
        "ReturnIn30DaysBefore",
        "MeanIn30DaysBefore",
    ]:
        assert key in sym["rolling_state"]
    dist = sym["return_distribution"]
    for key in ["grid", "pdf", "mean", "variance", "min", "max", "n_observations"]:
        assert key in dist


def test_symbols_have_distinct_distributions():
    btc = compute_daily_returns(_synthetic_close(400, 10, 0.005))
    eth = compute_daily_returns(_synthetic_close(400, 11, -0.005))
    btc_entry = build_symbol_entry(_Cfg, btc)
    eth_entry = build_symbol_entry(_Cfg, eth)
    btc_mean = btc_entry["return_distribution"]["mean"]
    eth_mean = eth_entry["return_distribution"]["mean"]
    assert btc_mean != eth_mean
    assert abs(btc_mean - eth_mean) > 1e-4


def test_run_pipeline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    days = pd.date_range("2020-01-01", periods=400, freq="D", tz="UTC")
    close = _synthetic_close(400, 20, 0.001)
    df = pd.DataFrame({
        "timestamp": days,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": np.arange(400, dtype="float64"),
    })
    df.to_csv(raw_dir / "BTCUSDT_daily.csv", index=False)

    config_text = (
        "symbols:\n"
        "  - BTCUSDT\n"
        "  - NEOUSDT\n"
        "data:\n"
        "  raw_dir: raw\n"
        "  output_path: out.json\n"
        "  timeframe: 1m\n"
        "  daily_resample: true\n"
        "rolling:\n"
        "  window_days: 30\n"
        "  variance_mode: population\n"
        "distribution:\n"
        "  method: histogram\n"
        "  bins: 100\n"
        "  min_observations: 60\n"
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(config_text, encoding="utf-8")

    run(str(cfg_path))

    out_path = tmp_path / "out.json"
    assert out_path.exists()
    with open(out_path, "r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert loaded["symbols"]["BTCUSDT"]["valid"] is True
    assert loaded["symbols"]["NEOUSDT"]["valid"] is False
    assert loaded["symbols"]["NEOUSDT"]["reason"] == "missing_daily_source"
    # Distribution metadata makes the return-decimal contract explicit.
    assert loaded["source_frequency"] == "1d"
    assert loaded["distribution_unit"] == "return_decimal"
    assert loaded["return_definition"] == "close_t / close_t_minus_1 - 1"
