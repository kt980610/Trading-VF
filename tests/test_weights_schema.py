import json
import os

import numpy as np

from src.portfolio import weights_writer as ww
from src.portfolio.config import PortfolioConfig
from src.portfolio.mvo import MVOStats


def _config():
    return PortfolioConfig(
        symbols=["BTCUSDT", "ETHUSDT"],
        weight_steps={"BTCUSDT": 0.01, "ETHUSDT": 0.01},
        lookback_days=30,
        integer_tolerance=1e-9,
    )


def _payload():
    config = _config()
    stats = MVOStats(
        symbols=["BTCUSDT"],
        mu=np.array([0.002]),
        variance=np.array([0.01]),
        sigma_reg=np.array([[0.01]]),
    )
    valid_symbols = ["BTCUSDT"]
    invalid = {"ETHUSDT": "not_enough_return_days"}
    continuous_w = np.array([0.237])
    discrete_k = np.array([11])
    discrete_w = np.array([0.22])
    return config, ww.build_payload(
        config, "2024-09-29", stats, valid_symbols, invalid, continuous_w, discrete_k, discrete_w
    )


def test_schema_top_level_keys():
    _, payload = _payload()
    for key in ["as_of_date", "lookback_days", "input_type", "objective", "symbols",
                "sum_weight_discrete", "cash_weight"]:
        assert key in payload


def test_valid_symbol_entry_fields():
    _, payload = _payload()
    btc = payload["symbols"]["BTCUSDT"]
    assert btc["valid"] is True
    assert np.isclose(btc["weight_discrete"], 0.22)
    assert np.isclose(btc["long_weight"], 0.11)
    assert np.isclose(btc["short_weight"], 0.11)
    assert btc["integer_constraint_ok"] is True
    assert "weight_continuous" in btc


def test_invalid_symbol_entry():
    _, payload = _payload()
    eth = payload["symbols"]["ETHUSDT"]
    assert eth["valid"] is False
    assert eth["reason"] == "not_enough_return_days"
    assert eth["weight_discrete"] == 0.0


def test_cash_weight_is_one_minus_sum():
    _, payload = _payload()
    assert np.isclose(payload["cash_weight"], 1.0 - payload["sum_weight_discrete"])


def test_long_short_divisible_by_step():
    _, payload = _payload()
    btc = payload["symbols"]["BTCUSDT"]
    step = btc["weight_step"]
    assert np.isclose(btc["long_weight"] / step, round(btc["long_weight"] / step))
    assert np.isclose(btc["short_weight"] / step, round(btc["short_weight"] / step))


def test_atomic_write_roundtrip(tmp_path):
    _, payload = _payload()
    out = str(tmp_path / "portfolio_weights.json")
    ww.write_atomic(payload, out)
    assert os.path.isfile(out)
    # Temp file must be gone after the atomic rename.
    assert not os.path.isfile(out + ".tmp")
    with open(out, "r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert loaded["symbols"]["BTCUSDT"]["weight_discrete"] == 0.22


def test_end_to_end_run(tmp_path, monkeypatch):
    import pandas as pd

    from src.portfolio.main_build_portfolio_weights import run

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "config").mkdir()

    btc_dates = pd.date_range("2024-09-01", "2024-09-28", freq="D")
    eth_dates = pd.date_range("2024-09-26", "2024-09-28", freq="D")
    rows = [{"date": str(d.date()), "symbol": "BTCUSDT", "daily_return": 0.01} for d in btc_dates]
    rows += [{"date": str(d.date()), "symbol": "ETHUSDT", "daily_return": 0.0} for d in eth_dates]
    pd.DataFrame(rows).to_csv(tmp_path / "data" / "simulation_results.csv", index=False)

    with open(tmp_path / "data" / "realized_symbol_returns.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"date": "2024-09-28", "symbol": "BTCUSDT", "realized_return": 0.02}) + "\n")

    config_text = (
        "portfolio:\n"
        "  lookback_days: 30\n"
        "  min_required_days: 5\n"
        "  base_capital_per_symbol: 1000\n"
        "  risk_aversion: 2.0\n"
        "  allow_cash: true\n"
        "  max_weight_per_symbol: 0.40\n"
        "  covariance_mode: sample\n"
        "  covariance_epsilon: 1.0e-6\n"
        "  default_weight_step: 0.01\n"
        "  symbols:\n"
        "    - BTCUSDT\n"
        "    - ETHUSDT\n"
        "  weight_steps:\n"
        "    BTCUSDT: 0.01\n"
        "    ETHUSDT: 0.01\n"
        "branch_and_bound:\n"
        "  enabled: true\n"
        "  max_nodes: 50000\n"
        "  max_runtime_ms: 5000\n"
        "  objective_tolerance: 1.0e-9\n"
        "paths:\n"
        "  simulated_returns: data/simulation_results.csv\n"
        "  realized_returns: data/realized_symbol_returns.jsonl\n"
        "  output_path: data/portfolio_weights.json\n"
    )
    cfg_path = tmp_path / "config" / "portfolio_config.yaml"
    cfg_path.write_text(config_text, encoding="utf-8")

    payload = run(str(cfg_path), as_of_date="2024-09-29")

    out_path = tmp_path / "data" / "portfolio_weights.json"
    assert out_path.exists()
    assert payload["symbols"]["BTCUSDT"]["valid"] is True
    assert payload["symbols"]["ETHUSDT"]["valid"] is False
    assert np.isclose(payload["cash_weight"], 1.0 - payload["sum_weight_discrete"])
