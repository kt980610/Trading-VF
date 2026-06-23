"""Tests for the candidate-only RF training CLI (src/main_train_rf.py).

The CLI must NEVER write to ``models/promoted``: it only emits candidate
artifacts under ``<models_staging>/rf/<run-id>/<SYMBOL>/`` plus a run-level
report under ``reports/rf_runs/<run-id>/``. These tests stub out the heavy data
pipeline (snapshot/minute-loader/simulator) and exercise the orchestration.
"""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import main_train_rf as mtr
from src import rf_classifier as rfc

# Compact grid so the per-symbol joint RF + threshold selection stays fast.
FAST_GRID = [
    {"n_estimators": 40, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 40, "max_depth": 6, "min_samples_leaf": 5, "max_features": "sqrt"},
]


def _synthetic_trades(symbol="BTCUSDT", n_trades=30, minutes=20, seed=0):
    """Economically learnable per-symbol trade dataset (beats the baseline)."""
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2023-01-01", tz="UTC")
    for tid in range(n_trades):
        entry_ts = base + pd.Timedelta(days=tid)
        for m in range(minutes):
            signal = rng.normal()
            terminal = m == minutes - 1
            rows.append(
                {
                    "trade_id": tid,
                    "leg_id": f"{tid}:both",
                    "symbol": symbol,
                    "side_code": 2,
                    "entry_timestamp": entry_ts,
                    "timestamp": entry_ts + pd.Timedelta(minutes=m),
                    "y": signal * 0.01,
                    "current_pnl": signal * 5.0,
                    "mdp_liq_risk_score": abs(signal) % 1.0,
                    "season_pre_halving_2y": 1.0,
                    "season_post_halving_2y": 0.0,
                    "season_unknown": 0.0,
                    "close_label": 1 if (signal > 0.5 or terminal) else 0,
                    "sample_weight": 1.0 + abs(signal),
                    "trade_pnl_rate_now": signal,
                    "net_close_pnl_now": 10.0 * signal,
                    "leg_liquidated_now": 0,
                    "fully_liquidated_now": 0,
                    "is_terminal": int(terminal),
                }
            )
    return pd.DataFrame(rows)


def _fake_config(base):
    paths = SimpleNamespace(
        rf_dataset_dir="data",
        models_promoted="models/promoted",
        models_staging="models/staging",
        rf_policy_report="reports/rf_policy_report.csv",
        distribution_snapshot="data/distribution_snapshot.json",
    )
    cfg = SimpleNamespace(
        paths=paths,
        rf_model=SimpleNamespace(
            min_training_rows=100, n_estimators=50, max_depth=0,
            random_state=42, include_optional_component_features=False,
        ),
        halving=SimpleNamespace(dates=None, season_seed=0),
        simulation=SimpleNamespace(fee_rate=0.0, funding_rate=0.0, slippage=0.0, close_epsilon=0.0),
        rolling=SimpleNamespace(window_days=30),
        news=SimpleNamespace(enabled=False, intraday_news_enabled=False),
        symbols=["BTCUSDT"],
    )

    def resolve(path):
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(str(base), path))

    cfg.resolve = resolve
    return cfg


def _patch_pipeline(monkeypatch, base, dataset=None):
    """Stub the heavy data pipeline so run() trains on a synthetic dataset."""
    cfg = _fake_config(base)
    df = dataset if dataset is not None else _synthetic_trades()

    monkeypatch.setattr(mtr, "load_config", lambda p: cfg)
    monkeypatch.setattr(mtr, "load_snapshot", lambda p: {"symbols": {"BTCUSDT": {"valid": True}}})
    monkeypatch.setattr(mtr, "load_minute_ohlcv", lambda c, s: pd.DataFrame({"close": [1.0, 2.0]}))
    monkeypatch.setattr(mtr, "build_cache_for_symbol", lambda s, e, window=30: object())
    monkeypatch.setattr(mtr, "_build_context_provider", lambda c, s: (lambda ts: {}))
    monkeypatch.setattr(mtr.sim, "generate_trade_dataset", lambda *a, **k: df.copy())

    orig_train = rfc.train_close_classifier

    def _fast_train(*a, **k):
        k.setdefault("param_grid", FAST_GRID)
        return orig_train(*a, **k)

    monkeypatch.setattr(mtr.rfc, "train_close_classifier", _fast_train)
    return cfg


@pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "a\\b", "../x", "with space", "x/../y"])
def test_validate_run_id_rejects_unsafe(bad):
    with pytest.raises(ValueError):
        mtr.validate_run_id(bad)


def test_validate_run_id_accepts_safe():
    assert mtr.validate_run_id("no_news_4y_v1") == "no_news_4y_v1"


def test_candidate_run_writes_staging_not_promoted(tmp_path, monkeypatch):
    _patch_pipeline(monkeypatch, tmp_path)
    out = mtr.run(str(tmp_path / "config.yaml"), run_id="no_news_4y_v1")

    staging_model = (
        tmp_path / "models" / "staging" / "rf" / "no_news_4y_v1" / "BTCUSDT" / rfc.MODEL_FILE
    )
    assert staging_model.is_file()
    # Scaler/schema/threshold also land in staging.
    staging_dir = staging_model.parent
    for fname in (rfc.SCALER_FILE, rfc.SCHEMA_FILE, rfc.THRESHOLD_FILE, rfc.METADATA_FILE):
        assert (staging_dir / fname).is_file()

    # NOTHING under models/promoted.
    promoted = tmp_path / "models" / "promoted"
    assert not promoted.exists() or not any(promoted.rglob("*"))

    # Dataset is scoped to the run id, not the global data dir.
    assert (tmp_path / "data" / "no_news_4y_v1" / "BTCUSDT").exists()

    assert out["mode"] == "candidate_only"


def test_summary_json_has_mode_and_run_metadata(tmp_path, monkeypatch):
    _patch_pipeline(monkeypatch, tmp_path)
    mtr.run(str(tmp_path / "config.yaml"), run_id="no_news_4y_v1")

    summary_path = tmp_path / "reports" / "rf_runs" / "no_news_4y_v1" / "summary.json"
    assert summary_path.is_file()
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    assert data["mode"] == "candidate_only"
    assert data["run_id"] == "no_news_4y_v1"
    assert data["symbols"] == ["BTCUSDT"]
    assert data["news_enabled"] is False
    assert data["news_intraday_enabled"] is False
    assert "git_commit" in data  # may be None, but key must exist
    assert "generated_at_utc" in data
    assert isinstance(data["symbols_report"], list) and data["symbols_report"]

    row = data["symbols_report"][0]
    for key in (
        "symbol", "status",
        "selected_n_estimators", "selected_max_depth",
        "selected_min_samples_leaf", "selected_max_features",
        "close_probability_threshold",
        "validation_total_pnl_per_minute_objective",
        "baseline_validation_objective",
        "test_total_pnl_per_minute_objective",
        "test_net_pnl", "test_max_drawdown", "test_turnover", "test_liquidation_count",
    ):
        assert key in row, key

    # CSV report exists too.
    assert (tmp_path / "reports" / "rf_runs" / "no_news_4y_v1" / "summary.csv").is_file()


def test_second_run_same_run_id_fails_without_overwrite(tmp_path, monkeypatch):
    _patch_pipeline(monkeypatch, tmp_path)
    mtr.run(str(tmp_path / "config.yaml"), run_id="no_news_4y_v1")

    staging_model = (
        tmp_path / "models" / "staging" / "rf" / "no_news_4y_v1" / "BTCUSDT" / rfc.MODEL_FILE
    )
    before = staging_model.read_bytes()

    # Re-running the SAME run-id must fail-closed before training, leaving the
    # existing candidate artifacts untouched.
    _patch_pipeline(monkeypatch, tmp_path)
    with pytest.raises(FileExistsError):
        mtr.run(str(tmp_path / "config.yaml"), run_id="no_news_4y_v1")

    assert staging_model.read_bytes() == before


def test_run_refuses_to_target_promoted(tmp_path, monkeypatch):
    # If models_staging is (mis)configured to equal models_promoted, the run must
    # refuse rather than write candidate artifacts under promoted.
    cfg = _patch_pipeline(monkeypatch, tmp_path)
    cfg.paths.models_staging = cfg.paths.models_promoted
    with pytest.raises(ValueError):
        mtr.run(str(tmp_path / "config.yaml"), run_id="no_news_4y_v1")
    promoted = tmp_path / "models" / "promoted"
    assert not promoted.exists() or not any(promoted.rglob("*"))
