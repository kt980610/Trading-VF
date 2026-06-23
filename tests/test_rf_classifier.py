"""Tests for the per-symbol RF binary close classifier (spec 7,9,10; tests 13-15,19-20)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import rf_classifier as rfc

SCALE_COLS = ["y", "current_pnl", "mdp_liq_risk_score"]
PASS_COLS = ["season_pre_halving_2y", "season_post_halving_2y", "season_unknown", "side_code"]

# Compact, deterministic grid so the per-symbol joint RF + threshold selection
# stays fast in tests while still exercising the >1-candidate selection path.
FAST_GRID = [
    {"n_estimators": 40, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 40, "max_depth": 6, "min_samples_leaf": 5, "max_features": "sqrt"},
]


def _synthetic_trades(n_trades=30, minutes=25, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2023-01-01", tz="UTC")
    for tid in range(n_trades):
        entry_ts = base + pd.Timedelta(days=tid)
        for m in range(minutes):
            signal = rng.normal()
            # Learnable: close when signal high; terminal minute always closes.
            terminal = m == minutes - 1
            close_label = 1 if (signal > 0.5 or terminal) else 0
            # Economically learnable net PnL: closing on a high-signal minute is
            # strongly profitable, so an RF that fires there beats the
            # continue-to-terminal baseline on the new sum(net/holding) objective.
            net_close_pnl_now = 10.0 * signal
            rows.append(
                {
                    "trade_id": tid,
                    "leg_id": f"{tid}:both",
                    "symbol": "BTCUSDT",
                    "side_code": 2,
                    "entry_timestamp": entry_ts,
                    "timestamp": entry_ts + pd.Timedelta(minutes=m),
                    "y": signal * 0.01,
                    "current_pnl": signal * 5.0,
                    "mdp_liq_risk_score": abs(signal) % 1.0,
                    "season_pre_halving_2y": 1.0,
                    "season_post_halving_2y": 0.0,
                    "season_unknown": 0.0,
                    "close_label": close_label,
                    "sample_weight": 1.0 + abs(signal),
                    "trade_pnl_rate_now": signal,
                    "net_close_pnl_now": net_close_pnl_now,
                    "leg_liquidated_now": 0,
                    "fully_liquidated_now": 0,
                    "is_terminal": int(terminal),
                }
            )
    return pd.DataFrame(rows)


def test_trade_group_split_no_trade_crosses_splits():
    df = _synthetic_trades()
    train, val, test = rfc.trade_group_split(df)
    s_train = set(train["trade_id"])
    s_val = set(val["trade_id"])
    s_test = set(test["trade_id"])
    assert s_train.isdisjoint(s_val)
    assert s_train.isdisjoint(s_test)
    assert s_val.isdisjoint(s_test)


def test_train_and_decide_and_threshold():
    df = _synthetic_trades()
    result = rfc.train_close_classifier(df, "BTCUSDT", SCALE_COLS, PASS_COLS, min_training_rows=100, param_grid=FAST_GRID)
    assert result["valid"] is True
    assert result["status"] == rfc.STATUS_OK
    assert 0.05 <= result["threshold"] <= 0.95
    policy = result["policy"]
    decisions = policy.decide(df.head(10))
    assert all(d in (rfc.DECISION_CLOSE, rfc.DECISION_CONTINUE) for d in decisions)


def test_scaler_is_fit_on_train_only():
    df = _synthetic_trades()
    train, _val, _test = rfc.trade_group_split(df)
    result = rfc.train_close_classifier(df, "BTCUSDT", SCALE_COLS, PASS_COLS, min_training_rows=100, param_grid=FAST_GRID)
    scaler = result["scaler"]
    # RobustScaler center_ is the median of the (train) fit data, not the full set.
    train_median_y = float(np.median(train["y"].to_numpy()))
    assert scaler.center_[0] == pytest.approx(train_median_y, rel=1e-6, abs=1e-9)


def test_schema_train_live_feature_order_matches():
    df = _synthetic_trades()
    result = rfc.train_close_classifier(df, "BTCUSDT", SCALE_COLS, PASS_COLS, min_training_rows=100, param_grid=FAST_GRID)
    schema = result["schema"]
    assert schema["final_feature_order"] == SCALE_COLS + PASS_COLS


def test_save_load_roundtrip_decisions_match(tmp_path):
    df = _synthetic_trades()
    result = rfc.train_close_classifier(df, "BTCUSDT", SCALE_COLS, PASS_COLS, min_training_rows=100, param_grid=FAST_GRID)
    model_dir = str(tmp_path / "BTCUSDT")
    rfc.save_artifacts(model_dir, result)
    for fname in (rfc.MODEL_FILE, rfc.SCALER_FILE, rfc.SCHEMA_FILE, rfc.THRESHOLD_FILE, rfc.METADATA_FILE):
        assert os.path.isfile(os.path.join(model_dir, fname))

    loaded, status = rfc.load_policy(model_dir)
    assert status == rfc.STATUS_OK
    sample = df.head(20)
    assert loaded.decide(sample) == result["policy"].decide(sample)


def _json_proba(export, row):
    """Reproduce predict_proba[:,1] from the framework-free JSON export."""
    scale_cols = export["scale_cols"]
    passthrough = export["passthrough_cols"]
    medians = export["imputer"]["medians"]
    center = export["scaler"]["center"]
    scale = export["scaler"]["scale"]
    x = []
    for j, c in enumerate(scale_cols):
        raw = row.get(c, np.nan)
        v = raw if np.isfinite(raw) else medians.get(c, 0.0)
        s = scale[j] if scale[j] else 1.0
        x.append((v - center[j]) / s)
    for c in passthrough:
        raw = row.get(c, 0.0)
        x.append(raw if np.isfinite(raw) else 0.0)
    total = 0.0
    for tree in export["trees"]:
        node = 0
        while tree["children_left"][node] != -1:
            feat = tree["feature"][node]
            if feat >= 0 and x[feat] <= tree["threshold"][node]:
                node = tree["children_left"][node]
            else:
                node = tree["children_right"][node]
        total += tree["value"][node]
    return total / len(export["trees"])


def test_json_export_matches_sklearn_predict_proba():
    df = _synthetic_trades()
    result = rfc.train_close_classifier(df, "BTCUSDT", SCALE_COLS, PASS_COLS, min_training_rows=100, param_grid=FAST_GRID)
    export = rfc.export_model_json(result)
    sample = df.head(15).reset_index(drop=True)
    sk = result["policy"].predict_proba_close(sample)
    for i in range(len(sample)):
        approx = _json_proba(export, sample.iloc[i].to_dict())
        assert abs(approx - sk[i]) < 1e-9, (i, approx, sk[i])


class _StubModel:
    """Returns P(close) straight from the single passthrough feature column."""

    classes_ = [0, 1]

    def predict_proba(self, X):
        p = np.asarray(X)[:, 0]
        return np.column_stack([1.0 - p, p])


def _eval_test_df():
    base = pd.Timestamp("2023-01-01", tz="UTC")
    rows = [
        # Trade A: RF fires at minute 1 (p=0.9); terminal at minute 2. No liq.
        dict(trade_id=0, timestamp=base, p_close_signal=0.1, net_close_pnl_now=10.0,
             trade_pnl_rate_now=10.0, close_label=0, is_terminal=0,
             leg_liquidated_now=0, fully_liquidated_now=0),
        dict(trade_id=0, timestamp=base + pd.Timedelta(minutes=1), p_close_signal=0.9,
             net_close_pnl_now=50.0, trade_pnl_rate_now=25.0, close_label=1, is_terminal=0,
             leg_liquidated_now=0, fully_liquidated_now=0),
        dict(trade_id=0, timestamp=base + pd.Timedelta(minutes=2), p_close_signal=0.2,
             net_close_pnl_now=5.0, trade_pnl_rate_now=1.67, close_label=1, is_terminal=1,
             leg_liquidated_now=0, fully_liquidated_now=0),
        # Trade B: RF never fires -> terminal full liquidation.
        dict(trade_id=1, timestamp=base + pd.Timedelta(minutes=10), p_close_signal=0.1,
             net_close_pnl_now=8.0, trade_pnl_rate_now=8.0, close_label=0, is_terminal=0,
             leg_liquidated_now=0, fully_liquidated_now=0),
        dict(trade_id=1, timestamp=base + pd.Timedelta(minutes=11), p_close_signal=0.2,
             net_close_pnl_now=-3.0, trade_pnl_rate_now=-1.5, close_label=1, is_terminal=1,
             leg_liquidated_now=1, fully_liquidated_now=1),
    ]
    return pd.DataFrame(rows)


def test_evaluate_close_policy_baseline_and_metrics():
    policy = rfc.ClosePolicy(
        scale_cols=[], passthrough_cols=["p_close_signal"], medians={},
        threshold=0.5, model=_StubModel(), scaler=None,
    )
    m = rfc.evaluate_close_policy(policy, _eval_test_df())

    # RF closes A at the 0.9 minute (50) and B at terminal (-3) -> 47.
    assert m["total_net_pnl"] == pytest.approx(47.0)
    # Baseline closes BOTH at terminal: A=5, B=-3 -> 2.
    assert m["baseline_total_net_pnl"] == pytest.approx(2.0)
    assert m["baseline_pnl_difference"] == pytest.approx(45.0)
    # Liquidations counted per-trade, once (only trade B is a full liquidation).
    assert m["liquidation_count"] == 1
    assert m["full_liquidation_count"] == 1
    assert m["baseline_full_liquidation_count"] == 1
    # Classification: only A's 0.9 minute predicts close.
    assert m["close_count"] == 1
    assert m["continue_count"] == 4
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0 / 3.0)
    # Drawdown: equity 50 -> 47 after B, peak-to-trough = 3.
    assert m["max_drawdown"] == pytest.approx(3.0)
    assert m["n_test_trades"] == 2
    assert m["n_test_minute_rows"] == 5


def test_evaluate_close_policy_empty():
    policy = rfc.ClosePolicy(
        scale_cols=[], passthrough_cols=["p_close_signal"], medians={},
        threshold=0.5, model=_StubModel(), scaler=None,
    )
    m = rfc.evaluate_close_policy(policy, pd.DataFrame())
    assert m["total_net_pnl"] == 0.0
    assert m["n_test_trades"] == 0


def test_missing_model_status_no_cross_symbol_fallback(tmp_path):
    loaded, status = rfc.load_policy(str(tmp_path / "ETHUSDT"))
    assert loaded is None
    assert status == rfc.STATUS_MISSING_MODEL


def test_total_objective_beats_mean_per_minute_threshold():
    # Two 2-minute trades. p decreases within each trade, so a LOW threshold fires
    # at minute 0 (early close) and a HIGH threshold (0.95) never fires (terminal).
    #
    #   minute0: net=1   (holding 1)   -> sum contribution 1 per trade
    #   terminal: net=100 (holding 2)  -> sum contribution 50 per trade
    #
    # sum(net/holding): early = 1+1 = 2 ; terminal = 50+50 = 100  -> picks terminal
    # avg pnl/minute  : early = 2/2 = 1 ; terminal = 200/4 = 50   -> also terminal
    # ...so to make them DISAGREE we instead compare against the *mean trade rate*
    # objective the old code used (trade_pnl_rate_now), which is decoupled below.
    val = pd.DataFrame(
        {
            "trade_id": [0, 0, 1, 1],
            "timestamp": pd.to_datetime(
                ["2023-01-01T00:00", "2023-01-01T00:01",
                 "2023-01-02T00:00", "2023-01-02T00:01"],
                utc=True,
            ),
            # Old objective (mean of this) is high at minute 0, low at terminal.
            "trade_pnl_rate_now": [100.0, 1.0, 100.0, 1.0],
            # New objective: terminal close is far more profitable in net PnL.
            "net_close_pnl_now": [1.0, 100.0, 1.0, 100.0],
        }
    )
    p_close = np.array([0.9, 0.1, 0.9, 0.1])

    thr, obj = rfc.select_threshold(val, p_close)
    # Sum(net/holding): firing at minute0 -> 1/1+1/1 = 2; terminal -> 100/2*2 = 100.
    assert obj == pytest.approx(100.0)
    assert thr > 0.9  # only thr>0.9 avoids firing -> holds to the profitable close

    # The OLD mean-trade-rate objective would instead pick a LOW threshold that
    # fires at minute 0 (mean rate 100 vs 1). Prove the two objectives disagree.
    def _mean_rate(thr_):
        reals = rfc._realizations(val, p_close >= thr_, baseline=False)
        return float(np.mean([r.rate for r in reals]))

    best_rate_thr = max(np.linspace(0.05, 0.95, 19), key=_mean_rate)
    assert best_rate_thr <= 0.9
    assert _mean_rate(best_rate_thr) == pytest.approx(100.0)
    # The selected (total-objective) threshold is NOT the mean-rate-optimal one.
    assert thr != pytest.approx(best_rate_thr)


def test_total_objective_picks_higher_total_over_higher_average():
    # Same trades but compare the chosen threshold's total objective against the
    # average-pnl-per-minute view: the selection maximizes the TOTAL.
    val = pd.DataFrame(
        {
            "trade_id": [0, 0, 1, 1],
            "timestamp": pd.to_datetime(
                ["2023-01-01T00:00", "2023-01-01T00:01",
                 "2023-01-02T00:00", "2023-01-02T00:01"],
                utc=True,
            ),
            "trade_pnl_rate_now": [5.0, 0.0, 5.0, 0.0],
            "net_close_pnl_now": [5.0, 12.0, 5.0, 12.0],
        }
    )
    p_close = np.array([0.9, 0.1, 0.9, 0.1])
    thr, obj = rfc.select_threshold(val, p_close)
    # terminal: 12/2+12/2 = 12 ; early: 5/1+5/1 = 10 -> total objective picks terminal.
    assert obj == pytest.approx(12.0)
    assert thr > 0.9


def test_insufficient_data_status():
    df = _synthetic_trades(n_trades=2, minutes=5)
    result = rfc.train_close_classifier(df, "BTCUSDT", SCALE_COLS, PASS_COLS, min_training_rows=250)
    assert result["valid"] is False
    assert result["status"] == rfc.STATUS_INSUFFICIENT_DATA


def test_test_split_not_used_for_hyperparameter_or_threshold_selection():
    # n_trades=30 -> train ids 0..14, validation 15..21, test 22..29. We mutate
    # ONLY the test-split trades; selection (RF params + threshold + validation
    # objective) must be byte-identical, while the held-out TEST evaluation moves.
    df1 = _synthetic_trades(n_trades=30, minutes=20, seed=3)
    df2 = df1.copy()
    test_mask = df2["trade_id"] >= 22
    assert test_mask.any()
    df2.loc[test_mask, "net_close_pnl_now"] = df2.loc[test_mask, "net_close_pnl_now"] * -1000.0
    df2.loc[test_mask, "trade_pnl_rate_now"] = df2.loc[test_mask, "trade_pnl_rate_now"] * -1000.0

    r1 = rfc.train_close_classifier(df1, "BTCUSDT", SCALE_COLS, PASS_COLS, min_training_rows=100, param_grid=FAST_GRID)
    r2 = rfc.train_close_classifier(df2, "BTCUSDT", SCALE_COLS, PASS_COLS, min_training_rows=100, param_grid=FAST_GRID)
    assert r1["valid"] and r2["valid"]
    assert r1["threshold"] == r2["threshold"]
    assert r1["selected_params"] == r2["selected_params"]

    m1, m2 = r1["metadata"], r2["metadata"]
    assert m1["validation_total_pnl_per_minute_objective"] == pytest.approx(
        m2["validation_total_pnl_per_minute_objective"]
    )
    assert m1["validation_trade_count"] == m2["validation_trade_count"]
    # The test evaluation reflects the mutated held-out trades (so it WAS isolated).
    assert m1["test_net_pnl"] != pytest.approx(m2["test_net_pnl"])


def test_per_symbol_artifacts_and_meta_are_separate(tmp_path):
    df_btc = _synthetic_trades(n_trades=30, minutes=20, seed=1)
    df_eth = _synthetic_trades(n_trades=30, minutes=20, seed=2)
    df_eth["symbol"] = "ETHUSDT"

    r_btc = rfc.train_close_classifier(df_btc, "BTCUSDT", SCALE_COLS, PASS_COLS, min_training_rows=100, param_grid=FAST_GRID)
    r_eth = rfc.train_close_classifier(df_eth, "ETHUSDT", SCALE_COLS, PASS_COLS, min_training_rows=100, param_grid=FAST_GRID)
    assert r_btc["valid"] and r_eth["valid"]
    assert r_btc["metadata"]["symbol"] == "BTCUSDT"
    assert r_eth["metadata"]["symbol"] == "ETHUSDT"

    d_btc = str(tmp_path / "BTCUSDT")
    d_eth = str(tmp_path / "ETHUSDT")
    rfc.save_artifacts(d_btc, r_btc)
    rfc.save_artifacts(d_eth, r_eth)

    _p_btc, s_btc = rfc.load_policy(d_btc)
    _p_eth, s_eth = rfc.load_policy(d_eth)
    assert s_btc == rfc.STATUS_OK and s_eth == rfc.STATUS_OK

    import json as _json
    with open(os.path.join(d_btc, rfc.METADATA_FILE)) as fh:
        meta_btc = _json.load(fh)
    with open(os.path.join(d_eth, rfc.METADATA_FILE)) as fh:
        meta_eth = _json.load(fh)
    assert meta_btc["symbol"] == "BTCUSDT"
    assert meta_eth["symbol"] == "ETHUSDT"
    assert "selected_rf_hyperparameters" in meta_btc
    assert "selected_rf_hyperparameters" in meta_eth

    # A symbol with no artifacts gets MISSING, never another symbol's model.
    missing, status = rfc.load_policy(str(tmp_path / "SOLUSDT"))
    assert missing is None and status == rfc.STATUS_MISSING_MODEL


def test_no_valid_policy_when_economic_gate_unmet():
    # An impossible minimum-validation-trade gate rejects every candidate, so the
    # symbol fails closed instead of promoting an ungated model.
    df = _synthetic_trades(n_trades=30, minutes=20, seed=5)
    result = rfc.train_close_classifier(
        df, "BTCUSDT", SCALE_COLS, PASS_COLS,
        min_training_rows=100, param_grid=FAST_GRID, min_validation_trades=10_000,
    )
    assert result["valid"] is False
    assert result["status"] == rfc.STATUS_NO_VALID_POLICY


def test_selected_metadata_contains_required_fields():
    df = _synthetic_trades(n_trades=30, minutes=20, seed=7)
    result = rfc.train_close_classifier(df, "BTCUSDT", SCALE_COLS, PASS_COLS, min_training_rows=100, param_grid=FAST_GRID)
    assert result["valid"] is True
    meta = result["metadata"]
    for key in (
        "selected_rf_hyperparameters",
        "selected_threshold",
        "validation_total_pnl_per_minute_objective",
        "validation_trade_count",
        "baseline_validation_objective",
        "baseline_improvement_required",
        "baseline_objective_epsilon",
        "test_total_pnl_per_minute_objective",
        "test_net_pnl",
        "test_max_drawdown",
        "test_turnover",
        "test_liquidation_count",
        "selection_objective_description",
    ):
        assert key in meta, key
    params = meta["selected_rf_hyperparameters"]
    assert set(params) == {"n_estimators", "max_depth", "min_samples_leaf", "max_features"}
    assert meta["baseline_improvement_required"] is True
    assert meta["baseline_objective_epsilon"] > 0


def _constant_ratio_trades(n_trades=30, minutes=8, seed=0):
    """Trades where net == holding_minutes at every close, so EVERY policy (early
    or terminal) has objective net/holding == 1.0 per trade.

    The total objective is therefore identical for all thresholds and exactly
    equals the continue-to-terminal baseline. Used to prove a baseline-tying
    candidate is rejected by the strict gate.
    """
    rows = []
    base = pd.Timestamp("2023-01-01", tz="UTC")
    for tid in range(n_trades):
        entry = base + pd.Timedelta(days=tid)
        for m in range(minutes):
            rows.append(
                {
                    "trade_id": tid,
                    "leg_id": f"{tid}:both",
                    "symbol": "BTCUSDT",
                    "side_code": 2,
                    "entry_timestamp": entry,
                    "timestamp": entry + pd.Timedelta(minutes=m),
                    "y": float(m),
                    "current_pnl": float(m),
                    "mdp_liq_risk_score": 0.0,
                    "season_pre_halving_2y": 1.0,
                    "season_post_halving_2y": 0.0,
                    "season_unknown": 0.0,
                    # Two classes so the RF can fit.
                    "close_label": int(m % 2 == 0),
                    "sample_weight": 1.0,
                    "trade_pnl_rate_now": 1.0,
                    # net == holding (pos+1) at every minute -> ratio always 1.0.
                    "net_close_pnl_now": float(m + 1),
                    "leg_liquidated_now": 0,
                    "fully_liquidated_now": 0,
                }
            )
    return pd.DataFrame(rows)


def test_baseline_tie_is_not_selected_under_strict_gate():
    # Every candidate/threshold objective ties the baseline exactly; the strict
    # gate (objective > baseline + epsilon) must reject all of them -> fail-closed.
    df = _constant_ratio_trades()
    result = rfc.train_close_classifier(
        df, "BTCUSDT", SCALE_COLS, PASS_COLS, min_training_rows=100, param_grid=FAST_GRID
    )
    assert result["valid"] is False
    assert result["status"] == rfc.STATUS_NO_VALID_POLICY
    assert result["baseline_improvement_required"] is True

    # Relaxing the gate (no baseline requirement) now allows a (tying) selection,
    # which confirms the tie was rejected *specifically* by the baseline gate.
    relaxed = rfc.train_close_classifier(
        df, "BTCUSDT", SCALE_COLS, PASS_COLS, min_training_rows=100,
        param_grid=FAST_GRID, require_baseline_improvement=False,
    )
    assert relaxed["valid"] is True


def test_invalid_result_does_not_write_or_overwrite_artifacts(tmp_path):
    # Train + persist a valid, deployable model first.
    good = rfc.train_close_classifier(
        _synthetic_trades(n_trades=30, minutes=20, seed=1), "BTCUSDT",
        SCALE_COLS, PASS_COLS, min_training_rows=100, param_grid=FAST_GRID,
    )
    assert good["valid"] is True
    model_dir = str(tmp_path / "BTCUSDT")
    rfc.save_artifacts(model_dir, good)
    model_path = os.path.join(model_dir, rfc.MODEL_FILE)
    before = open(model_path, "rb").read()

    # An insufficient-data (invalid) result must refuse to write and must NOT
    # overwrite the existing valid artifact.
    bad = rfc.train_close_classifier(
        _synthetic_trades(n_trades=2, minutes=5), "BTCUSDT",
        SCALE_COLS, PASS_COLS, min_training_rows=250,
    )
    assert bad["valid"] is False
    with pytest.raises(ValueError):
        rfc.save_artifacts(model_dir, bad)
    assert open(model_path, "rb").read() == before  # untouched

    # A no-valid-policy result is likewise refused, into a fresh dir (no files).
    nopolicy = rfc.train_close_classifier(
        _constant_ratio_trades(), "BTCUSDT", SCALE_COLS, PASS_COLS,
        min_training_rows=100, param_grid=FAST_GRID,
    )
    assert nopolicy["status"] == rfc.STATUS_NO_VALID_POLICY
    fresh_dir = str(tmp_path / "ETHUSDT")
    with pytest.raises(ValueError):
        rfc.save_artifacts(fresh_dir, nopolicy)
    assert not os.path.isfile(os.path.join(fresh_dir, rfc.MODEL_FILE))
