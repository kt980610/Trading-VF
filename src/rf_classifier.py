"""Per-symbol RF binary close classifier (spec sections 7, 9, 10).

Replaces the old ``RandomForestRegressor`` / ``predicted_continue_edge`` policy.

* Target is the binary ``close_label`` (1 = close now, 0 = continue).
* Live decision: ``p_close = predict_proba()[1]; close = p_close >= threshold``.
* Numeric features are imputed (train medians) then RobustScaler-transformed,
  fit on TRAIN ONLY. One-hot / code / id / binary columns are passed through
  unscaled. PDF/integral financial units are never rescaled outside this layer.
* For each symbol a small deterministic RF hyperparameter grid AND a
  close-probability threshold are jointly selected to MAXIMIZE the user economic
  objective ``sum(net_realized_pnl_i / max(holding_minutes_i, 1))`` on validation
  (net of fees/funding/slippage), behind economic safety gates. Ties break on
  lower drawdown, then turnover, then a simpler RF. The TEST split is never used
  for any selection.
* Each symbol has fully separate split + RF + hyperparameters + threshold +
  artifacts/meta; there is NO cross-symbol data, model or fallback.
"""

from __future__ import annotations

import json
import math
import os
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DECISION_CLOSE = "CLOSE"
DECISION_CONTINUE = "CONTINUE"

STATUS_OK = "ok"
STATUS_MISSING_MODEL = "missing_symbol_model"
STATUS_INSUFFICIENT_DATA = "insufficient_symbol_training_data"
STATUS_NO_SKLEARN = "sklearn_unavailable"
STATUS_NO_VALID_POLICY = "no_policy_passed_economic_gates"

MODEL_FILE = "rf_close_classifier.joblib"
SCALER_FILE = "feature_scaler.joblib"
SCHEMA_FILE = "feature_schema.json"
THRESHOLD_FILE = "close_threshold.json"
METADATA_FILE = "training_metadata.json"
# Portable, framework-free export so the live engine (e.g. the Rust runtime) can
# load the SAME model/scaler/threshold/schema without sklearn/joblib.
MODEL_JSON_FILE = "rf_close_classifier.json"

MODEL_VERSION = "rf_close_classifier_v1"


def _impute(frame: pd.DataFrame, cols: List[str], medians: Dict[str, float]) -> np.ndarray:
    out = np.empty((len(frame), len(cols)), dtype="float64")
    for j, c in enumerate(cols):
        raw = frame[c].to_numpy(dtype="float64") if c in frame.columns else np.full(len(frame), np.nan)
        med = float(medians.get(c, 0.0))
        out[:, j] = np.where(np.isfinite(raw), raw, med)
    return out


def _passthrough(frame: pd.DataFrame, cols: List[str]) -> np.ndarray:
    out = np.empty((len(frame), len(cols)), dtype="float64")
    for j, c in enumerate(cols):
        raw = frame[c].to_numpy(dtype="float64") if c in frame.columns else np.zeros(len(frame))
        out[:, j] = np.where(np.isfinite(raw), raw, 0.0)
    return out


@dataclass
class ClosePolicy:
    """Loaded close classifier: impute -> scale -> RF -> p_close >= threshold."""

    scale_cols: List[str]
    passthrough_cols: List[str]
    medians: Dict[str, float]
    threshold: float
    model: object = None
    scaler: object = None
    class_index: int = 1

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        scaled = self.scaler.transform(_impute(frame, self.scale_cols, self.medians)) if self.scale_cols else np.empty((len(frame), 0))
        passt = _passthrough(frame, self.passthrough_cols)
        return np.hstack([scaled, passt]) if passt.size or scaled.size else np.empty((len(frame), 0))

    def predict_proba_close(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None or len(frame) == 0:
            return np.zeros(len(frame))
        proba = self.model.predict_proba(self._matrix(frame))
        classes = list(self.model.classes_)
        idx = classes.index(1) if 1 in classes else (len(classes) - 1)
        return proba[:, idx]

    def decide(self, frame: pd.DataFrame) -> List[str]:
        p = self.predict_proba_close(frame)
        return [DECISION_CLOSE if pi >= self.threshold else DECISION_CONTINUE for pi in p]


def _time_ordered_trade_split(
    df: pd.DataFrame, val_fraction: float, test_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Single time-ordered, trade-group split; one trade never spans two splits."""
    if "trade_id" not in df.columns:
        n = len(df)
        c1 = int(n * (1 - val_fraction - test_fraction))
        c2 = int(n * (1 - test_fraction))
        return df.iloc[:c1], df.iloc[c1:c2], df.iloc[c2:]

    order_col = "entry_timestamp" if "entry_timestamp" in df.columns else "timestamp"
    trade_order = (
        df.groupby("trade_id")[order_col].min().sort_values().index.tolist()
        if order_col in df.columns
        else sorted(df["trade_id"].unique())
    )
    n = len(trade_order)
    c1 = int(n * (1 - val_fraction - test_fraction))
    c2 = int(n * (1 - test_fraction))
    train_ids = set(trade_order[:c1])
    val_ids = set(trade_order[c1:c2])
    test_ids = set(trade_order[c2:])
    return (
        df[df["trade_id"].isin(train_ids)].reset_index(drop=True),
        df[df["trade_id"].isin(val_ids)].reset_index(drop=True),
        df[df["trade_id"].isin(test_ids)].reset_index(drop=True),
    )


def trade_group_split(
    df: pd.DataFrame, val_fraction: float = 0.25, test_fraction: float = 0.25
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-ordered, trade-group split; one trade never spans two splits.

    When a ``train_window`` column is present with more than one window, the split
    is performed time-ordered WITHIN each window and concatenated, so every
    regime (and thus every ``halving_cycle_id``/"k" value) appears in train,
    validation and test instead of one regime monopolising a single split.
    """
    if "train_window" in df.columns and df["train_window"].nunique() > 1:
        trains, vals, tests = [], [], []
        for _w, g in df.groupby("train_window", sort=True):
            tr, va, te = _time_ordered_trade_split(
                g.reset_index(drop=True), val_fraction, test_fraction
            )
            trains.append(tr)
            vals.append(va)
            tests.append(te)
        return (
            pd.concat(trains, ignore_index=True),
            pd.concat(vals, ignore_index=True),
            pd.concat(tests, ignore_index=True),
        )
    return _time_ordered_trade_split(df, val_fraction, test_fraction)


# One resolved trade close. ``net`` is costs-inclusive net realized PnL,
# ``holding`` the holding duration in minutes (1-based close position in trade).
Realization = namedtuple(
    "Realization", "net rate holding order_key liquidated full_liquidated early"
)

# Close-probability thresholds scanned on validation.
_DEFAULT_THRESHOLDS = np.linspace(0.05, 0.95, 19)


def _objective(realizations: List["Realization"]) -> float:
    """User economic objective: SUM of ``net_realized_pnl_i / holding_minutes_i``.

    Net PnL already includes commission/funding/slippage/liquidation handling via
    the simulator's ``net_close_pnl_now``. This is a total (sum), never an average.
    """
    return float(sum(r.net / max(r.holding, 1) for r in realizations))


def _policy_metrics(realizations: List["Realization"]) -> Dict[str, float]:
    """Validation metrics for one (model, threshold) candidate."""
    n = len(realizations)
    ordered = sorted(realizations, key=lambda r: r.order_key)
    early = int(sum(r.early for r in realizations))
    return {
        "objective": _objective(realizations),
        "max_drawdown": _max_drawdown([r.net for r in ordered]),
        "turnover": early,
        "turnover_rate": (early / n) if n else 0.0,
        "n_trades": n,
        "liquidation_count": int(sum(r.liquidated for r in realizations)),
    }


def select_threshold(
    val_df: pd.DataFrame, p_close: np.ndarray, thresholds=None
) -> Tuple[float, float]:
    """Pick the threshold maximizing the validation total PnL/minute objective.

    Ties break on lower max drawdown, then lower turnover. Returns
    ``(threshold, objective)``.
    """
    if val_df is None or val_df.empty or "trade_id" not in val_df.columns:
        return 0.5, float("-inf")
    grid = _DEFAULT_THRESHOLDS if thresholds is None else thresholds
    best_thr, best_obj, best_key = 0.5, float("-inf"), None
    p = np.asarray(p_close)
    for thr in grid:
        reals = _realizations(val_df, p >= float(thr), baseline=False)
        m = _policy_metrics(reals)
        key = (-round(m["objective"], 9), m["max_drawdown"], m["turnover"], float(thr))
        if best_key is None or key < best_key:
            best_key, best_thr, best_obj = key, float(thr), m["objective"]
    return best_thr, best_obj


def _max_drawdown(pnls_in_order: List[float]) -> float:
    """Peak-to-trough drawdown of the cumulative realized-PnL equity curve."""
    equity = 0.0
    peak = 0.0
    mdd = 0.0
    for p in pnls_in_order:
        equity += float(p)
        peak = max(peak, equity)
        mdd = max(mdd, peak - equity)
    return float(mdd)


def _realize_one(group: pd.DataFrame, fire: np.ndarray, baseline: bool) -> "Realization":
    """Resolve one trade's close point and its costs-inclusive realization.

    * baseline -> close only at the terminal minute (continue everywhere else);
    * RF       -> close at the first minute ``fire`` is True, else the terminal.

    ``net`` is net of fees/funding/slippage (the simulator's ``net_close_pnl_now``)
    and ``holding`` is the holding duration in minutes (1-based close position in
    the trade). A trade contributes exactly once (no double-counting).
    """
    n = len(group)
    if baseline or not fire.any():
        pos = n - 1
        early = 0
    else:
        pos = int(np.argmax(fire))  # first True
        early = 1 if pos < n - 1 else 0
    row = group.iloc[pos]
    if "net_close_pnl_now" in group.columns:
        net = float(row["net_close_pnl_now"])
    else:  # fallback only when the simulator net column is absent
        net = float(row.get("trade_pnl_rate_now", 0.0)) * float(pos + 1)
    rate = float(row.get("trade_pnl_rate_now", net))
    holding = pos + 1
    order_key = row["timestamp"] if "timestamp" in group.columns else pos
    upto = group.iloc[: pos + 1]
    if "leg_liquidated_now" in group.columns:
        liq_occurred = int(upto["leg_liquidated_now"].fillna(0).astype(int).max() or 0)
    else:
        liq_occurred = 0
    if "fully_liquidated_now" in group.columns:
        full_liq = int(row.get("fully_liquidated_now", 0) or 0)
    else:
        full_liq = 0
    return Realization(net, rate, holding, order_key, liq_occurred, full_liq, early)


def _realizations(df: pd.DataFrame, fire: np.ndarray, baseline: bool) -> List["Realization"]:
    """Per-trade realizations for a fire mask aligned to ``df`` row order."""
    tmp = df.copy()
    tmp["_fire"] = np.asarray(fire)
    out: List["Realization"] = []
    for _tid, g in tmp.groupby("trade_id", sort=False):
        g = g.sort_values("timestamp") if "timestamp" in g.columns else g
        out.append(_realize_one(g, g["_fire"].to_numpy(), baseline))
    return out


# Small, deterministic RF candidate grid (kept CPU-feasible per coin). The
# config-driven (n_estimators, max_depth) point is always prepended in
# ``_candidate_grid`` so config still influences selection.
DEFAULT_RF_PARAM_GRID = [
    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 5, "max_features": "sqrt"},
    {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"n_estimators": 400, "max_depth": 16, "min_samples_leaf": 5, "max_features": 0.5},
    {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 2, "max_features": "log2"},
]


def _candidate_grid(base_n_estimators, base_max_depth, grid=None) -> List[Dict[str, object]]:
    """Deterministic, de-duplicated candidate list.

    When ``grid`` is provided it is used verbatim (de-duplicated). Otherwise the
    config-driven ``(n_estimators, max_depth)`` base point is prepended to the
    default grid so config still influences selection.
    """
    if grid is not None:
        source = list(grid)
    else:
        base = {
            "n_estimators": int(base_n_estimators or 300),
            "max_depth": (int(base_max_depth) if base_max_depth else None),
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        }
        source = [base, *DEFAULT_RF_PARAM_GRID]
    uniq: List[Dict[str, object]] = []
    seen = set()
    for p in source:
        key = (p["n_estimators"], p["max_depth"], p["min_samples_leaf"], str(p["max_features"]))
        if key not in seen:
            seen.add(key)
            uniq.append(dict(p))
    return uniq


def _rf_complexity(params: Dict[str, object]) -> Tuple:
    """Tie-breaker key; lower = simpler RF (preferred)."""
    depth = params["max_depth"] if params["max_depth"] else 10 ** 9
    mf_rank = {"log2": 0, "sqrt": 1}.get(params["max_features"], 2)
    return (
        int(params["n_estimators"]),
        int(depth),
        -int(params["min_samples_leaf"]),
        mf_rank,
    )


def evaluate_close_policy(policy: "ClosePolicy", test_df: pd.DataFrame) -> Dict[str, object]:
    """Evaluate a trained policy on the TEST split only (spec sections 8 & 10).

    Nothing is (re)fit here: the model, scaler and threshold come from the policy
    that was trained/selected on the train+validation splits. The baseline is the
    explicit "continue every minute, force close/liquidation only at the terminal
    minute" policy applied to the SAME test trades. All PnL is net of
    fees/funding/slippage, computed per-trade with no double-counting.
    """
    empty = {
        "precision": 0.0, "recall": 0.0, "close_count": 0, "continue_count": 0,
        "mean_trade_pnl_rate": 0.0, "total_net_pnl": 0.0,
        "total_net_pnl_per_minute": 0.0, "turnover": 0, "turnover_rate": 0.0,
        "max_drawdown": 0.0,
        "liquidation_count": 0, "full_liquidation_count": 0,
        "baseline_total_net_pnl": 0.0, "baseline_total_net_pnl_per_minute": 0.0,
        "baseline_max_drawdown": 0.0,
        "baseline_liquidation_count": 0, "baseline_full_liquidation_count": 0,
        "baseline_pnl_difference": 0.0, "n_test_trades": 0, "n_test_minute_rows": 0,
    }
    if test_df is None or len(test_df) == 0 or "trade_id" not in test_df.columns:
        return empty

    df = test_df.copy()
    if "timestamp" in df.columns:
        df = df.sort_values(["trade_id", "timestamp"]).reset_index(drop=True)
    else:
        df = df.sort_values(["trade_id"]).reset_index(drop=True)

    p_close = policy.predict_proba_close(df)
    fire_all = p_close >= policy.threshold
    df["_fire"] = fire_all

    # Per-minute classification metrics (predicted close vs close_label).
    y_pred = fire_all.astype(int)
    close_count = int(y_pred.sum())
    continue_count = int(len(y_pred) - close_count)
    precision = recall = 0.0
    if "close_label" in df.columns:
        y_true = df["close_label"].astype(int).to_numpy()
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

    rf_reals = _realizations(df, fire_all, baseline=False)
    base_reals = _realizations(df, fire_all, baseline=True)

    def _agg(reals):
        ordered = sorted(reals, key=lambda r: r.order_key)
        total = float(sum(r.net for r in ordered))
        mdd = _max_drawdown([r.net for r in ordered])
        liq = int(sum(r.liquidated for r in ordered))
        full_liq = int(sum(r.full_liquidated for r in ordered))
        mean_rate = float(np.mean([r.rate for r in ordered])) if ordered else 0.0
        early = int(sum(r.early for r in ordered))
        return total, mdd, liq, full_liq, mean_rate, early

    rf_total, rf_mdd, rf_liq, rf_full, rf_mean_rate, rf_early = _agg(rf_reals)
    base_total, base_mdd, base_liq, base_full, _, _ = _agg(base_reals)
    n_trades = len(rf_reals)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "close_count": close_count,
        "continue_count": continue_count,
        "mean_trade_pnl_rate": rf_mean_rate,
        "total_net_pnl": rf_total,
        "total_net_pnl_per_minute": _objective(rf_reals),
        "turnover": rf_early,
        "turnover_rate": (rf_early / n_trades) if n_trades else 0.0,
        "max_drawdown": rf_mdd,
        "liquidation_count": rf_liq,
        "full_liquidation_count": rf_full,
        "baseline_total_net_pnl": base_total,
        "baseline_total_net_pnl_per_minute": _objective(base_reals),
        "baseline_max_drawdown": base_mdd,
        "baseline_liquidation_count": base_liq,
        "baseline_full_liquidation_count": base_full,
        "baseline_pnl_difference": float(rf_total - base_total),
        "n_test_trades": int(n_trades),
        "n_test_minute_rows": int(len(df)),
    }


SELECTION_OBJECTIVE_DESCRIPTION = (
    "Maximize validation total PnL/minute = sum(net_realized_pnl_i / "
    "max(holding_minutes_i, 1)) (net of fees/funding/slippage). A candidate must "
    "STRICTLY beat the continue-to-terminal baseline (objective > baseline + "
    "objective_epsilon); a baseline-tying policy is never selected. Ties (between "
    "passing candidates) break on lower max drawdown, then lower turnover, then a "
    "simpler RF. RF hyperparameters and the close-probability threshold are "
    "selected jointly on train+validation only; the test split is held out."
)


def train_close_classifier(
    df: pd.DataFrame,
    symbol: str,
    scale_cols: List[str],
    passthrough_cols: List[str],
    min_training_rows: int = 250,
    min_trades: int = 5,
    n_estimators: int = 300,
    max_depth: int = 0,
    n_jobs: int = 4,
    random_state: int = 42,
    close_epsilon: float = 0.0,
    season_seed: int = 0,
    param_grid: Optional[List[Dict[str, object]]] = None,
    thresholds=None,
    min_validation_trades: int = 3,
    max_turnover_rate: float = 1.0,
    require_baseline_improvement: bool = True,
    objective_epsilon: float = 1e-9,
) -> Dict[str, object]:
    """Train a per-symbol binary close classifier; returns a result dict.

    RF hyperparameters (small deterministic grid) AND the close-probability
    threshold are selected JOINTLY to maximize the validation total PnL/minute
    objective, behind economic safety gates (min validation trades, no
    liquidation, finite objective, baseline improvement, turnover ceiling). The
    TEST split is never touched during selection; it is only used for the final
    evaluation. Everything is per-symbol with no cross-symbol fallback.
    """
    if len(df) < min_training_rows:
        return {"symbol": symbol, "valid": False, "status": STATUS_INSUFFICIENT_DATA, "n_rows": int(len(df))}
    if "trade_id" in df.columns and df["trade_id"].nunique() < min_trades:
        return {"symbol": symbol, "valid": False, "status": STATUS_INSUFFICIENT_DATA, "n_rows": int(len(df))}

    try:
        from sklearn.ensemble import RandomForestClassifier  # type: ignore
        from sklearn.preprocessing import RobustScaler  # type: ignore
    except ImportError:
        return {"symbol": symbol, "valid": False, "status": STATUS_NO_SKLEARN, "n_rows": int(len(df))}

    train_df, val_df, test_df = trade_group_split(df)
    if train_df.empty or train_df["close_label"].nunique() < 2:
        return {"symbol": symbol, "valid": False, "status": STATUS_INSUFFICIENT_DATA, "n_rows": int(len(df))}
    if val_df.empty or "trade_id" not in val_df.columns:
        return {"symbol": symbol, "valid": False, "status": STATUS_INSUFFICIENT_DATA, "n_rows": int(len(df))}

    medians = {c: float(np.nanmedian(train_df[c].to_numpy(dtype="float64"))) if c in train_df.columns else 0.0 for c in scale_cols}
    for c in medians:
        if not np.isfinite(medians[c]):
            medians[c] = 0.0

    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(_impute(train_df, scale_cols, medians)) if scale_cols else np.empty((len(train_df), 0))
    x_pass = _passthrough(train_df, passthrough_cols)
    x_train = np.hstack([x_scaled, x_pass]) if (x_scaled.size or x_pass.size) else np.empty((len(train_df), 0))
    y_train = train_df["close_label"].to_numpy(dtype="int64")
    w_train = train_df.get("sample_weight", pd.Series(1.0, index=train_df.index)).to_numpy(dtype="float64")

    threshold_grid = _DEFAULT_THRESHOLDS if thresholds is None else thresholds
    # Baseline = "continue to terminal" on the SAME validation trades.
    baseline_reals = _realizations(val_df, np.zeros(len(val_df), dtype=bool), baseline=True)
    baseline_objective = _objective(baseline_reals)

    candidates = _candidate_grid(n_estimators, max_depth, param_grid)
    best: Optional[Dict[str, object]] = None
    n_considered = 0
    for params in candidates:
        model = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            random_state=random_state,
            class_weight="balanced",
            n_jobs=n_jobs,
        )
        model.fit(x_train, y_train, sample_weight=w_train)
        cand_policy = ClosePolicy(
            scale_cols=list(scale_cols), passthrough_cols=list(passthrough_cols),
            medians=medians, threshold=0.5, model=model, scaler=scaler,
        )
        val_p = cand_policy.predict_proba_close(val_df)
        for thr in threshold_grid:
            reals = _realizations(val_df, val_p >= float(thr), baseline=False)
            m = _policy_metrics(reals)
            n_considered += 1
            # Economic safety gates: a failing candidate is never selected.
            if m["n_trades"] < min_validation_trades:
                continue
            if m["liquidation_count"] > 0:
                continue
            if not math.isfinite(m["objective"]):
                continue
            # STRICT baseline gate: a candidate must beat "continue to terminal"
            # by a small positive tolerance. A policy that merely ties the
            # baseline (e.g. never fires / closes only at the terminal minute) is
            # NOT an improvement and must never be selected.
            if require_baseline_improvement and m["objective"] <= baseline_objective + objective_epsilon:
                continue
            if m["turnover_rate"] > max_turnover_rate:
                continue
            key = (
                -round(m["objective"], 9),
                m["max_drawdown"],
                m["turnover"],
                _rf_complexity(params),
                float(thr),
            )
            if best is None or key < best["key"]:
                best = {
                    "key": key,
                    "params": dict(params),
                    "threshold": float(thr),
                    "model": model,
                    "metrics": dict(m),
                }

    if best is None:
        return {
            "symbol": symbol,
            "valid": False,
            "status": STATUS_NO_VALID_POLICY,
            "n_rows": int(len(df)),
            "n_candidates_considered": int(n_considered),
            "baseline_validation_objective": float(baseline_objective),
            "baseline_improvement_required": bool(require_baseline_improvement),
            "baseline_objective_epsilon": float(objective_epsilon),
        }

    model = best["model"]
    threshold = float(best["threshold"])
    selected_params = dict(best["params"])
    val_metrics = dict(best["metrics"])

    policy = ClosePolicy(
        scale_cols=list(scale_cols), passthrough_cols=list(passthrough_cols),
        medians=medians, threshold=threshold, model=model, scaler=scaler,
    )

    # Test-split-only evaluation vs the explicit continue-to-terminal baseline.
    evaluation = evaluate_close_policy(policy, test_df) if not test_df.empty else {}

    schema = {
        "model_version": MODEL_VERSION,
        "symbol": symbol,
        "scale_cols": list(scale_cols),
        "passthrough_cols": list(passthrough_cols),
        "final_feature_order": list(scale_cols) + list(passthrough_cols),
        "imputer": {"strategy": "median", "medians": medians},
        "scaler": {"type": "robust", "file": SCALER_FILE},
    }
    metadata = {
        "symbol": symbol,
        "model_version": MODEL_VERSION,
        "n_rows": int(len(df)),
        "n_minute_rows": int(len(df)),
        "n_trades_train": int(train_df["trade_id"].nunique()) if "trade_id" in train_df else None,
        "n_trades_validation": int(val_df["trade_id"].nunique()) if "trade_id" in val_df and not val_df.empty else 0,
        "n_trades_test": int(test_df["trade_id"].nunique()) if "trade_id" in test_df and not test_df.empty else 0,
        "close_rate": float(np.mean(y_train)) if len(y_train) else 0.0,
        "close_epsilon": float(close_epsilon),
        "selected_season_seed": int(season_seed),
        "random_state": int(random_state),
        # --- Selection (train + validation only) ---
        "selected_rf_hyperparameters": selected_params,
        "selected_threshold": float(threshold),
        "close_probability_threshold": float(threshold),
        "validation_total_pnl_per_minute_objective": float(val_metrics["objective"]),
        "validation_trade_count": int(val_metrics["n_trades"]),
        "validation_max_drawdown": float(val_metrics["max_drawdown"]),
        "validation_turnover": int(val_metrics["turnover"]),
        "validation_liquidation_count": int(val_metrics["liquidation_count"]),
        "baseline_validation_objective": float(baseline_objective),
        "baseline_improvement_required": bool(require_baseline_improvement),
        "baseline_objective_epsilon": float(objective_epsilon),
        "n_candidates_considered": int(n_considered),
        "selection_objective_description": SELECTION_OBJECTIVE_DESCRIPTION,
    }
    metadata.update(evaluation)
    # Explicit spec-named TEST metrics (held-out; not used for any selection).
    metadata["test_total_pnl_per_minute_objective"] = evaluation.get("total_net_pnl_per_minute")
    metadata["test_net_pnl"] = evaluation.get("total_net_pnl")
    metadata["test_max_drawdown"] = evaluation.get("max_drawdown")
    metadata["test_turnover"] = evaluation.get("turnover")
    metadata["test_liquidation_count"] = evaluation.get("liquidation_count")

    # In-sample PnL of the selected policy applied to the TRAINING split. This is
    # diagnostic only (NEVER used for selection); it shows what the model would
    # have earned on the data it was fit on (expected to be optimistic).
    train_eval = evaluate_close_policy(policy, train_df) if not train_df.empty else {}
    metadata["train_total_pnl_per_minute_objective"] = train_eval.get("total_net_pnl_per_minute")
    metadata["train_net_pnl"] = train_eval.get("total_net_pnl")
    metadata["train_baseline_net_pnl"] = train_eval.get("baseline_total_net_pnl")
    metadata["train_max_drawdown"] = train_eval.get("max_drawdown")
    metadata["train_turnover"] = train_eval.get("turnover")
    metadata["train_liquidation_count"] = train_eval.get("liquidation_count")
    metadata["train_trades_evaluated"] = train_eval.get("n_test_trades")
    return {
        "symbol": symbol,
        "valid": True,
        "status": STATUS_OK,
        "n_rows": int(len(df)),
        "model": model,
        "scaler": scaler,
        "policy": policy,
        "schema": schema,
        "threshold": float(threshold),
        "selected_params": selected_params,
        "metadata": metadata,
        "val_df": val_df,
        "test_df": test_df,
    }


def _export_classifier_tree(estimator, class1_idx: int) -> dict:
    """Export one sklearn classifier tree with leaf P(class=1) as the node value."""
    t = estimator.tree_
    proba1: List[float] = []
    for node in range(t.node_count):
        counts = t.value[node][0]
        total = float(counts.sum())
        proba1.append(float(counts[class1_idx] / total) if total > 0 else 0.0)
    return {
        "children_left": t.children_left.tolist(),
        "children_right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "threshold": t.threshold.tolist(),
        "value": proba1,
    }


def export_model_json(result: Dict[str, object]) -> dict:
    """Framework-free model export: RF (leaf P(close)) + RobustScaler + schema.

    The averaged per-tree leaf probability equals ``predict_proba()[:,1]``; the
    feature vector order is ``scale_cols + passthrough_cols`` with RobustScaler
    applied to the scaled block only (median imputation first).
    """
    model = result["model"]
    scaler = result["scaler"]
    schema = result["schema"]  # type: ignore[assignment]
    scale_cols = list(schema["scale_cols"])  # type: ignore[index]
    class1_idx = list(model.classes_).index(1) if 1 in list(model.classes_) else (len(model.classes_) - 1)
    center = list(map(float, getattr(scaler, "center_", [0.0] * len(scale_cols)))) if scale_cols else []
    scale = list(map(float, getattr(scaler, "scale_", [1.0] * len(scale_cols)))) if scale_cols else []
    return {
        "type": "random_forest_classifier",
        "model_version": MODEL_VERSION,
        "symbol": schema["symbol"],  # type: ignore[index]
        "threshold": float(result["threshold"]),
        "final_feature_order": list(schema["final_feature_order"]),  # type: ignore[index]
        "scale_cols": scale_cols,
        "passthrough_cols": list(schema["passthrough_cols"]),  # type: ignore[index]
        "imputer": {"strategy": "median", "medians": dict(schema["imputer"]["medians"])},  # type: ignore[index]
        "scaler": {"type": "robust", "center": center, "scale": scale},
        "trees": [_export_classifier_tree(est, class1_idx) for est in model.estimators_],
    }


def save_artifacts(model_dir: str, result: Dict[str, object]) -> Dict[str, str]:
    # Fail-closed: only a fully valid, deployable result may write model/scaler/
    # policy artifacts the live loader can read. An invalid result (no valid
    # policy, insufficient data, sklearn missing, ...) must never create OR
    # overwrite artifacts, so an existing valid model is left untouched. (A
    # caller may still persist a separate diagnostics/report file itself.)
    if (
        not result.get("valid")
        or result.get("status") != STATUS_OK
        or result.get("model") is None
        or result.get("scaler") is None
        or "schema" not in result
    ):
        raise ValueError(
            "refusing to write RF artifacts for a non-deployable result "
            f"(symbol={result.get('symbol')}, status={result.get('status')}, "
            f"valid={result.get('valid')}); existing artifacts left untouched"
        )

    import joblib  # type: ignore

    os.makedirs(model_dir, exist_ok=True)
    paths = {
        "model": os.path.join(model_dir, MODEL_FILE),
        "scaler": os.path.join(model_dir, SCALER_FILE),
        "schema": os.path.join(model_dir, SCHEMA_FILE),
        "threshold": os.path.join(model_dir, THRESHOLD_FILE),
        "metadata": os.path.join(model_dir, METADATA_FILE),
        "model_json": os.path.join(model_dir, MODEL_JSON_FILE),
    }
    joblib.dump(result["model"], paths["model"])
    joblib.dump(result["scaler"], paths["scaler"])
    with open(paths["schema"], "w", encoding="utf-8") as fh:
        json.dump(result["schema"], fh, indent=2)
    with open(paths["threshold"], "w", encoding="utf-8") as fh:
        json.dump({"close_probability_threshold": float(result["threshold"])}, fh, indent=2)
    with open(paths["metadata"], "w", encoding="utf-8") as fh:
        json.dump(result["metadata"], fh, indent=2)
    with open(paths["model_json"], "w", encoding="utf-8") as fh:
        json.dump(export_model_json(result), fh)
    return paths


def load_policy(model_dir: str) -> Tuple[Optional[ClosePolicy], str]:
    """Load a per-symbol close policy; (None, status) when missing/mismatched."""
    model_path = os.path.join(model_dir, MODEL_FILE)
    schema_path = os.path.join(model_dir, SCHEMA_FILE)
    thr_path = os.path.join(model_dir, THRESHOLD_FILE)
    if not (os.path.isfile(model_path) and os.path.isfile(schema_path)):
        return None, STATUS_MISSING_MODEL
    try:
        import joblib  # type: ignore
    except ImportError:
        return None, STATUS_NO_SKLEARN

    with open(schema_path, "r", encoding="utf-8") as fh:
        schema = json.load(fh)
    threshold = 0.5
    if os.path.isfile(thr_path):
        with open(thr_path, "r", encoding="utf-8") as fh:
            threshold = float(json.load(fh).get("close_probability_threshold", 0.5))
    model = joblib.load(model_path)
    scaler_path = os.path.join(model_dir, SCALER_FILE)
    scaler = joblib.load(scaler_path) if os.path.isfile(scaler_path) else None
    return (
        ClosePolicy(
            scale_cols=list(schema.get("scale_cols", [])),
            passthrough_cols=list(schema.get("passthrough_cols", [])),
            medians=dict(schema.get("imputer", {}).get("medians", {})),
            threshold=threshold,
            model=model,
            scaler=scaler,
        ),
        STATUS_OK,
    )
