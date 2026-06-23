"""Per-symbol Random Forest close/continue policy (spec sections 10, 15, 16).

Each symbol trains its own model on its own rows only. The forest is serialized
to a framework-independent JSON (``rf_close_decision.json``) so inference can run
on the server without scikit-learn. A deterministic baseline policy is used as a
fallback when a symbol has no model or too few training rows.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import feature_scaling as fs
from .rf_dataset import TARGET_COLUMN, model_feature_columns

DECISION_CONTINUE = "CONTINUE"
DECISION_CLOSE = "CLOSE"
REASON_NOT_ENOUGH_ROWS = "not_enough_training_rows"
REASON_NO_SKLEARN = "sklearn_unavailable"


def _export_tree(tree) -> dict:
    t = tree.tree_
    return {
        "children_left": t.children_left.tolist(),
        "children_right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "threshold": t.threshold.tolist(),
        "value": [float(v[0][0]) for v in t.value],
    }


def _predict_tree(tree: dict, x: np.ndarray) -> np.ndarray:
    children_left = tree["children_left"]
    children_right = tree["children_right"]
    feature = tree["feature"]
    threshold = tree["threshold"]
    value = tree["value"]

    out = np.empty(x.shape[0], dtype="float64")
    for i in range(x.shape[0]):
        node = 0
        while children_left[node] != -1:
            if x[i, feature[node]] <= threshold[node]:
                node = children_left[node]
            else:
                node = children_right[node]
        out[i] = value[node]
    return out


@dataclass
class RFPredictor:
    """Loads a JSON forest and predicts the policy-improvement value.

    When a fitted ``scaler`` is attached the same impute -> transform -> scale
    pipeline used in training is applied before inference, so training and
    serving (including the Rust live engine) share one feature space.
    """

    features: List[str]
    trees: List[dict]
    threshold: float = 0.0
    scaler: Optional[dict] = None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.scaler is not None:
            x = fs.transform_frame(frame, self.scaler)
        else:
            x = frame.reindex(columns=self.features).fillna(0.0).to_numpy(dtype="float64")
        if not self.trees:
            return np.zeros(x.shape[0])
        preds = np.zeros(x.shape[0], dtype="float64")
        for tree in self.trees:
            preds += _predict_tree(tree, x)
        return preds / len(self.trees)

    def decide(self, frame: pd.DataFrame) -> List[str]:
        values = self.predict(frame)
        return [DECISION_CONTINUE if v > self.threshold else DECISION_CLOSE for v in values]


@dataclass
class BaselineModel:
    """Deterministic fallback: continue when the summed return-edges are positive."""

    threshold: float = 0.0

    @staticmethod
    def predict(frame: pd.DataFrame) -> np.ndarray:
        long_edge = frame.get("LongEdge_Return", pd.Series(0.0, index=frame.index)).fillna(0.0)
        short_edge = frame.get("ShortEdge_Return", pd.Series(0.0, index=frame.index)).fillna(0.0)
        return (long_edge + short_edge).to_numpy(dtype="float64")

    def decide(self, frame: pd.DataFrame) -> List[str]:
        values = self.predict(frame)
        return [DECISION_CONTINUE if v > self.threshold else DECISION_CLOSE for v in values]


def chronological_split(df: pd.DataFrame, test_fraction: float = 0.25) -> tuple:
    """Chronological (not random) train/test split (spec section 15)."""
    df = df.sort_values("timestamp").reset_index(drop=True) if "timestamp" in df.columns else df
    n = len(df)
    cut = int(n * (1.0 - test_fraction))
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def train_rf(
    df: pd.DataFrame,
    symbol: str,
    min_training_rows: int = 250,
    n_estimators: int = 200,
    max_depth: int = 0,
    random_state: int = 42,
    decision_threshold: float = 0.0,
    include_optional: bool = False,
    scaling_enabled: bool = True,
    scaler_type: str = "robust",
    imputer: str = "median",
) -> Dict[str, object]:
    """Train a symbol-specific RF; returns a result dict (valid flag + artifacts).

    Features are imputed/transformed/scaled with a per-symbol scaler fit on the
    TRAIN split only; validation/test rows are only transformed (no leakage).
    """
    if len(df) < min_training_rows:
        return {"symbol": symbol, "valid": False, "reason": REASON_NOT_ENOUGH_ROWS, "n_rows": int(len(df))}

    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore
    except ImportError:
        return {"symbol": symbol, "valid": False, "reason": REASON_NO_SKLEARN, "n_rows": int(len(df))}

    features = model_feature_columns(df, include_optional)

    train_df, test_df = chronological_split(df)

    scaler: Optional[dict] = None
    if scaling_enabled:
        scaler = fs.fit_scaler(train_df, features, scaler_type=scaler_type, imputer=imputer)
        x_train = fs.transform_frame(train_df, scaler)
    else:
        x_train = train_df.reindex(columns=features).fillna(0.0).to_numpy(dtype="float64")
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype="float64")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=(max_depth or None),
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    model_json = {
        "type": "random_forest_regressor",
        "n_estimators": int(n_estimators),
        "features": features,
        "threshold": float(decision_threshold),
        "trees": [_export_tree(est) for est in model.estimators_],
    }
    metadata = {
        "symbol": symbol,
        "n_rows": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "random_state": int(random_state),
        "decision_threshold": float(decision_threshold),
        "scaling_enabled": bool(scaling_enabled),
        "scaler_type": scaler_type if scaling_enabled else None,
        "feature_importances": dict(zip(features, [float(v) for v in model.feature_importances_])),
    }
    return {
        "symbol": symbol,
        "valid": True,
        "reason": None,
        "n_rows": int(len(df)),
        "model_json": model_json,
        "features": features,
        "metadata": metadata,
        "scaler": scaler,
        "test_df": test_df,
    }


def predictor_from_json(model_json: dict, scaler: Optional[dict] = None) -> RFPredictor:
    return RFPredictor(
        features=list(model_json["features"]),
        trees=list(model_json["trees"]),
        threshold=float(model_json.get("threshold", 0.0)),
        scaler=scaler,
    )


def save_artifacts(
    model_dir: str,
    model_json: dict,
    features: List[str],
    metadata: dict,
    scaler: Optional[dict] = None,
) -> Dict[str, str]:
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "rf_close_decision.json")
    features_path = os.path.join(model_dir, "rf_close_decision_features.json")
    metadata_path = os.path.join(model_dir, "rf_model_metadata.json")

    with open(model_path, "w", encoding="utf-8") as fh:
        json.dump(model_json, fh)
    with open(features_path, "w", encoding="utf-8") as fh:
        json.dump(fs.features_metadata(features, scaler), fh, indent=2)
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    paths = {"model": model_path, "features": features_path, "metadata": metadata_path}
    if scaler is not None:
        paths.update(fs.save_artifacts(model_dir, scaler))
    return paths


def load_artifacts(model_dir: str) -> Optional[RFPredictor]:
    model_path = os.path.join(model_dir, "rf_close_decision.json")
    if not os.path.isfile(model_path):
        return None
    with open(model_path, "r", encoding="utf-8") as fh:
        model_json = json.load(fh)
    scaler = fs.load_scaler(model_dir)
    return predictor_from_json(model_json, scaler=scaler)


def get_policy(model_dir: str, fallback_threshold: float = 0.0):
    """Return the promoted RF predictor for a symbol, else the baseline model."""
    predictor = load_artifacts(model_dir)
    if predictor is not None:
        return predictor
    return BaselineModel(threshold=fallback_threshold)
