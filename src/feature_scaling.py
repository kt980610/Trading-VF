"""Per-symbol feature imputation + transform + scaling (spec sections 8-12).

Pipeline order (must match the Rust live engine exactly):

    1. impute missing values with the TRAIN median (raw space)
    2. apply a per-feature transform (log1p / sqrt / none)
    3. scale with a RobustScaler (median / IQR) or StandardScaler (mean / std)

Scalers are fit on TRAIN rows only and merely transform validation/test/live
rows (no leakage). Raw price levels are excluded from the ML feature set. The
MDP value math never uses scaling.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

EPSILON = 1e-12

# Volume-level magnitudes (positive, heavy-tailed) -> log1p before scaling.
VOLUME_LEVEL_FEATURES = {
    "predicted_daily_volume",
    "previous_day_real_volume",
    "current_minute_volume",
    "intraday_volume_so_far",
    "last_5m_volume",
    "last_15m_volume",
    "last_60m_volume",
    "previous_day_volume",
    "last_7d_mean_volume",
    "last_30d_mean_volume",
}


def transform_name(feat: str) -> str:
    """Classify the positive-skew transform for a feature name."""
    if feat in VOLUME_LEVEL_FEATURES:
        return "log1p"
    if feat.endswith("_count"):
        return "log1p"
    if "Var" in feat or feat.endswith("var_volume"):
        return "log1p"
    return "none"


def apply_transform(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype="float64")
    if name == "log1p":
        return np.log1p(np.maximum(arr, 0.0))
    if name == "sqrt":
        return np.sqrt(np.maximum(arr, 0.0))
    return arr


def _column(df: pd.DataFrame, feat: str) -> np.ndarray:
    if feat in df.columns:
        return df[feat].to_numpy(dtype="float64")
    return np.full(len(df), np.nan, dtype="float64")


def fit_scaler(
    df_train: pd.DataFrame,
    features: List[str],
    scaler_type: str = "robust",
    imputer: str = "median",
) -> Dict[str, object]:
    """Fit imputer medians, per-feature transforms and scaler center/scale."""
    medians: Dict[str, float] = {}
    transforms: Dict[str, str] = {}
    center: Dict[str, float] = {}
    scale: Dict[str, float] = {}

    for feat in features:
        raw = _column(df_train, feat)
        finite = raw[np.isfinite(raw)]
        med = float(np.median(finite)) if finite.size else 0.0
        medians[feat] = med

        imputed = np.where(np.isfinite(raw), raw, med)
        tname = transform_name(feat)
        transforms[feat] = tname
        t = apply_transform(tname, imputed)

        if scaler_type == "standard":
            c = float(np.mean(t)) if t.size else 0.0
            s = float(np.std(t)) if t.size else 1.0
        else:  # robust
            c = float(np.median(t)) if t.size else 0.0
            q75 = float(np.percentile(t, 75)) if t.size else 0.0
            q25 = float(np.percentile(t, 25)) if t.size else 0.0
            s = q75 - q25
        if not np.isfinite(s) or abs(s) < EPSILON:
            s = 1.0
        center[feat] = c
        scale[feat] = s

    return {
        "type": scaler_type,
        "imputer": imputer,
        "features": list(features),
        "transforms": transforms,
        "medians": medians,
        "center": center,
        "scale": scale,
    }


def transform_frame(df: pd.DataFrame, scaler: Dict[str, object]) -> np.ndarray:
    """Apply impute -> transform -> scale to ``df`` in the scaler feature order."""
    features: List[str] = list(scaler["features"])  # type: ignore[index]
    medians = scaler["medians"]  # type: ignore[index]
    transforms = scaler["transforms"]  # type: ignore[index]
    center = scaler["center"]  # type: ignore[index]
    scale = scaler["scale"]  # type: ignore[index]

    out = np.empty((len(df), len(features)), dtype="float64")
    for j, feat in enumerate(features):
        raw = _column(df, feat)
        med = float(medians.get(feat, 0.0))
        imputed = np.where(np.isfinite(raw), raw, med)
        t = apply_transform(str(transforms.get(feat, "none")), imputed)
        c = float(center.get(feat, 0.0))
        s = float(scale.get(feat, 1.0)) or 1.0
        out[:, j] = (t - c) / s
    return out


def save_artifacts(model_dir: str, scaler: Dict[str, object]) -> Dict[str, str]:
    """Write feature_imputer.json and feature_scaler.json."""
    os.makedirs(model_dir, exist_ok=True)
    imputer_path = os.path.join(model_dir, "feature_imputer.json")
    scaler_path = os.path.join(model_dir, "feature_scaler.json")

    imputer_doc = {
        "strategy": scaler.get("imputer", "median"),
        "features": scaler["features"],
        "medians": scaler["medians"],
    }
    scaler_doc = {
        "type": scaler.get("type", "robust"),
        "features": scaler["features"],
        "transforms": scaler["transforms"],
        "center": scaler["center"],
        "scale": scaler["scale"],
    }
    with open(imputer_path, "w", encoding="utf-8") as fh:
        json.dump(imputer_doc, fh, indent=2)
    with open(scaler_path, "w", encoding="utf-8") as fh:
        json.dump(scaler_doc, fh, indent=2)
    return {"imputer": imputer_path, "scaler": scaler_path}


def features_metadata(features: List[str], scaler: Optional[Dict[str, object]]) -> Dict[str, object]:
    """Rich rf_close_decision_features.json content (section 10/13)."""
    doc: Dict[str, object] = {
        "features": list(features),
        "raw_feature_names": list(features),
        "final_feature_order": list(features),
    }
    if scaler is not None:
        doc.update(
            {
                "transform_order": ["impute", "transform", "scale"],
                "transforms": scaler["transforms"],
                "imputer": {
                    "strategy": scaler.get("imputer", "median"),
                    "medians": scaler["medians"],
                },
                "scaler": {
                    "type": scaler.get("type", "robust"),
                    "center": scaler["center"],
                    "scale": scaler["scale"],
                },
                "imputer_path": "feature_imputer.json",
                "scaler_path": "feature_scaler.json",
            }
        )
    return doc


def load_scaler(model_dir: str) -> Optional[Dict[str, object]]:
    """Load a fitted scaler (imputer + scaler json) from a model directory."""
    imputer_path = os.path.join(model_dir, "feature_imputer.json")
    scaler_path = os.path.join(model_dir, "feature_scaler.json")
    if not (os.path.isfile(imputer_path) and os.path.isfile(scaler_path)):
        return None
    with open(imputer_path, "r", encoding="utf-8") as fh:
        imp = json.load(fh)
    with open(scaler_path, "r", encoding="utf-8") as fh:
        sca = json.load(fh)
    return {
        "type": sca.get("type", "robust"),
        "imputer": imp.get("strategy", "median"),
        "features": sca.get("features", []),
        "transforms": sca.get("transforms", {}),
        "medians": imp.get("medians", {}),
        "center": sca.get("center", {}),
        "scale": sca.get("scale", {}),
    }
