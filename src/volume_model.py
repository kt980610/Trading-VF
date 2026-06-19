"""Per-symbol daily volume prediction via walk-forward Kernel Ridge (section 14).

Target is ``next_day_log_volume = log1p(volume)`` and the prediction is mapped
back with ``expm1``. All features for day ``D`` are lagged so only information
known strictly before ``D`` is used; predictions are produced walk-forward
out-of-sample (train on ``< D``, predict ``D``). A pure-numpy RBF kernel-ridge
fallback is used when scikit-learn is unavailable so the module still runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .rolling_features import (
    compute_daily_returns,
    rolling_mean,
    rolling_mean_of_mean,
    rolling_var_of_mean,
    rolling_variance,
)

FEATURE_COLUMNS = [
    "Last30DaysMean",
    "Last30DaysVar",
    "Last30DaysMeanOfMean",
    "Last30DaysVarOfMean",
    "previous_day_volume",
    "previous_day_log_volume",
    "last_7d_mean_volume",
    "last_7d_var_volume",
    "last_30d_mean_volume",
    "last_30d_var_volume",
    "volume_change_1d",
    "volume_change_7d",
]


class _RBFKernelRidge:
    """Minimal RBF kernel-ridge fallback (used only if sklearn is missing)."""

    def __init__(self, alpha: float = 1.0, gamma: Optional[float] = None):
        self.alpha = float(alpha)
        self.gamma = gamma
        self._x = None
        self._dual = None

    def _kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        gamma = self.gamma if self.gamma else 1.0 / a.shape[1]
        sq = (
            np.sum(a ** 2, axis=1)[:, None]
            + np.sum(b ** 2, axis=1)[None, :]
            - 2.0 * a @ b.T
        )
        return np.exp(-gamma * np.maximum(sq, 0.0))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_RBFKernelRidge":
        x = np.asarray(x, dtype="float64")
        y = np.asarray(y, dtype="float64")
        self._x = x
        k = self._kernel(x, x)
        n = k.shape[0]
        self._dual = np.linalg.solve(k + self.alpha * np.eye(n), y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype="float64")
        k = self._kernel(x, self._x)
        return k @ self._dual


def _make_model(kernel: str, alpha: float, gamma: float):
    try:
        from sklearn.kernel_ridge import KernelRidge  # type: ignore

        return KernelRidge(kernel=kernel, alpha=alpha, gamma=(gamma or None))
    except ImportError:
        return _RBFKernelRidge(alpha=alpha, gamma=(gamma or None))


def build_feature_frame(
    daily: pd.DataFrame,
    window: int = 30,
    variance_mode: str = "population",
    news_daily: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build a lagged daily feature frame with the volume target.

    Every feature is shifted by one day so the row for day ``D`` only uses data
    from before ``D``; ``volume``/``log_volume`` are the (current-day) targets.
    """
    df = daily.copy().reset_index(drop=True)
    df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("1D")

    close = df["close"].astype("float64")
    volume = df["volume"].astype("float64")

    returns = compute_daily_returns(close)
    returns = returns.reindex(df.index)
    rmean = rolling_mean(returns, window)
    rvar = rolling_variance(returns, window, variance_mode)
    rmom = rolling_mean_of_mean(rmean.dropna(), window).reindex(df.index)
    rvom = rolling_var_of_mean(rmean.dropna(), window, variance_mode).reindex(df.index)

    feat = pd.DataFrame(index=df.index)
    feat["date"] = df["date"]
    # Lag rolling state by one day so it is known at the start of day D.
    feat["Last30DaysMean"] = rmean.shift(1)
    feat["Last30DaysVar"] = rvar.shift(1)
    feat["Last30DaysMeanOfMean"] = rmom.shift(1)
    feat["Last30DaysVarOfMean"] = rvom.shift(1)

    feat["previous_day_volume"] = volume.shift(1)
    feat["previous_day_log_volume"] = np.log1p(volume.shift(1))
    feat["last_7d_mean_volume"] = volume.shift(1).rolling(7, min_periods=7).mean()
    feat["last_7d_var_volume"] = volume.shift(1).rolling(7, min_periods=7).var(ddof=0)
    feat["last_30d_mean_volume"] = volume.shift(1).rolling(30, min_periods=30).mean()
    feat["last_30d_var_volume"] = volume.shift(1).rolling(30, min_periods=30).var(ddof=0)
    feat["volume_change_1d"] = volume.shift(1) / volume.shift(2) - 1.0
    feat["volume_change_7d"] = volume.shift(1) / volume.shift(8) - 1.0

    feat["volume"] = volume
    feat["log_volume"] = np.log1p(volume)

    if news_daily is not None and not news_daily.empty:
        news = news_daily.copy()
        news["date"] = pd.to_datetime(news["date"], utc=True).dt.floor("1D")
        feat = feat.merge(news.drop(columns=[c for c in ["symbol"] if c in news.columns]), on="date", how="left")

    return feat


def _extra_feature_columns(frame: pd.DataFrame) -> List[str]:
    """Global + correlation-weighted cross-coin news features (sections 5-6)."""
    extras = [
        c
        for c in frame.columns
        if c.endswith("_news_sentiment") or c.endswith("_news_count") or c.endswith("_weighted")
    ]
    return extras


def walk_forward_predict(
    frame: pd.DataFrame,
    symbol: str,
    kernel: str = "rbf",
    alpha: float = 1.0,
    gamma: float = 0.0,
    min_train_days: int = 120,
) -> pd.DataFrame:
    """Produce out-of-sample predicted daily volume for each eligible day."""
    feature_cols = FEATURE_COLUMNS + _extra_feature_columns(frame)
    usable = frame.dropna(subset=feature_cols + ["log_volume"]).reset_index(drop=True)

    records: List[Dict[str, float]] = []
    x_all = usable[feature_cols].to_numpy(dtype="float64")
    y_all = usable["log_volume"].to_numpy(dtype="float64")
    vol_all = usable["volume"].to_numpy(dtype="float64")
    dates = usable["date"].tolist()

    for d in range(min_train_days, len(usable)):
        x_train = x_all[:d]
        y_train = y_all[:d]
        x_test = x_all[d : d + 1]
        # Standardize using only training statistics (no leakage).
        mu = x_train.mean(axis=0)
        sd = x_train.std(axis=0)
        sd[sd == 0] = 1.0
        model = _make_model(kernel, alpha, gamma)
        model.fit((x_train - mu) / sd, y_train)
        pred_log = float(model.predict((x_test - mu) / sd)[0])
        records.append(
            {
                "date": str(pd.Timestamp(dates[d]).date()),
                "symbol": symbol,
                "predicted_log_volume": pred_log,
                "predicted_daily_volume": float(np.expm1(pred_log)),
                "real_volume": float(vol_all[d]),
                "real_log_volume": float(y_all[d]),
                "previous_day_real_volume": float(usable["previous_day_volume"].iloc[d]),
            }
        )
    return pd.DataFrame.from_records(records)


def evaluate_predictions(pred: pd.DataFrame, symbol: str) -> Dict[str, float]:
    """Compute the spec section-14 volume performance report row."""
    if pred.empty:
        return {"symbol": symbol, "n_days": 0}

    real = pred["real_volume"].to_numpy()
    predicted = pred["predicted_daily_volume"].to_numpy()
    real_log = pred["real_log_volume"].to_numpy()
    pred_log = pred["predicted_log_volume"].to_numpy()
    prev = pred["previous_day_real_volume"].to_numpy()

    real_dir = np.sign(real - prev)
    pred_dir = np.sign(predicted - prev)
    direction_acc = float(np.mean(real_dir == pred_dir)) if len(real) else 0.0
    corr = float(np.corrcoef(predicted, real)[0, 1]) if len(real) > 1 else 0.0

    return {
        "symbol": symbol,
        "n_days": int(len(pred)),
        "volume_mae": float(np.mean(np.abs(predicted - real))),
        "volume_rmse": float(np.sqrt(np.mean((predicted - real) ** 2))),
        "log_volume_mae": float(np.mean(np.abs(pred_log - real_log))),
        "log_volume_rmse": float(np.sqrt(np.mean((pred_log - real_log) ** 2))),
        "direction_accuracy_volume_change": direction_acc,
        "corr_pred_real_volume": corr,
    }


def train_and_save_final(
    frame: pd.DataFrame,
    symbol: str,
    models_dir: str,
    kernel: str = "rbf",
    alpha: float = 1.0,
    gamma: float = 0.0,
) -> Optional[str]:
    """Fit on all usable history and persist a joblib artifact."""
    feature_cols = FEATURE_COLUMNS + _extra_feature_columns(frame)
    usable = frame.dropna(subset=feature_cols + ["log_volume"])
    if usable.empty:
        return None

    x = usable[feature_cols].to_numpy(dtype="float64")
    y = usable["log_volume"].to_numpy(dtype="float64")
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    model = _make_model(kernel, alpha, gamma)
    model.fit((x - mu) / sd, y)

    out_dir = os.path.join(models_dir, symbol)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "volume_krr.joblib")
    try:
        import joblib  # type: ignore

        joblib.dump({"model": model, "mu": mu, "sd": sd, "features": feature_cols}, path)
    except ImportError:
        import pickle

        with open(path, "wb") as fh:
            pickle.dump({"model": model, "mu": mu, "sd": sd, "features": feature_cols}, fh)
    return path
