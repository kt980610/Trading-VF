"""Mean-Variance Optimization statistics and continuous solver (spec sections 4-5).

Objective (maximized):

    f(w) = mu^T w - risk_aversion * w^T Sigma_reg w

subject to w_i >= 0, w_i <= max_weight_per_symbol, and sum_i w_i <= 1
(equality when cash is not allowed). Solved with projected gradient ascent and an
exact Euclidean projection onto the capped simplex, so no external QP solver is
required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class MVOStats:
    symbols: List[str]
    mu: np.ndarray
    variance: np.ndarray
    sigma_reg: np.ndarray


def _ddof_for_mode(mode: str) -> int:
    return 1 if str(mode).lower() == "sample" else 0


def compute_statistics(R: pd.DataFrame, covariance_mode: str = "sample", epsilon: float = 1e-6) -> MVOStats:
    """Compute mu, per-symbol variance, and the regularized covariance matrix."""
    symbols = list(R.columns) if R is not None else []
    n = len(symbols)
    if n == 0:
        return MVOStats(symbols, np.zeros(0), np.zeros(0), np.zeros((0, 0)))

    aligned = R.dropna()
    ddof = _ddof_for_mode(covariance_mode)

    mu = aligned.mean().to_numpy(dtype="float64")

    if len(aligned) >= 2:
        variance = aligned.var(ddof=ddof).to_numpy(dtype="float64")
        sigma = np.cov(aligned.to_numpy(dtype="float64").T, ddof=ddof)
        sigma = np.atleast_2d(sigma)
        if sigma.shape != (n, n):
            sigma = sigma.reshape(n, n)
    else:
        variance = np.zeros(n)
        sigma = np.zeros((n, n))

    sigma_reg = sigma + epsilon * np.eye(n)
    return MVOStats(symbols, mu, variance, sigma_reg)


def objective(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, risk_aversion: float) -> float:
    w = np.asarray(w, dtype="float64")
    return float(mu @ w - risk_aversion * (w @ sigma @ w))


def _project_capped_simplex(v0: np.ndarray, cap: float, s: float, equality: bool) -> np.ndarray:
    """Euclidean projection onto {0 <= w <= cap, sum w = s or <= s}."""
    box = np.clip(v0, 0.0, cap)
    if not equality and box.sum() <= s + 1e-12:
        return box

    s_eff = min(s, cap * len(v0))
    lo = float(v0.min() - cap)
    hi = float(v0.max())
    for _ in range(100):
        tau = 0.5 * (lo + hi)
        w = np.clip(v0 - tau, 0.0, cap)
        if w.sum() > s_eff:
            lo = tau
        else:
            hi = tau
    return np.clip(v0 - hi, 0.0, cap)


def solve_continuous_mvo(
    mu: np.ndarray,
    sigma_reg: np.ndarray,
    risk_aversion: float,
    max_weight_per_symbol: float,
    allow_cash: bool = True,
    max_iter: int = 5000,
    tol: float = 1e-12,
) -> np.ndarray:
    """Projected-gradient solution of the continuous MVO problem."""
    mu = np.asarray(mu, dtype="float64")
    n = mu.shape[0]
    if n == 0:
        return np.zeros(0)

    sigma = np.asarray(sigma_reg, dtype="float64")
    cap = float(max_weight_per_symbol)
    equality = not allow_cash

    eig_max = float(np.linalg.eigvalsh(sigma).max()) if n else 0.0
    lipschitz = 2.0 * risk_aversion * max(eig_max, 0.0)
    lr = 1.0 / (lipschitz + 1e-9)
    lr = min(lr, 1e3)

    w = np.zeros(n)
    for _ in range(max_iter):
        grad = mu - 2.0 * risk_aversion * (sigma @ w)
        w_new = _project_capped_simplex(w + lr * grad, cap, 1.0, equality)
        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break
        w = w_new

    w[w < 1e-12] = 0.0
    return w
