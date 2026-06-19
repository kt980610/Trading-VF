"""Branch-and-Bound discrete weight optimization (spec sections 7-9).

Decision variable per symbol is an integer ``k_i`` with::

    w_i        = 2 * k_i * weight_step_i
    long_i     = short_i = k_i * weight_step_i

so both legs are automatically integer multiples of the step. We maximize the
MVO objective over the integer lattice subject to ``w_i <= max_weight`` and
``sum_i w_i <= 1``. With ~6-8 symbols a depth-first search with an optimistic
linear upper bound prunes effectively; on hitting the node/time budget the best
incumbent is returned.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List

import numpy as np

from .mvo import objective

STATUS_OPTIMAL = "optimal"
STATUS_TIMEOUT = "timeout"


@dataclass
class BnBResult:
    k: np.ndarray
    w_discrete: np.ndarray
    objective_value: float
    status: str
    nodes_explored: int


def k_max_vector(steps: np.ndarray, max_weight_per_symbol: float) -> np.ndarray:
    """Per-symbol upper bound on k from per-symbol and portfolio caps."""
    steps = np.asarray(steps, dtype="float64")
    out = np.zeros(len(steps), dtype="int64")
    for i, step in enumerate(steps):
        if step <= 0:
            out[i] = 0
            continue
        k_symbol = math.floor(max_weight_per_symbol / (2.0 * step))
        k_portfolio = math.floor(1.0 / (2.0 * step))
        out[i] = max(0, min(k_symbol, k_portfolio))
    return out


def weights_from_k(k: np.ndarray, steps: np.ndarray) -> np.ndarray:
    return 2.0 * np.asarray(k, dtype="float64") * np.asarray(steps, dtype="float64")


def _rounded_incumbent(w_continuous, steps, kmax, max_weight):
    """Floor the continuous solution to a feasible integer start, if possible."""
    steps = np.asarray(steps, dtype="float64")
    k = np.zeros(len(steps), dtype="int64")
    for i, step in enumerate(steps):
        if step <= 0:
            continue
        k[i] = min(int(math.floor(w_continuous[i] / (2.0 * step))), int(kmax[i]))
        k[i] = max(0, k[i])
    w = weights_from_k(k, steps)
    # Repair portfolio constraint by trimming the largest k until feasible.
    while w.sum() > 1.0 + 1e-12:
        idx = int(np.argmax(k))
        if k[idx] <= 0:
            break
        k[idx] -= 1
        w = weights_from_k(k, steps)
    return k


def solve_branch_and_bound(
    mu: np.ndarray,
    sigma_reg: np.ndarray,
    steps: np.ndarray,
    risk_aversion: float,
    max_weight_per_symbol: float,
    w_continuous: np.ndarray = None,
    max_nodes: int = 50000,
    max_runtime_ms: int = 5000,
    objective_tolerance: float = 1e-9,
) -> BnBResult:
    mu = np.asarray(mu, dtype="float64")
    steps = np.asarray(steps, dtype="float64")
    sigma = np.asarray(sigma_reg, dtype="float64")
    n = mu.shape[0]

    if n == 0:
        return BnBResult(np.zeros(0, dtype="int64"), np.zeros(0), 0.0, STATUS_OPTIMAL, 0)

    kmax = k_max_vector(steps, max_weight_per_symbol)
    w_max = weights_from_k(kmax, steps)
    mu_pos = np.maximum(mu, 0.0)

    # Incumbent: zero solution, optionally improved by the rounded continuous one.
    best_k = np.zeros(n, dtype="int64")
    best_val = objective(weights_from_k(best_k, steps), mu, sigma, risk_aversion)
    if w_continuous is not None and len(w_continuous) == n:
        cand_k = _rounded_incumbent(w_continuous, steps, kmax, max_weight_per_symbol)
        cand_w = weights_from_k(cand_k, steps)
        if cand_w.sum() <= 1.0 + 1e-12:
            cand_val = objective(cand_w, mu, sigma, risk_aversion)
            if cand_val > best_val:
                best_k, best_val = cand_k, cand_val

    deadline = time.perf_counter() + max_runtime_ms / 1000.0
    state = {"nodes": 0, "best_k": best_k.copy(), "best_val": best_val, "timeout": False}

    def upper_bound(idx: int, partial_k: np.ndarray, sum_w: float) -> float:
        # Optimistic: full linear value, dropping the (non-negative) quadratic.
        w_partial = weights_from_k(partial_k, steps)
        assigned_linear = float(mu[:idx] @ w_partial[:idx])
        remaining_cap = max(0.0, 1.0 - sum_w)
        bonus = 0.0
        for j in range(idx, n):
            bonus += mu_pos[j] * min(w_max[j], remaining_cap)
        return assigned_linear + bonus

    def dfs(idx: int, partial_k: np.ndarray, sum_w: float):
        if state["timeout"]:
            return
        state["nodes"] += 1
        if state["nodes"] > max_nodes or time.perf_counter() > deadline:
            state["timeout"] = True
            return

        if idx == n:
            val = objective(weights_from_k(partial_k, steps), mu, sigma, risk_aversion)
            if val > state["best_val"] + objective_tolerance:
                state["best_val"] = val
                state["best_k"] = partial_k.copy()
            return

        # Prune whole subtree if the optimistic bound cannot beat the incumbent.
        if upper_bound(idx, partial_k, sum_w) <= state["best_val"] + objective_tolerance:
            return

        step = steps[idx]
        for k in range(int(kmax[idx]) + 1):
            w_i = 2.0 * k * step
            if sum_w + w_i > 1.0 + 1e-12:
                break
            partial_k[idx] = k
            dfs(idx + 1, partial_k, sum_w + w_i)
            if state["timeout"]:
                break
        partial_k[idx] = 0

    dfs(0, np.zeros(n, dtype="int64"), 0.0)

    status = STATUS_TIMEOUT if state["timeout"] else STATUS_OPTIMAL
    best_k = state["best_k"]
    return BnBResult(
        k=best_k,
        w_discrete=weights_from_k(best_k, steps),
        objective_value=float(state["best_val"]),
        status=status,
        nodes_explored=int(state["nodes"]),
    )
