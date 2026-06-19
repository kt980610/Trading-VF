"""Assemble and atomically write portfolio_weights.json (spec section 10)."""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

OBJECTIVE_STRING = "mu^T w - risk_aversion * w^T Sigma w"
INPUT_TYPE = "hybrid_realized_over_simulated"


def _integer_ok(weight: float, step: float, tol: float) -> bool:
    if step <= 0:
        return weight == 0.0
    ratio = weight / step
    return abs(ratio - round(ratio)) < tol


def build_payload(
    config,
    as_of_date: str,
    stats,
    valid_symbols: List[str],
    invalid: Dict[str, str],
    continuous_w: np.ndarray,
    discrete_k: np.ndarray,
    discrete_w: np.ndarray,
) -> dict:
    """Build the section-10 schema dict for all configured symbols."""
    tol = config.integer_tolerance
    index = {s: i for i, s in enumerate(valid_symbols)}

    symbols_out: Dict[str, dict] = {}
    sum_discrete = 0.0

    for symbol in config.symbols:
        if symbol in index:
            i = index[symbol]
            step = config.weight_step_for(symbol)
            w_disc = float(discrete_w[i])
            long_w = w_disc / 2.0
            short_w = w_disc / 2.0
            sum_discrete += w_disc
            symbols_out[symbol] = {
                "valid": True,
                "mu": float(stats.mu[i]),
                "variance": float(stats.variance[i]),
                "weight_continuous": float(continuous_w[i]),
                "weight_step": float(step),
                "k": int(discrete_k[i]),
                "weight_discrete": w_disc,
                "long_weight": long_w,
                "short_weight": short_w,
                "integer_constraint_ok": bool(
                    _integer_ok(long_w, step, tol) and _integer_ok(short_w, step, tol)
                ),
                "weight_times_2_over_step": float(w_disc * 2.0 / step) if step > 0 else 0.0,
            }
        else:
            symbols_out[symbol] = {
                "valid": False,
                "reason": invalid.get(symbol, "not_enough_return_days"),
                "weight_discrete": 0.0,
                "long_weight": 0.0,
                "short_weight": 0.0,
            }

    return {
        "as_of_date": str(as_of_date),
        "lookback_days": int(config.lookback_days),
        "input_type": INPUT_TYPE,
        "objective": OBJECTIVE_STRING,
        "long_short_split": config.long_short_split,
        "risk_aversion": float(config.risk_aversion),
        "symbols": symbols_out,
        "sum_weight_discrete": float(sum_discrete),
        "cash_weight": float(1.0 - sum_discrete),
    }


def write_atomic(payload: dict, output_path: str) -> str:
    """Write JSON atomically: tmp file -> fsync -> rename."""
    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)
    tmp_path = output_path + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass

    os.replace(tmp_path, output_path)
    return output_path
