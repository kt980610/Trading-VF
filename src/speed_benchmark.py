"""Cached vs full-integral benchmark and equivalence check (spec sections 22-23).

The "full" path recomputes the integral over the grid on every minute; the
"cached" path uses cumulative lookups. Both must agree within tolerance (spec
test 32) and the cached path must be materially faster.
"""

from __future__ import annotations

import os
import time
from typing import Dict

import numpy as np
import pandas as pd

from .feature_inputs import RollingState
from .integral_cache import _cumtrapz, _pdf_eval, build_cache_for_symbol


def full_long_return_edge(symbol_entry: dict, y: float, long_liq_z: float, notional_long: float) -> float:
    """Recompute the long return edge from scratch (no cache)."""
    rd = symbol_entry["return_distribution"]
    z = np.asarray(rd["grid"], dtype="float64")
    f = _pdf_eval(rd["grid"], rd["pdf"], z)
    integrand = f * z
    denom = float(_cumtrapz(f, z)[-1]) or 1.0
    z_min, z_max = float(z[0]), float(z[-1])

    def integral(a: float, b: float) -> float:
        a = min(max(a, z_min), z_max)
        b = min(max(b, z_min), z_max)
        mask = (z >= a) & (z <= b)
        if mask.sum() < 2:
            return 0.0
        trapz = getattr(np, "trapezoid", np.trapz)
        return float(trapz(integrand[mask], z[mask]))

    right = notional_long * integral(y, z_max) / denom
    left = notional_long * integral(long_liq_z, y) / denom
    return right - left


def cached_long_return_edge(cache, y: float, long_liq_z: float, notional_long: float) -> float:
    right = notional_long * cache.integral_long("return", y, cache.z_max) / cache.denom
    left = notional_long * cache.integral_long("return", long_liq_z, y) / cache.denom
    return right - left


def compare_cache_vs_full(
    symbol_entry: dict, y: float, long_liq_z: float, notional_long: float, window: int = 30
):
    """Return (cached_edge, full_edge) for an equivalence assertion."""
    cache = build_cache_for_symbol("TEST", symbol_entry, window=window)
    cached = cached_long_return_edge(cache, y, long_liq_z, notional_long)
    full = full_long_return_edge(symbol_entry, y, long_liq_z, notional_long)
    return cached, full


def benchmark_symbol(symbol: str, symbol_entry: dict, n_minutes: int = 2000, window: int = 30) -> Dict[str, float]:
    """Time full vs cached integral evaluation over ``n_minutes`` synthetic minutes."""
    rd = symbol_entry["return_distribution"]
    z_min = float(np.min(rd["grid"]))
    z_max = float(np.max(rd["grid"]))
    rng = np.random.default_rng(0)
    ys = rng.uniform(z_min * 0.5, z_max * 0.5, size=n_minutes)
    liq = z_min * 0.9
    notional = 1000.0

    t0 = time.perf_counter()
    cache = build_cache_for_symbol(symbol, symbol_entry, window=window)
    cache_build_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    for y in ys:
        full_long_return_edge(symbol_entry, float(y), liq, notional)
    full_total = time.perf_counter() - t0

    t0 = time.perf_counter()
    for y in ys:
        cached_long_return_edge(cache, float(y), liq, notional)
    cached_total = time.perf_counter() - t0

    full_ms_per_min = full_total / n_minutes * 1000.0
    cached_ms_per_min = cached_total / n_minutes * 1000.0
    speedup = (full_total / cached_total) if cached_total > 0 else float("inf")
    minutes_per_sec = (n_minutes / cached_total) if cached_total > 0 else float("inf")

    return {
        "symbol": symbol,
        "old_full_integral_runtime_ms_per_minute": float(full_ms_per_min),
        "cached_integral_runtime_ms_per_minute": float(cached_ms_per_min),
        "cache_build_time_ms": float(cache_build_ms),
        "minutes_processed_per_second": float(minutes_per_sec),
        "speedup_ratio": float(speedup),
    }


def write_benchmark(rows, output_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(output_path, index=False)
    return output_path
