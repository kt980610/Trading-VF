"""Cumulative integral cache for fast per-minute edge lookups (spec section 22).

For each symbol/snapshot/rolling-state we precompute cumulative integrals of the
five PDF integrands over the return grid. At runtime an interval integral is a
cheap ``cum(b) - cum(a)`` lookup, so changing ``CurrentPrice`` (hence ``y``) or a
liquidation cutoff only moves the interval bounds -- the cache is never rebuilt.

The single shared denominator (spec section 7) is::

    denom = integral of pdfReturn(z) dz over [z_min, z_max]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .feature_inputs import (
    RollingState,
    mean_input,
    mean_of_mean_input,
    var_input,
    var_of_mean_input,
)

_COMPONENTS = ("return", "mean", "var", "mean_of_mean", "var_of_mean")


def _pdf_eval(grid: np.ndarray, pdf: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vectorized PDF lookup; 0 outside the grid (spec: pdf=0 out of range)."""
    grid = np.asarray(grid, dtype="float64")
    pdf = np.asarray(pdf, dtype="float64")
    x = np.asarray(x, dtype="float64")
    if grid.size == 0:
        return np.zeros_like(x)
    out = np.interp(x, grid, pdf)
    out[(x < grid[0]) | (x > grid[-1])] = 0.0
    return out


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral with a leading 0 (same length as x)."""
    y = np.asarray(y, dtype="float64")
    x = np.asarray(x, dtype="float64")
    out = np.zeros_like(y)
    if y.size < 2:
        return out
    dx = np.diff(x)
    incr = (y[:-1] + y[1:]) / 2.0 * dx
    out[1:] = np.cumsum(incr)
    return out


@dataclass
class IntegralCache:
    """Holds the z-grid plus cumulative integrals for one symbol/state."""

    symbol: str
    grid: np.ndarray
    denom: float
    cum_long: Dict[str, np.ndarray] = field(default_factory=dict)
    cum_short: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def z_min(self) -> float:
        return float(self.grid[0])

    @property
    def z_max(self) -> float:
        return float(self.grid[-1])

    def _clip(self, value: float) -> float:
        return float(min(max(float(value), self.z_min), self.z_max))

    def _interp_cum(self, cum: np.ndarray, x: float) -> float:
        return float(np.interp(self._clip(x), self.grid, cum))

    def integral_long(self, component: str, a: float, b: float) -> float:
        cum = self.cum_long[component]
        return self._interp_cum(cum, b) - self._interp_cum(cum, a)

    def integral_short(self, component: str, a: float, b: float) -> float:
        cum = self.cum_short[component]
        return self._interp_cum(cum, b) - self._interp_cum(cum, a)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "grid": self.grid.tolist(),
            "denom": float(self.denom),
            "cum_long": {k: v.tolist() for k, v in self.cum_long.items()},
            "cum_short": {k: v.tolist() for k, v in self.cum_short.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IntegralCache":
        return cls(
            symbol=data["symbol"],
            grid=np.asarray(data["grid"], dtype="float64"),
            denom=float(data["denom"]),
            cum_long={k: np.asarray(v, dtype="float64") for k, v in data["cum_long"].items()},
            cum_short={k: np.asarray(v, dtype="float64") for k, v in data["cum_short"].items()},
        )


def _component_pdf_values(symbol_entry: dict, state: RollingState, z: np.ndarray) -> Dict[str, np.ndarray]:
    """Evaluate each PDF at its transformed input across the z-grid."""
    rd = symbol_entry["return_distribution"]
    md = symbol_entry["mean_distribution"]
    vd = symbol_entry["variance_distribution"]
    mom = symbol_entry["mean_of_mean_distribution"]
    vom = symbol_entry["var_of_mean_distribution"]

    return {
        "return": _pdf_eval(rd["grid"], rd["pdf"], z),
        "mean": _pdf_eval(md["grid"], md["pdf"], mean_input(z, state)),
        "var": _pdf_eval(vd["grid"], vd["pdf"], var_input(z, state)),
        "mean_of_mean": _pdf_eval(mom["grid"], mom["pdf"], mean_of_mean_input(z, state)),
        "var_of_mean": _pdf_eval(vom["grid"], vom["pdf"], var_of_mean_input(z, state)),
    }


def build_cache_for_symbol(
    symbol: str,
    symbol_entry: dict,
    window: int = 30,
    grid_points: int = 0,
) -> IntegralCache:
    """Build an :class:`IntegralCache` from a snapshot symbol entry.

    ``grid_points`` of 0 reuses the return-distribution grid; a positive value
    resamples a uniform grid over ``[z_min, z_max]``.
    """
    state = RollingState.from_dict(symbol_entry["rolling_state"], window=window)
    base_grid = np.asarray(symbol_entry["return_distribution"]["grid"], dtype="float64")
    if grid_points and grid_points > 1:
        z = np.linspace(float(base_grid[0]), float(base_grid[-1]), int(grid_points))
    else:
        z = base_grid

    pdf_values = _component_pdf_values(symbol_entry, state, z)

    # Shared denominator: integral of pdfReturn over the full grid.
    denom = float(_cumtrapz(pdf_values["return"], z)[-1])
    if denom == 0.0 or not np.isfinite(denom):
        denom = 1.0

    cum_long: Dict[str, np.ndarray] = {}
    cum_short: Dict[str, np.ndarray] = {}
    for name in _COMPONENTS:
        f = pdf_values[name]
        cum_long[name] = _cumtrapz(f * z, z)
        cum_short[name] = _cumtrapz(f * (-z), z)

    return IntegralCache(symbol=symbol, grid=z, denom=denom, cum_long=cum_long, cum_short=cum_short)


def build_caches(snapshot: dict, window: int = 30, grid_points: int = 0) -> Dict[str, IntegralCache]:
    """Build caches for every valid symbol in a distribution snapshot."""
    caches: Dict[str, IntegralCache] = {}
    for symbol, entry in snapshot.get("symbols", {}).items():
        if not entry.get("valid"):
            continue
        caches[symbol] = build_cache_for_symbol(symbol, entry, window=window, grid_points=grid_points)
    return caches
