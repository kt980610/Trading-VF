"""Histogram-based probability distribution helpers."""

from __future__ import annotations

import numpy as np


def build_histogram_pdf(values: np.ndarray, bins: int) -> dict:
    values = np.asarray(values, dtype="float64")
    values = values[np.isfinite(values)]

    pdf, edges = np.histogram(values, bins=bins, density=True)
    grid = (edges[:-1] + edges[1:]) / 2.0
    bin_widths = np.diff(edges)
    cdf = np.cumsum(pdf * bin_widths)

    return {
        "grid": grid.astype("float64").tolist(),
        "pdf": pdf.astype("float64").tolist(),
        "cdf": cdf.astype("float64").tolist(),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "variance": float(np.var(values, ddof=0)),
        "n_observations": int(values.size),
    }


def pdf_at(grid, pdf, x: float) -> float:
    grid = np.asarray(grid, dtype="float64")
    pdf = np.asarray(pdf, dtype="float64")
    if grid.size == 0:
        return 0.0
    x = float(x)
    if x < grid[0] or x > grid[-1]:
        return 0.0
    return float(np.interp(x, grid, pdf))
