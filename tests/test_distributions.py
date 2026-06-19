import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.distributions import build_histogram_pdf, pdf_at


def test_histogram_pdf_normalizes():
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 1.0, 5000)
    result = build_histogram_pdf(values, bins=400)
    grid = np.asarray(result["grid"])
    pdf = np.asarray(result["pdf"])
    bin_width = grid[1] - grid[0]
    assert np.sum(pdf * bin_width) == pytest.approx(1.0, abs=1e-6)


def test_grid_pdf_lengths_equal():
    rng = np.random.default_rng(1)
    values = rng.normal(0.0, 1.0, 1000)
    result = build_histogram_pdf(values, bins=400)
    assert len(result["grid"]) == 400
    assert len(result["pdf"]) == 400
    assert len(result["grid"]) == len(result["pdf"])


def test_pdf_at_outside_grid_returns_zero():
    rng = np.random.default_rng(2)
    values = rng.normal(0.0, 1.0, 2000)
    result = build_histogram_pdf(values, bins=100)
    grid = result["grid"]
    pdf = result["pdf"]
    assert pdf_at(grid, pdf, grid[0] - 1.0) == 0.0
    assert pdf_at(grid, pdf, grid[-1] + 1.0) == 0.0


def test_pdf_at_inside_grid_positive():
    rng = np.random.default_rng(3)
    values = rng.normal(0.0, 1.0, 5000)
    result = build_histogram_pdf(values, bins=100)
    grid = result["grid"]
    pdf = result["pdf"]
    value = pdf_at(grid, pdf, 0.0)
    assert value > 0.0
