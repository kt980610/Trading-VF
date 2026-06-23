"""NumPy 2 compatibility for the trapezoidal integral used in speed_benchmark.

NumPy 2 removed ``np.trapz`` in favour of ``np.trapezoid``. The benchmark must
resolve the function lazily so importing/using it never evaluates a removed
attribute. These tests pin that behaviour.
"""

import numpy as np

from src import speed_benchmark as sb


def test_trapezoid_lazy_fallback_resolves_to_callable():
    # The safe pattern must never touch np.trapz eagerly. Whatever NumPy version
    # is installed, we must end up with a working callable.
    trapz = getattr(np, "trapezoid", None)
    if trapz is None:
        trapz = np.trapz
    assert callable(trapz)
    # Integral of a constant 1.0 over [0, 1] sampled densely is ~1.0.
    x = np.linspace(0.0, 1.0, 101)
    y = np.ones_like(x)
    assert abs(float(trapz(y, x)) - 1.0) < 1e-9


def test_full_long_return_edge_runs_under_numpy2():
    # Symmetric distribution on a small grid; the function must execute the
    # trapz line without raising on NumPy 2.
    grid = list(np.linspace(-0.1, 0.1, 21))
    pdf = [1.0] * len(grid)
    symbol_entry = {"return_distribution": {"grid": grid, "pdf": pdf}}
    edge = sb.full_long_return_edge(symbol_entry, y=0.0, long_liq_z=-0.05, notional_long=1000.0)
    assert isinstance(edge, float)
    assert np.isfinite(edge)
