import numpy as np

from src.integral_cache import build_cache_for_symbol
from src.speed_benchmark import cached_long_return_edge, full_long_return_edge
from tests.synthetic import make_symbol_entry


def test_cache_matches_full_integral_within_tolerance():
    entry = make_symbol_entry(seed=1)
    cache = build_cache_for_symbol("BTCUSDT", entry, window=30)
    grid = cache.grid

    # Use on-grid bounds so the cumulative lookup and full trapz integrate the
    # same discrete points (spec test 32).
    y = float(grid[len(grid) // 2])
    liq_z = float(grid[len(grid) // 4])
    notional = 1000.0

    cached = cached_long_return_edge(cache, y, liq_z, notional)
    full = full_long_return_edge(entry, y, liq_z, notional)
    assert np.isclose(cached, full, atol=1e-9, rtol=1e-6)


def test_denominator_is_positive():
    entry = make_symbol_entry(seed=2)
    cache = build_cache_for_symbol("ETHUSDT", entry, window=30)
    assert cache.denom > 0.0


def test_interval_integral_is_additive():
    entry = make_symbol_entry(seed=3)
    cache = build_cache_for_symbol("X", entry, window=30)
    g = cache.grid
    a, m, b = float(g[10]), float(g[40]), float(g[80])
    whole = cache.integral_long("return", a, b)
    parts = cache.integral_long("return", a, m) + cache.integral_long("return", m, b)
    assert np.isclose(whole, parts, atol=1e-12)


def test_out_of_grid_clipped():
    entry = make_symbol_entry(seed=4)
    cache = build_cache_for_symbol("X", entry, window=30)
    # Beyond-grid bounds clip to grid edges -> integral over full grid.
    far = cache.integral_long("return", cache.z_min - 5.0, cache.z_max + 5.0)
    inside = cache.integral_long("return", cache.z_min, cache.z_max)
    assert np.isclose(far, inside)
