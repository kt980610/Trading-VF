import numpy as np

from src.portfolio import branch_and_bound as bnb


def _setup():
    mu = np.array([0.02, 0.01])
    sigma = np.array([[0.001, 0.0], [0.0, 0.001]]) + 1e-6 * np.eye(2)
    steps = np.array([0.01, 0.005])
    return mu, sigma, steps


def test_discrete_weight_is_two_k_step():
    mu, sigma, steps = _setup()
    res = bnb.solve_branch_and_bound(mu, sigma, steps, risk_aversion=2.0, max_weight_per_symbol=0.4)
    assert np.allclose(res.w_discrete, 2.0 * res.k * steps)


def test_k_is_integer():
    mu, sigma, steps = _setup()
    res = bnb.solve_branch_and_bound(mu, sigma, steps, 2.0, 0.4)
    assert res.k.dtype.kind in ("i", "u")


def test_long_short_equal_half_weight():
    mu, sigma, steps = _setup()
    res = bnb.solve_branch_and_bound(mu, sigma, steps, 2.0, 0.4)
    long_w = res.k * steps
    short_w = res.k * steps
    assert np.allclose(long_w + short_w, res.w_discrete)
    assert np.allclose(long_w, res.w_discrete / 2.0)


def test_long_short_divisible_by_step():
    mu, sigma, steps = _setup()
    res = bnb.solve_branch_and_bound(mu, sigma, steps, 2.0, 0.4)
    long_ratio = (res.k * steps) / steps
    assert np.allclose(long_ratio, np.round(long_ratio))


def test_sum_weights_within_budget():
    mu, sigma, steps = _setup()
    res = bnb.solve_branch_and_bound(mu, sigma, steps, 2.0, 0.4)
    assert res.w_discrete.sum() <= 1.0 + 1e-12


def test_max_weight_per_symbol_respected():
    mu, sigma, steps = _setup()
    cap = 0.4
    res = bnb.solve_branch_and_bound(mu, sigma, steps, 2.0, cap)
    assert np.all(res.w_discrete <= cap + 1e-12)


def test_timeout_returns_incumbent():
    mu, sigma, steps = _setup()
    res = bnb.solve_branch_and_bound(mu, sigma, steps, 2.0, 0.4, max_nodes=1)
    assert res.status == bnb.STATUS_TIMEOUT
    # Incumbent must still be feasible.
    assert res.w_discrete.sum() <= 1.0 + 1e-12


def test_k_max_vector_bounds():
    steps = np.array([0.01, 0.005])
    kmax = bnb.k_max_vector(steps, max_weight_per_symbol=0.4)
    # 0.4 / (2*0.01) = 20 ; 0.4 / (2*0.005) = 40
    assert kmax[0] == 20
    assert kmax[1] == 40
