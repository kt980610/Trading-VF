import numpy as np
import pandas as pd

from src.portfolio import mvo as mvo_mod


def _R():
    return pd.DataFrame(
        {
            "BTCUSDT": [0.01, 0.02, 0.03, 0.00, 0.02],
            "ETHUSDT": [0.00, -0.01, 0.01, 0.02, 0.00],
        }
    )


def test_mu_is_column_mean():
    stats = mvo_mod.compute_statistics(_R(), covariance_mode="sample", epsilon=1e-6)
    assert np.isclose(stats.mu[0], _R()["BTCUSDT"].mean())
    assert np.isclose(stats.mu[1], _R()["ETHUSDT"].mean())


def test_variance_sample_mode():
    stats = mvo_mod.compute_statistics(_R(), covariance_mode="sample", epsilon=0.0)
    assert np.isclose(stats.variance[0], _R()["BTCUSDT"].var(ddof=1))


def test_covariance_shape():
    stats = mvo_mod.compute_statistics(_R(), covariance_mode="sample", epsilon=1e-6)
    assert stats.sigma_reg.shape == (2, 2)


def test_covariance_regularization_adds_epsilon_on_diagonal():
    eps = 1e-3
    raw = np.cov(_R().to_numpy().T, ddof=1)
    stats = mvo_mod.compute_statistics(_R(), covariance_mode="sample", epsilon=eps)
    assert np.allclose(np.diag(stats.sigma_reg) - np.diag(raw), eps)


def test_continuous_weights_non_negative_and_bounded_sum():
    stats = mvo_mod.compute_statistics(_R(), covariance_mode="sample", epsilon=1e-6)
    w = mvo_mod.solve_continuous_mvo(
        stats.mu, stats.sigma_reg, risk_aversion=2.0, max_weight_per_symbol=0.4, allow_cash=True
    )
    assert np.all(w >= -1e-9)
    assert w.sum() <= 1.0 + 1e-6


def test_continuous_weights_respect_cap():
    stats = mvo_mod.compute_statistics(_R(), covariance_mode="sample", epsilon=1e-6)
    cap = 0.3
    w = mvo_mod.solve_continuous_mvo(
        stats.mu, stats.sigma_reg, risk_aversion=0.1, max_weight_per_symbol=cap, allow_cash=True
    )
    assert np.all(w <= cap + 1e-6)


def test_zero_weights_when_all_negative_mu():
    R = pd.DataFrame({"A": [-0.02, -0.01, -0.03], "B": [-0.01, -0.02, -0.01]})
    stats = mvo_mod.compute_statistics(R, covariance_mode="sample", epsilon=1e-6)
    w = mvo_mod.solve_continuous_mvo(stats.mu, stats.sigma_reg, 2.0, 0.4, allow_cash=True)
    assert np.allclose(w, 0.0)
