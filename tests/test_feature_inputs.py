import numpy as np

from src.feature_inputs import (
    RollingState,
    mean_input,
    mean_of_mean_input,
    var_input,
    var_of_mean_input,
)


def _state():
    return RollingState(
        Last30DaysMean=0.01,
        Last30DaysVar=0.001,
        Last30DaysMeanOfMean=0.005,
        Last30DaysVarOfMean=0.0002,
        ReturnIn30DaysBefore=0.02,
        MeanIn30DaysBefore=0.004,
        window=30,
    )


def test_mean_input_formula():
    s = _state()
    z = 0.03
    expected = (30 * 0.01 + 0.03 - 0.02) / 30
    assert np.isclose(mean_input(z, s), expected)


def test_var_input_formula():
    s = _state()
    z = 0.03
    mi = (30 * 0.01 + 0.03 - 0.02) / 30
    expected = 0.001 + z ** 2 / 30 - 0.02 ** 2 / 30 + 0.01 ** 2 - mi ** 2
    assert np.isclose(var_input(z, s), expected)


def test_mean_of_mean_input_formula():
    s = _state()
    z = 0.03
    mi = mean_input(z, s)
    expected = (mi + 30 * 0.005 - 0.004) / 30
    assert np.isclose(mean_of_mean_input(z, s), expected)


def test_var_of_mean_input_is_squared_bracket():
    s = _state()
    z = 0.03
    mi = mean_input(z, s)
    inner = (30 * 0.005 + mi - 0.004) / 30
    bracket = 0.0002 + mi ** 2 / 30 - 0.004 ** 2 / 30 + 0.005 ** 2 - inner
    expected = bracket ** 2
    assert np.isclose(var_of_mean_input(z, s), expected)
    # The whole bracket is squared, hence always non-negative.
    assert var_of_mean_input(z, s) >= 0.0


def test_feature_inputs_vectorized():
    s = _state()
    z = np.array([-0.01, 0.0, 0.02])
    assert mean_input(z, s).shape == z.shape
    assert var_input(z, s).shape == z.shape
