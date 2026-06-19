"""Decision-variable input transforms (spec section 6).

These map a scenario return ``z`` plus the rolling state of a symbol into the
input arguments of the five PDFs. The formulas are implemented verbatim from the
specification and must not be simplified or rewritten.

Rolling state aliases:
    m30      = Last30DaysMean
    v30      = Last30DaysVar
    mom30    = Last30DaysMeanOfMean
    vom30    = Last30DaysVarOfMean
    r_old    = ReturnIn30DaysBefore
    mean_old = MeanIn30DaysBefore
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RollingState:
    """Rolling state values used by the feature-input formulas."""

    Last30DaysMean: float
    Last30DaysVar: float
    Last30DaysMeanOfMean: float
    Last30DaysVarOfMean: float
    ReturnIn30DaysBefore: float
    MeanIn30DaysBefore: float
    window: int = 30

    @classmethod
    def from_dict(cls, data: dict, window: int = 30) -> "RollingState":
        return cls(
            Last30DaysMean=float(data["Last30DaysMean"]),
            Last30DaysVar=float(data["Last30DaysVar"]),
            Last30DaysMeanOfMean=float(data["Last30DaysMeanOfMean"]),
            Last30DaysVarOfMean=float(data["Last30DaysVarOfMean"]),
            ReturnIn30DaysBefore=float(data["ReturnIn30DaysBefore"]),
            MeanIn30DaysBefore=float(data["MeanIn30DaysBefore"]),
            window=int(window),
        )


def mean_input(z, state: RollingState):
    """mean_input(z) = (30 * m30 + z - r_old) / 30"""
    w = state.window
    return (w * state.Last30DaysMean + z - state.ReturnIn30DaysBefore) / w


def var_input(z, state: RollingState):
    """var_input(z) = v30 + z^2/30 - r_old^2/30 + m30^2 - mean_input(z)^2"""
    w = state.window
    mi = mean_input(z, state)
    return (
        state.Last30DaysVar
        + (np.asarray(z, dtype="float64") ** 2) / w
        - (state.ReturnIn30DaysBefore ** 2) / w
        + state.Last30DaysMean ** 2
        - mi ** 2
    )


def mean_of_mean_input(z, state: RollingState):
    """mean_of_mean_input(z) = (mean_input(z) + 30 * mom30 - mean_old) / 30"""
    w = state.window
    mi = mean_input(z, state)
    return (mi + w * state.Last30DaysMeanOfMean - state.MeanIn30DaysBefore) / w


def var_of_mean_input(z, state: RollingState):
    """var_of_mean_input(z), implemented verbatim from spec section 6.4.

        ( vom30
          + mean_input(z)^2 / 30
          - mean_old^2 / 30
          + mom30^2
          - ((30 * mom30 + mean_input(z) - mean_old) / 30)
        )^2

    The trailing inner term equals ``mean_of_mean_input(z)``. The whole bracket
    is squared exactly as specified.
    """
    w = state.window
    mi = mean_input(z, state)
    inner = (w * state.Last30DaysMeanOfMean + mi - state.MeanIn30DaysBefore) / w
    bracket = (
        state.Last30DaysVarOfMean
        + (mi ** 2) / w
        - (state.MeanIn30DaysBefore ** 2) / w
        + state.Last30DaysMeanOfMean ** 2
        - inner
    )
    return bracket ** 2
