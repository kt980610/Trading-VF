"""Single-leg add-margin MDP for double-liquidation risk (spec sections 17-20).

This engine is intentionally isolated from the RF close/continue policy. It runs
ONLY after one leg has already liquidated (single-leg mode) and ONLY when the
remaining leg's liquidation price is closer than the first liquidation price was.

It uses the return-only integral exclusively -- no pdfMean/Var/MeanOfMean/
VarOfMean, no RF, no news, no volume, no MVO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .integral_cache import IntegralCache
from .pricing import LegState

ACTION_CLOSE = "CLOSE"
ACTION_ADD_MARGIN_CONTINUE = "ADD_MARGIN_CONTINUE"


@dataclass
class MDPDecision:
    triggered: bool
    action: str
    x_best: float
    continue_value: float
    close_value: float
    candidates: List[Tuple[float, float]] = field(default_factory=list)


def mdp_trigger(current_price: float, remaining_liq_price: float, first_liq_price: float) -> bool:
    """True when the remaining leg's liq price is closer than the first one's.

    abs(CurrentPrice - remaining_liq_price) < abs(CurrentPrice - first_liq_price)
    """
    return abs(float(current_price) - float(remaining_liq_price)) < abs(
        float(current_price) - float(first_liq_price)
    )


def enumerate_actions(
    remaining_balance_before: float,
    add_margin_step: float,
    max_add_margin_per_decision: float,
    max_total_added_margin: float,
    current_total_added_margin: float,
) -> List[float]:
    """All valid ``X = k * step`` add-margin actions (spec section 18)."""
    rb = float(remaining_balance_before)
    step = float(add_margin_step)

    # Not enough balance for a single step -> only the no-op action.
    if step <= 0 or rb < step:
        return [0.0]

    upper = rb
    upper = min(upper, float(max_add_margin_per_decision))
    upper = min(upper, float(max_total_added_margin) - float(current_total_added_margin))
    if upper < 0:
        upper = 0.0

    actions = [0.0]
    k = 1
    while True:
        x = k * step
        if x > upper + 1e-12:
            break
        # X can never exceed RemainingBalance (RemainingBalance_before - X >= 0).
        if rb - x < -1e-12:
            break
        actions.append(float(x))
        k += 1
    return actions


def _continuation_integral_long(
    cache: IntegralCache, leg: LegState, x: float, y: float, expected_costs: float
) -> float:
    n = leg.notional
    denom = cache.denom
    liq_z_x = leg.liq_z_after_add(x)
    right = n * cache.integral_long("return", y, cache.z_max) / denom
    left = n * cache.integral_long("return", liq_z_x, y) / denom
    return right - left - float(expected_costs)


def _continuation_integral_short(
    cache: IntegralCache, leg: LegState, x: float, y: float, expected_costs: float
) -> float:
    n = leg.notional
    denom = cache.denom
    liq_z_x = leg.liq_z_after_add(x)
    left = n * cache.integral_short("return", cache.z_min, y) / denom
    right = n * cache.integral_short("return", y, liq_z_x) / denom
    return left - right - float(expected_costs)


def decide(
    cache: IntegralCache,
    leg: LegState,
    current_price: float,
    remaining_balance_before: float,
    first_liq_price: float,
    add_margin_step: float,
    max_add_margin_per_decision: float = 1e18,
    max_total_added_margin: float = 1e18,
    current_total_added_margin: float = 0.0,
    expected_costs: float = 0.0,
) -> MDPDecision:
    """Return the MDP add-margin/close decision for the remaining leg."""
    remaining_liq_price = leg.liquidation_price_level()
    triggered = mdp_trigger(current_price, remaining_liq_price, first_liq_price)

    close_pnl = leg.current_pnl(current_price)
    close_value = close_pnl + float(remaining_balance_before)

    if not triggered:
        # No add-margin search; MDP does not act.
        return MDPDecision(
            triggered=False,
            action=ACTION_CLOSE,
            x_best=0.0,
            continue_value=float("nan"),
            close_value=float(close_value),
            candidates=[],
        )

    y = leg.current_return(current_price)
    actions = enumerate_actions(
        remaining_balance_before,
        add_margin_step,
        max_add_margin_per_decision,
        max_total_added_margin,
        current_total_added_margin,
    )

    candidates: List[Tuple[float, float]] = []
    for x in actions:
        rb_x = float(remaining_balance_before) - x
        if leg.side == "long":
            cont_integral = _continuation_integral_long(cache, leg, x, y, expected_costs)
        else:
            cont_integral = _continuation_integral_short(cache, leg, x, y, expected_costs)
        continue_value = rb_x + cont_integral
        candidates.append((float(x), float(continue_value)))

    # Best X: max value, tie-break to the smaller X.
    best_x, best_value = candidates[0]
    for x, value in candidates[1:]:
        if value > best_value + 1e-15 or (abs(value - best_value) <= 1e-15 and x < best_x):
            best_x, best_value = x, value

    if close_value >= best_value:
        action = ACTION_CLOSE
        chosen_x = 0.0
    else:
        action = ACTION_ADD_MARGIN_CONTINUE
        chosen_x = best_x

    return MDPDecision(
        triggered=True,
        action=action,
        x_best=float(chosen_x),
        continue_value=float(best_value),
        close_value=float(close_value),
        candidates=candidates,
    )
