"""MDP-as-feature-engine: per-minute integral/risk features (spec section 11).

The MDP NO LONGER produces an ``ADD_MARGIN`` action for the RF dataset, RF
target or RF live action logic. Here it behaves purely as a feature engine: each
minute it recomputes return-only integral and liquidation-risk values from the
current state and exposes them to the RF close classifier. It never adds margin
and never decides close/continue (that is solely the symbol's RF classifier).
"""

from __future__ import annotations

from typing import Dict, Optional

from .integral_cache import IntegralCache
from .pricing import LegState

MDP_FEATURES = [
    "mdp_return_integral_updated",
    "mdp_left_pnl_updated",
    "mdp_right_pnl_updated",
    "mdp_liq_risk_score",
    "mdp_expected_continue_value",
]


def _leg_integrals(cache: IntegralCache, leg: LegState, current_price: float):
    """Return-only (x=0, no added margin) left/right integral parts for a leg."""
    denom = cache.denom or 1.0
    n = leg.notional
    y = leg.current_return(current_price)
    liq_z = leg.liq_z()
    if leg.side == "long":
        right = n * cache.integral_long("return", y, cache.z_max) / denom
        left = n * cache.integral_long("return", liq_z, y) / denom
    else:
        right = n * cache.integral_short("return", y, liq_z) / denom
        left = n * cache.integral_short("return", cache.z_min, y) / denom
    return float(left), float(right), float(y), float(liq_z)


def _liq_risk(current_price: float, leg: LegState) -> float:
    """Bounded [0,1] proximity-to-liquidation score (1 = at liquidation)."""
    liq = leg.liquidation_price_level()
    entry = leg.entry_price or 1.0
    denom = abs(current_price - liq) + abs(entry - liq)
    if denom <= 0:
        return 1.0
    score = 1.0 - abs(current_price - liq) / denom
    return float(min(1.0, max(0.0, score)))


def compute_mdp_features(
    cache: Optional[IntegralCache],
    long_leg: Optional[LegState],
    short_leg: Optional[LegState],
    current_price: float,
) -> Dict[str, float]:
    """Per-minute MDP feature dict (zeros when nothing is open / no cache)."""
    feats = {name: 0.0 for name in MDP_FEATURES}
    if cache is None:
        return feats

    left_total = right_total = integral_total = 0.0
    risk_max = 0.0
    legs = [leg for leg in (long_leg, short_leg) if leg is not None]
    for leg in legs:
        left, right, _y, _liq = _leg_integrals(cache, leg, current_price)
        left_total += left
        right_total += right
        integral_total += right - left
        risk_max = max(risk_max, _liq_risk(current_price, leg))

    feats["mdp_left_pnl_updated"] = left_total
    feats["mdp_right_pnl_updated"] = right_total
    feats["mdp_return_integral_updated"] = integral_total
    feats["mdp_expected_continue_value"] = integral_total
    feats["mdp_liq_risk_score"] = risk_max
    return feats
