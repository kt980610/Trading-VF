"""Long/short continuation-edge components (spec sections 8 & 9).

All integrals are looked up from a prebuilt :class:`IntegralCache` and share the
single denominator. Long uses ``[liq_z, y]`` (left) and ``[y, z_max]`` (right);
short uses ``[z_min, y]`` (left) and ``[y, short_liq_z]`` (right). The component
naming matches the RF feature set in spec section 11.
"""

from __future__ import annotations

from typing import Dict, Optional

from .integral_cache import IntegralCache
from .pricing import LegState

MODE_HEDGED = "HEDGED_BOTH_ACTIVE"
MODE_LONG_ONLY = "LONG_ONLY_AFTER_SHORT_LIQ"
MODE_SHORT_ONLY = "SHORT_ONLY_AFTER_LONG_LIQ"

_COMPONENT_SUFFIX = {
    "return": "Return",
    "mean": "Mean",
    "var": "Var",
    "mean_of_mean": "MeanOfMean",
    "var_of_mean": "VarOfMean",
}


def compute_long_components(
    cache: IntegralCache,
    y: float,
    long_liq_z: float,
    notional_long: float,
) -> Dict[str, float]:
    """Right/Left/Edge for all five long components.

    LongRightPnL_C = N_long * integral_long(C, y, z_max) / denom
    LongLeftPnL_C  = N_long * integral_long(C, long_liq_z, y) / denom
    LongEdge_C     = Right - Left
    """
    denom = cache.denom
    z_max = cache.z_max
    out: Dict[str, float] = {}
    for comp, suffix in _COMPONENT_SUFFIX.items():
        right = notional_long * cache.integral_long(comp, y, z_max) / denom
        left = notional_long * cache.integral_long(comp, long_liq_z, y) / denom
        out[f"LongRightPnL_{suffix}"] = float(right)
        out[f"LongLeftPnL_{suffix}"] = float(left)
        out[f"LongEdge_{suffix}"] = float(right - left)
    return out


def compute_short_components(
    cache: IntegralCache,
    y: float,
    short_liq_z: float,
    notional_short: float,
) -> Dict[str, float]:
    """Right/Left/Edge for all five short components.

    ShortLeftPnL_C  = N_short * integral_short(C, z_min, y) / denom
    ShortRightPnL_C = N_short * integral_short(C, y, short_liq_z) / denom
    ShortEdge_C     = Left - Right
    """
    denom = cache.denom
    z_min = cache.z_min
    out: Dict[str, float] = {}
    for comp, suffix in _COMPONENT_SUFFIX.items():
        left = notional_short * cache.integral_short(comp, z_min, y) / denom
        right = notional_short * cache.integral_short(comp, y, short_liq_z) / denom
        out[f"ShortLeftPnL_{suffix}"] = float(left)
        out[f"ShortRightPnL_{suffix}"] = float(right)
        out[f"ShortEdge_{suffix}"] = float(left - right)
    return out


def long_edges_only(components: Dict[str, float]) -> Dict[str, float]:
    return {k: v for k, v in components.items() if k.startswith("LongEdge_")}


def short_edges_only(components: Dict[str, float]) -> Dict[str, float]:
    return {k: v for k, v in components.items() if k.startswith("ShortEdge_")}


def compute_features(
    cache: IntegralCache,
    mode: str,
    current_price: float,
    long_leg: Optional[LegState] = None,
    short_leg: Optional[LegState] = None,
    include_components: bool = False,
) -> Dict[str, float]:
    """Mode-aware edge features.

    * HEDGED_BOTH_ACTIVE        -> both long and short components
    * LONG_ONLY_AFTER_SHORT_LIQ -> only long components
    * SHORT_ONLY_AFTER_LONG_LIQ -> only short components
    """
    features: Dict[str, float] = {"mode": mode}

    want_long = mode in (MODE_HEDGED, MODE_LONG_ONLY) and long_leg is not None
    want_short = mode in (MODE_HEDGED, MODE_SHORT_ONLY) and short_leg is not None

    if want_long:
        y = long_leg.current_return(current_price)
        comp = compute_long_components(cache, y, long_leg.liq_z(), long_leg.notional)
        features.update(long_edges_only(comp))
        if include_components:
            features.update(comp)

    if want_short:
        y = short_leg.current_return(current_price)
        comp = compute_short_components(cache, y, short_leg.liq_z(), short_leg.notional)
        features.update(short_edges_only(comp))
        if include_components:
            features.update(comp)

    return features
