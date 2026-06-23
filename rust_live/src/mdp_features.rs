//! MDP-as-feature-engine: per-minute integral/risk features.
//!
//! Exact parity with the Python `src/mdp_features.py`. The MDP NO LONGER
//! produces an `ADD_MARGIN` action in the live engine; here it is purely a
//! feature engine. Each minute it recomputes return-only integral and
//! liquidation-risk values from the current state and exposes them to the RF
//! close classifier. It never adds margin and never decides close/continue
//! (that is solely the symbol's RF classifier).

use std::collections::HashMap;

use crate::artifacts::IntegralCache;
use crate::pricing::{LegState, Side};

pub const MDP_FEATURES: &[&str] = &[
    "mdp_return_integral_updated",
    "mdp_left_pnl_updated",
    "mdp_right_pnl_updated",
    "mdp_liq_risk_score",
    "mdp_expected_continue_value",
];

/// Return-only (x = 0, no added margin) left/right integral parts for a leg.
/// Mirrors `_leg_integrals`.
fn leg_integrals(cache: &IntegralCache, leg: &LegState, current_price: f64) -> (f64, f64) {
    let denom = if cache.denom == 0.0 { 1.0 } else { cache.denom };
    let n = leg.notional();
    let y = leg.current_return(current_price);
    let liq_z = leg.liq_z();
    match leg.side {
        Side::Long => {
            let right = n * cache.integral_long("return", y, cache.z_max()) / denom;
            let left = n * cache.integral_long("return", liq_z, y) / denom;
            (left, right)
        }
        Side::Short => {
            let right = n * cache.integral_short("return", y, liq_z) / denom;
            let left = n * cache.integral_short("return", cache.z_min(), y) / denom;
            (left, right)
        }
    }
}

/// Bounded [0,1] proximity-to-liquidation score (1 = at liquidation). Mirrors
/// `_liq_risk`.
fn liq_risk(current_price: f64, leg: &LegState) -> f64 {
    let liq = leg.liquidation_price_level();
    let entry = if leg.entry_price == 0.0 {
        1.0
    } else {
        leg.entry_price
    };
    let denom = (current_price - liq).abs() + (entry - liq).abs();
    if denom <= 0.0 {
        return 1.0;
    }
    let score = 1.0 - (current_price - liq).abs() / denom;
    score.clamp(0.0, 1.0)
}

/// Per-minute MDP feature map (zeros when nothing is open / no cache). Mirrors
/// `compute_mdp_features`.
pub fn compute_mdp_features(
    cache: Option<&IntegralCache>,
    long_leg: Option<&LegState>,
    short_leg: Option<&LegState>,
    current_price: f64,
) -> HashMap<String, f64> {
    let mut feats: HashMap<String, f64> = MDP_FEATURES
        .iter()
        .map(|k| ((*k).to_string(), 0.0))
        .collect();
    let cache = match cache {
        Some(c) => c,
        None => return feats,
    };

    let mut left_total = 0.0;
    let mut right_total = 0.0;
    let mut integral_total = 0.0;
    let mut risk_max = 0.0_f64;
    for leg in [long_leg, short_leg].into_iter().flatten() {
        let (left, right) = leg_integrals(cache, leg, current_price);
        left_total += left;
        right_total += right;
        integral_total += right - left;
        risk_max = risk_max.max(liq_risk(current_price, leg));
    }

    feats.insert("mdp_left_pnl_updated".into(), left_total);
    feats.insert("mdp_right_pnl_updated".into(), right_total);
    feats.insert("mdp_return_integral_updated".into(), integral_total);
    feats.insert("mdp_expected_continue_value".into(), integral_total);
    feats.insert("mdp_liq_risk_score".into(), risk_max);
    feats
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as Map;

    fn cache() -> IntegralCache {
        let grid = vec![-0.2, -0.1, 0.0, 0.1, 0.2];
        let mut cum_long = Map::new();
        cum_long.insert("return".to_string(), vec![0.0, 0.5, 1.0, 1.8, 3.0]);
        let mut cum_short = Map::new();
        cum_short.insert("return".to_string(), vec![0.0, 0.4, 0.9, 1.5, 2.2]);
        IntegralCache {
            symbol: "X".into(),
            grid,
            denom: 1.0,
            cum_long,
            cum_short,
        }
    }

    #[test]
    fn no_cache_is_zeros() {
        let f = compute_mdp_features(None, None, None, 100.0);
        assert!(MDP_FEATURES.iter().all(|k| f[*k] == 0.0));
    }

    #[test]
    fn open_leg_populates_features_and_risk_bounded() {
        let leg = LegState::new(Side::Long, 100.0, 1.0, 50.0); // liq ~95
        let f = compute_mdp_features(Some(&cache()), Some(&leg), None, 96.0);
        assert!(f["mdp_liq_risk_score"] >= 0.0 && f["mdp_liq_risk_score"] <= 1.0);
        // integral == right - left == expected_continue_value.
        assert!(
            (f["mdp_return_integral_updated"] - f["mdp_expected_continue_value"]).abs() < 1e-12
        );
    }
}
