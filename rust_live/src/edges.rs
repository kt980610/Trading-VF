//! Long/short continuation-edge features from the integral cache.
//!
//! Long uses `[liq_z, y]` (left) and `[y, z_max]` (right); short uses
//! `[z_min, y]` (left) and `[y, short_liq_z]` (right). Naming matches the RF
//! feature schema.

use std::collections::HashMap;

use crate::artifacts::IntegralCache;
use crate::pricing::LegState;
use crate::state::Mode;

const COMPONENTS: &[(&str, &str)] = &[
    ("return", "Return"),
    ("mean", "Mean"),
    ("var", "Var"),
    ("mean_of_mean", "MeanOfMean"),
    ("var_of_mean", "VarOfMean"),
];

pub fn compute_long_components(
    cache: &IntegralCache,
    y: f64,
    long_liq_z: f64,
    notional_long: f64,
    out: &mut HashMap<String, f64>,
) {
    let denom = if cache.denom == 0.0 { 1.0 } else { cache.denom };
    let z_max = cache.z_max();
    for (comp, suffix) in COMPONENTS {
        let right = notional_long * cache.integral_long(comp, y, z_max) / denom;
        let left = notional_long * cache.integral_long(comp, long_liq_z, y) / denom;
        out.insert(format!("LongRightPnL_{suffix}"), right);
        out.insert(format!("LongLeftPnL_{suffix}"), left);
        out.insert(format!("LongEdge_{suffix}"), right - left);
    }
}

pub fn compute_short_components(
    cache: &IntegralCache,
    y: f64,
    short_liq_z: f64,
    notional_short: f64,
    out: &mut HashMap<String, f64>,
) {
    let denom = if cache.denom == 0.0 { 1.0 } else { cache.denom };
    let z_min = cache.z_min();
    for (comp, suffix) in COMPONENTS {
        let left = notional_short * cache.integral_short(comp, z_min, y) / denom;
        let right = notional_short * cache.integral_short(comp, y, short_liq_z) / denom;
        out.insert(format!("ShortLeftPnL_{suffix}"), left);
        out.insert(format!("ShortRightPnL_{suffix}"), right);
        out.insert(format!("ShortEdge_{suffix}"), left - right);
    }
}

/// Mode-aware edge features. Inactive legs contribute nothing (defaults 0 are
/// filled later by the feature builder).
pub fn compute_features(
    cache: &IntegralCache,
    mode: Mode,
    current_price: f64,
    long: Option<&LegState>,
    short: Option<&LegState>,
    include_components: bool,
) -> HashMap<String, f64> {
    let mut out: HashMap<String, f64> = HashMap::new();

    let want_long = matches!(mode, Mode::HedgedBothActive | Mode::LongOnlyAfterShortLiq) && long.is_some();
    let want_short = matches!(mode, Mode::HedgedBothActive | Mode::ShortOnlyAfterLongLiq) && short.is_some();

    if want_long {
        let leg = long.unwrap();
        let y = leg.current_return(current_price);
        compute_long_components(cache, y, leg.liq_z(), leg.notional(), &mut out);
    }
    if want_short {
        let leg = short.unwrap();
        let y = leg.current_return(current_price);
        compute_short_components(cache, y, leg.liq_z(), leg.notional(), &mut out);
    }

    if !include_components {
        out.retain(|k, _| k.contains("Edge_"));
    }
    out
}
