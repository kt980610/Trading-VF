//! Single-leg add-margin MDP. RETURN-ONLY: uses pdfReturn integrals exclusively
//! (no Mean/Var/MeanOfMean/VarOfMean, no RF, no news, no volume, no MVO). Runs
//! only after one leg has liquidated and only when triggered.

use crate::artifacts::IntegralCache;
use crate::pricing::{LegState, Side};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MdpAction {
    Close,
    AddMarginContinue,
}

#[derive(Debug, Clone)]
pub struct MdpDecision {
    pub triggered: bool,
    pub action: MdpAction,
    pub x_best: f64,
    pub continue_value: f64,
    pub close_value: f64,
    pub candidates: Vec<(f64, f64)>,
}

pub fn mdp_trigger(current_price: f64, remaining_liq_price: f64, first_liq_price: f64) -> bool {
    (current_price - remaining_liq_price).abs() < (current_price - first_liq_price).abs()
}

pub fn enumerate_actions(
    remaining_balance_before: f64,
    add_margin_step: f64,
    max_add_margin_per_decision: f64,
    max_total_added_margin: f64,
    current_total_added_margin: f64,
) -> Vec<f64> {
    let rb = remaining_balance_before;
    let step = add_margin_step;
    if step <= 0.0 || rb < step {
        return vec![0.0];
    }
    let mut upper = rb;
    upper = upper.min(max_add_margin_per_decision);
    upper = upper.min(max_total_added_margin - current_total_added_margin);
    if upper < 0.0 {
        upper = 0.0;
    }

    let mut actions = vec![0.0];
    let mut k = 1i64;
    loop {
        let x = k as f64 * step;
        if x > upper + 1e-12 {
            break;
        }
        if rb - x < -1e-12 {
            break;
        }
        actions.push(x);
        k += 1;
    }
    actions
}

fn continuation_integral(
    cache: &IntegralCache,
    leg: &LegState,
    x: f64,
    y: f64,
    expected_costs: f64,
) -> f64 {
    let n = leg.notional();
    let denom = if cache.denom == 0.0 { 1.0 } else { cache.denom };
    let liq_z_x = leg.liq_z_after_add(x);
    match leg.side {
        Side::Long => {
            let right = n * cache.integral_long("return", y, cache.z_max()) / denom;
            let left = n * cache.integral_long("return", liq_z_x, y) / denom;
            right - left - expected_costs
        }
        Side::Short => {
            let left = n * cache.integral_short("return", cache.z_min(), y) / denom;
            let right = n * cache.integral_short("return", y, liq_z_x) / denom;
            left - right - expected_costs
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn decide(
    cache: &IntegralCache,
    leg: &LegState,
    current_price: f64,
    remaining_balance_before: f64,
    first_liq_price: f64,
    add_margin_step: f64,
    max_add_margin_per_decision: f64,
    max_total_added_margin: f64,
    current_total_added_margin: f64,
    expected_costs: f64,
) -> MdpDecision {
    let remaining_liq_price = leg.liquidation_price_level();
    let triggered = mdp_trigger(current_price, remaining_liq_price, first_liq_price);

    let close_value = leg.current_pnl(current_price) + remaining_balance_before;

    if !triggered {
        return MdpDecision {
            triggered: false,
            action: MdpAction::Close,
            x_best: 0.0,
            continue_value: f64::NAN,
            close_value,
            candidates: vec![],
        };
    }

    let y = leg.current_return(current_price);
    let actions = enumerate_actions(
        remaining_balance_before,
        add_margin_step,
        max_add_margin_per_decision,
        max_total_added_margin,
        current_total_added_margin,
    );

    let mut candidates: Vec<(f64, f64)> = Vec::with_capacity(actions.len());
    for x in actions {
        let rb_x = remaining_balance_before - x;
        let cont = continuation_integral(cache, leg, x, y, expected_costs);
        candidates.push((x, rb_x + cont));
    }

    // Best X: max value, tie-break to the smaller X.
    let mut best = candidates[0];
    for &(x, val) in candidates.iter().skip(1) {
        if val > best.1 + 1e-15 || ((val - best.1).abs() <= 1e-15 && x < best.0) {
            best = (x, val);
        }
    }

    let (action, chosen_x) = if close_value >= best.1 {
        (MdpAction::Close, 0.0)
    } else {
        (MdpAction::AddMarginContinue, best.0)
    };

    MdpDecision {
        triggered: true,
        action,
        x_best: chosen_x,
        continue_value: best.1,
        close_value,
        candidates,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::IntegralCache;
    use std::collections::HashMap;

    fn cache_return_only() -> IntegralCache {
        let grid = vec![-0.2, -0.1, 0.0, 0.1, 0.2];
        let mut cum_long = HashMap::new();
        // Monotone cumulative arrays (only the "return" component exists).
        cum_long.insert("return".to_string(), vec![0.0, 0.5, 1.0, 1.8, 3.0]);
        let mut cum_short = HashMap::new();
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
    fn trigger_logic() {
        assert!(mdp_trigger(100.0, 98.0, 90.0));
        assert!(!mdp_trigger(100.0, 85.0, 95.0));
    }

    #[test]
    fn actions_integer_multiples_and_bounded() {
        let a = enumerate_actions(55.0, 10.0, 1e18, 1e18, 0.0);
        assert_eq!(a, vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0]);
        assert!(a.iter().cloned().fold(0.0, f64::max) <= 55.0);
    }

    #[test]
    fn actions_only_zero_when_balance_low() {
        assert_eq!(enumerate_actions(5.0, 10.0, 1e18, 1e18, 0.0), vec![0.0]);
    }

    #[test]
    fn not_triggered_does_not_search() {
        let cache = cache_return_only();
        let leg = LegState::new(Side::Long, 100.0, 10.0, 200.0); // liq price 80
        let d = decide(&cache, &leg, 100.0, 500.0, 99.0, 10.0, 1e18, 1e18, 0.0, 0.0);
        assert!(!d.triggered);
        assert_eq!(d.action, MdpAction::Close);
        assert_eq!(d.x_best, 0.0);
    }

    #[test]
    fn uses_only_return_component() {
        // Cache has no mean/var components; decide must still work.
        let cache = cache_return_only();
        let leg = LegState::new(Side::Long, 100.0, 10.0, 50.0); // liq price 95
        let d = decide(&cache, &leg, 92.0, 500.0, 80.0, 10.0, 1e18, 1e18, 0.0, 0.0);
        assert!(d.triggered);
        assert!((d.x_best % 10.0).abs() < 1e-9);
    }
}
