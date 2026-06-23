//! Canonical MVO portfolio-weight sizing.
//!
//! The MVO + Branch-and-Bound artifact is the ONLY source of position sizing.
//! There is no fixed %-of-balance reserve, no net-PnL-based resizing and no
//! legacy baseline capital formula. For each symbol:
//!
//! ```text
//! total_equity         = wallet_balance + unrealized_pnl
//! target_margin[sym]   = total_equity * mvo_weight[sym]
//! target_notional[sym] = target_margin[sym] * permitted_leverage[sym]
//! ```
//!
//! The hedged architecture splits the symbol margin EQUALLY between the long and
//! short leg (`leg_margin = target_margin / 2`). Hard caps (MAX_ORDER_NOTIONAL,
//! exchange step-size / min-qty / min-notional) only ever REDUCE an order;
//! capital a cap prevents from being deployed stays as cash and is never shifted
//! to another symbol.

#[derive(Debug, Clone)]
pub struct SymbolFilters {
    pub step_size: f64,
    pub min_qty: f64,
    pub min_notional: f64,
}

impl Default for SymbolFilters {
    fn default() -> Self {
        Self {
            step_size: 0.001,
            min_qty: 0.0,
            min_notional: 0.0,
        }
    }
}

pub fn round_down_to_step(value: f64, step: f64) -> f64 {
    if step <= 0.0 {
        return value;
    }
    (value / step).floor() * step
}

/// One leg (long or short) sizing result.
#[derive(Debug, Clone)]
pub struct LegAllocation {
    /// `leg_margin * leverage`, before any hard cap.
    pub target_notional: f64,
    /// `target_notional` after the MAX_ORDER_NOTIONAL cap.
    pub capped_notional: f64,
    /// Order quantity after step / min filters; `None` when nothing tradeable.
    pub qty: Option<f64>,
    /// Why the realized notional is below `target_notional` (cash left behind).
    pub cap_reason: Option<String>,
}

impl LegAllocation {
    pub fn realized_notional(&self, price: f64) -> f64 {
        self.qty.unwrap_or(0.0) * price
    }
}

/// Size a single hedged leg from its margin budget.
pub fn size_leg(
    leg_margin: f64,
    leverage: f64,
    price: f64,
    filters: &SymbolFilters,
    max_order_notional_usdt: f64,
) -> LegAllocation {
    let lev = leverage.max(1.0);
    let target_notional = leg_margin.max(0.0) * lev;

    let mut capped = target_notional;
    let mut cap_reason: Option<String> = None;
    if max_order_notional_usdt > 0.0 && capped > max_order_notional_usdt {
        capped = max_order_notional_usdt;
        cap_reason = Some("max_order_notional_usdt".to_string());
    }

    let qty = if capped <= 0.0 || price <= 0.0 {
        if cap_reason.is_none() && target_notional > 0.0 {
            cap_reason = Some("zero_price".to_string());
        }
        None
    } else {
        let q = round_down_to_step(capped / price, filters.step_size);
        if q <= 0.0 || q < filters.min_qty {
            cap_reason = Some("min_qty".to_string());
            None
        } else if q * price < filters.min_notional {
            cap_reason = Some("min_notional".to_string());
            None
        } else {
            Some(q)
        }
    };

    LegAllocation {
        target_notional,
        capped_notional: capped,
        qty,
        cap_reason,
    }
}

/// Full per-symbol target allocation (hedged long + short).
#[derive(Debug, Clone)]
pub struct SymbolAllocation {
    pub weight: f64,
    /// `total_equity * weight`.
    pub target_margin: f64,
    /// `target_margin * leverage` (both legs combined).
    pub target_notional: f64,
    pub long: LegAllocation,
    pub short: LegAllocation,
    pub cap_reason: Option<String>,
}

impl SymbolAllocation {
    pub fn realized_notional(&self, price: f64) -> f64 {
        self.long.realized_notional(price) + self.short.realized_notional(price)
    }

    pub fn realized_margin(&self, price: f64, leverage: f64) -> f64 {
        self.realized_notional(price) / leverage.max(1.0)
    }

    /// Margin that the MVO target asked for but a hard cap / filter prevented
    /// from being deployed. This stays as cash.
    pub fn uninvested_margin(&self, price: f64, leverage: f64) -> f64 {
        (self.target_margin - self.realized_margin(price, leverage)).max(0.0)
    }

    pub fn total_qty(&self) -> f64 {
        self.long.qty.unwrap_or(0.0) + self.short.qty.unwrap_or(0.0)
    }
}

/// Canonical sizing: `target_margin = total_equity * weight`, split equally
/// between the long and short leg, each leg notional = `leg_margin * leverage`.
pub fn allocate_symbol(
    total_equity: f64,
    weight: f64,
    leverage: f64,
    price: f64,
    filters: &SymbolFilters,
    max_order_notional_usdt: f64,
) -> SymbolAllocation {
    let target_margin = (total_equity * weight).max(0.0);
    let leg_margin = target_margin / 2.0;
    let long = size_leg(
        leg_margin,
        leverage,
        price,
        filters,
        max_order_notional_usdt,
    );
    let short = size_leg(
        leg_margin,
        leverage,
        price,
        filters,
        max_order_notional_usdt,
    );
    let target_notional = target_margin * leverage.max(1.0);
    let cap_reason = long.cap_reason.clone().or_else(|| short.cap_reason.clone());
    SymbolAllocation {
        weight,
        target_margin,
        target_notional,
        long,
        short,
        cap_reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_down_applied() {
        assert!((round_down_to_step(1.2345, 0.001) - 1.234).abs() < 1e-9);
        assert!((round_down_to_step(7.0, 0.5) - 7.0).abs() < 1e-9);
    }

    #[test]
    fn target_margin_is_equity_times_weight() {
        let f = SymbolFilters::default();
        let a = allocate_symbol(1000.0, 0.4, 1.0, 100.0, &f, 0.0);
        assert!((a.target_margin - 400.0).abs() < 1e-9);
        assert!((a.target_notional - 400.0).abs() < 1e-9);
        // Equal hedged split: 200 margin per leg -> 2.0 qty each at price 100.
        assert!((a.long.qty.unwrap() - 2.0).abs() < 1e-9);
        assert_eq!(a.long.qty, a.short.qty);
        // No cap -> fully deployed, nothing left as cash.
        assert!(a.uninvested_margin(100.0, 1.0).abs() < 1e-9);
    }

    #[test]
    fn leverage_scales_notional_not_margin() {
        let f = SymbolFilters::default();
        let a = allocate_symbol(1000.0, 0.2, 5.0, 100.0, &f, 0.0);
        // margin 200, notional 1000, per leg notional 500 -> qty 5.0.
        assert!((a.target_margin - 200.0).abs() < 1e-9);
        assert!((a.target_notional - 1000.0).abs() < 1e-9);
        assert!((a.long.qty.unwrap() - 5.0).abs() < 1e-9);
        assert!(a.uninvested_margin(100.0, 5.0).abs() < 1e-9);
    }

    #[test]
    fn max_order_notional_cap_leaves_cash() {
        let f = SymbolFilters::default();
        // Per-leg target notional = 200, cap at 150 -> realized 300 total,
        // realized margin 300 -> 100 stays as cash.
        let a = allocate_symbol(1000.0, 0.4, 1.0, 100.0, &f, 150.0);
        assert_eq!(a.cap_reason.as_deref(), Some("max_order_notional_usdt"));
        assert!((a.long.qty.unwrap() - 1.5).abs() < 1e-9);
        assert!((a.realized_margin(100.0, 1.0) - 300.0).abs() < 1e-9);
        assert!((a.uninvested_margin(100.0, 1.0) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn min_notional_reject_leaves_full_margin_as_cash() {
        let f = SymbolFilters {
            step_size: 0.001,
            min_qty: 0.0,
            min_notional: 1_000_000.0,
        };
        let a = allocate_symbol(1000.0, 0.4, 1.0, 100.0, &f, 0.0);
        assert!(a.long.qty.is_none() && a.short.qty.is_none());
        assert_eq!(a.cap_reason.as_deref(), Some("min_notional"));
        assert!((a.uninvested_margin(100.0, 1.0) - 400.0).abs() < 1e-9);
    }

    #[test]
    fn zero_weight_allocates_nothing() {
        let f = SymbolFilters::default();
        let a = allocate_symbol(1000.0, 0.0, 1.0, 100.0, &f, 0.0);
        assert!((a.target_margin).abs() < 1e-9);
        assert!(a.long.qty.is_none() && a.short.qty.is_none());
    }

    #[test]
    fn cash_weight_zero_distributes_full_equity() {
        // Two symbols with weights summing to 1.0 (cash_weight 0) -> the sum of
        // target margins equals total equity.
        let f = SymbolFilters::default();
        let a = allocate_symbol(1000.0, 0.6, 1.0, 100.0, &f, 0.0);
        let b = allocate_symbol(1000.0, 0.4, 1.0, 100.0, &f, 0.0);
        assert!((a.target_margin + b.target_margin - 1000.0).abs() < 1e-9);
    }
}
