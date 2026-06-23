//! Price / payoff / notional / liquidation-cutoff math.
//!
//! Invariants: `ScenarioPrice(z) = EntryPrice * (1 + z)`, long payoff = `z`,
//! short payoff = `-z`, and PnL is sized by the open notional `N` only. Adding
//! margin changes margin and the liquidation cutoff, never `N`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Long,
    Short,
}

impl Side {
    pub fn as_str(&self) -> &'static str {
        match self {
            Side::Long => "long",
            Side::Short => "short",
        }
    }
    pub fn position_side(&self) -> &'static str {
        match self {
            Side::Long => "LONG",
            Side::Short => "SHORT",
        }
    }
}

pub fn scenario_price(entry_price: f64, z: f64) -> f64 {
    entry_price * (1.0 + z)
}

pub fn long_payoff_pct(z: f64) -> f64 {
    z
}

pub fn short_payoff_pct(z: f64) -> f64 {
    -z
}

pub fn notional(qty: f64, entry_price: f64) -> f64 {
    qty * entry_price
}

pub fn long_liq_z(entry_price: f64, n: f64, m: f64, liq_price: Option<f64>) -> f64 {
    match liq_price {
        Some(p) => p / entry_price - 1.0,
        None => -m / n,
    }
}

pub fn short_liq_z(entry_price: f64, n: f64, m: f64, liq_price: Option<f64>) -> f64 {
    match liq_price {
        Some(p) => p / entry_price - 1.0,
        None => m / n,
    }
}

pub fn liquidation_price_from_cutoff(entry_price: f64, liq_z: f64) -> f64 {
    entry_price * (1.0 + liq_z)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegState {
    pub side: Side,
    pub entry_price: f64,
    pub qty: f64,
    pub margin_current: f64,
    #[serde(default)]
    pub liquidation_price: Option<f64>,
}

impl LegState {
    pub fn new(side: Side, entry_price: f64, qty: f64, margin_current: f64) -> Self {
        Self {
            side,
            entry_price,
            qty,
            margin_current,
            liquidation_price: None,
        }
    }

    pub fn notional(&self) -> f64 {
        notional(self.qty, self.entry_price)
    }

    pub fn current_return(&self, current_price: f64) -> f64 {
        current_price / self.entry_price - 1.0
    }

    pub fn payoff_pct(&self, z: f64) -> f64 {
        match self.side {
            Side::Long => long_payoff_pct(z),
            Side::Short => short_payoff_pct(z),
        }
    }

    pub fn liq_z(&self) -> f64 {
        match self.side {
            Side::Long => long_liq_z(self.entry_price, self.notional(), self.margin_current, self.liquidation_price),
            Side::Short => short_liq_z(self.entry_price, self.notional(), self.margin_current, self.liquidation_price),
        }
    }

    /// Liquidation cutoff after adding margin `x` (notional unchanged).
    pub fn liq_z_after_add(&self, x: f64) -> f64 {
        let m_x = self.margin_current + x;
        match self.side {
            Side::Long => long_liq_z(self.entry_price, self.notional(), m_x, None),
            Side::Short => short_liq_z(self.entry_price, self.notional(), m_x, None),
        }
    }

    pub fn current_pnl(&self, current_price: f64) -> f64 {
        let y = self.current_return(current_price);
        self.notional() * self.payoff_pct(y)
    }

    pub fn liquidation_price_level(&self) -> f64 {
        match self.liquidation_price {
            Some(p) => p,
            None => liquidation_price_from_cutoff(self.entry_price, self.liq_z()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenario_uses_entry() {
        assert!((scenario_price(100.0, 0.1) - 110.0).abs() < 1e-9);
    }

    #[test]
    fn payoff_signs() {
        assert_eq!(long_payoff_pct(0.2), 0.2);
        assert_eq!(short_payoff_pct(0.2), -0.2);
    }

    #[test]
    fn add_margin_keeps_notional() {
        let mut leg = LegState::new(Side::Long, 100.0, 10.0, 100.0);
        let n_before = leg.notional();
        leg.margin_current += 50.0;
        assert!((leg.notional() - n_before).abs() < 1e-12);
    }

    #[test]
    fn liq_z_approximations() {
        assert!((long_liq_z(100.0, 1000.0, 100.0, None) + 0.1).abs() < 1e-12);
        assert!((short_liq_z(100.0, 1000.0, 100.0, None) - 0.1).abs() < 1e-12);
    }
}
