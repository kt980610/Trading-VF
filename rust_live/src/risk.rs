//! Centralized risk gate consulted by the `OrderExecutor` before every send.
//!
//! Holds the per-cycle account/risk snapshot and a duplicate-close guard so the
//! same symbol can never receive two close orders in one execution pass.

use std::collections::HashSet;

use crate::order::{OrderIntent, OrderIntentType};

#[derive(Debug, Clone)]
pub struct RiskManager {
    pub shadow_mode: bool,
    pub real_money: bool,
    pub min_balance: f64,
    pub kill_switch_active: bool,
    pub balance: f64,
    /// Guard against closing the SAME leg (symbol+side) twice in one pass. A
    /// hedged close legitimately closes both the long and short leg.
    closed_legs: HashSet<String>,
}

impl RiskManager {
    pub fn new(shadow_mode: bool, real_money: bool, min_balance: f64, kill_switch_active: bool, balance: f64) -> Self {
        Self {
            shadow_mode,
            real_money,
            min_balance,
            kill_switch_active,
            balance,
            closed_legs: HashSet::new(),
        }
    }

    /// Whether real Binance orders may be sent (real-money, non-shadow).
    pub fn orders_enabled(&self) -> bool {
        self.real_money && !self.shadow_mode
    }

    pub fn balance_ok(&self) -> bool {
        self.balance >= self.min_balance
    }

    /// Approve (or reject with a reason) an intent. Mutates the duplicate-close
    /// guard when a close is approved.
    pub fn approve(&mut self, intent: &OrderIntent, position_open: bool) -> Result<(), String> {
        // New risk is forbidden when killed or under-funded; flattening is allowed.
        if self.kill_switch_active && (intent.intent_type.is_open() || intent.intent_type.is_add_margin()) {
            return Err("kill_switch_active".into());
        }
        if !self.balance_ok() && (intent.intent_type.is_open() || intent.intent_type.is_add_margin()) {
            return Err("balance_below_min".into());
        }
        if let Some(q) = intent.quantity {
            if q <= 0.0 {
                return Err("non_positive_quantity".into());
            }
        }
        if intent.intent_type.is_add_margin() {
            match intent.margin_amount {
                Some(m) if m > 0.0 => {}
                _ => return Err("non_positive_margin".into()),
            }
        }
        if (intent.intent_type.is_close() || intent.intent_type.is_add_margin()) && !position_open {
            return Err("position_not_open".into());
        }
        if intent.intent_type.is_close() {
            let key = format!("{}:{}", intent.symbol, intent.position_side.position_side());
            if self.closed_legs.contains(&key) {
                return Err("duplicate_close".into());
            }
            self.closed_legs.insert(key);
        }
        let _ = OrderIntentType::CancelOpenOrders; // explicit: cancels are always allowed
        Ok(())
    }
}
