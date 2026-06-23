//! Kill-switch: halts trading and flattens positions when wallet balance is
//! too low. The trigger file blocks all trading until manually removed.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::RiskSettings;
use crate::exchange::{ExchangeClient, OrderRequest};
use crate::pricing::Side;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillSwitchRecord {
    pub triggered_at: String,
    pub reason: String,
    pub total_wallet_balance: f64,
    pub min_required: f64,
}

pub fn is_triggered_file_present(path: impl AsRef<Path>) -> bool {
    path.as_ref().exists()
}

pub fn balance_below_threshold(total_wallet_balance: f64, risk: &RiskSettings) -> bool {
    total_wallet_balance < risk.min_total_wallet_balance_usdt
}

pub fn write_trigger_file(
    path: impl AsRef<Path>,
    reason: &str,
    total_wallet_balance: f64,
    risk: &RiskSettings,
) -> anyhow::Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent)?;
    }
    let record = KillSwitchRecord {
        triggered_at: crate::clock::now_rfc3339(),
        reason: reason.to_string(),
        total_wallet_balance,
        min_required: risk.min_total_wallet_balance_usdt,
    };
    std::fs::write(path, serde_json::to_string_pretty(&record)?)?;
    Ok(())
}

/// Flatten everything: cancel open orders and reduceOnly-close every open
/// position. Honors the risk flags. Returns the list of placed order ids.
pub fn flatten_all(
    exchange: &dyn ExchangeClient,
    risk: &RiskSettings,
) -> anyhow::Result<Vec<String>> {
    let positions = exchange.positions()?;
    let mut order_ids = Vec::new();

    if risk.kill_switch_cancel_orders {
        let mut seen = std::collections::HashSet::new();
        for p in &positions {
            if seen.insert(p.symbol.clone()) {
                let _ = exchange.cancel_open_orders(&p.symbol);
            }
        }
    }

    if risk.kill_switch_close_positions {
        for p in &positions {
            if p.position_amt.abs() <= 0.0 {
                continue;
            }
            let side = match p.position_side {
                Side::Long => Side::Long,
                Side::Short => Side::Short,
            };
            let req = OrderRequest::close(&p.symbol, side, p.position_amt.abs());
            if let Ok(resp) = exchange.place_market_order(&req) {
                order_ids.push(resp.order_id);
            }
        }
    }

    Ok(order_ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_check() {
        let risk = RiskSettings::default();
        assert!(balance_below_threshold(99.99, &risk));
        assert!(!balance_below_threshold(100.0, &risk));
        assert!(!balance_below_threshold(150.0, &risk));
    }
}
