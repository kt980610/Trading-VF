//! Order intents produced by decision workers and consumed by the central
//! `OrderExecutor`. Workers NEVER call the exchange directly; they only emit
//! `OrderIntent`s that the executor sends sequentially after risk checks.

use serde::Serialize;

use crate::decision::Action;
use crate::pricing::Side;
use crate::state::SymbolState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum OrderIntentType {
    OpenLong,
    OpenShort,
    CloseLong,
    CloseShort,
    AddMarginLong,
    AddMarginShort,
    CancelOpenOrders,
}

impl OrderIntentType {
    pub fn is_close(&self) -> bool {
        matches!(self, OrderIntentType::CloseLong | OrderIntentType::CloseShort)
    }
    pub fn is_open(&self) -> bool {
        matches!(self, OrderIntentType::OpenLong | OrderIntentType::OpenShort)
    }
    pub fn is_add_margin(&self) -> bool {
        matches!(self, OrderIntentType::AddMarginLong | OrderIntentType::AddMarginShort)
    }
}

/// Execution priority (lower value = executed first). Spec section 7.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum Priority {
    KillSwitchClose = 0,
    LiquidationClose = 1,
    MdpAddMargin = 2,
    RfClose = 3,
    NewDayOpen = 4,
}

#[derive(Debug, Clone, Serialize)]
pub struct OrderIntent {
    pub timestamp: String,
    pub symbol: String,
    pub intent_type: OrderIntentType,
    pub position_side: Side,
    pub quantity: Option<f64>,
    pub margin_amount: Option<f64>,
    pub reduce_only: bool,
    pub reason: String,
    pub model_version: String,
    pub decision_id: String,
    pub priority: Priority,
}

impl OrderIntent {
    #[allow(clippy::too_many_arguments)]
    fn close(symbol: &str, side: Side, qty: f64, ts: &str, reason: &str, mv: &str, id: &str, prio: Priority) -> Self {
        OrderIntent {
            timestamp: ts.to_string(),
            symbol: symbol.to_string(),
            intent_type: if side == Side::Long {
                OrderIntentType::CloseLong
            } else {
                OrderIntentType::CloseShort
            },
            position_side: side,
            quantity: Some(qty),
            margin_amount: None,
            reduce_only: true,
            reason: reason.to_string(),
            model_version: mv.to_string(),
            decision_id: id.to_string(),
            priority: prio,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn open(symbol: &str, side: Side, qty: f64, ts: &str, reason: &str, mv: &str, id: &str) -> Self {
        OrderIntent {
            timestamp: ts.to_string(),
            symbol: symbol.to_string(),
            intent_type: if side == Side::Long {
                OrderIntentType::OpenLong
            } else {
                OrderIntentType::OpenShort
            },
            position_side: side,
            quantity: Some(qty),
            margin_amount: None,
            reduce_only: false,
            reason: reason.to_string(),
            model_version: mv.to_string(),
            decision_id: id.to_string(),
            priority: Priority::NewDayOpen,
        }
    }
}

/// Translate a worker's `Action` into one or more `OrderIntent`s. Closes read
/// the leg quantities from the (pre-mutation) symbol state.
#[allow(clippy::too_many_arguments)]
pub fn intents_from_action(
    action: &Action,
    symbol: &str,
    state: &SymbolState,
    timestamp: &str,
    reason: &str,
    model_version: &str,
    decision_id: &str,
    priority: Priority,
) -> Vec<OrderIntent> {
    let mut out = Vec::new();
    match action {
        Action::Hold => {}
        Action::CloseHedged => {
            if let Some(l) = &state.long {
                out.push(OrderIntent::close(symbol, Side::Long, l.qty, timestamp, reason, model_version, decision_id, priority));
            }
            if let Some(s) = &state.short {
                out.push(OrderIntent::close(symbol, Side::Short, s.qty, timestamp, reason, model_version, decision_id, priority));
            }
        }
        Action::CloseLong => {
            if let Some(l) = &state.long {
                out.push(OrderIntent::close(symbol, Side::Long, l.qty, timestamp, reason, model_version, decision_id, priority));
            }
        }
        Action::CloseShort => {
            if let Some(s) = &state.short {
                out.push(OrderIntent::close(symbol, Side::Short, s.qty, timestamp, reason, model_version, decision_id, priority));
            }
        }
        Action::AddMarginLong(x) => {
            if *x > 0.0 {
                out.push(OrderIntent {
                    timestamp: timestamp.to_string(),
                    symbol: symbol.to_string(),
                    intent_type: OrderIntentType::AddMarginLong,
                    position_side: Side::Long,
                    quantity: None,
                    margin_amount: Some(*x),
                    reduce_only: false,
                    reason: reason.to_string(),
                    model_version: model_version.to_string(),
                    decision_id: decision_id.to_string(),
                    priority: Priority::MdpAddMargin,
                });
            }
        }
        Action::AddMarginShort(x) => {
            if *x > 0.0 {
                out.push(OrderIntent {
                    timestamp: timestamp.to_string(),
                    symbol: symbol.to_string(),
                    intent_type: OrderIntentType::AddMarginShort,
                    position_side: Side::Short,
                    quantity: None,
                    margin_amount: Some(*x),
                    reduce_only: false,
                    reason: reason.to_string(),
                    model_version: model_version.to_string(),
                    decision_id: decision_id.to_string(),
                    priority: Priority::MdpAddMargin,
                });
            }
        }
    }
    out
}

/// Priority for an RF/MDP close based on the decision reason.
pub fn close_priority(reason: &str) -> Priority {
    if reason.starts_with("mdp") {
        Priority::LiquidationClose
    } else {
        Priority::RfClose
    }
}
