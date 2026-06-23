//! Central, single-threaded order execution layer.
//!
//! This is the ONLY place that talks to the exchange for order placement. It
//! drains a priority-sorted queue of `OrderIntent`s, runs every safety check via
//! the `RiskManager`, honours shadow mode and an optional rate limit, then sends
//! to Binance. Parallel decision workers never reach this layer directly.

use std::time::{Duration, Instant};

use crate::exchange::{ExchangeClient, OrderRequest};
use crate::order::{OrderIntent, OrderIntentType};
use crate::risk::RiskManager;
use crate::sizing::SymbolFilters;

#[derive(Debug, Clone, Default)]
pub struct ExecOutcome {
    pub order_ids: Vec<String>,
    pub sent: bool,
    pub skipped_reason: Option<String>,
}

impl ExecOutcome {
    fn skipped(reason: impl Into<String>) -> Self {
        ExecOutcome {
            order_ids: vec![],
            sent: false,
            skipped_reason: Some(reason.into()),
        }
    }
}

/// Per-intent context the executor needs that lives outside the intent itself.
#[derive(Debug, Clone, Default)]
pub struct ExecContext {
    pub position_open: bool,
    pub price: f64,
    pub filters: SymbolFilters,
}

pub struct OrderExecutor<'a, E: ExchangeClient> {
    exchange: &'a E,
    rate_limit_enabled: bool,
    min_spacing: Option<Duration>,
    last_send: Option<Instant>,
    pub sent_count: usize,
}

impl<'a, E: ExchangeClient> OrderExecutor<'a, E> {
    pub fn new(exchange: &'a E, rate_limit_enabled: bool, max_orders_per_second: f64) -> Self {
        let min_spacing = if rate_limit_enabled && max_orders_per_second > 0.0 {
            Some(Duration::from_secs_f64(1.0 / max_orders_per_second))
        } else {
            None
        };
        Self {
            exchange,
            rate_limit_enabled,
            min_spacing,
            last_send: None,
            sent_count: 0,
        }
    }

    fn rate_limit_wait(&mut self) {
        if !self.rate_limit_enabled {
            return;
        }
        if let (Some(spacing), Some(last)) = (self.min_spacing, self.last_send) {
            let elapsed = last.elapsed();
            if elapsed < spacing {
                std::thread::sleep(spacing - elapsed);
            }
        }
    }

    /// Run all pre-send checks then (if enabled and approved) send the order.
    pub fn execute(&mut self, intent: &OrderIntent, ctx: &ExecContext, risk: &mut RiskManager) -> ExecOutcome {
        // 1-7) Risk gate: kill-switch, balance, qty, position-open, duplicate.
        if let Err(reason) = risk.approve(intent, ctx.position_open) {
            return ExecOutcome::skipped(reason);
        }

        // 8-9) Binance filters / minNotional only enforced on opens (reduceOnly
        // closes can be below minNotional and are still accepted by Binance).
        if intent.intent_type.is_open() {
            if let Some(q) = intent.quantity {
                if q < ctx.filters.min_qty {
                    return ExecOutcome::skipped("below_min_qty");
                }
                if ctx.price > 0.0 && q * ctx.price < ctx.filters.min_notional {
                    return ExecOutcome::skipped("below_min_notional");
                }
            }
        }

        // Shadow / not real money: intent handled but never sent to Binance.
        if !risk.orders_enabled() {
            return ExecOutcome::skipped("shadow");
        }

        self.rate_limit_wait();
        let outcome = self.send(intent);
        self.last_send = Some(Instant::now());
        outcome
    }

    fn send(&mut self, intent: &OrderIntent) -> ExecOutcome {
        match intent.intent_type {
            OrderIntentType::OpenLong | OrderIntentType::OpenShort => {
                let q = intent.quantity.unwrap_or(0.0);
                let req = OrderRequest::open(&intent.symbol, intent.position_side, q);
                self.place(&req)
            }
            OrderIntentType::CloseLong | OrderIntentType::CloseShort => {
                let q = intent.quantity.unwrap_or(0.0);
                let req = OrderRequest::close(&intent.symbol, intent.position_side, q);
                self.place(&req)
            }
            OrderIntentType::AddMarginLong | OrderIntentType::AddMarginShort => {
                let amount = intent.margin_amount.unwrap_or(0.0);
                match self.exchange.add_isolated_margin(&intent.symbol, intent.position_side, amount) {
                    Ok(()) => {
                        self.sent_count += 1;
                        ExecOutcome { order_ids: vec![], sent: true, skipped_reason: None }
                    }
                    Err(e) => ExecOutcome::skipped(format!("add_margin_error:{e}")),
                }
            }
            OrderIntentType::CancelOpenOrders => match self.exchange.cancel_open_orders(&intent.symbol) {
                Ok(()) => {
                    self.sent_count += 1;
                    ExecOutcome { order_ids: vec![], sent: true, skipped_reason: None }
                }
                Err(e) => ExecOutcome::skipped(format!("cancel_error:{e}")),
            },
        }
    }

    fn place(&mut self, req: &OrderRequest) -> ExecOutcome {
        match self.exchange.place_market_order(req) {
            Ok(resp) => {
                self.sent_count += 1;
                ExecOutcome { order_ids: vec![resp.order_id], sent: true, skipped_reason: None }
            }
            Err(e) => ExecOutcome::skipped(format!("order_error:{e}")),
        }
    }
}
