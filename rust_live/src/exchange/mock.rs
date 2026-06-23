//! In-memory mock exchange used by tests and shadow runs.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::pricing::Side;
use crate::sizing::SymbolFilters;

use super::{ExchangeClient, OrderRequest, OrderResponse, PositionInfo};

#[derive(Default)]
pub struct MockExchange {
    pub balance: Mutex<f64>,
    pub positions: Mutex<Vec<PositionInfo>>,
    pub filters: Mutex<HashMap<String, SymbolFilters>>,
    pub mark_prices: Mutex<HashMap<String, f64>>,
    pub placed_orders: Mutex<Vec<OrderRequest>>,
    pub cancels: Mutex<Vec<String>>,
    pub leverage_calls: Mutex<Vec<(String, u32)>>,
    pub margin_adds: Mutex<Vec<(String, Side, f64)>>,
    next_order_id: Mutex<u64>,
}

impl MockExchange {
    pub fn new(balance: f64) -> Self {
        Self {
            balance: Mutex::new(balance),
            next_order_id: Mutex::new(1),
            ..Default::default()
        }
    }

    pub fn set_balance(&self, balance: f64) {
        *self.balance.lock().unwrap() = balance;
    }

    pub fn set_filters(&self, symbol: &str, filters: SymbolFilters) {
        self.filters.lock().unwrap().insert(symbol.to_string(), filters);
    }

    pub fn set_mark_price(&self, symbol: &str, price: f64) {
        self.mark_prices.lock().unwrap().insert(symbol.to_string(), price);
    }

    pub fn placed_count(&self) -> usize {
        self.placed_orders.lock().unwrap().len()
    }

    pub fn placed_snapshot(&self) -> Vec<OrderRequest> {
        self.placed_orders.lock().unwrap().clone()
    }
}

impl ExchangeClient for MockExchange {
    fn total_wallet_balance(&self) -> anyhow::Result<f64> {
        Ok(*self.balance.lock().unwrap())
    }

    fn positions(&self) -> anyhow::Result<Vec<PositionInfo>> {
        Ok(self.positions.lock().unwrap().clone())
    }

    fn symbol_filters(&self, symbol: &str) -> anyhow::Result<SymbolFilters> {
        Ok(self
            .filters
            .lock()
            .unwrap()
            .get(symbol)
            .cloned()
            .unwrap_or_default())
    }

    fn mark_price(&self, symbol: &str) -> anyhow::Result<f64> {
        self.mark_prices
            .lock()
            .unwrap()
            .get(symbol)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("no mark price for {symbol}"))
    }

    fn set_leverage(&self, symbol: &str, leverage: u32) -> anyhow::Result<()> {
        self.leverage_calls
            .lock()
            .unwrap()
            .push((symbol.to_string(), leverage));
        Ok(())
    }

    fn place_market_order(&self, req: &OrderRequest) -> anyhow::Result<OrderResponse> {
        let mut id = self.next_order_id.lock().unwrap();
        let order_id = format!("mock-{}", *id);
        *id += 1;
        self.placed_orders.lock().unwrap().push(req.clone());
        Ok(OrderResponse {
            order_id,
            status: "FILLED".to_string(),
        })
    }

    fn cancel_open_orders(&self, symbol: &str) -> anyhow::Result<()> {
        self.cancels.lock().unwrap().push(symbol.to_string());
        Ok(())
    }

    fn add_isolated_margin(&self, symbol: &str, position_side: Side, amount: f64) -> anyhow::Result<()> {
        self.margin_adds
            .lock()
            .unwrap()
            .push((symbol.to_string(), position_side, amount));
        Ok(())
    }
}
