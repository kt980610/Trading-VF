//! Exchange abstraction. The live engine talks to this trait so the decision
//! logic is testable with a mock and the real Binance client is swappable.

pub mod mock;

#[cfg(feature = "live-http")]
pub mod binance;

use crate::pricing::Side;
use crate::sizing::SymbolFilters;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl OrderSide {
    pub fn as_str(&self) -> &'static str {
        match self {
            OrderSide::Buy => "BUY",
            OrderSide::Sell => "SELL",
        }
    }
}

#[derive(Debug, Clone)]
pub struct OrderRequest {
    pub symbol: String,
    pub side: OrderSide,
    pub position_side: Side,
    pub quantity: f64,
    pub reduce_only: bool,
}

impl OrderRequest {
    /// Open long = BUY/LONG; open short = SELL/SHORT.
    pub fn open(symbol: &str, side: Side, quantity: f64) -> Self {
        let order_side = match side {
            Side::Long => OrderSide::Buy,
            Side::Short => OrderSide::Sell,
        };
        Self {
            symbol: symbol.to_string(),
            side: order_side,
            position_side: side,
            quantity,
            reduce_only: false,
        }
    }

    /// reduceOnly close: close long = SELL/LONG; close short = BUY/SHORT.
    pub fn close(symbol: &str, side: Side, quantity: f64) -> Self {
        let order_side = match side {
            Side::Long => OrderSide::Sell,
            Side::Short => OrderSide::Buy,
        };
        Self {
            symbol: symbol.to_string(),
            side: order_side,
            position_side: side,
            quantity,
            reduce_only: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OrderResponse {
    pub order_id: String,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct PositionInfo {
    pub symbol: String,
    pub position_side: Side,
    pub position_amt: f64,
    pub entry_price: f64,
    pub leverage: f64,
    pub liquidation_price: Option<f64>,
    pub unrealized_pnl: f64,
    pub isolated_margin: f64,
}

pub trait ExchangeClient {
    fn total_wallet_balance(&self) -> anyhow::Result<f64>;
    fn positions(&self) -> anyhow::Result<Vec<PositionInfo>>;
    fn symbol_filters(&self, symbol: &str) -> anyhow::Result<SymbolFilters>;
    fn mark_price(&self, symbol: &str) -> anyhow::Result<f64>;
    fn set_leverage(&self, symbol: &str, leverage: u32) -> anyhow::Result<()>;
    fn place_market_order(&self, req: &OrderRequest) -> anyhow::Result<OrderResponse>;
    fn cancel_open_orders(&self, symbol: &str) -> anyhow::Result<()>;
    fn add_isolated_margin(&self, symbol: &str, position_side: Side, amount: f64)
        -> anyhow::Result<()>;
}
