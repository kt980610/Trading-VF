//! Real Binance USD-M Futures client (compiled only with the `live-http`
//! feature). Hedge mode is assumed: LONG/SHORT positionSide are sent explicitly.

use hmac::{Hmac, Mac};
use sha2::Sha256;

use crate::config::ApiCredentials;
use crate::pricing::Side;
use crate::sizing::SymbolFilters;

use super::{ExchangeClient, OrderRequest, OrderResponse, PositionInfo};

type HmacSha256 = Hmac<Sha256>;

pub struct BinanceClient {
    creds: ApiCredentials,
    http: reqwest::blocking::Client,
}

fn to_query(params: &[(&str, String)]) -> String {
    params
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("&")
}

fn sign(secret: &str, query: &str) -> String {
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).expect("hmac key");
    mac.update(query.as_bytes());
    hex::encode(mac.finalize().into_bytes())
}

fn parse_f64(value: &serde_json::Value, key: &str) -> f64 {
    match value.get(key) {
        Some(serde_json::Value::String(s)) => s.parse().unwrap_or(0.0),
        Some(serde_json::Value::Number(n)) => n.as_f64().unwrap_or(0.0),
        _ => 0.0,
    }
}

impl BinanceClient {
    pub fn new(creds: ApiCredentials) -> anyhow::Result<Self> {
        let http = reqwest::blocking::Client::builder().build()?;
        Ok(Self { creds, http })
    }

    fn timestamp_ms() -> i64 {
        crate::clock::now_millis()
    }

    fn public_get(&self, path: &str, params: &[(&str, String)]) -> anyhow::Result<serde_json::Value> {
        let url = format!("{}{}", self.creds.base_url, path);
        let mut req = self.http.get(&url);
        if !params.is_empty() {
            req = req.query(params);
        }
        let resp = req.send()?.error_for_status()?;
        Ok(resp.json()?)
    }

    fn signed(
        &self,
        method: reqwest::Method,
        path: &str,
        mut params: Vec<(&str, String)>,
    ) -> anyhow::Result<serde_json::Value> {
        params.push(("timestamp", Self::timestamp_ms().to_string()));
        params.push(("recvWindow", "5000".to_string()));
        let query = to_query(&params);
        let signature = sign(&self.creds.api_secret, &query);
        let url = format!("{}{}?{}&signature={}", self.creds.base_url, path, query, signature);

        let resp = self
            .http
            .request(method, &url)
            .header("X-MBX-APIKEY", &self.creds.api_key)
            .send()?
            .error_for_status()?;
        Ok(resp.json()?)
    }

    pub fn ping(&self) -> anyhow::Result<()> {
        self.public_get("/fapi/v1/ping", &[])?;
        Ok(())
    }
}

impl ExchangeClient for BinanceClient {
    fn total_wallet_balance(&self) -> anyhow::Result<f64> {
        let v = self.signed(reqwest::Method::GET, "/fapi/v2/account", vec![])?;
        Ok(parse_f64(&v, "totalWalletBalance"))
    }

    fn positions(&self) -> anyhow::Result<Vec<PositionInfo>> {
        let v = self.signed(reqwest::Method::GET, "/fapi/v2/positionRisk", vec![])?;
        let mut out = Vec::new();
        if let Some(arr) = v.as_array() {
            for p in arr {
                let position_side = match p.get("positionSide").and_then(|s| s.as_str()) {
                    Some("SHORT") => Side::Short,
                    _ => Side::Long,
                };
                let liq = parse_f64(p, "liquidationPrice");
                out.push(PositionInfo {
                    symbol: p.get("symbol").and_then(|s| s.as_str()).unwrap_or("").to_string(),
                    position_side,
                    position_amt: parse_f64(p, "positionAmt"),
                    entry_price: parse_f64(p, "entryPrice"),
                    leverage: parse_f64(p, "leverage"),
                    liquidation_price: if liq > 0.0 { Some(liq) } else { None },
                    unrealized_pnl: parse_f64(p, "unRealizedProfit"),
                    isolated_margin: parse_f64(p, "isolatedMargin"),
                });
            }
        }
        Ok(out)
    }

    fn symbol_filters(&self, symbol: &str) -> anyhow::Result<SymbolFilters> {
        let v = self.public_get("/fapi/v1/exchangeInfo", &[("symbol", symbol.to_string())])?;
        let mut filters = SymbolFilters::default();
        if let Some(symbols) = v.get("symbols").and_then(|s| s.as_array()) {
            if let Some(sym) = symbols.iter().find(|s| s.get("symbol").and_then(|x| x.as_str()) == Some(symbol)) {
                if let Some(arr) = sym.get("filters").and_then(|f| f.as_array()) {
                    for f in arr {
                        match f.get("filterType").and_then(|t| t.as_str()) {
                            Some("LOT_SIZE") => {
                                filters.step_size = parse_f64(f, "stepSize");
                                filters.min_qty = parse_f64(f, "minQty");
                            }
                            Some("MIN_NOTIONAL") => {
                                filters.min_notional = parse_f64(f, "notional");
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        Ok(filters)
    }

    fn mark_price(&self, symbol: &str) -> anyhow::Result<f64> {
        let v = self.public_get("/fapi/v1/premiumIndex", &[("symbol", symbol.to_string())])?;
        Ok(parse_f64(&v, "markPrice"))
    }

    fn set_leverage(&self, symbol: &str, leverage: u32) -> anyhow::Result<()> {
        self.signed(
            reqwest::Method::POST,
            "/fapi/v1/leverage",
            vec![("symbol", symbol.to_string()), ("leverage", leverage.to_string())],
        )?;
        Ok(())
    }

    fn place_market_order(&self, req: &OrderRequest) -> anyhow::Result<OrderResponse> {
        let mut params = vec![
            ("symbol", req.symbol.clone()),
            ("side", req.side.as_str().to_string()),
            ("positionSide", req.position_side.position_side().to_string()),
            ("type", "MARKET".to_string()),
            ("quantity", format!("{}", req.quantity)),
        ];
        if req.reduce_only {
            params.push(("reduceOnly", "true".to_string()));
        }
        let v = self.signed(reqwest::Method::POST, "/fapi/v1/order", params)?;
        Ok(OrderResponse {
            order_id: v
                .get("orderId")
                .map(|x| x.to_string())
                .unwrap_or_default(),
            status: v.get("status").and_then(|s| s.as_str()).unwrap_or("").to_string(),
        })
    }

    fn cancel_open_orders(&self, symbol: &str) -> anyhow::Result<()> {
        self.signed(
            reqwest::Method::DELETE,
            "/fapi/v1/allOpenOrders",
            vec![("symbol", symbol.to_string())],
        )?;
        Ok(())
    }

    fn add_isolated_margin(&self, symbol: &str, position_side: Side, amount: f64) -> anyhow::Result<()> {
        // type=1 -> add isolated margin.
        self.signed(
            reqwest::Method::POST,
            "/fapi/v1/positionMargin",
            vec![
                ("symbol", symbol.to_string()),
                ("positionSide", position_side.position_side().to_string()),
                ("amount", format!("{amount}")),
                ("type", "1".to_string()),
            ],
        )?;
        Ok(())
    }
}
