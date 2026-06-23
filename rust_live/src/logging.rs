//! Append-only JSONL logging of decisions and realized returns.

use std::io::Write;
use std::path::Path;

use serde::Serialize;

pub fn append_jsonl<T: Serialize>(path: impl AsRef<Path>, record: &T) -> anyhow::Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let line = serde_json::to_string(record)?;
    writeln!(file, "{line}")?;
    file.flush()?;
    Ok(())
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveDecisionRecord {
    pub timestamp: String,
    pub symbol: String,
    pub mode: String,
    pub current_price: f64,
    pub y: f64,
    pub rf_model_version: String,
    pub feature_schema_version: Option<String>,
    /// `p_close = predict_proba()[:,1]` (None when no classifier was available).
    pub rf_prediction: Option<f64>,
    pub rf_threshold: f64,
    pub decision: String,
    pub reason: String,
    pub long_edge_return: f64,
    pub short_edge_return: f64,
    pub predicted_daily_volume: f64,
    pub current_minute_volume: f64,
    pub news_sentiment_summary: f64,
    /// News provenance of the as-of record actually used (intraday news only).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub news_mode: Option<String>,
    /// News provider that produced the as-of record (e.g. "gdelt").
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub news_source: Option<String>,
    /// Quality of the as-of instant: "exact_utc" (verified publish) or
    /// "observed_utc" (observation time, e.g. GDELT seendate).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub news_timestamp_quality: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub news_asof_timestamp: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub news_source_feature_date: Option<String>,
    pub mdp_trigger_active: bool,
    pub mdp_action: Option<String>,
    pub margin_added: f64,
    pub order_ids: Vec<String>,
    pub realized_pnl_if_closed: Option<f64>,
    pub shadow: bool,
}

/// One line per symbol emitted by the daily MVO rebalance. Captures the full
/// sizing decision so cash left behind by a hard cap is auditable.
#[derive(Debug, Clone, Serialize)]
pub struct PortfolioAllocationRecord {
    pub timestamp: String,
    pub symbol: String,
    pub portfolio_version: String,
    pub as_of_date: String,
    pub total_equity: f64,
    pub cash_weight: f64,
    pub weight: f64,
    pub target_margin: f64,
    pub target_notional: f64,
    pub current_margin: f64,
    pub current_notional: f64,
    pub delta_notional: f64,
    pub realized_margin: f64,
    pub uninvested_margin: f64,
    pub action: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cap_reason: Option<String>,
    pub shadow: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct RealizedReturnRecord {
    pub date: String,
    pub symbol: String,
    pub realized_pnl: f64,
    pub fees: f64,
    pub funding: f64,
    pub slippage: f64,
    pub model_version: String,
    pub portfolio_weight_used: f64,
}
