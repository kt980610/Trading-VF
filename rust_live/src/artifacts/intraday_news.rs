//! Reader for the live as-of intraday news artifact
//! (`data/news_features_intraday.jsonl`).
//!
//! Each line is one `(asof_timestamp, symbol)` snapshot produced by the Python
//! live news worker:
//!
//! ```json
//! {"asof_timestamp":"2026-06-20T12:00:00Z","symbol":"BTCUSDT",
//!  "feature_version":"news_v1","macro_news_sentiment":0.1, ...}
//! ```
//!
//! The reader keeps every numeric feature so the full RF news schema reaches the
//! feature vector unchanged, and indexes records per symbol sorted by their real
//! `asof_timestamp` (NEVER the fetch time). The engine selects the latest record
//! at or before the decision instant and enforces a freshness budget before it
//! lets a symbol open a NEW position.

use std::collections::HashMap;
use std::path::Path;

use crate::clock;

#[derive(Debug, Clone)]
pub struct IntradayRecord {
    pub asof_secs: i64,
    pub asof_timestamp: String,
    pub feature_version: String,
    /// Provenance: "intraday_asof" | "previous_completed_day_fallback".
    pub news_mode: String,
    /// Provenance: the news provider that produced this record (e.g. "gdelt").
    pub news_source: Option<String>,
    /// Provenance: representative timestamp quality of the window
    /// ("exact_utc" = verified publish, "observed_utc" = observation, e.g. GDELT
    /// seendate). Never used to relax leakage; logged for auditability only.
    pub timestamp_quality: Option<String>,
    /// Source daily date for the fallback mode (None for intraday_asof).
    pub source_feature_date: Option<String>,
    pub features: HashMap<String, f64>,
}

#[derive(Debug, Default)]
pub struct IntradayNews {
    // symbol -> records sorted ascending by asof_secs.
    by_symbol: HashMap<String, Vec<IntradayRecord>>,
}

impl IntradayNews {
    /// Load the artifact. A missing file yields an empty (but valid) reader so a
    /// never-run worker is indistinguishable from "no fresh news" at the gate.
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let mut out = IntradayNews::default();
        if !path.as_ref().exists() {
            return Ok(out);
        }
        let text = std::fs::read_to_string(path)?;
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let v: serde_json::Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let symbol = v
                .get("symbol")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string();
            let asof_timestamp = v
                .get("asof_timestamp")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string();
            let asof_secs = match clock::parse_rfc3339_secs(&asof_timestamp) {
                Some(s) => s,
                None => continue, // unparseable as-of => skip (never a silent zero)
            };
            let feature_version = v
                .get("feature_version")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string();
            // Provenance (defaults keep older artifacts readable).
            let news_mode = v
                .get("news_mode")
                .and_then(|s| s.as_str())
                .unwrap_or("intraday_asof")
                .to_string();
            let news_source = v
                .get("news_source")
                .and_then(|s| s.as_str())
                .map(|s| s.to_string());
            let timestamp_quality = v
                .get("timestamp_quality")
                .and_then(|s| s.as_str())
                .map(|s| s.to_string());
            let source_feature_date = v
                .get("source_feature_date")
                .and_then(|s| s.as_str())
                .map(|s| s.to_string());
            let mut features: HashMap<String, f64> = HashMap::new();
            if let Some(obj) = v.as_object() {
                for (k, val) in obj {
                    if matches!(
                        k.as_str(),
                        "symbol"
                            | "asof_timestamp"
                            | "feature_version"
                            | "news_mode"
                            | "news_source"
                            | "news_timestamp_quality"
                            | "timestamp_quality"
                            | "published_at"
                            | "source_seen_at"
                            | "available_at"
                            | "source_feature_date"
                    ) {
                        continue;
                    }
                    if let Some(f) = val.as_f64() {
                        features.insert(k.clone(), f);
                    }
                }
            }
            out.by_symbol
                .entry(symbol)
                .or_default()
                .push(IntradayRecord {
                    asof_secs,
                    asof_timestamp,
                    feature_version,
                    news_mode,
                    news_source,
                    timestamp_quality,
                    source_feature_date,
                    features,
                });
        }
        for recs in out.by_symbol.values_mut() {
            recs.sort_by_key(|r| r.asof_secs);
        }
        Ok(out)
    }

    pub fn is_empty(&self) -> bool {
        self.by_symbol.is_empty()
    }

    /// The most recent record for `symbol` whose `asof_secs <= cutoff_secs`.
    ///
    /// The caller passes the leakage-adjusted cutoff
    /// (`decision_secs - news_safety_lag_seconds`), so a record is only returned
    /// when `asof_timestamp <= decision_timestamp - news_safety_lag_seconds`.
    /// Records newer than the cutoff are never returned (no look-ahead leakage).
    pub fn latest_before(&self, symbol: &str, cutoff_secs: i64) -> Option<&IntradayRecord> {
        let recs = self.by_symbol.get(symbol)?;
        recs.iter().rev().find(|r| r.asof_secs <= cutoff_secs)
    }

    /// Age in seconds of `record` relative to the decision instant (>= 0).
    pub fn age_secs(record: &IntradayRecord, decision_secs: i64) -> i64 {
        (decision_secs - record.asof_secs).max(0)
    }
}
