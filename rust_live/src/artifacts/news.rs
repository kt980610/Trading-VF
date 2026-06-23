//! Reader for `news_features_daily.jsonl`.
//!
//! Passes through ALL numeric fields per (symbol, date) so the Python-computed
//! correlation-weighted cross-coin news features (e.g.
//! `news_sentiment_from_BTCUSDT_weighted`, `weighted_symbol_news_count`) reach
//! the RF feature vector unchanged.

use std::collections::HashMap;
use std::path::Path;

const CANONICAL_KEYS: &[&str] = &[
    "macro_news_sentiment",
    "policy_news_sentiment",
    "stock_market_news_sentiment",
    "crypto_market_news_sentiment",
    "symbol_specific_news_sentiment",
    "macro_news_count",
    "policy_news_count",
    "stock_market_news_count",
    "crypto_market_news_count",
    "symbol_specific_news_count",
];

#[derive(Debug, Default)]
pub struct NewsFeatures {
    // symbol -> date -> {feature -> value}
    by_symbol: HashMap<String, HashMap<String, HashMap<String, f64>>>,
}

impl NewsFeatures {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let mut out = NewsFeatures::default();
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
            let symbol = v.get("symbol").and_then(|s| s.as_str()).unwrap_or("").to_string();
            let date = v.get("date").and_then(|s| s.as_str()).unwrap_or("").to_string();
            if date.is_empty() {
                continue;
            }
            let mut feats: HashMap<String, f64> = HashMap::new();
            if let Some(obj) = v.as_object() {
                for (k, val) in obj {
                    if k == "symbol" || k == "date" {
                        continue;
                    }
                    if let Some(f) = val.as_f64() {
                        feats.insert(k.clone(), f);
                    }
                }
            }
            out.by_symbol
                .entry(symbol)
                .or_default()
                .insert(date, feats);
        }
        Ok(out)
    }

    /// Build an in-memory `NewsFeatures` holding a single `(symbol, date)` entry
    /// from a precomputed as-of feature map. The live engine uses this to splice
    /// the freshest intraday news vector into the existing daily-keyed lookup
    /// path without changing any decision/feature signatures.
    pub fn from_asof(date: &str, symbol: &str, features: &HashMap<String, f64>) -> Self {
        let mut by_symbol: HashMap<String, HashMap<String, HashMap<String, f64>>> = HashMap::new();
        let mut dm: HashMap<String, HashMap<String, f64>> = HashMap::new();
        dm.insert(date.to_string(), features.clone());
        by_symbol.insert(symbol.to_string(), dm);
        NewsFeatures { by_symbol }
    }

    /// Features of the most recent COMPLETED day STRICTLY before `date` (the D-1
    /// leakage-safe join), preferring a symbol-specific row, then the universal
    /// ("") row. Empty when none exists. Daily dates are "YYYY-MM-DD" so plain
    /// lexicographic ordering equals chronological ordering. This mirrors the
    /// Python training join (`artifact_join.build_previous_day_provider`), so a
    /// minute on day D uses day D-1's news in BOTH training and live (and in
    /// historical replay the same-day row, if present, is excluded).
    pub fn previous_day_features(&self, symbol: &str, date: &str) -> HashMap<String, f64> {
        fn newest_before(
            m: &HashMap<String, HashMap<String, f64>>,
            date: &str,
        ) -> Option<HashMap<String, f64>> {
            m.iter()
                .filter(|(d, _)| d.as_str() < date)
                .max_by(|a, b| a.0.cmp(b.0))
                .map(|(_, f)| f.clone())
        }
        if let Some(m) = self.by_symbol.get(symbol) {
            if let Some(f) = newest_before(m, date) {
                return f;
            }
        }
        if let Some(m) = self.by_symbol.get("") {
            if let Some(f) = newest_before(m, date) {
                return f;
            }
        }
        HashMap::new()
    }

    /// All numeric news features for (symbol, date); canonical keys default to 0.
    pub fn get(&self, symbol: &str, date: &str) -> HashMap<String, f64> {
        let mut out: HashMap<String, f64> = CANONICAL_KEYS.iter().map(|k| ((*k).to_string(), 0.0)).collect();
        if let Some(feats) = self.by_symbol.get(symbol).and_then(|m| m.get(date)) {
            for (k, v) in feats {
                out.insert(k.clone(), *v);
            }
        } else if let Some(feats) = self.by_symbol.get("").and_then(|m| m.get(date)) {
            for (k, v) in feats {
                out.insert(k.clone(), *v);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build() -> NewsFeatures {
        // Universal ("") rows written by the GKG daily producer (no symbol).
        let mut feats = NewsFeatures::default();
        for (date, val) in [("2024-04-30", 1.0), ("2024-05-01", 2.0), ("2024-05-02", 3.0)] {
            let mut m: HashMap<String, f64> = HashMap::new();
            m.insert("gkg_btc_count".into(), val);
            feats.by_symbol.entry(String::new()).or_default().insert(date.into(), m);
        }
        feats
    }

    #[test]
    fn previous_day_uses_strictly_earlier_universal_row() {
        let nf = build();
        // On 2024-05-02 the D-1 join must pick 2024-05-01 (not the same day).
        let f = nf.previous_day_features("BTCUSDT", "2024-05-02");
        assert_eq!(f.get("gkg_btc_count").copied(), Some(2.0));
    }

    #[test]
    fn previous_day_before_first_is_empty() {
        let nf = build();
        assert!(nf.previous_day_features("BTCUSDT", "2024-04-30").is_empty());
    }

    #[test]
    fn previous_day_prefers_symbol_specific_over_universal() {
        let mut nf = build();
        let mut m: HashMap<String, f64> = HashMap::new();
        m.insert("gkg_btc_count".into(), 99.0);
        nf.by_symbol.entry("BTCUSDT".into()).or_default().insert("2024-05-01".into(), m);
        let f = nf.previous_day_features("BTCUSDT", "2024-05-02");
        assert_eq!(f.get("gkg_btc_count").copied(), Some(99.0));
    }
}
