//! Reader for `portfolio_weights.json` produced by the MVO + B&B module.
//!
//! This artifact is the SOLE source of position sizing. The live engine reads
//! the per-symbol `weight_discrete` (the MVO margin fraction of total equity) and
//! enforces fail-safe gates (`sum(weights) + cash_weight == 1.0`, freshness,
//! `weight_semantics` and `portfolio_version`) before opening any new position.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

/// The only weight semantics the sizing engine understands.
pub const EXPECTED_WEIGHT_SEMANTICS: &str = "margin_fraction_of_total_equity";

#[derive(Debug, Clone, Deserialize)]
pub struct SymbolWeight {
    #[serde(default)]
    pub valid: bool,
    #[serde(default)]
    pub weight_discrete: f64,
    #[serde(default)]
    pub long_weight: f64,
    #[serde(default)]
    pub short_weight: f64,
    #[serde(default)]
    pub reason: Option<String>,
}

impl SymbolWeight {
    /// A symbol is tradeable only if valid and with positive discrete weight.
    pub fn tradeable(&self) -> bool {
        self.valid && self.weight_discrete > 0.0
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct PortfolioWeights {
    #[serde(default)]
    pub as_of_date: String,
    #[serde(default)]
    pub symbols: HashMap<String, SymbolWeight>,
    #[serde(default)]
    pub sum_weight_discrete: f64,
    #[serde(default)]
    pub cash_weight: f64,
    /// Canonical sizing provenance (new schema). Older artifacts omit these.
    #[serde(default)]
    pub weight_semantics: String,
    #[serde(default)]
    pub portfolio_version: String,
    #[serde(default)]
    pub selected_symbols: Vec<String>,
}

impl PortfolioWeights {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let w: PortfolioWeights = serde_json::from_str(&text)?;
        Ok(w)
    }

    pub fn get(&self, symbol: &str) -> Option<&SymbolWeight> {
        self.symbols.get(symbol)
    }

    /// MVO margin fraction for a symbol; `0.0` for unknown / non-tradeable.
    pub fn weight_for(&self, symbol: &str) -> f64 {
        match self.symbols.get(symbol) {
            Some(sw) if sw.tradeable() => sw.weight_discrete,
            _ => 0.0,
        }
    }

    /// `sum(weight_discrete) + cash_weight == 1.0` within tolerance.
    pub fn sum_valid(&self, tol: f64) -> bool {
        let sum: f64 = self.symbols.values().map(|s| s.weight_discrete).sum();
        (sum + self.cash_weight - 1.0).abs() <= tol
    }

    /// The artifact `as_of_date` is no older than `max_age_days` relative to
    /// `today` (a `YYYY-MM-DD` string). A future-dated artifact is also rejected.
    pub fn is_fresh(&self, today: &str, max_age_days: i64) -> bool {
        let asof = crate::clock::parse_rfc3339_secs(&format!("{}T00:00:00Z", self.as_of_date));
        let now = crate::clock::parse_rfc3339_secs(&format!("{today}T00:00:00Z"));
        match (asof, now) {
            (Some(a), Some(n)) => {
                let age = n - a;
                age >= 0 && age <= max_age_days.max(0) * 86400
            }
            _ => false,
        }
    }

    /// `weight_semantics` matches the engine's expectation. An empty string is
    /// accepted for backward compatibility with pre-canonical artifacts.
    pub fn semantics_ok(&self) -> bool {
        self.weight_semantics.is_empty() || self.weight_semantics == EXPECTED_WEIGHT_SEMANTICS
    }

    /// `portfolio_version` matches `expected`. An empty `expected` accepts any.
    pub fn version_ok(&self, expected: &str) -> bool {
        expected.is_empty() || self.portfolio_version == expected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> PortfolioWeights {
        let json = r#"{
            "as_of_date":"2024-09-29",
            "symbols":{
                "BTCUSDT":{"valid":true,"weight_discrete":0.22,"long_weight":0.11,"short_weight":0.11},
                "ETHUSDT":{"valid":false,"reason":"not_enough_return_days","weight_discrete":0.0}
            },
            "sum_weight_discrete":0.22,
            "cash_weight":0.78,
            "weight_semantics":"margin_fraction_of_total_equity",
            "portfolio_version":"mvo_bnb_v1",
            "selected_symbols":["BTCUSDT"]
        }"#;
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn parses_and_filters() {
        let w = sample();
        assert!(w.get("BTCUSDT").unwrap().tradeable());
        assert!(!w.get("ETHUSDT").unwrap().tradeable());
        assert!((w.weight_for("BTCUSDT") - 0.22).abs() < 1e-12);
        assert_eq!(w.weight_for("ETHUSDT"), 0.0);
        assert_eq!(w.weight_for("UNKNOWN"), 0.0);
    }

    #[test]
    fn sum_validation() {
        let w = sample();
        assert!(w.sum_valid(1e-6));
        let mut bad = sample();
        bad.cash_weight = 0.50; // 0.22 + 0.50 = 0.72 != 1.0
        assert!(!bad.sum_valid(1e-6));
    }

    #[test]
    fn freshness_and_version_and_semantics() {
        let w = sample();
        assert!(w.is_fresh("2024-09-30", 2));
        assert!(!w.is_fresh("2024-10-15", 2)); // too old
        assert!(!w.is_fresh("2024-09-28", 2)); // future-dated
        assert!(w.semantics_ok());
        assert!(w.version_ok("mvo_bnb_v1"));
        assert!(!w.version_ok("other"));
        assert!(w.version_ok(""));

        let mut wrong = sample();
        wrong.weight_semantics = "notional_fraction".to_string();
        assert!(!wrong.semantics_ok());
    }

    #[test]
    fn legacy_artifact_without_semantics_is_accepted() {
        let json = r#"{
            "as_of_date":"2024-09-29",
            "symbols":{"BTCUSDT":{"valid":true,"weight_discrete":0.2}},
            "sum_weight_discrete":0.2,
            "cash_weight":0.8
        }"#;
        let w: PortfolioWeights = serde_json::from_str(json).unwrap();
        assert!(w.semantics_ok());
        assert!(w.version_ok(""));
    }
}
