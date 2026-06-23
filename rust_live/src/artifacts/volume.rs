//! Reader for `predicted_daily_volume.jsonl`.

use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Default)]
pub struct VolumeRecord {
    pub predicted_daily_volume: f64,
    pub previous_day_real_volume: f64,
}

#[derive(Debug, Default)]
pub struct VolumeFeatures {
    // symbol -> date -> record
    by_symbol: HashMap<String, HashMap<String, VolumeRecord>>,
    // symbol -> latest record (by lexical date order)
    latest: HashMap<String, (String, VolumeRecord)>,
}

fn f(value: &serde_json::Value, key: &str) -> f64 {
    value.get(key).and_then(|v| v.as_f64()).unwrap_or(0.0)
}

impl VolumeFeatures {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let mut out = VolumeFeatures::default();
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
            if symbol.is_empty() || date.is_empty() {
                continue;
            }
            let rec = VolumeRecord {
                predicted_daily_volume: f(&v, "predicted_daily_volume"),
                previous_day_real_volume: f(&v, "previous_day_real_volume"),
            };
            out.by_symbol
                .entry(symbol.clone())
                .or_default()
                .insert(date.clone(), rec.clone());
            let replace = out
                .latest
                .get(&symbol)
                .map(|(d, _)| date.as_str() > d.as_str())
                .unwrap_or(true);
            if replace {
                out.latest.insert(symbol, (date, rec));
            }
        }
        Ok(out)
    }

    pub fn get(&self, symbol: &str, date: &str) -> Option<&VolumeRecord> {
        self.by_symbol.get(symbol).and_then(|m| m.get(date))
    }

    /// Latest available record for a symbol when the exact date is missing.
    pub fn latest(&self, symbol: &str) -> Option<&VolumeRecord> {
        self.latest.get(symbol).map(|(_, r)| r)
    }
}
