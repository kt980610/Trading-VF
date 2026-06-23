//! Persistent live position state (mode survives restarts).

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::pricing::LegState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    #[serde(rename = "HEDGED_BOTH_ACTIVE")]
    HedgedBothActive,
    #[serde(rename = "LONG_ONLY_AFTER_SHORT_LIQ")]
    LongOnlyAfterShortLiq,
    #[serde(rename = "SHORT_ONLY_AFTER_LONG_LIQ")]
    ShortOnlyAfterLongLiq,
}

impl Mode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Mode::HedgedBothActive => "HEDGED_BOTH_ACTIVE",
            Mode::LongOnlyAfterShortLiq => "LONG_ONLY_AFTER_SHORT_LIQ",
            Mode::ShortOnlyAfterLongLiq => "SHORT_ONLY_AFTER_LONG_LIQ",
        }
    }
    pub fn code(&self) -> f64 {
        match self {
            Mode::HedgedBothActive => 0.0,
            Mode::LongOnlyAfterShortLiq => 1.0,
            Mode::ShortOnlyAfterLongLiq => 2.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolState {
    pub mode: Mode,
    #[serde(default)]
    pub long: Option<LegState>,
    #[serde(default)]
    pub short: Option<LegState>,
    #[serde(default)]
    pub first_liq_price: Option<f64>,
    #[serde(default)]
    pub total_added_margin_long: f64,
    #[serde(default)]
    pub total_added_margin_short: f64,
}

impl SymbolState {
    pub fn n_open(&self) -> f64 {
        let mut n = 0.0;
        if let Some(l) = &self.long {
            n += l.notional();
        }
        if let Some(s) = &self.short {
            n += s.notional();
        }
        n
    }
    pub fn margin_open(&self) -> f64 {
        let mut m = 0.0;
        if let Some(l) = &self.long {
            m += l.margin_current;
        }
        if let Some(s) = &self.short {
            m += s.margin_current;
        }
        m
    }
    pub fn is_single_leg(&self) -> bool {
        self.long.is_some() ^ self.short.is_some()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LivePositionState {
    #[serde(default)]
    pub remaining_balance: f64,
    #[serde(default)]
    pub symbols: HashMap<String, SymbolState>,
}

impl LivePositionState {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        if !path.as_ref().exists() {
            return Ok(LivePositionState::default());
        }
        let text = std::fs::read_to_string(path)?;
        let s: LivePositionState = serde_json::from_str(&text)?;
        Ok(s)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.as_ref().with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(self)?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }
}
