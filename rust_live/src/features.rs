//! Assembles the RF close/continue feature vector each minute.

use std::collections::HashMap;

use crate::artifacts::{IntegralCache, NewsFeatures, VolumeFeatures};
use crate::edges;
use crate::halving_season;
use crate::mdp_features;
use crate::pricing::{LegState, Side};
use crate::state::SymbolState;

/// Live, minute-level volume measurements (from market data, not artifacts).
#[derive(Debug, Clone, Default)]
pub struct MarketVolume {
    pub current_minute_volume: f64,
    pub intraday_volume_so_far: f64,
    pub last_5m_volume: f64,
    pub last_15m_volume: f64,
    pub last_60m_volume: f64,
}

const SEASON_KEYS: &[&str] = &[
    "days_since_last_halving",
    "days_to_next_halving",
    "halving_cycle_progress",
    "halving_sin",
    "halving_cos",
];

pub struct FeatureBuilder<'a> {
    pub cache: &'a IntegralCache,
    pub volume: &'a VolumeFeatures,
    pub news: &'a NewsFeatures,
}

impl<'a> FeatureBuilder<'a> {
    /// Build the feature map for a symbol's current state. `primary` is the leg
    /// the RF decision is keyed on (the single open leg, or the long leg in a
    /// hedged position).
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        &self,
        symbol: &str,
        timestamp: &str,
        date: &str,
        state: &SymbolState,
        primary: &LegState,
        current_price: f64,
        remaining_balance: f64,
        hour_of_day: f64,
        day_of_week: f64,
        market_volume: &MarketVolume,
        halvings: &[i64],
        season_seed: u64,
    ) -> HashMap<String, f64> {
        let mut f = edges::compute_features(
            self.cache,
            state.mode,
            current_price,
            state.long.as_ref(),
            state.short.as_ref(),
            true,
        );

        // Ensure every edge key exists even when a leg is inactive.
        for prefix in ["Long", "Short"] {
            for suffix in ["Return", "Mean", "Var", "MeanOfMean", "VarOfMean"] {
                f.entry(format!("{prefix}Edge_{suffix}")).or_insert(0.0);
            }
        }

        let y = primary.current_return(current_price);
        let liq_price = primary.liquidation_price_level();
        let first_liq = state.first_liq_price.unwrap_or(0.0);

        let side_code = match primary.side {
            Side::Long => 0.0,
            Side::Short => 1.0,
        };

        f.insert("side_code".into(), side_code);
        f.insert("mode_code".into(), state.mode.code());
        f.insert("CurrentPrice".into(), current_price);
        f.insert("EntryPrice".into(), primary.entry_price);
        f.insert("y".into(), y);
        f.insert("current_pnl".into(), primary.current_pnl(current_price));
        f.insert("distance_to_liq".into(), current_price - liq_price);
        f.insert(
            "distance_to_liq_pct".into(),
            if current_price != 0.0 {
                (current_price - liq_price) / current_price
            } else {
                0.0
            },
        );
        f.insert(
            "distance_to_first_liq".into(),
            if first_liq != 0.0 {
                current_price - first_liq
            } else {
                0.0
            },
        );
        f.insert("remaining_balance".into(), remaining_balance);
        f.insert("N_open".into(), state.n_open());
        f.insert("M_open_current".into(), state.margin_open());
        f.insert("liquidation_cutoff".into(), primary.liq_z());
        f.insert("hour_of_day".into(), hour_of_day);
        f.insert("day_of_week".into(), day_of_week);

        // Volume features.
        let (predicted, prev) = match self
            .volume
            .get(symbol, date)
            .or_else(|| self.volume.latest(symbol))
        {
            Some(rec) => (rec.predicted_daily_volume, rec.previous_day_real_volume),
            None => (0.0, 0.0),
        };
        f.insert("predicted_daily_volume".into(), predicted);
        f.insert("previous_day_real_volume".into(), prev);
        f.insert(
            "predicted_volume_change_pct".into(),
            if prev != 0.0 {
                predicted / prev - 1.0
            } else {
                0.0
            },
        );
        f.insert(
            "current_minute_volume".into(),
            market_volume.current_minute_volume,
        );
        f.insert(
            "intraday_volume_so_far".into(),
            market_volume.intraday_volume_so_far,
        );
        f.insert(
            "intraday_volume_pct_of_predicted".into(),
            if predicted != 0.0 {
                market_volume.intraday_volume_so_far / predicted
            } else {
                0.0
            },
        );
        f.insert("last_5m_volume".into(), market_volume.last_5m_volume);
        f.insert("last_15m_volume".into(), market_volume.last_15m_volume);
        f.insert("last_60m_volume".into(), market_volume.last_60m_volume);

        // Leakage-free intraday volume ratios (mirror `simulator._volume_ratio`:
        // numerator / denominator only when the denominator is positive & finite).
        f.insert(
            "current_volume_to_predicted_daily_volume".into(),
            volume_ratio(market_volume.intraday_volume_so_far, predicted),
        );
        f.insert(
            "current_volume_to_previous_day_daily_volume".into(),
            volume_ratio(market_volume.intraday_volume_so_far, prev),
        );

        // News features (zeros when missing).
        for (k, v) in self.news.get(symbol, date) {
            f.insert(k, v);
        }

        // Legacy continuous season features default to 0 (not in the classifier
        // schema, but kept for backward-compatible logging/inspection).
        for k in SEASON_KEYS {
            f.entry((*k).to_string()).or_insert(0.0);
        }

        // Halving-season one-hot (sums to 1) + cycle id, deterministic SHA-256
        // tie-break — full parity with Python `halving_season.season_features`.
        halving_season::insert_season_features(&mut f, symbol, timestamp, halvings, season_seed);

        // MDP-as-feature-engine outputs (no ADD_MARGIN action).
        let mdp = mdp_features::compute_mdp_features(
            Some(self.cache),
            state.long.as_ref(),
            state.short.as_ref(),
            current_price,
        );
        for (k, v) in mdp {
            f.insert(k, v);
        }

        f
    }
}

/// `numerator / denominator`, or `0.0` when the denominator is non-positive or
/// non-finite. Mirrors Python `simulator._volume_ratio`.
fn volume_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator <= 0.0 || !denominator.is_finite() {
        0.0
    } else {
        numerator / denominator
    }
}
