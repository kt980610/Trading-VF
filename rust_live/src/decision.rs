//! Per-symbol live decision.
//!
//! The close/continue decision is made SOLELY by the symbol's RF binary close
//! classifier:
//!
//! ```text
//! p_close = classifier.predict_proba()[:,1]
//! p_close >= threshold -> CLOSE
//! p_close <  threshold -> CONTINUE
//! ```
//!
//! The MDP is now a pure feature engine (its `mdp_*` features are part of the
//! classifier input, computed in `features.rs`); it no longer produces any
//! independent CLOSE / CONTINUE / ADD_MARGIN action. When a symbol has no usable
//! classifier (missing artifact / schema mismatch / load error), the engine HOLDS
//! and logs an explicit reason — it never falls back to a regressor or edge policy.

use std::collections::HashMap;
use std::sync::Arc;

use crate::artifacts::{
    ClassifierError, IntegralCache, ModelSnapshot, NewsFeatures, Policy, VolumeFeatures,
};
use crate::config::MdpSettings;
use crate::features::{FeatureBuilder, MarketVolume};
use crate::logging::LiveDecisionRecord;
use crate::pricing::LegState;
use crate::state::{Mode, SymbolState};

/// Per-symbol sub-timings captured during a decision (for the perf log).
#[derive(Debug, Clone, Copy, Default)]
pub struct DecisionTiming {
    pub rf_inference_ms: f64,
    pub mdp_compute_ms: f64,
}

fn elapsed_ms(start: std::time::Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    Hold,
    CloseHedged,
    CloseLong,
    CloseShort,
    // Retained for the order/executor layer + kill-switch flows; the live
    // decision logic NEVER emits these (MDP is feature-only now).
    AddMarginLong(f64),
    AddMarginShort(f64),
}

#[derive(Debug, Clone)]
pub struct MarketContext {
    pub timestamp: String,
    pub date: String,
    pub current_price: f64,
    pub hour_of_day: f64,
    pub day_of_week: f64,
    pub market_volume: MarketVolume,
    /// Sorted halving epoch seconds (for the season one-hot features).
    pub halvings: Arc<Vec<i64>>,
    /// Global seed for the deterministic season tie-break.
    pub season_seed: u64,
}

#[derive(Debug, Clone)]
pub struct SymbolDecision {
    pub action: Action,
    pub record: LiveDecisionRecord,
}

fn news_summary(features: &HashMap<String, f64>) -> f64 {
    let keys = [
        "macro_news_sentiment",
        "policy_news_sentiment",
        "stock_market_news_sentiment",
        "crypto_market_news_sentiment",
        "symbol_specific_news_sentiment",
    ];
    let mut sum = 0.0;
    let mut n = 0.0;
    for k in keys {
        if let Some(v) = features.get(k) {
            sum += v;
            n += 1.0;
        }
    }
    if n > 0.0 {
        sum / n
    } else {
        0.0
    }
}

/// The leg a decision keys on: the long leg for hedged/long-only, the short leg
/// for short-only (falls back to the other leg if needed).
pub fn primary_leg(state: &SymbolState) -> Option<&LegState> {
    match state.mode {
        Mode::HedgedBothActive | Mode::LongOnlyAfterShortLiq => {
            state.long.as_ref().or(state.short.as_ref())
        }
        Mode::ShortOnlyAfterLongLiq => state.short.as_ref().or(state.long.as_ref()),
    }
}

/// Build the RF feature map for a symbol's current state.
#[allow(clippy::too_many_arguments)]
pub fn build_features(
    symbol: &str,
    state: &SymbolState,
    primary: &LegState,
    cache: &IntegralCache,
    volume: &VolumeFeatures,
    news: &NewsFeatures,
    mctx: &MarketContext,
    remaining_balance: f64,
) -> HashMap<String, f64> {
    let builder = FeatureBuilder {
        cache,
        volume,
        news,
    };
    builder.build(
        symbol,
        &mctx.timestamp,
        &mctx.date,
        state,
        primary,
        mctx.current_price,
        remaining_balance,
        mctx.hour_of_day,
        mctx.day_of_week,
        &mctx.market_volume,
        mctx.halvings.as_slice(),
        mctx.season_seed,
    )
}

/// Build the per-minute decision record skeleton (decision filled in later).
fn base_record(
    symbol: &str,
    state: &SymbolState,
    mctx: &MarketContext,
    features: &HashMap<String, f64>,
    model_version: &str,
    feature_schema_version: Option<String>,
    threshold: f64,
) -> LiveDecisionRecord {
    LiveDecisionRecord {
        timestamp: mctx.timestamp.clone(),
        symbol: symbol.to_string(),
        mode: state.mode.as_str().to_string(),
        current_price: mctx.current_price,
        y: features.get("y").copied().unwrap_or(0.0),
        rf_model_version: model_version.to_string(),
        feature_schema_version,
        rf_prediction: None,
        rf_threshold: threshold,
        decision: "CONTINUE".to_string(),
        reason: String::new(),
        long_edge_return: features.get("LongEdge_Return").copied().unwrap_or(0.0),
        short_edge_return: features.get("ShortEdge_Return").copied().unwrap_or(0.0),
        predicted_daily_volume: features
            .get("predicted_daily_volume")
            .copied()
            .unwrap_or(0.0),
        current_minute_volume: mctx.market_volume.current_minute_volume,
        news_sentiment_summary: news_summary(features),
        news_mode: None,
        news_source: None,
        news_timestamp_quality: None,
        news_asof_timestamp: None,
        news_source_feature_date: None,
        mdp_trigger_active: false,
        mdp_action: None,
        margin_added: 0.0,
        order_ids: vec![],
        realized_pnl_if_closed: None,
        shadow: false,
    }
}

/// Convenience wrapper: build features then decide with a resolved policy.
#[allow(clippy::too_many_arguments)]
pub fn decide_symbol(
    symbol: &str,
    state: &SymbolState,
    policy: &Policy,
    cache: &IntegralCache,
    volume: &VolumeFeatures,
    news: &NewsFeatures,
    mctx: &MarketContext,
    remaining_balance: f64,
    _mdp_cfg: &MdpSettings,
    _expected_costs: f64,
) -> SymbolDecision {
    let primary = primary_leg(state).expect("a leg must be open").clone();
    let features = build_features(
        symbol,
        state,
        &primary,
        cache,
        volume,
        news,
        mctx,
        remaining_balance,
    );
    decide_with_policy(symbol, state, policy, mctx, &features).0
}

/// Decide from an immutable snapshot. Pure and thread-safe (no exchange, no
/// registry) so it runs inside parallel workers. Always returns a decision: when
/// the classifier is missing the symbol HOLDS with an explicit reason (logged),
/// rather than falling back to any legacy policy.
#[allow(clippy::too_many_arguments)]
pub fn decide_symbol_snapshot(
    symbol: &str,
    state: &SymbolState,
    snapshot: &ModelSnapshot,
    cache: &IntegralCache,
    volume: &VolumeFeatures,
    news: &NewsFeatures,
    mctx: &MarketContext,
    remaining_balance: f64,
    _mdp_cfg: &MdpSettings,
    _fallback_to_baseline: bool,
    _expected_costs: f64,
) -> Option<(SymbolDecision, DecisionTiming)> {
    let primary = primary_leg(state)?.clone();
    let features = build_features(
        symbol,
        state,
        &primary,
        cache,
        volume,
        news,
        mctx,
        remaining_balance,
    );

    match snapshot.policy() {
        Some(policy) => Some(decide_with_policy(symbol, state, &policy, mctx, &features)),
        None => {
            // No usable classifier for this symbol -> explicit hold (no fallback).
            let mut record = base_record(symbol, state, mctx, &features, "none", None, 0.0);
            record.decision = "CONTINUE".to_string();
            record.reason = "missing_symbol_model".to_string();
            Some((
                SymbolDecision {
                    action: Action::Hold,
                    record,
                },
                DecisionTiming::default(),
            ))
        }
    }
}

/// Core close/continue decision from already-built features. Returns the decision
/// plus the RF inference sub-timing for the performance log.
pub fn decide_with_policy(
    symbol: &str,
    state: &SymbolState,
    policy: &Policy,
    mctx: &MarketContext,
    features: &HashMap<String, f64>,
) -> (SymbolDecision, DecisionTiming) {
    let primary: &LegState = primary_leg(state).expect("a leg must be open");
    let mut timing = DecisionTiming::default();

    let mut record = base_record(
        symbol,
        state,
        mctx,
        features,
        policy.version(),
        Some(policy.feature_schema_version().to_string()),
        policy.threshold(),
    );

    let rf_start = std::time::Instant::now();
    let p_close = policy.p_close(features);
    timing.rf_inference_ms = elapsed_ms(rf_start);

    match p_close {
        Ok(p) => {
            record.rf_prediction = Some(p);
            if p >= policy.threshold() {
                record.decision = "CLOSE".to_string();
                record.reason = format!("{}_close", policy.version());
                record.realized_pnl_if_closed = Some(primary.current_pnl(mctx.current_price));
                let action = match state.mode {
                    Mode::HedgedBothActive => Action::CloseHedged,
                    Mode::LongOnlyAfterShortLiq => Action::CloseLong,
                    Mode::ShortOnlyAfterLongLiq => Action::CloseShort,
                };
                (SymbolDecision { action, record }, timing)
            } else {
                record.decision = "CONTINUE".to_string();
                record.reason = format!("{}_continue", policy.version());
                (
                    SymbolDecision {
                        action: Action::Hold,
                        record,
                    },
                    timing,
                )
            }
        }
        Err(e) => {
            // Schema mismatch / load error -> safe hold, explicit reason, no order.
            record.rf_prediction = None;
            record.decision = "CONTINUE".to_string();
            record.reason = match e {
                ClassifierError::FeatureSchemaMismatch(_) => "feature_schema_mismatch".to_string(),
                ClassifierError::ClassifierLoadError(_) => "classifier_load_error".to_string(),
            };
            (
                SymbolDecision {
                    action: Action::Hold,
                    record,
                },
                timing,
            )
        }
    }
}
