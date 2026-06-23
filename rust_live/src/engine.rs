//! Live engine orchestration: startup checks, position opening, the minute tick,
//! kill-switch enforcement, and shadow/real-money order gating.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::artifacts::{
    CacheSet, IntradayNews, IntradayRecord, ModelRegistry, NewsFeatures, Policy, PortfolioWeights,
    VolumeFeatures,
};
use crate::config::Config;
use crate::decision::{decide_symbol, Action, MarketContext, SymbolDecision};
use crate::exchange::{ExchangeClient, OrderRequest};
use crate::executor::{ExecContext, OrderExecutor};
use crate::killswitch;
use crate::logging::{self, LiveDecisionRecord};
use crate::order::{self, OrderIntent};
use crate::parallel::{self, DecisionInput};
use crate::perf::CycleMetrics;
use crate::pricing::{LegState, Side};
use crate::risk::RiskManager;
use crate::state::{LivePositionState, Mode, SymbolState};

fn ms_since(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[derive(Debug, Clone, Default)]
pub struct StartupReport {
    pub issues: Vec<String>,
    pub can_open_new_trades: bool,
}

#[derive(Debug, Clone, Default)]
pub struct RebalanceSummary {
    pub opened: usize,
    pub reduced: usize,
    pub closed: usize,
    pub total_equity: f64,
}

/// Merge an additional market fill into a leg, computing a quantity-weighted
/// average entry price and the matching isolated margin.
fn merge_leg(
    existing: Option<LegState>,
    side: Side,
    price: f64,
    add_qty: f64,
    leverage: f64,
) -> LegState {
    match existing {
        Some(leg) => {
            let new_qty = leg.qty + add_qty;
            let new_entry = if new_qty > 0.0 {
                (leg.qty * leg.entry_price + add_qty * price) / new_qty
            } else {
                price
            };
            let margin = (new_qty * new_entry) / leverage.max(1.0);
            LegState::new(side, new_entry, new_qty, margin)
        }
        None => {
            let margin = (add_qty * price) / leverage.max(1.0);
            LegState::new(side, price, add_qty, margin)
        }
    }
}

pub struct Engine<E: ExchangeClient> {
    pub config: Config,
    pub exchange: E,
    pub registry: ModelRegistry,
    pub cache: Arc<CacheSet>,
    pub weights: Option<PortfolioWeights>,
    pub volume: Arc<VolumeFeatures>,
    pub news: Arc<NewsFeatures>,
    pub intraday_news: Arc<IntradayNews>,
    pub state: LivePositionState,
    pub halted: bool,
}

impl<E: ExchangeClient> Engine<E> {
    pub fn new(config: Config, exchange: E) -> anyhow::Result<Self> {
        let registry = ModelRegistry::new(config.paths.models_promoted.clone());
        let cache = Arc::new(
            CacheSet::load(&config.paths.integral_cache).unwrap_or(CacheSet {
                symbols: Default::default(),
            }),
        );
        let weights = PortfolioWeights::load(&config.paths.portfolio_weights).ok();
        let volume = Arc::new(
            VolumeFeatures::load(&config.paths.predicted_daily_volume).unwrap_or_default(),
        );
        let news =
            Arc::new(NewsFeatures::load(&config.paths.news_features_daily).unwrap_or_default());
        let intraday_news =
            Arc::new(IntradayNews::load(&config.paths.news_features_intraday).unwrap_or_default());
        let state = LivePositionState::load(&config.paths.live_position_state).unwrap_or_default();

        Ok(Self {
            config,
            exchange,
            registry,
            cache,
            weights,
            volume,
            news,
            intraday_news,
            state,
            halted: false,
        })
    }

    /// The single source of truth for the as-of news join, by `symbol +
    /// decision_secs`. Selects the newest record satisfying the leakage rule
    /// `asof_timestamp <= decision_secs - news_safety_lag_seconds` (identical to
    /// the Python training join). Returns the record only when it is also fresh
    /// enough (`age <= max_news_feature_age_minutes`); a stale/missing record
    /// yields `None`.
    fn asof_news_record(&self, symbol: &str, decision_secs: i64) -> Option<&IntradayRecord> {
        if !self.config.news.intraday_enabled {
            return None;
        }
        let cutoff = decision_secs - self.config.news.news_safety_lag_seconds.max(0);
        let rec = self.intraday_news.latest_before(symbol, cutoff)?;
        let max_age = self.config.news.max_news_feature_age_minutes.max(0) * 60;
        if IntradayNews::age_secs(rec, decision_secs) <= max_age {
            Some(rec)
        } else {
            None
        }
    }

    /// The news lookup a symbol's decision should use at `decision_secs`.
    ///
    /// With the intraday feature off this is always the daily artifact (legacy
    /// behavior). With it on we splice the freshest leakage-safe as-of record
    /// into a single-entry lookup so the SAME RF feature vector (hence both the
    /// close and continue probability) reflects current news. A missing/stale
    /// record falls back to the daily artifact (zero-news); the open-gate
    /// separately blocks NEW positions in that case while leaving close/risk
    /// management of existing positions untouched.
    fn news_for_symbol(&self, symbol: &str, decision_secs: i64, date: &str) -> Arc<NewsFeatures> {
        match self.asof_news_record(symbol, decision_secs) {
            Some(rec) => Arc::new(NewsFeatures::from_asof(date, symbol, &rec.features)),
            // DAILY (D-1) path: the daily artifact is keyed by the day news
            // OCCURRED, so a minute on day D must use the most recent completed
            // day strictly before D (mirrors the Python training join). Splice
            // that previous-day vector into a single-entry lookup keyed at
            // `date` so the existing `get(symbol, date)` returns it unchanged.
            None => {
                let prev = self.news.previous_day_features(symbol, date);
                Arc::new(NewsFeatures::from_asof(date, symbol, &prev))
            }
        }
    }

    /// Provenance of the as-of record used for `symbol` at `decision_secs`:
    /// `(news_mode, asof_timestamp, source_feature_date)`. `None` when no
    /// leakage-safe / fresh record is used (intraday off, or daily fallback).
    fn news_provenance(
        &self,
        symbol: &str,
        decision_secs: i64,
    ) -> Option<(
        String,
        String,
        Option<String>,
        Option<String>,
        Option<String>,
    )> {
        let rec = self.asof_news_record(symbol, decision_secs)?;
        Some((
            rec.news_mode.clone(),
            rec.asof_timestamp.clone(),
            rec.source_feature_date.clone(),
            rec.news_source.clone(),
            rec.timestamp_quality.clone(),
        ))
    }

    /// Whether the as-of news for `symbol` is fresh enough to OPEN a new position.
    /// Off => always allowed. On => requires a leakage-safe record no older than
    /// the configured budget; a missing artifact / failed worker (no usable
    /// record) blocks the open rather than silently opening on zero-news.
    fn news_allows_open(&self, symbol: &str, now_secs: i64) -> bool {
        if !self.config.news.intraday_enabled {
            return true;
        }
        match self.asof_news_record(symbol, now_secs) {
            Some(rec) => {
                // Schema-mismatch fail-safe: a record whose feature_version does
                // not match the configured expectation must not open a position
                // (the model was trained on a different news feature schema).
                let expected = &self.config.news.expected_feature_version;
                if !expected.is_empty() && &rec.feature_version != expected {
                    eprintln!(
                        "news_gate symbol={symbol} block_open=schema_mismatch expected={expected} got={}",
                        rec.feature_version
                    );
                    return false;
                }
                true
            }
            None => {
                eprintln!(
                    "news_gate symbol={symbol} block_open=stale_or_missing now={} lag_s={} max_age_s={}",
                    crate::clock::rfc3339(now_secs),
                    self.config.news.news_safety_lag_seconds,
                    self.config.news.max_news_feature_age_minutes * 60
                );
                false
            }
        }
    }

    /// Orders are only sent in real-money, non-shadow mode.
    pub fn orders_enabled(&self) -> bool {
        self.config.live.real_money && !self.config.live.shadow_mode
    }

    pub fn kill_switch_active(&self) -> bool {
        killswitch::is_triggered_file_present(&self.config.paths.kill_switch_triggered)
    }

    /// Canonical equity base for sizing: `wallet_balance + unrealized_pnl`.
    fn total_equity(&self) -> anyhow::Result<f64> {
        let balance = self.exchange.total_wallet_balance()?;
        let upnl: f64 = self
            .exchange
            .positions()
            .map(|ps| ps.iter().map(|p| p.unrealized_pnl).sum())
            .unwrap_or(0.0);
        Ok(balance + upnl)
    }

    /// Fail-safe gate: the MVO artifact must be present, internally consistent
    /// (`sum(weights) + cash_weight == 1.0`), fresh, and schema/version
    /// compatible before ANY new position may be opened. There is no legacy
    /// sizing fallback: when this returns `false` the engine only manages
    /// existing positions.
    pub fn portfolio_ready(&self) -> Vec<String> {
        let mut issues = Vec::new();
        let w = match self.weights.as_ref() {
            None => {
                issues.push("portfolio_weights.json missing or invalid".into());
                return issues;
            }
            Some(w) => w,
        };
        if w.symbols.is_empty() {
            issues.push("portfolio_weights has no symbols".into());
        }
        if !w.sum_valid(self.config.sizing.weight_sum_tolerance) {
            issues.push("portfolio weights + cash_weight != 1.0".into());
        }
        if !w.is_fresh(
            &crate::clock::today_date(),
            self.config.sizing.max_portfolio_age_days,
        ) {
            issues.push(format!(
                "portfolio_weights stale (as_of_date={}, max_age_days={})",
                w.as_of_date, self.config.sizing.max_portfolio_age_days
            ));
        }
        if !w.semantics_ok() {
            issues.push(format!(
                "portfolio weight_semantics unsupported: '{}'",
                w.weight_semantics
            ));
        }
        if !w.version_ok(&self.config.sizing.portfolio_version) {
            issues.push(format!(
                "portfolio_version mismatch (artifact='{}', expected='{}')",
                w.portfolio_version, self.config.sizing.portfolio_version
            ));
        }
        issues
    }

    pub fn portfolio_ok(&self) -> bool {
        self.portfolio_ready().is_empty()
    }

    fn save_state(&self) {
        let _ = self.state.save(&self.config.paths.live_position_state);
    }

    fn log_decision(&self, mut record: LiveDecisionRecord) {
        record.shadow = !self.orders_enabled();
        let _ = logging::append_jsonl(&self.config.paths.live_decisions, &record);
    }

    /// Send an order only when orders are enabled (shadow/real_money gate).
    fn execute_order(&self, req: &OrderRequest) -> Vec<String> {
        if !self.orders_enabled() {
            return vec![];
        }
        match self.exchange.place_market_order(req) {
            Ok(resp) => vec![resp.order_id],
            Err(_) => vec![],
        }
    }

    // ---------------------------------------------------------------------
    // Startup
    // ---------------------------------------------------------------------
    pub fn startup_checks(&mut self) -> StartupReport {
        let mut issues = Vec::new();

        if self.kill_switch_active() {
            issues.push("kill_switch_triggered.json present -> trading disabled".into());
        }

        for issue in self.portfolio_ready() {
            issues.push(issue);
        }

        if self.cache.symbols.is_empty() {
            issues.push("integral cache empty/missing".into());
        }

        if !std::path::Path::new(&self.config.paths.distribution_snapshot).exists() {
            issues.push("distribution_snapshot.json missing".into());
        }

        // Per-symbol RF presence (informational unless require_rf_for_new_trade).
        for symbol in self.config.symbols.clone() {
            if !self.registry.has_model(&symbol) {
                if self.config.live.require_rf_for_new_trade {
                    issues.push(format!(
                        "RF model missing for {symbol} (require_rf_for_new_trade=true)"
                    ));
                } else {
                    issues.push(format!("RF model missing for {symbol} (baseline fallback)"));
                }
            }
        }

        let can_open =
            !self.kill_switch_active() && self.portfolio_ok() && !self.cache.symbols.is_empty();
        StartupReport {
            issues,
            can_open_new_trades: can_open,
        }
    }

    /// Whether a given symbol is allowed to open a NEW trade right now.
    pub fn can_open_symbol(&mut self, symbol: &str) -> bool {
        if self.kill_switch_active() {
            return false;
        }
        // Fail-safe: missing / stale / invalid / schema-incompatible MVO artifact
        // blocks all new opens. No legacy sizing fallback.
        if !self.portfolio_ok() {
            return false;
        }
        let tradeable = self
            .weights
            .as_ref()
            .map(|w| w.weight_for(symbol) > 0.0)
            .unwrap_or(false);
        if !tradeable {
            return false;
        }
        if self.cache.get(symbol).is_none() {
            return false;
        }
        // As-of news freshness gate: do not OPEN new positions on stale/missing
        // news when the intraday feature is enabled (open-only; existing
        // positions keep being managed by the per-minute tick).
        if !self.news_allows_open(symbol, crate::clock::now_unix_secs()) {
            return false;
        }
        let has_rf = self.registry.has_model(symbol);
        if !has_rf {
            // require_rf_for_new_trade blocks; otherwise baseline must be allowed.
            if self.config.live.require_rf_for_new_trade {
                return false;
            }
            if !self.config.live.fallback_to_baseline_if_rf_missing {
                return false;
            }
        }
        true
    }

    // ---------------------------------------------------------------------
    // Opening / daily rebalance (canonical MVO portfolio-weight sizing)
    // ---------------------------------------------------------------------

    /// Backwards-compatible entry point: rebalance from the current (typically
    /// flat) state to the MVO targets and report how many symbols were opened.
    pub fn open_initial_positions(&mut self) -> anyhow::Result<usize> {
        Ok(self.rebalance_to_targets()?.opened)
    }

    /// Apply the daily MVO artifact: size every symbol off `total_equity *
    /// weight`, then open / reduce / close only the difference versus the current
    /// position. Symbols out of MVO scope (target weight 0) are closed. There is
    /// no fixed reserve and no net-PnL adjustment; capital a hard cap leaves
    /// undeployed stays as cash and is logged.
    pub fn rebalance_to_targets(&mut self) -> anyhow::Result<RebalanceSummary> {
        let mut summary = RebalanceSummary::default();
        if self.kill_switch_active() {
            return Ok(summary);
        }
        let balance = self.exchange.total_wallet_balance()?;
        let equity = self.total_equity()?;
        summary.total_equity = equity;

        // Fail-safe: without a valid MVO artifact we neither open nor rebalance.
        // Existing positions remain under normal close/risk management.
        if !self.portfolio_ok() {
            self.state.remaining_balance = balance;
            self.save_state();
            return Ok(summary);
        }

        let (cash_weight, as_of_date, portfolio_version) = self
            .weights
            .as_ref()
            .map(|w| {
                (
                    w.cash_weight,
                    w.as_of_date.clone(),
                    w.portfolio_version.clone(),
                )
            })
            .unwrap_or((0.0, String::new(), String::new()));
        let max_order_notional = self.config.sizing.max_order_notional_usdt;

        // 1) Close any open symbol that is out of MVO scope (target weight 0).
        let open_symbols: Vec<String> = self.state.symbols.keys().cloned().collect();
        for symbol in open_symbols {
            let target_weight = self
                .weights
                .as_ref()
                .map(|w| w.weight_for(&symbol))
                .unwrap_or(0.0);
            if target_weight > 0.0 {
                continue;
            }
            let price = self.exchange.mark_price(&symbol).unwrap_or(0.0);
            if let Some(state) = self.state.symbols.get(&symbol).cloned() {
                let current_notional = state.n_open();
                let _ = self.execute_action(&symbol, &Action::CloseHedged, &state, price);
                summary.closed += 1;
                self.log_allocation(logging::PortfolioAllocationRecord {
                    timestamp: crate::clock::now_rfc3339(),
                    symbol: symbol.clone(),
                    portfolio_version: portfolio_version.clone(),
                    as_of_date: as_of_date.clone(),
                    total_equity: equity,
                    cash_weight,
                    weight: 0.0,
                    target_margin: 0.0,
                    target_notional: 0.0,
                    current_margin: state.margin_open(),
                    current_notional,
                    delta_notional: -current_notional,
                    realized_margin: 0.0,
                    uninvested_margin: 0.0,
                    action: "close_out_of_scope".into(),
                    cap_reason: None,
                    shadow: !self.orders_enabled(),
                });
            }
        }

        // 2) Size and rebalance every configured, in-scope symbol.
        let symbols = self.config.symbols.clone();
        for symbol in symbols {
            let target_weight = self
                .weights
                .as_ref()
                .map(|w| w.weight_for(&symbol))
                .unwrap_or(0.0);
            if target_weight <= 0.0 {
                continue;
            }
            let price = match self.exchange.mark_price(&symbol) {
                Ok(p) if p > 0.0 => p,
                _ => continue,
            };
            let filters = self.exchange.symbol_filters(&symbol).unwrap_or_default();
            let leverage = self.config.leverage_for(&symbol);
            let alloc = crate::sizing::allocate_symbol(
                equity,
                target_weight,
                leverage,
                price,
                &filters,
                max_order_notional,
            );

            let current = self.state.symbols.get(&symbol).cloned();
            let action = if current.is_none() {
                // Open from flat (gated by the full open fail-safe / news / RF).
                if !self.can_open_symbol(&symbol) {
                    continue;
                }
                if self.open_symbol_from_flat(&symbol, &alloc, price, leverage) {
                    summary.opened += 1;
                    "open"
                } else {
                    continue;
                }
            } else {
                // Adjust each leg toward target; reduce-only when shrinking, add
                // when growing. No order when already at target (no double
                // allocation of the same capital).
                let (reduced, increased) =
                    self.adjust_symbol_to_target(&symbol, &alloc, price, leverage);
                if reduced {
                    summary.reduced += 1;
                }
                if reduced && increased {
                    "rebalance"
                } else if reduced {
                    "reduce"
                } else if increased {
                    "increase"
                } else {
                    "hold"
                }
            };

            let current_notional = self
                .state
                .symbols
                .get(&symbol)
                .map(|s| s.n_open())
                .unwrap_or(0.0);
            let current_margin = self
                .state
                .symbols
                .get(&symbol)
                .map(|s| s.margin_open())
                .unwrap_or(0.0);
            self.log_allocation(logging::PortfolioAllocationRecord {
                timestamp: crate::clock::now_rfc3339(),
                symbol: symbol.clone(),
                portfolio_version: portfolio_version.clone(),
                as_of_date: as_of_date.clone(),
                total_equity: equity,
                cash_weight,
                weight: target_weight,
                target_margin: alloc.target_margin,
                target_notional: alloc.target_notional,
                current_margin,
                current_notional,
                delta_notional: current_notional - alloc.realized_notional(price),
                realized_margin: alloc.realized_margin(price, leverage),
                uninvested_margin: alloc.uninvested_margin(price, leverage),
                action: action.into(),
                cap_reason: alloc.cap_reason.clone(),
                shadow: !self.orders_enabled(),
            });
        }

        self.state.remaining_balance = balance;
        self.save_state();
        Ok(summary)
    }

    /// Open a fresh hedged position from flat at the allocation target.
    fn open_symbol_from_flat(
        &mut self,
        symbol: &str,
        alloc: &crate::sizing::SymbolAllocation,
        price: f64,
        leverage: f64,
    ) -> bool {
        if alloc.long.qty.is_none() && alloc.short.qty.is_none() {
            return false;
        }
        if self.orders_enabled() {
            let _ = self.exchange.set_leverage(symbol, leverage.max(1.0) as u32);
        }
        let mut long_leg = None;
        let mut short_leg = None;
        if let Some(q) = alloc.long.qty {
            let _ = self.execute_order(&OrderRequest::open(symbol, Side::Long, q));
            let margin = (q * price) / leverage.max(1.0);
            long_leg = Some(LegState::new(Side::Long, price, q, margin));
        }
        if let Some(q) = alloc.short.qty {
            let _ = self.execute_order(&OrderRequest::open(symbol, Side::Short, q));
            let margin = (q * price) / leverage.max(1.0);
            short_leg = Some(LegState::new(Side::Short, price, q, margin));
        }
        self.state.symbols.insert(
            symbol.to_string(),
            SymbolState {
                mode: Mode::HedgedBothActive,
                long: long_leg,
                short: short_leg,
                first_liq_price: None,
                total_added_margin_long: 0.0,
                total_added_margin_short: 0.0,
            },
        );
        true
    }

    /// Move an existing hedged position toward the per-leg target quantities.
    /// Returns `(reduced, increased)`. A delta smaller than one step is a no-op
    /// (the capital is already allocated; it is not re-deployed).
    fn adjust_symbol_to_target(
        &mut self,
        symbol: &str,
        alloc: &crate::sizing::SymbolAllocation,
        price: f64,
        leverage: f64,
    ) -> (bool, bool) {
        let filters = self.exchange.symbol_filters(symbol).unwrap_or_default();
        let step = filters.step_size.max(0.0);
        let mut reduced = false;
        let mut increased = false;

        for side in [Side::Long, Side::Short] {
            let target_qty = match side {
                Side::Long => alloc.long.qty.unwrap_or(0.0),
                Side::Short => alloc.short.qty.unwrap_or(0.0),
            };
            let current_qty = self
                .state
                .symbols
                .get(symbol)
                .and_then(|s| match side {
                    Side::Long => s.long.as_ref(),
                    Side::Short => s.short.as_ref(),
                })
                .map(|l| l.qty)
                .unwrap_or(0.0);

            let delta = target_qty - current_qty;
            if delta.abs() <= step {
                continue;
            }

            if delta < 0.0 {
                // Reduce: reduceOnly close of the surplus quantity.
                let qty = crate::sizing::round_down_to_step(-delta, step);
                if qty <= 0.0 {
                    continue;
                }
                let _ = self.execute_order(&OrderRequest::close(symbol, side, qty));
                if let Some(st) = self.state.symbols.get_mut(symbol) {
                    let leg = match side {
                        Side::Long => st.long.as_mut(),
                        Side::Short => st.short.as_mut(),
                    };
                    if let Some(leg) = leg {
                        leg.qty = (leg.qty - qty).max(0.0);
                        leg.margin_current = (leg.qty * leg.entry_price) / leverage.max(1.0);
                    }
                }
                reduced = true;
            } else {
                // Increase: open the delta and update the leg with a weighted
                // average entry price.
                let qty = crate::sizing::round_down_to_step(delta, step);
                if qty <= 0.0 {
                    continue;
                }
                let _ = self.execute_order(&OrderRequest::open(symbol, side, qty));
                if let Some(st) = self.state.symbols.get_mut(symbol) {
                    match side {
                        Side::Long => {
                            st.long =
                                Some(merge_leg(st.long.take(), Side::Long, price, qty, leverage));
                        }
                        Side::Short => {
                            st.short = Some(merge_leg(
                                st.short.take(),
                                Side::Short,
                                price,
                                qty,
                                leverage,
                            ));
                        }
                    }
                }
                increased = true;
            }
        }

        (reduced, increased)
    }

    fn log_allocation(&self, record: logging::PortfolioAllocationRecord) {
        let _ = logging::append_jsonl(&self.config.paths.live_decisions, &record);
    }

    // ---------------------------------------------------------------------
    // Per-symbol tick
    // ---------------------------------------------------------------------
    fn resolve_policy(&mut self, symbol: &str) -> Option<Policy> {
        self.registry.policy_for(symbol)
    }

    /// Decide and (if enabled) execute for one symbol with a supplied market
    /// context. Returns the produced decision (already logged). When the symbol
    /// has no usable classifier the engine holds (no order, no legacy fallback).
    pub fn tick_symbol(&mut self, symbol: &str, mctx: &MarketContext) -> Option<SymbolDecision> {
        let state = self.state.symbols.get(symbol)?.clone();

        let cache = match self.cache.get(symbol) {
            Some(c) => c.clone(),
            None => {
                // Conservative fallback: no cache -> hold, log nothing actionable.
                return None;
            }
        };

        // Ensure a leg is actually open before deciding.
        let _ = crate::decision::primary_leg(&state)?;
        let policy = match self.resolve_policy(symbol) {
            Some(p) => p,
            None => return None,
        };

        let remaining_balance = self.state.remaining_balance;
        let decision_secs = crate::clock::parse_rfc3339_secs(&mctx.timestamp)
            .unwrap_or_else(crate::clock::now_unix_secs);
        let news_arc = self.news_for_symbol(symbol, decision_secs, &mctx.date);
        let mut decision = decide_symbol(
            symbol,
            &state,
            &policy,
            &cache,
            self.volume.as_ref(),
            news_arc.as_ref(),
            mctx,
            remaining_balance,
            &self.config.mdp,
            0.0,
        );

        if let Some((mode, asof, src, source, quality)) =
            self.news_provenance(symbol, decision_secs)
        {
            if mode == "previous_completed_day_fallback" {
                eprintln!(
                    "news_mode symbol={symbol} mode={mode} asof={asof} reason=date_only_or_ambiguous_timestamp source_feature_date={}",
                    src.as_deref().unwrap_or("null")
                );
            }
            decision.record.news_mode = Some(mode);
            decision.record.news_source = source;
            decision.record.news_timestamp_quality = quality;
            decision.record.news_asof_timestamp = Some(asof);
            decision.record.news_source_feature_date = src;
        }

        let order_ids = self.execute_action(symbol, &decision.action, &state, mctx.current_price);
        decision.record.order_ids = order_ids;
        self.log_decision(decision.record.clone());
        self.save_state();
        Some(decision)
    }

    fn execute_action(
        &mut self,
        symbol: &str,
        action: &Action,
        state: &SymbolState,
        current_price: f64,
    ) -> Vec<String> {
        let mut ids = Vec::new();
        match action {
            Action::Hold => {}
            Action::CloseHedged => {
                if let Some(l) = &state.long {
                    ids.extend(self.execute_order(&OrderRequest::close(symbol, Side::Long, l.qty)));
                }
                if let Some(s) = &state.short {
                    ids.extend(self.execute_order(&OrderRequest::close(
                        symbol,
                        Side::Short,
                        s.qty,
                    )));
                }
                self.record_realized(symbol, state, current_price);
                self.state.symbols.remove(symbol);
            }
            Action::CloseLong => {
                if let Some(l) = &state.long {
                    ids.extend(self.execute_order(&OrderRequest::close(symbol, Side::Long, l.qty)));
                }
                self.record_realized(symbol, state, current_price);
                self.state.symbols.remove(symbol);
            }
            Action::CloseShort => {
                if let Some(s) = &state.short {
                    ids.extend(self.execute_order(&OrderRequest::close(
                        symbol,
                        Side::Short,
                        s.qty,
                    )));
                }
                self.record_realized(symbol, state, current_price);
                self.state.symbols.remove(symbol);
            }
            Action::AddMarginLong(x) => {
                if *x > 0.0 {
                    if self.orders_enabled() {
                        let _ = self.exchange.add_isolated_margin(symbol, Side::Long, *x);
                    }
                    if let Some(st) = self.state.symbols.get_mut(symbol) {
                        if let Some(l) = st.long.as_mut() {
                            l.margin_current += x;
                        }
                        st.total_added_margin_long += x;
                    }
                    self.state.remaining_balance -= x;
                }
            }
            Action::AddMarginShort(x) => {
                if *x > 0.0 {
                    if self.orders_enabled() {
                        let _ = self.exchange.add_isolated_margin(symbol, Side::Short, *x);
                    }
                    if let Some(st) = self.state.symbols.get_mut(symbol) {
                        if let Some(s) = st.short.as_mut() {
                            s.margin_current += x;
                        }
                        st.total_added_margin_short += x;
                    }
                    self.state.remaining_balance -= x;
                }
            }
        }
        ids
    }

    fn record_realized(&self, symbol: &str, state: &SymbolState, current_price: f64) {
        let mut pnl = 0.0;
        if let Some(l) = &state.long {
            pnl += l.current_pnl(current_price);
        }
        if let Some(s) = &state.short {
            pnl += s.current_pnl(current_price);
        }
        let weight = self
            .weights
            .as_ref()
            .and_then(|w| w.get(symbol))
            .map(|sw| sw.weight_discrete)
            .unwrap_or(0.0);
        let record = logging::RealizedReturnRecord {
            date: crate::clock::today_date(),
            symbol: symbol.to_string(),
            realized_pnl: pnl,
            fees: 0.0,
            funding: 0.0,
            slippage: 0.0,
            model_version: "live".into(),
            portfolio_weight_used: weight,
        };
        let _ = logging::append_jsonl(&self.config.paths.realized_symbol_returns, &record);
    }

    // ---------------------------------------------------------------------
    // Kill switch
    // ---------------------------------------------------------------------
    pub fn enforce_kill_switch(&mut self, balance: f64) -> anyhow::Result<bool> {
        if !killswitch::balance_below_threshold(balance, &self.config.risk) {
            return Ok(false);
        }
        // 1) stop new orders implicitly (halted). 2) cancel + close (gated).
        if self.orders_enabled() {
            let _ = killswitch::flatten_all(&self.exchange, &self.config.risk);
        }
        killswitch::write_trigger_file(
            &self.config.paths.kill_switch_triggered,
            "total_wallet_balance below minimum",
            balance,
            &self.config.risk,
        )?;
        self.state.symbols.clear();
        self.save_state();
        self.halted = true;
        Ok(true)
    }

    /// One full minute tick: balance read, kill-switch, then manage positions.
    pub fn run_tick(
        &mut self,
        contexts: &std::collections::HashMap<String, MarketContext>,
    ) -> anyhow::Result<()> {
        if self.kill_switch_active() {
            self.halted = true;
            return Ok(());
        }
        let balance = self.exchange.total_wallet_balance()?;
        self.state.remaining_balance = balance;

        if self.enforce_kill_switch(balance)? {
            return Ok(());
        }

        let symbols: Vec<String> = self.state.symbols.keys().cloned().collect();
        for symbol in symbols {
            if let Some(mctx) = contexts.get(&symbol) {
                let _ = self.tick_symbol(&symbol, mctx);
            }
        }
        self.save_state();
        Ok(())
    }

    // ---------------------------------------------------------------------
    // Parallel decision cycle + centralized execution
    // ---------------------------------------------------------------------

    /// State-only mutation for a close (no order send): record realized PnL and
    /// drop the symbol. The order itself goes through `OrderExecutor`.
    fn apply_close_state(&mut self, symbol: &str, state: &SymbolState, current_price: f64) {
        self.record_realized(symbol, state, current_price);
        self.state.symbols.remove(symbol);
    }

    /// State-only mutation for add-margin (no order send).
    fn apply_add_margin_state(&mut self, symbol: &str, side: Side, x: f64) {
        if x <= 0.0 {
            return;
        }
        if let Some(st) = self.state.symbols.get_mut(symbol) {
            match side {
                Side::Long => {
                    if let Some(l) = st.long.as_mut() {
                        l.margin_current += x;
                    }
                    st.total_added_margin_long += x;
                }
                Side::Short => {
                    if let Some(s) = st.short.as_mut() {
                        s.margin_current += x;
                    }
                    st.total_added_margin_short += x;
                }
            }
        }
        self.state.remaining_balance -= x;
    }

    /// Build the immutable per-symbol decision inputs for this cycle. Model
    /// snapshots are captured here (sequentially) so workers never touch the
    /// registry and a hot-reload mid-cycle cannot change a worker's schema.
    fn build_decision_inputs(
        &mut self,
        contexts: &HashMap<String, MarketContext>,
    ) -> Vec<DecisionInput> {
        let symbols: Vec<String> = self
            .state
            .symbols
            .keys()
            .filter(|s| contexts.contains_key(*s))
            .cloned()
            .collect();

        let mut inputs = Vec::with_capacity(symbols.len());
        let remaining_balance = self.state.remaining_balance;
        let fallback = self.config.live.fallback_to_baseline_if_rf_missing;
        for symbol in symbols {
            let state = match self.state.symbols.get(&symbol) {
                Some(s) => s.clone(),
                None => continue,
            };
            let mctx = match contexts.get(&symbol) {
                Some(m) => m.clone(),
                None => continue,
            };
            let snapshot = Arc::new(self.registry.snapshot(&symbol));
            let decision_secs = crate::clock::parse_rfc3339_secs(&mctx.timestamp)
                .unwrap_or_else(crate::clock::now_unix_secs);
            let news = self.news_for_symbol(&symbol, decision_secs, &mctx.date);
            inputs.push(DecisionInput {
                symbol,
                state,
                snapshot,
                cache: Arc::clone(&self.cache),
                volume: Arc::clone(&self.volume),
                news,
                mctx,
                remaining_balance,
                mdp_cfg: self.config.mdp.clone(),
                fallback_to_baseline: fallback,
                expected_costs: 0.0,
                inject_delay: None,
            });
        }
        inputs
    }

    /// One full minute cycle with PARALLEL per-symbol decisions and a single,
    /// centralized order-execution pass. Falls back to the sequential `run_tick`
    /// when `live_parallel.enabled` is false.
    pub fn run_cycle(&mut self, contexts: &HashMap<String, MarketContext>) -> anyhow::Result<()> {
        if !self.config.live_parallel.enabled {
            return self.run_tick(contexts);
        }

        let cycle_start = Instant::now();
        let cycle_ts = crate::clock::now_rfc3339();

        if self.kill_switch_active() {
            self.halted = true;
            return Ok(());
        }

        let md_start = Instant::now();
        let balance = self.exchange.total_wallet_balance()?;
        self.state.remaining_balance = balance;
        let market_data_ms = ms_since(md_start);

        if self.enforce_kill_switch(balance)? {
            return Ok(());
        }

        // Capture immutable per-cycle inputs (model snapshots).
        let art_start = Instant::now();
        let inputs = self.build_decision_inputs(contexts);
        let artifact_lookup_ms = ms_since(art_start);
        let n_symbols = inputs.len();

        // ---- Parallel decision computation ----
        let dec_start = Instant::now();
        let max_workers = self.config.live_parallel.max_symbol_workers.max(1);
        let timeout_ms = self.config.live_parallel.decision_timeout_ms;
        let outputs = parallel::run_parallel(inputs, max_workers, timeout_ms);
        let decision_compute_ms = ms_since(dec_start);

        // Aggregate timings / slowest symbol / timeouts.
        let mut rf_inference_ms = 0.0;
        let mut mdp_compute_ms = 0.0;
        let mut slowest_symbol = String::new();
        let mut slowest_symbol_ms = 0.0;
        let mut n_timeouts = 0usize;
        for o in &outputs {
            rf_inference_ms += o.timing.rf_inference_ms;
            mdp_compute_ms += o.timing.mdp_compute_ms;
            if o.timed_out {
                n_timeouts += 1;
            }
            if o.compute_ms > slowest_symbol_ms {
                slowest_symbol_ms = o.compute_ms;
                slowest_symbol = o.symbol.clone();
            }
        }

        // ---- Build + prioritize the order intent queue ----
        let queue_start = Instant::now();
        let mut decisions: Vec<SymbolDecision> = Vec::new();
        let mut queue: Vec<(usize, OrderIntent)> = Vec::new();
        for o in outputs {
            let decision = match o.decision {
                Some(d) => d,
                None => continue, // timeout / no-policy => conservative hold
            };
            let idx = decisions.len();
            let symbol = decision.record.symbol.clone();
            if let Some(state) = self.state.symbols.get(&symbol) {
                let prio = order::close_priority(&decision.record.reason);
                let intents = order::intents_from_action(
                    &decision.action,
                    &symbol,
                    state,
                    &decision.record.timestamp,
                    &decision.record.reason,
                    &decision.record.rf_model_version,
                    &symbol,
                    prio,
                );
                for it in intents {
                    queue.push((idx, it));
                }
            }
            decisions.push(decision);
        }
        // Highest priority first (KillSwitchClose < ... < NewDayOpen).
        queue.sort_by(|a, b| {
            a.1.priority
                .cmp(&b.1.priority)
                .then(a.1.symbol.cmp(&b.1.symbol))
        });
        let order_queue_ms = ms_since(queue_start);

        // ---- Centralized, sequential execution ----
        let exec_start = Instant::now();
        let mut order_ids_by_idx: HashMap<usize, Vec<String>> = HashMap::new();
        {
            let mut risk = RiskManager::new(
                self.config.live.shadow_mode,
                self.config.live.real_money,
                self.config.risk.min_total_wallet_balance_usdt,
                false,
                balance,
            );
            let mut executor = OrderExecutor::new(
                &self.exchange,
                self.config.live_parallel.order_rate_limit_enabled,
                self.config.live_parallel.max_orders_per_second,
            );
            for (idx, intent) in &queue {
                let position_open = self.position_open_for(intent);
                let price = decisions[*idx].record.current_price;
                let filters = self
                    .exchange
                    .symbol_filters(&intent.symbol)
                    .unwrap_or_default();
                let ctx = ExecContext {
                    position_open,
                    price,
                    filters,
                };
                let outcome = executor.execute(intent, &ctx, &mut risk);
                order_ids_by_idx
                    .entry(*idx)
                    .or_default()
                    .extend(outcome.order_ids);
            }
        }
        let order_execution_ms = ms_since(exec_start);

        // ---- Apply state mutations + log decisions ----
        for (idx, mut decision) in decisions.into_iter().enumerate() {
            decision.record.order_ids = order_ids_by_idx.remove(&idx).unwrap_or_default();
            let symbol = decision.record.symbol.clone();
            let price = decision.record.current_price;
            let decision_secs = crate::clock::parse_rfc3339_secs(&decision.record.timestamp)
                .unwrap_or_else(crate::clock::now_unix_secs);
            if let Some((mode, asof, src, source, quality)) =
                self.news_provenance(&symbol, decision_secs)
            {
                if mode == "previous_completed_day_fallback" {
                    eprintln!(
                        "news_mode symbol={symbol} mode={mode} asof={asof} reason=date_only_or_ambiguous_timestamp source_feature_date={}",
                        src.as_deref().unwrap_or("null")
                    );
                }
                decision.record.news_mode = Some(mode);
                decision.record.news_source = source;
                decision.record.news_timestamp_quality = quality;
                decision.record.news_asof_timestamp = Some(asof);
                decision.record.news_source_feature_date = src;
            }
            let state_snapshot = self.state.symbols.get(&symbol).cloned();
            match (&decision.action, state_snapshot) {
                (Action::CloseHedged | Action::CloseLong | Action::CloseShort, Some(st)) => {
                    self.apply_close_state(&symbol, &st, price);
                }
                (Action::AddMarginLong(x), Some(_)) => {
                    self.apply_add_margin_state(&symbol, Side::Long, *x);
                }
                (Action::AddMarginShort(x), Some(_)) => {
                    self.apply_add_margin_state(&symbol, Side::Short, *x);
                }
                _ => {}
            }
            self.log_decision(decision.record);
        }
        self.save_state();

        // ---- Performance metrics ----
        let cycle_total_ms = ms_since(cycle_start);
        let metrics = CycleMetrics {
            cycle_timestamp: cycle_ts,
            n_symbols,
            decision_parallel_enabled: true,
            max_symbol_workers: max_workers,
            cycle_total_ms,
            market_data_ms,
            artifact_lookup_ms,
            decision_compute_ms,
            rf_inference_ms,
            mdp_compute_ms,
            order_queue_ms,
            order_execution_ms,
            slowest_symbol,
            slowest_symbol_ms,
            n_timeouts,
            alert: CycleMetrics::classify(cycle_total_ms),
        };
        crate::perf::append(&self.config.paths.live_performance, &metrics);

        Ok(())
    }

    /// Whether the position/leg an intent targets is currently open.
    fn position_open_for(&self, intent: &OrderIntent) -> bool {
        use crate::order::OrderIntentType::*;
        match self.state.symbols.get(&intent.symbol) {
            None => false,
            Some(st) => match intent.intent_type {
                CloseLong | AddMarginLong | OpenLong => st.long.is_some(),
                CloseShort | AddMarginShort | OpenShort => st.short.is_some(),
                CancelOpenOrders => true,
            },
        }
    }
}
