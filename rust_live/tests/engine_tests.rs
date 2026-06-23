//! Integration tests covering section 22 expectations.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use trading_live::config::{Config, Paths};
use trading_live::decision::{Action, MarketContext};
use trading_live::engine::Engine;
use trading_live::exchange::mock::MockExchange;
use trading_live::exchange::PositionInfo;
use trading_live::features::MarketVolume;
use trading_live::pricing::{LegState, Side};
use trading_live::sizing::SymbolFilters;
use trading_live::state::{Mode, SymbolState};

static COUNTER: AtomicU64 = AtomicU64::new(0);

struct TmpDir {
    path: PathBuf,
}

impl TmpDir {
    fn new() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!("tl_test_{}_{}_{}", std::process::id(), nanos, id));
        std::fs::create_dir_all(&path).unwrap();
        TmpDir { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn write(path: &Path, contents: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, contents).unwrap();
}

fn paths_for(dir: &Path) -> Paths {
    let p = |s: &str| dir.join(s).to_string_lossy().to_string();
    Paths {
        distribution_snapshot: p("data/distribution_snapshot.json"),
        integral_cache: p("data/integral_cache.json"),
        portfolio_weights: p("data/portfolio_weights.json"),
        predicted_daily_volume: p("data/predicted_daily_volume.jsonl"),
        news_features_daily: p("data/news_features_daily.jsonl"),
        news_features_intraday: p("data/news_features_intraday.jsonl"),
        correlation_matrix_daily: p("data/correlation_matrix_daily.jsonl"),
        models_promoted: p("models/promoted"),
        live_position_state: p("data/live_position_state.json"),
        live_decisions: p("data/live_decisions.jsonl"),
        realized_symbol_returns: p("data/realized_symbol_returns.jsonl"),
        kill_switch_triggered: p("data/kill_switch_triggered.json"),
        live_performance: p("data/live_performance.jsonl"),
    }
}

fn write_cache(dir: &Path, symbols: &[&str]) {
    let mut map = serde_json::Map::new();
    for s in symbols {
        map.insert(
            (*s).to_string(),
            serde_json::json!({
                "symbol": s,
                "grid": [-0.2, -0.1, 0.0, 0.1, 0.2],
                "denom": 1.0,
                "cum_long": {"return": [0.0, 0.5, 1.0, 1.8, 3.0]},
                "cum_short": {"return": [0.0, 0.4, 0.9, 1.5, 2.2]}
            }),
        );
    }
    let v = serde_json::json!({ "symbols": map });
    write(
        &dir.join("data/integral_cache.json"),
        &serde_json::to_string(&v).unwrap(),
    );
}

fn write_weights(dir: &Path, symbols: &[&str]) {
    write_weights_dated(dir, symbols, &trading_live::clock::today_date());
}

fn write_weights_dated(dir: &Path, symbols: &[&str], as_of_date: &str) {
    let mut map = serde_json::Map::new();
    let mut sum = 0.0;
    let mut selected = Vec::new();
    for s in symbols {
        map.insert(
            (*s).to_string(),
            serde_json::json!({
                "valid": true,
                "weight_discrete": 0.2,
                "long_weight": 0.1,
                "short_weight": 0.1
            }),
        );
        sum += 0.2;
        selected.push((*s).to_string());
    }
    let v = serde_json::json!({
        "as_of_date": as_of_date,
        "symbols": map,
        "sum_weight_discrete": sum,
        "cash_weight": 1.0 - sum,
        "weight_semantics": "margin_fraction_of_total_equity",
        "portfolio_version": "mvo_bnb_v1",
        "selected_symbols": selected
    });
    write(
        &dir.join("data/portfolio_weights.json"),
        &serde_json::to_string(&v).unwrap(),
    );
}

// Writes a per-symbol close classifier whose single leaf P(close) == `p_close`.
// CLOSE when p_close >= 0.5, otherwise CONTINUE.
fn write_model(dir: &Path, symbol: &str, p_close: f64) {
    let v = serde_json::json!({
        "type": "random_forest_classifier",
        "model_version": "rf_close_classifier_v1",
        "threshold": 0.5,
        "final_feature_order": ["LongEdge_Return"],
        "scale_cols": [],
        "passthrough_cols": ["LongEdge_Return"],
        "imputer": {"strategy": "median", "medians": {}},
        "scaler": {"type": "robust", "center": [], "scale": []},
        "trees": [{
            "children_left": [-1],
            "children_right": [-1],
            "feature": [-2],
            "threshold": [0.0],
            "value": [p_close]
        }]
    });
    write(
        &dir.join(format!("models/promoted/{symbol}/rf_close_classifier.json")),
        &serde_json::to_string(&v).unwrap(),
    );
}

fn write_intraday_news(dir: &Path, symbol: &str, asof_rfc3339: &str) {
    let line = serde_json::json!({
        "asof_timestamp": asof_rfc3339,
        "symbol": symbol,
        "feature_version": "news_v1",
        "macro_news_sentiment": 0.0,
        "macro_news_count": 0
    });
    write(
        &dir.join("data/news_features_intraday.jsonl"),
        &(serde_json::to_string(&line).unwrap() + "\n"),
    );
}

fn base_config(dir: &Path, symbols: &[&str]) -> Config {
    let mut cfg = Config::default();
    cfg.paths = paths_for(dir);
    cfg.symbols = symbols.iter().map(|s| s.to_string()).collect();
    cfg.exchange.default_leverage = 1.0;
    cfg
}

fn mock_with_prices(balance: f64, symbols: &[&str]) -> MockExchange {
    let ex = MockExchange::new(balance);
    for s in symbols {
        ex.set_mark_price(s, 100.0);
        ex.set_filters(
            s,
            SymbolFilters {
                step_size: 0.001,
                min_qty: 0.0,
                min_notional: 0.0,
            },
        );
    }
    ex
}

fn ctx() -> MarketContext {
    MarketContext {
        timestamp: "2024-09-29T00:00:00Z".into(),
        date: "2024-09-29".into(),
        current_price: 100.0,
        hour_of_day: 0.0,
        day_of_week: 0.0,
        market_volume: MarketVolume::default(),
        halvings: Arc::new(Vec::new()),
        season_seed: 0,
    }
}

fn hedged(symbol_price: f64, qty: f64) -> SymbolState {
    SymbolState {
        mode: Mode::HedgedBothActive,
        long: Some(LegState::new(
            Side::Long,
            symbol_price,
            qty,
            qty * symbol_price,
        )),
        short: Some(LegState::new(
            Side::Short,
            symbol_price,
            qty,
            qty * symbol_price,
        )),
        first_liq_price: None,
        total_added_margin_long: 0.0,
        total_added_margin_short: 0.0,
    }
}

// 1. no portfolio_weights -> no new trade
#[test]
fn no_weights_no_trade() {
    let dir = TmpDir::new();
    write_cache(dir.path(), &["BTCUSDT"]);
    write(&dir.path().join("data/distribution_snapshot.json"), "{}");
    // No weights file written.
    let cfg = base_config(dir.path(), &["BTCUSDT"]);
    let ex = mock_with_prices(1000.0, &["BTCUSDT"]);
    let mut engine = Engine::new(cfg, ex).unwrap();
    assert!(!engine.can_open_symbol("BTCUSDT"));
    let opened = engine.open_initial_positions().unwrap();
    assert_eq!(opened, 0);
    assert_eq!(engine.exchange.placed_count(), 0);
}

// 3. require_rf_for_new_trade true + RF missing -> no trade
#[test]
fn require_rf_blocks_when_missing() {
    let dir = TmpDir::new();
    write_cache(dir.path(), &["BTCUSDT"]);
    write_weights(dir.path(), &["BTCUSDT"]);
    let mut cfg = base_config(dir.path(), &["BTCUSDT"]);
    cfg.live.require_rf_for_new_trade = true;
    let ex = mock_with_prices(1000.0, &["BTCUSDT"]);
    let mut engine = Engine::new(cfg, ex).unwrap();
    assert!(!engine.can_open_symbol("BTCUSDT"));
    assert_eq!(engine.open_initial_positions().unwrap(), 0);
}

// 2. RF missing fallback works (require_rf false + baseline fallback)
#[test]
fn baseline_fallback_allows_open() {
    let dir = TmpDir::new();
    write_cache(dir.path(), &["BTCUSDT"]);
    write_weights(dir.path(), &["BTCUSDT"]);
    let mut cfg = base_config(dir.path(), &["BTCUSDT"]);
    cfg.live.require_rf_for_new_trade = false;
    cfg.live.fallback_to_baseline_if_rf_missing = true;
    let ex = mock_with_prices(1000.0, &["BTCUSDT"]);
    let mut engine = Engine::new(cfg, ex).unwrap();
    assert!(engine.can_open_symbol("BTCUSDT"));
    let opened = engine.open_initial_positions().unwrap();
    assert_eq!(opened, 1);
    // 18/19: shadow + real_money=false -> no orders sent.
    assert_eq!(engine.exchange.placed_count(), 0);
}

// 4 & 5. each symbol uses its OWN model; BTC model not used for ETH
// 15. RF CLOSE -> reduceOnly close orders
#[test]
fn per_symbol_models_and_reduce_only_close() {
    let dir = TmpDir::new();
    write_cache(dir.path(), &["BTCUSDT", "ETHUSDT"]);
    write_weights(dir.path(), &["BTCUSDT", "ETHUSDT"]);
    write_model(dir.path(), "BTCUSDT", 1.0); // p_close 1.0 >= 0.5 -> CLOSE
    write_model(dir.path(), "ETHUSDT", 0.0); // p_close 0.0 <  0.5 -> CONTINUE
    let mut cfg = base_config(dir.path(), &["BTCUSDT", "ETHUSDT"]);
    cfg.live.real_money = true;
    cfg.live.shadow_mode = false;
    let ex = mock_with_prices(1000.0, &["BTCUSDT", "ETHUSDT"]);
    let mut engine = Engine::new(cfg, ex).unwrap();
    engine.state.remaining_balance = 1000.0;
    engine
        .state
        .symbols
        .insert("BTCUSDT".into(), hedged(100.0, 1.0));
    engine
        .state
        .symbols
        .insert("ETHUSDT".into(), hedged(100.0, 1.0));

    let c = ctx();
    let btc = engine.tick_symbol("BTCUSDT", &c).unwrap();
    assert_eq!(btc.action, Action::CloseHedged);

    let eth = engine.tick_symbol("ETHUSDT", &c).unwrap();
    assert_eq!(eth.action, Action::Hold);

    // Only BTC produced orders, and they are reduceOnly.
    let orders = engine.exchange.placed_snapshot();
    assert_eq!(orders.len(), 2);
    assert!(orders
        .iter()
        .all(|o| o.reduce_only && o.symbol == "BTCUSDT"));

    // 20. decisions logged
    let decisions = std::fs::read_to_string(dir.path().join("data/live_decisions.jsonl")).unwrap();
    assert!(decisions.lines().count() >= 2);
}

// Spec section 2/7: a missing classifier blocks new opens AND, for an existing
// position, holds (no order) while logging an explicit error reason. No fallback
// to any legacy regressor/edge policy.
#[test]
fn missing_classifier_holds_logs_and_blocks_open() {
    let dir = TmpDir::new();
    write_cache(dir.path(), &["BTCUSDT"]);
    write_weights(dir.path(), &["BTCUSDT"]);
    // No classifier artifact written.
    let mut cfg = base_config(dir.path(), &["BTCUSDT"]);
    cfg.live.real_money = true;
    cfg.live.shadow_mode = false;
    cfg.live.require_rf_for_new_trade = true;
    let ex = mock_with_prices(1000.0, &["BTCUSDT"]);
    let mut engine = Engine::new(cfg, ex).unwrap();

    // No NEW position opened when the classifier is missing.
    assert!(!engine.can_open_symbol("BTCUSDT"));
    assert_eq!(engine.open_initial_positions().unwrap(), 0);

    // An EXISTING position is held (not closed); reason logged.
    engine.state.remaining_balance = 1000.0;
    engine
        .state
        .symbols
        .insert("BTCUSDT".into(), hedged(100.0, 1.0));
    let mut contexts = HashMap::new();
    contexts.insert("BTCUSDT".to_string(), ctx());
    engine.run_cycle(&contexts).unwrap();

    assert!(
        engine.state.symbols.contains_key("BTCUSDT"),
        "position must not be closed"
    );
    assert_eq!(
        engine.exchange.placed_count(),
        0,
        "no orders on missing classifier"
    );
    let decisions = std::fs::read_to_string(dir.path().join("data/live_decisions.jsonl")).unwrap();
    assert!(
        decisions.contains("missing_symbol_model"),
        "decision log must record the explicit missing-classifier reason"
    );
}

// 18 & 19. shadow / real_money=false -> no order even on CLOSE
#[test]
fn shadow_mode_sends_no_orders() {
    let dir = TmpDir::new();
    write_cache(dir.path(), &["BTCUSDT"]);
    write_weights(dir.path(), &["BTCUSDT"]);
    write_model(dir.path(), "BTCUSDT", 1.0);
    let mut cfg = base_config(dir.path(), &["BTCUSDT"]);
    cfg.live.shadow_mode = true;
    cfg.live.real_money = false;
    let ex = mock_with_prices(1000.0, &["BTCUSDT"]);
    let mut engine = Engine::new(cfg, ex).unwrap();
    engine.state.remaining_balance = 1000.0;
    engine
        .state
        .symbols
        .insert("BTCUSDT".into(), hedged(100.0, 1.0));

    let d = engine.tick_symbol("BTCUSDT", &ctx()).unwrap();
    assert_eq!(d.action, Action::CloseHedged);
    assert_eq!(engine.exchange.placed_count(), 0);
}

// 16. kill-switch triggers when balance < 100
#[test]
fn kill_switch_triggers_on_low_balance() {
    let dir = TmpDir::new();
    write_cache(dir.path(), &["BTCUSDT"]);
    write_weights(dir.path(), &["BTCUSDT"]);
    let mut cfg = base_config(dir.path(), &["BTCUSDT"]);
    cfg.live.real_money = true;
    cfg.live.shadow_mode = false;
    let ex = mock_with_prices(50.0, &["BTCUSDT"]);
    *ex.positions.lock().unwrap() = vec![PositionInfo {
        symbol: "BTCUSDT".into(),
        position_side: Side::Long,
        position_amt: 1.0,
        entry_price: 100.0,
        leverage: 1.0,
        liquidation_price: None,
        unrealized_pnl: 0.0,
        isolated_margin: 0.0,
    }];
    let mut engine = Engine::new(cfg, ex).unwrap();

    let triggered = engine.enforce_kill_switch(50.0).unwrap();
    assert!(triggered);
    assert!(Path::new(&engine.config.paths.kill_switch_triggered).exists());
    assert!(engine.halted);
    // Position flattened with a reduceOnly close.
    let orders = engine.exchange.placed_snapshot();
    assert!(orders.iter().any(|o| o.reduce_only));
}

// 17. kill_switch_triggered.json present -> bot does not open trades
#[test]
fn kill_switch_file_blocks_trading() {
    let dir = TmpDir::new();
    write_cache(dir.path(), &["BTCUSDT"]);
    write_weights(dir.path(), &["BTCUSDT"]);
    write_model(dir.path(), "BTCUSDT", 1.0);
    write(&dir.path().join("data/kill_switch_triggered.json"), "{}");
    let cfg = base_config(dir.path(), &["BTCUSDT"]);
    let ex = mock_with_prices(1000.0, &["BTCUSDT"]);
    let mut engine = Engine::new(cfg, ex).unwrap();
    assert!(engine.kill_switch_active());
    assert!(!engine.can_open_symbol("BTCUSDT"));
    assert_eq!(engine.open_initial_positions().unwrap(), 0);
}

// Live as-of news gate: a STALE intraday artifact blocks opening a new position
// (existing positions are unaffected, exercised elsewhere). Without the gate the
// baseline fallback would otherwise allow the open.
#[test]
fn stale_news_blocks_open() {
    let dir = TmpDir::new();
    write_cache(dir.path(), &["BTCUSDT"]);
    write_weights(dir.path(), &["BTCUSDT"]);
    // asof far in the past -> age >> max_news_feature_age_minutes.
    write_intraday_news(dir.path(), "BTCUSDT", "2020-01-01T00:00:00Z");
    let mut cfg = base_config(dir.path(), &["BTCUSDT"]);
    cfg.live.require_rf_for_new_trade = false;
    cfg.live.fallback_to_baseline_if_rf_missing = true;
    cfg.news.intraday_enabled = true;
    cfg.news.max_news_feature_age_minutes = 30;
    let ex = mock_with_prices(1000.0, &["BTCUSDT"]);
    let mut engine = Engine::new(cfg, ex).unwrap();
    assert!(!engine.can_open_symbol("BTCUSDT"));
    assert_eq!(engine.open_initial_positions().unwrap(), 0);
}

// Live as-of news gate: a MISSING intraday artifact (failed/never-run worker)
// blocks opening on zero-news rather than silently opening.
#[test]
fn missing_news_blocks_open() {
    let dir = TmpDir::new();
    write_cache(dir.path(), &["BTCUSDT"]);
    write_weights(dir.path(), &["BTCUSDT"]);
    // No intraday artifact written.
    let mut cfg = base_config(dir.path(), &["BTCUSDT"]);
    cfg.live.require_rf_for_new_trade = false;
    cfg.live.fallback_to_baseline_if_rf_missing = true;
    cfg.news.intraday_enabled = true;
    let ex = mock_with_prices(1000.0, &["BTCUSDT"]);
    let mut engine = Engine::new(cfg, ex).unwrap();
    assert!(!engine.can_open_symbol("BTCUSDT"));
    assert_eq!(engine.open_initial_positions().unwrap(), 0);
}

// Live as-of news gate: a FRESH intraday artifact permits the open as usual.
#[test]
fn fresh_news_allows_open() {
    let dir = TmpDir::new();
    write_cache(dir.path(), &["BTCUSDT"]);
    write_weights(dir.path(), &["BTCUSDT"]);
    // Fresh but past the safety-lag cutoff: asof = now - 10min (lag is 5min,
    // max age 30min) -> selectable and within the freshness budget.
    let asof = trading_live::clock::rfc3339(trading_live::clock::now_unix_secs() - 600);
    write_intraday_news(dir.path(), "BTCUSDT", &asof);
    let mut cfg = base_config(dir.path(), &["BTCUSDT"]);
    cfg.live.require_rf_for_new_trade = false;
    cfg.live.fallback_to_baseline_if_rf_missing = true;
    cfg.news.intraday_enabled = true;
    cfg.news.max_news_feature_age_minutes = 30;
    cfg.news.news_safety_lag_seconds = 300;
    let ex = mock_with_prices(1000.0, &["BTCUSDT"]);
    let mut engine = Engine::new(cfg, ex).unwrap();
    assert!(engine.can_open_symbol("BTCUSDT"));
    assert_eq!(engine.open_initial_positions().unwrap(), 1);
}

// Fail-safe: a fresh, selectable intraday record whose feature_version does NOT
// match the configured expectation (model trained on a different news schema)
// blocks new opens (schema mismatch), while a matching version opens normally.
#[test]
fn news_schema_mismatch_blocks_open() {
    let dir = TmpDir::new();
    write_cache(dir.path(), &["BTCUSDT"]);
    write_weights(dir.path(), &["BTCUSDT"]);
    let asof = trading_live::clock::rfc3339(trading_live::clock::now_unix_secs() - 600);
    // Artifact is stamped feature_version="news_v1".
    write_intraday_news(dir.path(), "BTCUSDT", &asof);
    let mut cfg = base_config(dir.path(), &["BTCUSDT"]);
    cfg.live.require_rf_for_new_trade = false;
    cfg.live.fallback_to_baseline_if_rf_missing = true;
    cfg.news.intraday_enabled = true;
    cfg.news.max_news_feature_age_minutes = 30;
    cfg.news.news_safety_lag_seconds = 300;
    cfg.news.expected_feature_version = "news_v2".to_string(); // mismatch
    let ex = mock_with_prices(1000.0, &["BTCUSDT"]);
    let mut engine = Engine::new(cfg, ex).unwrap();
    assert!(!engine.can_open_symbol("BTCUSDT"));
    assert_eq!(engine.open_initial_positions().unwrap(), 0);
}

// The same artifact opens normally once the expected feature_version matches.
#[test]
fn news_schema_match_allows_open() {
    let dir = TmpDir::new();
    write_cache(dir.path(), &["BTCUSDT"]);
    write_weights(dir.path(), &["BTCUSDT"]);
    let asof = trading_live::clock::rfc3339(trading_live::clock::now_unix_secs() - 600);
    write_intraday_news(dir.path(), "BTCUSDT", &asof);
    let mut cfg = base_config(dir.path(), &["BTCUSDT"]);
    cfg.live.require_rf_for_new_trade = false;
    cfg.live.fallback_to_baseline_if_rf_missing = true;
    cfg.news.intraday_enabled = true;
    cfg.news.max_news_feature_age_minutes = 30;
    cfg.news.news_safety_lag_seconds = 300;
    cfg.news.expected_feature_version = "news_v1".to_string(); // matches artifact
    let ex = mock_with_prices(1000.0, &["BTCUSDT"]);
    let mut engine = Engine::new(cfg, ex).unwrap();
    assert!(engine.can_open_symbol("BTCUSDT"));
    assert_eq!(engine.open_initial_positions().unwrap(), 1);
}

// No-news mode: with intraday news disabled the open-gate never blocks on news,
// even when NO news artifact exists at all (worker/timer not deployed). The
// no-news RF model is used and the position opens normally.
#[test]
fn no_news_mode_allows_open_without_news_artifact() {
    let dir = TmpDir::new();
    write_cache(dir.path(), &["BTCUSDT"]);
    write_weights(dir.path(), &["BTCUSDT"]);
    write_model(dir.path(), "BTCUSDT", 0.0); // no-news model (only LongEdge_Return)
    let mut cfg = base_config(dir.path(), &["BTCUSDT"]);
    cfg.news.intraday_enabled = false; // no-news configuration
                                       // Deliberately write no intraday/daily news artifacts.
    let ex = mock_with_prices(1000.0, &["BTCUSDT"]);
    let mut engine = Engine::new(cfg, ex).unwrap();
    assert!(engine.can_open_symbol("BTCUSDT"));
    assert_eq!(engine.open_initial_positions().unwrap(), 1);
}

// Fail-safe: a STALE MVO artifact (as_of_date far in the past) blocks new opens.
#[test]
fn stale_weights_block_open() {
    let dir = TmpDir::new();
    write_cache(dir.path(), &["BTCUSDT"]);
    write_weights_dated(dir.path(), &["BTCUSDT"], "2020-01-01");
    let mut cfg = base_config(dir.path(), &["BTCUSDT"]);
    cfg.live.require_rf_for_new_trade = false;
    cfg.live.fallback_to_baseline_if_rf_missing = true;
    cfg.sizing.max_portfolio_age_days = 2;
    let ex = mock_with_prices(1000.0, &["BTCUSDT"]);
    let mut engine = Engine::new(cfg, ex).unwrap();
    assert!(!engine.portfolio_ok());
    assert!(!engine.can_open_symbol("BTCUSDT"));
    assert_eq!(engine.open_initial_positions().unwrap(), 0);
}

// Fail-safe: weights whose sum + cash_weight != 1.0 block new opens.
#[test]
fn invalid_weight_sum_blocks_open() {
    let dir = TmpDir::new();
    write_cache(dir.path(), &["BTCUSDT"]);
    let v = serde_json::json!({
        "as_of_date": trading_live::clock::today_date(),
        "symbols": {"BTCUSDT": {"valid": true, "weight_discrete": 0.2}},
        "sum_weight_discrete": 0.2,
        "cash_weight": 0.5, // 0.2 + 0.5 = 0.7 != 1.0
        "weight_semantics": "margin_fraction_of_total_equity"
    });
    write(
        &dir.path().join("data/portfolio_weights.json"),
        &serde_json::to_string(&v).unwrap(),
    );
    let mut cfg = base_config(dir.path(), &["BTCUSDT"]);
    cfg.live.require_rf_for_new_trade = false;
    cfg.live.fallback_to_baseline_if_rf_missing = true;
    let ex = mock_with_prices(1000.0, &["BTCUSDT"]);
    let mut engine = Engine::new(cfg, ex).unwrap();
    assert!(!engine.can_open_symbol("BTCUSDT"));
    assert_eq!(engine.open_initial_positions().unwrap(), 0);
}

// Fail-safe: an incompatible weight_semantics (schema mismatch) blocks opens.
#[test]
fn wrong_semantics_blocks_open() {
    let dir = TmpDir::new();
    write_cache(dir.path(), &["BTCUSDT"]);
    let v = serde_json::json!({
        "as_of_date": trading_live::clock::today_date(),
        "symbols": {"BTCUSDT": {"valid": true, "weight_discrete": 0.2}},
        "sum_weight_discrete": 0.2,
        "cash_weight": 0.8,
        "weight_semantics": "notional_fraction"
    });
    write(
        &dir.path().join("data/portfolio_weights.json"),
        &serde_json::to_string(&v).unwrap(),
    );
    let mut cfg = base_config(dir.path(), &["BTCUSDT"]);
    cfg.live.require_rf_for_new_trade = false;
    cfg.live.fallback_to_baseline_if_rf_missing = true;
    let ex = mock_with_prices(1000.0, &["BTCUSDT"]);
    let mut engine = Engine::new(cfg, ex).unwrap();
    assert!(!engine.can_open_symbol("BTCUSDT"));
    assert_eq!(engine.open_initial_positions().unwrap(), 0);
}

// Rebalance must size off total_equity * weight and NOT re-allocate capital that
// is already deployed: an existing position already at target produces no order.
#[test]
fn rebalance_no_double_allocation() {
    let dir = TmpDir::new();
    write_cache(dir.path(), &["BTCUSDT"]);
    write_weights(dir.path(), &["BTCUSDT"]); // weight 0.2
    let mut cfg = base_config(dir.path(), &["BTCUSDT"]);
    cfg.live.real_money = true;
    cfg.live.shadow_mode = false;
    let ex = mock_with_prices(1000.0, &["BTCUSDT"]);
    let mut engine = Engine::new(cfg, ex).unwrap();
    // total_equity 1000 * 0.2 = 200 margin, 100/leg, lev 1, price 100 -> qty 1.0.
    engine
        .state
        .symbols
        .insert("BTCUSDT".into(), hedged(100.0, 1.0));
    let summary = engine.rebalance_to_targets().unwrap();
    assert_eq!(summary.opened, 0);
    assert_eq!(summary.reduced, 0);
    assert_eq!(
        engine.exchange.placed_count(),
        0,
        "position already at target must not be re-allocated"
    );
}

// Rebalance closes a symbol that the new MVO artifact dropped (target weight 0).
#[test]
fn rebalance_closes_out_of_scope_symbol() {
    let dir = TmpDir::new();
    write_cache(dir.path(), &["BTCUSDT", "ETHUSDT"]);
    write_weights(dir.path(), &["BTCUSDT"]); // only BTC is in scope now
    let mut cfg = base_config(dir.path(), &["BTCUSDT"]);
    cfg.live.real_money = true;
    cfg.live.shadow_mode = false;
    let ex = mock_with_prices(1000.0, &["BTCUSDT", "ETHUSDT"]);
    let mut engine = Engine::new(cfg, ex).unwrap();
    // An open ETH position that is no longer in the MVO portfolio.
    engine
        .state
        .symbols
        .insert("ETHUSDT".into(), hedged(100.0, 1.0));
    let summary = engine.rebalance_to_targets().unwrap();
    assert_eq!(summary.closed, 1);
    assert!(!engine.state.symbols.contains_key("ETHUSDT"));
    let orders = engine.exchange.placed_snapshot();
    assert!(orders
        .iter()
        .any(|o| o.reduce_only && o.symbol == "ETHUSDT"));
}

// 12 & 13 & 14. long/short equal split + step round down + minNotional handled
#[test]
fn opening_quantities_equal_and_rounded() {
    let dir = TmpDir::new();
    write_cache(dir.path(), &["BTCUSDT"]);
    write_weights(dir.path(), &["BTCUSDT"]);
    write_model(dir.path(), "BTCUSDT", 1.0);
    let mut cfg = base_config(dir.path(), &["BTCUSDT"]);
    cfg.live.real_money = true;
    cfg.live.shadow_mode = false;
    let ex = MockExchange::new(1000.0);
    ex.set_mark_price("BTCUSDT", 100.0);
    ex.set_filters(
        "BTCUSDT",
        SymbolFilters {
            step_size: 0.001,
            min_qty: 0.0,
            min_notional: 0.0,
        },
    );
    let mut engine = Engine::new(cfg, ex).unwrap();
    let opened = engine.open_initial_positions().unwrap();
    assert_eq!(opened, 1);
    let orders = engine.exchange.placed_snapshot();
    assert_eq!(orders.len(), 2);
    let long = orders
        .iter()
        .find(|o| o.position_side == Side::Long)
        .unwrap();
    let short = orders
        .iter()
        .find(|o| o.position_side == Side::Short)
        .unwrap();
    assert!((long.quantity - short.quantity).abs() < 1e-9);
}
