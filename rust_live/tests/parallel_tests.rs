//! Tests for parallel decision computation + centralized order execution
//! (spec section 15).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use trading_live::artifacts::close_classifier::CloseClassifier;
use trading_live::artifacts::{CacheSet, ModelSnapshot};
use trading_live::config::{Config, Paths};
use trading_live::decision::{Action, MarketContext};
use trading_live::engine::Engine;
use trading_live::exchange::mock::MockExchange;
use trading_live::executor::{ExecContext, OrderExecutor};
use trading_live::features::MarketVolume;
use trading_live::order::{intents_from_action, OrderIntent, OrderIntentType, Priority};
use trading_live::parallel::{self, DecisionInput};
use trading_live::pricing::{LegState, Side};
use trading_live::risk::RiskManager;
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
        path.push(format!("tl_par_{}_{}_{}", std::process::id(), nanos, id));
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

fn cache_json() -> serde_json::Value {
    serde_json::json!({
        "symbol": "X",
        "grid": [-0.2, -0.1, 0.0, 0.1, 0.2],
        "denom": 1.0,
        "cum_long": {"return": [0.0, 0.5, 1.0, 1.8, 3.0]},
        "cum_short": {"return": [0.0, 0.4, 0.9, 1.5, 2.2]}
    })
}

fn write_cache(dir: &Path, symbols: &[&str]) {
    let mut map = serde_json::Map::new();
    for s in symbols {
        let mut v = cache_json();
        v["symbol"] = serde_json::json!(s);
        map.insert((*s).to_string(), v);
    }
    write(
        &dir.join("data/integral_cache.json"),
        &serde_json::to_string(&serde_json::json!({ "symbols": map })).unwrap(),
    );
}

fn write_weights(dir: &Path, symbols: &[&str]) {
    let mut map = serde_json::Map::new();
    for s in symbols {
        map.insert(
            (*s).to_string(),
            serde_json::json!({"valid": true, "weight_discrete": 0.2, "long_weight": 0.1, "short_weight": 0.1}),
        );
    }
    write(
        &dir.join("data/portfolio_weights.json"),
        &serde_json::to_string(&serde_json::json!({
            "as_of_date": "2024-09-29", "symbols": map, "sum_weight_discrete": 0.4, "cash_weight": 0.6
        }))
        .unwrap(),
    );
}

// Per-symbol close classifier with single leaf P(close) == `p_close`.
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
            "children_left": [-1], "children_right": [-1],
            "feature": [-2], "threshold": [0.0], "value": [p_close]
        }]
    });
    write(
        &dir.join(format!("models/promoted/{symbol}/rf_close_classifier.json")),
        &serde_json::to_string(&v).unwrap(),
    );
}

fn base_config(dir: &Path, symbols: &[&str]) -> Config {
    let mut cfg = Config::default();
    cfg.paths = paths_for(dir);
    cfg.symbols = symbols.iter().map(|s| s.to_string()).collect();
    cfg.exchange.default_leverage = 1.0;
    // Disable the rate limit so tests never sleep.
    cfg.live_parallel.order_rate_limit_enabled = false;
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

fn hedged(price: f64, qty: f64) -> SymbolState {
    SymbolState {
        mode: Mode::HedgedBothActive,
        long: Some(LegState::new(Side::Long, price, qty, qty * price)),
        short: Some(LegState::new(Side::Short, price, qty, qty * price)),
        first_liq_price: None,
        total_added_margin_long: 0.0,
        total_added_margin_short: 0.0,
    }
}

fn contexts_for(symbols: &[&str]) -> HashMap<String, MarketContext> {
    symbols.iter().map(|s| ((*s).to_string(), ctx())).collect()
}

// 2/3/4/15: each symbol uses its own model in parallel; workers never send
// orders directly; only the central executor does -> BTC close, ETH hold.
#[test]
fn parallel_cycle_per_symbol_and_central_execution() {
    let dir = TmpDir::new();
    write_cache(dir.path(), &["BTCUSDT", "ETHUSDT"]);
    write_weights(dir.path(), &["BTCUSDT", "ETHUSDT"]);
    write_model(dir.path(), "BTCUSDT", 1.0); // p_close 1.0 -> CLOSE
    write_model(dir.path(), "ETHUSDT", 0.0); // p_close 0.0 -> CONTINUE
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

    engine
        .run_cycle(&contexts_for(&["BTCUSDT", "ETHUSDT"]))
        .unwrap();

    // BTC closed (removed), ETH still open.
    assert!(!engine.state.symbols.contains_key("BTCUSDT"));
    assert!(engine.state.symbols.contains_key("ETHUSDT"));

    // Only BTC reduceOnly close orders were placed by the central executor.
    let orders = engine.exchange.placed_snapshot();
    assert_eq!(orders.len(), 2);
    assert!(orders
        .iter()
        .all(|o| o.reduce_only && o.symbol == "BTCUSDT"));

    // Decisions + performance both logged.
    let decisions = std::fs::read_to_string(dir.path().join("data/live_decisions.jsonl")).unwrap();
    assert!(decisions.lines().count() >= 2);
    let perf = std::fs::read_to_string(dir.path().join("data/live_performance.jsonl")).unwrap();
    assert_eq!(perf.lines().count(), 1);
    assert!(perf.contains("\"decision_parallel_enabled\":true"));
}

// 3/6/12: shadow mode -> intents produced + logged, but no Binance order sent.
#[test]
fn shadow_mode_parallel_logs_but_sends_nothing() {
    let dir = TmpDir::new();
    write_cache(dir.path(), &["BTCUSDT"]);
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

    engine.run_cycle(&contexts_for(&["BTCUSDT"])).unwrap();

    assert_eq!(engine.exchange.placed_count(), 0);
    let decisions = std::fs::read_to_string(dir.path().join("data/live_decisions.jsonl")).unwrap();
    assert!(decisions.contains("CLOSE"));
    assert!(Path::new(&engine.config.paths.live_performance).exists());
}

// 7: parallel decisions == sequential decisions.
#[test]
fn parallel_equals_sequential() {
    let run = |parallel_enabled: bool| -> HashMap<String, String> {
        let dir = TmpDir::new();
        write_cache(dir.path(), &["BTCUSDT", "ETHUSDT"]);
        write_model(dir.path(), "BTCUSDT", 1.0);
        write_model(dir.path(), "ETHUSDT", 0.0);
        let mut cfg = base_config(dir.path(), &["BTCUSDT", "ETHUSDT"]);
        cfg.live.shadow_mode = true;
        cfg.live.real_money = false;
        cfg.live_parallel.enabled = parallel_enabled;
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
        engine
            .run_cycle(&contexts_for(&["BTCUSDT", "ETHUSDT"]))
            .unwrap();

        let text = std::fs::read_to_string(dir.path().join("data/live_decisions.jsonl")).unwrap();
        let mut out = HashMap::new();
        for line in text.lines() {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            let sym = v["symbol"].as_str().unwrap().to_string();
            let dec = v["decision"].as_str().unwrap().to_string();
            out.insert(sym, dec);
        }
        out
    };
    assert_eq!(run(true), run(false));
}

// Lower-level: run_parallel and run_sequential produce identical outputs.
fn make_input(symbol: &str, delay: Option<Duration>) -> DecisionInput {
    let set: CacheSet = serde_json::from_value(serde_json::json!({
        "symbols": { symbol: cache_json() }
    }))
    .unwrap();
    DecisionInput {
        symbol: symbol.to_string(),
        state: hedged(100.0, 1.0),
        snapshot: Arc::new(ModelSnapshot::default()),
        cache: Arc::new(set),
        volume: Arc::new(Default::default()),
        news: Arc::new(Default::default()),
        mctx: ctx(),
        remaining_balance: 1000.0,
        mdp_cfg: Default::default(),
        fallback_to_baseline: true,
        expected_costs: 0.0,
        inject_delay: delay,
    }
}

#[test]
fn scheduler_parallel_matches_sequential() {
    let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    let seq = parallel::run_sequential(symbols.iter().map(|s| make_input(s, None)).collect());
    let par = parallel::run_parallel(
        symbols.iter().map(|s| make_input(s, None)).collect(),
        8,
        5000,
    );
    assert_eq!(seq.len(), par.len());
    for s in &symbols {
        let a = seq.iter().find(|o| &o.symbol == s).unwrap();
        let b = par.iter().find(|o| &o.symbol == s).unwrap();
        let da = a.decision.as_ref().map(|d| d.record.decision.clone());
        let db = b.decision.as_ref().map(|d| d.record.decision.clone());
        assert_eq!(da, db);
    }
}

// 11: a slow symbol past the timeout yields a safe (timed_out) fallback.
#[test]
fn scheduler_timeout_falls_back() {
    let inputs = vec![
        make_input("BTCUSDT", None),
        make_input("ETHUSDT", Some(Duration::from_millis(400))),
    ];
    let out = parallel::run_parallel(inputs, 8, 50);
    let eth = out.iter().find(|o| o.symbol == "ETHUSDT").unwrap();
    assert!(eth.timed_out);
    assert!(eth.decision.is_none());
    let btc = out.iter().find(|o| o.symbol == "BTCUSDT").unwrap();
    assert!(!btc.timed_out);
    assert!(btc.decision.is_some());
}

// Spec section 4/6: the MDP is feature-only. Even single-leg AND near liquidation
// (where the OLD MDP would have triggered an add-margin/close), the decision is
// SOLELY the classifier's; no AddMargin, and no MDP-originated close/continue.
fn classifier_snapshot(p_close: f64) -> ModelSnapshot {
    let json = format!(
        r#"{{"type":"random_forest_classifier","model_version":"v","threshold":0.5,
            "final_feature_order":["LongEdge_Return"],"scale_cols":[],
            "passthrough_cols":["LongEdge_Return"],"imputer":{{"medians":{{}}}},
            "scaler":{{"center":[],"scale":[]}},
            "trees":[{{"children_left":[-1],"children_right":[-1],"feature":[-2],
                      "threshold":[0.0],"value":[{p_close}]}}]}}"#
    );
    ModelSnapshot {
        classifier: Some(Arc::new(CloseClassifier::from_json_str(&json).unwrap())),
    }
}

#[test]
fn mdp_is_feature_only_never_acts() {
    let set: CacheSet =
        serde_json::from_value(serde_json::json!({ "symbols": { "BTCUSDT": cache_json() } }))
            .unwrap();
    // Single long leg with margin 5 on notional 100 -> liq price ~95; at price 100
    // with first_liq 80 the legacy MDP trigger would fire.
    let near_liq = SymbolState {
        mode: Mode::LongOnlyAfterShortLiq,
        long: Some(LegState::new(Side::Long, 100.0, 1.0, 5.0)),
        short: None,
        first_liq_price: Some(80.0),
        total_added_margin_long: 0.0,
        total_added_margin_short: 0.0,
    };

    for (p_close, expect_close) in [(0.0_f64, false), (1.0_f64, true)] {
        let input = DecisionInput {
            symbol: "BTCUSDT".into(),
            state: near_liq.clone(),
            snapshot: Arc::new(classifier_snapshot(p_close)),
            cache: Arc::new(set.clone()),
            volume: Arc::new(Default::default()),
            news: Arc::new(Default::default()),
            mctx: ctx(),
            remaining_balance: 1000.0,
            mdp_cfg: Default::default(),
            fallback_to_baseline: true,
            expected_costs: 0.0,
            inject_delay: None,
        };
        let d = parallel::compute_one(&input).decision.unwrap();
        assert!(
            !matches!(
                d.action,
                Action::AddMarginLong(_) | Action::AddMarginShort(_)
            ),
            "MDP must never produce an AddMargin action"
        );
        assert!(
            !d.record.reason.starts_with("mdp"),
            "no MDP-originated reason"
        );
        if expect_close {
            assert_eq!(d.action, Action::CloseLong);
            assert_eq!(d.record.decision, "CLOSE");
        } else {
            assert_eq!(d.action, Action::Hold);
            assert_eq!(d.record.decision, "CONTINUE");
        }
    }
}

// 8: MDP trigger false (single leg, price far from liq) -> no AddMargin intent.
#[test]
fn mdp_trigger_false_no_add_margin() {
    let set: CacheSet = serde_json::from_value(serde_json::json!({
        "symbols": { "BTCUSDT": cache_json() }
    }))
    .unwrap();
    let state = SymbolState {
        mode: Mode::LongOnlyAfterShortLiq,
        long: Some(LegState::new(Side::Long, 100.0, 1.0, 100.0)),
        short: None,
        first_liq_price: Some(10.0), // current price 100 is far from any liq -> trigger false
        total_added_margin_long: 0.0,
        total_added_margin_short: 0.0,
    };
    let input = DecisionInput {
        symbol: "BTCUSDT".into(),
        state,
        snapshot: Arc::new(ModelSnapshot::default()),
        cache: Arc::new(set),
        volume: Arc::new(Default::default()),
        news: Arc::new(Default::default()),
        mctx: ctx(),
        remaining_balance: 1000.0,
        mdp_cfg: Default::default(),
        fallback_to_baseline: true,
        expected_costs: 0.0,
        inject_delay: None,
    };
    let out = parallel::compute_one(&input);
    let d = out.decision.unwrap();
    assert!(!matches!(
        d.action,
        Action::AddMarginLong(_) | Action::AddMarginShort(_)
    ));
}

// 5: executor does not send opens/adds under kill-switch; closes still allowed.
#[test]
fn risk_blocks_new_risk_under_kill_switch() {
    let mut risk = RiskManager::new(false, true, 100.0, true, 1000.0);
    let open = OrderIntent::open("BTCUSDT", Side::Long, 1.0, "t", "open", "v", "id");
    assert!(risk.approve(&open, true).is_err());

    let state = hedged(100.0, 1.0);
    let closes = intents_from_action(
        &Action::CloseHedged,
        "BTCUSDT",
        &state,
        "t",
        "rf_close",
        "v",
        "id",
        Priority::RfClose,
    );
    assert!(risk.approve(&closes[0], true).is_ok());
}

// 14: a second close for the SAME leg is rejected; a hedged close (long+short)
// is allowed because the two legs are distinct.
#[test]
fn risk_blocks_duplicate_close() {
    let mut risk = RiskManager::new(false, true, 100.0, false, 1000.0);
    let state = hedged(100.0, 1.0);
    let closes = intents_from_action(
        &Action::CloseHedged,
        "BTCUSDT",
        &state,
        "t",
        "rf_close",
        "v",
        "id",
        Priority::RfClose,
    );
    // Both legs of a hedged close are accepted (long + short).
    assert!(risk.approve(&closes[0], true).is_ok());
    assert!(risk.approve(&closes[1], true).is_ok());
    // Re-submitting the exact same leg is rejected as a duplicate.
    assert!(risk.approve(&closes[0], true).is_err());
}

// 13/15: kill-switch close sorts ahead of MDP add-margin and RF close.
#[test]
fn priority_ordering_is_applied() {
    let mut q = vec![
        OrderIntent {
            timestamp: "t".into(),
            symbol: "B".into(),
            intent_type: OrderIntentType::CloseLong,
            position_side: Side::Long,
            quantity: Some(1.0),
            margin_amount: None,
            reduce_only: true,
            reason: "rf_close".into(),
            model_version: "v".into(),
            decision_id: "1".into(),
            priority: Priority::RfClose,
        },
        OrderIntent {
            timestamp: "t".into(),
            symbol: "A".into(),
            intent_type: OrderIntentType::AddMarginLong,
            position_side: Side::Long,
            quantity: None,
            margin_amount: Some(10.0),
            reduce_only: false,
            reason: "mdp".into(),
            model_version: "v".into(),
            decision_id: "2".into(),
            priority: Priority::MdpAddMargin,
        },
        OrderIntent {
            timestamp: "t".into(),
            symbol: "C".into(),
            intent_type: OrderIntentType::CloseLong,
            position_side: Side::Long,
            quantity: Some(1.0),
            margin_amount: None,
            reduce_only: true,
            reason: "kill".into(),
            model_version: "v".into(),
            decision_id: "3".into(),
            priority: Priority::KillSwitchClose,
        },
    ];
    q.sort_by(|a, b| a.priority.cmp(&b.priority));
    assert_eq!(q[0].priority, Priority::KillSwitchClose);
    assert_eq!(q[1].priority, Priority::MdpAddMargin);
    assert_eq!(q[2].priority, Priority::RfClose);
}

// Executor sends in real mode, stays silent in shadow mode.
#[test]
fn executor_respects_shadow_mode() {
    let ex = mock_with_prices(1000.0, &["BTCUSDT"]);
    let intent = OrderIntent::open("BTCUSDT", Side::Long, 1.0, "t", "open", "v", "id");
    let fctx = ExecContext {
        position_open: true,
        price: 100.0,
        filters: SymbolFilters::default(),
    };

    // Shadow: not sent.
    let mut risk = RiskManager::new(true, false, 100.0, false, 1000.0);
    let mut exec = OrderExecutor::new(&ex, false, 2.0);
    let out = exec.execute(&intent, &fctx, &mut risk);
    assert!(!out.sent);
    assert_eq!(out.skipped_reason.as_deref(), Some("shadow"));
    assert_eq!(ex.placed_count(), 0);

    // Real money: sent.
    let mut risk2 = RiskManager::new(false, true, 100.0, false, 1000.0);
    let mut exec2 = OrderExecutor::new(&ex, false, 2.0);
    let out2 = exec2.execute(&intent, &fctx, &mut risk2);
    assert!(out2.sent);
    assert_eq!(ex.placed_count(), 1);
}
