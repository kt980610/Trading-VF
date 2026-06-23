//! Live engine configuration (YAML) and environment loading.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

fn default_true() -> bool {
    true
}
fn default_min_balance() -> f64 {
    100.0
}
fn default_leverage() -> f64 {
    1.0
}
fn default_add_margin_step() -> f64 {
    10.0
}
fn default_big() -> f64 {
    1e18
}
fn default_open_time() -> String {
    "00:00".to_string()
}
fn default_trade_mode() -> String {
    "day_start_only".to_string()
}
fn default_max_symbol_workers() -> usize {
    8
}
fn default_decision_timeout_ms() -> u64 {
    5000
}
fn default_order_executor_mode() -> String {
    "single_queue".to_string()
}
fn default_max_orders_per_second() -> f64 {
    2.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct LiveSettings {
    #[serde(default)]
    pub shadow_mode: bool,
    #[serde(default)]
    pub real_money: bool,
    #[serde(default = "default_true")]
    pub require_rf_for_new_trade: bool,
    #[serde(default = "default_true")]
    pub fallback_to_baseline_if_rf_missing: bool,
}

impl Default for LiveSettings {
    fn default() -> Self {
        Self {
            shadow_mode: true,
            real_money: false,
            require_rf_for_new_trade: true,
            fallback_to_baseline_if_rf_missing: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RiskSettings {
    #[serde(default = "default_min_balance")]
    pub min_total_wallet_balance_usdt: f64,
    #[serde(default = "default_true")]
    pub kill_switch_close_positions: bool,
    #[serde(default = "default_true")]
    pub kill_switch_cancel_orders: bool,
    #[serde(default = "default_true")]
    pub kill_switch_stop_trading: bool,
}

impl Default for RiskSettings {
    fn default() -> Self {
        Self {
            min_total_wallet_balance_usdt: 100.0,
            kill_switch_close_positions: true,
            kill_switch_cancel_orders: true,
            kill_switch_stop_trading: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TradeOpening {
    #[serde(default = "default_trade_mode")]
    pub mode: String,
    #[serde(default = "default_open_time")]
    pub open_time_utc: String,
}

impl Default for TradeOpening {
    fn default() -> Self {
        Self {
            mode: default_trade_mode(),
            open_time_utc: default_open_time(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct MdpSettings {
    #[serde(default = "default_add_margin_step")]
    pub add_margin_step: f64,
    #[serde(default = "default_big")]
    pub max_add_margin_per_decision: f64,
    #[serde(default = "default_big")]
    pub max_total_added_margin: f64,
}

impl Default for MdpSettings {
    fn default() -> Self {
        Self {
            add_margin_step: 10.0,
            max_add_margin_per_decision: 1e18,
            max_total_added_margin: 1e18,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct LiveParallelSettings {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_max_symbol_workers")]
    pub max_symbol_workers: usize,
    #[serde(default = "default_decision_timeout_ms")]
    pub decision_timeout_ms: u64,
    #[serde(default = "default_order_executor_mode")]
    pub order_executor_mode: String,
    #[serde(default = "default_true")]
    pub order_rate_limit_enabled: bool,
    #[serde(default = "default_max_orders_per_second")]
    pub max_orders_per_second: f64,
    #[serde(default = "default_true")]
    pub cancel_before_close: bool,
}

impl Default for LiveParallelSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            max_symbol_workers: default_max_symbol_workers(),
            decision_timeout_ms: default_decision_timeout_ms(),
            order_executor_mode: default_order_executor_mode(),
            order_rate_limit_enabled: true,
            max_orders_per_second: default_max_orders_per_second(),
            cancel_before_close: true,
        }
    }
}

/// Bitcoin halving anchors + the global seed used for the deterministic
/// season tie-break. Must match the Python `config.halving` values so the
/// live one-hot season features are identical to training.
#[derive(Debug, Clone, Deserialize)]
pub struct HalvingSettings {
    /// UTC halving timestamps (ISO-8601). Empty -> use the canonical defaults.
    #[serde(default)]
    pub dates: Vec<String>,
    #[serde(default)]
    pub season_seed: u64,
}

impl Default for HalvingSettings {
    fn default() -> Self {
        Self {
            dates: Vec::new(),
            season_seed: 0,
        }
    }
}

impl HalvingSettings {
    /// Parsed + sorted halving epoch seconds (canonical defaults when unset).
    pub fn epoch_secs(&self) -> Vec<i64> {
        if self.dates.is_empty() {
            crate::halving_season::default_halvings()
        } else {
            crate::halving_season::parse_halving_dates(&self.dates)
        }
    }
}

fn default_max_news_age_minutes() -> i64 {
    30
}
fn default_news_safety_lag_seconds() -> i64 {
    300
}

/// Live as-of news settings (mirror of the Python `news` block; only the fields
/// the live engine needs are deserialized, extra YAML keys are ignored).
#[derive(Debug, Clone, Deserialize)]
pub struct NewsSettings {
    /// When false (default) the engine ignores the intraday artifact entirely and
    /// behaves exactly as before (daily news, no freshness gate). Production
    /// enables this so RF inference uses up-to-date news.
    #[serde(default)]
    pub intraday_enabled: bool,
    /// A symbol may not OPEN a new position when the freshest selectable as-of
    /// record is older than this many minutes (existing positions are still
    /// managed normally).
    #[serde(default = "default_max_news_age_minutes")]
    pub max_news_feature_age_minutes: i64,
    /// Leakage guard applied at SELECTION time: the engine only ever uses a
    /// record whose `asof_timestamp <= decision_timestamp - news_safety_lag_seconds`.
    /// Must match the Python `news.news_safety_lag_seconds` so live == training.
    #[serde(default = "default_news_safety_lag_seconds")]
    pub news_safety_lag_seconds: i64,
    /// Expected intraday-artifact `feature_version`. When non-empty the engine
    /// refuses to OPEN on a record whose `feature_version` differs (schema
    /// mismatch fail-safe). Empty (default) disables the check for back-compat.
    #[serde(default)]
    pub expected_feature_version: String,
}

impl Default for NewsSettings {
    fn default() -> Self {
        Self {
            intraday_enabled: false,
            max_news_feature_age_minutes: default_max_news_age_minutes(),
            news_safety_lag_seconds: default_news_safety_lag_seconds(),
            expected_feature_version: String::new(),
        }
    }
}

fn default_max_portfolio_age_days() -> i64 {
    2
}
fn default_weight_sum_tolerance() -> f64 {
    1e-6
}

/// Canonical MVO portfolio-weight sizing settings. There is intentionally no
/// fixed deploy-capital or net-PnL knob: live sizing is always
/// `total_equity * mvo_weight` (see `crate::sizing`).
#[derive(Debug, Clone, Deserialize)]
pub struct SizingSettings {
    /// Per-order notional hard cap (USDT). `0` (default) means no cap. Capital a
    /// cap prevents from being deployed stays as cash; it is never reallocated.
    #[serde(default)]
    pub max_order_notional_usdt: f64,
    /// A portfolio_weights artifact older than this (relative to its `as_of_date`)
    /// is treated as stale and blocks opening new positions.
    #[serde(default = "default_max_portfolio_age_days")]
    pub max_portfolio_age_days: i64,
    /// Tolerance for the `sum(weights) + cash_weight == 1.0` invariant.
    #[serde(default = "default_weight_sum_tolerance")]
    pub weight_sum_tolerance: f64,
    /// Expected `portfolio_version`. Empty (default) accepts any version; when set
    /// the artifact must match exactly or new opens are blocked.
    #[serde(default)]
    pub portfolio_version: String,
}

impl Default for SizingSettings {
    fn default() -> Self {
        Self {
            max_order_notional_usdt: 0.0,
            max_portfolio_age_days: default_max_portfolio_age_days(),
            weight_sum_tolerance: default_weight_sum_tolerance(),
            portfolio_version: String::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExchangeSettings {
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    pub leverage: HashMap<String, f64>,
    #[serde(default = "default_leverage")]
    pub default_leverage: f64,
}

impl Default for ExchangeSettings {
    fn default() -> Self {
        Self {
            base_url: "https://fapi.binance.com".to_string(),
            leverage: HashMap::new(),
            default_leverage: 1.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Paths {
    #[serde(default = "p_distribution")]
    pub distribution_snapshot: String,
    #[serde(default = "p_cache")]
    pub integral_cache: String,
    #[serde(default = "p_weights")]
    pub portfolio_weights: String,
    #[serde(default = "p_volume")]
    pub predicted_daily_volume: String,
    #[serde(default = "p_news")]
    pub news_features_daily: String,
    #[serde(default = "p_news_intraday")]
    pub news_features_intraday: String,
    #[serde(default = "p_corr")]
    pub correlation_matrix_daily: String,
    #[serde(default = "p_models")]
    pub models_promoted: String,
    #[serde(default = "p_state")]
    pub live_position_state: String,
    #[serde(default = "p_decisions")]
    pub live_decisions: String,
    #[serde(default = "p_realized")]
    pub realized_symbol_returns: String,
    #[serde(default = "p_kill")]
    pub kill_switch_triggered: String,
    #[serde(default = "p_perf")]
    pub live_performance: String,
}

fn p_distribution() -> String {
    "data/distribution_snapshot.json".into()
}
fn p_cache() -> String {
    "data/integral_cache.json".into()
}
fn p_weights() -> String {
    "data/portfolio_weights.json".into()
}
fn p_volume() -> String {
    "data/predicted_daily_volume.jsonl".into()
}
fn p_news() -> String {
    "data/news_features_daily.jsonl".into()
}
fn p_news_intraday() -> String {
    "data/news_features_intraday.jsonl".into()
}
fn p_corr() -> String {
    "data/correlation_matrix_daily.jsonl".into()
}
fn p_models() -> String {
    "models/promoted".into()
}
fn p_state() -> String {
    "data/live_position_state.json".into()
}
fn p_decisions() -> String {
    "data/live_decisions.jsonl".into()
}
fn p_realized() -> String {
    "data/realized_symbol_returns.jsonl".into()
}
fn p_kill() -> String {
    "data/kill_switch_triggered.json".into()
}
fn p_perf() -> String {
    "data/live_performance.jsonl".into()
}

impl Default for Paths {
    fn default() -> Self {
        Self {
            distribution_snapshot: p_distribution(),
            integral_cache: p_cache(),
            portfolio_weights: p_weights(),
            predicted_daily_volume: p_volume(),
            news_features_daily: p_news(),
            news_features_intraday: p_news_intraday(),
            correlation_matrix_daily: p_corr(),
            models_promoted: p_models(),
            live_position_state: p_state(),
            live_decisions: p_decisions(),
            realized_symbol_returns: p_realized(),
            kill_switch_triggered: p_kill(),
            live_performance: p_perf(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub live: LiveSettings,
    #[serde(default)]
    pub risk: RiskSettings,
    #[serde(default)]
    pub trade_opening: TradeOpening,
    #[serde(default)]
    pub mdp: MdpSettings,
    #[serde(default)]
    pub halving: HalvingSettings,
    #[serde(default)]
    pub live_parallel: LiveParallelSettings,
    #[serde(default)]
    pub news: NewsSettings,
    #[serde(default)]
    pub sizing: SizingSettings,
    #[serde(default)]
    pub exchange: ExchangeSettings,
    #[serde(default)]
    pub symbols: Vec<String>,
    #[serde(default)]
    pub paths: Paths,
}

impl Config {
    pub fn from_yaml_str(text: &str) -> anyhow::Result<Self> {
        let cfg: Config = serde_yaml::from_str(text)?;
        Ok(cfg)
    }

    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_yaml_str(&text)
    }

    pub fn leverage_for(&self, symbol: &str) -> f64 {
        *self
            .exchange
            .leverage
            .get(symbol)
            .unwrap_or(&self.exchange.default_leverage)
    }
}

/// Binance credentials read from the environment.
#[derive(Debug, Clone)]
pub struct ApiCredentials {
    pub api_key: String,
    pub api_secret: String,
    pub base_url: String,
}

impl ApiCredentials {
    pub fn from_env(default_base_url: &str) -> anyhow::Result<Self> {
        let api_key = std::env::var("BINANCE_API_KEY")
            .map_err(|_| anyhow::anyhow!("BINANCE_API_KEY not set"))?;
        let api_secret = std::env::var("BINANCE_API_SECRET")
            .map_err(|_| anyhow::anyhow!("BINANCE_API_SECRET not set"))?;
        let base_url = std::env::var("BINANCE_FUTURES_BASE_URL")
            .unwrap_or_else(|_| default_base_url.to_string());
        Ok(Self {
            api_key,
            api_secret,
            base_url,
        })
    }
}
