//! Live as-of intraday news: reader selection/leakage, feature splicing, the
//! effect of current news on a FIXED RF model's `p_close`, and Python<->Rust
//! parity of the artifact contents.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Deserialize;
use serde_json::json;

use trading_live::artifacts::close_classifier::CloseClassifier;
use trading_live::artifacts::intraday_news::IntradayNews;
use trading_live::artifacts::{IntegralCache, NewsFeatures};
use trading_live::clock;
use trading_live::features::{FeatureBuilder, MarketVolume};
use trading_live::halving_season::default_halvings;
use trading_live::pricing::{LegState, Side};
use trading_live::state::{Mode, SymbolState};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn tmp_file(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "tl_intraday_{}_{}_{}_{}",
        std::process::id(),
        nanos,
        id,
        name
    ));
    p
}

fn write(path: &Path, contents: &str) {
    std::fs::write(path, contents).unwrap();
}

// A fixed single-split RF classifier keyed solely on `macro_news_sentiment`:
// x <= 0.5 -> p_close 0.2 (CONTINUE), x > 0.5 -> p_close 0.9 (CLOSE).
fn news_split_classifier() -> CloseClassifier {
    let v = json!({
        "type": "random_forest_classifier",
        "model_version": "rf_news_test_v1",
        "threshold": 0.5,
        "final_feature_order": ["macro_news_sentiment"],
        "scale_cols": [],
        "passthrough_cols": ["macro_news_sentiment"],
        "imputer": {"strategy": "median", "medians": {}},
        "scaler": {"type": "robust", "center": [], "scale": []},
        "trees": [{
            "children_left": [1, -1, -1],
            "children_right": [2, -1, -1],
            "feature": [0, -2, -2],
            "threshold": [0.5, 0.0, 0.0],
            "value": [0.0, 0.2, 0.9]
        }]
    });
    CloseClassifier::from_json_str(&serde_json::to_string(&v).unwrap()).unwrap()
}

#[test]
fn latest_before_picks_recent_and_excludes_future() {
    let path = tmp_file("sel.jsonl");
    let lines = [
        json!({"asof_timestamp":"2026-06-20T10:00:00Z","symbol":"BTCUSDT","macro_news_sentiment":0.1}),
        json!({"asof_timestamp":"2026-06-20T11:00:00Z","symbol":"BTCUSDT","macro_news_sentiment":0.2}),
        json!({"asof_timestamp":"2026-06-20T12:00:00Z","symbol":"BTCUSDT","macro_news_sentiment":0.3}),
    ];
    let body: String = lines
        .iter()
        .map(|l| serde_json::to_string(l).unwrap() + "\n")
        .collect();
    write(&path, &body);

    let news = IntradayNews::load(&path).unwrap();
    let decision = clock::parse_rfc3339_secs("2026-06-20T11:30:00Z").unwrap();
    let rec = news.latest_before("BTCUSDT", decision).expect("a record");
    // Picks 11:00 (the latest <= 11:30), never the future 12:00.
    assert_eq!(rec.asof_timestamp, "2026-06-20T11:00:00Z");
    assert_eq!(rec.features.get("macro_news_sentiment").copied(), Some(0.2));
    assert_eq!(IntradayNews::age_secs(rec, decision), 30 * 60);

    // A decision before the first record -> nothing usable.
    let early = clock::parse_rfc3339_secs("2026-06-20T09:00:00Z").unwrap();
    assert!(news.latest_before("BTCUSDT", early).is_none());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn selection_applies_safety_lag() {
    let path = tmp_file("lag.jsonl");
    let lines = [
        json!({"asof_timestamp":"2026-06-20T11:50:00Z","symbol":"BTCUSDT","macro_news_sentiment":0.1}),
        json!({"asof_timestamp":"2026-06-20T11:55:00Z","symbol":"BTCUSDT","macro_news_sentiment":0.2}),
        // Inside the 5-min lag window of the 12:00 decision -> must be excluded.
        json!({"asof_timestamp":"2026-06-20T11:58:00Z","symbol":"BTCUSDT","macro_news_sentiment":0.9}),
    ];
    let body: String = lines
        .iter()
        .map(|l| serde_json::to_string(l).unwrap() + "\n")
        .collect();
    write(&path, &body);

    let news = IntradayNews::load(&path).unwrap();
    let decision = clock::parse_rfc3339_secs("2026-06-20T12:00:00Z").unwrap();
    let lag = 300; // 5 minutes
    let rec = news
        .latest_before("BTCUSDT", decision - lag)
        .expect("record");
    // Newest with asof <= 11:55:00 (= 12:00 - 5min); the 11:58 record is excluded.
    assert_eq!(rec.asof_timestamp, "2026-06-20T11:55:00Z");
    assert_eq!(rec.features.get("macro_news_sentiment").copied(), Some(0.2));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn missing_artifact_loads_empty() {
    let news = IntradayNews::load("does/not/exist.jsonl").unwrap();
    assert!(news.is_empty());
    assert!(news.latest_before("BTCUSDT", 0).is_none());
}

#[test]
fn feature_builder_splices_asof_news() {
    let mut feats: HashMap<String, f64> = HashMap::new();
    feats.insert("macro_news_sentiment".into(), 0.77);
    feats.insert("weighted_symbol_news_count".into(), 3.0);
    let news = NewsFeatures::from_asof("2024-05-01", "BTCUSDT", &feats);

    let cache: IntegralCache = serde_json::from_value(json!({
        "symbol": "BTCUSDT",
        "grid": [-0.2, -0.1, 0.0, 0.1, 0.2],
        "denom": 1.0,
        "cum_long": {"return": [0.0, 0.5, 1.0, 1.8, 3.0]},
        "cum_short": {"return": [0.0, 0.4, 0.9, 1.5, 2.2]}
    }))
    .unwrap();
    let volume = Default::default();
    let builder = FeatureBuilder {
        cache: &cache,
        volume: &volume,
        news: &news,
    };
    let state = SymbolState {
        mode: Mode::HedgedBothActive,
        long: Some(LegState::new(Side::Long, 100.0, 1.0, 100.0)),
        short: Some(LegState::new(Side::Short, 100.0, 1.0, 100.0)),
        first_liq_price: None,
        total_added_margin_long: 0.0,
        total_added_margin_short: 0.0,
    };
    let primary = state.long.clone().unwrap();
    let halvings = default_halvings();
    let mv = MarketVolume::default();
    let live = builder.build(
        "BTCUSDT",
        "2024-05-01T00:00:00Z",
        "2024-05-01",
        &state,
        &primary,
        100.0,
        1000.0,
        0.0,
        0.0,
        &mv,
        &halvings,
        0,
    );
    assert_eq!(live.get("macro_news_sentiment").copied(), Some(0.77));
    assert_eq!(live.get("weighted_symbol_news_count").copied(), Some(3.0));
}

#[test]
fn p_close_varies_with_current_news_same_model() {
    let clf = news_split_classifier();
    let mut bearish: HashMap<String, f64> = HashMap::new();
    bearish.insert("macro_news_sentiment".into(), 0.0);
    let mut bullish: HashMap<String, f64> = HashMap::new();
    bullish.insert("macro_news_sentiment".into(), 0.8);

    let p_lo = clf.p_close(&bearish).unwrap();
    let p_hi = clf.p_close(&bullish).unwrap();
    assert!((p_lo - 0.2).abs() < 1e-12);
    assert!((p_hi - 0.9).abs() < 1e-12);
    // Same fixed model, different current news -> different decision.
    assert!(p_lo < clf.threshold && p_hi >= clf.threshold);
}

// End-to-end: the SAME price/volume/MDP inputs through the full feature builder,
// changing ONLY the as-of news vector, must move a fixed RF model's p_close.
#[test]
fn news_changes_p_close_through_full_feature_vector() {
    let clf = news_split_classifier();
    let cache: IntegralCache = serde_json::from_value(json!({
        "symbol": "BTCUSDT",
        "grid": [-0.2, -0.1, 0.0, 0.1, 0.2],
        "denom": 1.0,
        "cum_long": {"return": [0.0, 0.5, 1.0, 1.8, 3.0]},
        "cum_short": {"return": [0.0, 0.4, 0.9, 1.5, 2.2]}
    }))
    .unwrap();
    let volume = Default::default();
    let state = SymbolState {
        mode: Mode::HedgedBothActive,
        long: Some(LegState::new(Side::Long, 100.0, 1.0, 100.0)),
        short: Some(LegState::new(Side::Short, 100.0, 1.0, 100.0)),
        first_liq_price: None,
        total_added_margin_long: 0.0,
        total_added_margin_short: 0.0,
    };
    let primary = state.long.clone().unwrap();
    let halvings = default_halvings();
    let mv = MarketVolume::default();

    let build = |macro_sent: f64| {
        let mut feats: HashMap<String, f64> = HashMap::new();
        feats.insert("macro_news_sentiment".into(), macro_sent);
        let news = NewsFeatures::from_asof("2024-05-01", "BTCUSDT", &feats);
        let builder = FeatureBuilder {
            cache: &cache,
            volume: &volume,
            news: &news,
        };
        builder.build(
            "BTCUSDT",
            "2024-05-01T00:00:00Z",
            "2024-05-01",
            &state,
            &primary,
            100.0,  // identical price
            1000.0, // identical balance
            0.0,
            0.0,
            &mv, // identical volume
            &halvings,
            0,
        )
    };

    let bear = build(0.0);
    let bull = build(0.8);
    // Identical everything except the single news feature.
    for (k, v) in &bear {
        if k != "macro_news_sentiment" {
            assert_eq!(
                bull.get(k).copied(),
                Some(*v),
                "non-news feature {k} changed"
            );
        }
    }
    let p_bear = clf.p_close(&bear).unwrap();
    let p_bull = clf.p_close(&bull).unwrap();
    assert!(
        (p_bear - 0.2).abs() < 1e-12 && (p_bull - 0.9).abs() < 1e-12,
        "p_close must move with news: bear={p_bear} bull={p_bull}"
    );
}

// ---- Python <-> Rust parity of the intraday artifact ---------------------

#[derive(Deserialize)]
struct IntradayParity {
    decision_timestamp: String,
    safety_lag_seconds: i64,
    asof_timestamp: String,
    symbol: String,
    news_mode: String,
    #[serde(default)]
    news_source: Option<String>,
    #[serde(default)]
    timestamp_quality: Option<String>,
    source_feature_date: Option<String>,
    features: HashMap<String, f64>,
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

// The Python builder writes both the artifact and the expected feature map; Rust
// must read identical values for the as-of lookup (REQUIRES the fixture, like the
// other parity tests).
#[test]
fn intraday_features_match_python() {
    let dir = fixtures_dir();
    let artifact = dir.join("intraday_news.jsonl");
    let expected = dir.join("intraday_news_expected.json");
    let art = std::fs::read_to_string(&artifact).unwrap_or_else(|e| {
        panic!(
            "missing fixture {} ({e}); run `python scripts/gen_rust_parity_fixtures.py`",
            artifact.display()
        )
    });
    let exp_text = std::fs::read_to_string(&expected).unwrap_or_else(|e| {
        panic!(
            "missing fixture {} ({e}); run `python scripts/gen_rust_parity_fixtures.py`",
            expected.display()
        )
    });

    let path = tmp_file("parity.jsonl");
    write(&path, &art);
    let news = IntradayNews::load(&path).unwrap();

    let cases: Vec<IntradayParity> = serde_json::from_str(&exp_text).expect("parse expected");
    assert!(!cases.is_empty(), "expected >=1 intraday parity case");
    for c in &cases {
        // Mirror the engine selection rule: asof <= decision - news_safety_lag.
        let decision = clock::parse_rfc3339_secs(&c.decision_timestamp).expect("decision ts");
        let cutoff = decision - c.safety_lag_seconds;
        let rec = news
            .latest_before(&c.symbol, cutoff)
            .unwrap_or_else(|| panic!("no record for {} @ {}", c.symbol, c.decision_timestamp));
        assert_eq!(
            rec.asof_timestamp, c.asof_timestamp,
            "selected wrong as-of for {} @ {}",
            c.symbol, c.decision_timestamp
        );
        assert_eq!(
            rec.news_mode, c.news_mode,
            "news_mode mismatch for {} @ {}",
            c.symbol, c.decision_timestamp
        );
        assert_eq!(
            rec.source_feature_date, c.source_feature_date,
            "source_feature_date mismatch for {} @ {}",
            c.symbol, c.decision_timestamp
        );
        assert_eq!(
            rec.news_source, c.news_source,
            "news_source mismatch for {} @ {}",
            c.symbol, c.decision_timestamp
        );
        assert_eq!(
            rec.timestamp_quality, c.timestamp_quality,
            "timestamp_quality mismatch for {} @ {}",
            c.symbol, c.decision_timestamp
        );
        // GDELT-sourced fixture: the as-of instant is an observation time.
        if c.news_source.as_deref() == Some("gdelt") {
            assert_eq!(rec.timestamp_quality.as_deref(), Some("observed_utc"));
        }
        for (k, v) in &c.features {
            let got = rec.features.get(k).copied().unwrap_or(f64::NAN);
            assert!(
                (got - v).abs() <= 1e-9,
                "{} feature {k}: rust {got} vs python {v}",
                c.symbol
            );
        }
    }
    let _ = std::fs::remove_file(&path);
}
