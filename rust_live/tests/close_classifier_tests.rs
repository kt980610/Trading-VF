//! Python <-> Rust parity for the live close classifier + halving season.
//!
//! Fixtures are produced by `python scripts/gen_rust_parity_fixtures.py` and live
//! in `tests/fixtures/`. These tests REQUIRE the fixtures: if they are missing the
//! test FAILS (it does not skip), so a green run proves real parity was checked.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use serde::Deserialize;
use serde_json::json;

use trading_live::artifacts::close_classifier::CloseClassifier;
use trading_live::artifacts::IntegralCache;
use trading_live::features::{FeatureBuilder, MarketVolume};
use trading_live::halving_season::{default_halvings, parse_halving_dates, select_season};
use trading_live::pricing::{LegState, Side};
use trading_live::state::{Mode, SymbolState};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

#[derive(Deserialize)]
struct ParityRow {
    features: HashMap<String, f64>,
    p_close: f64,
    decision: String,
}

#[derive(Deserialize)]
struct HalvingCase {
    symbol: String,
    timestamp: String,
    seed: u64,
    halvings: Vec<String>,
    season: String,
    cycle_id: i64,
}

/// Hard requirement: fixtures must exist. Generate them with
/// `python scripts/gen_rust_parity_fixtures.py` before running `cargo test`.
fn require(name: &str) -> String {
    let path = fixtures_dir().join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing parity fixture {} ({e}); run `python scripts/gen_rust_parity_fixtures.py`",
            path.display()
        )
    })
}

fn load_classifier() -> CloseClassifier {
    CloseClassifier::from_json_str(&require("rf_close_classifier.json")).expect("load classifier")
}

// Requirement 2: >=15 real rows; Rust p_close == Python predict_proba[:,1] within
// 1e-9, and the threshold decision matches exactly.
#[test]
fn p_close_matches_python_predict_proba() {
    let clf = load_classifier();
    let rows: Vec<ParityRow> =
        serde_json::from_str(&require("parity_rows.json")).expect("parse rows");
    assert!(
        rows.len() >= 15,
        "expected >=15 parity rows, got {}",
        rows.len()
    );

    let mut max_abs_diff = 0.0_f64;
    for (i, row) in rows.iter().enumerate() {
        let p = clf.p_close(&row.features).expect("p_close");
        let d = (p - row.p_close).abs();
        max_abs_diff = max_abs_diff.max(d);
        assert!(
            d <= 1e-9,
            "row {i}: rust p_close {p} vs python {} (|d|={d})",
            row.p_close
        );
        let decision = if p >= clf.threshold {
            "CLOSE"
        } else {
            "CONTINUE"
        };
        assert_eq!(
            decision,
            row.decision.as_str(),
            "row {i}: decision mismatch"
        );
    }
    println!("parity rows={} max_abs_diff={max_abs_diff:.3e}", rows.len());
}

// Requirement 3: scale_cols + passthrough_cols == final_feature_order, in order,
// complete, and with no duplicates.
#[test]
fn feature_order_is_exact_and_unique() {
    let clf = load_classifier();
    let union: Vec<String> = clf
        .scale_cols
        .iter()
        .chain(clf.passthrough_cols.iter())
        .cloned()
        .collect();
    assert_eq!(
        union, clf.final_feature_order,
        "scale_cols + passthrough_cols must equal final_feature_order (same order)"
    );
    let set: HashSet<&String> = union.iter().collect();
    assert_eq!(set.len(), union.len(), "duplicate feature names in schema");
    assert!(!union.is_empty(), "empty feature schema");
}

// Requirement 4: every feature in the training schema is produced by the live
// runtime feature map, OR is a scale_col that is allowed to be median-imputed.
// Passthrough features must be present (they cannot be median-imputed).
#[test]
fn runtime_feature_map_covers_training_schema() {
    let clf = load_classifier();

    let cache: IntegralCache = serde_json::from_value(json!({
        "symbol": "BTCUSDT",
        "grid": [-0.2, -0.1, 0.0, 0.1, 0.2],
        "denom": 1.0,
        "cum_long": {"return": [0.0, 0.5, 1.0, 1.8, 3.0]},
        "cum_short": {"return": [0.0, 0.4, 0.9, 1.5, 2.2]}
    }))
    .unwrap();
    let volume = Default::default();
    let news = Default::default();
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

    let scale: HashSet<&String> = clf.scale_cols.iter().collect();
    for col in &clf.final_feature_order {
        if live.contains_key(col) {
            continue;
        }
        // Absent is only acceptable for a scale col that has a median to impute.
        assert!(
            scale.contains(col) && clf.medians.contains_key(col),
            "training feature '{col}' is neither produced by the live feature map \
             nor median-imputable (passthrough/categorical features must be present)"
        );
    }
}

// Requirement 9: halving overlap selection (incl. equidistant SHA-256 tie)
// matches Python exactly.
#[test]
fn halving_season_matches_python() {
    let cases: Vec<HalvingCase> =
        serde_json::from_str(&require("halving_cases.json")).expect("parse halving cases");
    assert!(cases.len() >= 5, "expected several halving cases");

    let mut saw_tie = false;
    for c in &cases {
        let halvings = parse_halving_dates(&c.halvings);
        let (label, cid) = select_season(&c.symbol, &c.timestamp, &halvings, c.seed);
        assert_eq!(
            label,
            c.season.as_str(),
            "season mismatch for {} @ {}",
            c.symbol,
            c.timestamp
        );
        assert_eq!(
            cid, c.cycle_id,
            "cycle_id mismatch for {} @ {}",
            c.symbol, c.timestamp
        );
        if c.halvings.len() == 2 {
            saw_tie = true;
        }
    }
    assert!(saw_tie, "fixtures must include an equidistant tie case");
}
