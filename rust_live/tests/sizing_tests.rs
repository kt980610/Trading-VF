//! Python <-> Rust portfolio-weight sizing parity.
//!
//! The Python MVO module writes `portfolio_weights.json` plus the canonical
//! sizing targets it implies (`target_margin = total_equity * weight`,
//! `target_notional = target_margin * leverage`). Rust's `sizing::allocate_symbol`
//! must reproduce those numbers exactly. Requires the fixtures produced by
//! `python scripts/gen_rust_parity_fixtures.py` (the test fails, never skips,
//! when they are missing).

use std::path::PathBuf;

use serde::Deserialize;

use trading_live::artifacts::portfolio_weights::PortfolioWeights;
use trading_live::sizing::{allocate_symbol, SymbolFilters};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

#[derive(Debug, Deserialize)]
struct ExpectedTarget {
    symbol: String,
    weight: f64,
    leverage: f64,
    target_margin: f64,
    target_notional: f64,
}

#[derive(Debug, Deserialize)]
struct SizingMeta {
    total_equity: f64,
    expected: Vec<ExpectedTarget>,
}

#[test]
fn sizing_targets_match_python() {
    let dir = fixtures_dir();
    let weights_path = dir.join("sizing_weights.json");
    let meta_path = dir.join("sizing_parity_expected.json");

    let weights = PortfolioWeights::load(&weights_path).unwrap_or_else(|e| {
        panic!(
            "missing fixture {} ({e}); run `python scripts/gen_rust_parity_fixtures.py`",
            weights_path.display()
        )
    });
    let meta_text = std::fs::read_to_string(&meta_path).unwrap_or_else(|e| {
        panic!(
            "missing fixture {} ({e}); run `python scripts/gen_rust_parity_fixtures.py`",
            meta_path.display()
        )
    });
    let meta: SizingMeta = serde_json::from_str(&meta_text).expect("parse sizing meta");

    // The artifact must advertise the canonical sizing semantics.
    assert_eq!(weights.weight_semantics, "margin_fraction_of_total_equity");
    assert!(weights.sum_valid(1e-6));
    assert!(!meta.expected.is_empty(), "expected >=1 sizing case");

    let filters = SymbolFilters::default();
    for exp in &meta.expected {
        // The MVO weight Rust reads from the artifact must match Python's.
        let weight = weights.weight_for(&exp.symbol);
        assert!(
            (weight - exp.weight).abs() < 1e-12,
            "{} weight {weight} != {}",
            exp.symbol,
            exp.weight
        );

        let alloc = allocate_symbol(
            meta.total_equity,
            weight,
            exp.leverage,
            100.0, // price/filters are irrelevant to the target math
            &filters,
            0.0,
        );
        assert!(
            (alloc.target_margin - exp.target_margin).abs() < 1e-9,
            "{} target_margin {} != {}",
            exp.symbol,
            alloc.target_margin,
            exp.target_margin
        );
        assert!(
            (alloc.target_notional - exp.target_notional).abs() < 1e-9,
            "{} target_notional {} != {}",
            exp.symbol,
            alloc.target_notional,
            exp.target_notional
        );
    }
}
