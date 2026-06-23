//! The live engine applies the SAME impute -> RobustScaler -> RF pipeline as
//! Python training, and NEVER falls back to a regressor/edge policy: a missing
//! classifier yields no policy, and a schema mismatch is surfaced as an error.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use trading_live::artifacts::close_classifier::ClassifierError;
use trading_live::artifacts::ModelRegistry;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn tmp_dir() -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!("tl_scale_{}_{}_{}", std::process::id(), nanos, id));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn write(path: &Path, contents: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, contents).unwrap();
}

// Classifier with one scaled feature `vol`: RobustScaler (center, scale), then a
// stump splitting the SCALED value at 0.5 (left leaf 0.2, right leaf 0.8).
fn write_classifier(dir: &Path, symbol: &str, scale_feature: &str, center: f64, scale: f64) {
    let v = serde_json::json!({
        "type": "random_forest_classifier",
        "model_version": "rf_close_classifier_v1",
        "threshold": 0.5,
        "final_feature_order": [scale_feature],
        "scale_cols": [scale_feature],
        "passthrough_cols": [],
        "imputer": {"strategy": "median", "medians": {"vol": 0.0}},
        "scaler": {"type": "robust", "center": [center], "scale": [scale]},
        "trees": [{
            "children_left": [1, -1, -1],
            "children_right": [2, -1, -1],
            "feature": [0, -2, -2],
            "threshold": [0.5, 0.0, 0.0],
            "value": [0.0, 0.2, 0.8]
        }]
    });
    write(
        &dir.join(symbol).join("rf_close_classifier.json"),
        &serde_json::to_string(&v).unwrap(),
    );
}

#[test]
fn scaler_is_applied_in_inference() {
    let dir = tmp_dir();
    write_classifier(&dir, "BTCUSDT", "vol", 0.0, 2.0);

    let mut reg = ModelRegistry::new(&dir);
    let policy = reg.policy_for("BTCUSDT").unwrap();
    assert!(policy.is_rf());

    // raw=1 -> scaled 0.5 -> left leaf 0.2.
    let mut feats = HashMap::new();
    feats.insert("vol".to_string(), 1.0);
    assert!((policy.p_close(&feats).unwrap() - 0.2).abs() < 1e-12);

    // Raw value 1.0 WITHOUT scaling would be > 0.5 -> 0.8, proving scaling ran.
    // raw=2 -> scaled 1.0 -> right leaf 0.8.
    feats.insert("vol".to_string(), 2.0);
    assert!((policy.p_close(&feats).unwrap() - 0.8).abs() < 1e-12);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn missing_scale_feature_uses_median() {
    let dir = tmp_dir();
    write_classifier(&dir, "BTCUSDT", "vol", 0.0, 1.0);

    let mut reg = ModelRegistry::new(&dir);
    let policy = reg.policy_for("BTCUSDT").unwrap();
    // vol absent -> median 0.0 -> scaled 0.0 <= 0.5 -> left leaf 0.2.
    let feats = HashMap::new();
    assert!((policy.p_close(&feats).unwrap() - 0.2).abs() < 1e-12);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn missing_feature_without_median_is_schema_error() {
    let dir = tmp_dir();
    // "no_median" is a scale col but has no median entry in the imputer.
    write_classifier(&dir, "BTCUSDT", "no_median", 0.0, 1.0);

    let mut reg = ModelRegistry::new(&dir);
    let policy = reg.policy_for("BTCUSDT").unwrap();
    let feats = HashMap::new();
    assert!(matches!(
        policy.p_close(&feats),
        Err(ClassifierError::FeatureSchemaMismatch(_))
    ));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn no_classifier_means_no_policy_and_no_fallback() {
    let dir = tmp_dir();
    // No model written at all.
    let mut reg = ModelRegistry::new(&dir);
    assert!(!reg.has_model("BTCUSDT"));
    // No baseline/edge fallback: a missing classifier yields no policy.
    assert!(reg.policy_for("BTCUSDT").is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn legacy_regressor_file_is_ignored() {
    let dir = tmp_dir();
    // Old artifact present, new classifier absent -> still no policy (no fallback
    // to the legacy regressor).
    let legacy = serde_json::json!({
        "type": "random_forest_regressor",
        "features": ["LongEdge_Return"],
        "threshold": 0.0,
        "trees": [{"children_left":[-1],"children_right":[-1],
                   "feature":[-2],"threshold":[0.0],"value":[1.0]}]
    });
    write(
        &dir.join("BTCUSDT").join("rf_close_decision.json"),
        &serde_json::to_string(&legacy).unwrap(),
    );
    let mut reg = ModelRegistry::new(&dir);
    assert!(!reg.has_model("BTCUSDT"));
    assert!(reg.policy_for("BTCUSDT").is_none());
    let _ = std::fs::remove_dir_all(&dir);
}
