//! Per-symbol model registry for the live close decision.
//!
//! The ONLY artifact consulted is `models/promoted/<SYMBOL>/rf_close_classifier.json`
//! (a binary close classifier producing `p_close`). Each symbol uses its OWN
//! classifier; there is NO cross-symbol fallback, NO regressor fallback and NO
//! baseline edge fallback. A missing or schema-mismatched artifact is surfaced as
//! an explicit error to the caller, which then holds (never silently closes,
//! opens or reverts to a legacy policy).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::close_classifier::{ClassifierError, CloseClassifier};

/// File name of the per-symbol classifier artifact.
pub const CLASSIFIER_FILE: &str = "rf_close_classifier.json";

/// The active close policy for a symbol: a loaded RF binary close classifier.
#[derive(Debug, Clone)]
pub struct Policy {
    pub classifier: Arc<CloseClassifier>,
}

impl Policy {
    pub fn new(classifier: Arc<CloseClassifier>) -> Self {
        Self { classifier }
    }

    /// `p_close = predict_proba()[:,1]`.
    pub fn p_close(&self, features: &HashMap<String, f64>) -> Result<f64, ClassifierError> {
        self.classifier.p_close(features)
    }

    pub fn threshold(&self) -> f64 {
        self.classifier.threshold
    }

    pub fn version(&self) -> &str {
        &self.classifier.model_version
    }

    pub fn feature_schema_version(&self) -> &str {
        self.classifier.feature_schema_version()
    }

    /// Always true now: a Policy only ever wraps a loaded RF classifier.
    pub fn is_rf(&self) -> bool {
        true
    }
}

/// An immutable per-symbol snapshot captured at cycle start.
///
/// Decision workers resolve their policy from this snapshot WITHOUT touching the
/// registry, so a model hot-reload mid-cycle never changes the schema a worker
/// already started with. `classifier == None` means the symbol has no usable
/// classifier (missing / load error) -> the worker holds with an explicit reason.
#[derive(Debug, Clone, Default)]
pub struct ModelSnapshot {
    pub classifier: Option<Arc<CloseClassifier>>,
}

impl ModelSnapshot {
    pub fn policy(&self) -> Option<Policy> {
        self.classifier.clone().map(Policy::new)
    }

    pub fn has_classifier(&self) -> bool {
        self.classifier.is_some()
    }
}

/// Loads and caches one close classifier per symbol from
/// `models/promoted/<SYMBOL>/rf_close_classifier.json`.
#[derive(Debug, Default)]
pub struct ModelRegistry {
    models_dir: PathBuf,
    cache: HashMap<String, Option<Arc<CloseClassifier>>>,
}

impl ModelRegistry {
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        Self {
            models_dir: models_dir.into(),
            cache: HashMap::new(),
        }
    }

    fn classifier_path(&self, symbol: &str) -> PathBuf {
        self.models_dir.join(symbol).join(CLASSIFIER_FILE)
    }

    pub fn load_for_symbol(&mut self, symbol: &str) -> Option<Arc<CloseClassifier>> {
        if !self.cache.contains_key(symbol) {
            let path = self.classifier_path(symbol);
            let loaded = if path.exists() {
                CloseClassifier::load(&path).ok().map(Arc::new)
            } else {
                None
            };
            self.cache.insert(symbol.to_string(), loaded);
        }
        self.cache.get(symbol).and_then(|m| m.clone())
    }

    pub fn has_model(&mut self, symbol: &str) -> bool {
        self.load_for_symbol(symbol).is_some()
    }

    /// Capture an immutable classifier snapshot for a symbol.
    pub fn snapshot(&mut self, symbol: &str) -> ModelSnapshot {
        ModelSnapshot {
            classifier: self.load_for_symbol(symbol),
        }
    }

    /// Resolve the active close policy for a symbol. Returns `None` when the
    /// symbol has no usable classifier (the caller then holds explicitly).
    pub fn policy_for(&mut self, symbol: &str) -> Option<Policy> {
        self.load_for_symbol(symbol).map(Policy::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL: &str = r#"{
        "type":"random_forest_classifier",
        "model_version":"rf_close_classifier_v1",
        "threshold":0.5,
        "final_feature_order":["LongEdge_Return"],
        "scale_cols":[],
        "passthrough_cols":["LongEdge_Return"],
        "imputer":{"medians":{}},
        "scaler":{"center":[],"scale":[]},
        "trees":[{"children_left":[-1],"children_right":[-1],"feature":[-2],
                  "threshold":[0.0],"value":[0.8]}]
    }"#;

    fn write_model(dir: &Path, symbol: &str) {
        let d = dir.join(symbol);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join(CLASSIFIER_FILE), MODEL).unwrap();
    }

    fn tmp() -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let mut p = std::env::temp_dir();
        p.push(format!("tl_reg_{}_{}", std::process::id(), nanos));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn resolves_classifier_when_present() {
        let dir = tmp();
        write_model(&dir, "BTCUSDT");
        let mut reg = ModelRegistry::new(&dir);
        assert!(reg.has_model("BTCUSDT"));
        let p = reg.policy_for("BTCUSDT").unwrap();
        assert_eq!(p.threshold(), 0.5);
        let mut f = HashMap::new();
        f.insert("LongEdge_Return".to_string(), 0.0);
        assert!((p.p_close(&f).unwrap() - 0.8).abs() < 1e-12);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_classifier_no_policy() {
        let dir = tmp();
        let mut reg = ModelRegistry::new(&dir);
        assert!(!reg.has_model("ETHUSDT"));
        assert!(reg.policy_for("ETHUSDT").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
