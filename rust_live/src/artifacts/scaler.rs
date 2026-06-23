//! Per-symbol feature imputer + transform + scaler reader.
//!
//! Applies the EXACT pipeline used in Python training (spec sections 8-12):
//! impute (train median) -> transform (log1p/sqrt/none) -> scale ((t-c)/s).
//! Loaded from `feature_imputer.json` + `feature_scaler.json` in a model dir.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, Default)]
struct ImputerDoc {
    #[serde(default)]
    medians: HashMap<String, f64>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct ScalerDoc {
    #[serde(default)]
    features: Vec<String>,
    #[serde(default)]
    transforms: HashMap<String, String>,
    #[serde(default)]
    center: HashMap<String, f64>,
    #[serde(default)]
    scale: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct FeatureScaler {
    pub features: Vec<String>,
    pub medians: HashMap<String, f64>,
    pub transforms: HashMap<String, String>,
    pub center: HashMap<String, f64>,
    pub scale: HashMap<String, f64>,
}

fn apply_transform(name: &str, v: f64) -> f64 {
    match name {
        "log1p" => v.max(0.0).ln_1p(),
        "sqrt" => v.max(0.0).sqrt(),
        _ => v,
    }
}

impl FeatureScaler {
    /// Load the scaler/imputer for a model directory; `None` if either is absent.
    pub fn load_from_dir(model_dir: impl AsRef<Path>) -> Option<Self> {
        let dir = model_dir.as_ref();
        let imputer_path = dir.join("feature_imputer.json");
        let scaler_path = dir.join("feature_scaler.json");
        if !imputer_path.exists() || !scaler_path.exists() {
            return None;
        }
        let imp: ImputerDoc = serde_json::from_str(&std::fs::read_to_string(imputer_path).ok()?).ok()?;
        let sca: ScalerDoc = serde_json::from_str(&std::fs::read_to_string(scaler_path).ok()?).ok()?;
        Some(FeatureScaler {
            features: sca.features,
            medians: imp.medians,
            transforms: sca.transforms,
            center: sca.center,
            scale: sca.scale,
        })
    }

    /// Impute -> transform -> scale a single feature given the raw value (if any).
    pub fn apply(&self, name: &str, raw: Option<f64>) -> f64 {
        let median = self.medians.get(name).copied().unwrap_or(0.0);
        let v = match raw {
            Some(x) if x.is_finite() => x,
            _ => median,
        };
        let t = apply_transform(self.transforms.get(name).map(|s| s.as_str()).unwrap_or("none"), v);
        let c = self.center.get(name).copied().unwrap_or(0.0);
        let mut s = self.scale.get(name).copied().unwrap_or(1.0);
        if s == 0.0 || !s.is_finite() {
            s = 1.0;
        }
        (t - c) / s
    }

    /// Build the scaled feature vector for the given feature order.
    pub fn transform_vector(&self, order: &[String], features: &HashMap<String, f64>) -> Vec<f64> {
        order
            .iter()
            .map(|name| self.apply(name, features.get(name).copied()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scaler() -> FeatureScaler {
        let mut medians = HashMap::new();
        medians.insert("y".to_string(), 2.0);
        medians.insert("vol".to_string(), 100.0);
        let mut transforms = HashMap::new();
        transforms.insert("y".to_string(), "none".to_string());
        transforms.insert("vol".to_string(), "log1p".to_string());
        let mut center = HashMap::new();
        center.insert("y".to_string(), 0.0);
        center.insert("vol".to_string(), 0.0);
        let mut scale = HashMap::new();
        scale.insert("y".to_string(), 2.0);
        scale.insert("vol".to_string(), 1.0);
        FeatureScaler {
            features: vec!["y".into(), "vol".into()],
            medians,
            transforms,
            center,
            scale,
        }
    }

    #[test]
    fn impute_with_median() {
        let s = scaler();
        // Missing y -> median 2.0 -> (2-0)/2 = 1.0
        assert!((s.apply("y", None) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn log1p_applied() {
        let s = scaler();
        let expected = (100f64).ln_1p();
        assert!((s.apply("vol", Some(100.0)) - expected).abs() < 1e-12);
    }

    #[test]
    fn negative_clamped_before_log1p() {
        let s = scaler();
        assert!((s.apply("vol", Some(-5.0)) - 0.0).abs() < 1e-12);
    }
}
