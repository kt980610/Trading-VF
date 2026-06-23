//! Reader for the Python-produced `rf_close_classifier.json` (per symbol).
//!
//! This is the ONLY model artifact the live engine consults for the close/continue
//! decision. The file is produced by `src/rf_classifier.py::export_model_json` and
//! mirrors `predict_proba()[:,1]` exactly:
//!
//! ```text
//! raw feature map
//!   -> median imputation (scale_cols only)
//!   -> RobustScaler transform (scale_cols only): (x - center) / scale
//!   -> passthrough_cols appended raw
//!   -> averaged per-tree leaf P(close)
//! ```
//!
//! Live decision: `p_close >= threshold -> CLOSE`. There is NO regressor / edge /
//! built-in fallback: a missing or schema-mismatched artifact is a hard error.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

/// Errors surfaced when loading or evaluating the classifier. These map directly
/// to the explicit failure reasons required by the live spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClassifierError {
    /// `scaler.center` / `scaler.scale` length does not match `scale_cols`, or a
    /// feature required for scaling is missing AND has no median to impute.
    FeatureSchemaMismatch(String),
    /// Empty tree list / structurally invalid model.
    ClassifierLoadError(String),
}

impl std::fmt::Display for ClassifierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClassifierError::FeatureSchemaMismatch(m) => write!(f, "feature_schema_mismatch: {m}"),
            ClassifierError::ClassifierLoadError(m) => write!(f, "classifier_load_error: {m}"),
        }
    }
}

impl std::error::Error for ClassifierError {}

#[derive(Debug, Clone, Deserialize)]
struct TreeJson {
    children_left: Vec<i64>,
    children_right: Vec<i64>,
    feature: Vec<i64>,
    threshold: Vec<f64>,
    value: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct ImputerJson {
    #[serde(default)]
    medians: HashMap<String, f64>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct ScalerJson {
    #[serde(default)]
    center: Vec<f64>,
    #[serde(default)]
    scale: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct ClassifierJson {
    #[serde(default)]
    model_version: String,
    threshold: f64,
    #[serde(default)]
    final_feature_order: Vec<String>,
    #[serde(default)]
    scale_cols: Vec<String>,
    #[serde(default)]
    passthrough_cols: Vec<String>,
    #[serde(default)]
    imputer: ImputerJson,
    #[serde(default)]
    scaler: ScalerJson,
    #[serde(default)]
    trees: Vec<TreeJson>,
}

#[derive(Debug, Clone)]
struct Tree {
    children_left: Vec<i64>,
    children_right: Vec<i64>,
    feature: Vec<i64>,
    threshold: Vec<f64>,
    value: Vec<f64>,
}

impl Tree {
    fn predict(&self, x: &[f64]) -> f64 {
        let mut node = 0usize;
        // children_left[node] == -1 marks a leaf (sklearn convention).
        while self.children_left[node] >= 0 {
            let fidx = self.feature[node];
            let go_left = if fidx < 0 {
                true
            } else {
                x.get(fidx as usize).copied().unwrap_or(0.0) <= self.threshold[node]
            };
            node = if go_left {
                self.children_left[node] as usize
            } else {
                self.children_right[node] as usize
            };
        }
        self.value[node]
    }
}

/// A loaded per-symbol close classifier.
#[derive(Debug, Clone)]
pub struct CloseClassifier {
    pub model_version: String,
    pub threshold: f64,
    pub final_feature_order: Vec<String>,
    pub scale_cols: Vec<String>,
    pub passthrough_cols: Vec<String>,
    pub medians: HashMap<String, f64>,
    pub center: Vec<f64>,
    pub scale: Vec<f64>,
    trees: Vec<Tree>,
}

impl CloseClassifier {
    pub fn from_json_str(text: &str) -> Result<Self, ClassifierError> {
        let doc: ClassifierJson = serde_json::from_str(text)
            .map_err(|e| ClassifierError::ClassifierLoadError(e.to_string()))?;
        Self::from_doc(doc)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, ClassifierError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| ClassifierError::ClassifierLoadError(e.to_string()))?;
        Self::from_json_str(&text)
    }

    fn from_doc(doc: ClassifierJson) -> Result<Self, ClassifierError> {
        if doc.trees.is_empty() {
            return Err(ClassifierError::ClassifierLoadError(
                "empty tree list".into(),
            ));
        }
        // RobustScaler center/scale must align 1:1 with scale_cols.
        if doc.scaler.center.len() != doc.scale_cols.len()
            || doc.scaler.scale.len() != doc.scale_cols.len()
        {
            return Err(ClassifierError::FeatureSchemaMismatch(format!(
                "scaler center/scale lengths ({}/{}) != scale_cols ({})",
                doc.scaler.center.len(),
                doc.scaler.scale.len(),
                doc.scale_cols.len()
            )));
        }
        // final_feature_order must equal scale_cols + passthrough_cols.
        if !doc.final_feature_order.is_empty() {
            let expected: Vec<&String> = doc
                .scale_cols
                .iter()
                .chain(doc.passthrough_cols.iter())
                .collect();
            if expected.len() != doc.final_feature_order.len()
                || expected
                    .iter()
                    .zip(doc.final_feature_order.iter())
                    .any(|(a, b)| *a != b)
            {
                return Err(ClassifierError::FeatureSchemaMismatch(
                    "final_feature_order != scale_cols + passthrough_cols".into(),
                ));
            }
        }
        let trees = doc
            .trees
            .into_iter()
            .map(|t| Tree {
                children_left: t.children_left,
                children_right: t.children_right,
                feature: t.feature,
                threshold: t.threshold,
                value: t.value,
            })
            .collect();
        Ok(Self {
            model_version: if doc.model_version.is_empty() {
                "rf_close_classifier".into()
            } else {
                doc.model_version
            },
            threshold: doc.threshold,
            final_feature_order: doc.final_feature_order,
            scale_cols: doc.scale_cols,
            passthrough_cols: doc.passthrough_cols,
            medians: doc.imputer.medians,
            center: doc.scaler.center,
            scale: doc.scaler.scale,
            trees,
        })
    }

    /// Feature-schema version exposed for logging (uses the model version).
    pub fn feature_schema_version(&self) -> &str {
        &self.model_version
    }

    /// Build the model input vector: imputed+scaled `scale_cols`, then raw
    /// `passthrough_cols`. Mirrors `ClosePolicy._matrix`.
    fn build_vector(&self, features: &HashMap<String, f64>) -> Result<Vec<f64>, ClassifierError> {
        let mut x = Vec::with_capacity(self.scale_cols.len() + self.passthrough_cols.len());
        for (j, col) in self.scale_cols.iter().enumerate() {
            let raw = match features.get(col) {
                Some(v) if v.is_finite() => *v,
                _ => match self.medians.get(col) {
                    Some(m) => *m,
                    None => {
                        return Err(ClassifierError::FeatureSchemaMismatch(format!(
                            "missing scale feature with no median: {col}"
                        )))
                    }
                },
            };
            let center = self.center[j];
            let mut scale = self.scale[j];
            if scale == 0.0 || !scale.is_finite() {
                scale = 1.0;
            }
            x.push((raw - center) / scale);
        }
        for col in &self.passthrough_cols {
            let v = match features.get(col) {
                Some(v) if v.is_finite() => *v,
                _ => 0.0,
            };
            x.push(v);
        }
        Ok(x)
    }

    /// `predict_proba()[:,1]`: averaged per-tree leaf P(close).
    pub fn p_close(&self, features: &HashMap<String, f64>) -> Result<f64, ClassifierError> {
        let x = self.build_vector(features)?;
        let sum: f64 = self.trees.iter().map(|t| t.predict(&x)).sum();
        Ok(sum / self.trees.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(scale_cols: &str, center: &str, scale: &str, pass: &str) -> String {
        format!(
            r#"{{
              "type":"random_forest_classifier",
              "model_version":"rf_close_classifier_v1",
              "threshold":0.5,
              "final_feature_order":[{scale_cols}{sep}{pass}],
              "scale_cols":[{scale_cols}],
              "passthrough_cols":[{pass}],
              "imputer":{{"strategy":"median","medians":{{"vol":1.0}}}},
              "scaler":{{"type":"robust","center":[{center}],"scale":[{scale}]}},
              "trees":[{{"children_left":[1,-1,-1],"children_right":[2,-1,-1],
                        "feature":[0,-2,-2],"threshold":[0.0,0.0,0.0],
                        "value":[0.0,0.2,0.8]}}]
            }}"#,
            sep = if scale_cols.is_empty() || pass.is_empty() {
                ""
            } else {
                ","
            }
        )
    }

    #[test]
    fn scaled_split_chooses_leaf() {
        let c = CloseClassifier::from_json_str(&doc("\"vol\"", "0.0", "2.0", "")).unwrap();
        // x = (raw-0)/2; raw=1 -> 0.5 > 0 -> right leaf 0.8.
        let mut f = HashMap::new();
        f.insert("vol".to_string(), 1.0);
        assert!((c.p_close(&f).unwrap() - 0.8).abs() < 1e-12);
        // raw=-1 -> -0.5 <= 0 -> left leaf 0.2.
        f.insert("vol".to_string(), -1.0);
        assert!((c.p_close(&f).unwrap() - 0.2).abs() < 1e-12);
    }

    #[test]
    fn missing_scale_feature_uses_median() {
        let c = CloseClassifier::from_json_str(&doc("\"vol\"", "0.0", "1.0", "")).unwrap();
        let f = HashMap::new(); // vol missing -> median 1.0 -> 1.0 > 0 -> 0.8
        assert!((c.p_close(&f).unwrap() - 0.8).abs() < 1e-12);
    }

    #[test]
    fn missing_feature_without_median_errors() {
        // scale col "nomed" has no median entry.
        let c = CloseClassifier::from_json_str(&doc("\"nomed\"", "0.0", "1.0", "")).unwrap();
        let f = HashMap::new();
        assert!(matches!(
            c.p_close(&f),
            Err(ClassifierError::FeatureSchemaMismatch(_))
        ));
    }

    #[test]
    fn center_scale_length_mismatch_is_schema_error() {
        let bad = doc("\"vol\"", "0.0,1.0", "1.0", "");
        assert!(matches!(
            CloseClassifier::from_json_str(&bad),
            Err(ClassifierError::FeatureSchemaMismatch(_))
        ));
    }

    #[test]
    fn empty_trees_is_load_error() {
        let bad = r#"{"threshold":0.5,"scale_cols":[],"passthrough_cols":[],
            "final_feature_order":[],"imputer":{"medians":{}},
            "scaler":{"center":[],"scale":[]},"trees":[]}"#;
        assert!(matches!(
            CloseClassifier::from_json_str(bad),
            Err(ClassifierError::ClassifierLoadError(_))
        ));
    }
}
