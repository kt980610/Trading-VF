//! Readers for the Python-produced artifacts consumed by the live engine.

pub mod close_classifier;
pub mod integral_cache;
pub mod intraday_news;
pub mod news;
pub mod portfolio_weights;
pub mod rf_model;
pub mod scaler;
pub mod volume;

pub use close_classifier::{ClassifierError, CloseClassifier};
pub use integral_cache::{CacheSet, IntegralCache};
pub use intraday_news::{IntradayNews, IntradayRecord};
pub use news::NewsFeatures;
pub use portfolio_weights::{PortfolioWeights, SymbolWeight};
pub use rf_model::{ModelRegistry, ModelSnapshot, Policy};
// `FeatureScaler` and the old regressor reader remain compiled for compatibility
// but are NOT used by the live close decision.
pub use scaler::FeatureScaler;
pub use volume::VolumeFeatures;
