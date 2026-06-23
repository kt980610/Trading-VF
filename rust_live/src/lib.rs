//! Live Binance Futures trade-execution engine.
//!
//! Python produces all artifacts (distribution snapshot, integral cache,
//! per-symbol RF models, predicted volume, news features, MVO/B&B portfolio
//! weights). This crate only *reads* those artifacts and executes trades. It
//! never shells out to Python in the live loop.

pub mod artifacts;
pub mod clock;
pub mod config;
pub mod decision;
pub mod edges;
pub mod exchange;
pub mod executor;
pub mod features;
pub mod halving_season;
pub mod killswitch;
pub mod logging;
pub mod mdp;
pub mod mdp_features;
pub mod order;
pub mod parallel;
pub mod perf;
pub mod pricing;
pub mod risk;
pub mod sizing;
pub mod state;

pub mod engine;

pub use pricing::Side;
