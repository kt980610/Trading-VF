//! Per-cycle latency metrics written to `data/live_performance.jsonl`.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct CycleMetrics {
    pub cycle_timestamp: String,
    pub n_symbols: usize,
    pub decision_parallel_enabled: bool,
    pub max_symbol_workers: usize,
    pub cycle_total_ms: f64,
    pub market_data_ms: f64,
    pub artifact_lookup_ms: f64,
    pub decision_compute_ms: f64,
    pub rf_inference_ms: f64,
    pub mdp_compute_ms: f64,
    pub order_queue_ms: f64,
    pub order_execution_ms: f64,
    pub slowest_symbol: String,
    pub slowest_symbol_ms: f64,
    pub n_timeouts: usize,
    /// Severity flag: "ok" (<10s), "warning" (>=10s), "critical" (>=60s).
    pub alert: String,
}

impl CycleMetrics {
    pub fn classify(total_ms: f64) -> String {
        if total_ms >= 60_000.0 {
            "critical".into()
        } else if total_ms >= 10_000.0 {
            "warning".into()
        } else {
            "ok".into()
        }
    }
}

pub fn append(path: impl AsRef<std::path::Path>, metrics: &CycleMetrics) {
    let _ = crate::logging::append_jsonl(path, metrics);
}
