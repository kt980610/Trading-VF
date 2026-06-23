//! Parallel decision scheduler.
//!
//! Each active symbol's decision (feature build, integral cache lookup, RF
//! inference, MDP trigger/return-only value, close/continue/add-margin) is pure
//! and independent, so we run them across an OS thread pool bounded by a counting
//! semaphore. Workers only PRODUCE decisions; order submission stays centralized.
//!
//! Tokio is intentionally not used: the equivalent guarantees (bounded
//! concurrency, per-symbol timeout with a safe fallback, deterministic results)
//! are achieved with `std::thread` + channels, which also keeps the build free of
//! the platform-specific async toolchain.

use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::artifacts::{CacheSet, ModelSnapshot, NewsFeatures, VolumeFeatures};
use crate::config::MdpSettings;
use crate::decision::{decide_symbol_snapshot, DecisionTiming, MarketContext, SymbolDecision};
use crate::state::SymbolState;

/// Everything a worker needs to decide one symbol. All fields are owned or
/// `Arc`-shared so the task is `'static` + `Send` and can outlive the cycle if a
/// stuck worker is abandoned past the timeout.
pub struct DecisionInput {
    pub symbol: String,
    pub state: SymbolState,
    pub snapshot: Arc<ModelSnapshot>,
    pub cache: Arc<CacheSet>,
    pub volume: Arc<VolumeFeatures>,
    pub news: Arc<NewsFeatures>,
    pub mctx: MarketContext,
    pub remaining_balance: f64,
    pub mdp_cfg: MdpSettings,
    pub fallback_to_baseline: bool,
    pub expected_costs: f64,
    /// Test-only hook to simulate a slow symbol (validates timeout fallback).
    pub inject_delay: Option<Duration>,
}

#[derive(Debug, Clone)]
pub struct DecisionOutput {
    pub symbol: String,
    pub decision: Option<SymbolDecision>,
    pub timing: DecisionTiming,
    pub compute_ms: f64,
    pub timed_out: bool,
}

impl DecisionOutput {
    fn fallback(symbol: &str, timed_out: bool) -> Self {
        DecisionOutput {
            symbol: symbol.to_string(),
            decision: None,
            timing: DecisionTiming::default(),
            compute_ms: 0.0,
            timed_out,
        }
    }
}

/// Pure per-symbol computation. No exchange, no registry, no shared mutation.
pub fn compute_one(input: &DecisionInput) -> DecisionOutput {
    if let Some(d) = input.inject_delay {
        std::thread::sleep(d);
    }
    let start = Instant::now();
    let cache = match input.cache.get(&input.symbol) {
        Some(c) => c,
        None => return DecisionOutput::fallback(&input.symbol, false),
    };
    let res = decide_symbol_snapshot(
        &input.symbol,
        &input.state,
        &input.snapshot,
        cache,
        &input.volume,
        &input.news,
        &input.mctx,
        input.remaining_balance,
        &input.mdp_cfg,
        input.fallback_to_baseline,
        input.expected_costs,
    );
    let compute_ms = start.elapsed().as_secs_f64() * 1000.0;
    match res {
        Some((decision, timing)) => DecisionOutput {
            symbol: input.symbol.clone(),
            decision: Some(decision),
            timing,
            compute_ms,
            timed_out: false,
        },
        None => DecisionOutput {
            symbol: input.symbol.clone(),
            decision: None,
            timing: DecisionTiming::default(),
            compute_ms,
            timed_out: false,
        },
    }
}

/// Run all decisions sequentially (parity baseline / parallelism disabled).
pub fn run_sequential(inputs: Vec<DecisionInput>) -> Vec<DecisionOutput> {
    inputs.iter().map(compute_one).collect()
}

/// A minimal counting semaphore (no extra crates / async runtime).
struct Semaphore {
    state: Mutex<usize>,
    cv: Condvar,
}

impl Semaphore {
    fn new(permits: usize) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(permits.max(1)),
            cv: Condvar::new(),
        })
    }
    fn acquire(&self) {
        let mut n = self.state.lock().unwrap();
        while *n == 0 {
            n = self.cv.wait(n).unwrap();
        }
        *n -= 1;
    }
    fn release(&self) {
        let mut n = self.state.lock().unwrap();
        *n += 1;
        self.cv.notify_one();
    }
}

/// Run decisions in parallel, bounded by `max_workers`, with a per-cycle
/// collection deadline of `timeout_ms`. Symbols that do not report by the
/// deadline get a safe `timed_out` fallback (the engine treats this as "hold").
pub fn run_parallel(inputs: Vec<DecisionInput>, max_workers: usize, timeout_ms: u64) -> Vec<DecisionOutput> {
    let n = inputs.len();
    if n == 0 {
        return Vec::new();
    }
    let symbols: Vec<String> = inputs.iter().map(|i| i.symbol.clone()).collect();
    let sem = Semaphore::new(max_workers);
    let (tx, rx) = mpsc::channel::<DecisionOutput>();

    for input in inputs {
        let tx = tx.clone();
        let sem = sem.clone();
        std::thread::spawn(move || {
            sem.acquire();
            let out = compute_one(&input);
            sem.release();
            let _ = tx.send(out);
        });
    }
    drop(tx);

    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    let mut collected: HashMap<String, DecisionOutput> = HashMap::new();
    while collected.len() < n {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        match rx.recv_timeout(deadline - now) {
            Ok(out) => {
                collected.insert(out.symbol.clone(), out);
            }
            Err(_) => break,
        }
    }

    // Preserve input order; missing symbols => timeout fallback. Stuck worker
    // threads are deliberately not joined so one slow symbol cannot stall the cycle.
    symbols
        .into_iter()
        .map(|s| collected.remove(&s).unwrap_or_else(|| DecisionOutput::fallback(&s, true)))
        .collect()
}
