//! Cumulative integral cache reader + interval lookup (mirrors the Python
//! `integral_cache.json`). The live engine never recomputes integrals; it only
//! does `cum(b) - cum(a)` lookups, so moving `y` or a liquidation cutoff only
//! shifts the interval bounds.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct IntegralCache {
    pub symbol: String,
    pub grid: Vec<f64>,
    pub denom: f64,
    pub cum_long: HashMap<String, Vec<f64>>,
    pub cum_short: HashMap<String, Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CacheSet {
    pub symbols: HashMap<String, IntegralCache>,
}

fn interp(grid: &[f64], ys: &[f64], x: f64) -> f64 {
    if grid.is_empty() || ys.is_empty() {
        return 0.0;
    }
    let lo = grid[0];
    let hi = grid[grid.len() - 1];
    let xc = x.clamp(lo, hi);

    // Binary search for the interval [grid[i], grid[i+1]] containing xc.
    let mut left = 0usize;
    let mut right = grid.len() - 1;
    if xc <= lo {
        return ys[0];
    }
    if xc >= hi {
        return ys[ys.len() - 1];
    }
    while right - left > 1 {
        let mid = (left + right) / 2;
        if grid[mid] <= xc {
            left = mid;
        } else {
            right = mid;
        }
    }
    let x0 = grid[left];
    let x1 = grid[right];
    let y0 = ys[left];
    let y1 = ys[right];
    if (x1 - x0).abs() < f64::EPSILON {
        return y0;
    }
    y0 + (y1 - y0) * (xc - x0) / (x1 - x0)
}

impl IntegralCache {
    pub fn z_min(&self) -> f64 {
        *self.grid.first().unwrap_or(&0.0)
    }
    pub fn z_max(&self) -> f64 {
        *self.grid.last().unwrap_or(&0.0)
    }

    fn lookup(&self, cum: &HashMap<String, Vec<f64>>, component: &str, a: f64, b: f64) -> f64 {
        match cum.get(component) {
            Some(arr) => interp(&self.grid, arr, b) - interp(&self.grid, arr, a),
            None => 0.0,
        }
    }

    pub fn integral_long(&self, component: &str, a: f64, b: f64) -> f64 {
        self.lookup(&self.cum_long, component, a, b)
    }

    pub fn integral_short(&self, component: &str, a: f64, b: f64) -> f64 {
        self.lookup(&self.cum_short, component, a, b)
    }
}

impl CacheSet {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let set: CacheSet = serde_json::from_str(&text)?;
        Ok(set)
    }

    pub fn get(&self, symbol: &str) -> Option<&IntegralCache> {
        self.symbols.get(symbol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> IntegralCache {
        // cum_long["return"] is the cumulative integral; differences give intervals.
        let grid = vec![-0.1, 0.0, 0.1];
        let mut cum_long = HashMap::new();
        cum_long.insert("return".to_string(), vec![0.0, 1.0, 3.0]);
        IntegralCache {
            symbol: "X".into(),
            grid,
            denom: 1.0,
            cum_long,
            cum_short: HashMap::new(),
        }
    }

    #[test]
    fn interval_is_difference() {
        let c = sample();
        assert!((c.integral_long("return", 0.0, 0.1) - 2.0).abs() < 1e-9);
        assert!((c.integral_long("return", -0.1, 0.1) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn out_of_grid_clamped() {
        let c = sample();
        let far = c.integral_long("return", -5.0, 5.0);
        let inside = c.integral_long("return", -0.1, 0.1);
        assert!((far - inside).abs() < 1e-9);
    }

    #[test]
    fn missing_component_zero() {
        let c = sample();
        assert_eq!(c.integral_long("mean", 0.0, 0.1), 0.0);
    }
}
