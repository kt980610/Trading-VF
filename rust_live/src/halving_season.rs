//! Bitcoin halving-season one-hot features with deterministic tie-breaking.
//!
//! Exact parity with the Python `src/halving_season.py`:
//!
//! * `season_pre_halving_2y + season_post_halving_2y + season_unknown == 1`.
//! * A timestamp can fall into BOTH the previous halving's post-window and the
//!   next halving's pre-window; the closest halving wins.
//! * Exact distance ties resolve via a reproducible SHA-256 keyed on
//!   `symbol|timestamp_iso|season_seed` (NEVER process-salted hashing), so the
//!   same tuple resolves to the same season across train/val/test/live.
//!
//! The candidate-window arithmetic is done on integer epoch seconds (robust to
//! `Z` vs `+00:00`). The SHA tie-break uses the timestamp string verbatim, so the
//! caller must pass the SAME ISO string Python used (its `Timestamp.isoformat()`).
//! Minute bars are whole-second UTC, for which RFC3339 == pandas `isoformat()`.

use sha2::{Digest, Sha256};

pub const WINDOW_DAYS: i64 = 730; // 2 years
pub const WINDOW_SECS: i64 = WINDOW_DAYS * 86_400;

pub const SEASON_PRE: &str = "season_pre_halving_2y";
pub const SEASON_POST: &str = "season_post_halving_2y";
pub const SEASON_UNKNOWN: &str = "season_unknown";
pub const SEASON_ONE_HOT: &[&str] = &[SEASON_PRE, SEASON_POST, SEASON_UNKNOWN];

/// Canonical Bitcoin halving timestamps (UTC). Override via config.halving.dates.
pub const DEFAULT_HALVING_DATES_UTC: &[&str] = &[
    "2012-11-28T00:00:00Z",
    "2016-07-09T00:00:00Z",
    "2020-05-11T00:00:00Z",
    "2024-04-20T00:00:00Z",
    "2028-04-20T00:00:00Z",
];

/// Days from the civil (proleptic Gregorian) date to the Unix epoch.
/// Howard Hinnant's algorithm; valid for the full range we care about.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = (y - era * 400) as i64; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
}

/// Parse the leading `YYYY-MM-DDtHH:MM:SS` (separator `T` or space) of an ISO
/// timestamp into Unix epoch seconds. Any trailing timezone suffix is ignored
/// (all artifacts are UTC). Returns `None` on a malformed prefix.
pub fn parse_epoch_secs(iso: &str) -> Option<i64> {
    let b = iso.as_bytes();
    if b.len() < 19 {
        return None;
    }
    let num = |s: &str| -> Option<i64> { s.parse::<i64>().ok() };
    let year = num(iso.get(0..4)?)?;
    if b[4] != b'-' {
        return None;
    }
    let month = num(iso.get(5..7)?)?;
    if b[7] != b'-' {
        return None;
    }
    let day = num(iso.get(8..10)?)?;
    // Separator is 'T', 't', or a space.
    if !(b[10] == b'T' || b[10] == b't' || b[10] == b' ') {
        return None;
    }
    let hour = num(iso.get(11..13)?)?;
    if b[13] != b':' {
        return None;
    }
    let minute = num(iso.get(14..16)?)?;
    if b[16] != b':' {
        return None;
    }
    let second = num(iso.get(17..19)?)?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let days = days_from_civil(year, month, day);
    Some(days * 86_400 + hour * 3_600 + minute * 60 + second)
}

/// Parse + sort halving date strings into epoch seconds (mirrors Python
/// `parse_halving_dates`). Unparseable entries are dropped.
pub fn parse_halving_dates(dates: &[String]) -> Vec<i64> {
    let mut out: Vec<i64> = dates.iter().filter_map(|d| parse_epoch_secs(d)).collect();
    out.sort_unstable();
    out
}

pub fn default_halvings() -> Vec<i64> {
    let mut out: Vec<i64> = DEFAULT_HALVING_DATES_UTC
        .iter()
        .filter_map(|d| parse_epoch_secs(d))
        .collect();
    out.sort_unstable();
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeasonKind {
    Pre,
    Post,
}

impl SeasonKind {
    pub fn label(&self) -> &'static str {
        match self {
            SeasonKind::Pre => SEASON_PRE,
            SeasonKind::Post => SEASON_POST,
        }
    }
    /// Alphabetical key used by Python's secondary sort ("post" < "pre").
    fn key(&self) -> &'static str {
        match self {
            SeasonKind::Post => "post",
            SeasonKind::Pre => "pre",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    kind: SeasonKind,
    halving_index: usize,
    halving_epoch: i64,
    distance: i64,
}

fn candidates(ts: i64, halvings: &[i64]) -> Vec<Candidate> {
    let mut out = Vec::new();
    for (idx, &h) in halvings.iter().enumerate() {
        if h - WINDOW_SECS <= ts && ts < h {
            out.push(Candidate {
                kind: SeasonKind::Pre,
                halving_index: idx,
                halving_epoch: h,
                distance: h - ts,
            });
        } else if h <= ts && ts < h + WINDOW_SECS {
            out.push(Candidate {
                kind: SeasonKind::Post,
                halving_index: idx,
                halving_epoch: h,
                distance: ts - h,
            });
        }
    }
    out
}

fn stable_pick(symbol: &str, ts_iso: &str, seed: u64, tied: &[Candidate]) -> Candidate {
    let mut ordered: Vec<Candidate> = tied.to_vec();
    ordered.sort_by(|a, b| {
        a.halving_epoch
            .cmp(&b.halving_epoch)
            .then_with(|| a.kind.key().cmp(b.kind.key()))
    });
    let stable_key = format!("{symbol}|{ts_iso}|{seed}");
    let digest = Sha256::digest(stable_key.as_bytes());
    let mut first8 = [0u8; 8];
    first8.copy_from_slice(&digest[..8]);
    let pick = (u64::from_be_bytes(first8) % ordered.len() as u64) as usize;
    ordered[pick]
}

/// Returns `(season_label, halving_cycle_id)` for one timestamp. `halving_cycle_id`
/// is the index of the chosen halving, or `-1` for `season_unknown`.
///
/// `ts_iso` is used both to derive epoch seconds (window membership) and verbatim
/// in the SHA tie-break key, exactly mirroring Python's `select_season`.
pub fn select_season(
    symbol: &str,
    ts_iso: &str,
    halvings: &[i64],
    seed: u64,
) -> (&'static str, i64) {
    let ts = match parse_epoch_secs(ts_iso) {
        Some(t) => t,
        None => return (SEASON_UNKNOWN, -1),
    };
    let cands = candidates(ts, halvings);
    if cands.is_empty() {
        return (SEASON_UNKNOWN, -1);
    }
    let chosen = if cands.len() == 1 {
        cands[0]
    } else {
        let min_dist = cands.iter().map(|c| c.distance).min().unwrap();
        let tied: Vec<Candidate> = cands
            .iter()
            .copied()
            .filter(|c| c.distance == min_dist)
            .collect();
        if tied.len() == 1 {
            tied[0]
        } else {
            stable_pick(symbol, ts_iso, seed, &tied)
        }
    };
    (chosen.kind.label(), chosen.halving_index as i64)
}

/// Insert the one-hot season features (sum == 1) plus `halving_cycle_id` into a
/// feature map, mirroring Python `season_features`.
pub fn insert_season_features(
    map: &mut std::collections::HashMap<String, f64>,
    symbol: &str,
    ts_iso: &str,
    halvings: &[i64],
    seed: u64,
) {
    let (label, cycle_id) = select_season(symbol, ts_iso, halvings, seed);
    for name in SEASON_ONE_HOT {
        map.insert((*name).to_string(), if *name == label { 1.0 } else { 0.0 });
    }
    map.insert("halving_cycle_id".to_string(), cycle_id as f64);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_parsing_matches_known_values() {
        // 1970-01-01T00:00:00 == 0.
        assert_eq!(parse_epoch_secs("1970-01-01T00:00:00Z"), Some(0));
        // 2020-05-11T00:00:00Z == 1589155200.
        assert_eq!(
            parse_epoch_secs("2020-05-11T00:00:00Z"),
            Some(1_589_155_200)
        );
        // Space separator + offset suffix is tolerated.
        assert_eq!(
            parse_epoch_secs("2024-04-20 00:00:00+00:00"),
            Some(1_713_571_200)
        );
    }

    #[test]
    fn one_hot_sums_to_one() {
        let halvings = default_halvings();
        for ts in [
            "2010-01-01T00:00:00Z",
            "2024-05-01T00:00:00Z",
            "2050-01-01T00:00:00Z",
        ] {
            let mut m = std::collections::HashMap::new();
            insert_season_features(&mut m, "BTCUSDT", ts, &halvings, 0);
            let s = m[SEASON_PRE] + m[SEASON_POST] + m[SEASON_UNKNOWN];
            assert!((s - 1.0).abs() < 1e-12, "ts {ts} sum {s}");
        }
    }

    #[test]
    fn unknown_when_far_from_any_halving() {
        let halvings = default_halvings();
        let (label, cid) = select_season("BTCUSDT", "2005-01-01T00:00:00Z", &halvings, 0);
        assert_eq!(label, SEASON_UNKNOWN);
        assert_eq!(cid, -1);
    }

    #[test]
    fn equidistant_tie_is_deterministic() {
        // Two halvings ~1000 days apart; midpoint is inside both windows and
        // equidistant -> the SHA tie-break must be stable across calls.
        let h0 = parse_epoch_secs("2020-01-01T00:00:00Z").unwrap();
        let h1 = h0 + 1000 * 86_400;
        let halvings = vec![h0, h1];
        // Exact midpoint: 2020-01-01 + 500 days == 2021-05-15T00:00:00Z.
        let mid_iso = "2021-05-15T00:00:00+00:00";
        let mid = parse_epoch_secs(mid_iso).unwrap();
        // Confirm equidistant.
        assert_eq!(mid - h0, h1 - mid);
        let a = select_season("BTCUSDT", mid_iso, &halvings, 0);
        let b = select_season("BTCUSDT", mid_iso, &halvings, 0);
        assert_eq!(a, b);
        assert!(a.1 == 0 || a.1 == 1);
    }
}
