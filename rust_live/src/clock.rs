//! Minimal UTC clock helpers (no external date dependency).

use std::time::{SystemTime, UNIX_EPOCH};

pub fn now_unix_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

pub fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Howard Hinnant's days->civil algorithm.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as i64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

fn parts(secs: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    let (y, mo, d) = civil_from_days(days);
    let hour = (rem / 3600) as u32;
    let min = ((rem % 3600) / 60) as u32;
    let sec = (rem % 60) as u32;
    (y, mo, d, hour, min, sec)
}

/// Howard Hinnant's civil->days algorithm (inverse of `civil_from_days`).
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = (y - era * 400) as i64; // [0, 399]
    let mp = if m > 2 { m - 3 } else { m + 9 } as i64; // [0, 11]
    let doy = (153 * mp + 2) / 5 + d as i64 - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}

/// Parse a UTC ISO-8601 / RFC3339 timestamp (`YYYY-MM-DDTHH:MM:SS[.fff]Z`) into
/// epoch seconds. Any timezone designator is treated as UTC (the live news
/// artifacts always emit a trailing `Z`); fractional seconds are ignored.
/// Returns `None` when the leading 19 characters are not a valid date-time.
pub fn parse_rfc3339_secs(s: &str) -> Option<i64> {
    let s = s.trim();
    if s.len() < 19 {
        return None;
    }
    let y: i64 = s.get(0..4)?.parse().ok()?;
    let mo: u32 = s.get(5..7)?.parse().ok()?;
    let d: u32 = s.get(8..10)?.parse().ok()?;
    let h: i64 = s.get(11..13)?.parse().ok()?;
    let mi: i64 = s.get(14..16)?.parse().ok()?;
    let se: i64 = s.get(17..19)?.parse().ok()?;
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) {
        return None;
    }
    let days = days_from_civil(y, mo, d);
    Some(days * 86400 + h * 3600 + mi * 60 + se)
}

pub fn date_str(secs: i64) -> String {
    let (y, mo, d, _, _, _) = parts(secs);
    format!("{y:04}-{mo:02}-{d:02}")
}

pub fn rfc3339(secs: i64) -> String {
    let (y, mo, d, h, mi, s) = parts(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

pub fn today_date() -> String {
    date_str(now_unix_secs())
}

pub fn now_rfc3339() -> String {
    rfc3339(now_unix_secs())
}

pub fn hour_of_day(secs: i64) -> f64 {
    let (_, _, _, h, _, _) = parts(secs);
    h as f64
}

/// Day of week with Monday = 0 .. Sunday = 6 (matches Python's weekday()).
pub fn day_of_week(secs: i64) -> f64 {
    let days = secs.div_euclid(86400);
    ((days + 3).rem_euclid(7)) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_is_thursday_1970() {
        assert_eq!(date_str(0), "1970-01-01");
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        // 1970-01-01 was a Thursday -> Monday0 index 3.
        assert_eq!(day_of_week(0), 3.0);
    }

    #[test]
    fn known_date() {
        // 2024-09-29T00:00:00Z = 1727568000
        assert_eq!(date_str(1_727_568_000), "2024-09-29");
        assert_eq!(hour_of_day(1_727_568_000 + 3600 * 5), 5.0);
    }

    #[test]
    fn parse_roundtrips_rfc3339() {
        let secs = 1_727_568_000 + 3600 * 5 + 60 * 7 + 9;
        let s = rfc3339(secs);
        assert_eq!(parse_rfc3339_secs(&s), Some(secs));
        assert_eq!(parse_rfc3339_secs("2024-09-29T05:07:09Z"), Some(secs));
        // Fractional seconds + offset designator are tolerated (treated as UTC).
        assert_eq!(parse_rfc3339_secs("2024-09-29T05:07:09.123Z"), Some(secs));
        assert_eq!(parse_rfc3339_secs("garbage"), None);
    }
}
