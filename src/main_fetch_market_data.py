"""Download public Binance USDⓈ-M Futures history into the two raw sources.

Produces, per symbol, BOTH:

* ``data/raw/{SYMBOL}_daily.csv``  (interval ``1d``)
* ``data/raw/{SYMBOL}_1m.csv``     (interval ``1m``)

Design notes (production-safe):

* Uses the official public archive at ``https://data.binance.vision`` (monthly
  zips for whole months, daily zips for the trailing partial month) instead of
  thousands of small REST calls. No API key is required.
* ``--market futures`` (default) pulls USDⓈ-M futures; ``--market spot`` pulls the
  spot archive, which is the only source for pre-2019 history (e.g. 2018, since
  Binance futures launched 2019-09). Both write the same {SYMBOL}_*.csv files and
  merge by timestamp, so a spot 2018 window can sit alongside futures 2022/2026.
* Daily output only contains CLOSED days (the current UTC day is dropped).
* Writes atomically (``.tmp`` then replace) and merges with any existing file so
  re-runs never create duplicate rows.
* Missing months/days are reported, not silently ignored.
* The live trading bot NEVER calls this; it is an offline data tool only.

Example::

    python -m src.main_fetch_market_data \
        --symbols BTCUSDT ETHUSDT SOLUSDT \
        --start 2022-01-01 --daily --minute
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Tuple

import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
DEFAULT_BASE_URL = "https://data.binance.vision"
# Archive prefix per market. USDⓈ-M futures only exist from ~2019-09; for older
# history (e.g. the 2018 halving regime) the spot archive is the only source.
_ARCHIVE_PREFIX = {
    "futures": "data/futures/um",
    "spot": "data/spot",
}

# Binance klines CSV column order (no/optional header row).
_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


# ---------------------------------------------------------------------------
# Pure helpers (no network) -- unit tested.
# ---------------------------------------------------------------------------
def month_iter(start: date, end: date) -> List[Tuple[int, int]]:
    """List of ``(year, month)`` covering ``[start, end]`` inclusive of months."""
    out: List[Tuple[int, int]] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return out


def _parse_open_time(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    sample = numeric.dropna()
    unit = "ms"
    if not sample.empty:
        median = float(sample.median())
        if median > 1e14:
            unit = "us"
        elif median > 1e11:
            unit = "ms"
        else:
            unit = "s"
    return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")


def parse_klines_csv(text: str) -> pd.DataFrame:
    """Parse a Binance klines CSV (with or without header) to OHLCV."""
    if not text or not text.strip():
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    first = text.splitlines()[0].lower()
    has_header = "open_time" in first or first.startswith("open time")
    df = pd.read_csv(
        io.StringIO(text),
        header=0 if has_header else None,
        names=None if has_header else _KLINE_COLS,
    )
    if has_header:
        df = df.rename(columns={c: c.strip().lower().replace(" ", "_") for c in df.columns})
    ts = _parse_open_time(df["open_time"])
    out = pd.DataFrame(
        {
            "timestamp": ts,
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df["volume"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["timestamp", "close"])
    return out[OHLCV_COLUMNS]


def combine_and_dedupe(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate, sort by timestamp and drop duplicate timestamps (keep last)."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    return df.reset_index(drop=True)[OHLCV_COLUMNS]


def only_closed_days(df: pd.DataFrame, now: Optional[datetime] = None) -> pd.DataFrame:
    """Drop rows belonging to the current (still-open) UTC day."""
    if df.empty:
        return df
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(timezone.utc).date()
    ts = pd.to_datetime(df["timestamp"], utc=True)
    keep = ts.dt.date < today
    return df[keep].reset_index(drop=True)


def atomic_write_csv(df: pd.DataFrame, path: str) -> str:
    """Write ``df`` to ``path`` via a temporary file then atomic replace."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


def _parse_existing_timestamps(series: pd.Series) -> pd.Series:
    """Parse an existing file's timestamp column to tz-aware UTC datetimes.

    Existing raw files may store timestamps as epoch integers (ms/us/s) OR as ISO
    strings. A plain ``pd.to_datetime`` on an epoch-MS *integer* column wrongly
    assumes nanoseconds (mapping 2022+ rows to 1970), so numeric columns must go
    through the unit-detecting parser. Mirrors ``data_loader._parse_timestamps``.
    """
    if pd.api.types.is_numeric_dtype(series):
        return _parse_open_time(series)
    return pd.to_datetime(series, utc=True, errors="coerce")


def merge_with_existing(path: str, df: pd.DataFrame) -> pd.DataFrame:
    """Merge newly fetched rows with any existing file (dedupe by timestamp)."""
    if os.path.isfile(path):
        try:
            existing = pd.read_csv(path)
            existing["timestamp"] = _parse_existing_timestamps(existing["timestamp"])
            existing = existing.dropna(subset=["timestamp"])
            keep = [c for c in OHLCV_COLUMNS if c in existing.columns]
            return combine_and_dedupe([existing[keep], df])
        except Exception as exc:  # noqa: BLE001
            print(f"warning: could not merge existing {path}: {exc}")
    return combine_and_dedupe([df])


# ---------------------------------------------------------------------------
# Network layer.
# ---------------------------------------------------------------------------
def _http_get(url: str, timeout: float = 60.0) -> Optional[bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    except urllib.error.URLError as exc:
        print(f"warning: network error for {url}: {exc}")
        return None


def _zip_to_text(payload: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        name = zf.namelist()[0]
        return zf.read(name).decode("utf-8")


def _download_archive(base_url, prefix, symbol, interval, period, label) -> Optional[pd.DataFrame]:
    url = f"{base_url}/{prefix}/{period}/klines/{symbol}/{interval}/{symbol}-{interval}-{label}.zip"
    payload = _http_get(url)
    if payload is None:
        return None
    try:
        return parse_klines_csv(_zip_to_text(payload))
    except Exception as exc:  # noqa: BLE001
        print(f"warning: failed to parse {url}: {exc}")
        return None


def fetch_symbol_interval(base_url, prefix, symbol, interval, start, end) -> Tuple[pd.DataFrame, List[str]]:
    """Fetch ``[start, end]`` for one interval; returns (frame, missing_labels)."""
    frames: List[pd.DataFrame] = []
    missing: List[str] = []

    months = month_iter(start, end)
    today = datetime.now(timezone.utc).date()
    current_month = (today.year, today.month)

    for (y, m) in months:
        if (y, m) == current_month:
            # Trailing partial month: use daily archives up to yesterday.
            d = date(y, m, 1)
            while d <= end and d < today:
                label = d.strftime("%Y-%m-%d")
                frame = _download_archive(base_url, prefix, symbol, interval, "daily", label)
                if frame is None:
                    missing.append(f"{interval}:{label}")
                else:
                    frames.append(frame)
                d += timedelta(days=1)
        else:
            label = f"{y:04d}-{m:02d}"
            frame = _download_archive(base_url, prefix, symbol, interval, "monthly", label)
            if frame is None:
                missing.append(f"{interval}:{label}")
            else:
                frames.append(frame)

    return combine_and_dedupe(frames), missing


def fetch_symbol(base_url, prefix, raw_dir, symbol, start, end, do_daily, do_minute) -> dict:
    report = {"symbol": symbol, "missing": []}
    os.makedirs(raw_dir, exist_ok=True)

    if do_daily:
        daily, missing = fetch_symbol_interval(base_url, prefix, symbol, "1d", start, end)
        daily = only_closed_days(daily)
        path = os.path.join(raw_dir, f"{symbol}_daily.csv")
        merged = merge_with_existing(path, daily)
        if not merged.empty:
            atomic_write_csv(merged, path)
        report["missing"].extend(missing)
        report["daily_rows"] = int(len(merged))
        print(f"{symbol} daily rows={len(merged)} missing={len(missing)}")

    if do_minute:
        minute, missing = fetch_symbol_interval(base_url, prefix, symbol, "1m", start, end)
        path = os.path.join(raw_dir, f"{symbol}_1m.csv")
        merged = merge_with_existing(path, minute)
        if not merged.empty:
            atomic_write_csv(merged, path)
        report["missing"].extend(missing)
        report["minute_rows"] = int(len(merged))
        print(f"{symbol} minute rows={len(merged)} missing={len(missing)}")

    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch public Binance USD-M futures OHLCV.")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", required=True, help="UTC start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="UTC end date YYYY-MM-DD (default: today)")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--market",
        choices=sorted(_ARCHIVE_PREFIX),
        default="futures",
        help="archive to pull from (default futures; use 'spot' for pre-2019 history)",
    )
    parser.add_argument("--daily", action="store_true", help="produce {SYMBOL}_daily.csv")
    parser.add_argument("--minute", action="store_true", help="produce {SYMBOL}_1m.csv")
    args = parser.parse_args(argv)

    if not args.daily and not args.minute:
        parser.error("specify at least one of --daily / --minute")

    prefix = _ARCHIVE_PREFIX[args.market]
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else datetime.now(timezone.utc).date()

    all_missing: List[str] = []
    for symbol in args.symbols:
        rep = fetch_symbol(args.base_url, prefix, args.raw_dir, symbol, start, end, args.daily, args.minute)
        all_missing.extend(f"{symbol} {x}" for x in rep["missing"])

    if all_missing:
        print(f"\nMISSING ({len(all_missing)} archives):")
        for item in all_missing:
            print(f"  {item}")
    else:
        print("\nNo missing archives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
