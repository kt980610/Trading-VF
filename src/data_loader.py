"""Loading of historical OHLCV data from two explicit raw sources.

Production rule (no implicit fallback): the daily and minute research/artifact
pipelines each read their OWN file and never substitute the other timeframe.

* ``load_daily_ohlcv``  -> reads ONLY ``data/raw/{SYMBOL}_daily.csv``
* ``load_minute_ohlcv`` -> reads ONLY ``data/raw/{SYMBOL}_1m.csv``

The ``timeframe`` config field is deprecated and MUST NOT influence source-file
selection. When a required source file is missing an explicit, machine-readable
reason is raised (``missing_daily_source`` / ``missing_minute_source``).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

REASON_MISSING_DAILY = "missing_daily_source"
REASON_MISSING_MINUTE = "missing_minute_source"
REASON_BAD_SCHEMA = "bad_source_schema"

_COLUMN_ALIASES = {
    "timestamp": "timestamp",
    "time": "timestamp",
    "date": "timestamp",
    "datetime": "timestamp",
    "open_time": "timestamp",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}


class DataSourceError(Exception):
    """Raised when a required raw data source is missing or malformed.

    ``reason`` is a stable, machine-readable code (see the ``REASON_*``
    constants) so callers can record a precise skip/error reason.
    """

    def __init__(self, reason: str, symbol: str, path: str, detail: str = ""):
        self.reason = reason
        self.symbol = symbol
        self.path = path
        self.detail = detail
        msg = f"{reason} for {symbol}: {path}"
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in _COLUMN_ALIASES:
            rename[col] = _COLUMN_ALIASES[key]
    return df.rename(columns=rename)


def _parse_timestamps(series: pd.Series) -> pd.Series:
    """Parse a timestamp column to tz-aware UTC datetimes."""
    if pd.api.types.is_numeric_dtype(series):
        numeric = series.astype("float64")
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
    return pd.to_datetime(series, utc=True, errors="coerce")


def _daily_path(config, symbol: str) -> str:
    raw_dir = config.resolve(config.data.raw_dir)
    return os.path.join(raw_dir, f"{symbol}{config.data.daily_file_suffix}")


def _minute_path(config, symbol: str) -> str:
    raw_dir = config.resolve(config.data.raw_dir)
    return os.path.join(raw_dir, f"{symbol}{config.data.minute_file_suffix}")


def _read_ohlcv(path: str, symbol: str, reason_missing: str) -> pd.DataFrame:
    """Read, validate and clean a single OHLCV source file.

    Enforces the canonical schema ``timestamp,open,high,low,close,volume``,
    parses timestamps as UTC, sorts ascending, drops duplicate timestamps and
    reports the number of corrupt rows removed.
    """
    df = pd.read_csv(path)
    df = _normalize_columns(df)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataSourceError(
            REASON_BAD_SCHEMA, symbol, path, detail=f"missing columns: {missing}"
        )

    n_in = len(df)
    df["timestamp"] = _parse_timestamps(df["timestamp"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Corrupt rows = unparseable timestamp or close. Reported, not silently kept.
    bad_mask = df["timestamp"].isna() | df["close"].isna()
    n_bad = int(bad_mask.sum())
    if n_bad:
        print(f"{symbol} source={os.path.basename(path)} dropped_corrupt_rows={n_bad}")
    df = df[~bad_mask].copy()

    df = df.sort_values("timestamp")
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset="timestamp", keep="last")
    n_dups = n_before_dedup - len(df)
    if n_dups:
        print(f"{symbol} source={os.path.basename(path)} dropped_duplicate_timestamps={n_dups}")

    df = df.reset_index(drop=True)
    if df.empty:
        raise DataSourceError(
            reason_missing, symbol, path, detail=f"no valid rows (in={n_in})"
        )
    return df[REQUIRED_COLUMNS]


def load_daily_ohlcv(config, symbol: str) -> Optional[pd.DataFrame]:
    """Load daily OHLCV from ``{SYMBOL}_daily.csv`` ONLY.

    Raises :class:`DataSourceError` (``missing_daily_source``) when the file is
    absent and ``data.require_daily_source`` is true; otherwise returns ``None``.
    Never falls back to the minute source.
    """
    path = _daily_path(config, symbol)
    if not os.path.isfile(path):
        if getattr(config.data, "require_daily_source", True):
            raise DataSourceError(REASON_MISSING_DAILY, symbol, path)
        return None

    df = _read_ohlcv(path, symbol, REASON_MISSING_DAILY)
    # Collapse to one row per UTC day (defensive; daily files are already daily).
    df["__day"] = df["timestamp"].dt.floor("1D")
    df = df.drop_duplicates(subset="__day", keep="last")
    df["timestamp"] = df["__day"]
    df = df.drop(columns="__day").reset_index(drop=True)
    return df


def load_minute_ohlcv(config, symbol: str) -> Optional[pd.DataFrame]:
    """Load minute OHLCV from ``{SYMBOL}_1m.csv`` ONLY.

    Raises :class:`DataSourceError` (``missing_minute_source``) when the file is
    absent and ``data.require_minute_source`` is true; otherwise returns ``None``.
    Never falls back to the daily source.
    """
    path = _minute_path(config, symbol)
    if not os.path.isfile(path):
        if getattr(config.data, "require_minute_source", True):
            raise DataSourceError(REASON_MISSING_MINUTE, symbol, path)
        return None
    return _read_ohlcv(path, symbol, REASON_MISSING_MINUTE)


# ---------------------------------------------------------------------------
# Deprecated aliases. Kept so older imports do not break, but they now enforce
# the strict, source-specific behavior (no timeframe-driven file selection).
# ---------------------------------------------------------------------------
def load_daily(config, symbol: str) -> Optional[pd.DataFrame]:  # pragma: no cover
    """Deprecated: use :func:`load_daily_ohlcv`."""
    return load_daily_ohlcv(config, symbol)


def load_minute(config, symbol: str) -> Optional[pd.DataFrame]:  # pragma: no cover
    """Deprecated: use :func:`load_minute_ohlcv`."""
    return load_minute_ohlcv(config, symbol)


def minute_returns(close: pd.Series) -> pd.Series:
    """Minute return series ``close_t / close_t_minus_1 - 1`` (return_decimal)."""
    close = pd.Series(close, dtype="float64")
    return (close / close.shift(1) - 1.0).iloc[1:]


__all__ = [
    "DataSourceError",
    "REASON_MISSING_DAILY",
    "REASON_MISSING_MINUTE",
    "REASON_BAD_SCHEMA",
    "REQUIRED_COLUMNS",
    "load_daily_ohlcv",
    "load_minute_ohlcv",
    "minute_returns",
]
