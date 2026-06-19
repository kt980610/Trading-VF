"""Loading and daily-normalizing of historical OHLCV data."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

_COLUMN_ALIASES = {
    "timestamp": "timestamp",
    "time": "timestamp",
    "date": "timestamp",
    "datetime": "timestamp",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}


def _candidate_paths(raw_dir: str, symbol: str, timeframe: str):
    return [
        os.path.join(raw_dir, f"{symbol}_{timeframe}.csv"),
        os.path.join(raw_dir, f"{symbol}_daily.csv"),
        os.path.join(raw_dir, f"{symbol}.csv"),
    ]


def _find_file(raw_dir: str, symbol: str, timeframe: str) -> Optional[str]:
    for path in _candidate_paths(raw_dir, symbol, timeframe):
        if os.path.isfile(path):
            return path
    return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in _COLUMN_ALIASES:
            rename[col] = _COLUMN_ALIASES[key]
    return df.rename(columns=rename)


def _parse_timestamps(series: pd.Series) -> pd.Series:
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
        return pd.to_datetime(numeric, unit=unit, utc=True)
    return pd.to_datetime(series, utc=True)


def _resample_daily(df: pd.DataFrame) -> pd.DataFrame:
    indexed = df.set_index("timestamp").sort_index()
    agg = {}
    if "open" in indexed.columns:
        agg["open"] = "first"
    if "high" in indexed.columns:
        agg["high"] = "max"
    if "low" in indexed.columns:
        agg["low"] = "min"
    agg["close"] = "last"
    agg["volume"] = "sum"
    daily = indexed.resample("1D").agg(agg)
    daily = daily.dropna(subset=["close"])
    return daily.reset_index()


def load_daily(config, symbol: str) -> Optional[pd.DataFrame]:
    raw_dir = config.resolve(config.data.raw_dir)
    timeframe = config.data.timeframe
    path = _find_file(raw_dir, symbol, timeframe)
    if path is None:
        return None

    df = pd.read_csv(path)
    df = _normalize_columns(df)

    if "timestamp" not in df.columns or "close" not in df.columns:
        return None
    if "volume" not in df.columns:
        df["volume"] = np.nan

    df["timestamp"] = _parse_timestamps(df["timestamp"])
    df = df.dropna(subset=["timestamp"])

    is_minute_source = "_daily.csv" not in os.path.basename(path).lower()
    if timeframe == "1m" and config.data.daily_resample and is_minute_source:
        df = _resample_daily(df)
    else:
        df = df.sort_values("timestamp")
        df["day"] = df["timestamp"].dt.floor("1D")
        df = df.drop_duplicates(subset="day", keep="last")
        df["timestamp"] = df["day"]
        df = df.drop(columns="day")

    df = df.sort_values("timestamp")
    df["day"] = df["timestamp"].dt.floor("1D")
    df = df.drop_duplicates(subset="day", keep="last").drop(columns="day")
    df = df.reset_index(drop=True)

    keep = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep]


def load_minute(config, symbol: str) -> Optional[pd.DataFrame]:
    """Load raw (non-resampled) bars for minute-level simulation.

    Returns OHLCV sorted ascending by parsed UTC timestamp. Used by the
    simulator/benchmark where intraday high/low are required for liquidation
    checks; falls back to whatever granularity the source file provides.
    """
    raw_dir = config.resolve(config.data.raw_dir)
    timeframe = config.data.timeframe
    path = _find_file(raw_dir, symbol, timeframe)
    if path is None:
        return None

    df = pd.read_csv(path)
    df = _normalize_columns(df)
    if "timestamp" not in df.columns or "close" not in df.columns:
        return None
    for col in ["open", "high", "low"]:
        if col not in df.columns:
            df[col] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = np.nan

    df["timestamp"] = _parse_timestamps(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    keep = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep]
