"""Load simulated and realized per-symbol daily return series (spec sections 1-2).

Both loaders return a normalized long frame with columns ``date``, ``symbol``,
``return`` (plus a ``source`` tag). Daily return is taken directly when present,
otherwise derived as ``daily_pnl / base_capital_per_symbol``.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd

# Candidate PnL columns, in preference order, when daily_return is absent.
_PNL_COLUMNS = ["daily_pnl", "rf_daily_pnl", "pnl", "realized_pnl"]


def _normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.date


def load_simulated_returns(path: str, base_capital_per_symbol: float) -> pd.DataFrame:
    """Load ``data/simulation_results.csv`` into (date, symbol, return, source)."""
    cols = ["date", "symbol", "return", "source"]
    if not os.path.isfile(path):
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(path)
    if "symbol" not in df.columns or "date" not in df.columns:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame()
    out["date"] = _normalize_date(df["date"])
    out["symbol"] = df["symbol"].astype(str)

    if "daily_return" in df.columns:
        out["return"] = df["daily_return"].astype(float)
    else:
        pnl_col = next((c for c in _PNL_COLUMNS if c in df.columns), None)
        if pnl_col is None:
            return pd.DataFrame(columns=cols)
        denom = float(base_capital_per_symbol) or 1.0
        out["return"] = df[pnl_col].astype(float) / denom

    out["source"] = "simulated"
    out = out.dropna(subset=["date", "return"]).reset_index(drop=True)
    return out


def load_realized_returns(path: str, base_capital_per_symbol: float) -> pd.DataFrame:
    """Load ``data/realized_symbol_returns.jsonl`` into (date, symbol, return, source)."""
    cols = ["date", "symbol", "return", "source"]
    if not os.path.isfile(path):
        return pd.DataFrame(columns=cols)

    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(records)
    if "symbol" not in df.columns or "date" not in df.columns:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame()
    out["date"] = _normalize_date(df["date"])
    out["symbol"] = df["symbol"].astype(str)

    if "realized_return" in df.columns:
        out["return"] = df["realized_return"].astype(float)
    elif "realized_pnl" in df.columns:
        cap = df["base_capital_per_symbol"] if "base_capital_per_symbol" in df.columns else None
        denom = cap.astype(float) if cap is not None else float(base_capital_per_symbol)
        out["return"] = df["realized_pnl"].astype(float) / denom
    else:
        return pd.DataFrame(columns=cols)

    out["source"] = "live_realized"
    out = out.dropna(subset=["date", "return"]).reset_index(drop=True)
    return out
