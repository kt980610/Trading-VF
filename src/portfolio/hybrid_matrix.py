"""Hybrid return assembly + rolling return matrix (spec sections 2-3).

Realized returns override simulated returns for the same (date, symbol). A rolling
lookback window ``[D - lookback_days, D - 1]`` is then pivoted into a return
matrix ``R[d, i]``; symbols with fewer than ``min_required_days`` observations are
flagged invalid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List

import pandas as pd

REASON_NOT_ENOUGH_DAYS = "not_enough_return_days"


@dataclass
class ReturnMatrix:
    R: pd.DataFrame                       # index=date, columns=valid symbols
    valid_symbols: List[str]
    invalid: Dict[str, str] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    window_start: date = None
    window_end: date = None


def build_hybrid_returns(simulated: pd.DataFrame, realized: pd.DataFrame) -> pd.DataFrame:
    """Combine the two sources with realized taking priority (override)."""
    frames = [f for f in (realized, simulated) if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "return", "source"])
    # Realized first so keep="first" wins for duplicate (date, symbol).
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "symbol"], keep="first").reset_index(drop=True)
    return combined


def _as_date(value) -> date:
    if isinstance(value, date):
        return value
    return pd.to_datetime(value, utc=True).date()


def build_return_matrix(
    hybrid: pd.DataFrame,
    symbols: List[str],
    as_of_date,
    lookback_days: int = 30,
    min_required_days: int = 25,
) -> ReturnMatrix:
    """Pivot the lookback window into R[d, i] and flag invalid symbols."""
    d = _as_date(as_of_date)
    window_start = d - timedelta(days=lookback_days)
    window_end = d - timedelta(days=1)

    invalid: Dict[str, str] = {}
    counts: Dict[str, int] = {}

    if hybrid is None or hybrid.empty:
        for s in symbols:
            invalid[s] = REASON_NOT_ENOUGH_DAYS
            counts[s] = 0
        return ReturnMatrix(pd.DataFrame(), [], invalid, counts, window_start, window_end)

    mask = (hybrid["date"] >= window_start) & (hybrid["date"] <= window_end)
    window = hybrid.loc[mask]

    pivot = window.pivot_table(index="date", columns="symbol", values="return", aggfunc="last")
    pivot = pivot.sort_index()

    valid_symbols: List[str] = []
    for s in symbols:
        n = int(pivot[s].count()) if s in pivot.columns else 0
        counts[s] = n
        if n >= min_required_days:
            valid_symbols.append(s)
        else:
            invalid[s] = REASON_NOT_ENOUGH_DAYS

    if valid_symbols:
        R = pivot[valid_symbols]
    else:
        R = pd.DataFrame()

    return ReturnMatrix(R, valid_symbols, invalid, counts, window_start, window_end)
