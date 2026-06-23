"""Cross-coin daily return correlation matrices (spec section 2).

For day ``D`` the correlation matrix is computed ONLY from daily returns dated
strictly before ``D`` (no leakage), within a rolling ``lookback_days`` window.
Pairs with fewer than ``min_required_days`` overlapping observations fall back to
configured constants (self -> 1.0, cross -> 0.0 by default).
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

EPSILON = 1e-9


def daily_returns_matrix(daily_by_symbol: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a date-indexed return matrix (columns = symbols).

    Each input frame must have ``timestamp`` and ``close``. Daily simple returns
    ``close_t / close_{t-1} - 1`` are aligned on the (UTC, day-floored) date.
    """
    series_by_symbol: Dict[str, pd.Series] = {}
    for symbol, daily in daily_by_symbol.items():
        if daily is None or daily.empty or "close" not in daily.columns:
            continue
        df = daily.copy()
        date = pd.to_datetime(df["timestamp"], utc=True).dt.floor("1D")
        close = df["close"].astype("float64")
        ret = close / close.shift(1) - 1.0
        s = pd.Series(ret.to_numpy(), index=date.to_numpy())
        s = s[~s.index.duplicated(keep="last")]
        series_by_symbol[symbol] = s

    if not series_by_symbol:
        return pd.DataFrame()

    matrix = pd.DataFrame(series_by_symbol).sort_index()
    return matrix


def _pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size < 2:
        return None
    sa = a.std()
    sb = b.std()
    if sa < EPSILON or sb < EPSILON:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def correlation_for_date(
    returns_matrix: pd.DataFrame,
    asof_date: pd.Timestamp,
    symbols: List[str],
    lookback_days: int = 90,
    min_required_days: int = 60,
    method: str = "pearson",
    fallback_self_corr: float = 1.0,
    fallback_cross_corr: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    """Compute the symmetric correlation matrix usable on day ``asof_date``."""
    asof = pd.Timestamp(asof_date)
    if asof.tzinfo is None:
        asof = asof.tz_localize("utc")
    window_start = asof - pd.Timedelta(days=lookback_days)

    corr: Dict[str, Dict[str, float]] = {i: {} for i in symbols}

    if returns_matrix is None or returns_matrix.empty:
        for i in symbols:
            for j in symbols:
                corr[i][j] = fallback_self_corr if i == j else fallback_cross_corr
        return corr

    idx = pd.to_datetime(returns_matrix.index, utc=True)
    # Strict no-leakage cutoff: only returns BEFORE the day start.
    mask = np.asarray((idx >= window_start) & (idx < asof))
    window = returns_matrix[mask]

    for a_i, i in enumerate(symbols):
        for j in symbols[a_i:]:
            if i == j:
                value = fallback_self_corr
            else:
                value = fallback_cross_corr
                if i in window.columns and j in window.columns:
                    sub = window[[i, j]].dropna()
                    if len(sub) >= min_required_days:
                        r = _pearson(
                            sub[i].to_numpy(dtype="float64"),
                            sub[j].to_numpy(dtype="float64"),
                        )
                        if r is not None:
                            value = r
            corr[i][j] = value
            corr[j][i] = value
    return corr


def build_daily_matrices(
    returns_matrix: pd.DataFrame,
    symbols: List[str],
    dates: Iterable[pd.Timestamp],
    lookback_days: int = 90,
    min_required_days: int = 60,
    method: str = "pearson",
    fallback_self_corr: float = 1.0,
    fallback_cross_corr: float = 0.0,
) -> List[Dict[str, object]]:
    """Produce one correlation-matrix record per date (jsonl rows)."""
    rows: List[Dict[str, object]] = []
    for d in dates:
        d = pd.Timestamp(d)
        if d.tzinfo is None:
            d = d.tz_localize("utc")
        corr = correlation_for_date(
            returns_matrix, d, symbols,
            lookback_days=lookback_days,
            min_required_days=min_required_days,
            method=method,
            fallback_self_corr=fallback_self_corr,
            fallback_cross_corr=fallback_cross_corr,
        )
        rows.append(
            {
                "date": str(d.date()),
                "lookback_days": int(lookback_days),
                "symbols": list(symbols),
                "corr": corr,
            }
        )
    return rows


def write_jsonl(rows: Iterable[Dict[str, object]], output_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return output_path


class CorrelationProvider:
    """Lookup of correlation weights by date with safe fallbacks."""

    def __init__(
        self,
        by_date: Dict[str, Dict[str, Dict[str, float]]],
        fallback_self_corr: float = 1.0,
        fallback_cross_corr: float = 0.0,
    ):
        self.by_date = by_date
        self.sorted_dates = sorted(by_date.keys())
        self.fallback_self_corr = fallback_self_corr
        self.fallback_cross_corr = fallback_cross_corr

    @classmethod
    def from_rows(cls, rows: Iterable[Dict[str, object]], **kwargs) -> "CorrelationProvider":
        by_date = {str(r["date"]): r.get("corr", {}) for r in rows}
        return cls(by_date, **kwargs)

    @classmethod
    def load(cls, path: str, **kwargs) -> "CorrelationProvider":
        rows: List[Dict[str, object]] = []
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        return cls.from_rows(rows, **kwargs)

    def matrix_for(self, date: str) -> Dict[str, Dict[str, float]]:
        if date in self.by_date:
            return self.by_date[date]
        # Use the most recent matrix strictly before this date if present.
        prior = [d for d in self.sorted_dates if d < date]
        if prior:
            return self.by_date[prior[-1]]
        return {}

    def weight(self, date: str, target: str, source: str) -> float:
        matrix = self.matrix_for(date)
        row = matrix.get(target)
        if row is not None and source in row:
            return float(row[source])
        if target == source:
            return self.fallback_self_corr
        return self.fallback_cross_corr
