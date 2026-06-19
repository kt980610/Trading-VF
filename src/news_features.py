"""Daily news-feature aggregation with strict no-leakage cutoff (spec section 13).

For day ``D`` (UTC) only news with ``timestamp < D_start`` may be used. Features
are aggregated per (date, symbol) across the five categories. Sentiment comes
from an explicit ``sentiment_score`` when present, otherwise from a pluggable
backend (a tiny built-in lexicon by default, with hooks for a precomputed table,
FinBERT or VADER).
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

CATEGORIES = ["macro", "policy", "stock_market", "crypto_market", "symbol_specific"]

# Coin-specific (symbol_specific) news is correlation-weighted; the other four
# categories are global and used as-is across all coins (spec section 5).
GLOBAL_CATEGORIES = ["macro", "policy", "stock_market", "crypto_market"]

WEIGHTED_NEWS_AGGREGATES = [
    "weighted_symbol_news_sentiment",
    "weighted_symbol_news_count",
    "weighted_positive_symbol_news_count",
    "weighted_negative_symbol_news_count",
]

EPSILON = 1e-9


def cross_news_source_features(universe) -> list:
    """Per-source correlation-weighted sentiment feature names."""
    return [f"news_sentiment_from_{s}_weighted" for s in universe]

_POSITIVE = {
    "surge", "rally", "gain", "gains", "bull", "bullish", "up", "soar", "soars",
    "beat", "beats", "record", "high", "growth", "boost", "optimism", "approve",
    "approved", "support", "win", "wins", "strong", "rise", "rises", "jump",
    "positive", "upgrade", "adopt", "adoption", "inflow", "inflows",
}
_NEGATIVE = {
    "crash", "plunge", "drop", "drops", "bear", "bearish", "down", "fall",
    "falls", "miss", "misses", "low", "loss", "losses", "fear", "ban", "banned",
    "reject", "rejected", "weak", "decline", "slump", "selloff", "negative",
    "downgrade", "hack", "hacked", "lawsuit", "outflow", "outflows", "fraud",
}

_CATEGORY_KEYWORDS = {
    "macro": {"inflation", "cpi", "gdp", "unemployment", "fed", "interest rate", "recession"},
    "policy": {"regulation", "regulatory", "sec", "law", "government", "policy", "congress"},
    "stock_market": {"stock", "equity", "nasdaq", "s&p", "dow", "shares", "earnings"},
    "crypto_market": {"crypto", "bitcoin", "ethereum", "blockchain", "defi", "altcoin", "token"},
}


def _lexicon_sentiment(text: str) -> float:
    if not text:
        return 0.0
    words = str(text).lower().replace(",", " ").replace(".", " ").split()
    pos = sum(1 for w in words if w in _POSITIVE)
    neg = sum(1 for w in words if w in _NEGATIVE)
    total = pos + neg
    if total == 0:
        return 0.0
    return float((pos - neg) / total)


def _infer_category(row: pd.Series) -> str:
    if isinstance(row.get("symbol"), str) and row.get("symbol"):
        return "symbol_specific"
    text = f"{row.get('title', '')} {row.get('body', '')} {row.get('query', '')}".lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return category
    return "macro"


def load_news(path: str) -> pd.DataFrame:
    """Load raw news from JSONL or CSV into a normalized DataFrame."""
    if not os.path.isfile(path):
        return pd.DataFrame(
            columns=["timestamp", "source", "title", "body", "query", "symbol", "category", "sentiment_score"]
        )

    if path.lower().endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        df = pd.DataFrame(records)
    elif path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as fh:
            df = pd.DataFrame(json.load(fh))
    else:
        df = pd.read_csv(path)

    for col in ["source", "title", "body", "query", "symbol", "category"]:
        if col not in df.columns:
            df[col] = None
    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = np.nan

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    return df


def _score_row(row: pd.Series, backend: str, table: Optional[dict]) -> float:
    score = row.get("sentiment_score")
    if score is not None and pd.notna(score):
        return float(score)
    if backend == "table" and table is not None:
        key = str(row.get("timestamp").date())
        if key in table:
            return float(table[key])
    # lexicon (default) and unimplemented external backends fall back to lexicon
    text = f"{row.get('title', '')} {row.get('body', '')}"
    return _lexicon_sentiment(text)


def prepare_news(df: pd.DataFrame, backend: str = "lexicon", table: Optional[dict] = None) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        df["category"] = []
        df["score"] = []
        return df
    # Ensure UTC datetime even when called directly (not via load_news).
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["category"] = df.apply(
        lambda r: r["category"] if isinstance(r.get("category"), str) and r["category"] in CATEGORIES
        else _infer_category(r),
        axis=1,
    )
    df["score"] = df.apply(lambda r: _score_row(r, backend, table), axis=1)
    return df


def _empty_feature_row(date_str: str, symbol: str) -> Dict[str, float]:
    row: Dict[str, float] = {"date": date_str, "symbol": symbol}
    for cat in CATEGORIES:
        row[f"{cat}_news_sentiment"] = 0.0
        row[f"{cat}_news_count"] = 0
    return row


def daily_features_for_symbol(
    news: pd.DataFrame,
    symbol: str,
    day_starts: Iterable[pd.Timestamp],
    lookback_days: int = 1,
) -> List[Dict[str, float]]:
    """Aggregate per-day features for one symbol with ``timestamp < D_start``."""
    out: List[Dict[str, float]] = []
    ts = news["timestamp"] if not news.empty else pd.Series([], dtype="datetime64[ns, UTC]")

    for day_start in day_starts:
        day_start = pd.Timestamp(day_start)
        if day_start.tzinfo is None:
            day_start = day_start.tz_localize("utc")
        date_str = str(day_start.date())
        row = _empty_feature_row(date_str, symbol)

        if news.empty:
            out.append(row)
            continue

        window_start = day_start - pd.Timedelta(days=lookback_days)
        # Strict no-leakage cutoff: timestamp < day_start.
        mask = (ts >= window_start) & (ts < day_start)
        window = news.loc[mask]
        if window.empty:
            out.append(row)
            continue

        for cat in CATEGORIES:
            if cat == "symbol_specific":
                sub = window[(window["category"] == cat) & (window["symbol"] == symbol)]
            else:
                sub = window[window["category"] == cat]
            row[f"{cat}_news_count"] = int(len(sub))
            row[f"{cat}_news_sentiment"] = float(sub["score"].mean()) if len(sub) else 0.0
        out.append(row)
    return out


def _symbol_specific_stats(window: pd.DataFrame, source_symbol: str) -> Dict[str, float]:
    """Raw symbol-specific sentiment/count for one source coin in the window."""
    sub = window[(window["category"] == "symbol_specific") & (window["symbol"] == source_symbol)]
    count = int(len(sub))
    if count == 0:
        return {"sentiment": 0.0, "count": 0, "pos": 0, "neg": 0}
    scores = sub["score"].astype("float64")
    return {
        "sentiment": float(scores.mean()),
        "count": count,
        "pos": int((scores > 0).sum()),
        "neg": int((scores < 0).sum()),
    }


def daily_cross_features_for_symbol(
    news: pd.DataFrame,
    target: str,
    universe: List[str],
    day_starts: Iterable[pd.Timestamp],
    corr_provider=None,
    lookback_days: int = 1,
) -> List[Dict[str, float]]:
    """Per-day news features for ``target`` with correlation-weighted cross-coin news.

    Global categories (macro/policy/stock_market/crypto_market) are used as-is.
    Coin-specific news from every source ``j`` is weighted by ``corr[target, j]``:
    sentiment uses signed correlation, counts use ``abs`` correlation (section 4).
    """
    out: List[Dict[str, float]] = []
    ts = news["timestamp"] if not news.empty else pd.Series([], dtype="datetime64[ns, UTC]")

    def weight(date_str: str, source: str) -> float:
        if corr_provider is not None:
            return float(corr_provider.weight(date_str, target, source))
        return 1.0 if source == target else 0.0

    for day_start in day_starts:
        day_start = pd.Timestamp(day_start)
        if day_start.tzinfo is None:
            day_start = day_start.tz_localize("utc")
        date_str = str(day_start.date())

        row = _empty_feature_row(date_str, target)
        for agg in WEIGHTED_NEWS_AGGREGATES:
            row[agg] = 0.0
        for src in universe:
            row[f"news_sentiment_from_{src}_weighted"] = 0.0

        if news.empty:
            out.append(row)
            continue

        window_start = day_start - pd.Timedelta(days=lookback_days)
        mask = (ts >= window_start) & (ts < day_start)
        window = news.loc[mask]
        if window.empty:
            out.append(row)
            continue

        # Global categories + the target's own symbol-specific news (unchanged).
        for cat in CATEGORIES:
            if cat == "symbol_specific":
                sub = window[(window["category"] == cat) & (window["symbol"] == target)]
            else:
                sub = window[window["category"] == cat]
            row[f"{cat}_news_count"] = int(len(sub))
            row[f"{cat}_news_sentiment"] = float(sub["score"].mean()) if len(sub) else 0.0

        # Correlation-weighted cross-coin symbol-specific news.
        stats = {src: _symbol_specific_stats(window, src) for src in universe}
        num = 0.0
        den = 0.0
        w_count = 0.0
        w_pos = 0.0
        w_neg = 0.0
        for src in universe:
            w = weight(date_str, src)
            st = stats[src]
            num += w * st["sentiment"]
            den += abs(w)
            w_count += abs(w) * st["count"]
            w_pos += abs(w) * st["pos"]
            w_neg += abs(w) * st["neg"]
            row[f"news_sentiment_from_{src}_weighted"] = float(w * st["sentiment"])

        row["weighted_symbol_news_sentiment"] = float(num / den) if den > EPSILON else 0.0
        row["weighted_symbol_news_count"] = float(w_count)
        row["weighted_positive_symbol_news_count"] = float(w_pos)
        row["weighted_negative_symbol_news_count"] = float(w_neg)
        out.append(row)
    return out


def write_jsonl(rows: Iterable[Dict[str, float]], output_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return output_path
