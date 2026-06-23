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
import re
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

CATEGORIES = ["macro", "policy", "stock_market", "crypto_market", "symbol_specific"]

# Timestamp quality buckets for a raw timestamp value:
#   exact_utc    -> a verified PUBLISHER publish time, full date+time+UTC tz.
#   observed_utc -> an exact-UTC instant that is an OBSERVATION time (e.g. GDELT
#                   ``seendate``: when the article was first seen in the global
#                   stream, NOT a verified publisher publish time). It pins a real
#                   UTC second, so for leakage/window purposes it is as trustworthy
#                   as ``exact_utc`` and is treated identically; the distinct label
#                   exists only so provenance never mislabels it as a publish time.
#   date_only    -> a calendar date is known but the precise instant / tz is not.
#   invalid      -> not even a date can be parsed.
TS_EXACT = "exact_utc"
TS_OBSERVED = "observed_utc"
TS_DATE_ONLY = "date_only"
TS_INVALID = "invalid"

# Qualities that pin a real UTC instant and may therefore enter the intraday
# rolling window / availability proxy. NOTE: adding ``observed_utc`` here does NOT
# relax the as-of safety rule -- the same ``available_at <= cutoff`` test applies;
# it only recognises that an observation instant is a valid availability anchor.
TS_INTRADAY_QUALITIES = (TS_EXACT, TS_OBSERVED)

_TZ_SUFFIX_RE = re.compile(r"(?:[+-]\d{2}:?\d{2}|Z)$")


def classify_timestamp_quality(value) -> str:
    """Classify how reliably a raw ``published_at`` value pins a UTC instant.

    A news item only enters the intraday as-of feature when this returns
    ``exact_utc`` (full date+time AND an explicit timezone, e.g. trailing ``Z``
    or ``+00:00``). Date-only or timezone-ambiguous values are ``date_only`` and
    must NOT be assumed to occur at 00:00 UTC for intraday purposes; unparseable
    values are ``invalid``.
    """
    if value is None:
        return TS_INVALID
    if isinstance(value, (pd.Timestamp, datetime)):
        # An already-parsed object is exact only when it carries a tz.
        return TS_EXACT if getattr(value, "tzinfo", None) is not None else TS_DATE_ONLY
    s = str(value).strip()
    if not s:
        return TS_INVALID
    parsed = pd.to_datetime(s, utc=True, errors="coerce")
    if parsed is pd.NaT or pd.isna(parsed):
        return TS_INVALID
    has_time = (("T" in s) or (" " in s)) and (":" in s)
    has_tz = bool(_TZ_SUFFIX_RE.search(s)) or s.upper().endswith("UTC")
    if has_time and has_tz:
        return TS_EXACT
    return TS_DATE_ONLY

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
        # Classify the RAW value before it is coerced so we can tell an exact UTC
        # instant from a date-only / tz-ambiguous one (drives intraday vs fallback).
        if "timestamp_quality" not in df.columns:
            df["timestamp_quality"] = df["timestamp"].map(classify_timestamp_quality)
        # ``format="mixed"`` parses each value independently so a date-only string
        # ("2026-06-20") next to an exact instant ("2026-06-20T11:50:00Z") does not
        # poison the whole column into NaT.
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], utc=True, errors="coerce", format="mixed"
        )
    elif "timestamp_quality" not in df.columns:
        df["timestamp_quality"] = TS_INVALID
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


def _aggregate_window(
    window: pd.DataFrame,
    target: str,
    universe: List[str],
    weight_fn,
) -> Dict[str, float]:
    """Cross-coin correlation-weighted aggregation over an already-sliced window.

    Shared by the daily and the intraday as-of builders so both produce byte-for-
    byte identical feature columns from the same window of rows.
    """
    row: Dict[str, float] = {}
    for cat in CATEGORIES:
        row[f"{cat}_news_sentiment"] = 0.0
        row[f"{cat}_news_count"] = 0
    for agg in WEIGHTED_NEWS_AGGREGATES:
        row[agg] = 0.0
    for src in universe:
        row[f"news_sentiment_from_{src}_weighted"] = 0.0

    if window is None or window.empty:
        return row

    for cat in CATEGORIES:
        if cat == "symbol_specific":
            sub = window[(window["category"] == cat) & (window["symbol"] == target)]
        else:
            sub = window[window["category"] == cat]
        row[f"{cat}_news_count"] = int(len(sub))
        row[f"{cat}_news_sentiment"] = float(sub["score"].mean()) if len(sub) else 0.0

    stats = {src: _symbol_specific_stats(window, src) for src in universe}
    num = den = w_count = w_pos = w_neg = 0.0
    for src in universe:
        w = float(weight_fn(src))
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
    return row


def asof_features_for_symbol(
    news: pd.DataFrame,
    target: str,
    universe: List[str],
    asof: pd.Timestamp,
    corr_provider=None,
    lookback_hours: int = 24,
    safety_lag_seconds: int = 300,
) -> Dict[str, float]:
    """As-of news feature vector for ``target`` at the instant ``asof`` (UTC).

    Strict leakage + rolling-window rule (identical for live inference and any
    per-minute training row):

    ``window_start < published_at <= asof - safety_lag_seconds`` where
    ``window_start = asof - lookback_hours``.

    The window is a true rolling 24h lookback from the decision instant; it does
    NOT reset on a UTC day boundary, and news published after the (lagged)
    cutoff -- including later in the same day -- can never enter an earlier row.
    """
    asof = pd.Timestamp(asof)
    if asof.tzinfo is None:
        asof = asof.tz_localize("utc")
    else:
        asof = asof.tz_convert("utc")

    cutoff = asof - pd.Timedelta(seconds=int(safety_lag_seconds))
    window_start = asof - pd.Timedelta(hours=int(lookback_hours))
    # Correlation weights are keyed by the calendar date of the lagged cutoff.
    date_str = str(cutoff.date())

    def weight(source: str) -> float:
        if corr_provider is not None:
            return float(corr_provider.weight(date_str, target, source))
        return 1.0 if source == target else 0.0

    if news is None or news.empty:
        return _aggregate_window(news, target, universe, weight)

    # Intraday as-of uses ONLY instant-pinned news (verified publish OR observation
    # time): a date-only / tz-ambiguous item has no trustworthy instant, so it must
    # not be placed inside the rolling window.
    if "timestamp_quality" in news.columns:
        news = news[news["timestamp_quality"].isin(TS_INTRADAY_QUALITIES)]
    if news.empty:
        return _aggregate_window(news, target, universe, weight)

    ts = news["timestamp"]
    mask = (ts > window_start) & (ts <= cutoff)
    window = news.loc[mask]
    return _aggregate_window(window, target, universe, weight)


def asof_news_provider(
    news: pd.DataFrame,
    symbol: str,
    universe: List[str],
    corr_provider=None,
    lookback_hours: int = 24,
    safety_lag_seconds: int = 300,
):
    """Return ``provider(ts) -> {news feature: value}`` for one symbol.

    This is the SINGLE join used identically by training (per-minute rows) and by
    live inference: given a decision instant ``ts`` it returns the rolling
    ``lookback_hours`` news as-of ``ts - safety_lag_seconds``. The safety lag is
    applied to the lookup instant (not double-counted in the window), exactly
    mirroring the Rust engine which selects the newest artifact record with
    ``asof_timestamp <= ts - safety_lag_seconds``.
    """

    def provider(ts) -> Dict[str, float]:
        asof = pd.Timestamp(ts)
        if asof.tzinfo is None:
            asof = asof.tz_localize("utc")
        else:
            asof = asof.tz_convert("utc")
        effective = asof - pd.Timedelta(seconds=int(safety_lag_seconds))
        return asof_features_for_symbol(
            news,
            symbol,
            universe,
            effective,
            corr_provider=corr_provider,
            lookback_hours=lookback_hours,
            safety_lag_seconds=0,
        )

    return provider


def load_daily_ctx(path: str, symbol: str) -> Dict[str, Dict[str, float]]:
    """Load the final daily news artifact as ``{date: news features}`` for one
    symbol. Used as the previous-completed-day fallback source; the daily builder
    only emits parseable-timestamp news so this is already quality-verified."""
    ctx: Dict[str, Dict[str, float]] = {}
    if not path or not os.path.exists(path):
        return ctx
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("symbol") != symbol:
                continue
            date = row.get("date")
            if not date:
                continue
            ctx[str(date)] = {k: v for k, v in row.items() if k not in ("date", "symbol")}
    return ctx


def previous_completed_day_features(
    daily_ctx: Dict[str, Dict[str, float]], decision_ts
) -> Tuple[Optional[str], Dict[str, float]]:
    """``(source_date, feats)`` for the greatest daily date strictly before the
    decision's UTC calendar day.

    Never assumes a publish time and never returns a same-day or future row, so
    date-only / after-decision news can never leak into the fallback.
    """
    if not daily_ctx:
        return None, {}
    d = pd.Timestamp(decision_ts)
    d = d.tz_convert("utc") if d.tzinfo is not None else d.tz_localize("utc")
    ddate = d.date()
    chosen: Optional[str] = None
    for ds in sorted(daily_ctx.keys()):
        try:
            cur = pd.Timestamp(ds).date()
        except (ValueError, TypeError):
            continue
        if cur < ddate:
            chosen = ds
        else:
            break
    if chosen is None:
        return None, {}
    return chosen, dict(daily_ctx[chosen])


def _iso_or_none(ts) -> Optional[str]:
    """Format a timestamp as canonical UTC ISO-8601, or None when missing."""
    if ts is None:
        return None
    try:
        if pd.isna(ts):
            return None
    except (TypeError, ValueError):
        return None
    ts = pd.Timestamp(ts)
    ts = ts.tz_convert("utc") if ts.tzinfo is not None else ts.tz_localize("utc")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_available_at(
    news: pd.DataFrame,
    *,
    live: bool,
    historical_lag_seconds: int,
) -> pd.Series:
    """Per-row ``available_at`` (UTC): when the SCORED news is usable by the model.

    ``NaT`` marks a row that cannot enter an intraday window (date-only / invalid
    with no real availability time), routing it to the daily fallback instead.

    * ``live=True``  -> real ingestion/scoring completion time (``scored_at``,
      else ``fetched_at``); an exact-UTC ``published_at`` is proxied only when no
      real time was recorded. A date-only item with a real availability time is
      still usable live.
    * ``live=False`` (historical/backtest) -> conservative proxy
      ``published_at + historical_lag`` for exact-UTC rows ONLY; any recorded
      fetch/score time is ignored (a bulk historical fetch is not a realistic
      availability instant), so date-only/invalid rows are never intraday.
    """
    idx = news.index
    empty = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]")
    if news.empty:
        return empty
    published = (
        news["timestamp"]
        if "timestamp" in news.columns
        else empty
    )
    if "timestamp_quality" in news.columns:
        quality = news["timestamp_quality"]
    else:
        quality = pd.Series(TS_EXACT, index=idx)
    # Both a verified publish instant (exact_utc) and an observation instant
    # (observed_utc, e.g. GDELT seendate) anchor a real UTC second, so the same
    # conservative historical proxy applies to both.
    is_exact = quality.isin(TS_INTRADAY_QUALITIES)
    proxy = published + pd.Timedelta(seconds=int(historical_lag_seconds))

    avail = empty.copy()
    if live:
        for col in ("scored_at", "fetched_at"):
            if col in news.columns:
                real = pd.to_datetime(news[col], utc=True, errors="coerce")
                avail = avail.where(avail.notna(), real)
        need_proxy = avail.isna() & is_exact
        avail = avail.mask(need_proxy, proxy)
    else:
        avail = avail.mask(is_exact, proxy)
    return avail


def _provenance(mode, quality, source_date, published, available, source=None) -> Dict[str, object]:
    return {
        "published_at": published,
        "available_at": available,
        "news_mode": mode,
        "timestamp_quality": quality,
        "source_feature_date": source_date,
        "news_source": source,
    }


# Numeric coverage/provenance features (added to the model schema only when a
# ``news_source`` is supplied, e.g. for the GDELT provider). They never replace
# the sentiment features: ``gdelt_tone`` is the raw source tone, kept distinct
# from FinBERT sentiment.
COVERAGE_FEATURE_KEYS = ["news_available", "news_coverage_ratio", "gdelt_tone"]


def _window_mean_tone(window: Optional[pd.DataFrame]) -> float:
    if window is None or window.empty or "gdelt_tone" not in window.columns:
        return 0.0
    tone = pd.to_numeric(window["gdelt_tone"], errors="coerce").dropna()
    return float(tone.mean()) if not tone.empty else 0.0


def _window_quality(window: Optional[pd.DataFrame]) -> str:
    """Representative timestamp_quality of an as-of window for provenance.

    A verified publish instant dominates over an observation instant; with neither
    present (or no quality column) we fall back to ``exact_utc``. This is provenance
    only and never changes which rows entered the window.
    """
    if window is None or window.empty or "timestamp_quality" not in window.columns:
        return TS_EXACT
    qualities = set(window["timestamp_quality"].dropna().tolist())
    if TS_EXACT in qualities:
        return TS_EXACT
    if TS_OBSERVED in qualities:
        return TS_OBSERVED
    return TS_EXACT


def _coverage_features(
    available: float, coverage_ratio: Optional[float], tone: float
) -> Dict[str, float]:
    return {
        "news_available": float(available),
        "news_coverage_ratio": float(1.0 if coverage_ratio is None else coverage_ratio),
        "gdelt_tone": float(tone),
    }


def asof_news_for_decision(
    news: pd.DataFrame,
    symbol: str,
    universe: List[str],
    decision_ts,
    corr_provider=None,
    lookback_hours: int = 24,
    safety_lag_seconds: int = 300,
    daily_ctx: Optional[Dict[str, Dict[str, float]]] = None,
    live: bool = False,
    historical_lag_seconds: int = 300,
    news_source: Optional[str] = None,
    coverage_ratio: Optional[float] = None,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Unified live + training news join keyed on ``available_at``.

    The leakage rule is ``available_at <= decision_ts - safety_lag_seconds`` where
    ``available_at`` is when the bot actually had the scored news (NOT
    ``published_at``). The cutoff is ALWAYS the current decision time, never the
    trade entry time: a headline that became available after a position opened but
    before this close decision is admissible.

    Returns ``(features, provenance)`` with provenance
    ``{published_at, available_at, news_mode, timestamp_quality, source_feature_date}``:

    * news available within the rolling window -> ``intraday_asof``.
    * else, if date-only/tz-ambiguous news is relevant to the window ->
      ``previous_completed_day_fallback`` using the prior completed UTC day's
      daily feature (never a 00:00 assumption, never future/after-decision news).
    * genuinely no usable news -> ``intraday_asof`` zeros.
    """
    decision = pd.Timestamp(decision_ts)
    decision = (
        decision.tz_convert("utc") if decision.tzinfo is not None else decision.tz_localize("utc")
    )
    cutoff = decision - pd.Timedelta(seconds=int(safety_lag_seconds))
    window_start = cutoff - pd.Timedelta(hours=int(lookback_hours))
    cutoff_iso = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    date_str = str(cutoff.date())

    def weight(source: str) -> float:
        if corr_provider is not None:
            return float(corr_provider.weight(date_str, symbol, source))
        return 1.0 if source == symbol else 0.0

    def finalize(feats, available_flag, tone, provenance):
        # Coverage/provenance features are only added to the model schema when a
        # provider (news_source) is declared, so the no-news / legacy paths keep
        # their exact feature set.
        if news_source is not None:
            feats = dict(feats)
            feats.update(_coverage_features(available_flag, coverage_ratio, tone))
        return feats, provenance

    if news is None or news.empty:
        feats = _aggregate_window(None, symbol, universe, weight)
        return finalize(
            feats, 0.0, 0.0,
            _provenance("intraday_asof", TS_EXACT, None, None, cutoff_iso, news_source),
        )

    avail = compute_available_at(
        news, live=live, historical_lag_seconds=historical_lag_seconds
    )
    in_window = avail.notna() & (avail > window_start) & (avail <= cutoff)
    window = news.loc[in_window]
    if not window.empty:
        feats = _aggregate_window(window, symbol, universe, weight)
        pub = window["timestamp"].max() if "timestamp" in window.columns else None
        available = _iso_or_none(avail.loc[in_window].max()) or cutoff_iso
        return finalize(
            feats, 1.0, _window_mean_tone(window),
            _provenance(
                "intraday_asof", _window_quality(window), None,
                _iso_or_none(pub), available, news_source,
            ),
        )

    # No intraday-usable news -> previous-completed-day fallback when date-only
    # (timestamp-ambiguous) news is relevant to the window.
    has_date_only = False
    if "timestamp_quality" in news.columns:
        date_only = news[news["timestamp_quality"] == TS_DATE_ONLY]
        date_only = date_only[date_only["timestamp"].notna()]
        if not date_only.empty:
            dd = date_only["timestamp"].dt.date
            has_date_only = any(window_start.date() <= d <= cutoff.date() for d in dd)

    if has_date_only:
        src, dfeats = previous_completed_day_features(daily_ctx or {}, decision)
        base = _aggregate_window(None, symbol, list(universe), lambda s: 0.0)
        for key in base:
            if key in dfeats:
                base[key] = dfeats[key]
        return finalize(
            base, 0.0, 0.0,
            _provenance(
                "previous_completed_day_fallback", TS_DATE_ONLY, src, None, cutoff_iso, news_source
            ),
        )

    feats = _aggregate_window(None, symbol, universe, weight)
    return finalize(
        feats, 0.0, 0.0,
        _provenance("intraday_asof", TS_EXACT, None, None, cutoff_iso, news_source),
    )


def write_jsonl(rows: Iterable[Dict[str, float]], output_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return output_path
