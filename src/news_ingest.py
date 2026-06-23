"""Historical + live news ingestion into a canonical, immutable raw store.

Pipeline:

1. Build a configurable set of (category, symbol, query) specs.
2. For each UTC day and each query, ask the provider for that day's articles.
3. Normalise to the canonical schema, drop articles with no publish timestamp,
   dedupe, and write a per-day immutable JSONL partition.
4. Record a coverage manifest line for every (day, query): ``ok`` or ``failed``.
   A failed day/query is NEVER counted as "0 news".

The publish timestamp (``timestamp``) always comes from the provider's publish
time, normalised to UTC. ``fetched_at`` is recorded separately and is never used
as a feature time.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Default query set (spec section 4). Overridable via config.
# ---------------------------------------------------------------------------
DEFAULT_CATEGORY_QUERIES: Dict[str, List[str]] = {
    "macro": ["FOMC", "Federal Reserve", "CPI", "inflation", "jobs report", "recession"],
    "policy": ["crypto regulation", "SEC crypto", "stablecoin regulation", "MiCA"],
    "stock_market": ["S&P 500", "Nasdaq", "equities", "risk assets"],
    "crypto_market": ["bitcoin", "ethereum", "cryptocurrency", "crypto market"],
}

DEFAULT_SYMBOL_QUERIES: Dict[str, str] = {
    "BTCUSDT": "Bitcoin",
    "ETHUSDT": "Ethereum",
    "SOLUSDT": "Solana",
    "ADAUSDT": "Cardano",
    "XRPUSDT": "XRP OR Ripple",
    "BNBUSDT": "BNB OR Binance Coin",
    "NEOUSDT": "NEO cryptocurrency",
    "LINKUSDT": "Chainlink",
}

CANONICAL_FIELDS = [
    "timestamp",
    "published_at",
    "source_seen_at",
    "available_at",
    "source",
    "provider",
    "provider_id",
    "url",
    "title",
    "body",
    "query",
    "category",
    "symbol",
    "timestamp_quality",
    "gdelt_tone",
    "fetched_at",
]

COVERAGE_OK = "ok"
COVERAGE_FAILED = "failed"


@dataclass(frozen=True)
class QuerySpec:
    category: str
    symbol: Optional[str]
    query: str

    def label(self) -> str:
        return f"{self.category}:{self.symbol or '*'}:{self.query}"


def build_query_specs(
    symbols: Iterable[str],
    category_queries: Optional[Dict[str, List[str]]] = None,
    symbol_queries: Optional[Dict[str, str]] = None,
) -> List[QuerySpec]:
    """Expand configured queries into a flat, ordered list of specs."""
    category_queries = category_queries or DEFAULT_CATEGORY_QUERIES
    symbol_queries = symbol_queries or DEFAULT_SYMBOL_QUERIES

    specs: List[QuerySpec] = []
    for category, queries in category_queries.items():
        for q in queries:
            specs.append(QuerySpec(category=category, symbol=None, query=str(q)))
    for symbol in symbols:
        q = symbol_queries.get(symbol)
        if q:
            specs.append(QuerySpec(category="symbol_specific", symbol=symbol, query=str(q)))
    return specs


# ---------------------------------------------------------------------------
# Timestamp + identity helpers.
# ---------------------------------------------------------------------------
def normalize_timestamp(raw) -> Optional[str]:
    """Parse a provider publish timestamp to canonical UTC ISO-8601.

    Returns ``None`` for missing/unparseable timestamps so callers can drop and
    count the record (a publish timestamp is mandatory).
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    dt: Optional[datetime] = None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        try:
            import pandas as pd  # local import keeps this module light

            ts = pd.to_datetime(s, utc=True, errors="coerce")
            if pd.isna(ts):
                return None
            dt = ts.to_pydatetime()
        except Exception:  # noqa: BLE001
            return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def stable_provider_id(article: dict) -> str:
    """Stable id for dedupe: hash of the article URL, else title+publishedAt."""
    url = article.get("url")
    if url:
        return "u:" + hashlib.sha256(str(url).encode("utf-8")).hexdigest()[:24]
    base = f"{article.get('title', '')}|{article.get('publishedAt', '')}"
    return "h:" + hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]


def canonical_record(
    article: dict, spec: QuerySpec, provider_name: str, fetched_at: str
) -> Optional[dict]:
    """Convert a raw provider article to a canonical record, or None if no instant.

    Two distinct source instants are kept separate and never conflated:

    * ``published_at``  -- a VERIFIED publisher publish time (e.g. NewsAPI
      ``publishedAt``). Quality is classified from the raw string
      (``exact_utc`` / ``date_only`` / ``invalid``).
    * ``source_seen_at`` -- an OBSERVATION time (e.g. GDELT ``seendate``: when the
      article was first seen in the global stream). It is NOT a publish time, so it
      is labelled ``observed_utc``.

    ``timestamp`` is the single instant used for as-of windowing: the verified
    publish time when present, otherwise the observation time. The leakage rule is
    unchanged; only the provenance labelling is made precise.
    """
    from .news_features import (
        TS_OBSERVED,
        classify_timestamp_quality,
    )

    # Verified publisher publish time. Legacy providers pass it via ``publishedAt``.
    raw_published = article.get("published_at")
    if raw_published is None:
        raw_published = article.get("publishedAt")
    # Observation time (GDELT seendate); ``source_seen_at`` is the canonical key.
    raw_observed = article.get("source_seen_at")
    if raw_observed is None:
        raw_observed = article.get("observed_at")

    published_at = normalize_timestamp(raw_published) if raw_published else None
    source_seen_at = normalize_timestamp(raw_observed) if raw_observed else None

    if published_at is not None:
        # A verified publish time wins as the windowing instant; its quality is
        # classified from the raw string so date-only/tz-ambiguous is preserved.
        ts = published_at
        quality = classify_timestamp_quality(raw_published)
    elif source_seen_at is not None:
        # Only an observation instant is available -> windowing uses it, labelled
        # observed_utc so it is never mistaken for a verified publish time.
        ts = source_seen_at
        quality = TS_OBSERVED
    else:
        return None

    source = article.get("source") or {}
    source_name = source.get("name") if isinstance(source, dict) else None
    tone = article.get("gdelt_tone")
    return {
        "timestamp": ts,
        "published_at": published_at,
        "source_seen_at": source_seen_at,
        # ``available_at`` is stamped at scoring time (live) or proxied at feature
        # time (historical: source_seen_at/published_at + lag); raw record leaves
        # it unset.
        "available_at": None,
        "source": source_name or "",
        "provider": provider_name,
        "provider_id": stable_provider_id(article),
        "url": article.get("url") or "",
        "title": article.get("title") or "",
        # Only description/summary if the provider supplies it (never full body).
        "body": article.get("description") or "",
        "query": spec.query,
        "category": spec.category,
        "symbol": spec.symbol,
        "timestamp_quality": quality,
        # GDELT source tone (numeric) when present; kept distinct from FinBERT.
        "gdelt_tone": float(tone) if isinstance(tone, (int, float)) else None,
        "fetched_at": fetched_at,
    }


def content_dedupe_hash(title, body) -> str:
    """Stable hash of normalised ``title`` + ``body`` for content-level dedupe."""
    base = f"{(title or '').strip().lower()}\n{(body or '').strip().lower()}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]


def dedupe_records(records: Iterable[dict]) -> List[dict]:
    """Drop duplicates for the same (category, symbol), keeping first seen.

    A record is a duplicate when it shares a ``provider_id`` (URL/title hash) OR
    the same normalised title+body content hash with an already-kept record in the
    same (category, symbol) bucket. The content hash catches the same story
    syndicated under different URLs.
    """
    seen_ids = set()
    seen_content = set()
    out: List[dict] = []
    for rec in records:
        id_key = (rec.get("provider_id"), rec.get("category"), rec.get("symbol"))
        if id_key in seen_ids:
            continue
        content_key = None
        title = rec.get("title")
        body = rec.get("body")
        if (title or "").strip() or (body or "").strip():
            content_key = (
                content_dedupe_hash(title, body),
                rec.get("category"),
                rec.get("symbol"),
            )
            if content_key in seen_content:
                continue
        seen_ids.add(id_key)
        if content_key is not None:
            seen_content.add(content_key)
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Filesystem layout (date-partitioned immutable JSONL).
# ---------------------------------------------------------------------------
def partition_dir(raw_path: str) -> str:
    """Directory holding per-day partitions, derived from the configured path."""
    base, _ = os.path.splitext(raw_path)
    return base  # e.g. data/raw/news.jsonl -> data/raw/news/


def partition_path(raw_path: str, day: date) -> str:
    return os.path.join(partition_dir(raw_path), f"news_{day.isoformat()}.jsonl")


def _atomic_write_jsonl(rows: Iterable[dict], path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _read_jsonl(path: str) -> List[dict]:
    out: List[dict] = []
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def read_jsonl(path: str) -> List[dict]:
    """Public reader for a single JSONL file (one record per line)."""
    return _read_jsonl(path)


def write_jsonl_atomic(rows: Iterable[dict], path: str) -> None:
    """Public atomic JSONL writer."""
    _atomic_write_jsonl(rows, path)


def list_partition_files(raw_path: str) -> List[str]:
    """All per-day partition files plus the legacy single file if present."""
    files: List[str] = []
    pdir = partition_dir(raw_path)
    if os.path.isdir(pdir):
        for name in sorted(os.listdir(pdir)):
            if name.endswith(".jsonl"):
                files.append(os.path.join(pdir, name))
    if os.path.isfile(raw_path):
        files.append(raw_path)
    return files


def load_raw_records(raw_path: str) -> List[dict]:
    """Load all canonical records from partitions and the legacy single file."""
    records: List[dict] = []
    pdir = partition_dir(raw_path)
    if os.path.isdir(pdir):
        for name in sorted(os.listdir(pdir)):
            if name.endswith(".jsonl"):
                records.extend(_read_jsonl(os.path.join(pdir, name)))
    if os.path.isfile(raw_path):
        records.extend(_read_jsonl(raw_path))
    return dedupe_records(records)


# ---------------------------------------------------------------------------
# Coverage manifest.
# ---------------------------------------------------------------------------
def load_manifest(path: str) -> List[dict]:
    return _read_jsonl(path)


def manifest_key(entry: dict) -> Tuple[str, str, str, str]:
    return (
        str(entry.get("date")),
        str(entry.get("category")),
        str(entry.get("symbol")),
        str(entry.get("query")),
    )


def write_manifest(entries: Iterable[dict], path: str) -> None:
    _atomic_write_jsonl(entries, path)


def merge_manifest(existing: List[dict], new_entries: List[dict]) -> List[dict]:
    """Overlay new entries on existing, keyed by (date, category, symbol, query)."""
    index = {manifest_key(e): e for e in existing}
    for e in new_entries:
        index[manifest_key(e)] = e
    return sorted(index.values(), key=manifest_key)


class NewsValidationError(Exception):
    """Raised in strict mode when news coverage is insufficient for training."""


def _record_date(rec: dict) -> Optional[date]:
    ts = rec.get("timestamp")
    if not ts:
        return None
    try:
        return date.fromisoformat(str(ts)[:10])
    except ValueError:
        return None


def validate_for_training(
    records: List[dict],
    manifest: List[dict],
    start: date,
    end: date,
    require_scores: bool = False,
    lookback_days: int = 1,
) -> None:
    """Strict-mode gate. Raises :class:`NewsValidationError` on any deficiency."""
    if not records:
        raise NewsValidationError("no raw news records found (run main_fetch_news)")
    if not manifest:
        raise NewsValidationError("coverage manifest missing/empty (run main_fetch_news)")
    fails = coverage_failures(manifest, start, end)
    if fails:
        sample = ", ".join(sorted({f"{f.get('date')}/{f.get('query')}" for f in fails})[:5])
        raise NewsValidationError(
            f"{len(fails)} failed coverage cells in [{start}, {end}] (e.g. {sample}); "
            f"refetch before training (failed != zero news)"
        )
    # News for day D uses [D - lookback, D); pad the window start accordingly.
    window_start = start - timedelta(days=max(0, lookback_days))
    in_range = [
        r for r in records
        if (lambda d: d is not None and window_start <= d <= end)(_record_date(r))
    ]
    if not in_range:
        raise NewsValidationError(
            f"zero raw news within training range [{start}, {end}] "
            f"(refusing to train on all-zero news features)"
        )
    if require_scores:
        # Text-less records (no title/body, e.g. GDELT URL-only hits) are NEVER
        # scorable: their absent sentiment is expected and must not fail strict
        # training. Only records that carry real text need a score.
        def _scorable(r: dict) -> bool:
            return bool((str(r.get("title") or "").strip()) or (str(r.get("body") or "").strip()))

        unscored = [r for r in in_range if _scorable(r) and r.get("sentiment_score") is None]
        if unscored:
            raise NewsValidationError(
                f"{len(unscored)} unscored records in range "
                f"(run main_score_news --backend finbert)"
            )


def ingestion_report(
    records: List[dict],
    manifest: List[dict],
    start: date,
    end: date,
    enabled: bool,
    strict: bool,
) -> dict:
    """Summarise coverage/records for ``reports/news_ingestion_report.json``."""
    category_counts: Dict[str, int] = {}
    symbol_counts: Dict[str, int] = {}
    timestamps: List[str] = []
    for r in records:
        category_counts[str(r.get("category"))] = category_counts.get(str(r.get("category")), 0) + 1
        sym = r.get("symbol")
        if sym:
            symbol_counts[str(sym)] = symbol_counts.get(str(sym), 0) + 1
        if r.get("timestamp"):
            timestamps.append(str(r.get("timestamp")))

    covered_days = set()
    failed_cells = 0
    ok_cells = 0
    for e in manifest:
        try:
            d = date.fromisoformat(str(e.get("date")))
        except ValueError:
            continue
        if not (start <= d <= end):
            continue
        if e.get("coverage_status") == COVERAGE_OK:
            ok_cells += 1
            covered_days.add(d)
        else:
            failed_cells += 1
    all_days = set(day_range(start, end))
    missing_days = sorted(d.isoformat() for d in (all_days - covered_days))

    return {
        "enabled": enabled,
        "strict_coverage": strict,
        "range_start": start.isoformat(),
        "range_end": end.isoformat(),
        "raw_record_count": len(records),
        "category_counts": category_counts,
        "symbol_counts": symbol_counts,
        "min_timestamp": min(timestamps) if timestamps else None,
        "max_timestamp": max(timestamps) if timestamps else None,
        "coverage_ok_cells": ok_cells,
        "coverage_failed_cells": failed_cells,
        "missing_day_count": len(missing_days),
        "missing_days": missing_days[:50],
    }


def coverage_ratio_for_days(manifest: List[dict], days: Iterable[date]) -> Optional[float]:
    """Fraction of ``ok`` (day, query) cells over ``days``.

    Returns ``None`` when the manifest has no cells for those days (unknown
    coverage is NOT silently treated as full coverage)."""
    want = {d.isoformat() for d in days}
    ok = 0
    total = 0
    for e in manifest:
        if str(e.get("date")) in want:
            total += 1
            if e.get("coverage_status") == COVERAGE_OK:
                ok += 1
    if total == 0:
        return None
    return ok / total


def coverage_failures(manifest: List[dict], start: date, end: date) -> List[dict]:
    """Entries with coverage_status != ok whose date is within [start, end]."""
    out: List[dict] = []
    for e in manifest:
        try:
            d = date.fromisoformat(str(e.get("date")))
        except ValueError:
            continue
        if start <= d <= end and e.get("coverage_status") != COVERAGE_OK:
            out.append(e)
    return out


# ---------------------------------------------------------------------------
# Day helpers.
# ---------------------------------------------------------------------------
def last_completed_utc_day(now: Optional[datetime] = None) -> date:
    now = now or datetime.now(timezone.utc)
    return now.astimezone(timezone.utc).date() - timedelta(days=1)


def day_range(start: date, end: date) -> List[date]:
    if end < start:
        return []
    out = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def resume_day(manifest: List[dict], backfill_start: date) -> date:
    """First day to (re)fetch: the day after the last fully-ok day, else start."""
    ok_days = []
    for e in manifest:
        if e.get("coverage_status") == COVERAGE_OK:
            try:
                ok_days.append(date.fromisoformat(str(e.get("date"))))
            except ValueError:
                continue
    if not ok_days:
        return backfill_start
    return max(ok_days) + timedelta(days=1)


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------
# Structured manifest diagnostics. These NEVER carry the request URL, query
# string, API key or request headers -- only the server's own response is
# described, so a failed cell can be triaged without re-running.
MANIFEST_ERROR_FIELDS = (
    "error",
    "error_code",
    "http_status",
    "content_type",
    "response_excerpt",
    "retry_after_seconds",
)


def _record_error_fields(entry: dict, exc) -> None:
    entry["error"] = getattr(exc, "reason", None)
    entry["error_code"] = getattr(exc, "error_code", None)
    entry["http_status"] = getattr(exc, "http_status", None)
    entry["content_type"] = getattr(exc, "content_type", None)
    entry["response_excerpt"] = getattr(exc, "response_excerpt", None)
    entry["retry_after_seconds"] = getattr(exc, "retry_after_seconds", None)


def _clear_error_fields(entry: dict) -> None:
    for key in MANIFEST_ERROR_FIELDS:
        entry[key] = None


def probe_cell(
    provider,
    spec: "QuerySpec",
    day: date,
    manifest_path: str,
    language: str = "en",
    fetched_at: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Run a single-query, single-day, single-HTTP-call probe and record it.

    Writes one classified entry into the coverage manifest and returns the probe
    result (including ``ok``). Intended as a gate: do not run a day-level fetch
    unless the probe is ``ok``.
    """
    if not hasattr(provider, "probe"):
        raise ValueError(f"provider {getattr(provider, 'name', '?')} has no probe mode")

    fetched_at = fetched_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    provider_name = getattr(provider, "name", "unknown")
    result = provider.probe(spec.query, day, language=language)

    entry = {
        "date": day.isoformat(),
        "category": spec.category,
        "symbol": spec.symbol,
        "query": spec.query,
        "provider": provider_name,
        "fetched_at": fetched_at,
        "probe": True,
        "article_count": int(result.get("article_count", 0)),
    }
    if result.get("ok"):
        entry["coverage_status"] = COVERAGE_OK
        _clear_error_fields(entry)
    else:
        entry["coverage_status"] = COVERAGE_FAILED
        entry["error"] = result.get("response_class")
        entry["error_code"] = result.get("error_code")
        entry["http_status"] = result.get("http_status")
        entry["content_type"] = result.get("content_type")
        entry["response_excerpt"] = result.get("response_excerpt")
        entry["retry_after_seconds"] = result.get("retry_after_seconds")

    merged = merge_manifest(load_manifest(manifest_path), [entry])
    write_manifest(merged, manifest_path)
    log(
        f"PROBE {day} {spec.label()} class={result.get('response_class')} "
        f"http={result.get('http_status')} articles={result.get('article_count')}"
    )
    result["manifest_path"] = manifest_path
    return result


def fetch_range(
    provider,
    specs: List[QuerySpec],
    start: date,
    end: date,
    raw_path: str,
    manifest_path: str,
    language: str = "en",
    fetched_at: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Fetch [start, end] for every spec, writing partitions + coverage manifest.

    Returns a summary dict. Failed (day, query) pairs are recorded with
    ``coverage_status=failed`` and contribute zero records (but are NOT treated
    as a successful empty day).
    """
    from .news_provider import NewsProviderError

    fetched_at = fetched_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    provider_name = getattr(provider, "name", "unknown")

    new_manifest: List[dict] = []
    summary = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "days": 0,
        "ok_cells": 0,
        "failed_cells": 0,
        "records_written": 0,
        "dropped_no_timestamp": 0,
    }

    for day in day_range(start, end):
        summary["days"] += 1
        day_records: List[dict] = []
        for spec in specs:
            entry = {
                "date": day.isoformat(),
                "category": spec.category,
                "symbol": spec.symbol,
                "query": spec.query,
                "provider": provider_name,
                "fetched_at": fetched_at,
            }
            try:
                articles = provider.fetch_day(spec.query, day, language=language)
            except NewsProviderError as exc:
                entry["coverage_status"] = COVERAGE_FAILED
                entry["article_count"] = 0
                _record_error_fields(entry, exc)
                summary["failed_cells"] += 1
                new_manifest.append(entry)
                log(
                    f"FAILED {day} {spec.label()} reason={exc.reason} "
                    f"code={exc.error_code} http={exc.http_status}"
                )
                continue

            kept = 0
            dropped = 0
            for art in articles:
                rec = canonical_record(art, spec, provider_name, fetched_at)
                if rec is None:
                    dropped += 1
                    continue
                day_records.append(rec)
                kept += 1
            summary["dropped_no_timestamp"] += dropped
            entry["coverage_status"] = COVERAGE_OK
            entry["article_count"] = kept
            _clear_error_fields(entry)
            summary["ok_cells"] += 1
            new_manifest.append(entry)

        deduped = dedupe_records(day_records)
        _atomic_write_jsonl(deduped, partition_path(raw_path, day))
        summary["records_written"] += len(deduped)

    merged = merge_manifest(load_manifest(manifest_path), new_manifest)
    write_manifest(merged, manifest_path)
    summary["manifest_path"] = manifest_path
    summary["partition_dir"] = partition_dir(raw_path)
    return summary
