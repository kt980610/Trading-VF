"""CLI: fetch real, date-stamped news into the canonical raw store.

Examples::

    # Backfill a fixed range.
    python -m src.main_fetch_news \
        --config config/distribution_config.yaml \
        --start 2022-01-01 --end 2026-06-19 --provider newsapi

    # Incremental update (resume after the last fully-covered day -> yesterday).
    python -m src.main_fetch_news \
        --config config/distribution_config.yaml \
        --incremental --provider newsapi

The API key is read ONLY from the environment variable named by
``news.api_key_env`` (default ``NEWSAPI_API_KEY``). It is never written to disk,
logs, manifests or artifacts. This tool is offline/operational only; the live
bot never calls it.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import load_config
    from src import news_ingest as ni
    from src.news_provider import NewsProviderError, get_provider, provider_requires_api_key
else:
    from .config import load_config
    from . import news_ingest as ni
    from .news_provider import NewsProviderError, get_provider, provider_requires_api_key


def _parse_day(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _check_doc_history_window(news, provider_name: str, start_day: date, last_day: date) -> None:
    """Fail fast if a GDELT-DOC fetch reaches further back than DOC reliably covers.

    DOC 2.0 is a recent/live source; older training backfill needs a dedicated
    historical provider (GKG/bulk or BigQuery), not DOC.
    """
    if provider_name.strip().lower() != "gdelt":
        return
    earliest = last_day - timedelta(days=int(news.gdelt_doc_max_lookback_days))
    if start_day < earliest:
        raise SystemExit(
            "unsupported_history_window: our safety policy treats GDELT DOC 2.0 as "
            f"recent/live only (trailing {news.gdelt_doc_max_lookback_days} days, "
            f">= {earliest.isoformat()}), but requested start={start_day.isoformat()}. "
            "This is a self-imposed guardrail, not an official GDELT cutoff; use a "
            "dedicated historical provider (GDELT GKG/bulk archive or BigQuery) for "
            "older dates."
        )


def _make_provider(news, provider_name: str, api_key: str, transport=None):
    provider_kwargs: dict = {}
    if provider_name.strip().lower() == "gdelt":
        if news.gdelt_base_url:
            provider_kwargs["base_url"] = news.gdelt_base_url
        provider_kwargs["max_records"] = news.gdelt_max_records
        provider_kwargs["min_request_interval"] = news.gdelt_min_request_interval_seconds
    return get_provider(provider_name, api_key, transport=transport, **provider_kwargs)


def run_probe(
    config_path: str,
    query: str = "bitcoin",
    day: str = "2026-06-19",
    provider_name: str = None,
    transport=None,
) -> dict:
    """Single query, single day, single HTTP call. Gates a real fetch: returns a
    classification dict and writes one safe manifest entry. Does NOT bisect or
    retry, and does NOT run a day-level fetch."""
    config = load_config(config_path)
    news = config.news
    provider_name = provider_name or news.provider

    api_key = os.environ.get(news.api_key_env, "").strip()
    if provider_requires_api_key(provider_name) and not api_key:
        raise SystemExit(
            f"missing API key: set environment variable {news.api_key_env} "
            f"(never store the key in config)"
        )

    manifest_path = config.resolve(news.coverage_manifest_path)
    provider = _make_provider(news, provider_name, api_key, transport)
    spec = ni.QuerySpec("crypto_market", None, query)
    probe_day = _parse_day(day)

    print(f"probing {provider_name}: query={query!r} day={probe_day} (single HTTP call)")
    result = ni.probe_cell(provider, spec, probe_day, manifest_path, language=news.language)
    print(
        "probe result: class={response_class} http_status={http_status} "
        "content_type={content_type} articles={article_count} "
        "retry_after={retry_after_seconds}".format(**result)
    )
    if result.get("response_excerpt"):
        print(f"response excerpt: {result['response_excerpt']}")
    print(f"coverage manifest -> {result['manifest_path']}")
    if result.get("ok"):
        print("PROBE OK: safe to run a day-level fetch.")
    else:
        print("PROBE FAILED: do NOT run a day-level fetch until this is resolved.")
    return result


def run(
    config_path: str,
    start: str = None,
    end: str = None,
    provider_name: str = None,
    incremental: bool = False,
    transport=None,
    now: datetime = None,
) -> dict:
    config = load_config(config_path)
    news = config.news

    provider_name = provider_name or news.provider

    # GDELT is free/keyless; only key-based providers (e.g. the smoke-test-only
    # NewsAPI adapter) require an environment-provided API key.
    api_key = os.environ.get(news.api_key_env, "").strip()
    if provider_requires_api_key(provider_name) and not api_key:
        raise SystemExit(
            f"missing API key: set environment variable {news.api_key_env} "
            f"(never store the key in config)"
        )

    raw_path = config.resolve(news.raw_path)
    manifest_path = config.resolve(news.coverage_manifest_path)

    category_queries = news.queries or None
    symbol_queries = news.symbol_queries or None
    specs = ni.build_query_specs(config.symbols, category_queries, symbol_queries)

    backfill_start = _parse_day(news.backfill_start)
    last_day = ni.last_completed_utc_day(now)

    if incremental:
        manifest = ni.load_manifest(manifest_path)
        start_day = ni.resume_day(manifest, backfill_start)
        end_day = last_day
    else:
        start_day = _parse_day(start) if start else backfill_start
        end_day = _parse_day(end) if end else last_day

    if end_day < start_day:
        print(f"nothing to fetch: start={start_day} > end={end_day} (already up to date)")
        return {"days": 0, "ok_cells": 0, "failed_cells": 0, "records_written": 0}

    _check_doc_history_window(news, provider_name, start_day, last_day)

    provider = _make_provider(news, provider_name, api_key, transport)

    print(
        f"fetching {provider_name} news {start_day}..{end_day} "
        f"({len(specs)} queries/day, language={news.language})"
    )
    summary = ni.fetch_range(
        provider,
        specs,
        start_day,
        end_day,
        raw_path=raw_path,
        manifest_path=manifest_path,
        language=news.language,
    )
    print(
        "done: days={days} ok_cells={ok_cells} failed_cells={failed_cells} "
        "records={records_written} dropped_no_ts={dropped_no_timestamp}".format(**summary)
    )
    print(f"raw partitions -> {summary['partition_dir']}")
    print(f"coverage manifest -> {summary['manifest_path']}")
    if summary["failed_cells"]:
        print(
            f"WARNING: {summary['failed_cells']} (day,query) cells failed; "
            f"strict training will fail until they are refetched."
        )
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch real dated news (offline tool).")
    parser.add_argument("--config", default="config/distribution_config.yaml")
    parser.add_argument("--start", default=None, help="UTC start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="UTC end date YYYY-MM-DD")
    parser.add_argument("--provider", default=None, help="override configured provider")
    parser.add_argument("--incremental", action="store_true", help="resume from manifest")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="single query/day/HTTP-call diagnostic probe; gates a real fetch",
    )
    parser.add_argument("--probe-query", default="bitcoin", help="probe query (default: bitcoin)")
    parser.add_argument(
        "--probe-date", default="2026-06-19", help="probe UTC day YYYY-MM-DD (default: 2026-06-19)"
    )
    args = parser.parse_args(argv)
    if args.probe:
        result = run_probe(
            args.config,
            query=args.probe_query,
            day=args.probe_date,
            provider_name=args.provider,
        )
        return 0 if result.get("ok") else 1
    run(
        args.config,
        start=args.start,
        end=args.end,
        provider_name=args.provider,
        incremental=args.incremental,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
