"""CLI: build leakage-safe daily news features for every symbol (section 13).

Strict coverage (``news.enabled`` and ``news.strict_coverage``) refuses to
produce all-zero news features: missing raw news, a missing/failed coverage
manifest, or zero raw news across the training range is a hard error. When
``news.enabled`` is false, any existing news feature output is removed so no news
columns leak into the training schema.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import load_config
    from src.data_loader import DataSourceError, load_daily_ohlcv
    from src import news_features as nf
    from src import news_ingest as ni
    from src import correlation as corr_mod
else:
    from .config import load_config
    from .data_loader import DataSourceError, load_daily_ohlcv
    from . import news_features as nf
    from . import news_ingest as ni
    from . import correlation as corr_mod


def _write_report(path: str, report: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)


def _load_scored_news(config):
    """Load + score the raw canonical news as a prepared DataFrame, or None."""
    news = config.news
    raw_path = config.resolve(news.raw_path)
    records = ni.load_raw_records(raw_path)
    if not records:
        return None, []
    df = pd.DataFrame(records)
    news_df = nf.prepare_news(df, backend=news.sentiment_backend)
    return news_df, records


def run_intraday(config_path: str, asof: str | None = None) -> str:
    """Build the as-of intraday news feature artifact (one record per symbol).

    Strict: when the live worker produced no raw news at all we REFUSE to emit an
    all-zero artifact (no fake "0 news"). A genuinely empty rolling window for an
    individual symbol is fine and yields explicit zeros for that symbol only.
    The file is written atomically so the Rust engine never reads a half-written
    artifact.
    """
    config = load_config(config_path)
    news = config.news
    out_path = config.resolve(news.intraday_output_path)

    if not news.enabled:
        if os.path.isfile(out_path):
            os.remove(out_path)
        print("news disabled: removed intraday news artifact")
        return out_path

    asof_ts = pd.Timestamp(asof) if asof else pd.Timestamp.now(tz="UTC")
    if asof_ts.tzinfo is None:
        asof_ts = asof_ts.tz_localize("utc")
    else:
        asof_ts = asof_ts.tz_convert("utc")

    news_df, records = _load_scored_news(config)
    if news.strict_coverage and not records:
        raise ni.NewsValidationError(
            "no raw news available for intraday build; refusing to write all-zero "
            "intraday artifact (run main_fetch_news/main_score_news first)"
        )
    if news_df is None:
        news_df = nf.prepare_news(pd.DataFrame(columns=ni.CANONICAL_FIELDS))

    corr_provider = corr_mod.CorrelationProvider.load(
        config.resolve(config.correlation.output_path),
        fallback_self_corr=config.correlation.fallback_self_corr,
        fallback_cross_corr=config.correlation.fallback_cross_corr,
    )
    universe = list(config.symbols)
    asof_iso = asof_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    # Previous-completed-day fallback comes from the final daily artifact (already
    # quality-verified: the daily builder drops unparseable timestamps).
    daily_path = config.resolve(news.output_path)

    # Coverage ratio over the UTC day(s) the rolling 24h window can touch, read
    # from the ingestion coverage manifest (unknown coverage -> 1.0 so an absent
    # manifest does not penalise a smoke run; strict gating lives in training).
    from datetime import timedelta as _td

    manifest = ni.load_manifest(config.resolve(news.coverage_manifest_path))
    window_days = [(asof_ts - _td(days=k)).date() for k in (0, 1)]
    coverage_ratio = ni.coverage_ratio_for_days(manifest, window_days)
    if coverage_ratio is None:
        coverage_ratio = 1.0

    rows = []
    for symbol in universe:
        # The snapshot is the pure rolling-window news AS OF its run time
        # (no content lag): the safety lag is enforced at CONSUMPTION time, both
        # by the Rust engine (select newest asof <= decision - lag) and by the
        # training join (asof = t - lag). This keeps a single, non-double-counted
        # lag and makes the live/train join semantics identical. Date-only /
        # tz-ambiguous news is excluded from the as-of window and instead routed
        # to the previous-completed-UTC-day fallback with explicit provenance.
        daily_ctx = nf.load_daily_ctx(daily_path, symbol)
        feats, provenance = nf.asof_news_for_decision(
            news_df,
            symbol,
            universe,
            asof_ts,
            corr_provider=corr_provider,
            lookback_hours=news.feature_lookback_hours,
            safety_lag_seconds=0,
            daily_ctx=daily_ctx,
            live=True,
            historical_lag_seconds=news.historical_news_availability_lag_seconds,
            news_source=news.provider,
            coverage_ratio=coverage_ratio,
        )
        # The representative window instant is a verified publish time only when
        # the provenance quality says so; for an observation (observed_utc, e.g.
        # GDELT) it is recorded as source_seen_at and published_at stays null so
        # the artifact never mislabels an observation as a publish time.
        inst = provenance["published_at"]
        observed = provenance["timestamp_quality"] == nf.TS_OBSERVED
        record = {
            "asof_timestamp": asof_iso,
            "symbol": symbol,
            "feature_version": news.feature_version,
            "published_at": None if observed else inst,
            "source_seen_at": inst if observed else None,
            "available_at": provenance["available_at"],
            "news_mode": provenance["news_mode"],
            "news_source": provenance["news_source"],
            "timestamp_quality": provenance["timestamp_quality"],
            "source_feature_date": provenance["source_feature_date"],
        }
        record.update(feats)
        rows.append(record)

    ni.write_jsonl_atomic(rows, out_path)
    print(f"intraday news artifact written to {out_path} asof={asof_iso} symbols={len(rows)}")
    return out_path


def run(config_path: str) -> str:
    config = load_config(config_path)
    news = config.news
    output_path = config.resolve(news.output_path)
    report_path = config.resolve(news.report_path)

    bstart = datetime.strptime(news.backfill_start, "%Y-%m-%d").date()
    last_day = ni.last_completed_utc_day()

    # Disabled: cleanly drop news columns by removing any stale feature output.
    if not news.enabled:
        if os.path.isfile(output_path):
            os.remove(output_path)
        report = ni.ingestion_report([], [], bstart, last_day, enabled=False, strict=news.strict_coverage)
        report["news_feature_days"] = 0
        _write_report(report_path, report)
        print("news disabled: removed news feature output; no news columns will be added")
        return output_path

    raw_path = config.resolve(news.raw_path)
    manifest_path = config.resolve(news.coverage_manifest_path)
    records = ni.load_raw_records(raw_path)
    manifest = ni.load_manifest(manifest_path)

    df = pd.DataFrame(records) if records else pd.DataFrame(columns=ni.CANONICAL_FIELDS)
    news_df = nf.prepare_news(df, backend=news.sentiment_backend)

    corr_provider = corr_mod.CorrelationProvider.load(
        config.resolve(config.correlation.output_path),
        fallback_self_corr=config.correlation.fallback_self_corr,
        fallback_cross_corr=config.correlation.fallback_cross_corr,
    )
    universe = list(config.symbols)

    # Load daily series first so we know the real training range before validating.
    per_symbol = []
    day_starts_all = []
    for symbol in config.symbols:
        try:
            daily = load_daily_ohlcv(config, symbol)
        except DataSourceError as exc:
            print(f"{symbol} skipped reason={exc.reason}")
            continue
        if daily is None or daily.empty:
            print(f"{symbol} skipped reason=missing_daily_source")
            continue
        day_starts = daily["timestamp"].dt.floor("1D").tolist()
        per_symbol.append((symbol, day_starts))
        day_starts_all.extend(day_starts)

    if day_starts_all:
        start = min(day_starts_all)
        end = max(day_starts_all)
        start = (start.tz_convert("UTC") if start.tzinfo else start).date()
        end = (end.tz_convert("UTC") if end.tzinfo else end).date()
    else:
        start, end = bstart, last_day

    if news.strict_coverage:
        ni.validate_for_training(
            records,
            manifest,
            start,
            end,
            require_scores=(news.sentiment_backend == "finbert"),
        )

    all_rows = []
    for symbol, day_starts in per_symbol:
        rows = nf.daily_cross_features_for_symbol(
            news_df, symbol, universe, day_starts, corr_provider=corr_provider
        )
        all_rows.extend(rows)
        print(f"{symbol} news_days={len(rows)} raw_news={len(news_df)}")

    nf.write_jsonl(all_rows, output_path)

    report = ni.ingestion_report(records, manifest, start, end, enabled=True, strict=news.strict_coverage)
    report["news_feature_days"] = len(all_rows)
    report["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_report(report_path, report)

    print(f"news features written to {output_path}")
    print(f"news ingestion report written to {report_path}")
    return output_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build news features.")
    parser.add_argument("--config", default="config/distribution_config.yaml")
    parser.add_argument(
        "--intraday",
        action="store_true",
        help="build the as-of intraday artifact (data/news_features_intraday.jsonl)",
    )
    parser.add_argument(
        "--asof",
        default=None,
        help="UTC ISO-8601 decision instant for --intraday (default: now)",
    )
    args = parser.parse_args(argv)
    if args.intraday:
        run_intraday(args.config, asof=args.asof)
    else:
        run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
