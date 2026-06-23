"""Unit tests for canonical news ingestion (no network)."""

import os
import sys
from datetime import date

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import news_ingest as ni
from src.news_ingest import QuerySpec
from src.news_provider import NewsProviderError


class FakeProvider:
    name = "newsapi"

    def __init__(self, fail_queries=None, per_day=1, empty_queries=None):
        self.fail_queries = set(fail_queries or [])
        self.empty_queries = set(empty_queries or [])
        self.per_day = per_day

    def fetch_day(self, query, day, language="en"):
        if query in self.fail_queries:
            raise NewsProviderError("rate_limited", "boom")
        if query in self.empty_queries:
            return []
        return [
            {
                "source": {"name": "src"},
                "title": f"{query} {day.isoformat()} {i}",
                "description": "desc",
                "url": f"https://e.com/{query}/{day.isoformat()}/{i}",
                "publishedAt": f"{day.isoformat()}T12:00:00Z",
            }
            for i in range(self.per_day)
        ]


# --------------------------------------------------------------------------- #
# Timestamp normalisation.
# --------------------------------------------------------------------------- #
def test_normalize_timestamp_utc_and_offset():
    assert ni.normalize_timestamp("2022-01-01T12:00:00Z") == "2022-01-01T12:00:00Z"
    # -05:00 offset normalises to UTC (+5h).
    assert ni.normalize_timestamp("2022-01-01T07:00:00-05:00") == "2022-01-01T12:00:00Z"


def test_normalize_timestamp_missing_returns_none():
    assert ni.normalize_timestamp(None) is None
    assert ni.normalize_timestamp("") is None
    assert ni.normalize_timestamp("not-a-date") is None


def test_canonical_record_drops_when_no_publish_timestamp():
    spec = QuerySpec("macro", None, "FOMC")
    art = {"title": "x", "description": "y", "url": "https://e.com/x"}  # no publishedAt
    assert ni.canonical_record(art, spec, "newsapi", "2026-01-01T00:00:00Z") is None


def test_canonical_record_uses_publish_time_not_fetch_time():
    spec = QuerySpec("macro", None, "FOMC")
    art = {"title": "x", "description": "y", "url": "https://e.com/x",
           "publishedAt": "2022-03-04T05:06:07Z", "source": {"name": "s"}}
    rec = ni.canonical_record(art, spec, "newsapi", "2026-06-20T00:00:00Z")
    assert rec["timestamp"] == "2022-03-04T05:06:07Z"
    assert rec["fetched_at"] == "2026-06-20T00:00:00Z"
    assert rec["category"] == "macro" and rec["symbol"] is None


def test_dedupe_by_provider_id_category_symbol():
    spec = QuerySpec("macro", None, "FOMC")
    art = {"title": "x", "description": "y", "url": "https://e.com/x",
           "publishedAt": "2022-01-01T00:00:00Z"}
    r1 = ni.canonical_record(art, spec, "newsapi", "f")
    r2 = ni.canonical_record(art, spec, "newsapi", "f")
    out = ni.dedupe_records([r1, r2])
    assert len(out) == 1


# --------------------------------------------------------------------------- #
# fetch_range orchestration + coverage manifest.
# --------------------------------------------------------------------------- #
def _paths(tmp_path):
    raw = str(tmp_path / "data" / "raw" / "news.jsonl")
    manifest = str(tmp_path / "data" / "news_coverage_manifest.jsonl")
    return raw, manifest


def test_fetch_range_writes_records_and_ok_manifest(tmp_path):
    raw, manifest = _paths(tmp_path)
    specs = [QuerySpec("macro", None, "FOMC"), QuerySpec("symbol_specific", "BTCUSDT", "Bitcoin")]
    provider = FakeProvider(per_day=1)
    summary = ni.fetch_range(
        provider, specs, date(2022, 1, 1), date(2022, 1, 2), raw, manifest,
        fetched_at="2026-01-01T00:00:00Z", log=lambda *_: None,
    )
    assert summary["days"] == 2
    assert summary["failed_cells"] == 0
    assert summary["ok_cells"] == 4
    records = ni.load_raw_records(raw)
    assert len(records) == 4  # 2 days x 2 queries, all unique URLs
    man = ni.load_manifest(manifest)
    assert all(e["coverage_status"] == ni.COVERAGE_OK for e in man)


def test_failed_query_is_marked_failed_not_zero(tmp_path):
    raw, manifest = _paths(tmp_path)
    specs = [QuerySpec("macro", None, "FOMC")]
    provider = FakeProvider(fail_queries={"FOMC"})
    summary = ni.fetch_range(
        provider, specs, date(2022, 1, 1), date(2022, 1, 1), raw, manifest,
        log=lambda *_: None,
    )
    assert summary["failed_cells"] == 1
    man = ni.load_manifest(manifest)
    assert man[0]["coverage_status"] == ni.COVERAGE_FAILED
    assert man[0]["error"] == "rate_limited"
    # Failed coverage must register as a failure for strict validation, NOT zero.
    fails = ni.coverage_failures(man, date(2022, 1, 1), date(2022, 1, 1))
    assert len(fails) == 1


def test_drops_articles_without_timestamp_and_counts(tmp_path):
    raw, manifest = _paths(tmp_path)

    class NoTsProvider:
        name = "newsapi"

        def fetch_day(self, query, day, language="en"):
            return [
                {"title": "ok", "url": "https://e.com/a", "publishedAt": "2022-01-01T00:00:00Z"},
                {"title": "bad", "url": "https://e.com/b"},  # no publishedAt -> dropped
            ]

    summary = ni.fetch_range(
        NoTsProvider(), [QuerySpec("macro", None, "FOMC")],
        date(2022, 1, 1), date(2022, 1, 1), raw, manifest, log=lambda *_: None,
    )
    assert summary["dropped_no_timestamp"] == 1
    assert summary["records_written"] == 1


def test_incremental_resume_day(tmp_path):
    raw, manifest = _paths(tmp_path)
    provider = FakeProvider(per_day=1)
    ni.fetch_range(
        provider, [QuerySpec("macro", None, "FOMC")],
        date(2022, 1, 1), date(2022, 1, 3), raw, manifest, log=lambda *_: None,
    )
    man = ni.load_manifest(manifest)
    assert ni.resume_day(man, date(2022, 1, 1)) == date(2022, 1, 4)


# --------------------------------------------------------------------------- #
# Strict validation.
# --------------------------------------------------------------------------- #
def _rec(ts="2022-01-01T12:00:00Z", scored=True):
    r = {"timestamp": ts, "category": "macro", "symbol": None, "title": "t", "body": "b",
         "provider_id": "u:" + ts, "url": "https://e.com/" + ts}
    if scored:
        r["sentiment_score"] = 0.1
    return r


def test_strict_fails_with_no_records():
    with pytest.raises(ni.NewsValidationError):
        ni.validate_for_training([], [{"date": "2022-01-01", "coverage_status": "ok"}],
                                 date(2022, 1, 1), date(2022, 1, 2))


def test_strict_fails_with_empty_manifest():
    with pytest.raises(ni.NewsValidationError):
        ni.validate_for_training([_rec()], [], date(2022, 1, 1), date(2022, 1, 2))


def test_strict_fails_with_failed_coverage():
    man = [{"date": "2022-01-01", "category": "macro", "symbol": None, "query": "FOMC",
            "coverage_status": "failed"}]
    with pytest.raises(ni.NewsValidationError):
        ni.validate_for_training([_rec()], man, date(2022, 1, 1), date(2022, 1, 1))


def test_strict_fails_when_zero_in_range():
    man = [{"date": "2030-01-01", "coverage_status": "ok"}]
    with pytest.raises(ni.NewsValidationError):
        ni.validate_for_training([_rec("2022-01-01T12:00:00Z")], man,
                                 date(2030, 1, 1), date(2030, 1, 2))


def test_strict_requires_scores_when_finbert():
    man = [{"date": "2022-01-01", "coverage_status": "ok"}]
    with pytest.raises(ni.NewsValidationError):
        ni.validate_for_training([_rec(scored=False)], man,
                                 date(2022, 1, 1), date(2022, 1, 1), require_scores=True)


def test_strict_passes_happy_path():
    man = [{"date": "2022-01-01", "category": "macro", "symbol": None, "query": "FOMC",
            "coverage_status": "ok"}]
    ni.validate_for_training([_rec()], man, date(2022, 1, 1), date(2022, 1, 1),
                             require_scores=True)  # should NOT raise


def test_ingestion_report_fields(tmp_path):
    man = [{"date": "2022-01-01", "coverage_status": "ok"},
           {"date": "2022-01-02", "coverage_status": "failed"}]
    report = ni.ingestion_report([_rec()], man, date(2022, 1, 1), date(2022, 1, 2),
                                 enabled=True, strict=True)
    assert report["raw_record_count"] == 1
    assert report["coverage_ok_cells"] == 1
    assert report["coverage_failed_cells"] == 1
    assert report["min_timestamp"] == "2022-01-01T12:00:00Z"
    assert report["category_counts"]["macro"] == 1
