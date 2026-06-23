"""GDELT provider + ingestion/scoring tests (no network).

Covers historical backfill, time-window bisection (truncation avoidance),
incremental checkpoint resume, URL/content-hash dedupe, the keyless factory, and
the rule that FinBERT is never run on text-less records (their GDELT tone is kept
as a separate feature, never relabelled as sentiment).
"""

import os
import sys
import urllib.parse
from datetime import date, datetime

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import news_ingest as ni
from src import news_provider as np
from src import news_scoring as ns
from src.news_ingest import QuerySpec
from src.news_provider import (
    GdeltProvider,
    HttpResponse,
    NewsProviderError,
    get_provider,
    provider_requires_api_key,
)


# --------------------------------------------------------------------------- #
# Transport helpers (GDELT DOC 2.0 ArtList JSON shape).
# --------------------------------------------------------------------------- #
def _article(uid: str, hour: str, tone=None, title=None):
    art = {
        "url": f"https://gdelt.example/{uid}",
        "title": title if title is not None else f"headline {uid}",
        "seendate": f"20220101T{hour}0000Z",
        "domain": "example.com",
    }
    if tone is not None:
        art["tone"] = tone
    return art


def _transport(resolver):
    def transport(url, headers, timeout):
        params = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
        return HttpResponse(status=200, body={"articles": resolver(params)})

    return transport


def _span_seconds(params) -> float:
    fmt = "%Y%m%d%H%M%S"
    s = datetime.strptime(params["startdatetime"], fmt)
    e = datetime.strptime(params["enddatetime"], fmt)
    return (e - s).total_seconds()


class FakeClock:
    """Deterministic monotonic clock + sleeper (advances on every sleep)."""

    def __init__(self, start: float = 1000.0):
        self.t = float(start)
        self.sleeps = []

    def time(self) -> float:
        return self.t

    def sleep(self, secs: float) -> None:
        self.sleeps.append(secs)
        self.t += secs


def make_gdelt(transport, *, min_request_interval: float = 0.0, **kw):
    """Build a GdeltProvider that never sleeps for real (fake clock + sleeper)."""
    clk = FakeClock()
    prov = GdeltProvider(
        transport=transport,
        sleep=clk.sleep,
        clock=clk.time,
        min_request_interval=min_request_interval,
        **kw,
    )
    prov.clk = clk  # exposed for sleep/pacing assertions
    return prov


# --------------------------------------------------------------------------- #
# Factory / keyless.
# --------------------------------------------------------------------------- #
def test_gdelt_is_keyless_and_registered():
    assert provider_requires_api_key("gdelt") is False
    assert provider_requires_api_key("newsapi") is True
    prov = get_provider(
        "gdelt", api_key="", transport=_transport(lambda p: []), min_request_interval=0.0
    )
    assert isinstance(prov, GdeltProvider)
    assert prov.name == "gdelt"


# --------------------------------------------------------------------------- #
# Seendate parsing + normalisation.
# --------------------------------------------------------------------------- #
def test_parse_seendate_variants():
    from src.news_provider import _parse_gdelt_seendate

    assert _parse_gdelt_seendate("20220101T120000Z") == "2022-01-01T12:00:00Z"
    assert _parse_gdelt_seendate("20220101120000") == "2022-01-01T12:00:00Z"
    assert _parse_gdelt_seendate("garbage") is None
    assert _parse_gdelt_seendate(None) is None


def test_normalize_article_is_observation_not_publish():
    good = GdeltProvider._normalize_article(_article("a", "08", tone=2.5))
    # seendate is an OBSERVATION time -> source_seen_at, never a publish time.
    assert good["source_seen_at"] == "2022-01-01T08:00:00Z"
    assert good["published_at"] is None
    assert "publishedAt" not in good
    assert good["gdelt_tone"] == 2.5
    assert good["description"] == ""  # ArtList has no body; never invented
    assert GdeltProvider._normalize_article({"url": "x", "title": "t"}) is None


# --------------------------------------------------------------------------- #
# Backfill (happy path) + manifest.
# --------------------------------------------------------------------------- #
def test_gdelt_backfill_writes_records_and_manifest(tmp_path):
    prov = make_gdelt(
        _transport(lambda p: [_article("a", "08", tone=1.0), _article("b", "09")]),
        max_records=250,
    )
    raw = str(tmp_path / "raw" / "news.jsonl")
    manifest = str(tmp_path / "manifest.jsonl")
    specs = [QuerySpec("crypto_market", None, "bitcoin")]
    summary = ni.fetch_range(
        prov, specs, date(2022, 1, 1), date(2022, 1, 1), raw, manifest,
        log=lambda *_: None,
    )
    assert summary["failed_cells"] == 0 and summary["ok_cells"] == 1
    recs = ni.load_raw_records(raw)
    assert len(recs) == 2
    assert {r["provider"] for r in recs} == {"gdelt"}
    # Tone preserved on the canonical record; GDELT instants are observation
    # times -> observed_utc, stored as source_seen_at (never published_at).
    toned = [r for r in recs if r["gdelt_tone"] is not None]
    assert toned and toned[0]["gdelt_tone"] == 1.0
    assert all(r["timestamp_quality"] == "observed_utc" for r in recs)
    assert all(r["source_seen_at"] is not None and r["published_at"] is None for r in recs)
    # The windowing instant equals the observation time for GDELT rows.
    assert all(r["timestamp"] == r["source_seen_at"] for r in recs)
    man = ni.load_manifest(manifest)
    assert all(e["coverage_status"] == ni.COVERAGE_OK for e in man)


# --------------------------------------------------------------------------- #
# Time-window bisection avoids silent truncation.
# --------------------------------------------------------------------------- #
def test_gdelt_bisects_saturated_day():
    # max_records=2: the full-day window saturates (2), each 12h half returns the
    # one article whose instant falls inside it, so the merged result recovers
    # both without truncation.
    pool = [("A", 0), ("B", 12)]

    def resolver(params):
        fmt = "%Y%m%d%H%M%S"
        s = datetime.strptime(params["startdatetime"], fmt)
        e = datetime.strptime(params["enddatetime"], fmt)
        out = []
        for uid, hour in pool:
            inst = s.replace(hour=hour, minute=0, second=0)
            if s <= inst <= e:
                out.append(_article(uid, f"{hour:02d}"))
        return out

    prov = make_gdelt(_transport(resolver), max_records=2)
    arts = prov.fetch_day("bitcoin", date(2022, 1, 1))
    urls = sorted(a["url"] for a in arts)
    assert urls == ["https://gdelt.example/A", "https://gdelt.example/B"]


def test_gdelt_raises_truncated_when_saturated_at_min_window():
    # Always saturated regardless of window -> recursion bottoms out and surfaces
    # result_truncated (recorded as failed coverage, never silent "0 news").
    prov = make_gdelt(
        _transport(lambda p: [_article("A", "00"), _article("B", "01")]),
        max_records=2,
        min_window_seconds=900,
    )
    with pytest.raises(NewsProviderError) as exc:
        prov.fetch_day("bitcoin", date(2022, 1, 1))
    assert exc.value.reason == "result_truncated"


# --------------------------------------------------------------------------- #
# Incremental checkpoint resume.
# --------------------------------------------------------------------------- #
def test_gdelt_incremental_resumes_after_last_ok_day(tmp_path):
    prov = make_gdelt(_transport(lambda p: [_article("a", "08")]))
    raw = str(tmp_path / "raw" / "news.jsonl")
    manifest = str(tmp_path / "manifest.jsonl")
    specs = [QuerySpec("crypto_market", None, "bitcoin")]
    ni.fetch_range(prov, specs, date(2022, 1, 1), date(2022, 1, 3), raw, manifest,
                   log=lambda *_: None)
    man = ni.load_manifest(manifest)
    assert ni.resume_day(man, date(2022, 1, 1)) == date(2022, 1, 4)


# --------------------------------------------------------------------------- #
# Dedupe: URL/provider_id and content hash.
# --------------------------------------------------------------------------- #
def test_dedupe_by_url_provider_id():
    spec = QuerySpec("crypto_market", None, "bitcoin")
    art = GdeltProvider._normalize_article(_article("same", "08"))
    r1 = ni.canonical_record(art, spec, "gdelt", "f")
    r2 = ni.canonical_record(art, spec, "gdelt", "f")  # identical URL
    assert len(ni.dedupe_records([r1, r2])) == 1


def test_dedupe_by_content_hash_across_urls():
    spec = QuerySpec("crypto_market", None, "bitcoin")
    a = GdeltProvider._normalize_article(_article("url1", "08", title="Bitcoin surges to new high"))
    b = GdeltProvider._normalize_article(_article("url2", "09", title="Bitcoin surges to new high"))
    r1 = ni.canonical_record(a, spec, "gdelt", "f")
    r2 = ni.canonical_record(b, spec, "gdelt", "f")
    out = ni.dedupe_records([r1, r2])
    assert len(out) == 1


# --------------------------------------------------------------------------- #
# FinBERT must never fabricate a score for text-less records.
# --------------------------------------------------------------------------- #
class RecordingScorer:
    model_name = "ProsusAI/finbert"
    model_version = "finbert-1"

    def __init__(self):
        self.seen = []

    def score_texts(self, texts):
        self.seen.extend(texts)
        return [0.5 for _ in texts]


def test_finbert_skips_textless_records():
    scorer = RecordingScorer()
    cache = {}
    records = [
        {"title": "Bitcoin rallies hard", "body": ""},   # scorable
        {"title": "", "body": "", "gdelt_tone": -3.0},   # text-less GDELT hit
    ]
    new = ns.score_records(records, scorer, cache)
    assert new == 1
    # The empty text was never sent to FinBERT.
    assert "" not in scorer.seen
    assert scorer.seen == ["Bitcoin rallies hard"]
    # Scorable record got a real score; text-less one is explicitly None.
    assert records[0]["sentiment_score"] == 0.5
    assert records[1]["sentiment_score"] is None
    assert "sentiment_model" not in records[1]
    # Tone is untouched (kept separate, never relabelled as sentiment).
    assert records[1]["gdelt_tone"] == -3.0


# --------------------------------------------------------------------------- #
# Strict coverage still fails for a GDELT manifest with failed cells.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Production-safety: non-JSON / empty / malformed bodies never crash.
# --------------------------------------------------------------------------- #
class _FakeUrlopen:
    """Minimal context-manager stand-in for urllib.request.urlopen."""

    def __init__(self, body: str, status: int = 200, content_type: str = "text/html"):
        self._body = body.encode("utf-8")
        self.status = status
        self.headers = {"Content-Type": content_type}

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_urllib_transport_never_raises_on_non_json(monkeypatch):
    # The original bug: json.loads on an HTML/empty body raised JSONDecodeError
    # and killed the whole fetch with Exit 1. Now it must classify, not crash.
    cases = [
        ("<!DOCTYPE html><html>503 Service Unavailable</html>", "text/html"),
        ("", "text/html"),
        ("{not valid json", "application/json"),
    ]
    for body, ctype in cases:
        monkeypatch.setattr(
            np.urllib.request, "urlopen",
            lambda req, timeout=0, _b=body, _c=ctype: _FakeUrlopen(_b, 200, _c),
        )
        resp = np._urllib_transport("https://api.gdeltproject.org/x", {}, 1.0)
        assert resp.status == 200
        assert resp.json_ok is False  # never silently "ok"
    # Valid JSON still parses.
    monkeypatch.setattr(
        np.urllib.request, "urlopen",
        lambda req, timeout=0: _FakeUrlopen('{"articles": []}', 200, "application/json"),
    )
    ok = np._urllib_transport("https://api.gdeltproject.org/x", {}, 1.0)
    assert ok.json_ok is True and ok.body == {"articles": []}


def _const_transport(resp: HttpResponse):
    def t(url, headers, timeout):
        return resp

    return t


def _count_transport(resp: HttpResponse):
    state = {"n": 0}

    def t(url, headers, timeout):
        state["n"] += 1
        return resp

    t.calls = state
    return t


def _seq_transport(responses):
    seq = list(responses)
    state = {"n": 0}

    def t(url, headers, timeout):
        state["n"] += 1
        return seq.pop(0)

    t.calls = state
    return t


def test_gdelt_html_200_retries_then_non_json_response():
    resp = HttpResponse(200, {}, "text/html; charset=utf-8",
                        "<html><body>error</body></html>", json_ok=False)
    prov = make_gdelt(_const_transport(resp), max_retries=2)
    with pytest.raises(NewsProviderError) as ei:
        prov.fetch_day("bitcoin", date(2022, 1, 1))
    assert ei.value.reason == "non_json_response"
    assert "status=200" in ei.value.detail and "html" in ei.value.detail.lower()
    assert len(prov.clk.sleeps) == 3  # retried (max_retries+1 attempts -> 3 backoffs)


def test_gdelt_empty_200_retries_then_empty_response():
    resp = HttpResponse(200, {}, "text/html", "", json_ok=False)
    prov = make_gdelt(_const_transport(resp), max_retries=1)
    with pytest.raises(NewsProviderError) as ei:
        prov.fetch_day("bitcoin", date(2022, 1, 1))
    assert ei.value.reason == "empty_response"
    assert "status=200" in ei.value.detail


def test_gdelt_invalid_json_200_retries_then_invalid_json_response():
    resp = HttpResponse(200, {}, "application/json", "{not json", json_ok=False)
    prov = make_gdelt(_const_transport(resp), max_retries=1)
    with pytest.raises(NewsProviderError) as ei:
        prov.fetch_day("bitcoin", date(2022, 1, 1))
    assert ei.value.reason == "invalid_json_response"


def test_bad_gdelt_response_marks_failed_not_crash(tmp_path):
    # The whole fetch must not die on the first cell: it records failed coverage.
    resp = HttpResponse(200, {}, "text/html", "<html>oops</html>", json_ok=False)
    prov = make_gdelt(_const_transport(resp), max_retries=1)
    raw = str(tmp_path / "raw" / "news.jsonl")
    manifest = str(tmp_path / "manifest.jsonl")
    summary = ni.fetch_range(
        prov, [QuerySpec("crypto_market", None, "bitcoin")],
        date(2022, 1, 1), date(2022, 1, 1), raw, manifest, log=lambda *_: None,
    )
    assert summary["failed_cells"] == 1
    man = ni.load_manifest(manifest)
    assert man[0]["coverage_status"] == "failed"
    assert man[0]["error"] == "non_json_response"
    # Structured diagnostics are persisted (no URL/key/header).
    assert man[0]["error_code"] == "html_or_waf_body"
    assert man[0]["http_status"] == 200
    assert "html" in (man[0]["content_type"] or "")
    assert man[0]["response_excerpt"]


def test_response_excerpt_redacts_secrets_and_truncates():
    redacted = np._sanitize_excerpt("upstream error apikey=SECRET123 token: a.b.c rest")
    assert "SECRET123" not in redacted
    assert "<redacted>" in redacted
    long = np._sanitize_excerpt("x" * 5000)
    assert "truncated" in long and len(long) < 5000


def test_gdelt_sends_accept_and_user_agent_and_json_path():
    captured = {}

    def t(url, headers, timeout):
        captured["url"] = url
        captured["headers"] = dict(headers)
        return HttpResponse(200, {"articles": []})

    make_gdelt(t).fetch_day("bitcoin", date(2022, 1, 1))
    assert captured["headers"].get("Accept") == "application/json"
    assert captured["headers"].get("User-Agent")
    assert "format=json" in captured["url"] and "mode=artlist" in captured["url"]
    # English-language DOC syntax + UTC datetime bounds.
    assert "sourcelang%3Aenglish" in captured["url"] or "sourcelang:english" in captured["url"]
    assert "startdatetime=20220101000000" in captured["url"]
    assert "enddatetime=20220101235959" in captured["url"]


# --------------------------------------------------------------------------- #
# Query builder + language normalisation (fail-fast on bad codes).
# --------------------------------------------------------------------------- #
def test_gdelt_query_builder_term_shapes_and_language():
    prov = make_gdelt(_transport(lambda p: []))
    # ISO-639-1 'en' is normalised to GDELT's 'english', NOT written as 'en'.
    assert "sourcelang:en " not in prov.build_query("bitcoin", "en") + " "
    # Single keyword: bare, NEVER parenthesised (the bug Hetzner exposed).
    assert prov.build_query("bitcoin", "en") == "bitcoin sourcelang:english"
    # Single multi-word phrase: quoted, never parenthesised.
    assert prov.build_query("jobs report", "en") == '"jobs report" sourcelang:english'
    # Real alias/OR list: parenthesised.
    assert prov.build_query("bitcoin OR BTC", "en") == "(bitcoin OR BTC) sourcelang:english"
    assert (
        prov.build_query("bitcoin OR BTC OR cryptocurrency", "en")
        == "(bitcoin OR BTC OR cryptocurrency) sourcelang:english"
    )
    # Lowercase 'or' is not an operator -> treated as a phrase, not an OR list.
    assert prov.build_query("bitcoin or btc", "en") == '"bitcoin or btc" sourcelang:english'


def test_gdelt_invalid_language_fails_fast():
    prov = make_gdelt(_transport(lambda p: []))
    with pytest.raises(ValueError):
        prov.build_query("bitcoin", "xx")


def test_probe_query_string_is_visible_and_secret_free():
    captured = {}

    def t(url, headers, timeout):
        captured["url"] = url
        return HttpResponse(200, {"articles": []}, "application/json", "{}", json_ok=True)

    make_gdelt(t).probe("bitcoin", date(2026, 6, 19))
    params = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(captured["url"]).query))
    # Single keyword -> NO parentheses (would trip GDELT's OR'd-statements rule).
    assert params["query"] == "bitcoin sourcelang:english"
    assert "(" not in params["query"] and ")" not in params["query"]
    assert params["mode"] == "artlist" and params["format"] == "json"
    low = captured["url"].lower()
    assert "apikey" not in low and "x-api-key" not in low


# --------------------------------------------------------------------------- #
# The real Hetzner body: a query-validation error served as text/html. It MUST
# classify as invalid_request, not be retried as a generic non_json_response.
# --------------------------------------------------------------------------- #
def test_gdelt_ord_statements_validation_is_invalid_request():
    resp = HttpResponse(
        200, {}, "text/html; charset=utf-8",
        "Searches may only be used around OR'd statements.", json_ok=False,
    )
    transport = _count_transport(resp)
    prov = make_gdelt(transport, max_retries=5)
    with pytest.raises(NewsProviderError) as ei:
        prov.fetch_day("bitcoin", date(2026, 6, 19))
    assert ei.value.reason == "invalid_request"
    assert ei.value.error_code == "gdelt_query_validation"
    assert ei.value.http_status == 200
    assert "html" in (ei.value.content_type or "")
    assert ei.value.response_excerpt and "OR'd" in ei.value.response_excerpt
    assert transport.calls["n"] == 1  # NOT retried despite html content type
    assert prov.clk.sleeps == []


# --------------------------------------------------------------------------- #
# GDELT DOC is recent/live only: a 2022 backfill must fail fast, not run.
# --------------------------------------------------------------------------- #
def test_gdelt_doc_rejects_unsupported_history_window():
    from datetime import timezone

    from src import main_fetch_news as mfn

    cfg = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "distribution_config.yaml",
    )
    with pytest.raises(SystemExit) as ei:
        mfn.run(
            cfg, start="2022-01-01", end="2022-01-02", provider_name="gdelt",
            now=datetime(2026, 6, 20, tzinfo=timezone.utc),
        )
    assert "unsupported_history_window" in str(ei.value)


# --------------------------------------------------------------------------- #
# Pacing: 429 Retry-After + adaptive global cooldown, applied across queries.
# --------------------------------------------------------------------------- #
def test_gdelt_429_retry_after_applies_global_cooldown():
    seq = [
        HttpResponse(429, {}, "application/json", "", json_ok=False, retry_after=7.0),
        HttpResponse(200, {"articles": [_article("a", "08")]}, "application/json",
                     "{}", json_ok=True),
    ]
    prov = make_gdelt(_seq_transport(seq), max_retries=2, min_request_interval=1.0)
    arts = prov.fetch_day("bitcoin", date(2022, 1, 1))
    assert len(arts) == 1
    # The server-specified Retry-After (7s) gated the retry, not the 1s interval.
    assert any(abs(s - 7.0) < 1e-9 for s in prov.clk.sleeps)


def test_gdelt_429_without_retry_after_uses_adaptive_cooldown():
    seq = [
        HttpResponse(429, {}, "application/json", "", json_ok=False),
        HttpResponse(429, {}, "application/json", "", json_ok=False),
        HttpResponse(200, {"articles": []}, "application/json", "{}", json_ok=True),
    ]
    # base cooldown = max(backoff_base, min_interval) = 2.0; doubles on the 2nd 429.
    prov = make_gdelt(
        _seq_transport(seq), max_retries=5, min_request_interval=2.0, backoff_base=2.0
    )
    prov.fetch_day("bitcoin", date(2022, 1, 1))
    assert any(abs(s - 2.0) < 1e-9 for s in prov.clk.sleeps)  # first cooldown
    assert any(abs(s - 4.0) < 1e-9 for s in prov.clk.sleeps)  # adaptive doubled


def test_global_pacing_applies_between_queries(tmp_path):
    prov = make_gdelt(_transport(lambda p: [_article("a", "08")]), min_request_interval=2.0)
    specs = [
        QuerySpec("crypto_market", None, "bitcoin"),
        QuerySpec("crypto_market", None, "ethereum"),
    ]
    raw = str(tmp_path / "raw" / "news.jsonl")
    manifest = str(tmp_path / "manifest.jsonl")
    ni.fetch_range(prov, specs, date(2022, 1, 1), date(2022, 1, 1), raw, manifest,
                   log=lambda *_: None)
    # The second query waited on the SAME global gate (no per-query burst).
    assert any(abs(s - 2.0) < 1e-9 for s in prov.clk.sleeps)


# --------------------------------------------------------------------------- #
# Plain-text GDELT validation error is a permanent, non-retryable invalid_request.
# --------------------------------------------------------------------------- #
def test_gdelt_plain_text_validation_is_nonretryable_invalid_request():
    # Even with an html-ish content type, the BODY identifies a query problem.
    resp = HttpResponse(200, {}, "text/html",
                        "Your query was too short. Please make it longer.", json_ok=False)
    transport = _count_transport(resp)
    prov = make_gdelt(transport, max_retries=5)
    with pytest.raises(NewsProviderError) as ei:
        prov.fetch_day("bitcoin", date(2022, 1, 1))
    assert ei.value.reason == "invalid_request"
    assert ei.value.error_code == "gdelt_query_validation"
    assert transport.calls["n"] == 1  # NOT retried
    assert prov.clk.sleeps == []  # no backoff, no cooldown


def test_gdelt_invalid_request_records_full_classification(tmp_path):
    resp = HttpResponse(200, {}, "text/plain",
                        "Your query was too long. Please make it shorter.", json_ok=False)
    prov = make_gdelt(_const_transport(resp), max_retries=3)
    raw = str(tmp_path / "raw" / "news.jsonl")
    manifest = str(tmp_path / "manifest.jsonl")
    ni.fetch_range(prov, [QuerySpec("crypto_market", None, "bitcoin")],
                   date(2022, 1, 1), date(2022, 1, 1), raw, manifest, log=lambda *_: None)
    man = ni.load_manifest(manifest)
    assert man[0]["error"] == "invalid_request"
    assert man[0]["error_code"] == "gdelt_query_validation"
    assert man[0]["http_status"] == 200
    assert man[0]["response_excerpt"] and "too long" in man[0]["response_excerpt"]


# --------------------------------------------------------------------------- #
# Single-request probe mode (no retry, no bisection) + safe manifest record.
# --------------------------------------------------------------------------- #
def test_probe_single_http_call_ok():
    transport = _count_transport(
        HttpResponse(200, {"articles": [_article("a", "08")]}, "application/json",
                     "{}", json_ok=True)
    )
    prov = make_gdelt(transport)
    res = prov.probe("bitcoin", date(2026, 6, 19))
    assert res["ok"] is True and res["response_class"] == "ok"
    assert res["article_count"] == 1
    assert transport.calls["n"] == 1  # exactly one HTTP call


def test_probe_classifies_and_writes_manifest(tmp_path):
    resp = HttpResponse(200, {}, "text/plain", "Your query was too short.", json_ok=False)
    transport = _count_transport(resp)
    prov = make_gdelt(transport)
    manifest = str(tmp_path / "manifest.jsonl")
    res = ni.probe_cell(prov, QuerySpec("crypto_market", None, "bitcoin"),
                        date(2026, 6, 19), manifest, log=lambda *_: None)
    assert res["ok"] is False
    assert res["response_class"] == "invalid_request"
    assert transport.calls["n"] == 1  # single call, no retry/bisection
    man = ni.load_manifest(manifest)
    assert man[0]["coverage_status"] == "failed"
    assert man[0]["error"] == "invalid_request"
    assert man[0]["error_code"] == "gdelt_query_validation"
    assert man[0]["probe"] is True


def test_strict_coverage_fails_on_gdelt_failure():
    man = [{"date": "2022-01-01", "category": "crypto_market", "symbol": None,
            "query": "bitcoin", "provider": "gdelt", "coverage_status": "failed"}]
    rec = {"timestamp": "2022-01-01T12:00:00Z", "category": "crypto_market",
           "symbol": None, "title": "t", "body": "b", "provider_id": "u:x",
           "url": "https://g/x", "sentiment_score": 0.1}
    with pytest.raises(ni.NewsValidationError):
        ni.validate_for_training([rec], man, date(2022, 1, 1), date(2022, 1, 1))
