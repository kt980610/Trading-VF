"""Integration tests for the news pipeline (fetch -> raw -> daily features).

All network access is faked. These exercise strict validation, leakage, the
disabled path, and that the API key never lands in any output artifact.
"""

import json
import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import main_build_news as mbn
from src import main_fetch_news as mfn
from src import news_ingest as ni
from src.news_provider import HttpResponse


def _abs(p):
    return str(p).replace("\\", "/")


def _write_config(tmp_path, *, enabled=True, strict=True, backend="lexicon"):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "symbols:\n"
        "  - BTCUSDT\n"
        "data:\n"
        f'  raw_dir: "{_abs(raw_dir)}"\n'
        '  daily_file_suffix: "_daily.csv"\n'
        '  minute_file_suffix: "_1m.csv"\n'
        "  require_daily_source: true\n"
        "  require_minute_source: false\n"
        "news:\n"
        f"  enabled: {'true' if enabled else 'false'}\n"
        f"  strict_coverage: {'true' if strict else 'false'}\n"
        "  provider: newsapi\n"
        "  api_key_env: NEWSAPI_API_KEY\n"
        f'  raw_path: "{_abs(tmp_path / "data" / "news.jsonl")}"\n'
        f'  coverage_manifest_path: "{_abs(tmp_path / "data" / "manifest.jsonl")}"\n'
        f'  output_path: "{_abs(tmp_path / "data" / "news_features_daily.jsonl")}"\n'
        f'  report_path: "{_abs(tmp_path / "reports" / "news_report.json")}"\n'
        f"  sentiment_backend: {backend}\n"
        "  backfill_start: 2022-01-01\n"
        "  language: en\n"
        "correlation:\n"
        f'  output_path: "{_abs(tmp_path / "data" / "correlation.jsonl")}"\n',
        encoding="utf-8",
    )
    return cfg


def _write_daily_csv(tmp_path, symbol="BTCUSDT", days=("2022-01-02", "2022-01-03", "2022-01-04")):
    path = tmp_path / "data" / "raw" / f"{symbol}_daily.csv"
    lines = ["timestamp,open,high,low,close,volume"]
    for i, d in enumerate(days):
        lines.append(f"{d}T00:00:00Z,100,101,99,100,{1000 + i}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Strict validation through the CLI entrypoint.
# --------------------------------------------------------------------------- #
def test_strict_build_fails_without_raw_news(tmp_path):
    cfg = _write_config(tmp_path, enabled=True, strict=True)
    _write_daily_csv(tmp_path)
    with pytest.raises(ni.NewsValidationError):
        mbn.run(str(cfg))


def test_disabled_removes_news_feature_output(tmp_path):
    cfg = _write_config(tmp_path, enabled=False)
    out = tmp_path / "data" / "news_features_daily.jsonl"
    out.write_text('{"stale": true}\n', encoding="utf-8")
    mbn.run(str(cfg))
    assert not out.exists()
    # A report is still produced documenting the disabled state.
    assert (tmp_path / "reports" / "news_report.json").exists()


def test_nonzero_news_produces_daily_features_with_leakage_cutoff(tmp_path):
    cfg = _write_config(tmp_path, enabled=True, strict=True, backend="lexicon")
    _write_daily_csv(tmp_path)

    raw_records = [
        # Counts for 2022-01-02 (in [2022-01-01, 2022-01-02)).
        {"timestamp": "2022-01-01T12:00:00Z", "title": "fed rally", "body": "inflation",
         "category": "macro", "symbol": None, "provider_id": "u:1",
         "url": "https://e.com/1", "query": "FOMC", "sentiment_score": 0.5},
        # Same-day (>= day start) -> leakage; must NOT count for 2022-01-02.
        {"timestamp": "2022-01-02T05:00:00Z", "title": "later", "body": "",
         "category": "macro", "symbol": None, "provider_id": "u:2",
         "url": "https://e.com/2", "query": "FOMC", "sentiment_score": -0.9},
    ]
    raw_path = tmp_path / "data" / "news.jsonl"
    raw_path.write_text("\n".join(json.dumps(r) for r in raw_records) + "\n", encoding="utf-8")

    manifest = [
        {"date": d, "category": "macro", "symbol": None, "query": "FOMC",
         "coverage_status": "ok"}
        for d in ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"]
    ]
    (tmp_path / "data" / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m) for m in manifest) + "\n", encoding="utf-8"
    )

    mbn.run(str(cfg))

    out = tmp_path / "data" / "news_features_daily.jsonl"
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_date = {r["date"]: r for r in rows if r["symbol"] == "BTCUSDT"}
    # Only the 2022-01-01 article is inside the leakage-safe window for 2022-01-02.
    assert by_date["2022-01-02"]["macro_news_count"] == 1
    # Report exists and records a nonzero raw count.
    report = json.loads((tmp_path / "reports" / "news_report.json").read_text(encoding="utf-8"))
    assert report["raw_record_count"] >= 1


# --------------------------------------------------------------------------- #
# API key secrecy.
# --------------------------------------------------------------------------- #
def test_fetch_does_not_leak_api_key_into_artifacts(tmp_path, monkeypatch):
    secret = "SUPER-SECRET-NEWSAPI-KEY-987"
    monkeypatch.setenv("NEWSAPI_API_KEY", secret)
    cfg = _write_config(tmp_path, enabled=True, strict=True)

    captured_headers = []

    def transport(url, headers, timeout):
        captured_headers.append(dict(headers))
        assert secret not in url  # key must never be in the URL
        return HttpResponse(
            status=200,
            body={
                "status": "ok",
                "totalResults": 1,
                "articles": [
                    {"source": {"name": "s"}, "title": "t", "description": "d",
                     "url": "https://e.com/a", "publishedAt": "2022-01-01T08:00:00Z"}
                ],
            },
        )

    mfn.run(
        str(cfg),
        start="2022-01-01",
        end="2022-01-01",
        provider_name="newsapi",
        transport=transport,
        now=datetime(2022, 1, 5, tzinfo=timezone.utc),
    )

    # The key is used (header) but must not appear in any produced file.
    assert any(h.get("X-Api-Key") == secret for h in captured_headers)

    produced = []
    for root, _dirs, files in os.walk(tmp_path):
        for name in files:
            if name == "config.yaml":
                continue
            with open(os.path.join(root, name), "r", encoding="utf-8", errors="ignore") as fh:
                produced.append(fh.read())
    blob = "\n".join(produced)
    assert secret not in blob
    # Sanity: some raw news was actually written.
    assert "https://e.com/a" in blob
