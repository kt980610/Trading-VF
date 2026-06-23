"""Live as-of intraday news features: leakage, 5-minute safety lag, the rolling
24h window across a UTC day boundary, the strict/atomic intraday artifact, API
key secrecy, and that current news can move a FIXED RF model's p_close."""

import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import main_build_news as mbn
from src import news_features as nf
from src import news_ingest as ni

UNIVERSE = ["BTCUSDT", "ETHUSDT"]


def _news(rows):
    return nf.prepare_news(pd.DataFrame(rows), backend="lexicon")


def _macro(ts, score):
    return {"timestamp": ts, "category": "macro", "symbol": None,
            "title": "", "body": "", "sentiment_score": score}


# --------------------------------------------------------------------------- #
# Pure as-of aggregation: leakage + safety lag.
# --------------------------------------------------------------------------- #
def test_asof_excludes_future_and_lagged_news():
    asof = pd.Timestamp("2026-06-20T12:00:00Z")
    news = _news([
        _macro("2026-06-20T11:50:00Z", 0.5),   # 10 min before -> counts
        _macro("2026-06-20T11:58:00Z", -0.9),   # 2 min before  -> inside 300s lag -> excluded
        _macro("2026-06-20T12:05:00Z", 1.0),    # after decision -> excluded (no look-ahead)
    ])
    feats = nf.asof_features_for_symbol(
        news, "BTCUSDT", UNIVERSE, asof, safety_lag_seconds=300, lookback_hours=24
    )
    assert feats["macro_news_count"] == 1
    assert feats["macro_news_sentiment"] == pytest.approx(0.5)


def test_asof_safety_lag_boundary_is_inclusive():
    asof = pd.Timestamp("2026-06-20T12:00:00Z")
    # Exactly at the cutoff (asof - 300s) is allowed; one second later is not.
    news = _news([
        _macro("2026-06-20T11:55:00Z", 0.4),    # == cutoff -> counts
        _macro("2026-06-20T11:55:01Z", -0.4),    # > cutoff  -> excluded
    ])
    feats = nf.asof_features_for_symbol(news, "BTCUSDT", UNIVERSE, asof, safety_lag_seconds=300)
    assert feats["macro_news_count"] == 1
    assert feats["macro_news_sentiment"] == pytest.approx(0.4)


# --------------------------------------------------------------------------- #
# Rolling 24h window does not reset at the UTC day boundary.
# --------------------------------------------------------------------------- #
def test_rolling_window_crosses_utc_day_boundary():
    asof = pd.Timestamp("2026-06-21T00:10:00Z")
    news = _news([
        _macro("2026-06-20T23:00:00Z", 0.3),    # previous UTC day, within 24h -> counts
        _macro("2026-06-19T23:00:00Z", -0.7),    # > 24h before asof -> excluded
    ])
    feats = nf.asof_features_for_symbol(
        news, "BTCUSDT", UNIVERSE, asof, lookback_hours=24, safety_lag_seconds=300
    )
    # The day boundary did NOT wipe the signal; the >24h item dropped out.
    assert feats["macro_news_count"] == 1
    assert feats["macro_news_sentiment"] == pytest.approx(0.3)


def test_asof_empty_window_is_explicit_zero():
    asof = pd.Timestamp("2026-06-20T12:00:00Z")
    news = _news([_macro("2026-06-15T00:00:00Z", 0.9)])  # far outside 24h
    feats = nf.asof_features_for_symbol(news, "BTCUSDT", UNIVERSE, asof)
    assert feats["macro_news_count"] == 0
    assert feats["macro_news_sentiment"] == 0.0
    assert feats["weighted_symbol_news_count"] == 0.0


# --------------------------------------------------------------------------- #
# Intraday artifact via the CLI: strict, schema-complete, atomic.
# --------------------------------------------------------------------------- #
def _write_intraday_config(tmp_path, *, enabled=True, strict=True):
    out = tmp_path / "data" / "news_features_intraday.jsonl"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "symbols:\n  - BTCUSDT\n  - ETHUSDT\n"
        "news:\n"
        f"  enabled: {'true' if enabled else 'false'}\n"
        f"  strict_coverage: {'true' if strict else 'false'}\n"
        "  provider: newsapi\n"
        "  api_key_env: NEWSAPI_API_KEY\n"
        f'  raw_path: "{str(tmp_path / "data" / "news.jsonl").replace(chr(92), "/")}"\n'
        f'  intraday_output_path: "{str(out).replace(chr(92), "/")}"\n'
        "  sentiment_backend: lexicon\n"
        "  feature_lookback_hours: 24\n"
        "  news_safety_lag_seconds: 300\n"
        "  max_news_feature_age_minutes: 30\n"
        "  feature_version: news_test_v1\n"
        "correlation:\n"
        f'  output_path: "{str(tmp_path / "data" / "correlation.jsonl").replace(chr(92), "/")}"\n',
        encoding="utf-8",
    )
    return cfg, out


def _write_raw(tmp_path, rows):
    raw = tmp_path / "data" / "news.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return raw


def test_intraday_artifact_is_schema_complete_and_atomic(tmp_path):
    cfg, out = _write_intraday_config(tmp_path)
    _write_raw(tmp_path, [
        {"timestamp": "2026-06-20T10:00:00Z", "category": "macro", "symbol": None,
         "title": "fed", "body": "", "sentiment_score": 0.5, "url": "https://e/1", "provider_id": "1"},
        {"timestamp": "2026-06-20T09:00:00Z", "category": "symbol_specific", "symbol": "BTCUSDT",
         "title": "btc", "body": "", "sentiment_score": 0.2, "url": "https://e/2", "provider_id": "2"},
    ])

    mbn.run_intraday(str(cfg), asof="2026-06-20T12:00:00Z")

    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert {r["symbol"] for r in rows} == {"BTCUSDT", "ETHUSDT"}
    btc = next(r for r in rows if r["symbol"] == "BTCUSDT")
    assert btc["asof_timestamp"] == "2026-06-20T12:00:00Z"
    assert btc["feature_version"] == "news_test_v1"
    # Full RF news schema present.
    for cat in nf.CATEGORIES:
        assert f"{cat}_news_sentiment" in btc and f"{cat}_news_count" in btc
    for agg in nf.WEIGHTED_NEWS_AGGREGATES:
        assert agg in btc
    for src in UNIVERSE:
        assert f"news_sentiment_from_{src}_weighted" in btc
    assert btc["macro_news_count"] == 1
    # Atomic write leaves no temp file behind.
    assert not (out.parent / (out.name + ".tmp")).exists()


def test_intraday_strict_refuses_all_zero_without_raw(tmp_path):
    cfg, out = _write_intraday_config(tmp_path, strict=True)
    with pytest.raises(ni.NewsValidationError):
        mbn.run_intraday(str(cfg), asof="2026-06-20T12:00:00Z")
    assert not out.exists()  # no fake zero-news artifact written


def test_intraday_disabled_removes_artifact(tmp_path):
    cfg, out = _write_intraday_config(tmp_path, enabled=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('{"stale": true}\n', encoding="utf-8")
    mbn.run_intraday(str(cfg))
    assert not out.exists()


def test_training_and_live_use_same_asof_join(tmp_path):
    """The live artifact record selected for decision D (run at D - lag) must hold
    exactly the features the training join produces for D -- same join, no drift."""
    cfg, out = _write_intraday_config(tmp_path)
    rows = [
        {"timestamp": "2026-06-20T09:00:00Z", "category": "macro", "symbol": None,
         "title": "", "body": "", "sentiment_score": 0.5, "url": "https://e/1", "provider_id": "1"},
        {"timestamp": "2026-06-20T09:30:00Z", "category": "symbol_specific", "symbol": "BTCUSDT",
         "title": "", "body": "", "sentiment_score": 0.2, "url": "https://e/2", "provider_id": "2"},
        # Published inside the 5-min lag of a 12:00 decision -> excluded by both.
        {"timestamp": "2026-06-20T11:58:00Z", "category": "macro", "symbol": None,
         "title": "", "body": "", "sentiment_score": -1.0, "url": "https://e/3", "provider_id": "3"},
    ]
    _write_raw(tmp_path, rows)

    # Live worker snapshot at run time W = decision - lag = 12:00 - 300s.
    mbn.run_intraday(str(cfg), asof="2026-06-20T11:55:00Z")
    art = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    live_btc = next(r for r in art if r["symbol"] == "BTCUSDT")

    # Training join at decision D = 12:00 uses the SAME available_at as-of join.
    news_df = nf.prepare_news(pd.DataFrame(rows), backend="lexicon")
    train_feats, _prov = nf.asof_news_for_decision(
        news_df, "BTCUSDT", UNIVERSE, pd.Timestamp("2026-06-20T12:00:00Z"),
        lookback_hours=24, safety_lag_seconds=300, live=False, historical_lag_seconds=300,
    )

    for key, val in train_feats.items():
        assert live_btc[key] == pytest.approx(val), key
    # 11:58 -> available 12:03 > cutoff -> excluded; only the 09:00 macro counts.
    assert train_feats["macro_news_count"] == 1


def test_api_key_never_in_intraday_artifact(tmp_path, monkeypatch):
    secret = "SECRET-NEWSAPI-KEY-INTRADAY-42"
    monkeypatch.setenv("NEWSAPI_API_KEY", secret)
    cfg, out = _write_intraday_config(tmp_path)
    _write_raw(tmp_path, [
        {"timestamp": "2026-06-20T10:00:00Z", "category": "macro", "symbol": None,
         "title": "fed", "body": "", "sentiment_score": 0.5, "url": "https://e/1", "provider_id": "1"},
    ])
    mbn.run_intraday(str(cfg), asof="2026-06-20T12:00:00Z")
    assert secret not in out.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# A fixed RF model produces different p_close for different current news.
# --------------------------------------------------------------------------- #
def test_p_close_changes_with_news_same_model():
    sk = pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from src.rf_classifier import ClosePolicy

    n = 200
    macro = [(-1.0 if i % 2 == 0 else 1.0) for i in range(n)]
    X = pd.DataFrame({"macro_news_sentiment": macro})
    y = [0 if v < 0 else 1 for v in macro]
    model = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=0)
    model.fit(X[["macro_news_sentiment"]].to_numpy(), y)

    policy = ClosePolicy(
        scale_cols=[], passthrough_cols=["macro_news_sentiment"], medians={},
        threshold=0.5, model=model, scaler=None,
    )
    p_bear = policy.predict_proba_close(pd.DataFrame({"macro_news_sentiment": [-1.0]}))[0]
    p_bull = policy.predict_proba_close(pd.DataFrame({"macro_news_sentiment": [1.0]}))[0]
    assert p_bull != p_bear
    assert p_bull > p_bear


# --------------------------------------------------------------------------- #
# Timestamp quality classification.
# --------------------------------------------------------------------------- #
def test_timestamp_quality_classification():
    assert nf.classify_timestamp_quality("2026-06-20T08:00:00Z") == "exact_utc"
    assert nf.classify_timestamp_quality("2026-06-20T08:00:00+00:00") == "exact_utc"
    assert nf.classify_timestamp_quality("2026-06-20T08:00:00+02:00") == "exact_utc"
    # Naive (no tz) -> ambiguous tz -> date_only, never silently treated as UTC.
    assert nf.classify_timestamp_quality("2026-06-20T08:00:00") == "date_only"
    assert nf.classify_timestamp_quality("2026-06-20") == "date_only"
    assert nf.classify_timestamp_quality("not a date") == "invalid"
    assert nf.classify_timestamp_quality("") == "invalid"
    assert nf.classify_timestamp_quality(None) == "invalid"


# --------------------------------------------------------------------------- #
# Decision cutoff uses the CURRENT decision time, not the entry time.
# --------------------------------------------------------------------------- #
def test_same_day_news_after_decision_is_not_used():
    decision = pd.Timestamp("2026-06-20T12:00:00Z")
    news = _news([
        _macro("2026-06-20T11:50:00Z", 0.5),    # before decision-lag -> used
        _macro("2026-06-20T12:30:00Z", -1.0),    # AFTER decision -> never used
    ])
    feats, prov = nf.asof_news_for_decision(
        news, "BTCUSDT", UNIVERSE, decision, lookback_hours=24, safety_lag_seconds=300
    )
    assert prov["news_mode"] == "intraday_asof"
    assert feats["macro_news_count"] == 1
    assert feats["macro_news_sentiment"] == pytest.approx(0.5)


def test_news_after_entry_before_close_decision_is_used():
    """A position opened at 10:00; the close decision is evaluated at 13:00. News
    published at 12:00 (after entry, before close decision - lag) MUST be used."""
    close_decision = pd.Timestamp("2026-06-20T13:00:00Z")
    news = _news([_macro("2026-06-20T12:00:00Z", -0.8)])
    feats, prov = nf.asof_news_for_decision(
        news, "BTCUSDT", UNIVERSE, close_decision, lookback_hours=24, safety_lag_seconds=300
    )
    assert prov["news_mode"] == "intraday_asof"
    assert feats["macro_news_count"] == 1
    assert feats["macro_news_sentiment"] == pytest.approx(-0.8)


# --------------------------------------------------------------------------- #
# Date-only / ambiguous-tz news: excluded from intraday, triggers fallback.
# --------------------------------------------------------------------------- #
def test_date_only_news_excluded_from_intraday():
    decision = pd.Timestamp("2026-06-20T12:00:00Z")
    # Same calendar day but only a date (no precise instant) -> never intraday.
    news = _news([{"timestamp": "2026-06-20", "category": "macro", "symbol": None,
                   "title": "", "body": "", "sentiment_score": 0.9}])
    feats = nf.asof_features_for_symbol(
        news, "BTCUSDT", UNIVERSE, decision, lookback_hours=24, safety_lag_seconds=300
    )
    assert feats["macro_news_count"] == 0
    assert feats["macro_news_sentiment"] == 0.0


def test_date_only_news_triggers_previous_completed_day_fallback():
    decision = pd.Timestamp("2026-06-20T12:00:00Z")
    news = _news([{"timestamp": "2026-06-20", "category": "macro", "symbol": None,
                   "title": "", "body": "", "sentiment_score": 0.9}])
    daily_ctx = {
        "2026-06-19": {"macro_news_sentiment": 0.42, "macro_news_count": 3},
        "2026-06-18": {"macro_news_sentiment": -0.1, "macro_news_count": 1},
    }
    feats, prov = nf.asof_news_for_decision(
        news, "BTCUSDT", UNIVERSE, decision, lookback_hours=24,
        safety_lag_seconds=300, daily_ctx=daily_ctx,
    )
    assert prov["news_mode"] == "previous_completed_day_fallback"
    assert prov["timestamp_quality"] == "date_only"
    assert prov["source_feature_date"] == "2026-06-19"
    assert feats["macro_news_sentiment"] == pytest.approx(0.42)
    assert feats["macro_news_count"] == 3
    # Schema stays complete even in fallback mode.
    for agg in nf.WEIGHTED_NEWS_AGGREGATES:
        assert agg in feats


def test_fallback_binds_to_correct_day_after_utc_day_change():
    """At 00:05 UTC on D, the previous completed UTC day is D-1; the fallback must
    bind to D-1's daily row, never a same-day or future row."""
    decision = pd.Timestamp("2026-06-21T00:05:00Z")
    # Date-only news on D-1 (relevant to the rolling window, but no precise instant).
    news = _news([{"timestamp": "2026-06-20", "category": "macro", "symbol": None,
                   "title": "", "body": "", "sentiment_score": 0.3}])
    daily_ctx = {
        "2026-06-21": {"macro_news_sentiment": 9.9, "macro_news_count": 99},   # same day -> forbidden
        "2026-06-20": {"macro_news_sentiment": 0.55, "macro_news_count": 5},   # D-1 -> expected
        "2026-06-19": {"macro_news_sentiment": -0.2, "macro_news_count": 2},
    }
    feats, prov = nf.asof_news_for_decision(
        news, "BTCUSDT", UNIVERSE, decision, lookback_hours=24,
        safety_lag_seconds=300, daily_ctx=daily_ctx,
    )
    assert prov["news_mode"] == "previous_completed_day_fallback"
    assert prov["source_feature_date"] == "2026-06-20"
    assert feats["macro_news_sentiment"] == pytest.approx(0.55)
    assert feats["macro_news_count"] == 5


def test_exact_utc_news_preferred_over_date_only_same_day():
    """When BOTH exact-UTC and date-only news exist, intraday uses the exact-UTC
    one (no fallback); the date-only item is simply dropped."""
    decision = pd.Timestamp("2026-06-20T12:00:00Z")
    news = _news([
        _macro("2026-06-20T11:50:00Z", -0.5),                       # exact_utc -> used
        {"timestamp": "2026-06-20", "category": "macro", "symbol": None,
         "title": "", "body": "", "sentiment_score": 0.9},          # date_only -> dropped
    ])
    daily_ctx = {"2026-06-19": {"macro_news_sentiment": 0.42, "macro_news_count": 3}}
    feats, prov = nf.asof_news_for_decision(
        news, "BTCUSDT", UNIVERSE, decision, lookback_hours=24,
        safety_lag_seconds=300, daily_ctx=daily_ctx,
    )
    assert prov["news_mode"] == "intraday_asof"
    assert prov["source_feature_date"] is None
    assert feats["macro_news_count"] == 1
    assert feats["macro_news_sentiment"] == pytest.approx(-0.5)


# --------------------------------------------------------------------------- #
# available_at rule: real ingestion/scoring time, never published_at.
# --------------------------------------------------------------------------- #
def test_news_available_after_entry_before_close_is_used():
    """Position opened 10:00; close decision at 13:00. A headline whose scoring
    COMPLETED at 12:00 (after entry, before close decision - lag) is used."""
    close_decision = pd.Timestamp("2026-06-20T13:00:00Z")
    news = _news([{"timestamp": "2026-06-20T11:00:00Z", "category": "macro", "symbol": None,
                   "title": "", "body": "", "sentiment_score": -0.7,
                   "scored_at": "2026-06-20T12:00:00Z"}])
    feats, prov = nf.asof_news_for_decision(
        news, "BTCUSDT", UNIVERSE, close_decision, lookback_hours=24,
        safety_lag_seconds=300, live=True,
    )
    assert prov["news_mode"] == "intraday_asof"
    assert prov["available_at"] == "2026-06-20T12:00:00Z"
    assert feats["macro_news_count"] == 1
    assert feats["macro_news_sentiment"] == pytest.approx(-0.7)


def test_published_but_not_yet_scored_news_is_not_used():
    """A headline published long ago but only SCORED by the bot after the decision
    cutoff is not yet available to the model and must be excluded."""
    decision = pd.Timestamp("2026-06-20T12:00:00Z")
    news = _news([{"timestamp": "2026-06-20T08:00:00Z", "category": "macro", "symbol": None,
                   "title": "", "body": "", "sentiment_score": 0.9,
                   "scored_at": "2026-06-20T11:59:00Z"}])  # scored inside the 300s lag
    feats, prov = nf.asof_news_for_decision(
        news, "BTCUSDT", UNIVERSE, decision, lookback_hours=24,
        safety_lag_seconds=300, live=True,
    )
    assert feats["macro_news_count"] == 0
    assert feats["macro_news_sentiment"] == 0.0


def test_historical_available_at_proxy_is_published_plus_lag():
    """Historical (live=False): available_at = published_at + 300s. An item at the
    cutoff boundary is included; one second of published time later is excluded."""
    decision = pd.Timestamp("2026-06-20T12:00:00Z")  # cutoff = 11:55:00
    news = _news([
        _macro("2026-06-20T11:50:00Z", 0.5),    # available 11:55:00 == cutoff -> used
        _macro("2026-06-20T11:50:01Z", -0.5),    # available 11:55:01 > cutoff   -> excluded
    ])
    feats, prov = nf.asof_news_for_decision(
        news, "BTCUSDT", UNIVERSE, decision, lookback_hours=24,
        safety_lag_seconds=300, live=False, historical_lag_seconds=300,
    )
    assert prov["news_mode"] == "intraday_asof"
    assert feats["macro_news_count"] == 1
    assert feats["macro_news_sentiment"] == pytest.approx(0.5)
    assert prov["available_at"] == "2026-06-20T11:55:00Z"


def test_live_uses_real_availability_even_when_published_is_date_only():
    """Live: a date-only published_at is fine as long as the bot actually fetched
    and scored it (real availability time), so it can enter the intraday window."""
    decision = pd.Timestamp("2026-06-20T12:00:00Z")
    news = _news([{"timestamp": "2026-06-20", "category": "macro", "symbol": None,
                   "title": "", "body": "", "sentiment_score": 0.3,
                   "scored_at": "2026-06-20T11:50:00Z"}])
    feats, prov = nf.asof_news_for_decision(
        news, "BTCUSDT", UNIVERSE, decision, lookback_hours=24,
        safety_lag_seconds=300, live=True,
    )
    assert prov["news_mode"] == "intraday_asof"
    assert feats["macro_news_count"] == 1
    assert feats["macro_news_sentiment"] == pytest.approx(0.3)
