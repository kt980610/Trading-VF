"""GDELT RF-feature semantics: tone is a SEPARATE numeric feature (never a
FinBERT score), coverage/provenance features only appear when a provider source
is declared, and the training/live as-of join produces an identical schema.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import news_features as nf
from src import rf_dataset as rd


def _news_df():
    raw = pd.DataFrame(
        [
            {"timestamp": "2026-06-20T10:00:00Z", "category": "macro", "symbol": None,
             "title": "Fed holds rates", "body": "", "sentiment_score": 0.3, "gdelt_tone": 2.0},
            {"timestamp": "2026-06-20T10:30:00Z", "category": "macro", "symbol": None,
             "title": "", "body": "", "sentiment_score": None, "gdelt_tone": -4.0},
        ]
    )
    return nf.prepare_news(raw, backend="lexicon")


def test_prepare_news_keeps_gdelt_tone_column():
    df = _news_df()
    assert "gdelt_tone" in df.columns
    assert set(df["gdelt_tone"].tolist()) == {2.0, -4.0}


def test_gdelt_tone_is_separate_from_sentiment():
    df = _news_df()
    feats, prov = nf.asof_news_for_decision(
        df, "BTCUSDT", ["BTCUSDT"], pd.Timestamp("2026-06-20T12:00:00Z"),
        lookback_hours=24, safety_lag_seconds=300, live=True, news_source="gdelt",
    )
    # Tone is the mean of the in-window source tone, NOT the sentiment.
    assert feats["gdelt_tone"] == pytest.approx((2.0 + -4.0) / 2.0)
    # Sentiment features still exist and are independent of tone.
    assert "macro_news_sentiment" in feats
    assert feats["macro_news_sentiment"] != feats["gdelt_tone"]
    assert prov["news_source"] == "gdelt"


def test_coverage_features_present_only_with_source():
    df = _news_df()
    asof = pd.Timestamp("2026-06-20T12:00:00Z")

    with_src, _ = nf.asof_news_for_decision(
        df, "BTCUSDT", ["BTCUSDT"], asof, lookback_hours=24, safety_lag_seconds=300,
        live=True, news_source="gdelt", coverage_ratio=0.75,
    )
    for key in ("news_available", "news_coverage_ratio", "gdelt_tone"):
        assert key in with_src
    assert with_src["news_available"] == 1.0
    assert with_src["news_coverage_ratio"] == pytest.approx(0.75)

    # Without a declared source the legacy schema is unchanged (no coverage cols).
    without_src, _ = nf.asof_news_for_decision(
        df, "BTCUSDT", ["BTCUSDT"], asof, lookback_hours=24, safety_lag_seconds=300,
        live=True,
    )
    for key in ("news_available", "news_coverage_ratio", "gdelt_tone"):
        assert key not in without_src


def test_news_available_zero_when_window_empty():
    df = _news_df()
    # Decision far before any news -> empty window -> news_available 0.
    feats, _ = nf.asof_news_for_decision(
        df, "BTCUSDT", ["BTCUSDT"], pd.Timestamp("2026-06-19T00:00:00Z"),
        lookback_hours=24, safety_lag_seconds=300, live=True, news_source="gdelt",
    )
    assert feats["news_available"] == 0.0
    assert feats["gdelt_tone"] == 0.0


def test_train_live_asof_schema_parity():
    """The same as-of join feeds training and live: identical feature schema."""
    df = _news_df()
    asof = pd.Timestamp("2026-06-20T12:00:00Z")
    train, _ = nf.asof_news_for_decision(
        df, "BTCUSDT", ["BTCUSDT"], asof, lookback_hours=24, safety_lag_seconds=300,
        live=False, news_source="gdelt",
    )
    live, _ = nf.asof_news_for_decision(
        df, "BTCUSDT", ["BTCUSDT"], asof, lookback_hours=24, safety_lag_seconds=300,
        live=True, news_source="gdelt",
    )
    assert set(train.keys()) == set(live.keys())
    assert "gdelt_tone" in train and "news_available" in train


def test_observed_utc_historical_available_at_uses_source_seen_at():
    """Historical proxy for a GDELT (observed_utc) row is source_seen_at + lag,
    and the safety rule is unchanged (observed instants are eligible, just like
    verified publish instants)."""
    seen = "2026-06-20T10:00:00Z"
    df = nf.prepare_news(
        pd.DataFrame([
            {"timestamp": seen, "category": "macro", "symbol": None,
             "title": "Fed", "body": "", "sentiment_score": 0.3,
             "timestamp_quality": nf.TS_OBSERVED},
        ]),
        backend="lexicon",
    )
    lag = 300
    avail = nf.compute_available_at(df, live=False, historical_lag_seconds=lag)
    expected = pd.Timestamp(seen) + pd.Timedelta(seconds=lag)
    assert avail.iloc[0] == expected

    # The observed row IS admissible to the rolling window once available (rule
    # not relaxed: a decision before available_at sees nothing).
    before, _ = nf.asof_news_for_decision(
        df, "BTCUSDT", ["BTCUSDT"], expected - pd.Timedelta(seconds=1),
        lookback_hours=24, safety_lag_seconds=0, live=False, news_source="gdelt",
    )
    after, prov = nf.asof_news_for_decision(
        df, "BTCUSDT", ["BTCUSDT"], expected + pd.Timedelta(seconds=1),
        lookback_hours=24, safety_lag_seconds=0, live=False, news_source="gdelt",
    )
    assert before["news_available"] == 0.0
    assert after["news_available"] == 1.0
    # Provenance reflects the observation semantics, not a publish time.
    assert prov["timestamp_quality"] == nf.TS_OBSERVED


def test_provenance_features_enter_close_schema_only_when_present():
    base = {
        "LongEdge_Return": 0.1, "ShortEdge_Return": 0.1,
        "side_code": 0, "mode_code": 0,
    }
    df_no = pd.DataFrame([base])
    cols_no = rd.close_classifier_feature_columns(df_no)
    assert "gdelt_tone" not in cols_no
    assert "news_available" not in cols_no

    df_yes = pd.DataFrame([{**base, "gdelt_tone": 1.0, "news_available": 1.0,
                            "news_coverage_ratio": 1.0}])
    cols_yes = rd.close_classifier_feature_columns(df_yes)
    assert "gdelt_tone" in cols_yes
    assert "news_available" in cols_yes
    assert "news_coverage_ratio" in cols_yes
