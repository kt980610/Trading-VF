import pandas as pd

from src import news_features as nf


def _news_df():
    return pd.DataFrame(
        [
            # Included: 12h before day start, macro.
            {"timestamp": "2021-01-01T12:00:00Z", "title": "macro update", "body": "",
             "category": "macro", "symbol": None, "sentiment_score": 0.5},
            # Excluded: same day, after day start (leakage).
            {"timestamp": "2021-01-02T01:00:00Z", "title": "macro later", "body": "",
             "category": "macro", "symbol": None, "sentiment_score": -0.9},
            # Excluded: older than the 1-day lookback window.
            {"timestamp": "2020-12-28T00:00:00Z", "title": "old", "body": "",
             "category": "macro", "symbol": None, "sentiment_score": 0.1},
        ]
    )


def test_only_news_before_day_start_used():
    news = nf.prepare_news(_news_df())
    day_start = pd.Timestamp("2021-01-02T00:00:00Z")
    rows = nf.daily_features_for_symbol(news, "BTCUSDT", [day_start], lookback_days=1)
    row = rows[0]
    # Only the 2021-01-01T12:00 macro item is in [D-1day, D).
    assert row["macro_news_count"] == 1
    assert row["macro_news_sentiment"] == 0.5


def test_no_leakage_from_same_day_news():
    news = nf.prepare_news(_news_df())
    day_start = pd.Timestamp("2021-01-02T00:00:00Z")
    rows = nf.daily_features_for_symbol(news, "BTCUSDT", [day_start], lookback_days=1)
    # The -0.9 same-day item must not pull the sentiment negative.
    assert rows[0]["macro_news_sentiment"] == 0.5


def test_empty_news_yields_zero_features():
    empty = nf.prepare_news(pd.DataFrame(columns=["timestamp", "title", "body", "category", "symbol", "sentiment_score"]))
    rows = nf.daily_features_for_symbol(empty, "BTCUSDT", [pd.Timestamp("2021-01-02T00:00:00Z")])
    for cat in nf.CATEGORIES:
        assert rows[0][f"{cat}_news_count"] == 0
        assert rows[0][f"{cat}_news_sentiment"] == 0.0


def test_symbol_specific_matches_symbol():
    df = pd.DataFrame(
        [
            {"timestamp": "2021-01-01T10:00:00Z", "title": "btc news", "body": "",
             "category": "symbol_specific", "symbol": "BTCUSDT", "sentiment_score": 0.7},
            {"timestamp": "2021-01-01T10:00:00Z", "title": "eth news", "body": "",
             "category": "symbol_specific", "symbol": "ETHUSDT", "sentiment_score": -0.3},
        ]
    )
    news = nf.prepare_news(df)
    rows = nf.daily_features_for_symbol(news, "BTCUSDT", [pd.Timestamp("2021-01-02T00:00:00Z")], lookback_days=2)
    assert rows[0]["symbol_specific_news_count"] == 1
    assert rows[0]["symbol_specific_news_sentiment"] == 0.7
