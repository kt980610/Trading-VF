import pandas as pd

from src import correlation as corr
from src import news_features as nf

UNIVERSE = ["ETHUSDT", "BTCUSDT"]
DAY_START = pd.Timestamp("2021-01-02T00:00:00Z")


def _provider(eth_btc=0.5):
    rows = [
        {
            "date": "2021-01-02",
            "lookback_days": 90,
            "symbols": UNIVERSE,
            "corr": {
                "ETHUSDT": {"ETHUSDT": 1.0, "BTCUSDT": eth_btc},
                "BTCUSDT": {"ETHUSDT": eth_btc, "BTCUSDT": 1.0},
            },
        }
    ]
    return corr.CorrelationProvider.from_rows(rows)


def _news():
    df = pd.DataFrame(
        [
            {"timestamp": "2021-01-01T12:00:00Z", "title": "btc", "body": "",
             "category": "symbol_specific", "symbol": "BTCUSDT", "sentiment_score": 0.8},
            {"timestamp": "2021-01-01T12:00:00Z", "title": "eth", "body": "",
             "category": "symbol_specific", "symbol": "ETHUSDT", "sentiment_score": 0.4},
            {"timestamp": "2021-01-01T12:00:00Z", "title": "macro", "body": "",
             "category": "macro", "symbol": None, "sentiment_score": 0.2},
        ]
    )
    return nf.prepare_news(df)


def _row(target, eth_btc=0.5):
    return nf.daily_cross_features_for_symbol(
        _news(), target, UNIVERSE, [DAY_START], _provider(eth_btc), lookback_days=2
    )[0]


def test_self_weight_is_one():
    row = _row("ETHUSDT")
    assert abs(row["news_sentiment_from_ETHUSDT_weighted"] - 0.4) < 1e-9


def test_cross_weight_uses_correlation():
    row = _row("ETHUSDT", eth_btc=0.5)
    assert abs(row["news_sentiment_from_BTCUSDT_weighted"] - 0.4) < 1e-9  # 0.5 * 0.8


def test_weighted_sentiment_is_signed_weighted_average():
    row = _row("ETHUSDT", eth_btc=0.5)
    # (1.0*0.4 + 0.5*0.8) / (1.0 + 0.5)
    assert abs(row["weighted_symbol_news_sentiment"] - (0.8 / 1.5)) < 1e-9


def test_count_uses_absolute_correlation():
    row = _row("ETHUSDT", eth_btc=-0.5)
    # |1.0|*1 + |-0.5|*1 = 1.5
    assert abs(row["weighted_symbol_news_count"] - 1.5) < 1e-9
    # Sentiment keeps the sign.
    assert row["news_sentiment_from_BTCUSDT_weighted"] < 0


def test_global_news_available_for_all_coins():
    eth = _row("ETHUSDT")
    btc = _row("BTCUSDT")
    assert eth["macro_news_sentiment"] == 0.2
    assert btc["macro_news_sentiment"] == 0.2


def test_btc_row_includes_weighted_eth_feature():
    row = _row("BTCUSDT", eth_btc=0.7)
    assert "news_sentiment_from_ETHUSDT_weighted" in row
    assert abs(row["news_sentiment_from_ETHUSDT_weighted"] - 0.7 * 0.4) < 1e-9
