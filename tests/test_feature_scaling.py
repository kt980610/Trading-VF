import numpy as np
import pandas as pd

from src import feature_scaling as fs
from src import rf_dataset as rfd


def test_transform_classification():
    assert fs.transform_name("predicted_daily_volume") == "log1p"
    assert fs.transform_name("current_minute_volume") == "log1p"
    assert fs.transform_name("macro_news_count") == "log1p"
    assert fs.transform_name("weighted_symbol_news_count") == "log1p"
    assert fs.transform_name("LongEdge_Var") == "log1p"
    assert fs.transform_name("y") == "none"
    assert fs.transform_name("distance_to_liq_pct") == "none"


def test_log1p_transform_values():
    out = fs.apply_transform("log1p", np.array([0.0, 9.0, -5.0]))
    # Negative clamped to 0 before log1p.
    assert np.allclose(out, np.log1p([0.0, 9.0, 0.0]))


def test_imputer_uses_train_median():
    train = pd.DataFrame({"y": [1.0, 2.0, 3.0, np.nan]})
    scaler = fs.fit_scaler(train, ["y"], scaler_type="robust")
    assert scaler["medians"]["y"] == 2.0
    # Missing test value imputed with the train median -> centered to 0.
    x = fs.transform_frame(pd.DataFrame({"y": [np.nan]}), scaler)
    assert np.isclose(x[0, 0], 0.0)


def test_per_symbol_scalers_differ():
    a = pd.DataFrame({"y": [1.0, 2.0, 3.0, 100.0]})
    b = pd.DataFrame({"y": [1.0, 1.0, 1.0, 1.0]})
    sa = fs.fit_scaler(a, ["y"])
    sb = fs.fit_scaler(b, ["y"])
    assert (sa["center"]["y"], sa["scale"]["y"]) != (sb["center"]["y"], sb["scale"]["y"])


def test_transform_only_uses_train_statistics():
    train = pd.DataFrame({"y": [0.0, 10.0, 20.0, 30.0]})
    scaler = fs.fit_scaler(train, ["y"])
    x = fs.transform_frame(pd.DataFrame({"y": [10.0]}), scaler)
    expected = (10.0 - scaler["center"]["y"]) / scaler["scale"]["y"]
    assert np.isclose(x[0, 0], expected)


def test_volume_feature_is_log1p_scaled():
    train = pd.DataFrame({"predicted_daily_volume": [10.0, 100.0, 1000.0, 10000.0]})
    scaler = fs.fit_scaler(train, ["predicted_daily_volume"])
    assert scaler["transforms"]["predicted_daily_volume"] == "log1p"
    x = fs.transform_frame(pd.DataFrame({"predicted_daily_volume": [100.0]}), scaler)
    t = np.log1p(100.0)
    expected = (t - scaler["center"]["predicted_daily_volume"]) / scaler["scale"]["predicted_daily_volume"]
    assert np.isclose(x[0, 0], expected)


def test_raw_price_excluded_from_ml_features():
    cols = rfd.numeric_feature_columns()
    assert "CurrentPrice" not in cols
    assert "EntryPrice" not in cols
    assert "y" in cols and "distance_to_liq_pct" in cols


def test_model_feature_order_is_deterministic_and_includes_weighted():
    df = pd.DataFrame(
        {
            "LongEdge_Return": [0.0],
            "y": [0.0],
            "predicted_daily_volume": [0.0],
            "macro_news_sentiment": [0.0],
            "news_sentiment_from_ETHUSDT_weighted": [0.0],
            "side_code": [0],
            "mode_code": [0],
            "symbol": ["BTCUSDT"],
            rfd.TARGET_COLUMN: [0.0],
        }
    )
    cols1 = rfd.model_feature_columns(df)
    cols2 = rfd.model_feature_columns(df)
    assert cols1 == cols2
    assert "news_sentiment_from_ETHUSDT_weighted" in cols1
    assert "CurrentPrice" not in cols1
    assert "symbol" not in cols1 and rfd.TARGET_COLUMN not in cols1
