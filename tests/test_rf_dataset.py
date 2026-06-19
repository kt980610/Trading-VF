import pandas as pd

from src import rf_dataset as rfd


def _row(symbol, edge=0.0):
    features = {k: 0.0 for k in rfd.base_feature_columns()}
    features["LongEdge_Return"] = edge
    features["side"] = "long"
    features["mode"] = "HEDGED_BOTH_ACTIVE"
    return rfd.make_row(symbol, pd.Timestamp("2021-01-01", tz="UTC"), features, pnl_if_continue=2.0, pnl_if_close=1.0)


def test_policy_improvement_is_continue_minus_close():
    assert rfd.policy_improvement(5.0, 2.0) == 3.0


def test_target_column_set_on_row():
    row = _row("BTCUSDT")
    assert row[rfd.TARGET_COLUMN] == 1.0  # 2.0 - 1.0


def test_split_by_symbol_isolates_symbol():
    rows = [_row("BTCUSDT", 0.1), _row("BTCUSDT", 0.2), _row("ETHUSDT", 0.3)]
    df = rfd.to_frame(rows)
    btc = rfd.split_by_symbol(df, "BTCUSDT")
    eth = rfd.split_by_symbol(df, "ETHUSDT")
    assert set(btc["symbol"].unique()) == {"BTCUSDT"}
    assert len(btc) == 2
    assert "BTCUSDT" not in set(eth["symbol"].unique())


def test_frame_contains_required_feature_columns():
    df = rfd.to_frame([_row("BTCUSDT")])
    for col in rfd.INTEGRAL_EDGE_FEATURES:
        assert col in df.columns
    for col in ["predicted_daily_volume", "previous_day_real_volume", "current_minute_volume"]:
        assert col in df.columns


def test_numeric_feature_columns_use_encoded_categoricals():
    cols = rfd.numeric_feature_columns()
    assert "side_code" in cols and "mode_code" in cols
    assert "side" not in cols and "mode" not in cols


def test_encoding_applied_in_frame():
    df = rfd.to_frame([_row("BTCUSDT")])
    assert df["side_code"].iloc[0] == rfd.SIDE_ENCODING["long"]
    assert df["mode_code"].iloc[0] == rfd.MODE_ENCODING["HEDGED_BOTH_ACTIVE"]
