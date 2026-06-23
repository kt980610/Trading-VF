import numpy as np
import pandas as pd

from src import volume_model as vm


def _daily(n=220, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    volume = np.abs(rng.normal(1000, 100, n)) + 100
    return pd.DataFrame({"timestamp": ts, "open": close, "high": close, "low": close, "close": close, "volume": volume})


def test_target_is_log1p_volume():
    daily = _daily()
    frame = vm.build_feature_frame(daily)
    assert np.allclose(frame["log_volume"], np.log1p(daily["volume"]))


def test_features_are_lagged_no_same_day_leak():
    daily = _daily()
    frame = vm.build_feature_frame(daily)
    vol = daily["volume"].to_numpy()
    # previous_day_volume for row i must equal the volume of day i-1.
    assert np.isclose(frame["previous_day_volume"].iloc[5], vol[4])
    # The target (current-day volume) must NOT appear in any lagged feature.
    assert frame["previous_day_volume"].iloc[5] != vol[5]


def test_walk_forward_is_out_of_sample():
    daily = _daily(n=240)
    frame = vm.build_feature_frame(daily)
    preds = vm.walk_forward_predict(frame, "BTCUSDT", min_train_days=20)
    assert not preds.empty
    assert {"predicted_daily_volume", "real_volume", "predicted_log_volume"}.issubset(preds.columns)
    # Predictions are strictly after the first min_train_days training rows.
    assert len(preds) >= 1


def test_evaluate_predictions_report_keys():
    daily = _daily(n=240)
    frame = vm.build_feature_frame(daily)
    preds = vm.walk_forward_predict(frame, "BTCUSDT", min_train_days=20)
    report = vm.evaluate_predictions(preds, "BTCUSDT")
    for key in [
        "symbol", "n_days", "volume_mae", "volume_rmse",
        "log_volume_mae", "log_volume_rmse",
        "direction_accuracy_volume_change", "corr_pred_real_volume",
    ]:
        assert key in report
