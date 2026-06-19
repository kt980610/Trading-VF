import os

import pandas as pd

from src import rf_model as rfm


def _stump_model_json():
    return {
        "type": "random_forest_regressor",
        "n_estimators": 1,
        "features": ["LongEdge_Return"],
        "threshold": 0.0,
        "trees": [
            {
                "children_left": [1, -1, -1],
                "children_right": [2, -1, -1],
                "feature": [0, -2, -2],
                "threshold": [0.0, -2.0, -2.0],
                "value": [0.0, -1.0, 1.0],
            }
        ],
    }


def test_baseline_decides_on_summed_return_edges():
    model = rfm.BaselineModel(threshold=0.0)
    frame = pd.DataFrame(
        {"LongEdge_Return": [0.5, -0.5], "ShortEdge_Return": [0.1, 0.1]}
    )
    assert model.decide(frame) == [rfm.DECISION_CONTINUE, rfm.DECISION_CLOSE]


def test_train_rf_not_enough_rows():
    df = pd.DataFrame({"timestamp": [1, 2], rfm.TARGET_COLUMN: [0.1, 0.2]})
    result = rfm.train_rf(df, "BTCUSDT", min_training_rows=250)
    assert result["valid"] is False
    assert result["reason"] == rfm.REASON_NOT_ENOUGH_ROWS


def test_predictor_traverses_tree():
    predictor = rfm.predictor_from_json(_stump_model_json())
    frame = pd.DataFrame({"LongEdge_Return": [0.5, -0.5]})
    assert list(predictor.predict(frame)) == [1.0, -1.0]
    assert predictor.decide(frame) == [rfm.DECISION_CONTINUE, rfm.DECISION_CLOSE]


def test_save_and_load_artifacts(tmp_path):
    model_dir = str(tmp_path / "BTCUSDT")
    model_json = _stump_model_json()
    paths = rfm.save_artifacts(model_dir, model_json, model_json["features"], {"symbol": "BTCUSDT"})
    assert os.path.isfile(paths["model"])
    assert os.path.isfile(paths["features"])
    assert os.path.isfile(paths["metadata"])

    predictor = rfm.load_artifacts(model_dir)
    frame = pd.DataFrame({"LongEdge_Return": [0.5]})
    assert list(predictor.predict(frame)) == [1.0]


def test_get_policy_falls_back_to_baseline(tmp_path):
    policy = rfm.get_policy(str(tmp_path / "missing_symbol"))
    assert isinstance(policy, rfm.BaselineModel)
