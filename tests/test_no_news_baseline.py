"""No-news baseline vs news-enabled training context.

When ``news.enabled: false`` the RF training context must carry NO news feature
columns (even if a stale daily news artifact exists on disk), so the RF training
schema excludes them instead of learning on zero/constant news features. When
``news.enabled: true`` the legacy daily news join still injects them.
"""

from __future__ import annotations

import json

from src.config import load_config
from src.main_train_rf import _build_context_provider
from src import rf_dataset as rfd


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _setup(tmp_path, news_enabled: bool):
    data = tmp_path / "data"
    # Previous-completed-day (2024-09-28) volume + news artifacts.
    _write_jsonl(
        data / "predicted_daily_volume.jsonl",
        [{"symbol": "BTCUSDT", "date": "2024-09-28",
          "predicted_daily_volume": 100.0, "previous_day_real_volume": 80.0}],
    )
    _write_jsonl(
        data / "news_features_daily.jsonl",
        [{"symbol": "BTCUSDT", "date": "2024-09-28",
          "macro_news_sentiment": 0.7, "macro_news_count": 3}],
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "symbols:\n  - BTCUSDT\n"
        "news:\n"
        f"  enabled: {'true' if news_enabled else 'false'}\n"
        "  intraday_news_enabled: false\n",
        encoding="utf-8",
    )
    return load_config(str(cfg_path))


def _news_keys(feats):
    return [
        k for k in feats
        if k.endswith("_news_sentiment") or k.endswith("_news_count") or k.endswith("_weighted")
    ]


def test_no_news_baseline_injects_no_news_columns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _setup(tmp_path, news_enabled=False)
    provider = _build_context_provider(config, "BTCUSDT")
    feats = provider("2024-09-29T00:05:00Z")
    # Volume context is still present; news columns are completely absent.
    assert feats.get("predicted_daily_volume") == 100.0
    assert _news_keys(feats) == []


def test_news_enabled_injects_daily_news_columns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _setup(tmp_path, news_enabled=True)
    provider = _build_context_provider(config, "BTCUSDT")
    feats = provider("2024-09-29T00:05:00Z")
    assert feats.get("predicted_daily_volume") == 100.0
    assert feats.get("macro_news_sentiment") == 0.7
    assert "macro_news_sentiment" in _news_keys(feats)


def test_training_schema_excludes_news_when_columns_absent():
    # The RF close-classifier schema is df-driven: with no news columns present
    # (the no-news baseline), none of the NEWS_FEATURES enter the feature order.
    import pandas as pd

    df = pd.DataFrame(
        {
            "LongEdge_Return": [0.1, -0.2],
            "ShortEdge_Return": [0.0, 0.3],
            "side_code": [0, 1],
            "mode_code": [0, 0],
            rfd.TARGET_COLUMN: [0.0, 1.0],
        }
    )
    cols = rfd.close_classifier_feature_columns(df)
    assert all(c not in cols for c in rfd.NEWS_FEATURES)
    assert rfd.news_like_columns(df) == []
    # The edge features are still there (it is a real, trainable schema).
    assert "LongEdge_Return" in cols


def test_training_schema_includes_news_when_columns_present():
    import pandas as pd

    df = pd.DataFrame(
        {
            "LongEdge_Return": [0.1, -0.2],
            "side_code": [0, 1],
            "mode_code": [0, 0],
            "macro_news_sentiment": [0.7, -0.1],
            "macro_news_count": [3.0, 1.0],
            rfd.TARGET_COLUMN: [0.0, 1.0],
        }
    )
    cols = rfd.close_classifier_feature_columns(df)
    assert "macro_news_sentiment" in cols
    assert "macro_news_count" in cols
