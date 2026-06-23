"""Tests for the disjoint training-window design.

Covers:
* ``config._parse_training_windows`` (string + mapping forms),
* ``rf_classifier.trade_group_split`` window-awareness (every ``train_window``
  appears in train/validation/test instead of one regime monopolising a split).
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import _parse_training_windows  # noqa: E402
from src import rf_classifier as rfc  # noqa: E402


def test_parse_windows_string_slash():
    out = _parse_training_windows(
        ["2018-08-01/2018-11-01", "2022-06-01/2022-09-01", "2026-04-21/2026-06-21"]
    )
    assert out == [
        ("2018-08-01", "2018-11-01"),
        ("2022-06-01", "2022-09-01"),
        ("2026-04-21", "2026-06-21"),
    ]


def test_parse_windows_mapping_form():
    out = _parse_training_windows(
        [{"start": "2018-08-01", "end": "2018-11-01"}]
    )
    assert out == [("2018-08-01", "2018-11-01")]


def test_parse_windows_alt_delims_and_garbage():
    assert _parse_training_windows(["2018-08-01..2018-11-01"]) == [
        ("2018-08-01", "2018-11-01")
    ]
    assert _parse_training_windows(["2018-08-01,2018-11-01"]) == [
        ("2018-08-01", "2018-11-01")
    ]
    # Non-list / malformed entries are ignored, not crashed on.
    assert _parse_training_windows(None) == []
    assert _parse_training_windows(["nodelimiter"]) == []


def _synthetic_trades(n_per_window: int, windows=(0, 1, 2)) -> pd.DataFrame:
    """One row per trade across several windows with distinct k per window."""
    rows = []
    for w in windows:
        base_day = pd.Timestamp("2018-01-01", tz="UTC") + pd.Timedelta(days=400 * w)
        for t in range(n_per_window):
            tid = w * 100_000_000 + t
            rows.append(
                {
                    "trade_id": tid,
                    "entry_timestamp": base_day + pd.Timedelta(minutes=t),
                    "timestamp": base_day + pd.Timedelta(minutes=t),
                    "train_window": w,
                    "halving_cycle_id": 2 + w,
                    "close_label": t % 2,
                }
            )
    return pd.DataFrame(rows)


def test_window_aware_split_covers_all_regimes():
    df = _synthetic_trades(n_per_window=20)  # 60 trades, 3 windows, k in {2,3,4}
    train, val, test = rfc.trade_group_split(df)
    # Every window/k must appear in every split (the whole point of the design).
    for part in (train, val, test):
        assert set(part["halving_cycle_id"]) == {2, 3, 4}
        assert set(part["train_window"]) == {0, 1, 2}
    # Splits are disjoint by trade.
    ids = [set(p["trade_id"]) for p in (train, val, test)]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert len(train) + len(val) + len(test) == len(df)


def test_single_window_falls_back_to_time_ordered():
    df = _synthetic_trades(n_per_window=40, windows=(1,))
    train, val, test = rfc.trade_group_split(df)
    # Time-ordered: train holds the earliest trades, test the latest.
    assert train["entry_timestamp"].max() <= val["entry_timestamp"].min()
    assert val["entry_timestamp"].max() <= test["entry_timestamp"].min()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
