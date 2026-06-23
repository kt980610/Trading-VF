"""Network-free tests for the Binance archive fetch tool's pure helpers."""

import os
import sys
from datetime import date, datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main_fetch_market_data import (
    atomic_write_csv,
    combine_and_dedupe,
    merge_with_existing,
    month_iter,
    only_closed_days,
    parse_klines_csv,
)


def test_month_iter_inclusive():
    months = month_iter(date(2022, 11, 1), date(2023, 2, 15))
    assert months == [(2022, 11), (2022, 12), (2023, 1), (2023, 2)]


def test_parse_klines_csv_headerless_ms():
    # open_time(ms), o,h,l,c,v, close_time, qv, n, tbb, tbq, ignore
    text = (
        "1640995200000,46000,47000,45000,46500,1000,1640995259999,1,1,1,1,0\n"
        "1640995260000,46500,46800,46400,46700,900,1640995319999,1,1,1,1,0\n"
    )
    df = parse_klines_csv(text)
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df["close"].iloc[0] == 46500
    assert str(df["timestamp"].iloc[0]) == "2022-01-01 00:00:00+00:00"


def test_parse_klines_csv_with_header():
    text = (
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_base,taker_buy_quote,ignore\n"
        "1640995200000,46000,47000,45000,46500,1000,1,1,1,1,1,0\n"
    )
    df = parse_klines_csv(text)
    assert len(df) == 1
    assert df["high"].iloc[0] == 47000


def test_combine_and_dedupe_keeps_last_and_sorts():
    a = parse_klines_csv("1640995200000,1,1,1,10,1,1,1,1,1,1,0\n")
    b = parse_klines_csv(
        "1640995260000,1,1,1,20,1,1,1,1,1,1,0\n"
        "1640995200000,1,1,1,99,1,1,1,1,1,1,0\n"  # duplicate ts, newer value 99
    )
    out = combine_and_dedupe([a, b])
    assert len(out) == 2
    # duplicate resolved to the last-seen value, order ascending
    assert out["close"].iloc[0] == 99
    assert out["close"].iloc[1] == 20


def test_only_closed_days_drops_today():
    now = datetime(2024, 3, 10, 12, 0, tzinfo=timezone.utc)
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-03-08", "2024-03-09", "2024-03-10"], utc=True
            ),
            "open": [1, 2, 3], "high": [1, 2, 3], "low": [1, 2, 3],
            "close": [1, 2, 3], "volume": [1, 2, 3],
        }
    )
    out = only_closed_days(df, now=now)
    assert list(out["timestamp"].dt.strftime("%Y-%m-%d")) == ["2024-03-08", "2024-03-09"]


def test_merge_with_existing_epoch_ms_not_misparsed(tmp_path):
    # Regression: an existing file storing epoch-MS *integers* must not be read as
    # nanoseconds (which would map 2022 rows to 1970-01-01) when merging.
    path = str(tmp_path / "BTCUSDT_1m.csv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("timestamp,open,high,low,close,volume\n")
        fh.write("1654041600000,1,1,1,30,1\n")  # 2022-06-01 00:00:00 UTC, ms int
        fh.write("1654041660000,1,1,1,31,1\n")  # 2022-06-01 00:01:00 UTC
    new = parse_klines_csv("1654041720000,1,1,1,32,1,1,1,1,1,1,0\n")  # +2 min
    merged = merge_with_existing(path, new)
    years = set(merged["timestamp"].dt.year.tolist())
    assert years == {2022}, f"epoch-ms misparsed: years={years}"
    assert len(merged) == 3
    assert list(merged["close"]) == [30, 31, 32]


def test_atomic_write_and_merge_no_duplicates(tmp_path):
    path = str(tmp_path / "BTCUSDT_daily.csv")
    first = parse_klines_csv(
        "1640995200000,1,1,1,10,1,1,1,1,1,1,0\n"
        "1641081600000,1,1,1,11,1,1,1,1,1,1,0\n"
    )
    atomic_write_csv(first, path)
    assert os.path.isfile(path)

    # Re-fetch overlapping + new row; merge must not duplicate the overlap.
    second = parse_klines_csv(
        "1641081600000,1,1,1,11,1,1,1,1,1,1,0\n"
        "1641168000000,1,1,1,12,1,1,1,1,1,1,0\n"
    )
    merged = merge_with_existing(path, second)
    assert len(merged) == 3
    assert list(merged["close"]) == [10, 11, 12]
