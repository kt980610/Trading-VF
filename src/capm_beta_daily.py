"""Produce the daily per-symbol CAPM beta artifact (``capm_beta_daily.jsonl``).

Reads the market series CSV (``date,total_market_cap``) -- produced either by
``scripts/build_market_index.py`` (local cap-weighted index, full history) or
``scripts/fetch_market_cap.py`` (CoinGecko, last 365d only) -- and each symbol's
daily close, aligns them on a continuous daily calendar, and
computes the rolling beta per :mod:`src.capm_beta`. Output rows are
``{"date": "YYYY-MM-DD", "symbol": SYMBOL, "capm_beta": float}`` and are joined
into training/live with the existing D-1 (previous-completed-day) rule, so there
is no same-day look-ahead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Dict, List, Optional

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import load_config
    from src.data_loader import DataSourceError, load_daily_ohlcv
    from src import capm_beta as cb
else:
    from .config import load_config
    from .data_loader import DataSourceError, load_daily_ohlcv
    from . import capm_beta as cb


def _load_market_caps(path: str) -> "pd.Series":
    df = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}
    if "date" not in cols or "total_market_cap" not in cols:
        raise SystemExit(
            f"FATAL: {path} must have columns 'date,total_market_cap' (got {list(df.columns)})"
        )
    dates = pd.to_datetime(df[cols["date"]], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    caps = pd.to_numeric(df[cols["total_market_cap"]], errors="coerce")
    s = pd.Series(caps.values, index=dates).dropna()
    s = s[s.index.notna()]
    return s.groupby(level=0).last().sort_index()


def _coin_close_by_day(daily_df: "pd.DataFrame") -> "pd.Series":
    d = daily_df.copy()
    day = d["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    s = pd.Series(pd.to_numeric(d["close"], errors="coerce").values, index=day).dropna()
    return s.groupby(level=0).last().sort_index()


def compute_symbol_beta(
    coin_close: "pd.Series",
    market_caps: "pd.Series",
    return_lag: int,
    window: int,
) -> List[Dict[str, object]]:
    """Aligned rolling beta for one symbol -> ``[{date, capm_beta}, ...]``."""
    if coin_close.empty or market_caps.empty:
        return []
    start = max(coin_close.index.min(), market_caps.index.min())
    end = min(coin_close.index.max(), market_caps.index.max())
    if start > end:
        return []
    cal = pd.date_range(start, end, freq="D")
    coin_a = coin_close.reindex(cal).ffill()
    mkt_a = market_caps.reindex(cal).ffill()
    dates = [d.strftime("%Y-%m-%d") for d in cal]
    pairs = cb.rolling_beta_by_date(
        dates, coin_a.tolist(), mkt_a.tolist(), return_lag=return_lag, window=window
    )
    return [{"date": d, "capm_beta": float(b)} for d, b in pairs]


def _write_jsonl_atomic(rows: List[Dict[str, object]], path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".capm_beta_", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def run(
    config_path: str,
    market_cap: Optional[str] = None,
    output: Optional[str] = None,
) -> str:
    config = load_config(config_path)
    mcap_path = market_cap or config.resolve(config.beta.market_cap_path)
    out_path = output or config.resolve(config.beta.output_path)
    return_lag = int(config.beta.return_lag_days)
    window = int(config.beta.window_days)

    if not mcap_path or not os.path.isfile(mcap_path):
        raise SystemExit(
            f"FATAL: market-cap series not found: {mcap_path!r} "
            f"(run scripts/build_market_index.py first)"
        )
    market_caps = _load_market_caps(mcap_path)
    if market_caps.empty:
        raise SystemExit(f"FATAL: no usable rows in {mcap_path!r}")

    all_rows: List[Dict[str, object]] = []
    per_symbol_counts: Dict[str, int] = {}
    for symbol in config.symbols:
        try:
            daily_df = load_daily_ohlcv(config, symbol)
        except DataSourceError as exc:
            print(f"{symbol} skipped reason={exc.reason}")
            continue
        if daily_df is None or daily_df.empty:
            print(f"{symbol} skipped reason=missing_daily_source")
            continue
        coin_close = _coin_close_by_day(daily_df)
        sym_rows = compute_symbol_beta(coin_close, market_caps, return_lag, window)
        for r in sym_rows:
            all_rows.append({"date": r["date"], "symbol": symbol, "capm_beta": r["capm_beta"]})
        per_symbol_counts[symbol] = len(sym_rows)
        if sym_rows:
            print(
                f"{symbol} beta days={len(sym_rows)} "
                f"range={sym_rows[0]['date']}..{sym_rows[-1]['date']}"
            )
        else:
            print(f"{symbol} beta days=0 (insufficient overlap/history)")

    if not all_rows:
        raise SystemExit(
            "FATAL: no beta rows produced (no symbol/market-cap overlap); refusing "
            "to write an empty capm_beta_daily artifact"
        )

    all_rows.sort(key=lambda r: (r["date"], r["symbol"]))
    _write_jsonl_atomic(all_rows, out_path)
    print(
        f"capm beta written to {out_path}: rows={len(all_rows)} "
        f"symbols={sum(1 for v in per_symbol_counts.values() if v)} "
        f"return_lag={return_lag} window={window}"
    )
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build daily per-symbol CAPM beta artifact.")
    p.add_argument("--config", default="config/distribution_config.yaml")
    p.add_argument("--market-cap", default=None, help="override beta.market_cap_path")
    p.add_argument("--output", default=None, help="override beta.output_path")
    args = p.parse_args(argv)
    run(args.config, args.market_cap, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
