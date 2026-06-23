"""Build a *cap-weighted crypto market index* from local daily OHLCV.

CoinGecko's free Demo API caps historical data at the last 365 days, so it cannot
provide the 2018/2022 market-cap history the CAPM beta needs. Instead we build a
broad market-value series from the daily closes we already train on:

    market_value(day) = sum_i  close_i(day) * circulating_supply_i

The per-coin *current* circulating supply (a constant weight) is fetched once from
the keyless ``/coins/markets`` endpoint. Because supply is effectively constant
over any 30-day return window, the SERIES' returns -- all the beta leg consumes --
faithfully track cap-weighted market moves.

Output CSV matches ``scripts/fetch_market_cap.py`` exactly so the downstream
beta producer is unchanged: ``date,total_market_cap,n_constituents``.

Caveats:
* Membership is whatever symbols have daily data on a given day. Within a single
  training window the membership is constant, so in-window beta returns are clean;
  level jumps only occur across the multi-year gaps between windows (never inside).
* ``total_market_cap`` here is an index *level* (a cap proxy), not the true global
  cap; only its returns matter for beta.

Example (run on the server):
    .venv/bin/python scripts/build_market_index.py \
        --config config/distribution_config.yaml \
        --output data/market_cap_daily.csv --rate-limit-seconds 3
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Repo root for ``src`` imports; script dir for sibling-script imports.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from src.config import load_config  # noqa: E402
from src.data_loader import DataSourceError, load_daily_ohlcv  # noqa: E402
from src.capm_beta_daily import _coin_close_by_day  # noqa: E402
from fetch_market_cap import FetchError, _request_json, _write_csv_atomic  # noqa: E402

# Binance-style symbol -> CoinGecko coin id. Extend here when adding symbols.
_DEFAULT_SYMBOL_IDS: Dict[str, str] = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "SOLUSDT": "solana",
    "ADAUSDT": "cardano",
    "XRPUSDT": "ripple",
    "BNBUSDT": "binancecoin",
    "NEOUSDT": "neo",
    "LINKUSDT": "chainlink",
}

_API_BASE = "https://api.coingecko.com/api/v3"


def _parse_symbol_ids(spec: str) -> Dict[str, str]:
    """Parse ``SYM=id,SYM2=id2`` overrides; empty -> {}."""
    out: Dict[str, str] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise SystemExit(f"FATAL: bad --symbol-ids entry {chunk!r} (want SYM=coingecko_id)")
        sym, cid = chunk.split("=", 1)
        sym, cid = sym.strip().upper(), cid.strip().lower()
        if sym and cid:
            out[sym] = cid
    return out


def fetch_supplies(coin_ids: List[str], api_key: Optional[str]) -> Dict[str, float]:
    """Fetch current circulating supply per coin id via keyless /coins/markets."""
    ids_param = ",".join(coin_ids)
    url = (
        f"{_API_BASE}/coins/markets?vs_currency=usd&ids={ids_param}"
        f"&order=market_cap_desc&per_page=250&page=1&sparkline=false"
    )
    data = _request_json(url, api_key)
    if not isinstance(data, list):
        raise FetchError("unexpected /coins/markets response (not a list)")
    out: Dict[str, float] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        cid = row.get("id")
        supply = row.get("circulating_supply")
        if not cid or supply is None:
            continue
        try:
            sup_f = float(supply)
        except (TypeError, ValueError):
            continue
        if sup_f > 0.0:
            out[str(cid)] = sup_f
    return out


def build_index(
    per_symbol_value: Dict[str, "pd.Series"]
) -> List[Tuple[str, float, int]]:
    """Sum per-symbol cap-value series by date -> ``[(date, level, n), ...]``."""
    if not per_symbol_value:
        return []
    frame = pd.DataFrame(per_symbol_value).sort_index()
    totals = frame.sum(axis=1, skipna=True)
    counts = frame.notna().sum(axis=1)
    rows: List[Tuple[str, float, int]] = []
    for ts, total in totals.items():
        n = int(counts.loc[ts])
        if n <= 0 or not (total > 0.0):
            continue
        rows.append((ts.strftime("%Y-%m-%d"), float(total), n))
    return rows


def run(
    config_path: str,
    output: Optional[str],
    rate_limit_seconds: float,
    api_key_env: str,
    symbol_ids_override: str,
) -> str:
    config = load_config(config_path)
    out_path = output or config.resolve(config.beta.market_cap_path)
    api_key = os.environ.get(api_key_env) or None

    symbol_ids = dict(_DEFAULT_SYMBOL_IDS)
    symbol_ids.update(_parse_symbol_ids(symbol_ids_override))

    # Only symbols we have a CoinGecko id mapping for can be weighted.
    symbols = [s for s in config.symbols if s in symbol_ids]
    missing_map = [s for s in config.symbols if s not in symbol_ids]
    if missing_map:
        print(f"WARN: no coin-id mapping for {missing_map}; excluded from index")
    if not symbols:
        raise SystemExit("FATAL: no symbols with a coin-id mapping; nothing to build")

    coin_ids = sorted({symbol_ids[s] for s in symbols})
    supplies_by_id = fetch_supplies(coin_ids, api_key)
    print(f"supplies fetched for {len(supplies_by_id)}/{len(coin_ids)} coins")

    per_symbol_value: Dict[str, "pd.Series"] = {}
    for symbol in symbols:
        cid = symbol_ids[symbol]
        supply = supplies_by_id.get(cid)
        if supply is None:
            print(f"{symbol} skipped reason=no_circulating_supply ({cid})")
            continue
        try:
            daily_df = load_daily_ohlcv(config, symbol)
        except DataSourceError as exc:
            print(f"{symbol} skipped reason={exc.reason}")
            continue
        if daily_df is None or daily_df.empty:
            print(f"{symbol} skipped reason=missing_daily_source")
            continue
        close = _coin_close_by_day(daily_df)
        if close.empty:
            print(f"{symbol} skipped reason=empty_close")
            continue
        per_symbol_value[symbol] = close * supply
        print(
            f"{symbol} ({cid}) days={len(close)} "
            f"range={close.index.min().strftime('%Y-%m-%d')}.."
            f"{close.index.max().strftime('%Y-%m-%d')} supply={supply:.4g}"
        )

    if not per_symbol_value:
        raise SystemExit("FATAL: no symbol daily data usable; refusing to write empty index")

    rows = build_index(per_symbol_value)
    if not rows:
        raise SystemExit("FATAL: index aggregation produced no rows")
    _write_csv_atomic(out_path, rows)
    print(
        f"WROTE {out_path} rows={len(rows)} "
        f"range={rows[0][0]}..{rows[-1][0]} from {len(per_symbol_value)} symbols"
    )
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build cap-weighted market index from local daily OHLCV.")
    p.add_argument("--config", default="config/distribution_config.yaml")
    p.add_argument("--output", default=None, help="override beta.market_cap_path")
    p.add_argument("--rate-limit-seconds", type=float, default=3.0, help="(reserved) sleep between API calls")
    p.add_argument("--api-key-env", default="COINGECKO_DEMO_API_KEY", help="env var holding an optional CoinGecko Demo key")
    p.add_argument("--symbol-ids", default="", help="override map, e.g. 'NEOUSDT=neo,LINKUSDT=chainlink'")
    args = p.parse_args(argv)
    run(args.config, args.output, args.rate_limit_seconds, args.api_key_env, args.symbol_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
