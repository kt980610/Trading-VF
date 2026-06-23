"""Build a daily *total crypto market cap* proxy from CoinGecko (free tier).

CoinGecko's historical GLOBAL market-cap chart is a paid-only endpoint, so this
tool approximates the total market cap by summing the per-coin historical market
caps of the current top-N coins (the top ~30 cover >90% of the true total, and
we only need the SERIES' returns for the CAPM market leg). The free, keyless
``/coins/{id}/market_chart?days=max`` endpoint provides daily history per coin.

Output CSV: ``date,total_market_cap,n_constituents`` (one row per UTC day,
ascending), written atomically.

Notes / caveats:
* Constituent membership uses *today's* top-N (mild survivorship bias). Within any
  single ~4-month training window the membership is effectively constant, so the
  30-day returns used for beta are unaffected.
* Free tier is rate-limited and occasionally flaky; downloads retry with backoff.
* HTTPS + TLS verification only. An optional CoinGecko Demo key (header
  ``x-cg-demo-api-key``) is read from an env var to raise the rate limit.

Example (run on the server):
    .venv/bin/python scripts/fetch_market_cap.py \
        --top-n 30 --output data/market_cap_daily.csv --rate-limit-seconds 3
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

_API_BASE = "https://api.coingecko.com/api/v3"
_USER_AGENT = "trading-vf-marketcap/1.0 (read-only research)"
_TIMEOUT_SECONDS = 60
_MAX_ATTEMPTS = 5
_BACKOFF_SECONDS = 5.0


class FetchError(Exception):
    """Non-retryable fetch failure."""


def _request_json(url: str, api_key: Optional[str]) -> object:
    if not url.startswith("https://"):
        raise FetchError(f"refusing non-HTTPS url: {url}")
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
    if api_key:
        headers["x-cg-demo-api-key"] = api_key
    context = ssl.create_default_context()
    req = urllib.request.Request(url, headers=headers, method="GET")
    last_err = "unknown error"
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS, context=context) as resp:
                status = getattr(resp, "status", None) or 200
                if status != 200:
                    raise FetchError(f"http_status={status} for {url}")
                return json.loads(resp.read().decode("utf-8"))
        except FetchError:
            raise
        except urllib.error.HTTPError as exc:
            if exc.code == 429 or 500 <= exc.code < 600:
                last_err = f"http_status={exc.code}"
            else:
                raise FetchError(f"http_status={exc.code} for {url}") from None
        except OSError as exc:  # timeout / connection / ssl
            last_err = f"network error: {exc}"
        if attempt < _MAX_ATTEMPTS:
            time.sleep(_BACKOFF_SECONDS * attempt)
    raise FetchError(f"{last_err} for {url} after {_MAX_ATTEMPTS} attempts")


def fetch_top_ids(top_n: int, api_key: Optional[str]) -> List[str]:
    url = (
        f"{_API_BASE}/coins/markets?vs_currency=usd&order=market_cap_desc"
        f"&per_page={int(top_n)}&page=1&sparkline=false"
    )
    data = _request_json(url, api_key)
    if not isinstance(data, list):
        raise FetchError("unexpected /coins/markets response (not a list)")
    ids = [str(row["id"]) for row in data if isinstance(row, dict) and row.get("id")]
    if not ids:
        raise FetchError("no coin ids returned by /coins/markets")
    return ids


def _to_utc_day(ms: float) -> str:
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def fetch_coin_market_caps(coin_id: str, api_key: Optional[str]) -> Dict[str, float]:
    """Return ``{utc_date: market_cap}`` (last sample per day) for one coin."""
    # No ``interval`` param: free tier auto-returns daily granularity for days=max.
    url = f"{_API_BASE}/coins/{coin_id}/market_chart?vs_currency=usd&days=max"
    data = _request_json(url, api_key)
    if not isinstance(data, dict) or "market_caps" not in data:
        raise FetchError(f"unexpected market_chart response for {coin_id}")
    out: Dict[str, float] = {}
    for point in data["market_caps"]:
        try:
            ms, cap = point[0], point[1]
        except (TypeError, IndexError):
            continue
        if cap is None:
            continue
        try:
            cap_f = float(cap)
        except (TypeError, ValueError):
            continue
        if cap_f <= 0.0:
            continue
        out[_to_utc_day(float(ms))] = cap_f  # last write per day wins (latest)
    return out


def aggregate_total(
    per_coin: Dict[str, Dict[str, float]]
) -> List[Tuple[str, float, int]]:
    """Sum per-coin caps by date -> ``[(date, total_cap, n_constituents), ...]``."""
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for caps in per_coin.values():
        for date, cap in caps.items():
            totals[date] = totals.get(date, 0.0) + cap
            counts[date] = counts.get(date, 0) + 1
    return [(d, totals[d], counts[d]) for d in sorted(totals)]


def _write_csv_atomic(path: str, rows: List[Tuple[str, float, int]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="mktcap_", suffix=".csv", dir=os.path.dirname(os.path.abspath(path)) or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write("date,total_market_cap,n_constituents\n")
            for date, cap, n in rows:
                fh.write(f"{date},{cap:.6f},{n}\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build daily total crypto market cap proxy (CoinGecko free tier).")
    p.add_argument("--top-n", type=int, default=30, help="number of top coins to sum (default 30)")
    p.add_argument("--coins", default="", help="optional explicit comma list of CoinGecko ids (overrides --top-n)")
    p.add_argument("--output", default="data/market_cap_daily.csv")
    p.add_argument("--rate-limit-seconds", type=float, default=3.0, help="sleep between coin requests")
    p.add_argument("--api-key-env", default="COINGECKO_DEMO_API_KEY", help="env var holding an optional CoinGecko Demo key")
    args = p.parse_args(argv)

    api_key = os.environ.get(args.api_key_env) or None
    if args.coins.strip():
        ids = [c.strip() for c in args.coins.split(",") if c.strip()]
    else:
        ids = fetch_top_ids(args.top_n, api_key)
    print(f"constituents={len(ids)}: {', '.join(ids)}")

    per_coin: Dict[str, Dict[str, float]] = {}
    for idx, cid in enumerate(ids, start=1):
        if idx > 1 and args.rate_limit_seconds > 0:
            time.sleep(args.rate_limit_seconds)
        try:
            caps = fetch_coin_market_caps(cid, api_key)
        except FetchError as exc:
            print(f"[{idx}/{len(ids)}] {cid} FAILED: {exc}", file=sys.stderr)
            continue
        per_coin[cid] = caps
        days = len(caps)
        first = min(caps) if caps else "-"
        last = max(caps) if caps else "-"
        print(f"[{idx}/{len(ids)}] {cid} days={days} range={first}..{last}")

    if not per_coin:
        print("FATAL: no coin data fetched", file=sys.stderr)
        return 2

    rows = aggregate_total(per_coin)
    _write_csv_atomic(args.output, rows)
    print(
        f"WROTE {args.output} rows={len(rows)} "
        f"range={rows[0][0]}..{rows[-1][0]} from {len(per_coin)} coins"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
