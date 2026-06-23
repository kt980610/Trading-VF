"""CLI: turn the GKG slot-feature SQLite backfill into daily news features.

Reads the compact per-slot aggregates produced by
``scripts/gkg_historical_feature_backfill.py`` (table ``gkg_slot_features``) and
rolls them up into ONE market-wide row per calendar day, written to
``news_features_daily.jsonl`` (the artifact both RF training and the Rust live
engine already consume).

Leakage / time semantics (D-1 design):
* Each output row is keyed by the calendar day the news OCCURRED (``date`` =
  ``observed_utc[:10]``), exactly like the volume daily artifact.
* The row carries NO ``symbol`` field: GKG news is market-wide, so it applies to
  every symbol (training accepts ``symbol=None`` rows; the Rust reader stores
  them under the universal ``""`` symbol).
* Consumers apply the previous-completed-day (D-1) join themselves: training via
  ``build_previous_day_provider`` and the live engine via
  ``NewsFeatures::previous_day_features``. So a minute on day D uses day D-1's
  news; this file never injects same-day look-ahead.

Column naming: every emitted feature uses the ``gkg_`` prefix so it is picked up
by the df-driven news-column discovery (``rf_dataset.news_like_columns``) and
flows, unchanged, into the RF feature schema and the Rust feature vector.

Groups (unique-by-URL within each slot, then pooled across the day's slots):
* crypto / btc / altcoins  -> DIRECT (URL+title) crypto buckets
* macro_conflict / macro_rates / macro_politics / macro_gold / macro_fx
  -> GKG theme-code macro buckets
Each group emits ``gkg_<group>_count`` (sum over the day's slots),
``gkg_<group>_tone_mean`` and ``gkg_<group>_tone_vol`` (count-pooled document
tone). ``gkg_coverage`` is the fraction of the day's slots that parsed OK.

Example:
    python -m src.gkg_news_daily --config config/distribution_config.yaml \
        --gkg-db data/gkg_4y_2h.sqlite
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import tempfile
from typing import Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import load_config
else:
    from .config import load_config

# group -> (count_column, tone_mean_column, tone_vol_column) in gkg_slot_features.
_GROUP_SOURCE = {
    "crypto": ("crypto_candidates", "crypto_tone_mean", "crypto_tone_vol"),
    "btc": ("btc_count", "btc_tone_mean", "btc_tone_vol"),
    "altcoins": ("altcoins_count", "altcoins_tone_mean", "altcoins_tone_vol"),
    "macro_conflict": ("macro_conflict_count", "macro_conflict_tone_mean", "macro_conflict_tone_vol"),
    "macro_rates": ("macro_rates_count", "macro_rates_tone_mean", "macro_rates_tone_vol"),
    "macro_politics": ("macro_politics_count", "macro_politics_tone_mean", "macro_politics_tone_vol"),
    "macro_gold": ("macro_gold_count", "macro_gold_tone_mean", "macro_gold_tone_vol"),
    "macro_fx": ("macro_fx_count", "macro_fx_tone_mean", "macro_fx_tone_vol"),
}


def _num(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


class _Pool:
    """Count-weighted pooled mean/variance for one group across a day's slots."""

    __slots__ = ("n", "wsum", "w2sum")

    def __init__(self):
        self.n = 0  # total candidate count (sum over slots)
        self.wsum = 0.0  # sum(count_i * mean_i)
        self.w2sum = 0.0  # sum(count_i * (vol_i^2 + mean_i^2)) = sum of x^2

    def add(self, count, mean, vol) -> None:
        c = int(_num(count))
        if c <= 0:
            return
        m = _num(mean)
        v = _num(vol)
        self.n += c
        self.wsum += c * m
        self.w2sum += c * (v * v + m * m)

    def finalize(self):
        if self.n <= 0:
            return 0, 0.0, 0.0
        mean = self.wsum / self.n
        var = max(0.0, self.w2sum / self.n - mean * mean)
        return self.n, mean, math.sqrt(var)


def _table_columns(conn: sqlite3.Connection) -> set:
    cur = conn.execute("PRAGMA table_info(gkg_slot_features)")
    return {row[1] for row in cur.fetchall()}


def aggregate_daily(conn: sqlite3.Connection) -> List[Dict[str, object]]:
    """Roll per-slot OK rows up into one market-wide row per calendar day."""
    cols = _table_columns(conn)
    if not cols:
        raise SystemExit("FATAL: table 'gkg_slot_features' not found in --gkg-db")

    needed = {"observed_utc", "status"}
    for c_count, c_mean, c_vol in _GROUP_SOURCE.values():
        needed.update((c_count, c_mean, c_vol))
    missing = needed - cols
    if missing:
        raise SystemExit(
            "FATAL: --gkg-db is missing expected columns "
            f"{sorted(missing)}; rebuild it with the current backfill tool"
        )

    # Per-day accumulators.
    pools: Dict[str, Dict[str, _Pool]] = {}
    slots_total: Dict[str, int] = {}
    slots_ok: Dict[str, int] = {}

    select_cols = ["observed_utc", "status"] + [
        col for g in _GROUP_SOURCE.values() for col in g
    ]
    query = f"SELECT {', '.join(select_cols)} FROM gkg_slot_features"
    for row in conn.execute(query):
        rec = dict(zip(select_cols, row))
        observed = rec.get("observed_utc") or ""
        if len(observed) < 10:
            continue
        date = observed[:10]
        slots_total[date] = slots_total.get(date, 0) + 1
        if rec.get("status") != "ok":
            continue
        slots_ok[date] = slots_ok.get(date, 0) + 1
        day_pools = pools.setdefault(date, {g: _Pool() for g in _GROUP_SOURCE})
        for g, (c_count, c_mean, c_vol) in _GROUP_SOURCE.items():
            day_pools[g].add(rec.get(c_count), rec.get(c_mean), rec.get(c_vol))

    rows: List[Dict[str, object]] = []
    for date in sorted(pools):
        out: Dict[str, object] = {"date": date}
        for g in _GROUP_SOURCE:
            n, mean, vol = pools[date][g].finalize()
            out[f"gkg_{g}_count"] = int(n)
            out[f"gkg_{g}_tone_mean"] = float(mean)
            out[f"gkg_{g}_tone_vol"] = float(vol)
        total = slots_total.get(date, 0)
        ok = slots_ok.get(date, 0)
        out["gkg_coverage"] = float(ok / total) if total else 0.0
        rows.append(out)
    return rows


def _write_jsonl_atomic(rows: List[Dict[str, object]], path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".gkg_news_", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def run(config_path: str, gkg_db: Optional[str], output: Optional[str], min_coverage: float) -> str:
    config = load_config(config_path)
    db_path = gkg_db or config.resolve(config.news.gkg_features_db)
    out_path = output or config.resolve(config.paths.news_features_daily)

    if not db_path or not os.path.isfile(db_path):
        raise SystemExit(f"FATAL: GKG feature DB not found: {db_path!r}")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = aggregate_daily(conn)
    finally:
        conn.close()

    if min_coverage > 0.0:
        rows = [r for r in rows if float(r.get("gkg_coverage", 0.0)) >= min_coverage]

    if not rows:
        raise SystemExit(
            "FATAL: no daily news rows produced (empty/uncovered GKG DB); refusing "
            "to write an empty news_features_daily artifact"
        )

    _write_jsonl_atomic(rows, out_path)
    first, last = rows[0]["date"], rows[-1]["date"]
    print(
        f"gkg daily news written to {out_path}: days={len(rows)} "
        f"range={first}..{last} min_coverage={min_coverage}"
    )
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build daily market-wide news features from the GKG SQLite backfill."
    )
    parser.add_argument("--config", default="config/distribution_config.yaml")
    parser.add_argument("--gkg-db", default=None, help="override news.gkg_features_db")
    parser.add_argument("--output", default=None, help="override paths.news_features_daily")
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.0,
        help="drop days whose slot coverage is below this fraction (default 0 = keep all)",
    )
    args = parser.parse_args(argv)
    if not (0.0 <= args.min_coverage <= 1.0):
        raise SystemExit(f"FATAL: --min-coverage={args.min_coverage} must be in [0, 1]")
    run(args.config, args.gkg_db, args.output, args.min_coverage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
