"""CLI: build and persist per-symbol cumulative integral caches (section 22/24)."""

from __future__ import annotations

import argparse
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import load_config
    from src.snapshot_writer import load_snapshot
    from src.integral_cache import build_caches
else:
    from .config import load_config
    from .snapshot_writer import load_snapshot
    from .integral_cache import build_caches


def run(config_path: str) -> str:
    config = load_config(config_path)
    snapshot = load_snapshot(config.resolve(config.paths.distribution_snapshot))
    caches = build_caches(
        snapshot,
        window=config.rolling.window_days,
        grid_points=config.integration.grid_points,
    )

    payload = {
        "created_at": snapshot.get("created_at"),
        "window_days": config.rolling.window_days,
        # The PDF grid is a daily return grid (return_decimal), never price levels.
        "pdf_source_frequency": "1d",
        "grid_unit": "return_decimal",
        "symbols": {symbol: cache.to_dict() for symbol, cache in caches.items()},
    }
    out_path = config.resolve(config.paths.integral_cache)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    for symbol, cache in caches.items():
        print(f"{symbol} cache_grid={len(cache.grid)} denom={cache.denom:.6g}")
    print(f"integral cache written to {out_path}")
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build cumulative integral caches.")
    parser.add_argument("--config", default="config/distribution_config.yaml")
    args = parser.parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
