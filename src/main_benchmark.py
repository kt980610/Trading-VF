"""CLI: cached vs full integral speed benchmark (section 23)."""

from __future__ import annotations

import argparse
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import load_config
    from src.snapshot_writer import load_snapshot
    from src import speed_benchmark as sb
else:
    from .config import load_config
    from .snapshot_writer import load_snapshot
    from . import speed_benchmark as sb


def run(config_path: str, n_minutes: int = 2000) -> str:
    config = load_config(config_path)
    snapshot = load_snapshot(config.resolve(config.paths.distribution_snapshot))

    rows = []
    for symbol in config.symbols:
        entry = snapshot.get("symbols", {}).get(symbol)
        if not entry or not entry.get("valid"):
            print(f"{symbol} skipped reason=no_valid_distribution")
            continue
        metrics = sb.benchmark_symbol(symbol, entry, n_minutes=n_minutes, window=config.rolling.window_days)
        rows.append(metrics)
        print(f"{symbol} speedup={metrics['speedup_ratio']:.1f}x cached_ms/min={metrics['cached_integral_runtime_ms_per_minute']:.4f}")

    out_path = config.resolve(config.paths.speed_benchmark)
    sb.write_benchmark(rows, out_path)
    print(f"speed benchmark written to {out_path}")
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark cached vs full integral evaluation.")
    parser.add_argument("--config", default="config/distribution_config.yaml")
    parser.add_argument("--minutes", type=int, default=2000)
    args = parser.parse_args(argv)
    run(args.config, n_minutes=args.minutes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
