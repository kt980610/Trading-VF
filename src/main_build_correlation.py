"""CLI: build leakage-safe daily cross-coin correlation matrices (section 2)."""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import load_config
    from src.data_loader import load_daily
    from src import correlation as corr
else:
    from .config import load_config
    from .data_loader import load_daily
    from . import correlation as corr


def run(config_path: str) -> str:
    config = load_config(config_path)
    output_path = config.resolve(config.correlation.output_path)

    daily_by_symbol = {}
    for symbol in config.symbols:
        daily = load_daily(config, symbol)
        if daily is None or daily.empty:
            print(f"{symbol} skipped reason=no_data_file")
            continue
        daily_by_symbol[symbol] = daily

    symbols = [s for s in config.symbols if s in daily_by_symbol]
    matrix = corr.daily_returns_matrix(daily_by_symbol)
    dates = list(pd.to_datetime(matrix.index, utc=True)) if not matrix.empty else []

    rows = corr.build_daily_matrices(
        matrix, symbols, dates,
        lookback_days=config.correlation.lookback_days,
        min_required_days=config.correlation.min_required_days,
        method=config.correlation.method,
        fallback_self_corr=config.correlation.fallback_self_corr,
        fallback_cross_corr=config.correlation.fallback_cross_corr,
    )
    corr.write_jsonl(rows, output_path)
    print(f"correlation matrices written to {output_path} ({len(rows)} days, {len(symbols)} symbols)")
    return output_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build daily cross-coin correlation matrices.")
    parser.add_argument("--config", default="config/distribution_config.yaml")
    args = parser.parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
