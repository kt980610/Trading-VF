"""CLI entrypoint to build the distribution snapshot."""

from __future__ import annotations

import argparse
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import load_config
    from src.data_loader import load_daily
    from src.distributions import build_histogram_pdf
    from src.rolling_features import (
        compute_daily_returns,
        compute_rolling_state,
        rolling_mean,
        rolling_mean_of_mean,
        rolling_var_of_mean,
        rolling_variance,
    )
    from src.snapshot_writer import build_snapshot, write_snapshot
else:
    from .config import load_config
    from .data_loader import load_daily
    from .distributions import build_histogram_pdf
    from .rolling_features import (
        compute_daily_returns,
        compute_rolling_state,
        rolling_mean,
        rolling_mean_of_mean,
        rolling_var_of_mean,
        rolling_variance,
    )
    from .snapshot_writer import build_snapshot, write_snapshot


def build_symbol_entry(config, returns) -> dict:
    min_obs = config.distribution.min_observations
    window = config.rolling.window_days
    mode = config.rolling.variance_mode
    bins = config.distribution.bins

    returns = returns.dropna()
    if len(returns) < min_obs:
        return {"valid": False, "reason": "not_enough_observations"}

    rmean = rolling_mean(returns, window)
    rvar = rolling_variance(returns, window, mode)
    rmean_of_mean = rolling_mean_of_mean(rmean, window)
    rvar_of_mean = rolling_var_of_mean(rmean, window, mode)

    series_map = {
        "return_distribution": returns,
        "mean_distribution": rmean.dropna(),
        "variance_distribution": rvar.dropna(),
        "mean_of_mean_distribution": rmean_of_mean.dropna(),
        "var_of_mean_distribution": rvar_of_mean.dropna(),
    }

    for series in series_map.values():
        if len(series) < min_obs:
            return {"valid": False, "reason": "not_enough_observations"}

    rolling_state = compute_rolling_state(returns, rmean, window, mode)
    if rolling_state is None:
        return {"valid": False, "reason": "not_enough_observations"}

    entry = {
        "valid": True,
        "n_daily_returns": int(len(returns)),
    }
    for key, series in series_map.items():
        entry[key] = build_histogram_pdf(series.to_numpy(), bins)
    entry["rolling_state"] = rolling_state
    return entry


def run(config_path: str) -> dict:
    config = load_config(config_path)
    symbols_out = {}

    for symbol in config.symbols:
        df = load_daily(config, symbol)
        if df is None:
            symbols_out[symbol] = {"valid": False, "reason": "no_data_file"}
            print(f"{symbol} skipped reason=no_data_file")
            continue

        returns = compute_daily_returns(df["close"])
        entry = build_symbol_entry(config, returns)
        symbols_out[symbol] = entry

        if entry.get("valid"):
            print(f"{symbol} valid n={entry['n_daily_returns']}")
        else:
            print(f"{symbol} invalid reason={entry.get('reason')}")

    snapshot = build_snapshot(
        config.distribution.method,
        config.distribution.bins,
        symbols_out,
    )
    output_path = config.resolve(config.data.output_path)
    write_snapshot(snapshot, output_path)
    print(f"snapshot written to {output_path}")
    return snapshot


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build a distribution snapshot.")
    parser.add_argument(
        "--config",
        default="config/distribution_config.yaml",
        help="Path to the YAML config file.",
    )
    args = parser.parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
