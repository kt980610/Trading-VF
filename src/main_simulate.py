"""CLI: per-symbol minute-level PnL simulation, RF vs baseline (section 21)."""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import load_config
    from src.data_loader import load_minute
    from src.snapshot_writer import load_snapshot
    from src.integral_cache import build_cache_for_symbol
    from src import rf_model as rfm
    from src import simulator as sim
    from src.parallel import parallel_map
else:
    from .config import load_config
    from .data_loader import load_minute
    from .snapshot_writer import load_snapshot
    from .integral_cache import build_cache_for_symbol
    from . import rf_model as rfm
    from . import simulator as sim
    from .parallel import parallel_map


def _simulate_one(args) -> pd.DataFrame:
    config, symbol, entry = args
    minute_df = load_minute(config, symbol)
    if minute_df is None or minute_df.empty:
        return pd.DataFrame()

    cache = build_cache_for_symbol(symbol, entry, window=config.rolling.window_days)
    account = sim.AccountConfig()

    promoted_dir = os.path.join(config.resolve(config.paths.models_promoted), symbol)
    model_promoted = os.path.isfile(os.path.join(promoted_dir, "rf_close_decision.json"))
    rf_policy = rfm.get_policy(promoted_dir, fallback_threshold=config.simulation.decision_threshold)
    baseline_policy = rfm.BaselineModel(threshold=config.simulation.decision_threshold)

    def run(policy):
        return sim.simulate_symbol(
            minute_df, cache, policy, symbol, account,
            fee_rate=config.simulation.fee_rate,
            funding_rate=config.simulation.funding_rate,
            slippage=config.simulation.slippage,
            add_margin_step=config.mdp.add_margin_step,
            max_add_margin_per_decision=config.mdp.max_add_margin_per_decision,
            max_total_added_margin=config.mdp.max_total_added_margin,
        )

    rf_res = run(rf_policy)
    base_res = run(baseline_policy)
    if rf_res.empty:
        return pd.DataFrame()

    merged = rf_res.merge(base_res, on=["date", "symbol"], how="outer", suffixes=("_rf", "_base")).fillna(0.0)
    out = pd.DataFrame()
    out["date"] = merged["date"]
    out["symbol"] = merged["symbol"]
    out["rf_daily_pnl"] = merged["daily_pnl_rf"]
    out["baseline_daily_pnl"] = merged["daily_pnl_base"]
    out["rf_minus_baseline"] = merged["daily_pnl_rf"] - merged["daily_pnl_base"]
    out["rf_close_count"] = merged["close_count_rf"]
    out["rf_continue_count"] = merged["continue_count_rf"]
    out["rf_liquidation_count"] = merged["liquidation_count_rf"]
    out["baseline_liquidation_count"] = merged["liquidation_count_base"]
    out["double_liquidation_count"] = merged["double_liquidation_count_rf"]
    out["avg_hold_minutes"] = merged["hold_minutes_rf"] / merged["close_count_rf"].replace(0, 1)
    out["model_promoted"] = model_promoted
    out["reason_if_not_promoted"] = "" if model_promoted else "no_promoted_model"
    return out


def run(config_path: str) -> str:
    config = load_config(config_path)
    snapshot = load_snapshot(config.resolve(config.paths.distribution_snapshot))

    jobs = []
    for symbol in config.symbols:
        entry = snapshot.get("symbols", {}).get(symbol)
        if entry and entry.get("valid"):
            jobs.append((config, symbol, entry))
        else:
            print(f"{symbol} skipped reason=no_valid_distribution")

    workers = config.simulation.max_workers if config.simulation.parallel else 1
    frames = parallel_map(_simulate_one, jobs, max_workers=workers)
    frames = [f for f in frames if not f.empty]

    for f in frames:
        sym = f["symbol"].iloc[0]
        print(f"{sym} rf_pnl={f['rf_daily_pnl'].sum():.2f} base_pnl={f['baseline_daily_pnl'].sum():.2f}")

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_path = config.resolve(config.paths.simulation_results)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"simulation results written to {out_path}")
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run minute-level strategy simulation.")
    parser.add_argument("--config", default="config/distribution_config.yaml")
    args = parser.parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
