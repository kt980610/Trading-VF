"""CLI: build per-symbol RF datasets, train, evaluate and promote (sections 10-16)."""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import load_config
    from src.data_loader import load_minute
    from src.snapshot_writer import load_snapshot
    from src.integral_cache import build_cache_for_symbol
    from src import rf_dataset as rfd
    from src import rf_model as rfm
    from src import rf_promotion as rfp
    from src import simulator as sim
else:
    from .config import load_config
    from .data_loader import load_minute
    from .snapshot_writer import load_snapshot
    from .integral_cache import build_cache_for_symbol
    from . import rf_dataset as rfd
    from . import rf_model as rfm
    from . import rf_promotion as rfp
    from . import simulator as sim


def _build_context_provider(config, symbol):
    """date -> volume/news context features (intraday minute features default 0)."""
    ctx = {}

    vol_path = config.resolve(config.paths.predicted_daily_volume)
    if os.path.isfile(vol_path):
        with open(vol_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("symbol") != symbol:
                    continue
                prev = float(rec.get("previous_day_real_volume", 0.0) or 0.0)
                pred = float(rec.get("predicted_daily_volume", 0.0) or 0.0)
                change = (pred / prev - 1.0) if prev else 0.0
                ctx.setdefault(rec["date"], {}).update(
                    {
                        "predicted_daily_volume": pred,
                        "previous_day_real_volume": prev,
                        "predicted_volume_change_pct": change,
                    }
                )

    news_path = config.resolve(config.paths.news_features_daily)
    if os.path.isfile(news_path):
        with open(news_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("symbol") not in (symbol, None):
                    continue
                date = rec.get("date")
                feats = {
                    k: v
                    for k, v in rec.items()
                    if k.endswith("_news_sentiment")
                    or k.endswith("_news_count")
                    or k.endswith("_weighted")
                }
                ctx.setdefault(date, {}).update(feats)

    def provider(ts):
        return ctx.get(str(pd.Timestamp(ts).date()), {})

    return provider


def _evaluate_policy(minute_df, cache, policy, symbol, account, config):
    res = sim.simulate_symbol(
        minute_df, cache, policy, symbol, account,
        fee_rate=config.simulation.fee_rate,
        funding_rate=config.simulation.funding_rate,
        slippage=config.simulation.slippage,
        add_margin_step=config.mdp.add_margin_step,
        max_add_margin_per_decision=config.mdp.max_add_margin_per_decision,
        max_total_added_margin=config.mdp.max_total_added_margin,
    )
    if res.empty:
        return 0.0, 0
    return float(res["daily_pnl"].sum()), int(res["liquidation_count"].sum())


def run(config_path: str) -> dict:
    config = load_config(config_path)
    snapshot = load_snapshot(config.resolve(config.paths.distribution_snapshot))
    account = sim.AccountConfig()
    include_optional = config.rf_model.include_optional_component_features

    report_rows = []
    for symbol in config.symbols:
        entry = snapshot.get("symbols", {}).get(symbol)
        if not entry or not entry.get("valid"):
            print(f"{symbol} skipped reason=no_valid_distribution")
            continue
        minute_df = load_minute(config, symbol)
        if minute_df is None or minute_df.empty:
            print(f"{symbol} skipped reason=no_data_file")
            continue

        cache = build_cache_for_symbol(symbol, entry, window=config.rolling.window_days)
        provider = _build_context_provider(config, symbol)

        rows = sim.generate_training_rows(
            minute_df, cache, symbol, account,
            context_provider=provider, include_optional=include_optional,
        )
        df = rfd.to_frame(rows, include_optional=include_optional)
        df = rfd.split_by_symbol(df, symbol)
        rfd.write_dataset(df, config.resolve(config.paths.rf_dataset_dir), symbol)

        result = rfm.train_rf(
            df, symbol,
            min_training_rows=config.rf_model.min_training_rows,
            n_estimators=config.rf_model.n_estimators,
            max_depth=config.rf_model.max_depth,
            random_state=config.rf_model.random_state,
            decision_threshold=config.rf_model.decision_threshold,
            include_optional=include_optional,
            scaling_enabled=config.feature_scaling.enabled,
            scaler_type=config.feature_scaling.scaler,
            imputer=config.feature_scaling.imputer,
        )
        if not result.get("valid"):
            print(f"{symbol} rf invalid reason={result.get('reason')} -> baseline fallback")
            report_rows.append({"symbol": symbol, "model_promoted": False, "reason_if_not_promoted": result.get("reason")})
            continue

        staging_dir = os.path.join(config.resolve(config.paths.models_staging), symbol)
        rfm.save_artifacts(
            staging_dir, result["model_json"], result["features"], result["metadata"],
            scaler=result.get("scaler"),
        )

        candidate_policy = rfm.predictor_from_json(result["model_json"], scaler=result.get("scaler"))
        baseline_policy = rfm.BaselineModel(threshold=config.rf_model.decision_threshold)

        # Chronological validation/test split of the minute data.
        n = len(minute_df)
        val_df = minute_df.iloc[: int(n * 0.5)].reset_index(drop=True)
        test_df = minute_df.iloc[int(n * 0.5):].reset_index(drop=True)

        cand_val_pnl, cand_val_liq = _evaluate_policy(val_df, cache, candidate_policy, symbol, account, config)
        cand_test_pnl, cand_test_liq = _evaluate_policy(test_df, cache, candidate_policy, symbol, account, config)
        base_val_pnl, base_val_liq = _evaluate_policy(val_df, cache, baseline_policy, symbol, account, config)
        base_test_pnl, base_test_liq = _evaluate_policy(test_df, cache, baseline_policy, symbol, account, config)

        candidate = rfp.PolicyMetrics(cand_val_pnl, cand_test_pnl, cand_test_liq)
        baseline = rfp.PolicyMetrics(base_val_pnl, base_test_pnl, base_test_liq)

        outcome = rfp.evaluate_and_promote(
            symbol, candidate, baseline,
            staging_root=config.resolve(config.paths.models_staging),
            promoted_root=config.resolve(config.paths.models_promoted),
            archive_root=config.resolve(config.paths.models_archive),
        )
        report_rows.append(
            {
                "symbol": symbol,
                "rf_validation_pnl": cand_val_pnl,
                "baseline_validation_pnl": base_val_pnl,
                "rf_test_pnl": cand_test_pnl,
                "baseline_test_pnl": base_test_pnl,
                "rf_liquidation_count": cand_test_liq,
                "baseline_liquidation_count": base_test_liq,
                "model_promoted": outcome["model_promoted"],
                "reason_if_not_promoted": outcome["reason_if_not_promoted"],
            }
        )
        print(f"{symbol} rf trained promoted={outcome['model_promoted']}")

    report_path = config.resolve(config.paths.rf_policy_report)
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    pd.DataFrame(report_rows).to_csv(report_path, index=False)
    print(f"rf policy report written to {report_path}")
    return {"report": report_path, "rows": report_rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train per-symbol RF close/continue models.")
    parser.add_argument("--config", default="config/distribution_config.yaml")
    args = parser.parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
