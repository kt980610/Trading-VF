"""CLI: per-symbol walk-forward daily volume prediction (section 14)."""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import load_config
    from src.data_loader import DataSourceError, load_daily_ohlcv
    from src import volume_model as vm
else:
    from .config import load_config
    from .data_loader import DataSourceError, load_daily_ohlcv
    from . import volume_model as vm


def _load_news_daily(path: str):
    if not os.path.isfile(path):
        return None
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records) if records else None


def run(config_path: str) -> str:
    config = load_config(config_path)
    out_path = config.resolve(config.paths.predicted_daily_volume)
    report_path = config.resolve(config.paths.volume_report)
    models_dir = config.resolve(config.paths.volume_models)
    news_daily = _load_news_daily(config.resolve(config.paths.news_features_daily))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    report_rows = []
    with open(out_path, "w", encoding="utf-8") as out_fh:
        for symbol in config.symbols:
            try:
                daily = load_daily_ohlcv(config, symbol)
            except DataSourceError as exc:
                print(f"{symbol} skipped reason={exc.reason}")
                continue
            if daily is None or daily.empty:
                print(f"{symbol} skipped reason=missing_daily_source")
                continue

            symbol_news = None
            if news_daily is not None and "symbol" in news_daily.columns:
                symbol_news = news_daily[news_daily["symbol"] == symbol]

            frame = vm.build_feature_frame(
                daily,
                window=config.rolling.window_days,
                variance_mode=config.rolling.variance_mode,
                news_daily=symbol_news,
            )
            preds = vm.walk_forward_predict(
                frame,
                symbol,
                kernel=config.volume.kernel,
                alpha=config.volume.alpha,
                gamma=config.volume.gamma,
                min_train_days=config.volume.min_train_days,
            )
            for _, row in preds.iterrows():
                out_fh.write(json.dumps(row.to_dict()) + "\n")

            report_rows.append(vm.evaluate_predictions(preds, symbol))
            vm.train_and_save_final(
                frame, symbol, models_dir,
                kernel=config.volume.kernel, alpha=config.volume.alpha, gamma=config.volume.gamma,
            )
            print(f"{symbol} volume_days={len(preds)}")

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    pd.DataFrame(report_rows).to_csv(report_path, index=False)
    print(f"predicted volume written to {out_path}")
    print(f"volume report written to {report_path}")
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build predicted daily volume.")
    parser.add_argument("--config", default="config/distribution_config.yaml")
    args = parser.parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
