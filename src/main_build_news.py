"""CLI: build leakage-safe daily news features for every symbol (section 13)."""

from __future__ import annotations

import argparse
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import load_config
    from src.data_loader import load_daily
    from src import news_features as nf
    from src import correlation as corr_mod
else:
    from .config import load_config
    from .data_loader import load_daily
    from . import news_features as nf
    from . import correlation as corr_mod


def run(config_path: str) -> str:
    config = load_config(config_path)
    raw_path = config.resolve(config.news.raw_path)
    output_path = config.resolve(config.news.output_path)

    raw = nf.load_news(raw_path)
    news = nf.prepare_news(raw, backend=config.news.sentiment_backend)

    # Correlation provider (built earlier); falls back to self=1/cross=0 if absent.
    corr_provider = corr_mod.CorrelationProvider.load(
        config.resolve(config.correlation.output_path),
        fallback_self_corr=config.correlation.fallback_self_corr,
        fallback_cross_corr=config.correlation.fallback_cross_corr,
    )
    universe = list(config.symbols)

    all_rows = []
    for symbol in config.symbols:
        daily = load_daily(config, symbol)
        if daily is None or daily.empty:
            print(f"{symbol} skipped reason=no_data_file")
            continue
        day_starts = daily["timestamp"].dt.floor("1D").tolist()
        rows = nf.daily_cross_features_for_symbol(
            news, symbol, universe, day_starts, corr_provider=corr_provider
        )
        all_rows.extend(rows)
        print(f"{symbol} news_days={len(rows)} raw_news={len(news)}")

    nf.write_jsonl(all_rows, output_path)
    print(f"news features written to {output_path}")
    return output_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build daily news features.")
    parser.add_argument("--config", default="config/distribution_config.yaml")
    args = parser.parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
