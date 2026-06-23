"""CLI: build per-symbol RF datasets, train and evaluate CANDIDATE models only.

This tool NEVER writes to ``models/promoted``. Every run is scoped by a required
``--run-id`` and emits candidate artifacts under
``<models_staging>/rf/<run-id>/<SYMBOL>/`` plus a run-level report under
``reports/rf_runs/<run-id>/``. Promotion is a separate, explicit, auditable step
performed elsewhere; there is no promotion path here.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import load_config
    from src.artifact_join import build_previous_day_provider
    from src.data_loader import DataSourceError, load_minute_ohlcv
    from src.halving_season import parse_halving_dates
    from src.snapshot_writer import load_snapshot
    from src.integral_cache import build_cache_for_symbol
    from src import rf_classifier as rfc
    from src import rf_dataset as rfd
    from src import simulator as sim
    from src import news_features as nf
    from src import news_ingest as ni
    from src import correlation as corr_mod
else:
    from .config import load_config
    from .artifact_join import build_previous_day_provider
    from .data_loader import DataSourceError, load_minute_ohlcv
    from .halving_season import parse_halving_dates
    from .snapshot_writer import load_snapshot
    from .integral_cache import build_cache_for_symbol
    from . import rf_classifier as rfc
    from . import rf_dataset as rfd
    from . import simulator as sim
    from . import news_features as nf
    from . import news_ingest as ni
    from . import correlation as corr_mod


def _build_volume_provider(config, symbol):
    """ts -> volume context using only the PREVIOUS completed day (leakage-safe)."""
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
    return build_previous_day_provider(ctx)


def _build_daily_news_into(config, symbol, ctx):
    """Legacy daily news join: previous-completed-day, keyed by date."""
    news_path = config.resolve(config.paths.news_features_daily)
    if not os.path.isfile(news_path):
        return
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
                or k.startswith("gkg_")  # GKG daily market-wide features
            }
            ctx.setdefault(date, {}).update(feats)


def _build_beta_provider(config, symbol):
    """ts -> {'capm_beta': ...} using the PREVIOUS completed day (leakage-safe)."""
    ctx = {}
    if getattr(config, "beta", None) is not None and config.beta.enabled:
        path = config.resolve(config.beta.output_path)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("symbol") not in (symbol, None):
                        continue
                    date = rec.get("date")
                    if date is None or "capm_beta" not in rec:
                        continue
                    ctx.setdefault(date, {})["capm_beta"] = float(rec["capm_beta"])
    return build_previous_day_provider(ctx)


def _build_asof_news_provider(config, symbol):
    """ts -> as-of news via the SAME ``symbol + timestamp`` join used live.

    Returns ``None`` when intraday news is not active so the caller falls back to
    the legacy daily join.
    """
    news = config.news
    if not (news.enabled and news.intraday_news_enabled):
        return None

    records = ni.load_raw_records(config.resolve(news.raw_path))
    df = pd.DataFrame(records) if records else pd.DataFrame(columns=ni.CANONICAL_FIELDS)
    news_df = nf.prepare_news(df, backend=news.sentiment_backend)

    corr_provider = corr_mod.CorrelationProvider.load(
        config.resolve(config.correlation.output_path),
        fallback_self_corr=config.correlation.fallback_self_corr,
        fallback_cross_corr=config.correlation.fallback_cross_corr,
    )
    universe = list(config.symbols)
    # Same timestamp-quality + previous-completed-day fallback the live worker
    # applies, so every training minute row sees the live news semantics.
    daily_ctx = nf.load_daily_ctx(config.resolve(config.paths.news_features_daily), symbol)

    def provider(ts):
        feats, _provenance = nf.asof_news_for_decision(
            news_df,
            symbol,
            universe,
            ts,
            corr_provider=corr_provider,
            lookback_hours=news.feature_lookback_hours,
            safety_lag_seconds=news.news_safety_lag_seconds,
            daily_ctx=daily_ctx,
            live=False,
            historical_lag_seconds=news.historical_news_availability_lag_seconds,
        )
        return feats

    return provider


def _build_context_provider(config, symbol):
    """ts -> per-minute volume + news context.

    Volume always uses the leakage-safe previous-completed-day join. News uses
    the as-of ``symbol + timestamp`` join when ``news.intraday_news_enabled`` is
    set (identical semantics to the live engine), otherwise the legacy
    previous-completed-day daily join. Live and training never diverge.
    """
    vol_provider = _build_volume_provider(config, symbol)
    asof_news = _build_asof_news_provider(config, symbol)
    beta_provider = _build_beta_provider(config, symbol)

    if asof_news is not None:
        def provider(ts):
            merged = dict(vol_provider(ts))
            merged.update(asof_news(ts))
            merged.update(beta_provider(ts))
            return merged

        return provider

    # Legacy daily path: merge daily news into a date-keyed ctx alongside volume.
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
    # No-news baseline: when news is disabled, never inject news columns (not even
    # a stale daily artifact). The dataset frame then carries no `_news_` columns,
    # so the RF training schema excludes them entirely instead of training on
    # zero/constant news features.
    if config.news.enabled:
        _build_daily_news_into(config, symbol, ctx)
    daily_provider = build_previous_day_provider(ctx)

    def provider(ts):
        merged = dict(daily_provider(ts))
        merged.update(beta_provider(ts))
        return merged

    return provider


RUN_MODE = "candidate_only"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def validate_run_id(run_id: str) -> str:
    """Reject empty, unsafe or path-traversing run ids (e.g. ``no_news_4y_v1``)."""
    if not run_id or not isinstance(run_id, str):
        raise ValueError("--run-id must be a non-empty string")
    if run_id in (".", ".."):
        raise ValueError("--run-id must not be '.' or '..'")
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(
            "--run-id may only contain letters, digits, '.', '_' and '-' "
            f"(got {run_id!r}); path separators and traversal are not allowed"
        )
    return run_id


def _git_commit_hash():
    """Best-effort short git commit hash; ``None`` if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return (out.stdout or "").strip() or None
    except Exception:
        return None
    return None


def _assert_not_under_promoted(path: str, promoted_root: str) -> None:
    """Fail-closed if a candidate output path would land under models/promoted."""
    p = os.path.abspath(path)
    root = os.path.abspath(promoted_root)
    try:
        common = os.path.commonpath([p, root])
    except ValueError:
        return  # different drive / not comparable -> not under promoted
    if common == root:
        raise ValueError(
            f"refusing to target the promoted model directory: {p} is under {root}. "
            "main_train_rf only writes candidate artifacts to models_staging."
        )


def _summary_row(symbol, status, meta=None, n_minute_rows=None):
    """One per-symbol report row (consistent columns for the CSV/JSON)."""
    meta = meta or {}
    selected = meta.get("selected_rf_hyperparameters", {}) or {}
    return {
        "symbol": symbol,
        "status": status,
        "n_minute_rows": n_minute_rows if n_minute_rows is not None else meta.get("n_minute_rows"),
        "selected_n_estimators": selected.get("n_estimators"),
        "selected_max_depth": selected.get("max_depth"),
        "selected_min_samples_leaf": selected.get("min_samples_leaf"),
        "selected_max_features": selected.get("max_features"),
        "close_probability_threshold": meta.get("close_probability_threshold"),
        "validation_total_pnl_per_minute_objective": meta.get(
            "validation_total_pnl_per_minute_objective"
        ),
        "validation_trade_count": meta.get("validation_trade_count"),
        "baseline_validation_objective": meta.get("baseline_validation_objective"),
        "train_net_pnl": meta.get("train_net_pnl"),
        "train_baseline_net_pnl": meta.get("train_baseline_net_pnl"),
        "train_total_pnl_per_minute_objective": meta.get("train_total_pnl_per_minute_objective"),
        "train_max_drawdown": meta.get("train_max_drawdown"),
        "train_liquidation_count": meta.get("train_liquidation_count"),
        "test_total_pnl_per_minute_objective": meta.get("test_total_pnl_per_minute_objective"),
        "test_net_pnl": meta.get("test_net_pnl"),
        "test_max_drawdown": meta.get("test_max_drawdown"),
        "test_turnover": meta.get("test_turnover"),
        "test_liquidation_count": meta.get("test_liquidation_count"),
    }


def _downcast_floats(df):
    """Downcast float64 columns to float32 to roughly halve memory.

    sklearn tree models cast inputs to float32 internally anyway, so this is
    lossless for training while making the in-memory dataset far smaller.
    """
    if df is None or df.empty:
        return df
    f64 = [c for c in df.columns if str(df[c].dtype) == "float64"]
    if f64:
        df[f64] = df[f64].astype("float32")
    return df


def _generate_windowed_dataset(minute_df, windows, cache, symbol, account, gen_kwargs,
                               max_hold_minutes=None, entry_every_days=7,
                               spill_dir=None, flush_rows=250_000):
    """Generate the trade dataset for each disjoint training window separately.

    Each window is simulated on its own minute slice (so no trade ever spans a
    multi-year gap), tagged with ``train_window`` and given globally-unique
    ``trade_id``/``leg_id`` values, then concatenated. ``halving_cycle_id`` (the
    "k" class) is computed per timestamp inside ``generate_trade_dataset`` and is
    therefore 2/3/4 for the 2018/2022/2026 windows automatically.
    """
    mdf = minute_df.reset_index(drop=True)
    ts = mdf["timestamp"]
    parts = []
    for w_idx, (start_raw, end_raw) in enumerate(windows):
        start = pd.Timestamp(start_raw, tz="UTC")
        end = pd.Timestamp(end_raw, tz="UTC")
        in_win = (ts >= start) & (ts < end)
        label = f"{start.date()}..{end.date()}"
        if not bool(in_win.any()):
            print(f"{symbol} window[{w_idx}] {label} rows=0 (no minute data, skipped)")
            continue
        # One independent (overlapping) episode per UTC day in the window; each
        # runs forward to full close/liquidation within its contiguous segment.
        win_ts = ts[in_win]
        daily_idx = win_ts.groupby(win_ts.dt.floor("1D")).head(1).index.to_numpy()
        stride = max(1, int(entry_every_days))
        entry_idx = daily_idx[::stride]
        # Forward sim never crosses the regime (window) end, so a k=3 trade can't
        # leak into k=4 and compute stays bounded.
        window_end_idx = int((ts < end).sum())
        spill_path = (
            os.path.join(spill_dir, f"{symbol}_w{w_idx}") if spill_dir else None
        )
        d = sim.generate_overlapping_trade_dataset(
            mdf, cache, symbol, account, entry_idx,
            max_end_index=window_end_idx, max_hold_minutes=max_hold_minutes,
            spill_path=spill_path, flush_rows=flush_rows,
            **gen_kwargs
        )
        if d is None or d.empty:
            print(f"{symbol} window[{w_idx}] {label} entries={len(entry_idx)} trades=0 (skipped)")
            continue
        d = d.copy()
        d["trade_id"] = d["trade_id"].astype("int64") + w_idx * 100_000_000
        if "side_label" in d.columns:
            d["leg_id"] = d["trade_id"].astype(str) + ":" + d["side_label"].astype(str)
        d["train_window"] = int(w_idx)
        kcol = d["halving_cycle_id"] if "halving_cycle_id" in d.columns else None
        kdesc = f" k={sorted(set(int(v) for v in kcol))}" if kcol is not None else ""
        print(
            f"{symbol} window[{w_idx}] {label} entries={len(entry_idx)} "
            f"trades={d['trade_id'].nunique()}{kdesc}"
        )
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def run(config_path: str, run_id: str) -> dict:
    """Train per-symbol RF CANDIDATES for ``run_id``; never writes to promoted."""
    run_id = validate_run_id(run_id)
    config = load_config(config_path)
    snapshot = load_snapshot(config.resolve(config.paths.distribution_snapshot))
    account = sim.AccountConfig()
    include_optional = config.rf_model.include_optional_component_features
    halvings = parse_halving_dates(config.halving.dates or None)
    season_seed = int(config.halving.season_seed)
    close_epsilon = float(getattr(config.simulation, "close_epsilon", 0.0))

    # Candidate-only output layout (NO promoted path anywhere).
    promoted_root = config.resolve(config.paths.models_promoted)
    staging_root = os.path.join(config.resolve(config.paths.models_staging), "rf", run_id)
    report_dir = config.resolve(os.path.join("reports", "rf_runs", run_id))
    dataset_root = os.path.join(config.resolve(config.paths.rf_dataset_dir), run_id)

    # Never let a candidate output resolve under models/promoted.
    for candidate_path in (staging_root, dataset_root, report_dir):
        _assert_not_under_promoted(candidate_path, promoted_root)

    # Fail-closed: do not overwrite an existing run's artifacts or reports.
    for existing in (staging_root, report_dir, dataset_root):
        if os.path.exists(existing):
            raise FileExistsError(
                f"run-id '{run_id}' already has output at {existing}; "
                "choose a new --run-id (existing candidate artifacts/reports are kept)"
            )

    os.makedirs(staging_root, exist_ok=False)
    os.makedirs(report_dir, exist_ok=False)

    report_rows = []
    for symbol in config.symbols:
        entry = snapshot.get("symbols", {}).get(symbol)
        if not entry or not entry.get("valid"):
            print(f"{symbol} skipped reason=no_valid_distribution")
            report_rows.append(_summary_row(symbol, "no_valid_distribution"))
            continue
        try:
            minute_df = load_minute_ohlcv(config, symbol)
        except DataSourceError as exc:
            print(f"{symbol} skipped reason={exc.reason}")
            report_rows.append(_summary_row(symbol, f"data_source_error:{exc.reason}"))
            continue
        if minute_df is None or minute_df.empty:
            print(f"{symbol} skipped reason=missing_minute_source")
            report_rows.append(_summary_row(symbol, "missing_minute_source"))
            continue

        cache = build_cache_for_symbol(symbol, entry, window=config.rolling.window_days)
        provider = _build_context_provider(config, symbol)

        gen_kwargs = dict(
            halvings=halvings, season_seed=season_seed,
            context_provider=provider,
            fee_rate=config.simulation.fee_rate,
            funding_rate=config.simulation.funding_rate,
            slippage=config.simulation.slippage,
            close_epsilon=close_epsilon,
            include_optional=include_optional,
        )
        training_windows = getattr(getattr(config, "training", None), "windows", None)
        if training_windows:
            tcfg = getattr(config, "training", None)
            max_hold_days = int(getattr(tcfg, "max_hold_days", 0) or 0)
            max_hold_minutes = max_hold_days * 1440 if max_hold_days > 0 else None
            entry_every_days = int(getattr(tcfg, "entry_every_days", 7) or 1)
            spill_enabled = bool(getattr(tcfg, "spill_to_disk", True))
            flush_rows = int(getattr(tcfg, "spill_flush_rows", 250_000) or 250_000)
            spill_dir = os.path.join(dataset_root, "_spill") if spill_enabled else None
            df = _generate_windowed_dataset(
                minute_df, training_windows, cache, symbol, account, gen_kwargs,
                max_hold_minutes=max_hold_minutes, entry_every_days=entry_every_days,
                spill_dir=spill_dir, flush_rows=flush_rows,
            )
        else:
            df = sim.generate_trade_dataset(minute_df, cache, symbol, account, **gen_kwargs)
        df = _downcast_floats(df)
        if df is None or df.empty:
            reason = "no_rows_in_training_windows" if training_windows else "no_trades"
            print(f"{symbol} skipped reason={reason}")
            report_rows.append(_summary_row(symbol, reason))
            continue
        df = rfd.split_by_symbol(df, symbol)
        # Datasets are also scoped to the run so they never clobber other runs.
        # The FULL dataset is always written to disk.
        rfd.write_dataset(df, os.path.join(dataset_root, symbol), symbol)
        n_full_rows = int(len(df))

        # sklearn RandomForest.fit() needs the whole matrix in RAM; for daily,
        # uncapped, multi-window episodes this is millions of highly-redundant
        # per-minute rows and can OOM. Train on a representative subsample (full
        # data stays on disk). Sampling keeps every trade group present.
        train_max_rows = int(getattr(getattr(config, "training", None), "train_max_rows", 0) or 0)
        if train_max_rows > 0 and len(df) > train_max_rows:
            df_fit = df.sample(n=train_max_rows, random_state=config.rf_model.random_state).sort_index()
            print(f"{symbol} fit subsample {len(df_fit)}/{n_full_rows} rows "
                  f"(trade_groups={df_fit['trade_id'].nunique() if 'trade_id' in df_fit.columns else 'n/a'})")
        else:
            df_fit = df
        del df

        _features, scale_cols, passthrough_cols = rfd.close_classifier_column_split(df_fit, include_optional)
        result = rfc.train_close_classifier(
            df_fit, symbol, scale_cols, passthrough_cols,
            min_training_rows=config.rf_model.min_training_rows,
            n_estimators=config.rf_model.n_estimators,
            max_depth=config.rf_model.max_depth,
            n_jobs=getattr(config.rf_model, "n_jobs", 4),
            random_state=config.rf_model.random_state,
            close_epsilon=close_epsilon,
            season_seed=season_seed,
        )
        df = df_fit
        if not result.get("valid"):
            status = result.get("status")
            print(f"{symbol} rf classifier skipped status={status}")
            report_rows.append(_summary_row(symbol, status, n_minute_rows=int(len(df))))
            continue

        # Candidate artifacts -> staging only. save_artifacts is itself
        # fail-closed (invalid results never write). Double-check the target is
        # not under promoted before writing.
        model_dir = os.path.join(staging_root, symbol)
        _assert_not_under_promoted(model_dir, promoted_root)
        rfc.save_artifacts(model_dir, result)
        meta = result["metadata"]
        selected = meta.get("selected_rf_hyperparameters", {})
        report_rows.append(_summary_row(symbol, result["status"], meta=meta))
        print(
            f"{symbol} rf candidate trained "
            f"threshold={meta['close_probability_threshold']:.3f} "
            f"rf={selected.get('n_estimators')}/{selected.get('max_depth')}/"
            f"{selected.get('min_samples_leaf')}/{selected.get('max_features')} "
            f"val_obj={meta.get('validation_total_pnl_per_minute_objective'):.4f} "
            f"train_pnl={meta.get('train_net_pnl')} "
            f"test_obj={meta.get('test_total_pnl_per_minute_objective')} "
            f"-> {model_dir}"
        )

    summary = {
        "run_id": run_id,
        "mode": RUN_MODE,
        "config_path": os.path.abspath(config_path),
        "symbols": list(config.symbols),
        "news_enabled": bool(config.news.enabled),
        "news_intraday_enabled": bool(config.news.intraday_news_enabled),
        "git_commit": _git_commit_hash(),
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "staging_dir": os.path.abspath(staging_root),
        "dataset_dir": os.path.abspath(dataset_root),
        "promoted_dir_untouched": os.path.abspath(promoted_root),
        "symbols_report": report_rows,
    }
    summary_json = os.path.join(report_dir, "summary.json")
    summary_csv = os.path.join(report_dir, "summary.csv")
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    pd.DataFrame(report_rows).to_csv(summary_csv, index=False)
    print(f"candidate run '{run_id}' report written to {report_dir} (mode={RUN_MODE})")
    return {
        "run_id": run_id,
        "mode": RUN_MODE,
        "staging_dir": staging_root,
        "dataset_dir": dataset_root,
        "summary_json": summary_json,
        "summary_csv": summary_csv,
        "rows": report_rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Train per-symbol RF CANDIDATE models (never promotes; staging only)."
    )
    parser.add_argument("--config", default="config/distribution_config.yaml")
    parser.add_argument(
        "--run-id",
        required=True,
        help="Required candidate run id, e.g. no_news_4y_v1 (no path separators).",
    )
    args = parser.parse_args(argv)
    run(args.config, args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
