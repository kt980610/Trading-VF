"""Configuration loading for the distribution snapshot builder.

The config is intentionally simple YAML. We use PyYAML when available, but fall
back to a tiny built-in parser so the tool runs even on a bare Python install.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    output_path: str = "data/distribution_snapshot.json"
    # Two explicit raw sources; suffixes drive file selection (NOT timeframe).
    daily_file_suffix: str = "_daily.csv"
    minute_file_suffix: str = "_1m.csv"
    require_daily_source: bool = True
    require_minute_source: bool = True
    # Deprecated: retained for backward-compatible config parsing only. These no
    # longer influence which source file is loaded.
    timeframe: str = "1m"
    daily_resample: bool = True


@dataclass
class RollingConfig:
    window_days: int = 30
    variance_mode: str = "population"  # "population" or "sample"


@dataclass
class DistributionConfig:
    method: str = "histogram"
    bins: int = 400
    min_observations: int = 60


@dataclass
class IntegrationConfig:
    """Bounds/resolution used when building integral caches."""

    grid_points: int = 0  # 0 -> reuse the snapshot return-distribution grid
    clip_to_grid: bool = True


@dataclass
class RFModelConfig:
    scope: str = "per_symbol"
    fallback_to_baseline_if_symbol_model_missing: bool = True
    min_training_rows: int = 250
    n_estimators: int = 200
    max_depth: int = 0  # 0 -> None
    random_state: int = 42
    # Cap parallel tree-building workers. -1 = all cores (high peak RAM); a small
    # value (e.g. 4) bounds memory because each worker holds its own tree state.
    n_jobs: int = 4
    decision_threshold: float = 0.0
    train_frequency: str = "monthly"
    history_years: int = 4
    include_optional_component_features: bool = False


@dataclass
class VolumeConfig:
    enabled: bool = True
    kernel: str = "rbf"
    alpha: float = 1.0
    gamma: float = 0.0  # 0 -> sklearn default (None)
    min_train_days: int = 120
    walk_forward: bool = True


@dataclass
class NewsConfig:
    enabled: bool = True
    # When true, missing/failed coverage or all-zero raw news is a hard error
    # during feature building (no silent zero-feature training).
    strict_coverage: bool = True
    provider: str = "newsapi"
    # API key is read ONLY from this environment variable. The key itself is
    # NEVER stored in config, logs, manifests or artifacts.
    api_key_env: str = "NEWSAPI_API_KEY"
    # GDELT (free, keyless) adapter settings. Empty base_url -> provider default.
    gdelt_base_url: str = ""
    gdelt_max_records: int = 250
    # Minimum seconds between ANY two GDELT requests (one global limiter for the
    # whole run; no per-query bursts). GDELT throttles aggressively.
    gdelt_min_request_interval_seconds: float = 5.0
    # OUR safety policy (NOT an official GDELT limit): DOC 2.0 is a recent/live
    # source, so we only trust it for a trailing window and fail fast
    # (unsupported_history_window) for older dates rather than silently running.
    # Tune this to taste; it is a self-imposed guardrail, not a documented cutoff.
    gdelt_doc_max_lookback_days: int = 90
    raw_path: str = "data/raw/news.jsonl"
    coverage_manifest_path: str = "data/news_coverage_manifest.jsonl"
    output_path: str = "data/news_features_daily.jsonl"
    report_path: str = "reports/news_ingestion_report.json"
    sentiment_cache_path: str = "data/news_sentiment_cache.jsonl"
    backfill_start: str = "2022-01-01"
    language: str = "en"
    categories: List[str] = field(
        default_factory=lambda: [
            "macro",
            "policy",
            "stock_market",
            "crypto_market",
            "symbol_specific",
        ]
    )
    sentiment_backend: str = "finbert"  # lexicon | table | finbert | vader
    sentiment_table_path: str = ""
    # Optional query overrides. Empty -> module defaults in src.news_ingest.
    # ``queries``: {category: [query strings]}; ``symbol_queries``: {symbol: query}.
    queries: Dict[str, Any] = field(default_factory=dict)
    symbol_queries: Dict[str, Any] = field(default_factory=dict)
    # --- Live / intraday as-of feature settings ----------------------------
    # When true, BOTH live inference and training join news by the as-of
    # ``symbol + timestamp`` rule (rolling window, safety lag) instead of the
    # legacy previous-completed-day daily join. Default false (safety).
    intraday_news_enabled: bool = False
    # Rolling window (hours) looked back from each decision instant.
    feature_lookback_hours: int = 24
    # Leakage guard: only news ``available_at <= decision - lag`` is used, where
    # ``available_at`` is the ingestion/scoring completion time (live) or, for
    # historical training without that record, the conservative proxy below.
    news_safety_lag_seconds: int = 300
    # Historical proxy: when no real availability time exists, a news item is only
    # considered available at ``published_at + this`` (models fetch+score delay).
    historical_news_availability_lag_seconds: int = 300
    # Rust refuses to OPEN new positions when the freshest as-of record is older
    # than this many minutes (existing positions still managed normally).
    max_news_feature_age_minutes: int = 30
    # Separate intraday artifact, written atomically by the live news worker.
    intraday_output_path: str = "data/news_features_intraday.jsonl"
    # Version tag stamped onto every intraday record (schema/version guard).
    feature_version: str = "news_v1"
    # GKG historical/live source: the SQLite slot-feature backfill produced by
    # scripts/gkg_historical_feature_backfill.py. src.gkg_news_daily rolls it up
    # into the daily news artifact (market-wide, D-1 leakage-safe).
    gkg_features_db: str = "data/gkg_4y_2h.sqlite"


@dataclass
class CorrelationConfig:
    enabled: bool = True
    lookback_days: int = 90
    min_required_days: int = 60
    method: str = "pearson"
    fallback_self_corr: float = 1.0
    fallback_cross_corr: float = 0.0
    output_path: str = "data/correlation_matrix_daily.jsonl"


@dataclass
class FeatureScalingConfig:
    enabled: bool = True
    scope: str = "per_symbol"
    scaler: str = "robust"  # robust | standard
    imputer: str = "median"
    fit_on_train_only: bool = True


@dataclass
class HalvingConfig:
    enabled: bool = True
    season_seed: int = 0
    dates: List[str] = field(default_factory=list)  # empty -> module defaults


@dataclass
class BetaConfig:
    """CAPM rolling-beta daily feature settings.

    The market leg ("crypto market value") is a daily total-market-cap proxy in
    ``market_cap_path`` (built by ``scripts/fetch_market_cap.py``); the asset leg
    is each symbol's daily close. ``beta = Cov(R_coin_L, R_mkt_L)/Var(R_mkt_L)``
    over a trailing ``window_days`` of ``return_lag_days``-day returns, joined D-1.
    """

    enabled: bool = True
    market_cap_path: str = "data/market_cap_daily.csv"
    output_path: str = "data/capm_beta_daily.jsonl"
    return_lag_days: int = 30
    window_days: int = 90
    market_top_n: int = 30


@dataclass
class TrainingConfig:
    """Disjoint UTC date windows the RF is trained on.

    ``windows`` is a list of ``(start, end)`` pairs (end EXCLUSIVE). When empty
    the full minute history is used (legacy behaviour). Each window is generated
    and split independently so every regime appears in train/validation/test; the
    halving cycle index (``halving_cycle_id``, i.e. the user's "k") is derived per
    timestamp and needs no separate variable.
    """

    windows: List[tuple] = field(default_factory=list)
    # Open one overlapping episode every ``entry_every_days`` days inside each
    # window (1 = daily, 7 = weekly). Larger strides mean fewer simultaneous
    # episodes -> far less memory, while episodes still run to true
    # liquidation/full-close (no holding cap needed).
    entry_every_days: int = 7
    # Optional hard cap on how long any single episode may stay open (0/None =
    # disabled). Kept as a safety knob; with weekly entries it is off by default
    # so the "hold to liquidation/close" principle is preserved.
    max_hold_days: int = 0
    # Spill generated per-minute rows to compact float32 Parquet parts during
    # generation instead of holding millions of fat dicts in RAM. This is the
    # main OOM fix: peak generation memory becomes ~spill_flush_rows, and the
    # full matrix is reassembled (compactly) only when needed for fit.
    spill_to_disk: bool = True
    spill_flush_rows: int = 250_000
    # Cap the number of per-minute rows used to FIT the RF (0 = use all). The
    # full dataset is still generated and written to disk; only training samples
    # a representative subset so sklearn's in-RAM fit does not OOM. Rows are
    # sampled while keeping every trade group present.
    train_max_rows: int = 3_000_000


@dataclass
class MDPConfig:
    add_margin_step: float = 10.0
    max_add_margin_per_decision: float = 1e18
    max_total_added_margin: float = 1e18


@dataclass
class SimulationConfig:
    fee_rate: float = 0.0004
    funding_rate: float = 0.0
    slippage: float = 0.0
    decision_threshold: float = 0.0
    close_epsilon: float = 0.0
    use_rf: bool = True
    parallel: bool = True
    max_workers: int = 0  # 0 -> auto


@dataclass
class PathsConfig:
    distribution_snapshot: str = "data/distribution_snapshot.json"
    integral_cache: str = "data/integral_cache.json"
    predicted_daily_volume: str = "data/predicted_daily_volume.jsonl"
    news_features_daily: str = "data/news_features_daily.jsonl"
    correlation_matrix_daily: str = "data/correlation_matrix_daily.jsonl"
    rf_dataset_dir: str = "data"
    simulation_results: str = "data/simulation_results.csv"
    volume_report: str = "reports/volume_model_report.csv"
    rf_policy_report: str = "reports/rf_policy_report.csv"
    speed_benchmark: str = "reports/speed_benchmark.csv"
    models_promoted: str = "models/promoted"
    models_staging: str = "models/staging"
    models_archive: str = "models/archive"
    volume_models: str = "models/volume"


@dataclass
class Config:
    symbols: List[str] = field(default_factory=list)
    data: DataConfig = field(default_factory=DataConfig)
    rolling: RollingConfig = field(default_factory=RollingConfig)
    distribution: DistributionConfig = field(default_factory=DistributionConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    rf_model: RFModelConfig = field(default_factory=RFModelConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)
    feature_scaling: FeatureScalingConfig = field(default_factory=FeatureScalingConfig)
    halving: HalvingConfig = field(default_factory=HalvingConfig)
    beta: BetaConfig = field(default_factory=BetaConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mdp: MDPConfig = field(default_factory=MDPConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    # Directory of the config file, used to resolve relative data paths.
    base_dir: str = "."

    def resolve(self, path: str) -> str:
        """Resolve a path relative to the current working directory.

        Data paths in the config (e.g. ``data/raw``) are interpreted relative to
        where the command is run from (the repo root), matching the spec layout.
        """
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(os.getcwd(), path))


def _coerce_scalar(value: str) -> Any:
    """Convert a raw YAML scalar string into a Python value."""
    text = value.strip()
    if text == "" or text in {"~", "null", "None"}:
        return None
    low = text.lower()
    if low in {"true", "yes"}:
        return True
    if low in {"false", "no"}:
        return False
    # Strip surrounding quotes.
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _strip_comment(line: str) -> str:
    """Remove an unquoted trailing ``#`` comment from a line."""
    in_single = in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i]
    return line


def _fallback_parse(text: str) -> Dict[str, Any]:
    """Minimal YAML parser for the simple structures used by this project.

    Supports nested maps (2-space indentation), block lists (``- item``),
    scalars, booleans and comments. It is not a general YAML implementation.
    """
    # Pre-clean lines into (indent, stripped_text) tuples, skipping blanks.
    lines: List[tuple[int, str]] = []
    for raw_line in text.splitlines():
        line = _strip_comment(raw_line)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        lines.append((indent, line.strip()))

    root: Dict[str, Any] = {}
    # Stack of (indent, container) pairs.
    stack: List[tuple[int, Any]] = [(-1, root)]

    i = 0
    n = len(lines)
    while i < n:
        indent, stripped = lines[i]

        # Pop containers that are no longer ancestors of this indent level.
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()
        container = stack[-1][1]

        if stripped.startswith("- "):
            item = _coerce_scalar(stripped[2:])
            if isinstance(container, list):
                container.append(item)
            i += 1
            continue

        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()

        if value == "":
            # Determine child type by peeking the next deeper line.
            child: Any = {}
            if i + 1 < n:
                next_indent, next_stripped = lines[i + 1]
                if next_indent > indent and next_stripped.startswith("- "):
                    child = []
            if isinstance(container, dict):
                container[key] = child
            stack.append((indent, child))
        else:
            if isinstance(container, dict):
                container[key] = _coerce_scalar(value)
        i += 1

    return root


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("Config root must be a mapping.")
        return data
    except ImportError:
        return _fallback_parse(text)


def _parse_training_windows(raw_windows: Any) -> List[tuple]:
    """Parse ``training.windows`` into a list of ``(start, end)`` string pairs.

    Accepts either mapping items (``{start: ..., end: ...}``) or scalar strings
    using ``/``, ``..`` or ``,`` as the start/end delimiter (e.g.
    ``"2018-08-01/2018-11-01"``). The scalar form is used so the project's tiny
    fallback YAML parser (which cannot parse a list of maps) still works when
    PyYAML is absent.
    """
    if not isinstance(raw_windows, list):
        return []
    windows: List[tuple] = []
    for item in raw_windows:
        start = end = None
        if isinstance(item, dict):
            start = item.get("start")
            end = item.get("end")
        elif isinstance(item, str):
            for sep in ("/", "..", ","):
                if sep in item:
                    parts = item.split(sep)
                    if len(parts) == 2:
                        start, end = parts[0], parts[1]
                    break
        if start is None or end is None:
            continue
        start = str(start).strip()
        end = str(end).strip()
        if start and end:
            windows.append((start, end))
    return windows


def load_config(path: str) -> Config:
    """Load a :class:`Config` from a YAML file."""
    raw = _load_yaml(path)
    base_dir = os.path.dirname(os.path.abspath(path))

    symbols = raw.get("symbols") or []
    if not isinstance(symbols, list):
        raise ValueError("`symbols` must be a list.")
    symbols = [str(s).strip() for s in symbols if str(s).strip()]

    data_raw = raw.get("data") or {}
    rolling_raw = raw.get("rolling") or {}
    dist_raw = raw.get("distribution") or {}
    integ_raw = raw.get("integration") or {}
    rf_raw = raw.get("rf_model") or {}
    vol_raw = raw.get("volume") or {}
    news_raw = raw.get("news") or {}
    corr_raw = raw.get("correlation") or {}
    scaling_raw = raw.get("feature_scaling") or {}
    halving_raw = raw.get("halving") or {}
    beta_raw = raw.get("beta") or {}
    training_raw = raw.get("training") or {}
    mdp_raw = raw.get("mdp") or {}
    sim_raw = raw.get("simulation") or {}
    paths_raw = raw.get("paths") or {}

    data_cfg = DataConfig(
        raw_dir=str(data_raw.get("raw_dir", "data/raw")),
        output_path=str(data_raw.get("output_path", "data/distribution_snapshot.json")),
        daily_file_suffix=str(data_raw.get("daily_file_suffix", "_daily.csv")),
        minute_file_suffix=str(data_raw.get("minute_file_suffix", "_1m.csv")),
        require_daily_source=bool(data_raw.get("require_daily_source", True)),
        require_minute_source=bool(data_raw.get("require_minute_source", True)),
        timeframe=str(data_raw.get("timeframe", "1m")),
        daily_resample=bool(data_raw.get("daily_resample", True)),
    )
    rolling_cfg = RollingConfig(
        window_days=int(rolling_raw.get("window_days", 30)),
        variance_mode=str(rolling_raw.get("variance_mode", "population")).lower(),
    )
    dist_cfg = DistributionConfig(
        method=str(dist_raw.get("method", "histogram")).lower(),
        bins=int(dist_raw.get("bins", 400)),
        min_observations=int(dist_raw.get("min_observations", 60)),
    )
    integ_cfg = IntegrationConfig(
        grid_points=int(integ_raw.get("grid_points", 0)),
        clip_to_grid=bool(integ_raw.get("clip_to_grid", True)),
    )
    rf_cfg = RFModelConfig(
        scope=str(rf_raw.get("scope", "per_symbol")),
        fallback_to_baseline_if_symbol_model_missing=bool(
            rf_raw.get("fallback_to_baseline_if_symbol_model_missing", True)
        ),
        min_training_rows=int(rf_raw.get("min_training_rows", 250)),
        n_estimators=int(rf_raw.get("n_estimators", 200)),
        max_depth=int(rf_raw.get("max_depth", 0)),
        random_state=int(rf_raw.get("random_state", 42)),
        n_jobs=int(rf_raw.get("n_jobs", 4)),
        decision_threshold=float(rf_raw.get("decision_threshold", 0.0)),
        train_frequency=str(rf_raw.get("train_frequency", "monthly")),
        history_years=int(rf_raw.get("history_years", 4)),
        include_optional_component_features=bool(
            rf_raw.get("include_optional_component_features", False)
        ),
    )
    vol_cfg = VolumeConfig(
        enabled=bool(vol_raw.get("enabled", True)),
        kernel=str(vol_raw.get("kernel", "rbf")),
        alpha=float(vol_raw.get("alpha", 1.0)),
        gamma=float(vol_raw.get("gamma", 0.0)),
        min_train_days=int(vol_raw.get("min_train_days", 120)),
        walk_forward=bool(vol_raw.get("walk_forward", True)),
    )
    news_categories = news_raw.get("categories")
    if not isinstance(news_categories, list) or not news_categories:
        news_categories = NewsConfig().categories
    news_queries = news_raw.get("queries")
    news_queries = news_queries if isinstance(news_queries, dict) else {}
    news_symbol_queries = news_raw.get("symbol_queries")
    news_symbol_queries = news_symbol_queries if isinstance(news_symbol_queries, dict) else {}
    news_cfg = NewsConfig(
        enabled=bool(news_raw.get("enabled", True)),
        strict_coverage=bool(news_raw.get("strict_coverage", True)),
        provider=str(news_raw.get("provider", "newsapi")),
        api_key_env=str(news_raw.get("api_key_env", "NEWSAPI_API_KEY")),
        gdelt_base_url=str(news_raw.get("gdelt_base_url", "")),
        gdelt_max_records=int(news_raw.get("gdelt_max_records", 250)),
        gdelt_min_request_interval_seconds=float(
            news_raw.get("gdelt_min_request_interval_seconds", 5.0)
        ),
        gdelt_doc_max_lookback_days=int(
            news_raw.get("gdelt_doc_max_lookback_days", 90)
        ),
        raw_path=str(news_raw.get("raw_path", "data/raw/news.jsonl")),
        coverage_manifest_path=str(
            news_raw.get("coverage_manifest_path", "data/news_coverage_manifest.jsonl")
        ),
        output_path=str(news_raw.get("output_path", "data/news_features_daily.jsonl")),
        report_path=str(news_raw.get("report_path", "reports/news_ingestion_report.json")),
        sentiment_cache_path=str(
            news_raw.get("sentiment_cache_path", "data/news_sentiment_cache.jsonl")
        ),
        backfill_start=str(news_raw.get("backfill_start", "2022-01-01")),
        language=str(news_raw.get("language", "en")),
        categories=[str(c) for c in news_categories],
        sentiment_backend=str(news_raw.get("sentiment_backend", "finbert")),
        sentiment_table_path=str(news_raw.get("sentiment_table_path", "")),
        queries=news_queries,
        symbol_queries=news_symbol_queries,
        intraday_news_enabled=bool(news_raw.get("intraday_news_enabled", False)),
        feature_lookback_hours=int(news_raw.get("feature_lookback_hours", 24)),
        news_safety_lag_seconds=int(news_raw.get("news_safety_lag_seconds", 300)),
        historical_news_availability_lag_seconds=int(
            news_raw.get("historical_news_availability_lag_seconds", 300)
        ),
        max_news_feature_age_minutes=int(news_raw.get("max_news_feature_age_minutes", 30)),
        intraday_output_path=str(
            news_raw.get("intraday_output_path", "data/news_features_intraday.jsonl")
        ),
        feature_version=str(news_raw.get("feature_version", "news_v1")),
        gkg_features_db=str(news_raw.get("gkg_features_db", "data/gkg_4y_2h.sqlite")),
    )
    corr_cfg = CorrelationConfig(
        enabled=bool(corr_raw.get("enabled", True)),
        lookback_days=int(corr_raw.get("lookback_days", 90)),
        min_required_days=int(corr_raw.get("min_required_days", 60)),
        method=str(corr_raw.get("method", "pearson")).lower(),
        fallback_self_corr=float(corr_raw.get("fallback_self_corr", 1.0)),
        fallback_cross_corr=float(corr_raw.get("fallback_cross_corr", 0.0)),
        output_path=str(corr_raw.get("output_path", "data/correlation_matrix_daily.jsonl")),
    )
    scaling_cfg = FeatureScalingConfig(
        enabled=bool(scaling_raw.get("enabled", True)),
        scope=str(scaling_raw.get("scope", "per_symbol")),
        scaler=str(scaling_raw.get("scaler", "robust")).lower(),
        imputer=str(scaling_raw.get("imputer", "median")).lower(),
        fit_on_train_only=bool(scaling_raw.get("fit_on_train_only", True)),
    )
    halving_dates = halving_raw.get("dates")
    if not isinstance(halving_dates, list):
        halving_dates = []
    halving_cfg = HalvingConfig(
        enabled=bool(halving_raw.get("enabled", True)),
        season_seed=int(halving_raw.get("season_seed", 0)),
        dates=[str(d) for d in halving_dates],
    )
    beta_cfg = BetaConfig(
        enabled=bool(beta_raw.get("enabled", True)),
        market_cap_path=str(beta_raw.get("market_cap_path", "data/market_cap_daily.csv")),
        output_path=str(beta_raw.get("output_path", "data/capm_beta_daily.jsonl")),
        return_lag_days=int(beta_raw.get("return_lag_days", 30)),
        window_days=int(beta_raw.get("window_days", 90)),
        market_top_n=int(beta_raw.get("market_top_n", 30)),
    )
    training_cfg = TrainingConfig(
        windows=_parse_training_windows(training_raw.get("windows")),
        entry_every_days=int(training_raw.get("entry_every_days", 7)),
        max_hold_days=int(training_raw.get("max_hold_days", 0)),
        spill_to_disk=bool(training_raw.get("spill_to_disk", True)),
        spill_flush_rows=int(training_raw.get("spill_flush_rows", 250_000)),
        train_max_rows=int(training_raw.get("train_max_rows", 3_000_000)),
    )
    mdp_cfg = MDPConfig(
        add_margin_step=float(mdp_raw.get("add_margin_step", 10.0)),
        max_add_margin_per_decision=float(mdp_raw.get("max_add_margin_per_decision", 1e18)),
        max_total_added_margin=float(mdp_raw.get("max_total_added_margin", 1e18)),
    )
    sim_cfg = SimulationConfig(
        fee_rate=float(sim_raw.get("fee_rate", 0.0004)),
        funding_rate=float(sim_raw.get("funding_rate", 0.0)),
        slippage=float(sim_raw.get("slippage", 0.0)),
        decision_threshold=float(sim_raw.get("decision_threshold", 0.0)),
        close_epsilon=float(sim_raw.get("close_epsilon", 0.0)),
        use_rf=bool(sim_raw.get("use_rf", True)),
        parallel=bool(sim_raw.get("parallel", True)),
        max_workers=int(sim_raw.get("max_workers", 0)),
    )
    paths_defaults = PathsConfig()
    paths_cfg = PathsConfig(
        distribution_snapshot=str(
            paths_raw.get("distribution_snapshot", paths_defaults.distribution_snapshot)
        ),
        integral_cache=str(paths_raw.get("integral_cache", paths_defaults.integral_cache)),
        predicted_daily_volume=str(
            paths_raw.get("predicted_daily_volume", paths_defaults.predicted_daily_volume)
        ),
        news_features_daily=str(
            paths_raw.get("news_features_daily", paths_defaults.news_features_daily)
        ),
        correlation_matrix_daily=str(
            paths_raw.get("correlation_matrix_daily", paths_defaults.correlation_matrix_daily)
        ),
        rf_dataset_dir=str(paths_raw.get("rf_dataset_dir", paths_defaults.rf_dataset_dir)),
        simulation_results=str(
            paths_raw.get("simulation_results", paths_defaults.simulation_results)
        ),
        volume_report=str(paths_raw.get("volume_report", paths_defaults.volume_report)),
        rf_policy_report=str(paths_raw.get("rf_policy_report", paths_defaults.rf_policy_report)),
        speed_benchmark=str(paths_raw.get("speed_benchmark", paths_defaults.speed_benchmark)),
        models_promoted=str(paths_raw.get("models_promoted", paths_defaults.models_promoted)),
        models_staging=str(paths_raw.get("models_staging", paths_defaults.models_staging)),
        models_archive=str(paths_raw.get("models_archive", paths_defaults.models_archive)),
        volume_models=str(paths_raw.get("volume_models", paths_defaults.volume_models)),
    )

    return Config(
        symbols=symbols,
        data=data_cfg,
        rolling=rolling_cfg,
        distribution=dist_cfg,
        integration=integ_cfg,
        rf_model=rf_cfg,
        volume=vol_cfg,
        news=news_cfg,
        correlation=corr_cfg,
        feature_scaling=scaling_cfg,
        halving=halving_cfg,
        beta=beta_cfg,
        training=training_cfg,
        mdp=mdp_cfg,
        simulation=sim_cfg,
        paths=paths_cfg,
        base_dir=base_dir,
    )
