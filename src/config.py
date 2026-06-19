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
    raw_path: str = "data/raw/news.jsonl"
    output_path: str = "data/news_features_daily.jsonl"
    categories: List[str] = field(
        default_factory=lambda: [
            "macro",
            "policy",
            "stock_market",
            "crypto_market",
            "symbol_specific",
        ]
    )
    sentiment_backend: str = "lexicon"  # lexicon | table | finbert | vader
    sentiment_table_path: str = ""


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
    mdp_raw = raw.get("mdp") or {}
    sim_raw = raw.get("simulation") or {}
    paths_raw = raw.get("paths") or {}

    data_cfg = DataConfig(
        raw_dir=str(data_raw.get("raw_dir", "data/raw")),
        output_path=str(data_raw.get("output_path", "data/distribution_snapshot.json")),
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
    news_cfg = NewsConfig(
        enabled=bool(news_raw.get("enabled", True)),
        raw_path=str(news_raw.get("raw_path", "data/raw/news.jsonl")),
        output_path=str(news_raw.get("output_path", "data/news_features_daily.jsonl")),
        categories=[str(c) for c in news_categories],
        sentiment_backend=str(news_raw.get("sentiment_backend", "lexicon")),
        sentiment_table_path=str(news_raw.get("sentiment_table_path", "")),
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
        mdp=mdp_cfg,
        simulation=sim_cfg,
        paths=paths_cfg,
        base_dir=base_dir,
    )
