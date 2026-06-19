"""Configuration for the portfolio (MVO + B&B) module."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class BranchAndBoundConfig:
    enabled: bool = True
    max_nodes: int = 50000
    max_runtime_ms: int = 5000
    objective_tolerance: float = 1e-9


@dataclass
class PortfolioPaths:
    simulated_returns: str = "data/simulation_results.csv"
    realized_returns: str = "data/realized_symbol_returns.jsonl"
    output_path: str = "data/portfolio_weights.json"


@dataclass
class PortfolioConfig:
    enabled: bool = True
    update_frequency: str = "daily"
    update_time_utc: str = "00:10"
    lookback_days: int = 30
    min_required_days: int = 25
    base_capital_per_symbol: float = 1000.0
    risk_aversion: float = 2.0
    allow_cash: bool = True
    max_weight_per_symbol: float = 0.40
    covariance_mode: str = "sample"
    covariance_epsilon: float = 1e-6
    long_short_split: str = "equal"
    integer_tolerance: float = 1e-9
    default_weight_step: float = 0.01
    symbols: List[str] = field(default_factory=list)
    weight_steps: Dict[str, float] = field(default_factory=dict)
    bnb: BranchAndBoundConfig = field(default_factory=BranchAndBoundConfig)
    paths: PortfolioPaths = field(default_factory=PortfolioPaths)
    base_dir: str = "."

    def resolve(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(os.getcwd(), path))

    def weight_step_for(self, symbol: str) -> float:
        return float(self.weight_steps.get(symbol, self.default_weight_step))


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
        from ..config import _fallback_parse  # reuse minimal parser

        return _fallback_parse(text)


def load_portfolio_config(path: str) -> PortfolioConfig:
    raw = _load_yaml(path)
    pf = raw.get("portfolio") or {}
    bnb_raw = raw.get("branch_and_bound") or {}
    paths_raw = raw.get("paths") or {}

    symbols = pf.get("symbols") or []
    symbols = [str(s).strip() for s in symbols if str(s).strip()]

    weight_steps_raw = pf.get("weight_steps") or {}
    weight_steps = {str(k): float(v) for k, v in weight_steps_raw.items()}

    paths_default = PortfolioPaths()
    paths = PortfolioPaths(
        simulated_returns=str(paths_raw.get("simulated_returns", paths_default.simulated_returns)),
        realized_returns=str(paths_raw.get("realized_returns", paths_default.realized_returns)),
        output_path=str(paths_raw.get("output_path", paths_default.output_path)),
    )

    bnb = BranchAndBoundConfig(
        enabled=bool(bnb_raw.get("enabled", True)),
        max_nodes=int(bnb_raw.get("max_nodes", 50000)),
        max_runtime_ms=int(bnb_raw.get("max_runtime_ms", 5000)),
        objective_tolerance=float(bnb_raw.get("objective_tolerance", 1e-9)),
    )

    return PortfolioConfig(
        enabled=bool(pf.get("enabled", True)),
        update_frequency=str(pf.get("update_frequency", "daily")),
        update_time_utc=str(pf.get("update_time_utc", "00:10")),
        lookback_days=int(pf.get("lookback_days", 30)),
        min_required_days=int(pf.get("min_required_days", 25)),
        base_capital_per_symbol=float(pf.get("base_capital_per_symbol", 1000.0)),
        risk_aversion=float(pf.get("risk_aversion", 2.0)),
        allow_cash=bool(pf.get("allow_cash", True)),
        max_weight_per_symbol=float(pf.get("max_weight_per_symbol", 0.40)),
        covariance_mode=str(pf.get("covariance_mode", "sample")).lower(),
        covariance_epsilon=float(pf.get("covariance_epsilon", 1e-6)),
        long_short_split=str(pf.get("long_short_split", "equal")),
        integer_tolerance=float(pf.get("integer_tolerance", 1e-9)),
        default_weight_step=float(pf.get("default_weight_step", 0.01)),
        symbols=symbols,
        weight_steps=weight_steps,
        bnb=bnb,
        paths=paths,
        base_dir=os.path.dirname(os.path.abspath(path)),
    )
