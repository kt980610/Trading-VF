"""CLI: build daily MVO + Branch-and-Bound portfolio weights (spec section 15)."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.portfolio.config import load_portfolio_config
    from src.portfolio import returns_loader as rl
    from src.portfolio import hybrid_matrix as hm
    from src.portfolio import mvo as mvo_mod
    from src.portfolio import branch_and_bound as bnb_mod
    from src.portfolio import weights_writer as ww
else:
    from .config import load_portfolio_config
    from . import returns_loader as rl
    from . import hybrid_matrix as hm
    from . import mvo as mvo_mod
    from . import branch_and_bound as bnb_mod
    from . import weights_writer as ww


def run(config_path: str, as_of_date: str = None) -> dict:
    config = load_portfolio_config(config_path)
    if as_of_date is None:
        as_of_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 2-4. Load and merge returns (realized overrides simulated).
    simulated = rl.load_simulated_returns(
        config.resolve(config.paths.simulated_returns), config.base_capital_per_symbol
    )
    realized = rl.load_realized_returns(
        config.resolve(config.paths.realized_returns), config.base_capital_per_symbol
    )
    hybrid = hm.build_hybrid_returns(simulated, realized)

    # 5. Rolling return matrix.
    rm = hm.build_return_matrix(
        hybrid, config.symbols, as_of_date,
        lookback_days=config.lookback_days,
        min_required_days=config.min_required_days,
    )

    # 6. mu / Sigma.
    stats = mvo_mod.compute_statistics(rm.R, config.covariance_mode, config.covariance_epsilon)

    import numpy as np

    steps = np.array([config.weight_step_for(s) for s in rm.valid_symbols], dtype="float64")

    if len(rm.valid_symbols) == 0:
        continuous_w = np.zeros(0)
        bnb_result = bnb_mod.BnBResult(np.zeros(0, dtype="int64"), np.zeros(0), 0.0, bnb_mod.STATUS_OPTIMAL, 0)
    else:
        # 7. Continuous MVO.
        continuous_w = mvo_mod.solve_continuous_mvo(
            stats.mu, stats.sigma_reg, config.risk_aversion,
            config.max_weight_per_symbol, allow_cash=config.allow_cash,
        )
        # 8. Branch and Bound discrete weights.
        bnb_result = bnb_mod.solve_branch_and_bound(
            stats.mu, stats.sigma_reg, steps, config.risk_aversion,
            config.max_weight_per_symbol, w_continuous=continuous_w,
            max_nodes=config.bnb.max_nodes,
            max_runtime_ms=config.bnb.max_runtime_ms,
            objective_tolerance=config.bnb.objective_tolerance,
        )

    payload = ww.build_payload(
        config, as_of_date, stats, rm.valid_symbols, rm.invalid,
        continuous_w, bnb_result.k, bnb_result.w_discrete,
    )

    out_path = config.resolve(config.paths.output_path)
    ww.write_atomic(payload, out_path)

    # 10. Console summary.
    for symbol in config.symbols:
        entry = payload["symbols"][symbol]
        if entry["valid"]:
            print(
                f"{symbol} valid w_disc={entry['weight_discrete']:.4f} "
                f"long={entry['long_weight']:.4f} short={entry['short_weight']:.4f} k={entry['k']}"
            )
        else:
            print(f"{symbol} invalid reason={entry['reason']}")
    print(f"sum_weight_discrete={payload['sum_weight_discrete']:.4f} cash={payload['cash_weight']:.4f}")
    print(f"bnb_status={bnb_result.status} nodes={bnb_result.nodes_explored}")
    print(f"portfolio weights written to {out_path}")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build MVO + B&B portfolio weights.")
    parser.add_argument("--config", default="config/portfolio_config.yaml")
    parser.add_argument("--as-of-date", default=None, help="YYYY-MM-DD (UTC). Defaults to today.")
    args = parser.parse_args(argv)
    run(args.config, args.as_of_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
