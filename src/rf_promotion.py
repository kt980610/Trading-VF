"""Per-symbol model promotion/evaluation (spec section 16).

Promotion is decided independently per symbol -- a strong BTCUSDT model never
promotes ETHUSDT. A candidate must beat its OWN baseline on both validation and
test PnL and must not materially increase liquidations.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple


@dataclass
class PolicyMetrics:
    validation_pnl: float
    test_pnl: float
    liquidation_count: int


def should_promote(
    candidate: PolicyMetrics,
    baseline: PolicyMetrics,
    max_liq_ratio: float = 1.0,
) -> Tuple[bool, Optional[str]]:
    """Apply the spec section-16 promotion gates. Returns (promote, reason)."""
    if candidate.validation_pnl <= baseline.validation_pnl:
        return False, "validation_pnl_not_better"
    if candidate.test_pnl <= baseline.test_pnl:
        return False, "test_pnl_not_better"
    if candidate.liquidation_count > baseline.liquidation_count * max_liq_ratio:
        return False, "liquidation_count_too_high"
    return True, None


def _archive_month_dir(archive_root: str, symbol: str) -> str:
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    return os.path.join(archive_root, month, symbol)


def archive_current_promoted(symbol: str, promoted_root: str, archive_root: str) -> Optional[str]:
    """Copy the currently promoted symbol artifacts into the monthly archive."""
    src = os.path.join(promoted_root, symbol)
    if not os.path.isdir(src):
        return None
    dst = _archive_month_dir(archive_root, symbol)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def promote_symbol(
    symbol: str,
    staging_root: str,
    promoted_root: str,
    archive_root: str,
) -> Optional[str]:
    """Promote staging artifacts to promoted, archiving the previous version."""
    staging = os.path.join(staging_root, symbol)
    if not os.path.isdir(staging):
        return None

    archive_current_promoted(symbol, promoted_root, archive_root)

    promoted = os.path.join(promoted_root, symbol)
    if os.path.isdir(promoted):
        shutil.rmtree(promoted)
    shutil.copytree(staging, promoted)
    return promoted


def evaluate_and_promote(
    symbol: str,
    candidate: PolicyMetrics,
    baseline: PolicyMetrics,
    staging_root: str,
    promoted_root: str,
    archive_root: str,
    max_liq_ratio: float = 1.0,
) -> dict:
    """Decide and (if eligible) perform promotion for a single symbol."""
    promote, reason = should_promote(candidate, baseline, max_liq_ratio)
    promoted_path = None
    if promote:
        promoted_path = promote_symbol(symbol, staging_root, promoted_root, archive_root)
    return {
        "symbol": symbol,
        "model_promoted": bool(promote and promoted_path is not None),
        "reason_if_not_promoted": None if promote else reason,
        "promoted_path": promoted_path,
        "candidate": candidate.__dict__,
        "baseline": baseline.__dict__,
    }
