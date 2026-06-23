"""Trade-based binary close labels and PnL-rate targets (spec section 8).

A *trade* spans from its entry minute to its normal close or complete
liquidation. All decision minutes of one trade share a ``trade_id`` and must
never be split across train/validation/test.

For each decision minute ``t``::

    elapsed_minutes      = max(1, t_index - entry_index + 1)
    trade_pnl_rate_now   = net_close_pnl_now / elapsed_minutes

where ``net_close_pnl_now`` already includes fees, funding and slippage. The
binary label compares "close now" against the best achievable rate at any LATER
decision minute of the SAME trade::

    best_future_rate = max(trade_pnl_rate(u) for u after t)   # 0 at the end
    close_label      = int(trade_pnl_rate_now >= best_future_rate + close_epsilon)

The terminal minute (normal close or full liquidation) always labels ``1``.
``close_epsilon`` suppresses label flips from tiny noise. ``decision_value_gap``
(= |now - best_future|) drives the regression-style sample weight so the model
optimizes realized trade PnL, not bare classification accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class MinuteDecision:
    """One decision minute inside a trade (ordered by time)."""

    index: int  # absolute minute index in the source frame
    net_close_pnl_now: float  # PnL if fully closed now, net of fees/funding/slippage


@dataclass
class LabeledMinute:
    index: int
    elapsed_minutes: int
    trade_pnl_rate_now: float
    best_future_trade_pnl_rate: float
    decision_value_gap: float
    close_label: int
    sample_weight: float


def trade_pnl_rate(net_close_pnl_now: float, elapsed_minutes: int) -> float:
    return float(net_close_pnl_now) / float(max(1, int(elapsed_minutes)))


def label_trade(
    minutes: List[MinuteDecision],
    entry_index: int,
    close_epsilon: float = 0.0,
    weight_floor: float = 1.0,
) -> List[LabeledMinute]:
    """Label every decision minute of ONE trade (spec section 8).

    The last minute is the terminal (close/liquidation) and is forced to label 1.
    """
    if not minutes:
        return []

    rates: List[float] = []
    elapsed: List[int] = []
    for m in minutes:
        e = max(1, m.index - entry_index + 1)
        elapsed.append(e)
        rates.append(trade_pnl_rate(m.net_close_pnl_now, e))

    n = len(minutes)
    out: List[LabeledMinute] = []
    for i in range(n):
        is_terminal = i == n - 1
        # Best achievable rate strictly AFTER this minute (0.0 if none -> closing
        # now is at least as good as not trading further).
        if is_terminal:
            best_future = 0.0
        else:
            best_future = max(rates[i + 1:]) if i + 1 < n else 0.0
        gap = abs(rates[i] - best_future)
        if is_terminal:
            label = 1
        else:
            label = int(rates[i] >= best_future + close_epsilon)
        out.append(
            LabeledMinute(
                index=minutes[i].index,
                elapsed_minutes=elapsed[i],
                trade_pnl_rate_now=rates[i],
                best_future_trade_pnl_rate=best_future,
                decision_value_gap=gap,
                close_label=label,
                sample_weight=float(weight_floor) + abs(float(gap)),
            )
        )
    return out
