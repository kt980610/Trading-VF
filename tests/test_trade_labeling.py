"""Tests for trade-based PnL-rate labeling (spec section 8; tests 16,17,19)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trade_labeling import MinuteDecision, label_trade, trade_pnl_rate


def test_trade_pnl_rate_is_net_pnl_over_elapsed():
    # Net PnL already includes fees/funding/slippage upstream.
    assert trade_pnl_rate(100.0, 10) == 10.0
    assert trade_pnl_rate(100.0, 0) == 100.0  # elapsed floored at 1


def test_terminal_minute_label_is_one():
    minutes = [
        MinuteDecision(index=0, net_close_pnl_now=5.0),
        MinuteDecision(index=1, net_close_pnl_now=8.0),
        MinuteDecision(index=2, net_close_pnl_now=3.0),  # terminal
    ]
    labels = label_trade(minutes, entry_index=0)
    assert labels[-1].close_label == 1


def test_close_label_one_when_now_beats_future():
    # Minute 0: rate 10/1 = 10. Future best rate lower -> should close now (1).
    minutes = [
        MinuteDecision(index=0, net_close_pnl_now=10.0),  # rate 10
        MinuteDecision(index=1, net_close_pnl_now=10.0),  # rate 5
        MinuteDecision(index=2, net_close_pnl_now=9.0),   # rate 3 (terminal)
    ]
    labels = label_trade(minutes, entry_index=0)
    assert labels[0].close_label == 1
    assert labels[0].best_future_trade_pnl_rate == 5.0


def test_close_label_zero_when_future_is_better():
    # Minute 0 rate 1; later minute has higher rate -> continue (label 0).
    minutes = [
        MinuteDecision(index=0, net_close_pnl_now=1.0),   # rate 1
        MinuteDecision(index=1, net_close_pnl_now=20.0),  # rate 10
        MinuteDecision(index=2, net_close_pnl_now=3.0),   # rate 1 (terminal)
    ]
    labels = label_trade(minutes, entry_index=0)
    assert labels[0].close_label == 0


def test_epsilon_prevents_flip_on_noise():
    minutes = [
        MinuteDecision(index=0, net_close_pnl_now=10.0),   # rate 10
        MinuteDecision(index=1, net_close_pnl_now=19.98),  # rate ~9.99
        MinuteDecision(index=2, net_close_pnl_now=1.0),    # terminal
    ]
    # now (10.0) marginally beats the future best (9.99). Without epsilon that
    # noisy edge labels CLOSE(1); a small epsilon requires a real margin so the
    # noisy close is suppressed -> CONTINUE(0).
    no_eps = label_trade(minutes, entry_index=0, close_epsilon=0.0)
    with_eps = label_trade(minutes, entry_index=0, close_epsilon=0.1)
    assert no_eps[0].close_label == 1
    assert with_eps[0].close_label == 0


def test_sample_weight_grows_with_value_gap():
    minutes = [
        MinuteDecision(index=0, net_close_pnl_now=1.0),
        MinuteDecision(index=1, net_close_pnl_now=100.0),
        MinuteDecision(index=2, net_close_pnl_now=2.0),
    ]
    labels = label_trade(minutes, entry_index=0)
    # Minute 0 has a large gap to the future best, so its weight exceeds the floor.
    assert labels[0].sample_weight > 1.0
