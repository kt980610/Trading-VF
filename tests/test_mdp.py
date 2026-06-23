import numpy as np

from src import mdp as mdp_mod
from src.integral_cache import build_cache_for_symbol
from src.pricing import LegState
from tests.synthetic import make_symbol_entry


def test_trigger_true_when_remaining_liq_closer():
    assert mdp_mod.mdp_trigger(current_price=100.0, remaining_liq_price=98.0, first_liq_price=90.0)


def test_trigger_false_when_remaining_liq_farther():
    assert not mdp_mod.mdp_trigger(current_price=100.0, remaining_liq_price=85.0, first_liq_price=95.0)


def test_actions_are_integer_multiples_of_step():
    actions = mdp_mod.enumerate_actions(55.0, 10.0, 1e18, 1e18, 0.0)
    assert actions == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    assert all(np.isclose(a % 10.0, 0.0) for a in actions)


def test_actions_never_exceed_remaining_balance():
    actions = mdp_mod.enumerate_actions(55.0, 10.0, 1e18, 1e18, 0.0)
    assert max(actions) <= 55.0


def test_actions_only_zero_when_balance_below_step():
    assert mdp_mod.enumerate_actions(5.0, 10.0, 1e18, 1e18, 0.0) == [0.0]


def test_actions_respect_per_decision_and_total_caps():
    assert mdp_mod.enumerate_actions(100.0, 10.0, 25.0, 1e18, 0.0) == [0.0, 10.0, 20.0]
    assert mdp_mod.enumerate_actions(100.0, 10.0, 1e18, 30.0, 5.0) == [0.0, 10.0, 20.0]


def test_decide_uses_only_return_component():
    entry = make_symbol_entry(seed=21)
    cache = build_cache_for_symbol("S", entry, window=30)
    # Strip every non-return component to prove the MDP only touches "return".
    cache.cum_long = {"return": cache.cum_long["return"]}
    cache.cum_short = {"return": cache.cum_short["return"]}

    leg = LegState("long", entry_price=100.0, qty=10.0, margin_current=50.0)
    decision = mdp_mod.decide(
        cache, leg, current_price=92.0, remaining_balance_before=500.0,
        first_liq_price=80.0, add_margin_step=10.0,
    )
    assert decision.action in (mdp_mod.ACTION_CLOSE, mdp_mod.ACTION_ADD_MARGIN_CONTINUE)
    assert np.isclose(decision.x_best % 10.0, 0.0)


def test_decide_not_triggered_does_not_search():
    entry = make_symbol_entry(seed=22)
    cache = build_cache_for_symbol("S", entry, window=30)
    leg = LegState("long", entry_price=100.0, qty=10.0, margin_current=200.0)  # liq price 80
    decision = mdp_mod.decide(
        cache, leg, current_price=100.0, remaining_balance_before=500.0,
        first_liq_price=99.0, add_margin_step=10.0,
    )
    assert decision.triggered is False
    assert decision.action == mdp_mod.ACTION_CLOSE
    assert decision.x_best == 0.0


def test_decide_closes_when_costs_dominate():
    entry = make_symbol_entry(seed=23)
    cache = build_cache_for_symbol("S", entry, window=30)
    leg = LegState("long", entry_price=100.0, qty=10.0, margin_current=50.0)
    decision = mdp_mod.decide(
        cache, leg, current_price=92.0, remaining_balance_before=500.0,
        first_liq_price=80.0, add_margin_step=10.0, expected_costs=1e9,
    )
    assert decision.triggered is True
    assert decision.action == mdp_mod.ACTION_CLOSE
