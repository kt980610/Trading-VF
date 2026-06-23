import numpy as np

from src import pricing


def test_scenario_price_uses_entry_not_current():
    # ScenarioPrice(z) = EntryPrice * (1 + z), independent of current/final price.
    assert np.isclose(pricing.scenario_price(100.0, 0.1), 110.0)
    assert np.isclose(pricing.scenario_price(200.0, -0.05), 190.0)


def test_scenario_price_not_current_times_one_plus_z():
    entry, current, z = 100.0, 130.0, 0.1
    scenario = pricing.scenario_price(entry, z)
    assert scenario == entry * (1 + z)
    assert scenario != current * (1 + z)


def test_long_payoff_is_z():
    for z in (-0.2, 0.0, 0.15):
        assert pricing.long_payoff_pct(z) == z


def test_short_payoff_is_minus_z():
    for z in (-0.2, 0.0, 0.15):
        assert pricing.short_payoff_pct(z) == -z


def test_long_short_liq_z_approximations():
    # long_liq_z = -M/N, short_liq_z = M/N
    assert np.isclose(pricing.long_liq_z(100.0, 1000.0, 100.0), -0.1)
    assert np.isclose(pricing.short_liq_z(100.0, 1000.0, 100.0), 0.1)


def test_real_liquidation_price_overrides_approximation():
    z = pricing.long_liq_z(100.0, 1000.0, 100.0, liquidation_price_long=95.0)
    assert np.isclose(z, -0.05)


def test_add_margin_does_not_change_notional():
    leg = pricing.LegState(side="long", entry_price=100.0, qty=10.0, margin_current=100.0)
    notional_before = leg.notional
    leg.margin_current = pricing.add_margin(leg.margin_current, 50.0)
    assert leg.notional == notional_before
    assert leg.margin_current == 150.0


def test_liq_z_after_add_widens_cutoff():
    leg = pricing.LegState(side="long", entry_price=100.0, qty=10.0, margin_current=100.0)
    before = leg.liq_z()
    after = leg.liq_z_after_add(100.0)
    # More margin -> liquidation cutoff further from 0 (more negative for long).
    assert after < before
