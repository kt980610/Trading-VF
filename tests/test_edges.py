from src import edges as edges_mod
from src.integral_cache import build_cache_for_symbol
from src.pricing import LegState
from tests.synthetic import make_symbol_entry

LONG_EDGES = [
    "LongEdge_Return", "LongEdge_Mean", "LongEdge_Var",
    "LongEdge_MeanOfMean", "LongEdge_VarOfMean",
]
SHORT_EDGES = [
    "ShortEdge_Return", "ShortEdge_Mean", "ShortEdge_Var",
    "ShortEdge_MeanOfMean", "ShortEdge_VarOfMean",
]


def _legs(entry):
    cache = build_cache_for_symbol("S", entry, window=30)
    long_leg = LegState("long", entry_price=100.0, qty=10.0, margin_current=100.0)
    short_leg = LegState("short", entry_price=100.0, qty=10.0, margin_current=100.0)
    return cache, long_leg, short_leg


def test_five_long_edges_computed():
    entry = make_symbol_entry(seed=5)
    cache, long_leg, _ = _legs(entry)
    comp = edges_mod.compute_long_components(cache, y=0.0, long_liq_z=long_leg.liq_z(), notional_long=long_leg.notional)
    for key in LONG_EDGES:
        assert key in comp


def test_five_short_edges_computed():
    entry = make_symbol_entry(seed=6)
    cache, _, short_leg = _legs(entry)
    comp = edges_mod.compute_short_components(cache, y=0.0, short_liq_z=short_leg.liq_z(), notional_short=short_leg.notional)
    for key in SHORT_EDGES:
        assert key in comp


def test_long_edge_is_right_minus_left():
    entry = make_symbol_entry(seed=7)
    cache, long_leg, _ = _legs(entry)
    comp = edges_mod.compute_long_components(cache, 0.01, long_leg.liq_z(), long_leg.notional)
    assert comp["LongEdge_Return"] == comp["LongRightPnL_Return"] - comp["LongLeftPnL_Return"]


def test_short_edge_is_left_minus_right():
    entry = make_symbol_entry(seed=8)
    cache, _, short_leg = _legs(entry)
    comp = edges_mod.compute_short_components(cache, 0.01, short_leg.liq_z(), short_leg.notional)
    assert comp["ShortEdge_Return"] == comp["ShortLeftPnL_Return"] - comp["ShortRightPnL_Return"]


def test_single_leg_mode_only_remaining_leg():
    entry = make_symbol_entry(seed=9)
    cache, long_leg, short_leg = _legs(entry)

    long_only = edges_mod.compute_features(
        cache, edges_mod.MODE_LONG_ONLY, 100.0, long_leg=long_leg, short_leg=short_leg
    )
    assert any(k.startswith("LongEdge_") for k in long_only)
    assert not any(k.startswith("ShortEdge_") for k in long_only)

    short_only = edges_mod.compute_features(
        cache, edges_mod.MODE_SHORT_ONLY, 100.0, long_leg=long_leg, short_leg=short_leg
    )
    assert any(k.startswith("ShortEdge_") for k in short_only)
    assert not any(k.startswith("LongEdge_") for k in short_only)


def test_symbols_do_not_mix():
    btc = make_symbol_entry(seed=10, scale=0.02)
    eth = make_symbol_entry(seed=11, scale=0.05)
    cache_btc = build_cache_for_symbol("BTCUSDT", btc, window=30)
    cache_eth = build_cache_for_symbol("ETHUSDT", eth, window=30)
    leg = LegState("long", 100.0, 10.0, 100.0)
    e_btc = edges_mod.compute_long_components(cache_btc, 0.0, leg.liq_z(), leg.notional)
    e_eth = edges_mod.compute_long_components(cache_eth, 0.0, leg.liq_z(), leg.notional)
    assert e_btc["LongEdge_Return"] != e_eth["LongEdge_Return"]
