"""Tests for halving-season one-hot features (spec section 6, tests 10-12)."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.halving_season import (
    SEASON_ONE_HOT,
    SEASON_POST,
    SEASON_PRE,
    SEASON_UNKNOWN,
    parse_halving_dates,
    season_features,
    select_season,
)


def test_one_hot_sums_to_one_everywhere():
    halvings = parse_halving_dates()
    for ts in ["2013-01-01", "2018-06-01", "2021-01-01", "2024-06-01", "2027-06-01", "1990-01-01"]:
        feats = season_features("BTCUSDT", ts, halvings)
        total = sum(feats[name] for name in SEASON_ONE_HOT)
        assert total == 1.0, (ts, feats)


def test_unknown_when_no_window():
    # Far from any halving window (>2y before first halving).
    label, cycle = select_season("BTCUSDT", "2000-01-01", parse_halving_dates())
    assert label == SEASON_UNKNOWN
    assert cycle == -1


def test_overlap_resolves_to_single_season_nearest_halving():
    # Construct two halvings 3 years apart so a midpoint is inside BOTH the
    # earlier post-window and the later pre-window; nearest one must win.
    halvings = parse_halving_dates(["2020-01-01T00:00:00Z", "2023-01-01T00:00:00Z"])
    # 100 days after the 2020 halving: inside 2020 post (dist 100d) and inside
    # 2023 pre (dist ~995d). Exactly ONE season, and it is post (nearest).
    feats = season_features("BTCUSDT", "2020-04-10", halvings)
    assert sum(feats[n] for n in SEASON_ONE_HOT) == 1.0
    assert feats[SEASON_POST] == 1.0
    assert feats[SEASON_PRE] == 0.0


def test_nearest_halving_is_selected():
    halvings = parse_halving_dates(["2020-01-01T00:00:00Z", "2023-01-01T00:00:00Z"])
    # 100 days before 2023 halving -> inside 2023 pre (dist 100d) and 2020 post
    # (dist ~995d). Nearest is the 2023 pre window.
    label, cycle = select_season("BTCUSDT", "2022-09-23", halvings)
    assert label == SEASON_PRE
    assert cycle == 1


def _equidistant_setup():
    # Two halvings ~1000 days apart (< 2*WINDOW) so the midpoint is strictly
    # inside BOTH the earlier post-window and the later pre-window, and is exactly
    # equidistant from each halving -> forces the SHA-256 tie-break.
    halvings = parse_halving_dates(["2020-01-01T00:00:00Z", "2022-09-27T00:00:00Z"])
    mid = halvings[0] + (halvings[1] - halvings[0]) / 2
    return halvings, mid


def test_equal_distance_same_seed_is_reproducible():
    halvings, mid = _equidistant_setup()
    a = select_season("BTCUSDT", mid, halvings, season_seed=7)
    b = select_season("BTCUSDT", mid, halvings, season_seed=7)
    assert a == b


def test_equal_distance_different_seed_can_differ():
    halvings, mid = _equidistant_setup()
    # Sweep seeds; for an equidistant point both pre/post are valid and the
    # SHA-256 pick must not be constant across all seeds.
    picks = {select_season("BTCUSDT", mid, halvings, season_seed=s)[0] for s in range(40)}
    assert len(picks) >= 2


def test_features_have_cycle_id():
    feats = season_features("BTCUSDT", "2024-06-01", parse_halving_dates())
    assert "halving_cycle_id" in feats
