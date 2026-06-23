"""Bitcoin halving-season one-hot features with deterministic tie-breaking (spec 6).

Each timestamp is mapped to exactly one season one-hot:

    season_pre_halving_2y + season_post_halving_2y + season_unknown == 1

Because halving cycles are not exactly four years, a timestamp can fall inside
BOTH the previous halving's ``post_halving_2y`` window AND the next halving's
``pre_halving_2y`` window. Selection rules:

1. Gather every pre/post candidate window the timestamp falls into.
2. One candidate  -> pick it.
3. Many candidates -> pick the one whose halving is closest in absolute time.
4. Exact ties     -> NEVER use runtime randomness or Python's process-salted
   ``hash()``. Use a reproducible SHA-256 keyed on
   ``symbol|timestamp|global_season_seed`` so the SAME tuple always resolves to
   the SAME season across train/val/test/backtest/live.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pandas as pd

WINDOW_DAYS = 730  # 2 years
SEASON_PRE = "season_pre_halving_2y"
SEASON_POST = "season_post_halving_2y"
SEASON_UNKNOWN = "season_unknown"
SEASON_ONE_HOT = [SEASON_PRE, SEASON_POST, SEASON_UNKNOWN]

# Canonical Bitcoin halving timestamps (UTC). Future anchors let pre-windows be
# assigned for recent dates; override via config.halving.dates.
DEFAULT_HALVING_DATES_UTC: List[str] = [
    "2012-11-28T00:00:00Z",
    "2016-07-09T00:00:00Z",
    "2020-05-11T00:00:00Z",
    "2024-04-20T00:00:00Z",
    "2028-04-20T00:00:00Z",
]


def parse_halving_dates(dates: Optional[Sequence[str]] = None) -> List[pd.Timestamp]:
    raw = list(dates) if dates else list(DEFAULT_HALVING_DATES_UTC)
    out = [pd.Timestamp(d) for d in raw]
    out = [t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC") for t in out]
    return sorted(out)


@dataclass(frozen=True)
class _Candidate:
    kind: str  # "pre" or "post"
    halving_index: int
    halving_ts: pd.Timestamp
    distance: pd.Timedelta  # absolute time between timestamp and that halving


def _to_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _candidates(ts: pd.Timestamp, halvings: List[pd.Timestamp]) -> List[_Candidate]:
    window = pd.Timedelta(days=WINDOW_DAYS)
    out: List[_Candidate] = []
    for idx, h in enumerate(halvings):
        if h - window <= ts < h:  # pre-halving window
            out.append(_Candidate("pre", idx, h, h - ts))
        elif h <= ts < h + window:  # post-halving window
            out.append(_Candidate("post", idx, h, ts - h))
    return out


def _stable_pick(symbol: str, ts: pd.Timestamp, seed, tied: List[_Candidate]) -> _Candidate:
    """Deterministically choose among equidistant candidates (SHA-256, no hash())."""
    ordered = sorted(tied, key=lambda c: (c.halving_ts.value, c.kind))
    stable_key = f"{symbol}|{ts.isoformat()}|{seed}"
    digest = hashlib.sha256(stable_key.encode("utf-8")).digest()
    pick = int.from_bytes(digest[:8], "big") % len(ordered)
    return ordered[pick]


def select_season(
    symbol: str,
    timestamp,
    halvings: List[pd.Timestamp],
    season_seed=0,
) -> Tuple[str, int]:
    """Return ``(season_label, halving_cycle_id)`` for one timestamp.

    ``halving_cycle_id`` is the index of the chosen halving, or ``-1`` when no
    window applies (``season_unknown``).
    """
    ts = _to_utc(timestamp)
    cands = _candidates(ts, halvings)
    if not cands:
        return SEASON_UNKNOWN, -1
    if len(cands) == 1:
        chosen = cands[0]
    else:
        min_dist = min(c.distance for c in cands)
        tied = [c for c in cands if c.distance == min_dist]
        chosen = tied[0] if len(tied) == 1 else _stable_pick(symbol, ts, season_seed, tied)
    label = SEASON_PRE if chosen.kind == "pre" else SEASON_POST
    return label, int(chosen.halving_index)


def season_features(
    symbol: str,
    timestamp,
    halvings: Optional[List[pd.Timestamp]] = None,
    season_seed=0,
) -> dict:
    """One-hot season features for a single row (sums to exactly 1)."""
    halvings = halvings if halvings is not None else parse_halving_dates()
    label, cycle_id = select_season(symbol, timestamp, halvings, season_seed=season_seed)
    feats = {name: 0.0 for name in SEASON_ONE_HOT}
    feats[label] = 1.0
    feats["halving_cycle_id"] = float(cycle_id)
    return feats
