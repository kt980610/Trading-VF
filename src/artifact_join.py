"""Leakage-safe join of daily artifacts onto minute-level rows.

Daily distribution/volume/news artifacts must NEVER be joined onto a minute row
of the SAME calendar day, because the daily artifact for day ``D`` is only fully
known after day ``D`` closes. A minute on day ``D`` may only use the artifact of
the most recent COMPLETED day strictly before ``D`` (i.e. ``D-1`` or earlier).
"""

from __future__ import annotations

import bisect
from datetime import date
from typing import Callable, Dict

import pandas as pd


def _to_date(value) -> date:
    return pd.Timestamp(value).tz_localize(None).date() if pd.Timestamp(value).tzinfo else pd.Timestamp(value).date()


def build_previous_day_provider(
    ctx_by_date: Dict[str, Dict[str, float]]
) -> Callable[[object], Dict[str, float]]:
    """Return ``provider(ts) -> features`` using the last COMPLETED day < ts.date.

    ``ctx_by_date`` maps ``"YYYY-MM-DD"`` to a feature dict. For a minute at date
    ``D`` the provider returns the entry for the greatest available date strictly
    less than ``D`` (empty dict if none), preventing same-day look-ahead.
    """
    sorted_dates = sorted(ctx_by_date.keys())
    parsed = [pd.Timestamp(d).date() for d in sorted_dates]

    def provider(ts) -> Dict[str, float]:
        d = pd.Timestamp(ts).date()
        # Largest index with parsed[idx] < d.
        idx = bisect.bisect_left(parsed, d) - 1
        if idx < 0:
            return {}
        return ctx_by_date[sorted_dates[idx]]

    return provider
