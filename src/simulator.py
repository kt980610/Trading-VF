"""Minute-level strategy simulation and RF training-data generation (section 21).

The simulator manages a hedged position (long + short leg), checks liquidation
every minute using candle high/low, computes the edge features from the prebuilt
integral cache, asks a policy (RF predictor or baseline) whether to close or
continue, and -- once a single leg remains -- consults the MDP add-margin engine.

No lookahead: features and decisions at minute ``t`` only use data up to ``t``;
decisions take effect going forward. Per-symbol simulation is independent and
deterministic, so a parallel run reproduces a sequential run exactly. Forward
information is used ONLY to build supervised labels in ``generate_training_rows``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from . import edges as edges_mod
from . import mdp as mdp_mod
from .halving_season import season_features
from .integral_cache import IntegralCache
from .mdp_features import compute_mdp_features
from .pricing import LegState
from .rf_dataset import SIDE_ENCODING, MODE_ENCODING, make_row
from .trade_labeling import MinuteDecision, label_trade


@dataclass
class AccountConfig:
    initial_balance: float = 10000.0
    leverage: float = 10.0
    notional_per_leg: float = 1000.0
    reopen: bool = True


@dataclass
class PositionState:
    long_leg: Optional[LegState] = None
    short_leg: Optional[LegState] = None
    mode: str = edges_mod.MODE_HEDGED
    remaining_balance: float = 0.0
    total_added_margin_long: float = 0.0
    total_added_margin_short: float = 0.0
    first_liq_price: Optional[float] = None
    open: bool = False
    entry_index: int = -1

    def active_long(self) -> bool:
        return self.long_leg is not None

    def active_short(self) -> bool:
        return self.short_leg is not None

    def n_open(self) -> float:
        n = 0.0
        if self.long_leg is not None:
            n += self.long_leg.notional
        if self.short_leg is not None:
            n += self.short_leg.notional
        return n

    def margin_open(self) -> float:
        m = 0.0
        if self.long_leg is not None:
            m += self.long_leg.margin_current
        if self.short_leg is not None:
            m += self.short_leg.margin_current
        return m


def open_position(account_balance: float, entry_price: float, account: AccountConfig) -> PositionState:
    """Open a fresh hedged position, deducting both legs' margin from balance."""
    notional = account.notional_per_leg
    margin = notional / account.leverage
    qty = notional / entry_price

    long_leg = LegState(side="long", entry_price=entry_price, qty=qty, margin_current=margin)
    short_leg = LegState(side="short", entry_price=entry_price, qty=qty, margin_current=margin)

    return PositionState(
        long_leg=long_leg,
        short_leg=short_leg,
        mode=edges_mod.MODE_HEDGED,
        remaining_balance=account_balance - 2.0 * margin,
        open=True,
    )


def _state_features(
    state: PositionState,
    current_price: float,
) -> Dict[str, float]:
    """Assemble the section-11 state features for the active position."""
    long_active = state.active_long()
    short_active = state.active_short()

    if long_active and not short_active:
        side = "long"
        leg = state.long_leg
    elif short_active and not long_active:
        side = "short"
        leg = state.short_leg
    else:
        side = "both"
        leg = state.long_leg if long_active else state.short_leg

    entry_price = leg.entry_price if leg is not None else float("nan")
    y = leg.current_return(current_price) if leg is not None else 0.0
    current_pnl = 0.0
    if long_active:
        current_pnl += state.long_leg.current_pnl(current_price)
    if short_active:
        current_pnl += state.short_leg.current_pnl(current_price)

    liq_levels = []
    if long_active:
        liq_levels.append(state.long_leg.liquidation_price_level())
    if short_active:
        liq_levels.append(state.short_leg.liquidation_price_level())
    if liq_levels:
        nearest_liq = min(liq_levels, key=lambda p: abs(current_price - p))
        distance_to_liq = abs(current_price - nearest_liq)
    else:
        distance_to_liq = 0.0

    distance_to_first_liq = (
        abs(current_price - state.first_liq_price) if state.first_liq_price is not None else 0.0
    )
    liquidation_cutoff = leg.liq_z() if leg is not None else 0.0

    return {
        "side": side,
        "mode": state.mode,
        "CurrentPrice": float(current_price),
        "EntryPrice": float(entry_price),
        "y": float(y),
        "current_pnl": float(current_pnl),
        "distance_to_liq": float(distance_to_liq),
        "distance_to_liq_pct": float(distance_to_liq / current_price) if current_price else 0.0,
        "distance_to_first_liq": float(distance_to_first_liq),
        "remaining_balance": float(state.remaining_balance),
        "N_open": float(state.n_open()),
        "M_open_current": float(state.margin_open()),
        "liquidation_cutoff": float(liquidation_cutoff),
        "hour_of_day": 0.0,
        "day_of_week": 0.0,
    }


def _time_features(ts) -> Dict[str, float]:
    ts = pd.Timestamp(ts)
    return {"hour_of_day": float(ts.hour), "day_of_week": float(ts.dayofweek)}


def build_feature_row(
    cache: IntegralCache,
    state: PositionState,
    current_price: float,
    timestamp,
    context: Optional[Dict[str, float]] = None,
    include_optional: bool = False,
) -> Dict[str, float]:
    """Edge + state + external context features for one decision point."""
    features = edges_mod.compute_features(
        cache,
        state.mode,
        current_price,
        long_leg=state.long_leg,
        short_leg=state.short_leg,
        include_components=include_optional,
    )
    features.update(_state_features(state, current_price))
    features.update(_time_features(timestamp))
    if context:
        features.update(context)
    return features


def _check_liquidations(state: PositionState, high: float, low: float) -> Dict[str, bool]:
    """Detect intraday liquidations from candle extremes (spec section 21)."""
    long_liq = False
    short_liq = False
    if state.active_long():
        if low <= state.long_leg.liquidation_price_level():
            long_liq = True
    if state.active_short():
        if high >= state.short_leg.liquidation_price_level():
            short_liq = True
    return {"long": long_liq, "short": short_liq}


def _realize_leg(leg: LegState, price: float) -> float:
    return leg.notional * leg.payoff_pct(leg.current_return(price))


def simulate_symbol(
    minute_df: pd.DataFrame,
    cache: IntegralCache,
    policy,
    symbol: str,
    account: AccountConfig,
    fee_rate: float = 0.0004,
    funding_rate: float = 0.0,
    slippage: float = 0.0,
    use_mdp: bool = True,
    add_margin_step: float = 10.0,
    max_add_margin_per_decision: float = 1e18,
    max_total_added_margin: float = 1e18,
    context_provider: Optional[Callable[[object], Dict[str, float]]] = None,
    include_optional: bool = False,
) -> pd.DataFrame:
    """Run a minute-level simulation; returns a per-day results frame."""
    df = minute_df.reset_index(drop=True)
    n = len(df)

    balance = account.initial_balance
    state: Optional[PositionState] = None

    daily: Dict[str, Dict[str, float]] = {}

    def day_key(ts) -> str:
        return str(pd.Timestamp(ts).date())

    def bump(ts, **deltas):
        key = day_key(ts)
        row = daily.setdefault(
            key,
            {
                "date": key,
                "symbol": symbol,
                "realized_pnl": 0.0,
                "fees": 0.0,
                "funding": 0.0,
                "slippage": 0.0,
                "close_count": 0,
                "continue_count": 0,
                "liquidation_count": 0,
                "double_liquidation_count": 0,
                "add_margin_count": 0,
                "hold_minutes": 0,
            },
        )
        for k, v in deltas.items():
            row[k] += v

    def close_all(ts, price, reason: str):
        nonlocal balance, state
        realized = 0.0
        if state.active_long():
            realized += _realize_leg(state.long_leg, price)
        if state.active_short():
            realized += _realize_leg(state.short_leg, price)
        # Return committed margin plus realized pnl to the free balance.
        balance = state.remaining_balance + state.margin_open() + realized
        fees = fee_rate * state.n_open()
        slip = slippage * state.n_open()
        bump(ts, realized_pnl=realized, fees=fees, slippage=slip)
        state.open = False

    for i in range(n):
        row = df.iloc[i]
        ts = row["timestamp"]
        close_price = float(row["close"])
        high = float(row.get("high", close_price))
        low = float(row.get("low", close_price))

        if state is None or not state.open:
            if not account.reopen and state is not None:
                break
            if balance <= 0:
                break
            state = open_position(balance, close_price, account)
            state.entry_index = i
            continue

        bump(ts, hold_minutes=1)

        # 1-2. Liquidation check on candle extremes.
        liqs = _check_liquidations(state, high, low)
        if liqs["long"] and liqs["short"]:
            # Double liquidation: both legs lost.
            long_loss = _realize_leg(state.long_leg, state.long_leg.liquidation_price_level())
            short_loss = _realize_leg(state.short_leg, state.short_leg.liquidation_price_level())
            bump(ts, realized_pnl=long_loss + short_loss, liquidation_count=2, double_liquidation_count=1)
            balance = state.remaining_balance
            state.open = False
            continue
        if liqs["long"]:
            loss = _realize_leg(state.long_leg, state.long_leg.liquidation_price_level())
            bump(ts, realized_pnl=loss, liquidation_count=1)
            state.first_liq_price = state.long_leg.liquidation_price_level()
            state.long_leg = None
            state.mode = edges_mod.MODE_SHORT_ONLY
            if not state.active_short():
                balance = state.remaining_balance
                state.open = False
                continue
        if liqs["short"]:
            loss = _realize_leg(state.short_leg, state.short_leg.liquidation_price_level())
            bump(ts, realized_pnl=loss, liquidation_count=1)
            state.first_liq_price = state.short_leg.liquidation_price_level()
            state.short_leg = None
            state.mode = edges_mod.MODE_LONG_ONLY
            if not state.active_long():
                balance = state.remaining_balance
                state.open = False
                continue

        single_leg = state.active_long() ^ state.active_short()

        # MDP only after a single leg remains and only when triggered.
        if use_mdp and single_leg:
            leg = state.long_leg if state.active_long() else state.short_leg
            decision = mdp_mod.decide(
                cache,
                leg,
                close_price,
                state.remaining_balance,
                state.first_liq_price if state.first_liq_price is not None else leg.liquidation_price_level(),
                add_margin_step=add_margin_step,
                max_add_margin_per_decision=max_add_margin_per_decision,
                max_total_added_margin=max_total_added_margin,
                current_total_added_margin=(
                    state.total_added_margin_long if state.active_long() else state.total_added_margin_short
                ),
            )
            if decision.triggered:
                if decision.action == mdp_mod.ACTION_CLOSE:
                    bump(ts, close_count=1)
                    close_all(ts, close_price, "mdp_close")
                else:
                    leg.margin_current += decision.x_best
                    state.remaining_balance -= decision.x_best
                    if state.active_long():
                        state.total_added_margin_long += decision.x_best
                    else:
                        state.total_added_margin_short += decision.x_best
                    bump(ts, add_margin_count=1, continue_count=1)
                continue

        # Normal RF/baseline close/continue decision.
        features = build_feature_row(
            cache,
            state,
            close_price,
            ts,
            context=context_provider(ts) if context_provider else None,
            include_optional=include_optional,
        )
        frame = pd.DataFrame([features])
        decision = policy.decide(frame)[0]
        if decision == "CLOSE":
            bump(ts, close_count=1)
            close_all(ts, close_price, "policy_close")
        else:
            bump(ts, continue_count=1)

    if state is not None and state.open:
        last = df.iloc[n - 1]
        close_all(last["timestamp"], float(last["close"]), "end_of_data")

    result = pd.DataFrame(list(daily.values()))
    if result.empty:
        return result
    result["daily_pnl"] = (
        result["realized_pnl"] - result["fees"] + result["funding"] - result["slippage"]
    )
    return result.sort_values("date").reset_index(drop=True)


def generate_training_rows(
    minute_df: pd.DataFrame,
    cache: IntegralCache,
    symbol: str,
    account: AccountConfig,
    context_provider: Optional[Callable[[object], Dict[str, float]]] = None,
    include_optional: bool = False,
) -> List[Dict[str, object]]:
    """Generate RF dataset rows under an always-hold rollout.

    For each minute in a held episode the label is::

        target = episode_final_pnl - close_now_pnl

    which is the realized improvement of continuing over closing immediately.
    Forward information is used for the label only, never for features.
    """
    df = minute_df.reset_index(drop=True)
    n = len(df)
    rows: List[Dict[str, object]] = []
    balance = account.initial_balance
    i = 0

    while i < n:
        if balance <= 0:
            break
        entry_price = float(df.iloc[i]["close"])
        state = open_position(balance, entry_price, account)

        # Roll the episode forward holding until both legs are gone or data ends.
        episode: List[Dict[str, object]] = []
        j = i + 1
        episode_final_pnl = 0.0
        realized_so_far = 0.0
        while j < n and state.open:
            row = df.iloc[j]
            close_price = float(row["close"])
            high = float(row.get("high", close_price))
            low = float(row.get("low", close_price))

            liqs = _check_liquidations(state, high, low)
            if liqs["long"] and state.active_long():
                realized_so_far += _realize_leg(state.long_leg, state.long_leg.liquidation_price_level())
                state.first_liq_price = state.long_leg.liquidation_price_level()
                state.long_leg = None
                state.mode = edges_mod.MODE_SHORT_ONLY
            if liqs["short"] and state.active_short():
                realized_so_far += _realize_leg(state.short_leg, state.short_leg.liquidation_price_level())
                state.first_liq_price = state.short_leg.liquidation_price_level()
                state.short_leg = None
                state.mode = edges_mod.MODE_LONG_ONLY
            if not state.active_long() and not state.active_short():
                state.open = False
                break

            features = build_feature_row(
                cache,
                state,
                close_price,
                row["timestamp"],
                context=context_provider(row["timestamp"]) if context_provider else None,
                include_optional=include_optional,
            )
            close_now = realized_so_far
            if state.active_long():
                close_now += state.long_leg.current_pnl(close_price)
            if state.active_short():
                close_now += state.short_leg.current_pnl(close_price)
            episode.append(
                {"features": features, "timestamp": row["timestamp"], "close_now": close_now}
            )
            j += 1

        # Episode resolution: realize whatever remains at the last seen price.
        last_price = float(df.iloc[min(j, n - 1)]["close"])
        if state.active_long():
            realized_so_far += _realize_leg(state.long_leg, last_price)
        if state.active_short():
            realized_so_far += _realize_leg(state.short_leg, last_price)
        episode_final_pnl = realized_so_far

        for item in episode:
            rows.append(
                make_row(
                    symbol=symbol,
                    timestamp=item["timestamp"],
                    features=item["features"],
                    pnl_if_continue=episode_final_pnl,
                    pnl_if_close=item["close_now"],
                )
            )

        balance += episode_final_pnl
        i = max(j, i + 1)

    return rows


def _volume_ratio(numerator: float, denominator: float) -> float:
    d = float(denominator)
    if d <= 0 or not np.isfinite(d):
        return 0.0
    return float(numerator) / d


def generate_trade_dataset(
    minute_df: pd.DataFrame,
    cache: IntegralCache,
    symbol: str,
    account: AccountConfig,
    halvings=None,
    season_seed: int = 0,
    context_provider: Optional[Callable[[object], Dict[str, float]]] = None,
    fee_rate: float = 0.0004,
    funding_rate: float = 0.0,
    slippage: float = 0.0,
    close_epsilon: float = 0.0,
    include_optional: bool = False,
) -> pd.DataFrame:
    """Trade-grouped, minute-level RF *close classifier* dataset (spec sections 8/11).

    Each held episode (open -> full close/liquidation) is one trade with a shared
    ``trade_id``. No ADD_MARGIN is taken; the MDP is used only to emit per-minute
    integral/risk features. Net PnL includes fees, funding and slippage. Binary
    ``close_label`` and ``sample_weight`` come from :mod:`trade_labeling`.
    """
    df = minute_df.reset_index(drop=True)
    n = len(df)
    if n == 0:
        return pd.DataFrame()

    day = df["timestamp"].dt.floor("1D")
    cum_vol = df.assign(__day=day).groupby("__day")["volume"].cumsum().to_numpy()

    rows: List[Dict[str, object]] = []
    balance = account.initial_balance
    trade_id = 0
    i = 0

    while i < n:
        if balance <= 0:
            break
        episode, decisions, j, realized_so_far, state = _simulate_episode(
            df, n, cum_vol, i, balance, cache, symbol, account,
            halvings=halvings, season_seed=season_seed, context_provider=context_provider,
            fee_rate=fee_rate, funding_rate=funding_rate, slippage=slippage,
            include_optional=include_optional,
        )
        if decisions:
            _emit_episode_rows(rows, episode, decisions, trade_id, df, i, close_epsilon, symbol)

        # Resolve episode PnL at last seen price for balance bookkeeping.
        last_price = float(df.iloc[min(j, n - 1)]["close"])
        if state.active_long():
            realized_so_far += _realize_leg(state.long_leg, last_price)
        if state.active_short():
            realized_so_far += _realize_leg(state.short_leg, last_price)
        balance += realized_so_far
        trade_id += 1
        i = max(j, i + 1)

    return pd.DataFrame(rows)


def _simulate_episode(
    df, n, cum_vol, i, balance, cache, symbol, account, *,
    halvings=None, season_seed=0, context_provider=None,
    fee_rate=0.0004, funding_rate=0.0, slippage=0.0, include_optional=False,
    end_index=None,
):
    """Simulate ONE hedged episode opened at row ``i`` forward until full
    close/liquidation or ``end_index`` (exclusive, defaults to ``n``).

    Returns ``(episode, decisions, j, realized_so_far, state)``. ``end_index``
    lets callers stop before a data discontinuity so an episode never spans a gap.
    """
    stop = n if end_index is None else min(int(end_index), n)
    entry_price = float(df.iloc[i]["close"])
    state = open_position(balance, entry_price, account)

    episode: List[Dict[str, object]] = []
    decisions: List[MinuteDecision] = []
    realized_so_far = 0.0
    j = i + 1
    while j < stop and state.open:
        row = df.iloc[j]
        ts = row["timestamp"]
        close_price = float(row["close"])
        high = float(row.get("high", close_price))
        low = float(row.get("low", close_price))

        liqs = _check_liquidations(state, high, low)
        long_liq_now = bool(liqs["long"]) and state.active_long()
        if long_liq_now:
            realized_so_far += _realize_leg(state.long_leg, state.long_leg.liquidation_price_level())
            state.first_liq_price = state.long_leg.liquidation_price_level()
            state.long_leg = None
            state.mode = edges_mod.MODE_SHORT_ONLY
        short_liq_now = bool(liqs["short"]) and state.active_short()
        if short_liq_now:
            realized_so_far += _realize_leg(state.short_leg, state.short_leg.liquidation_price_level())
            state.first_liq_price = state.short_leg.liquidation_price_level()
            state.short_leg = None
            state.mode = edges_mod.MODE_LONG_ONLY
        leg_liquidated_now = 1 if (long_liq_now or short_liq_now) else 0
        fully_closed = not state.active_long() and not state.active_short()
        fully_liquidated_now = 1 if (fully_closed and leg_liquidated_now) else 0

        ctx = context_provider(ts) if context_provider else {}
        features = build_feature_row(
            cache, state, close_price, ts, context=ctx, include_optional=include_optional
        )
        features.update(season_features(symbol, ts, halvings, season_seed=season_seed))
        features.update(
            compute_mdp_features(cache, state.long_leg, state.short_leg, close_price)
        )
        current_minute_volume = float(row.get("volume", 0.0) or 0.0)
        intraday_volume = float(cum_vol[j])
        predicted = float(ctx.get("predicted_daily_volume", 0.0) or 0.0)
        prev_day = float(ctx.get("previous_day_real_volume", 0.0) or 0.0)
        features["current_minute_volume"] = current_minute_volume
        features["intraday_volume_so_far"] = intraday_volume
        features["current_volume_to_predicted_daily_volume"] = _volume_ratio(intraday_volume, predicted)
        features["current_volume_to_previous_day_daily_volume"] = _volume_ratio(intraday_volume, prev_day)

        n_open = state.n_open()
        elapsed = max(1, j - i + 1)
        gross_close_now = realized_so_far
        if state.active_long():
            gross_close_now += state.long_leg.current_pnl(close_price)
        if state.active_short():
            gross_close_now += state.short_leg.current_pnl(close_price)
        close_costs = (fee_rate + slippage) * n_open + funding_rate * n_open * elapsed
        net_close_pnl_now = gross_close_now - close_costs

        side_label = features.get("side", "both")
        episode.append(
            {
                "features": dict(features),
                "timestamp": ts,
                "side_label": side_label,
                "net_close_pnl_now": float(net_close_pnl_now),
                "leg_liquidated_now": int(leg_liquidated_now),
                "fully_liquidated_now": int(fully_liquidated_now),
            }
        )
        decisions.append(MinuteDecision(index=j, net_close_pnl_now=float(net_close_pnl_now)))

        if fully_closed:
            state.open = False
            break
        j += 1

    return episode, decisions, j, realized_so_far, state


def _downcast_floats_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 columns to float32 (lossless for tree training)."""
    f64 = [c for c in df.columns if str(df[c].dtype) == "float64"]
    if f64:
        df[f64] = df[f64].astype("float32")
    return df


class _ParquetSpiller:
    """Buffer per-minute row dicts and flush them to numbered Parquet parts.

    This keeps generation memory bounded: instead of accumulating millions of
    fat row dicts in RAM (~10 KB/row), we periodically write a compact float32
    Parquet part to ``base_path.partNNNN.parquet`` and free the buffer. The
    parts are read back (compactly) only when the full matrix is needed for fit.
    """

    def __init__(self, base_path: str, flush_rows: int = 250_000):
        self.base = base_path
        self.flush_rows = max(1, int(flush_rows))
        self.buf: List[Dict[str, object]] = []
        self.parts: List[str] = []
        self.total_rows = 0
        os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)

    def add(self, rows: List[Dict[str, object]]):
        if not rows:
            return
        self.buf.extend(rows)
        if len(self.buf) >= self.flush_rows:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        chunk = pd.DataFrame(self.buf)
        _downcast_floats_inplace(chunk)
        path = f"{self.base}.part{len(self.parts):04d}.parquet"
        try:
            chunk.to_parquet(path, index=False)
        except Exception:
            path = path.replace(".parquet", ".csv")
            chunk.to_csv(path, index=False)
        self.parts.append(path)
        self.total_rows += len(chunk)
        self.buf = []
        del chunk

    def read_back(self) -> pd.DataFrame:
        """Concatenate all spilled parts into one compact DataFrame, then
        delete the part files. Returns an empty frame if nothing was written."""
        self.flush()
        if not self.parts:
            return pd.DataFrame()
        frames = []
        for p in self.parts:
            frames.append(pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p))
        out = pd.concat(frames, ignore_index=True)
        del frames
        for p in self.parts:
            try:
                os.remove(p)
            except OSError:
                pass
        self.parts = []
        return out


def _emit_episode_rows(rows, episode, decisions, trade_id, df, i, close_epsilon, symbol):
    """Label one episode and append its per-minute rows to ``rows``."""
    labeled = label_trade(decisions, entry_index=i, close_epsilon=close_epsilon)
    entry_ts = df.iloc[i]["timestamp"]
    n_ep = len(episode)
    for k, (item, lab) in enumerate(zip(episode, labeled)):
        feats = item["features"]
        side_label = item["side_label"]
        row_out: Dict[str, object] = dict(feats)
        row_out["side_code"] = int(SIDE_ENCODING.get(side_label, -1))
        row_out["mode_code"] = int(MODE_ENCODING.get(feats.get("mode", ""), -1))
        row_out["symbol"] = symbol
        row_out["trade_id"] = trade_id
        row_out["leg_id"] = f"{trade_id}:{side_label}"
        row_out["side_label"] = side_label
        row_out["entry_timestamp"] = entry_ts
        row_out["timestamp"] = item["timestamp"]
        row_out["close_label"] = int(lab.close_label)
        row_out["sample_weight"] = float(lab.sample_weight)
        row_out["trade_pnl_rate_now"] = float(lab.trade_pnl_rate_now)
        row_out["best_future_trade_pnl_rate"] = float(lab.best_future_trade_pnl_rate)
        row_out["decision_value_gap"] = float(lab.decision_value_gap)
        # Realization columns for leakage-free, per-trade evaluation (net of
        # fees/funding/slippage; no double-counting). The terminal minute is the
        # forced close/liquidation used by the baseline.
        row_out["net_close_pnl_now"] = float(item["net_close_pnl_now"])
        row_out["leg_liquidated_now"] = int(item["leg_liquidated_now"])
        row_out["fully_liquidated_now"] = int(item["fully_liquidated_now"])
        row_out["is_terminal"] = int(k == n_ep - 1)
        rows.append(row_out)


def _contiguous_segment_bounds(df, max_gap_seconds: float = 86400.0):
    """Return sorted segment boundaries [b0=0, b1, ..., n] where a new segment
    starts whenever consecutive timestamps differ by more than ``max_gap_seconds``.

    Used so an overlapping episode never simulates ACROSS a data discontinuity
    (e.g. the multi-year gap between the 2018 spot block and the 2022+ block).
    """
    n = len(df)
    if n <= 1:
        return [0, n]
    ts_idx = pd.DatetimeIndex(df["timestamp"])
    if ts_idx.tz is not None:
        ts_idx = ts_idx.tz_convert("UTC").tz_localize(None)
    ts_ns = ts_idx.values.astype("datetime64[ns]").astype("int64")
    gap_ns = max_gap_seconds * 1e9
    bounds = [0]
    for k in range(1, n):
        if (ts_ns[k] - ts_ns[k - 1]) > gap_ns:
            bounds.append(k)
    bounds.append(n)
    return bounds


def generate_overlapping_trade_dataset(
    minute_df: pd.DataFrame,
    cache: IntegralCache,
    symbol: str,
    account: AccountConfig,
    entry_indices,
    halvings=None,
    season_seed: int = 0,
    context_provider: Optional[Callable[[object], Dict[str, float]]] = None,
    fee_rate: float = 0.0004,
    funding_rate: float = 0.0,
    slippage: float = 0.0,
    close_epsilon: float = 0.0,
    include_optional: bool = False,
    max_end_index: Optional[int] = None,
    max_hold_minutes: Optional[int] = None,
    spill_path: Optional[str] = None,
    flush_rows: int = 250_000,
) -> pd.DataFrame:
    """Open ONE independent hedged episode at each ``entry_indices`` row and
    simulate it forward (within its contiguous data segment, and never past
    ``max_end_index``) until full close/liquidation or that boundary.

    ``max_hold_minutes`` caps how long any single episode may stay open (the
    liquidation/full-close still ends it earlier when it happens). This bounds
    memory/compute for overlapping episodes that would otherwise run to the
    window end: full close requires BOTH legs to liquidate, which is rare, so
    most episodes would otherwise hold to the boundary and explode row counts.

    Unlike :func:`generate_trade_dataset` (sequential, non-overlapping, with
    balance threading), episodes here OVERLAP in time and each starts from a fresh
    ``account.initial_balance``, so every entry yields its own trade group. This is
    used for windowed training, where one entry per day produces enough trade
    groups to train the close classifier. Episodes never cross a data gap.
    """
    import bisect

    df = minute_df.reset_index(drop=True)
    n = len(df)
    if n == 0:
        return pd.DataFrame()

    day = df["timestamp"].dt.floor("1D")
    cum_vol = df.assign(__day=day).groupby("__day")["volume"].cumsum().to_numpy()
    bounds = _contiguous_segment_bounds(df)
    hard_end = n if max_end_index is None else min(int(max_end_index), n)

    spiller = _ParquetSpiller(spill_path, flush_rows) if spill_path else None
    rows: List[Dict[str, object]] = []
    trade_id = 0
    for raw_i in entry_indices:
        i = int(raw_i)
        if i < 0 or i >= n:
            continue
        seg = bisect.bisect_right(bounds, i) - 1
        seg_end = bounds[seg + 1] if 0 <= seg < len(bounds) - 1 else n
        seg_end = min(seg_end, hard_end)
        if max_hold_minutes is not None:
            seg_end = min(seg_end, i + 1 + int(max_hold_minutes))
        episode, decisions, _j, _realized, _state = _simulate_episode(
            df, n, cum_vol, i, account.initial_balance, cache, symbol, account,
            halvings=halvings, season_seed=season_seed, context_provider=context_provider,
            fee_rate=fee_rate, funding_rate=funding_rate, slippage=slippage,
            include_optional=include_optional, end_index=seg_end,
        )
        if decisions:
            if spiller is not None:
                # Spill this episode's rows to disk so RAM stays bounded; the
                # full (compact float32) matrix is reassembled only at the end.
                ep_rows: List[Dict[str, object]] = []
                _emit_episode_rows(ep_rows, episode, decisions, trade_id, df, i, close_epsilon, symbol)
                spiller.add(ep_rows)
            else:
                _emit_episode_rows(rows, episode, decisions, trade_id, df, i, close_epsilon, symbol)
        trade_id += 1

    if spiller is not None:
        return spiller.read_back()
    return pd.DataFrame(rows)
