"""Price, payoff, notional, margin and liquidation-cutoff math.

Implements spec sections 2-4. The key invariants enforced here:

* ``ScenarioPrice(z) = EntryPrice * (1 + z)`` (NEVER CurrentPrice/FinalPrice).
* Long payoff = ``z``; short payoff = ``-z``.
* PnL is sized by the *open notional* ``N`` only. Adding margin changes margin
  and liquidation cutoff, never ``N``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def scenario_price(entry_price: float, z: float) -> float:
    """Final-scenario price for return ``z`` relative to the entry price."""
    return float(entry_price) * (1.0 + float(z))


def y_long(current_price: float, entry_price_long: float) -> float:
    return float(current_price) / float(entry_price_long) - 1.0


def y_short(current_price: float, entry_price_short: float) -> float:
    return float(current_price) / float(entry_price_short) - 1.0


def long_payoff_pct(z: float) -> float:
    return float(z)


def short_payoff_pct(z: float) -> float:
    return -float(z)


def notional(qty: float, entry_price: float) -> float:
    return float(qty) * float(entry_price)


def risk_leverage(notional_value: float, margin_current: float) -> float:
    if margin_current == 0:
        return float("inf")
    return float(notional_value) / float(margin_current)


def long_liq_z(
    entry_price_long: float,
    notional_long: float,
    margin_long_current: float,
    liquidation_price_long: Optional[float] = None,
) -> float:
    """Long liquidation cutoff in ``z`` units.

    Uses the real Binance liquidation price when available; otherwise the
    approximation ``-M_long_current / N_long``.
    """
    if liquidation_price_long is not None:
        return float(liquidation_price_long) / float(entry_price_long) - 1.0
    return -float(margin_long_current) / float(notional_long)


def short_liq_z(
    entry_price_short: float,
    notional_short: float,
    margin_short_current: float,
    liquidation_price_short: Optional[float] = None,
) -> float:
    """Short liquidation cutoff in ``z`` units.

    Real liquidation price when available; otherwise approximation
    ``M_short_current / N_short``.
    """
    if liquidation_price_short is not None:
        return float(liquidation_price_short) / float(entry_price_short) - 1.0
    return float(margin_short_current) / float(notional_short)


def liquidation_price_from_cutoff(entry_price: float, liq_z: float) -> float:
    """Inverse of the cutoff: price level that triggers liquidation."""
    return float(entry_price) * (1.0 + float(liq_z))


def add_margin(margin_current: float, x: float) -> float:
    """M_X = M_current + X. Notional is intentionally NOT touched here."""
    return float(margin_current) + float(x)


@dataclass
class LegState:
    """State of a single position leg. ``notional`` is fixed after open."""

    side: str  # "long" or "short"
    entry_price: float
    qty: float
    margin_current: float
    liquidation_price: Optional[float] = None

    @property
    def notional(self) -> float:
        return notional(self.qty, self.entry_price)

    def current_return(self, current_price: float) -> float:
        return float(current_price) / float(self.entry_price) - 1.0

    def payoff_pct(self, z: float) -> float:
        return long_payoff_pct(z) if self.side == "long" else short_payoff_pct(z)

    def liq_z(self) -> float:
        if self.side == "long":
            return long_liq_z(
                self.entry_price, self.notional, self.margin_current, self.liquidation_price
            )
        return short_liq_z(
            self.entry_price, self.notional, self.margin_current, self.liquidation_price
        )

    def liq_z_after_add(self, x: float) -> float:
        """Liquidation cutoff after adding margin ``x`` (notional unchanged)."""
        m_x = add_margin(self.margin_current, x)
        if self.side == "long":
            return long_liq_z(self.entry_price, self.notional, m_x, None)
        return short_liq_z(self.entry_price, self.notional, m_x, None)

    def current_pnl(self, current_price: float) -> float:
        """Realized-if-closed-now PnL = N_open * payoff_pct(y)."""
        y = self.current_return(current_price)
        return self.notional * self.payoff_pct(y)

    def liquidation_price_level(self) -> float:
        if self.liquidation_price is not None:
            return float(self.liquidation_price)
        return liquidation_price_from_cutoff(self.entry_price, self.liq_z())
