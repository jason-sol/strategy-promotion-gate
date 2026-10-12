"""Strategy contract: pure functions over prepared bar DataFrames.

Strategies never talk to the broker. The live engine and the backtester feed
them the same prepared DataFrames so behaviour is identical in both.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Action(Enum):
    LONG = "long"
    SHORT = "short"
    EXIT = "exit"
    HOLD = "hold"


@dataclass(frozen=True)
class Signal:
    action: Action
    reason: str = ""


@dataclass
class PositionState:
    symbol: str
    side: str  # "long" | "short"
    qty: float
    entry_price: float
    stop_price: float
    # Best favorable price since entry: max close for longs, min close for shorts.
    high_water_mark: float
    entry_fee: float = 0.0

    def update_high_water_mark(self, price: float) -> None:
        if self.side == "long":
            self.high_water_mark = max(self.high_water_mark, price)
        else:
            self.high_water_mark = min(self.high_water_mark, price)

    def pnl_at(self, price: float) -> float:
        gross = self.qty * (price - self.entry_price)
        return gross if self.side == "long" else -gross


class Strategy(ABC):
    @property
    @abstractmethod
    def min_bars(self) -> int:
        """Bars of history required before signals are valid."""

    @property
    def stop_atr_mult(self) -> float:
        """Hard-stop distance in ATRs. Strategies with an ATR trailing stop set
        this to the trail distance so the hard stop doesn't fire first and make
        the trail dead code. Sizing scales quantity to the stop distance, so the
        dollar risk per trade comes from RiskConfig (risk_per_trade times the
        bounded vol-target scalar) regardless of this value."""
        return 1.0

    @abstractmethod
    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of bars with indicator columns added (must include 'atr')."""

    @abstractmethod
    def signal(self, df: pd.DataFrame, i: int, position: PositionState | None) -> Signal:
        """Evaluate the prepared frame at integer index i given current position."""


HOLD = Signal(Action.HOLD)


def vol_gate_blocks_entry(row: pd.Series, percentile: float | None) -> bool:
    """True when the realized-vol percentile gate should suppress a NEW entry.

    Open (False) when the gate is off or vol_pct is NaN (warmup): a missing
    feature must never manufacture phantom suppression. Exits ignore the gate.
    """
    if percentile is None:
        return False
    vol_pct = row.get("vol_pct")
    return pd.notna(vol_pct) and vol_pct > percentile


def volume_gate_blocks_entry(row: pd.Series, mult: float | None) -> bool:
    """True when volume confirmation should suppress a NEW entry.

    Open (False) when the gate is off, or when either the bar's volume or its
    trailing average is NaN: a missing feature must never manufacture phantom
    suppression. Exits ignore the gate.
    """
    if mult is None:
        return False
    avg = row.get("volume_avg")
    volume = row.get("volume")
    return pd.notna(avg) and pd.notna(volume) and volume < mult * avg


def trailing_stop_level(position: PositionState, atr: float, mult: float) -> float:
    if position.side == "long":
        return position.high_water_mark - mult * atr
    return position.high_water_mark + mult * atr


def trailing_stop_hit(position: PositionState, close: float, atr: float, mult: float) -> bool:
    level = trailing_stop_level(position, atr, mult)
    if position.side == "long":
        return close <= level
    return close >= level


def effective_stop(position: PositionState, atr: float, mult: float) -> float:
    """Level that will actually trigger an exit: the tighter-binding of the
    persisted entry stop and the ATR trail."""
    trail = trailing_stop_level(position, atr, mult)
    if position.side == "long":
        return max(position.stop_price, trail)
    return min(position.stop_price, trail)


def resolve_stop_and_hwm(position: PositionState, meta: dict | None) -> tuple[float, float]:
    """Stop and high-water mark for a broker position: persisted state if we
    have it, else a stop 1% from entry (refined on the next bar)."""
    if meta is not None:
        return meta["stop_price"], meta["high_water_mark"]
    if position.side == "long":
        return position.entry_price * 0.99, position.entry_price
    return position.entry_price * 1.01, position.entry_price
