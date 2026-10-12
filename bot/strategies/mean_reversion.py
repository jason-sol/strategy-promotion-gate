"""Mean reversion on stddev bands: fade moves beyond entry_z sigma from the SMA."""

import pandas as pd

from bot.indicators import atr, rolling_std, sma
from bot.strategies.base import HOLD, Action, PositionState, Signal, Strategy


class MeanReversion(Strategy):
    def __init__(self, sma_period: int, entry_z: float, allow_short: bool, atr_period: int):
        self.sma_period = sma_period
        self.entry_z = entry_z
        self.allow_short = allow_short
        self.atr_period = atr_period

    @property
    def min_bars(self) -> int:
        return max(self.sma_period, self.atr_period) + 1

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        df["sma"] = sma(df["close"], self.sma_period)
        df["sd"] = rolling_std(df["close"], self.sma_period)
        df["atr"] = atr(df, self.atr_period)
        return df

    def signal(self, df: pd.DataFrame, i: int, position: PositionState | None) -> Signal:
        if i + 1 < self.min_bars:
            return HOLD
        row = df.iloc[i]
        if pd.isna(row["sma"]) or pd.isna(row["sd"]):
            return HOLD

        if position is not None:
            if position.side == "long" and row["close"] >= row["sma"]:
                return Signal(Action.EXIT, "reverted to mean")
            if position.side == "short" and row["close"] <= row["sma"]:
                return Signal(Action.EXIT, "reverted to mean")
            return HOLD

        if row["sd"] <= 0:
            return HOLD
        z = (row["close"] - row["sma"]) / row["sd"]
        if z <= -self.entry_z:
            return Signal(Action.LONG, f"z={z:.2f} below -{self.entry_z}")
        if z >= self.entry_z and self.allow_short:
            return Signal(Action.SHORT, f"z={z:.2f} above {self.entry_z}")
        return HOLD
