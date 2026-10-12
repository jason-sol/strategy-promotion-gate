"""Donchian-style breakout with volume confirmation. Long/flat only:
crypto is not shortable on Alpaca, and the trailing stop is software-managed
because Alpaca does not support broker-side trailing stops for crypto."""

import pandas as pd

from bot.indicators import atr, rolling_high, rolling_low, sma
from bot.strategies.base import HOLD, Action, PositionState, Signal, Strategy, trailing_stop_hit


class MomentumBreakout(Strategy):
    def __init__(self, channel_period: int, volume_mult: float, trail_atr_mult: float, atr_period: int):
        self.channel_period = channel_period
        self.volume_mult = volume_mult
        self.trail_atr_mult = trail_atr_mult
        self.atr_period = atr_period

    @property
    def min_bars(self) -> int:
        return max(self.channel_period, self.atr_period) + 2

    @property
    def stop_atr_mult(self) -> float:
        return self.trail_atr_mult

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        # shift(1): the channel is built from *prior* bars so the breakout bar
        # is compared against a level it did not itself set.
        df["channel_high"] = rolling_high(df["high"], self.channel_period).shift(1)
        df["channel_low"] = rolling_low(df["low"], self.channel_period).shift(1)
        df["vol_avg"] = sma(df["volume"], self.channel_period).shift(1)
        df["atr"] = atr(df, self.atr_period)
        return df

    def signal(self, df: pd.DataFrame, i: int, position: PositionState | None) -> Signal:
        if i + 1 < self.min_bars:
            return HOLD
        row = df.iloc[i]
        if pd.isna(row["channel_high"]) or pd.isna(row["vol_avg"]):
            return HOLD

        if position is not None:
            if row["close"] < row["channel_low"]:
                return Signal(Action.EXIT, "channel breakdown")
            if trailing_stop_hit(position, row["close"], row["atr"], self.trail_atr_mult):
                return Signal(Action.EXIT, "trailing stop")
            return HOLD

        volume_ok = row["volume"] >= self.volume_mult * row["vol_avg"]
        if row["close"] > row["channel_high"] and volume_ok:
            return Signal(Action.LONG, "breakout with volume")
        return HOLD
