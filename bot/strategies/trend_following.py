"""EMA crossover trend following. Entries on the cross itself; exits on the
opposite cross or an ATR trailing stop."""

import pandas as pd

from bot.indicators import add_vol_features, atr, ema, sma
from bot.strategies.base import (
    HOLD, Action, PositionState, Signal, Strategy, trailing_stop_hit,
    vol_gate_blocks_entry, volume_gate_blocks_entry,
)


class TrendFollowing(Strategy):
    def __init__(self, fast_period: int, slow_period: int, trail_atr_mult: float,
                 atr_period: int, allow_short: bool, vol_gate_percentile: float | None = None,
                 vol_gate_lookback: int = 20, vol_gate_rank_window: int = 252,
                 volume_gate_mult: float | None = None, volume_gate_lookback: int = 20):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.trail_atr_mult = trail_atr_mult
        self.atr_period = atr_period
        self.allow_short = allow_short
        self.vol_gate_percentile = vol_gate_percentile
        self.vol_gate_lookback = vol_gate_lookback
        self.vol_gate_rank_window = vol_gate_rank_window
        self.volume_gate_mult = volume_gate_mult
        self.volume_gate_lookback = volume_gate_lookback

    @property
    def min_bars(self) -> int:
        # The slow EMA is the longest recursive span here, and an EMA never
        # fully forgets its seed: span-N still carries ~0.03% of it after 4N
        # bars and materially more before that. Live sees only min_bars of
        # history, so a shorter warm-up reads the same bar differently than the
        # backtest that validated the parameters.
        ema_warmup = self.slow_period * 4
        # The volume average is shifted a bar, so it needs lookback + 1.
        volume_warmup = (self.volume_gate_lookback + 1
                         if self.volume_gate_mult is not None else 0)
        return max(ema_warmup, volume_warmup)

    @property
    def stop_atr_mult(self) -> float:
        return self.trail_atr_mult

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        df["ema_fast"] = ema(df["close"], self.fast_period)
        df["ema_slow"] = ema(df["close"], self.slow_period)
        df["atr"] = atr(df, self.atr_period)
        if self.vol_gate_percentile is not None:
            df = add_vol_features(df, self.vol_gate_lookback, self.vol_gate_rank_window)
        if self.volume_gate_mult is not None:
            # shift(1): the average is built from PRIOR bars so the signal
            # bar's own volume does not set the level it is judged against.
            df["volume_avg"] = sma(df["volume"], self.volume_gate_lookback).shift(1)
        return df

    def signal(self, df: pd.DataFrame, i: int, position: PositionState | None) -> Signal:
        if i + 1 < self.min_bars:
            return HOLD
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        fast_above = row["ema_fast"] > row["ema_slow"]
        prev_fast_above = prev["ema_fast"] > prev["ema_slow"]

        if position is not None:
            if position.side == "long" and not fast_above:
                return Signal(Action.EXIT, "death cross")
            if position.side == "short" and fast_above:
                return Signal(Action.EXIT, "golden cross")
            if trailing_stop_hit(position, row["close"], row["atr"], self.trail_atr_mult):
                return Signal(Action.EXIT, "trailing stop")
            return HOLD

        if vol_gate_blocks_entry(row, self.vol_gate_percentile):
            return HOLD
        if volume_gate_blocks_entry(row, self.volume_gate_mult):
            return HOLD

        if fast_above and not prev_fast_above:
            return Signal(Action.LONG, "golden cross")
        if not fast_above and prev_fast_above and self.allow_short:
            return Signal(Action.SHORT, "death cross")
        return HOLD
