"""Ensemble time-series momentum, long/flat on daily bars.

A single lookback picks up timing luck (a 9- vs 10-month window can diverge
wildly). Voting the sign of momentum across a symmetric band of lookbacks and
going long only on a net-positive vote is robust to any one window's noise. The
12-1 convention (skip the most recent month) sidesteps short-term reversal.
Treated as crisis-protection / drawdown control, not an alpha source.
"""

import numpy as np
import pandas as pd

from bot.indicators import add_vol_features, atr
from bot.strategies.base import HOLD, Action, PositionState, Signal, Strategy, vol_gate_blocks_entry


class TsMomentum(Strategy):
    def __init__(self, lookbacks: list[int] | None = None, skip_days: int = 21,
                 atr_period: int = 14, stop_atr_mult: float = 3.0,
                 vol_gate_percentile: float | None = None,
                 vol_gate_lookback: int = 20, vol_gate_rank_window: int = 252):
        self.lookbacks = list(lookbacks) if lookbacks is not None else [126, 147, 168, 189, 210, 231, 252]
        self.skip_days = skip_days
        self.atr_period = atr_period
        self._stop_atr_mult = stop_atr_mult
        self.vol_gate_percentile = vol_gate_percentile
        self.vol_gate_lookback = vol_gate_lookback
        self.vol_gate_rank_window = vol_gate_rank_window

    @property
    def min_bars(self) -> int:
        # The longest ts_score component needs max(lookbacks)+skip_days bars of
        # history; adding atr_period guarantees a Wilder-warmed ATR for
        # sizing/stops even if lookbacks are configured small, plus settle slack.
        return max(self.lookbacks) + self.skip_days + self.atr_period + 5

    @property
    def stop_atr_mult(self) -> float:
        return self._stop_atr_mult

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        close = df["close"]
        signs = pd.DataFrame(index=df.index)
        for lb in self.lookbacks:
            # (lb)-day momentum over the window ending skip_days before this bar.
            mom = close.shift(self.skip_days) / close.shift(lb + self.skip_days) - 1
            signs[lb] = np.sign(mom)
        # skipna=False so any un-warmed lookback keeps ts_score NaN during warmup.
        df["ts_score"] = signs.mean(axis=1, skipna=False)
        df["atr"] = atr(df, self.atr_period)
        if self.vol_gate_percentile is not None:
            df = add_vol_features(df, self.vol_gate_lookback, self.vol_gate_rank_window)
        return df

    def signal(self, df: pd.DataFrame, i: int, position: PositionState | None) -> Signal:
        if i + 1 < self.min_bars:
            return HOLD
        row = df.iloc[i]
        score = row["ts_score"]
        if pd.isna(score):
            return HOLD

        if position is not None:
            if score <= 0:
                return Signal(Action.EXIT, "momentum vote flat/negative")
            return HOLD

        if vol_gate_blocks_entry(row, self.vol_gate_percentile):
            return HOLD

        if score > 0:
            return Signal(Action.LONG, f"ts_score={score:.2f}")
        return HOLD
