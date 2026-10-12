"""Connors-style RSI(2) pullback buying on daily bars, long-only.

Buy sharp dips inside a long-term uptrend (close above the 200-SMA regime
filter), exit on the snap-back above the 5-SMA. Wide ATR stop: tight stops are
known to destroy this edge, so the stop is disaster insurance, not an exit.
"""

import pandas as pd

from bot.indicators import add_vol_features, atr, rsi, sma
from bot.strategies.base import HOLD, Action, PositionState, Signal, Strategy, vol_gate_blocks_entry


class RsiMeanReversion(Strategy):
    def __init__(self, rsi_period: int = 2, entry_below: float = 10.0,
                 regime_period: int = 200, exit_period: int = 5,
                 stop_mult: float = 3.0, atr_period: int = 14,
                 confirm_days: int = 1, vol_gate_percentile: float | None = None,
                 vol_gate_lookback: int = 20, vol_gate_rank_window: int = 252):
        self.rsi_period = rsi_period
        self.entry_below = entry_below
        self.regime_period = regime_period
        self.exit_period = exit_period
        self.stop_mult = stop_mult
        self.atr_period = atr_period
        self.confirm_days = confirm_days
        self.vol_gate_percentile = vol_gate_percentile
        self.vol_gate_lookback = vol_gate_lookback
        self.vol_gate_rank_window = vol_gate_rank_window

    @property
    def min_bars(self) -> int:
        return self.regime_period + 10

    @property
    def stop_atr_mult(self) -> float:
        return self.stop_mult

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        df["rsi"] = rsi(df["close"], self.rsi_period)
        df["regime_sma"] = sma(df["close"], self.regime_period)
        df["exit_sma"] = sma(df["close"], self.exit_period)
        df["atr"] = atr(df, self.atr_period)
        # Require close above the SMA for confirm_days straight bars to cut whipsaw.
        above = df["close"] > df["regime_sma"]
        df["regime_ok"] = above.rolling(self.confirm_days).min().fillna(0).astype(bool)
        if self.vol_gate_percentile is not None:
            df = add_vol_features(df, self.vol_gate_lookback, self.vol_gate_rank_window)
        return df

    def signal(self, df: pd.DataFrame, i: int, position: PositionState | None) -> Signal:
        if i + 1 < self.min_bars:
            return HOLD
        row = df.iloc[i]
        if pd.isna(row["regime_sma"]) or pd.isna(row["rsi"]):
            return HOLD

        if position is not None:
            if row["close"] > row["exit_sma"]:
                return Signal(Action.EXIT, "snap-back above exit SMA")
            return HOLD

        if vol_gate_blocks_entry(row, self.vol_gate_percentile):
            return HOLD

        if row["regime_ok"] and row["rsi"] < self.entry_below:
            return Signal(Action.LONG, f"rsi={row['rsi']:.1f} in uptrend")
        return HOLD
