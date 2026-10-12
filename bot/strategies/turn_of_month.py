"""Turn-of-month calendar tilt, long/flat on daily bars.

Institutional month-end rebalancing flows lift broad equity indices in a narrow
window around the month turn (documented ~+10bps/day). This harvests that by
holding through the last few business days of the month and the first few of the
next, flat otherwise. Not an alpha engine — a cheap calendar overlay.
"""

import numpy as np
import pandas as pd

from bot.indicators import add_vol_features, atr
from bot.strategies.base import HOLD, Action, PositionState, Signal, Strategy, vol_gate_blocks_entry


class TurnOfMonth(Strategy):
    def __init__(self, entry_window_bdays: int = 2, exit_day_of_month: int = 3,
                 atr_period: int = 14, stop_atr_mult: float = 3.0,
                 vol_gate_percentile: float | None = None,
                 vol_gate_lookback: int = 20, vol_gate_rank_window: int = 252):
        # entry_window_bdays=N puts the last N business days of the month in the
        # entry window (N=2 -> the final two).
        self.entry_window_bdays = entry_window_bdays
        self.exit_day_of_month = exit_day_of_month
        self.atr_period = atr_period
        # Wide disaster stop, not the real exit: the calendar window is what
        # flattens the book. This ATR stop only guards against a crash inside
        # the window, so a loose default (3 ATR) is intentional.
        self._stop_atr_mult = stop_atr_mult
        self.vol_gate_percentile = vol_gate_percentile
        self.vol_gate_lookback = vol_gate_lookback
        self.vol_gate_rank_window = vol_gate_rank_window

    @property
    def min_bars(self) -> int:
        # The window is derived from each bar's own calendar date, so the signal
        # itself needs no price history — but ATR sets the stop distance AND the
        # position size, and Wilder smoothing carries a real share of its seed
        # for several multiples of its period (seven leaves under 0.1%), so a
        # short warm-up sizes the same bar differently live than in backtest.
        # The floor is a deliberate round number: 7x the default atr_period is
        # 98, and 100 also keeps the window clear of the realized-vol lookbacks
        # the sizing overlays need, however short atr_period is configured.
        return max(self.atr_period * 7, 100)

    @property
    def stop_atr_mult(self) -> float:
        return self._stop_atr_mult

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("TurnOfMonth.prepare requires a DatetimeIndex (calendar-derived window)")
        # Window depends only on the calendar date; drop tz/time so period and
        # bdate math stay on plain dates (production feeds tz-aware UTC bars).
        idx = df.index.normalize().tz_localize(None) if df.index.tz else df.index.normalize()

        # Business days strictly after this bar within its calendar month, from
        # the CALENDAR alone (never from future bars, which would be lookahead).
        # pd.bdate_range is weekend-only, not holiday-aware: around a holiday
        # month-end the true last trading day still counts a phantom holiday
        # business day ahead of it, so the entry window can shrink by up to a
        # day. Accepted approximation for a coarse tilt vs an exchange-calendar
        # dependency.
        month_ends = idx + pd.offsets.MonthEnd(0)
        bdays_remaining = np.array(
            [len(pd.bdate_range(d, e)) - 1 for d, e in zip(idx, month_ends)]
        )
        # Backward-looking position among the month's bars seen so far.
        tbday_of_month = df.groupby(idx.to_period("M")).cumcount().to_numpy() + 1

        df["in_tom_window"] = (bdays_remaining < self.entry_window_bdays) | (
            tbday_of_month <= self.exit_day_of_month
        )
        df["atr"] = atr(df, self.atr_period)
        if self.vol_gate_percentile is not None:
            df = add_vol_features(df, self.vol_gate_lookback, self.vol_gate_rank_window)
        return df

    def signal(self, df: pd.DataFrame, i: int, position: PositionState | None) -> Signal:
        if i + 1 < self.min_bars:
            return HOLD
        row = df.iloc[i]
        in_window = bool(row["in_tom_window"])

        if position is not None:
            if not in_window:
                return Signal(Action.EXIT, "outside turn-of-month window")
            return HOLD

        if not in_window:
            return HOLD
        if vol_gate_blocks_entry(row, self.vol_gate_percentile):
            return HOLD
        return Signal(Action.LONG, "turn-of-month window")
