"""Post-earnings-announcement drift (PEAD), long-only — SIGNAL-LOGGING-ONLY.

Buy the drift after a large POSITIVE earnings surprise: stocks with high
standardized unexpected earnings (SUE) tend to keep drifting up for weeks as
the market underreacts. Entry starts on day 1 of the drift window (day 0 is the
announcement-reaction bar, which we deliberately skip to avoid the gap), holds
through `drift_window` trading days, then exits. The ATR stop is wide disaster
insurance, not the real exit — mirrors RsiMeanReversion/TurnOfMonth philosophy.

NOT in the live lineup and NOT in research EXPERIMENTS by design. The free
Finnhub tier that feeds `events` is not point-in-time (current consensus, ~4
quarters of surprise history), so a backtest would silently use estimates that
did not exist at the historical bar. The sleeve accumulates paper-forward
evidence first; a backtest on this data would be dishonest. See bot/earnings.py.

`events` is a DataFrame with columns:
- announce_date [date]  — the earnings announcement date
- hour ["bmo"/"amc"/...] — BMO announces before the open (actionable that bar),
  AMC/unknown/empty after the close (first actionable on the NEXT bar; the
  conservative default avoids lookahead)
- sue [float]           — standardized surprise; NaN never manufactures a signal
"""

import numpy as np
import pandas as pd

from bot.indicators import atr
from bot.strategies.base import HOLD, Action, PositionState, Signal, Strategy


class PeadDrift(Strategy):
    def __init__(self, events: pd.DataFrame, sue_min: float = 1.0,
                 drift_window: int = 10, atr_period: int = 14,
                 stop_atr_mult: float = 3.0):
        self.events = events
        self.sue_min = sue_min
        self.drift_window = drift_window
        self.atr_period = atr_period
        # Wide disaster stop, not the exit: the drift window flattens the book.
        self._stop_atr_mult = stop_atr_mult

    @property
    def min_bars(self) -> int:
        return self.atr_period + 5

    @property
    def stop_atr_mult(self) -> float:
        return self._stop_atr_mult

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("PeadDrift.prepare requires a DatetimeIndex (event date mapping)")
        norm = df.index.normalize().tz_localize(None) if df.index.tz else df.index.normalize()

        n = len(df)
        days_since = np.full(n, np.nan)
        sue = np.full(n, np.nan)
        active = np.zeros(n, dtype=bool)

        # Oldest first so an overlapping newer event overwrites -> most recent wins.
        for _, ev in self._sorted_events():
            s = float(ev["sue"])
            if np.isnan(s):
                continue  # a missing feature must never manufacture a signal
            ad = pd.Timestamp(ev["announce_date"]).normalize()
            hour = str(ev["hour"]).strip().lower()
            # BMO is actionable on the announce bar (>=); AMC/unknown only from
            # the next bar (>), the conservative no-lookahead default.
            side = "left" if hour == "bmo" else "right"
            p0 = int(norm.searchsorted(ad, side=side))
            if p0 >= n:
                continue
            for k in range(self.drift_window + 1):
                p = p0 + k
                if p >= n:
                    break
                days_since[p] = k
                sue[p] = s
                active[p] = True

        df["atr"] = atr(df, self.atr_period)
        df["days_since_announcement"] = days_since
        df["standardized_surprise"] = sue
        df["event_active"] = active
        return df

    def _sorted_events(self):
        if self.events is None or len(self.events) == 0:
            return []
        ev = self.events.copy()
        ev["_ad"] = pd.to_datetime(ev["announce_date"])
        return list(ev.sort_values("_ad").iterrows())

    def signal(self, df: pd.DataFrame, i: int, position: PositionState | None) -> Signal:
        if i + 1 < self.min_bars:
            return HOLD
        row = df.iloc[i]
        active = bool(row["event_active"])
        days_since = row["days_since_announcement"]

        if position is not None:
            if not active or days_since > self.drift_window:
                return Signal(Action.EXIT, "drift window elapsed")
            return HOLD

        if not active:
            return HOLD
        sue = row["standardized_surprise"]
        # Entry from day 1: day 0 is the announcement-reaction bar (the gap),
        # which this sleeve deliberately sits out.
        if pd.notna(sue) and sue >= self.sue_min and 1 <= days_since <= self.drift_window:
            return Signal(Action.LONG, f"pead sue={sue:.2f} day={int(days_since)}")
        return HOLD
