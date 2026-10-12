"""Plain helpers shared across test modules. Fixtures live in conftest.py; what
is here is imported by name, so it stays out of pytest's fixture machinery."""

import csv

import pandas as pd

from bot.data import LatestQuote

# Quote consistent with the 102.0 decision close the Engine factories stub, so
# the decision-price guardrail trusts the bar.
# Used by test_main_helpers.py and test_cold_start.py.
Q102 = LatestQuote(bid=101.95, ask=102.05, mid=102.0, spread_bps=10.0)


def daily_bars(closes, spread=1.0, freq="B", start="2020-01-01") -> pd.DataFrame:
    """OHLCV frame on a tz-aware DatetimeIndex. Business days by default: the
    weekend gaps are what resample-to-calendar-day code has to survive, so a
    gapless index would hide the seam. Pass freq for a contiguous or intraday
    index instead. The conftest make_bars fixture is the position-indexed
    counterpart, for tests that never touch dates."""
    closes = list(closes)
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + spread for c in closes],
            "low": [c - spread for c in closes],
            "close": closes,
            "volume": [1000.0] * len(closes),
        },
        index=idx,
    )


def patch_main_paths(monkeypatch, tmp_path):
    """Point bot.main's file constants at tmp_path. Used by the Engine factories
    in test_main_helpers.py, test_cold_start.py and test_ops.py."""
    monkeypatch.setattr("bot.main.STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setattr("bot.main.TRADES_CSV", str(tmp_path / "trades.csv"))
    monkeypatch.setattr("bot.main.DAILY_PNL_CSV", str(tmp_path / "pnl.csv"))
    monkeypatch.setattr("bot.main.EXECUTION_CSV", str(tmp_path / "fills.csv"))
    monkeypatch.setattr("bot.main.HALT_FILE", str(tmp_path / "HALT"))


def trades_rows(tmp_path):
    """Rows of the trades.csv a path-patched Engine wrote, header included.
    Used by test_main_helpers.py and test_cold_start.py."""
    return list(csv.reader((tmp_path / "trades.csv").open()))
