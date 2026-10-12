"""Position accounting for the simulator plus CSV logging shared with live trading."""

import csv
import math
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone

from bot.strategies.base import PositionState
from config import INSTRUMENTS


def slippage_bps(side: str, decision_mid: float, fill_price: float) -> float:
    """Signed give-up of a fill against the decision mid, in basis points. A buy
    filled above mid (or a sell filled below mid) is positive (paid the spread);
    the opposite is negative (captured it). A non-positive or NaN mid yields NaN."""
    if not decision_mid > 0 or math.isnan(decision_mid):
        return float("nan")
    direction = 1.0 if side in ("buy", "long") else -1.0
    return direction * (fill_price - decision_mid) / decision_mid * 10_000


class SimAccount:
    """Cash/position bookkeeping for backtests. Short sales credit proceeds to
    cash and carry a mark-to-market liability until closed."""

    def __init__(self, cash: float):
        self.cash = cash
        self.positions: dict[str, PositionState] = {}

    def open_position(self, symbol: str, side: str, qty: float, fill_price: float,
                      stop_price: float, fee: float = 0.0) -> PositionState:
        if side == "long":
            self.cash -= qty * fill_price
        else:
            self.cash += qty * fill_price
        self.cash -= fee
        pos = PositionState(
            symbol=symbol, side=side, qty=qty, entry_price=fill_price,
            stop_price=stop_price, high_water_mark=fill_price, entry_fee=fee,
        )
        self.positions[symbol] = pos
        return pos

    def close_position(self, symbol: str, fill_price: float, fee: float = 0.0) -> float:
        pos = self.positions.pop(symbol)
        if pos.side == "long":
            self.cash += pos.qty * fill_price
        else:
            self.cash -= pos.qty * fill_price
        self.cash -= fee
        return pos.pnl_at(fill_price) - fee - pos.entry_fee

    def equity(self, prices: dict[str, float]) -> float:
        value = self.cash
        for symbol, pos in self.positions.items():
            price = prices.get(symbol, pos.entry_price)
            value += pos.qty * price if pos.side == "long" else -pos.qty * price
        return value


class _CsvLog:
    header: list[str] = []

    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(self.header)

    def _append(self, row: list) -> None:
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)


class TradeLog(_CsvLog):
    header = ["timestamp", "instrument", "direction", "entry_price",
              "exit_price", "pnl", "position_size"]

    def record(self, timestamp, instrument, direction, entry_price,
               exit_price, pnl, position_size) -> None:
        self._append([timestamp, instrument, direction, entry_price,
                      exit_price, pnl, position_size])


class DailyPnlLog(_CsvLog):
    header = ["date", "equity", "pnl"]

    def record(self, date, equity, pnl) -> None:
        self._append([date, equity, pnl])


class ExecutionLog(_CsvLog):
    """Measured live execution quality: one row per fill/place/cancel, so realized
    slippage can be diffed against the backtest's assumed slippage_pct."""

    header = ["timestamp", "symbol", "side", "event", "decision_mid",
              "decision_spread_bps", "limit_price", "fill_price", "fill_qty",
              "slippage_bps"]

    def record(self, timestamp, symbol, side, event, decision_mid,
               decision_spread_bps, limit_price, fill_price, fill_qty,
               slippage_bps) -> None:
        self._append([timestamp, symbol, side, event, decision_mid,
                      decision_spread_bps, limit_price, fill_price, fill_qty,
                      slippage_bps])


# Only these events move shares; placements and cancellations have no fill to
# measure. Crypto maker orders only ever rest on the entry side. A broker stop
# fires with no decision quote to measure against, so it is its own kind rather
# than an exit: folded together, unmeasurable stops would hide behind the
# measurable exits instead of showing up as a gap.
FILL_KIND = {"entry": "entry", "crypto_maker_filled": "entry",
             "exit": "exit", "stop_fill": "stop"}

# Below this, a symbol's measured median is noise: do not act on it.
SLIPPAGE_MIN_SAMPLE = 20


@dataclass(frozen=True)
class SlippageStat:
    symbol: str
    kind: str
    count: int          # fills with a usable decision quote
    dropped: int        # booked fills with none, so nothing to measure
    mean_bps: float
    median_bps: float
    worst_bps: float
    median_spread_bps: float


@dataclass
class _Bucket:
    slips: list[float] = field(default_factory=list)
    spreads: list[float] = field(default_factory=list)
    dropped: int = 0


def assumed_slippage_bps(symbol: str) -> float | None:
    """The backtest's cost assumption for a symbol, converted from a fraction of
    price to basis points. None for anything outside the configured lineup."""
    cfg = next((c for c in INSTRUMENTS if c.symbol == symbol), None)
    return cfg.slippage_pct * 10_000 if cfg else None


def _numeric(row: dict, column: str) -> float:
    try:
        return float(row[column])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _row_timestamp(value) -> datetime | None:
    try:
        ts = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def slippage_summary(path: str, since: datetime | None = None) -> list[SlippageStat]:
    """Realized slippage per symbol and fill kind, in ExecutionLog's own sign
    convention (positive = paid). `since` trims to a trailing window; None
    reports since inception. A booked fill with no usable decision quote is
    counted as dropped rather than ignored, so a kind nothing can be measured
    for still shows up. An unreadable log yields nothing rather than raising,
    so callers can probe it blind.

    The per-row guards below defend against a hand-edited or truncated log:
    ExecutionLog's own output is always well-formed."""
    buckets: dict[tuple[str, str], _Bucket] = {}
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                kind = FILL_KIND.get((row.get("event") or "").strip())
                symbol = (row.get("symbol") or "").strip()
                if kind is None or not symbol:
                    continue
                if since is not None:
                    ts = _row_timestamp(row.get("timestamp"))
                    # An unplaceable row cannot be shown to be in the window.
                    if ts is None or ts < since:
                        continue
                bucket = buckets.setdefault((symbol, kind), _Bucket())
                mid, slip, fill = (_numeric(row, "decision_mid"),
                                   _numeric(row, "slippage_bps"),
                                   _numeric(row, "fill_price"))
                if not mid > 0 or math.isnan(slip) or math.isnan(fill):
                    bucket.dropped += 1
                    continue
                bucket.slips.append(slip)
                spread = _numeric(row, "decision_spread_bps")
                if not math.isnan(spread):
                    bucket.spreads.append(spread)
    except (OSError, csv.Error, UnicodeDecodeError):
        return []

    nan = float("nan")
    return [
        SlippageStat(
            symbol=symbol, kind=kind, count=len(b.slips), dropped=b.dropped,
            mean_bps=statistics.fmean(b.slips) if b.slips else nan,
            median_bps=statistics.median(b.slips) if b.slips else nan,
            worst_bps=max(b.slips) if b.slips else nan,
            median_spread_bps=statistics.median(b.spreads) if b.spreads else nan,
        )
        for (symbol, kind), b in sorted(buckets.items())
    ]
