"""Live/paper trading loop.

Usage:
    uv run python -m bot.main            # trade against the configured Alpaca account
    uv run python -m bot.main --dry-run  # compute and log signals, submit nothing

Wakes every minute; each instrument is evaluated once per bar of its timeframe —
closed bars only, except daily equities inside the near-close submission window,
which decide on the session's forming bar because that is the bar they fill on.
Equities only trade while the market is open; crypto trades 24/7.
A HALT file (created by the drawdown circuit breaker, or by you) stops all
trading until removed.

Restart invariant
-----------------
The broker is the only authority on what is open; state.json is an accelerator,
never authoritative trading state. Anything added here has to keep that shape —
reconstructible from the broker, or safe to lose:

- Every cycle re-reads positions and resting orders from the broker and rebuilds
  what is missing: a position with no persisted stop gets a recomputed ATR stop,
  a stop with no position is cancelled, a stale stop is re-placed rather than
  acted on.
- Entry dedupe asks the broker as well as the state file. The resting-order
  marker covers the same-bar case and the broker check is additive to it, so a
  lost volume cannot let one signal double a position.
- A lost state file can only weaken protection — a forgiven drawdown, a reset
  cooldown — never produce a wrong action on open risk.
- peak_equity has no broker-side source, so a lost peak re-seeds at current
  equity: it can fail to halt, never halt spuriously.
- Crypto stops are the one carve-out, because there is no broker-side stop to
  defer to. They are software-enforced from persisted state, so a persisted
  crypto stop the live price is already through closes the position on the first
  poll after a restart, right or not. The direction is de-risking: an unwanted
  exit, never unwanted exposure.

The HALT sentinel sits outside all of it — a decision, not a cache, and a
restart must obey it.
"""

import argparse
import hashlib
import logging
import math
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

from zoneinfo import ZoneInfo

from bot import commands  # noqa: F401  (bot.main.commands is a patch point)
from bot.broker import Broker, rejection_reason
from bot.ops import Ops, warn_unvalidated_instruments  # noqa: F401  (re-export)
from bot.halt import halt_tripped, trip_halt
from bot.data import (
    MarketData,
    bar_boundary,
    decision_bars,
    et_session_date,
    is_daily_equity,
)
from bot.portfolio import DailyPnlLog, ExecutionLog, TradeLog, slippage_bps
from bot.risk_manager import RiskManager
from bot.strategies import build_strategy
from bot.strategies.base import (
    Action,
    PositionState,
    effective_stop,
    resolve_stop_and_hwm,
)
from bot.state import StateStore, parse_utc, utc_day
from config import (
    DAILY_PNL_CSV,
    EXECUTION,
    EXECUTION_CSV,
    HALT_FILE,
    MARKET_DATA_FEED,
    PROTECTIONS,
    RISK,
    STATE_FILE,
    TRADES_CSV,
    AssetClass,
    enabled_instruments,
)

log = logging.getLogger("bot")

POLL_SECONDS = 60
BAR_WINDOW_EXTRA = 60
MAX_BACKOFF_SECONDS = 900
# A pending fill unresolved past this is dropped unmeasured. Roughly three
# equity sessions: a marketable limit that hasn't filled by then never will.
PENDING_FILL_STALE = timedelta(days=3)
TERMINAL_UNFILLED_STATUSES = frozenset({"canceled", "cancelled", "expired", "rejected"})
# A processed bar whose close is restated by more than this between two fetches
# is a corporate action, not a data blip: dividends restate by far less.
CORPORATE_ACTION_RESCALE_PCT = 0.01


def _order_status(order) -> str:
    return str(getattr(order.status, "value", order.status)).lower()


def _filled_at(order) -> datetime | None:
    """Timezone-aware fill instant of a broker order, or None when the broker
    reports none (or something unparseable) — an order we cannot date must never
    be matched to a position by recency."""
    value = getattr(order, "filled_at", None)
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def decision_bar_key(cfg, now_utc: datetime) -> str:
    """Identity of the bar this cycle will decide on, for the once-per-bar
    dedupe. A daily equity bar is stamped at ET midnight and may be consumed
    while it is still forming, so its identity is the ET session date; keying off
    the UTC boundary would let the same session be traded twice."""
    if is_daily_equity(cfg.asset_class, cfg.timeframe_minutes):
        return et_session_date(now_utc).isoformat()
    return bar_boundary(now_utc, cfg.timeframe_minutes).isoformat()


def marketable_limit_price(signal_close: float, side: str, limit_bps: float) -> float:
    """Cross the spread by limit_bps: buys reach up through the offer, sells down
    through the bid. Rounded to the penny (Alpaca sub-penny rule, stocks >= $1)."""
    direction = 1.0 if side == "buy" else -1.0
    return round(signal_close * (1 + direction * limit_bps / 10_000), 2)


def quote_anchored_limit_price(quote, side: str, limit_bps: float) -> float:
    """Marketable limit anchored to the live quote: a buy crosses up from the ask,
    a sell down through the bid, by limit_bps. Rounded to the penny."""
    anchor = quote.ask if side == "buy" else quote.bid
    direction = 1.0 if side == "buy" else -1.0
    return round(anchor * (1 + direction * limit_bps / 10_000), 2)


def equity_entry_limit_price(signal_close: float, side: str, limit_bps: float,
                             quote) -> float:
    """Prefer the live quote (buys cross the ask, sells the bid); fall back to the
    stale signal-bar close when the quote is missing or its anchor is unusable."""
    if quote is not None:
        anchor = quote.ask if side == "buy" else quote.bid
        if anchor == anchor and anchor > 0:  # finite and positive
            return quote_anchored_limit_price(quote, side, limit_bps)
    return marketable_limit_price(signal_close, side, limit_bps)


def passive_limit_price(quote, side: str, passive_bps: float) -> float:
    """Passive (liquidity-posting) limit anchored to the mid: a buy rests below the
    mid, a sell above, by passive_bps. Rounded to the penny."""
    direction = -1.0 if side == "buy" else 1.0
    return round(quote.mid * (1 + direction * passive_bps / 10_000), 2)


def _stable_unit_fraction(symbol: str, session_date: str) -> float:
    """Deterministic value in [0, 1) from (symbol, session date). Not cryptographic
    — this yields unpredictability to other participants, not a security control."""
    digest = hashlib.sha256(f"{symbol}|{session_date}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def jitter_offset_minutes(symbol: str, session_date: str, jitter_max_minutes: float,
                          window_minutes: float, ladder_enabled: bool) -> float:
    """Deterministic per-day submission offset (minutes) from the near-close window
    start, capped so it stays inside the window and leaves room to submit — and,
    with the ladder on, at least two poll cycles to escalate before the close."""
    if jitter_max_minutes <= 0:
        return 0.0
    safety = (2 if ladder_enabled else 1) * POLL_SECONDS / 60
    cap = min(jitter_max_minutes, window_minutes - safety)
    if cap <= 0:
        return 0.0
    return _stable_unit_fraction(symbol, session_date) * cap


def crypto_maker_limit_price(signal_close: float, offset_bps: float,
                             price_increment: float = 1.0) -> float:
    """Rest a buy below the signal close by offset_bps so the fill earns the
    maker rate instead of crossing the spread. Floored to price_increment: BTC/USD
    limits must be whole dollars (increment 1) or Alpaca rejects them, and a
    rejection is indistinguishable from a non-fill in the fill-rate logs."""
    raw = signal_close * (1 - offset_bps / 10_000)
    return math.floor(raw / price_increment) * price_increment


def in_near_close_window(now_utc: datetime, market_close_utc: datetime,
                         minutes: int) -> bool:
    """True within the last `minutes` before the close. The window start (exactly
    `minutes` out) is inclusive; only the close instant itself is excluded."""
    remaining = (market_close_utc - now_utc).total_seconds()
    return 0 < remaining <= minutes * 60


def in_open_auction_window(now_utc: datetime, market_open_utc: datetime,
                           minutes: int = 5) -> bool:
    """True within the first `minutes` after the open, when spreads are widest."""
    elapsed = (now_utc - market_open_utc).total_seconds()
    return 0 <= elapsed < minutes * 60


class Engine:
    def __init__(self, broker: Broker, data: MarketData, dry_run: bool):
        self.broker = broker
        self.data = data
        self.dry_run = dry_run
        self.risk = RiskManager(RISK)
        self.state = StateStore(STATE_FILE)
        self.trade_log = TradeLog(TRADES_CSV)
        self.daily_log = DailyPnlLog(DAILY_PNL_CSV)
        self.execution_log = ExecutionLog(EXECUTION_CSV)
        self.strategies = {cfg.symbol: (cfg, build_strategy(cfg))
                           for cfg in enabled_instruments()}
        self.ops = Ops(self)
        self.last_boundary: dict[str, datetime] = {}
        self.last_heartbeat: datetime | None = None
        self._session_cache: tuple[object, tuple[datetime, datetime] | None] | None = None
        # ET session each symbol's entry stop leg was submitted in. In memory on
        # purpose: after a restart, a missing stop is treated as lost.
        self._entry_leg_sessions: dict[str, date] = {}

    def positions(self) -> dict[str, PositionState]:
        """Broker truth overlaid with bot-side stop/high-water-mark state."""
        positions = self.broker.get_positions()
        for symbol, pos in positions.items():
            meta = self.state.position_meta(symbol)
            if meta is None and self.exit_in_flight(symbol):
                # The close is submitted but the fill isn't reflected yet. The
                # engine has deliberately forgotten this position: resurrecting
                # its state would book the settled close a second time.
                pos.stop_price, pos.high_water_mark = resolve_stop_and_hwm(pos, None)
                continue
            if meta is None:
                # Position exists at the broker but we have no memory of it
                # (e.g. state file lost): rebuild the sized stop rather than
                # leaving the position on a placeholder.
                pos.stop_price, pos.high_water_mark, degraded = \
                    self._recovered_stop_and_hwm(pos)
                self.state.set_position_meta(symbol, pos.stop_price,
                                             pos.high_water_mark, degraded=degraded)
            else:
                pos.stop_price, pos.high_water_mark = resolve_stop_and_hwm(pos, meta)
            self.state.set_last_seen_position(symbol, pos.side, pos.qty, pos.entry_price)
        # Drop state for positions that no longer exist (broker-side stop filled).
        for symbol in list(self.state.data["positions"]):
            if symbol not in positions and not self._entry_pending(symbol):
                self.reconcile_vanished_position(symbol)
                self.state.clear_position_meta(symbol)
                self.state.clear_last_seen_position(symbol)
        # A failed exit resolves itself the moment the position is flat, however
        # it got there; leaving the alert up would make it permanent.
        for symbol in list(self.state.data["exit_failures"]):
            if symbol not in positions:
                self.state.clear_exit_failure(symbol)
        return positions

    def exit_in_flight(self, symbol: str) -> bool:
        """A close order is submitted and awaiting its fill."""
        return (self.state.pending_fill(symbol) or {}).get("event") == "exit"

    def _entry_pending(self, symbol: str) -> bool:
        """An entry is working with no position to show for it yet. The intended
        stop is persisted before the order is submitted, so it has to outlive the
        whole placement-to-fill window rather than look like a vanished position."""
        if self.state.resting_order(symbol) or self.state.ladder_order(symbol):
            return True
        return ((self.state.pending_fill(symbol) or {}).get("event")
                in ("entry", "crypto_maker_placed"))

    def _latest_atr(self, symbol: str) -> float | None:
        """ATR from the same prepared frame the strategy trades on, or None when
        bars or indicators are unavailable."""
        entry = self.strategies.get(symbol)
        if entry is None:
            return None
        cfg, strategy = entry
        try:
            bars = self.data.get_recent_bars(
                symbol, cfg.asset_class, cfg.timeframe_minutes,
                bars=strategy.min_bars + BAR_WINDOW_EXTRA)
            bars = decision_bars(bars, cfg.asset_class, cfg.timeframe_minutes,
                                 datetime.now(timezone.utc))
            value = float(strategy.prepare(bars)["atr"].iloc[-1])
        except Exception:
            log.exception("%s: ATR unavailable", symbol)
            return None
        return None if math.isnan(value) else value

    def _atr_sized_stop(self, pos: PositionState, atr_value: float,
                        mult: float) -> float:
        distance = mult * atr_value
        return (pos.entry_price - distance if pos.side == "long"
                else pos.entry_price + distance)

    def _recovered_stop_and_hwm(self, pos: PositionState) -> tuple[float, float, bool]:
        """(stop, high-water mark, degraded) for a broker position we have no
        persisted stop for. Recomputes the strategy's ATR-sized stop; the flat
        percentage fallback is a last resort and is flagged for refinement."""
        entry = self.strategies.get(pos.symbol)
        atr_value = self._latest_atr(pos.symbol) if entry is not None else None
        if entry is None or atr_value is None:
            stop, hwm = resolve_stop_and_hwm(pos, None)
            log.warning("%s: no persisted stop and no ATR — DEGRADED stop %.2f "
                        "(flat %% of entry, not the sized stop); refining on the "
                        "next processed bar", pos.symbol, stop)
            return stop, hwm, True
        mult = entry[1].stop_atr_mult
        stop = self._atr_sized_stop(pos, atr_value, mult)
        log.warning("%s: no persisted stop — recomputed %.1f x ATR stop at %.2f",
                    pos.symbol, mult, stop)
        return stop, pos.entry_price, False

    def reconcile_vanished_position(self, symbol: str) -> None:
        """A tracked position gone from the broker means an order closed it out of
        band — almost always the resting stop. Book it before the state is
        dropped, otherwise the exit never reaches trades.csv or fills.csv."""
        if self.dry_run:
            return
        last = self.state.last_seen_position(symbol)
        seen_at = parse_utc(last.get("seen_at")) if last else None
        if seen_at is None:
            # No dated snapshot means no way to tell this position's exit from an
            # unrelated older fill; booking on a guess writes a phantom trade.
            log.warning("%s: position vanished with no dated last-seen snapshot — "
                        "exit UNRECONCILED, nothing booked", symbol)
            return
        try:
            orders = self.broker.closed_orders(symbol)
        except Exception:
            log.exception("%s: position vanished and closed orders could not be "
                          "queried — exit UNRECONCILED", symbol)
            return

        def closed_this_position(candidate) -> bool:
            if (_order_status(candidate) != "filled"
                    or candidate.filled_avg_price is None):
                return False
            filled_at = _filled_at(candidate)
            return filled_at is not None and filled_at >= seen_at

        order = next((o for o in orders if closed_this_position(o)), None)
        if order is None:
            log.warning("%s: position vanished with no filled order newer than the "
                        "last-seen snapshot — exit UNRECONCILED", symbol)
            return
        fill_qty = float(order.filled_qty)
        fill_price = float(order.filled_avg_price)
        is_stop = "stop" in str(order.order_type).lower()
        event = "stop_fill" if is_stop else "exit"
        exit_side = "sell" if last["side"] == "long" else "buy"
        gross = fill_qty * (fill_price - last["entry_price"])
        pnl = gross if last["side"] == "long" else -gross
        log.info("%s: EXIT %s qty=%s entry=%.2f fill=%.2f pnl~%.2f (broker %s)",
                 symbol, last["side"], fill_qty, last["entry_price"], fill_price,
                 pnl, event)
        self.trade_log.record(
            _filled_at(order).isoformat(), symbol, last["side"], last["entry_price"],
            fill_price, round(pnl, 2), fill_qty)
        self._write_fill_row(symbol, event, exit_side, float("nan"), float("nan"),
                             None, fill_price, fill_qty)
        self._record_exit_outcome(symbol, pnl)

    def _record_exit_outcome(self, symbol: str, pnl: float) -> None:
        """Advance or clear the symbol's losing streak, and pause its entries
        once the streak reaches the limit. Driven from the two booking sites, so
        the streak counts exactly the exits trades.csv books — a split fill
        books, and so counts, twice."""
        if PROTECTIONS.cooldown_after_losses <= 0:
            return
        if pnl >= 0:
            self.state.set_losing_streak(symbol, 0)
            return
        streak = self.state.losing_streak(symbol) + 1
        if streak < PROTECTIONS.cooldown_after_losses:
            self.state.set_losing_streak(symbol, streak)
            return
        until = datetime.now(timezone.utc) + timedelta(days=PROTECTIONS.cooldown_days)
        log.warning("%s: COOLDOWN — %d consecutive losing exits; entries paused "
                    "for %dd, until %s", symbol, streak, PROTECTIONS.cooldown_days,
                    until.date().isoformat())
        self.state.start_cooldown(symbol, until, streak, PROTECTIONS.cooldown_days)
        self.state.set_losing_streak(symbol, 0)

    def cooldown_blocked(self, symbol: str) -> bool:
        """Entry gate for a symbol standing down after a losing streak. Exits,
        stops and the protective sweeps are deliberately ungated — a cooldown
        stops new risk, it does not abandon the risk already on."""
        record = self.state.cooldown(symbol)
        if record is None:
            return False
        log.info("%s: entry blocked — cooldown after %d consecutive losses, "
                 "until %s", symbol, record["losses"], record["until"][:10])
        return True

    def check_daily_loss_limit(self, equity: float) -> bool:
        """Halt on a fast intraday bleed. Deliberately no flatten: the drawdown
        breaker owns that, and this is the earlier, softer brake — the sweeps
        keep every open position on its stop while a human decides.

        Fires once per UTC day. Clearing the halt is the operator deciding to
        trade on, and re-tripping while equity is still underwater would
        overrule that decision every cycle, leaving no way to resume before
        midnight. The breaker underneath is what still answers for the day
        getting worse."""
        limit = PROTECTIONS.daily_loss_limit_pct
        if limit <= 0:
            return False
        day_start = self.state.seed_day_start_equity(equity)
        if day_start <= 0 or equity >= (1 - limit) * day_start:
            return False
        if self.state.daily_loss_tripped_today():
            return False
        log.error("DAILY LOSS LIMIT: equity %.2f is more than %.1f%% below the "
                  "day's start %.2f — halting, positions keep their stops",
                  equity, limit * 100, day_start)
        # Sentinel first: a write that fails must leave the limit armed, not
        # spend the day's one shot on a halt that never landed.
        trip_halt(f"daily loss limit: equity={equity:.2f} "
                  f"day_start={day_start:.2f}", HALT_FILE)
        self.state.record_daily_loss_trip()
        return True

    def check_circuit_breaker(self, equity: float) -> bool:
        peak = max(self.state.data.get("peak_equity") or 0.0, equity)
        self.state.data["peak_equity"] = peak
        self.state.save()
        if self.risk.drawdown_halted(equity, peak):
            log.error("CIRCUIT BREAKER: equity %.2f is more than %.1f%% below peak "
                      "%.2f — closing everything and halting",
                      equity, self.risk.risk.max_drawdown * 100, peak)
            # Halt first: a close that fails must never leave the bot trading on.
            trip_halt(f"drawdown halt: equity={equity:.2f} peak={peak:.2f}", HALT_FILE)
            if not self.dry_run:
                self._flatten_everything()
            return True
        return False

    def _flatten_everything(self) -> None:
        """Best-effort close of every position after the breaker trips. A close
        that fails has already had its protective stop cancelled, so the stop goes
        straight back on — the halted cycle's sweep is the only thing left after
        this, and it must not find the position naked."""
        self._cancel_every_resting_order()
        try:
            open_positions = self.positions()
        except Exception:
            log.exception("CIRCUIT BREAKER: positions unreadable — flatten deferred "
                          "to the next cycle, existing stops still rest")
            return
        for symbol, pos in open_positions.items():
            try:
                self.broker.close_position(symbol)
            except Exception as exc:
                log.error("%s: CIRCUIT BREAKER close FAILED (%s) — position still "
                          "open, flatten it manually", symbol, exc)
                self.state.record_exit_failure(
                    symbol, f"circuit-breaker close failed: {exc}")
                entry = self.strategies.get(symbol)
                if entry is None:
                    log.error("%s: untracked symbol — no stop can be restored",
                              symbol)
                else:
                    self._restore_protective_stop(entry[0], pos)

    def _cancel_every_resting_order(self) -> None:
        """Quiesce the book before flattening, across the whole lineup rather
        than the symbols that happen to hold a position. A resting entry on a
        flat symbol is invisible to the close loop, and left live it can fill
        after the halt and open the exposure the breaker just shut down."""
        for symbol in self.strategies:
            try:
                self.broker.cancel_open_orders(symbol)
            except Exception as exc:
                log.error("%s: CIRCUIT BREAKER cancel FAILED (%s) — an order may "
                          "still be resting, check it manually", symbol, exc)
                self.state.record_cancel_failure(symbol, str(exc))

    def keeps_forming_bar(self, cfg, now_utc: datetime,
                          session: tuple[datetime, datetime] | None) -> bool:
        """Whether a daily equity decision may use the session's still-forming
        bar. Inside the near-close window it must: the backtest signals and fills
        on the same bar's close, and a submission ~10 minutes before the close
        fills within a few bps of that close. Deciding on the previous session's
        bar instead — which dropping every forming bar forces — signals on a
        price a whole day away from the one we actually pay."""
        if not is_daily_equity(cfg.asset_class, cfg.timeframe_minutes):
            return False
        return session is not None and in_near_close_window(
            now_utc, session[1], EXECUTION.near_close_minutes)

    def check_startup_windows(self) -> None:
        return self.ops.check_startup_windows(BAR_WINDOW_EXTRA)

    def process_instrument(self, symbol: str, now_utc: datetime | None = None,
                           session: tuple[datetime, datetime] | None = None) -> bool:
        """True when this bar was evaluated and must not be retried before the
        next boundary; False when the bar was skipped or failed and a later cycle
        should try again."""
        cfg, strategy = self.strategies[symbol]
        now = now_utc or datetime.now(timezone.utc)
        fetched = self.data.get_recent_bars(
            symbol, cfg.asset_class, cfg.timeframe_minutes,
            bars=strategy.min_bars + BAR_WINDOW_EXTRA)
        # Completed bars only, so signals see what the backtest saw (a seconds-old
        # bar has near-zero volume and a meaningless close) — except the near-close
        # daily equity case, where the forming bar IS the fill bar.
        completed = decision_bars(fetched, cfg.asset_class, cfg.timeframe_minutes, now)
        self.apply_corporate_action_rescale(symbol, completed)
        keep_forming = self.keeps_forming_bar(cfg, now, session)
        bars = (decision_bars(fetched, cfg.asset_class, cfg.timeframe_minutes, now,
                              keep_forming=True) if keep_forming else completed)
        boundary = bar_boundary(now, cfg.timeframe_minutes)
        if len(bars) < strategy.min_bars:
            # No decision at all on a window the strategy cannot compute its
            # signal from. Broker-side stops and the protective sweeps are
            # unaffected: stops, fills and vanished positions reconcile in the
            # cycle's sweep, which runs before this.
            detail = f"{len(bars)} bars available, {strategy.min_bars} needed"
            log.error("%s: SHORT decision window — %s; no decision taken this bar",
                      symbol, detail)
            self.state.record_short_window(symbol, detail)
            return True
        self.state.clear_short_window(symbol)
        df = strategy.prepare(bars)
        i = len(df) - 1
        close = float(df["close"].iloc[i])
        atr_value = float(df["atr"].iloc[i])

        positions = self.positions()
        pos = positions.get(symbol)
        if pos is not None:
            pos.update_high_water_mark(close)
            meta = self.state.position_meta(symbol)
            if meta is not None and meta.get("degraded"):
                pos.stop_price = self._atr_sized_stop(pos, atr_value,
                                                      strategy.stop_atr_mult)
                log.warning("%s: degraded stop refined to %.2f (%.1f x ATR)",
                            symbol, pos.stop_price, strategy.stop_atr_mult)
            self.state.set_position_meta(symbol, pos.stop_price, pos.high_water_mark)

        entry_blocked = False
        cancelled_resting = False
        if cfg.asset_class == AssetClass.CRYPTO:
            entry_blocked, cancelled_resting = self.reconcile_crypto_resting(
                symbol, boundary, pos)

        signal = strategy.signal(df, i, pos)
        log.info("%s close=%.2f action=%s %s", symbol, close, signal.action.value,
                 signal.reason)

        if signal.action == Action.EXIT and pos is not None:
            if self.exit_in_flight(symbol):
                log.info("%s: exit already working — not resubmitting", symbol)
                return True
            return self.exit_position(cfg, pos, close, signal.reason, atr_value)
        elif signal.action in (Action.LONG, Action.SHORT) and pos is None and not entry_blocked:
            if self.cooldown_blocked(symbol):
                # Bar consumed: a later cycle on the same bar would find the
                # same cooldown, so there is nothing to retry.
                return True
            side = "long" if signal.action == Action.LONG else "short"
            quote = self._decision_quote(symbol, cfg.asset_class)
            if not self._decision_price_trusted(symbol, close, quote):
                return False
            if not self._within_spread_limit(symbol, quote):
                # Skip WITHOUT marking the bar evaluated (mirrors the run_cycle
                # submission-window gate): a later cycle can retry if it narrows.
                return False
            if self._entry_order_outstanding(cfg):
                # A retried bar must not stack a second entry on an order that is
                # already working.
                log.warning("%s: entry suppressed — an order is already outstanding",
                            symbol)
                if cfg.asset_class == AssetClass.CRYPTO:
                    # Crypto reaching here at all is an anomaly: the same-bar
                    # case is already handled by the resting-order marker above,
                    # so the order is either one this cycle just cancelled or one
                    # the engine never knew about. Only the latter is marked — a
                    # symbol that has silently stopped taking entries must reach
                    # the briefing, and only an accepted entry clears the marker,
                    # so marking a cancel that settles a beat later would leave a
                    # standing alert with nothing to resolve it. Equity is never
                    # marked here: its suppression also covers an ordinary
                    # working entry limit, which would read as a standing alert
                    # for a normal unfilled order.
                    if not cancelled_resting:
                        self.state.record_entry_refusal(
                            symbol, "entry suppressed: unexpected resting order "
                                    "at the broker")
                    # The broker releases a cancel asynchronously, so an order
                    # this cycle cancelled can still read as outstanding for a
                    # beat. Retry on the next poll rather than spending the bar
                    # — for a daily strategy that costs a whole signal. An order
                    # the engine did not place has no cancel to settle, so it
                    # retries for as long as it rests: never stacking a second
                    # entry, and never silently.
                    return False
                return True
            rvol = self.risk.realized_vol_for(df["close"], i, cfg)
            self.enter_position(cfg, side, close, atr_value, positions, boundary, rvol, quote)
        return True

    def _entry_order_outstanding(self, cfg) -> bool:
        """Whether an order for this symbol is already working at the broker.

        Entry dedupe asks the broker, not only the state file: a resting order is
        the broker's fact and state.json only mirrors it, so a lost volume must
        not let one signal double a position. Every asset class is checked —
        crypto's resting-order marker covers the same-bar case and this is
        additive, not a substitute for it. Callers differ on whether a
        suppression should also consume the bar; see process_instrument."""
        if self.dry_run:
            return False
        return self.broker.has_open_order(cfg.symbol)

    def enter_position(self, cfg, side: str, close: float, atr_value: float,
                       positions: dict[str, PositionState], boundary: datetime,
                       realized_vol: float | None = None, quote=None) -> None:
        symbol = cfg.symbol
        # Re-read the switch here, not once per cycle: a cycle walks six
        # instruments over minutes, and a trip landing partway through must stop
        # the orders that come after it. Exits are deliberately not gated — the
        # breaker trips the switch and then flattens through them.
        if halt_tripped(HALT_FILE):
            log.warning("%s: entry refused — HALT is set", symbol)
            return
        _, strategy = self.strategies[symbol]
        equity = self.broker.account_equity()
        sizing = self.risk.approve_entry(
            symbol, side, equity, close, atr_value,
            cfg.asset_class, positions, strategy.stop_atr_mult,
            realized_vol=realized_vol)
        if sizing is None:
            log.info("%s: %s entry blocked (correlation filter or zero size)", symbol, side)
            return
        log.info("%s: ENTER %s qty=%s stop=%.2f (atr=%.2f)",
                 symbol, side, sizing.qty, sizing.stop_price, atr_value)
        self._log_risk_budget(symbol, sizing, close, equity, realized_vol)
        if self.dry_run:
            return
        order_side = "buy" if side == "long" else "sell"
        # Persist intended stop BEFORE submitting so a crash between order and
        # state write still leaves a stop for reconciliation to restore.
        self.state.set_position_meta(symbol, sizing.stop_price, close)
        if cfg.asset_class == AssetClass.US_EQUITY:
            if EXECUTION.ladder_enabled and quote is not None and quote.mid > 0:
                # Passive leg: post at mid carrying the OTO stop, mark the ladder
                # BEFORE submitting so a crash still leaves a recoverable stop and
                # positions() preserves it until the leg fills. The reconcile sweep
                # escalates to a marketable cross on the next in-window cycle.
                passive_price = passive_limit_price(quote, order_side,
                                                    EXECUTION.ladder_passive_bps)
                self.state.set_ladder_order(symbol, {
                    "order_id": None,
                    "boundary": boundary.isoformat(),
                    "stop_price": sizing.stop_price,
                    "side": order_side,
                    "total_qty": sizing.qty,
                    "signal_close": close,
                    "submitted_at": datetime.now(timezone.utc).isoformat(),
                })
                order = self.broker.limit_order(symbol, order_side, sizing.qty,
                                                cfg.asset_class, passive_price,
                                                stop_loss=sizing.stop_price)
                if self._entry_was_refused(symbol, order):
                    return
                marker = self.state.ladder_order(symbol)
                marker["order_id"] = str(order.id)
                self.state.set_ladder_order(symbol, marker)
                self._set_pending_fill(symbol, order, order_side, "entry", quote,
                                       passive_price)
                log.info("%s: ladder passive entry placed limit=%.2f qty=%s "
                         "(%.1fbps inside mid)", symbol, passive_price, sizing.qty,
                         EXECUTION.ladder_passive_bps)
            else:
                limit_price = equity_entry_limit_price(close, order_side,
                                                       EXECUTION.limit_bps, quote)
                order = self.broker.limit_order(symbol, order_side, sizing.qty,
                                                cfg.asset_class, limit_price,
                                                stop_loss=sizing.stop_price)
                if self._entry_was_refused(symbol, order):
                    return
                self._set_pending_fill(symbol, order, order_side, "entry", quote,
                                       limit_price)
            self._entry_leg_sessions[symbol] = et_session_date(datetime.now(timezone.utc))
        else:
            # Passive maker entry: a GTC limit resting below the close earns the
            # maker rate but may never fill. Mark the order resting BEFORE
            # submitting: crypto has no broker-side stop, so positions() must
            # preserve the intended stop for the whole placement-to-fill window,
            # and it keys that off the resting-order marker.
            limit_price = crypto_maker_limit_price(close, EXECUTION.crypto_maker_offset_bps)
            self.state.set_resting_order(symbol, boundary.isoformat())
            order = self.broker.limit_order(symbol, order_side, sizing.qty, cfg.asset_class,
                                            limit_price)
            if self._entry_was_refused(symbol, order):
                return
            self._set_pending_fill(symbol, order, order_side, "crypto_maker_placed",
                                   quote, limit_price)
            self._log_crypto_maker(symbol, "crypto_maker_placed", None, None)
            log.info("%s: resting maker entry placed limit=%.2f qty=%s (%.1fbps below close)",
                     symbol, limit_price, sizing.qty, EXECUTION.crypto_maker_offset_bps)
        self.state.clear_entry_refusal(symbol)

    def _log_risk_budget(self, symbol: str, sizing, price: float, equity: float,
                         realized_vol: float | None) -> None:
        """Report what the size about to be submitted risks against the budget
        its sizing intended — the configured fraction carrying the vol-target
        scalar, not a flat fraction. Never gates: the order goes out either
        way, and an over-budget entry is a sizing bug to be seen, not a trade to
        be silently changed."""
        check = self.risk.check_risk_budget(sizing.qty, price, sizing.stop_price,
                                            equity, realized_vol)
        log.log(logging.INFO if check.within_budget else logging.WARNING,
                "%s: risk %.3f%% of equity vs budget %.3f%% — %s", symbol,
                check.realized_pct * 100, check.budget_pct * 100, check.note)

    def _entry_was_refused(self, symbol: str, order) -> bool:
        """True when the broker declined the entry, after undoing the state the
        submission optimistically wrote.

        The stop meta and the resting/ladder marker are both persisted before the
        order goes out, so a crash in between still leaves something to
        reconcile. A refusal comes back as an Order rather than an exception, so
        without this those markers outlive an order that never existed: the stop
        meta shadows a position that was never opened, a stale resting marker
        suppresses the next bar's entry, and a ladder marker sends the escalation
        sweep chasing a dead order id."""
        refusal = rejection_reason(order)
        if refusal is None:
            return False
        log.error("%s: ENTRY REFUSED (%s) — nothing rests at the broker, "
                  "rolling back", symbol, refusal)
        self.state.clear_position_meta(symbol)
        self.state.clear_resting_order(symbol)
        self.state.clear_ladder_order(symbol)
        self.state.record_entry_refusal(symbol, refusal)
        return True

    def reconcile_crypto_resting(self, symbol: str, boundary: datetime,
                                 pos: PositionState | None) -> tuple[bool, bool]:
        """Reconcile a resting crypto entry placed on a prior cycle. Returns
        (entry_blocked, cancelled): entry_blocked when an unfilled order still
        rests for the current bar and a second entry must be suppressed;
        cancelled when this cycle cancelled a stale order, which the broker may
        still report open until the cancel settles."""
        tracked = self.state.resting_order(symbol)
        if tracked is None:
            return False, False
        # A dry run mutates nothing: it may run against the shared state.json a
        # real run persisted, and must never cancel orders or erase its stop.
        if pos is not None:
            # Filled (possibly partially). Cancel any unfilled remainder so it
            # can't rest GTC and grow the position past its sized qty. The stop
            # was persisted before submission and preserved while resting, so
            # supervision already protects the position with the correct stop.
            log.info("%s: resting maker entry filled at %.2f — supervision active",
                     symbol, pos.entry_price)
            if not self.dry_run:
                self.broker.cancel_open_orders(symbol)
                self._log_crypto_maker(symbol, "crypto_maker_filled",
                                       pos.entry_price, pos.qty)
                self.state.clear_pending_fill(symbol)
                self.state.clear_resting_order(symbol)
            return False, False
        if self.broker.has_open_order(symbol):
            if tracked == boundary.isoformat():
                log.info("%s: entry suppressed — maker order still resting this bar", symbol)
                # same bar, still resting: do not place a second order
                return True, False
            # Boundary advanced, still unfilled: the passive limit never filled.
            log.info("%s: resting maker entry unfilled at next bar — cancelling", symbol)
            if self.dry_run:
                return False, False
            self.broker.cancel_open_orders(symbol)
            self._log_crypto_maker(symbol, "crypto_maker_cancelled", None, None)
            self.state.clear_pending_fill(symbol)
            self.state.clear_resting_order(symbol)
            self.state.clear_position_meta(symbol)
            return False, True
        # No position and no open order: the order vanished (external cancel, or a
        # fill not yet reflected in this cycle's position snapshot). Drop only the
        # marker and never wipe the stop here — positions() reconciles it next
        # cycle, keeping it if a fill materialises and clearing it otherwise.
        if not self.dry_run:
            self.state.clear_pending_fill(symbol)
            self.state.clear_resting_order(symbol)
        return False, False

    def _decision_quote(self, symbol: str, asset_class: AssetClass):
        """Top-of-book at the decision instant, or None if the fetch fails.
        A quote failure must never block an order — measurement is best-effort."""
        try:
            return self.data.latest_quote(symbol, asset_class)
        except Exception:
            log.exception("%s: quote fetch failed — recording NaN decision mid", symbol)
            return None

    def _decision_price_trusted(self, symbol: str, close: float, quote) -> bool:
        """Guard against acting on a bar the live market disagrees with — a stale
        or mis-stamped bar fills nowhere near the price the signal saw. Fail-open
        with no usable quote: the resting protective stop is the real backstop and
        a quote outage must not freeze the engine."""
        if quote is None or math.isnan(quote.mid) or quote.mid <= 0:
            return True
        divergence = abs(close - quote.mid) / quote.mid
        if divergence <= EXECUTION.max_decision_divergence_pct:
            self.state.clear_data_divergence(symbol)
            return True
        detail = (f"decision close {close:.2f} is {divergence * 100:.1f}% from the "
                  f"live mid {quote.mid:.2f}")
        log.error("%s: DISTRUSTED decision bar — %s; skipping this cycle",
                  symbol, detail)
        self.state.record_data_divergence(symbol, detail)
        return False

    def apply_corporate_action_rescale(self, symbol: str, bars: pd.DataFrame) -> None:
        """Bars are split/dividend-adjusted; persisted stops, high-water marks and
        entry prices are raw broker prices. When the bar we last decided on comes
        back at a different price the whole series has been restated, so those
        levels are rescaled by the same factor rather than left a corporate action
        away from the market."""
        if bars.empty:
            return
        previous = self.state.last_bar(symbol)
        self.state.set_last_bar(symbol, bars.index[-1].isoformat(),
                                float(bars["close"].iloc[-1]))
        if previous is None or not previous.get("close", 0) > 0:
            return
        prior_ts = pd.Timestamp(previous["timestamp"])
        if prior_ts not in bars.index:
            return
        factor = float(bars.loc[prior_ts, "close"]) / previous["close"]
        if abs(factor - 1.0) <= CORPORATE_ACTION_RESCALE_PCT:
            return
        log.warning("%s: bar history restated %.4fx (corporate action) — rescaling "
                    "persisted stop, high-water mark and entry price", symbol, factor)
        self.state.rescale_price_levels(symbol, factor)

    def _within_spread_limit(self, symbol: str, quote) -> bool:
        """Spread-aware entry gate. Fail-open: with the gate disabled, no quote,
        or a NaN spread, the entry proceeds and the reason is logged."""
        limit = EXECUTION.max_spread_bps
        if limit is None:
            return True
        if quote is None or math.isnan(quote.spread_bps):
            log.warning("%s: spread gate could not evaluate (no quote) — allowing entry",
                        symbol)
            return True
        if quote.spread_bps > limit:
            log.info("%s: entry skipped — spread %.1fbps exceeds max %.1fbps",
                     symbol, quote.spread_bps, limit)
            return False
        return True

    def _set_pending_fill(self, symbol: str, order, side: str, event: str,
                          quote, limit_price,
                          position: PositionState | None = None) -> None:
        payload = {
            "order_id": str(order.id),
            "side": side,
            "event": event,
            "decision_mid": quote.mid if quote is not None else float("nan"),
            "decision_spread_bps": quote.spread_bps if quote is not None else float("nan"),
            "limit_price": limit_price,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        if position is not None:
            # trades.csv is booked from the reconciled fill, so the closing
            # position's identity has to outlive the state we drop at submission.
            payload["position_side"] = position.side
            payload["entry_price"] = position.entry_price
            payload["qty"] = position.qty
        self.state.set_pending_fill(symbol, payload)

    def _write_fill_row(self, symbol: str, event: str, side: str, decision_mid: float,
                        decision_spread_bps: float, limit_price, fill_price, fill_qty) -> None:
        slip = (slippage_bps(side, decision_mid, fill_price)
                if fill_price is not None else float("nan"))
        self.execution_log.record(
            datetime.now(timezone.utc).isoformat(), symbol, side, event, decision_mid,
            decision_spread_bps, limit_price, fill_price, fill_qty, slip)

    def _log_crypto_maker(self, symbol: str, event: str, fill_price, fill_qty) -> None:
        """Log a maker lifecycle event, reusing the pending-fill decision quote so
        a fill's slippage is measured against the mid at placement time."""
        payload = self.state.pending_fill(symbol) or {}
        self._write_fill_row(
            symbol, event, payload.get("side", "buy"),
            payload.get("decision_mid", float("nan")),
            payload.get("decision_spread_bps", float("nan")),
            payload.get("limit_price"), fill_price, fill_qty)

    def reconcile_pending_fills(self) -> None:
        """Resolve equity entry/exit fills captured at submission: on fill, write
        the measured-execution row and drop the marker; drop markers unresolved
        past PENDING_FILL_STALE (no row). Crypto maker fills are reconciled in
        reconcile_crypto_resting, so their markers are skipped here."""
        if self.dry_run:
            return
        now = datetime.now(timezone.utc)
        for symbol in list(self.state.data.get("pending_fills", {})):
            payload = self.state.pending_fill(symbol)
            if payload is None or payload.get("event") not in ("entry", "exit"):
                continue
            if now - datetime.fromisoformat(payload["submitted_at"]) > PENDING_FILL_STALE:
                log.warning("%s: pending %s fill (order %s) unresolved after %s — dropping",
                            symbol, payload["event"], payload["order_id"], PENDING_FILL_STALE)
                self.state.clear_pending_fill(symbol)
                continue
            try:
                fill_qty, fill_price, status = self.broker.get_order_fill(payload["order_id"])
            except Exception:
                log.exception("%s: could not query pending fill %s; will retry",
                              symbol, payload["order_id"])
                continue
            if status in TERMINAL_UNFILLED_STATUSES:
                # Nothing to measure, but the marker must go: while it stands the
                # engine treats the exit as in flight and holds off stop
                # supervision for the position the close never touched.
                log.warning("%s: %s order %s ended %s after filling %s", symbol,
                            payload["event"], payload["order_id"], status, fill_qty)
                self._book_partial_fill(symbol, payload, fill_price, fill_qty)
                if payload["event"] == "exit":
                    self.state.record_exit_failure(symbol, f"close order {status}")
                self.state.clear_pending_fill(symbol)
                continue
            if status != "filled":
                continue
            self._write_fill_row(
                symbol, payload["event"], payload["side"], payload["decision_mid"],
                payload["decision_spread_bps"], payload["limit_price"], fill_price, fill_qty)
            self._book_reconciled_exit(symbol, payload, fill_price, fill_qty)
            self.state.clear_pending_fill(symbol)

    def _book_partial_fill(self, symbol: str, payload: dict, fill_price,
                           fill_qty) -> None:
        """Shares that changed hands under an order that then ended without
        completing — cancelled by a retried close, or expired. They have left the
        position, so they must be measured and booked before the marker recording
        them is dropped or overwritten."""
        if not fill_qty or fill_price is None:
            return
        self._write_fill_row(symbol, payload["event"], payload["side"],
                             payload["decision_mid"], payload["decision_spread_bps"],
                             payload["limit_price"], fill_price, fill_qty)
        self._book_reconciled_exit(symbol, payload, fill_price, fill_qty)

    def _settle_superseded_close(self, symbol: str, new_order_id: str) -> None:
        """A retried close cancels the working one, whose partial fill is about to
        lose the only marker that records it."""
        payload = self.state.pending_fill(symbol)
        if (payload is None or payload.get("event") != "exit"
                or payload.get("order_id") == new_order_id):
            return
        try:
            fill_qty, fill_price, _ = self.broker.get_order_fill(payload["order_id"])
        except Exception:
            log.exception("%s: superseded close %s could not be measured — any "
                          "partial fill is UNBOOKED", symbol, payload["order_id"])
            return
        self._book_partial_fill(symbol, payload, fill_price, fill_qty)

    def _book_reconciled_exit(self, symbol: str, payload: dict, fill_price: float,
                              fill_qty: float) -> None:
        """trades.csv records the price we actually got out at. Booking it at
        submission time prints the decision bar's close instead, which is not the
        fill and puts the whole P&L record at odds with the broker's."""
        if payload["event"] != "exit" or "entry_price" not in payload:
            return
        gross = fill_qty * (fill_price - payload["entry_price"])
        pnl = gross if payload["position_side"] == "long" else -gross
        log.info("%s: EXIT filled %s qty=%s entry=%.2f fill=%.2f pnl=%.2f",
                 symbol, payload["position_side"], fill_qty, payload["entry_price"],
                 fill_price, pnl)
        self.trade_log.record(datetime.now(timezone.utc).isoformat(), symbol,
                              payload["position_side"], payload["entry_price"],
                              fill_price, round(pnl, 2), fill_qty)
        self._record_exit_outcome(symbol, pnl)

    def exit_position(self, cfg, pos: PositionState, close: float, reason: str,
                      atr_value: float | None = None) -> bool:
        """False when the broker refused the close: the position is still open and
        the bar must be retried on the next cycle."""
        symbol = cfg.symbol
        pnl = pos.pnl_at(close)
        log.info("%s: EXIT %s qty=%s entry=%.2f now=%.2f pnl~%.2f (%s)",
                 symbol, pos.side, pos.qty, pos.entry_price, close, pnl, reason)
        if self.dry_run:
            return True
        quote = self._decision_quote(symbol, cfg.asset_class)
        if not self._decision_price_trusted(symbol, close, quote):
            # Nothing was cancelled, so the resting protective stop still covers
            # the position while the bar it would have exited on is distrusted.
            return False
        try:
            order = self.broker.close_position(symbol)
        except Exception as exc:
            log.error("%s: EXIT FAILED (%s) — position still open, retrying next "
                      "cycle", symbol, exc)
            self.state.record_exit_failure(symbol, str(exc))
            self._restore_protective_stop(cfg, pos, atr_value)
            return False
        refusal = rejection_reason(order)
        if refusal is not None:
            log.error("%s: EXIT REFUSED (%s) — position still open, retrying next "
                      "cycle", symbol, refusal)
            self.state.record_exit_failure(symbol, refusal)
            self._restore_protective_stop(cfg, pos, atr_value)
            return False
        self._settle_superseded_close(symbol, str(order.id))
        self.state.clear_exit_failure(symbol)
        self.state.clear_position_meta(symbol)
        self.state.clear_last_seen_position(symbol)
        exit_side = "sell" if pos.side == "long" else "buy"
        self._set_pending_fill(symbol, order, exit_side, "exit", quote, None,
                               position=pos)
        return True

    def intended_stop(self, cfg, pos: PositionState,
                      atr_value: float | None = None) -> float:
        """Level the engine actually enforces: the persisted entry stop, tightened
        by the ATR trail for strategies that trail. A non-trailing strategy's stop
        must not follow the high-water mark."""
        mult = cfg.params.get("trail_atr_mult")
        if mult is None:
            return pos.stop_price
        if atr_value is None:
            atr_value = self._latest_atr(cfg.symbol)
        if atr_value is None:
            log.warning("%s: ATR unavailable — falling back to the entry stop %.2f",
                        cfg.symbol, pos.stop_price)
            return pos.stop_price
        return effective_stop(pos, atr_value, mult)

    def _restore_protective_stop(self, cfg, pos: PositionState,
                                 atr_value: float | None = None) -> None:
        """A failed close has already cancelled the resting stop, so put one back
        at once rather than waiting for the next reconciliation sweep."""
        if cfg.asset_class != AssetClass.US_EQUITY:
            return  # crypto stops are software-managed and the meta is preserved
        try:
            # A close that timed out waiting for its cancel never cancelled the
            # stop: the position is still covered, and a second stop is rejected.
            if self.broker.has_open_stop(cfg.symbol):
                log.warning("%s: stop still resting after the failed close — "
                            "leaving it in place", cfg.symbol)
                return
        except Exception:
            log.exception("%s: could not check for a resting stop; re-placing",
                          cfg.symbol)
        level = self.intended_stop(cfg, pos, atr_value)
        stop_side = "sell" if pos.side == "long" else "buy"
        try:
            order = self.broker.stop_order(cfg.symbol, stop_side, pos.qty, level,
                                           cfg.asset_class)
        except Exception:
            log.exception("%s: could not restore the protective stop after a "
                          "failed exit — POSITION IS NAKED", cfg.symbol)
            return
        refusal = rejection_reason(order)
        if refusal is not None:
            log.error("%s: protective stop was refused (%s) after a failed exit "
                      "— POSITION IS NAKED", cfg.symbol, refusal)
            return
        log.warning("%s: protective stop restored at %.2f after the failed exit",
                    cfg.symbol, level)

    def supervise_crypto_stops(self, positions: dict[str, PositionState]) -> None:
        """Runs every poll cycle, not just on bar close: crypto has no
        broker-side stop, so mid-bar crashes must be caught here."""
        for symbol, (cfg, _) in self.strategies.items():
            pos = positions.get(symbol)
            if cfg.asset_class != AssetClass.CRYPTO or pos is None:
                continue
            if self.exit_in_flight(symbol):
                # A working close leaves the position on a placeholder stop the
                # market is usually already through — closing again here would
                # double up on the same exit.
                continue
            price = self.data.latest_price(symbol, cfg.asset_class)
            if pos.side == "long" and price <= pos.stop_price:
                self.exit_position(cfg, pos, price, "software hard stop")

    def reconcile_equity_stops(self, positions: dict[str, PositionState]) -> None:
        """An equity position must always have a resting broker stop: re-place
        one that was rejected, cancelled, lost, or expired. The stop attached at
        entry is an OTO leg carrying the entry's DAY time-in-force, and Alpaca
        refuses to change that on an advanced order, so it expires at the close
        and is re-placed GTC here. Stops cannot trigger outside regular hours,
        so that gap exposes nothing."""
        for symbol, pos in positions.items():
            entry = self.strategies.get(symbol)
            if entry is None or entry[0].asset_class != AssetClass.US_EQUITY:
                continue
            # One symbol's rejection must not starve every other symbol's stop.
            try:
                if self.exit_in_flight(symbol):
                    continue  # never re-arm a stop under a working close
                if self.broker.has_open_stop(symbol):
                    continue
                level = self.intended_stop(entry[0], pos)
                if self._entry_leg_expired(symbol):
                    log.info("%s: entry stop expired at the close — re-placing "
                             "GTC at %.2f", symbol, level)
                else:
                    log.warning("%s: protective stop missing — re-placing at %.2f",
                                symbol, level)
                if not self.dry_run:
                    stop_side = "sell" if pos.side == "long" else "buy"
                    order = self.broker.stop_order(symbol, stop_side, pos.qty, level,
                                                   entry[0].asset_class)
                    refusal = rejection_reason(order)
                    if refusal is not None:
                        log.error("%s: stop re-placement REFUSED (%s) — position "
                                  "is unprotected until the next sweep",
                                  symbol, refusal)
            except Exception:
                log.exception("%s: protective stop could not be re-placed — "
                              "retrying next cycle", symbol)

    def _entry_leg_expired(self, symbol: str) -> bool:
        """Whether a missing stop is this session's entry leg expiring at the
        close. Excused once: the GTC replacement never expires, so losing that
        too is a real loss."""
        session = self._entry_leg_sessions.pop(symbol, None)
        if session != et_session_date(datetime.now(timezone.utc)):
            return False
        # Only the log level rides on this; the re-placement must not wait on it.
        try:
            return not self.broker.market_is_open()
        except Exception:
            return False

    def cancel_orphan_stops(self, positions: dict[str, PositionState]) -> None:
        """A protective stop outlives its position whenever the position leaves by
        another route — a manual flatten in the broker UI, or an exit that fills
        while a re-placed stop survives. Nothing expires it, and while it rests
        has_open_order blocks every future entry for that symbol. Crypto is
        exempt: its stops are software-supervised, so a resting crypto order is
        always a maker entry."""
        if self.dry_run:
            return
        for symbol, (cfg, _) in self.strategies.items():
            if cfg.asset_class != AssetClass.US_EQUITY or symbol in positions:
                continue
            if (self._entry_pending(symbol) or self.exit_in_flight(symbol)
                    or self.state.position_meta(symbol) is not None):
                continue
            # One symbol's failure must not starve the rest of the sweep.
            try:
                if not self.broker.has_open_stop(symbol):
                    continue
                log.warning("cancelling orphan stop for %s — no position", symbol)
                self.broker.cancel_open_orders(symbol)
            except Exception:
                log.exception("%s: orphan stop could not be cancelled — retrying "
                              "next cycle", symbol)

    def _log_ladder_partial(self, symbol: str, fill_price, fill_qty) -> None:
        """Record a passive partial fill before it is cancelled, reusing the
        placement decision quote so its favourable execution is measured."""
        payload = self.state.pending_fill(symbol) or {}
        self._write_fill_row(
            symbol, "entry", payload.get("side", "buy"),
            payload.get("decision_mid", float("nan")),
            payload.get("decision_spread_bps", float("nan")),
            payload.get("limit_price"), fill_price, fill_qty)

    def reconcile_equity_ladders(self, positions: dict[str, PositionState],
                                 session: tuple[datetime, datetime] | None,
                                 now_utc: datetime) -> None:
        """Escalate or clean up opt-in rest-then-cross ladders. Runs every cycle
        (not just on bar close) so the passive leg is escalated to a marketable
        cross on the very next cycle while still inside the near-close window, and
        a ladder left incomplete by a restart is cancelled once the window has
        closed. A dry run mutates nothing."""
        if self.dry_run:
            return
        for symbol in list(self.state.data.get("ladder_orders", {})):
            marker = self.state.ladder_order(symbol)
            if marker is None:
                continue
            entry = self.strategies.get(symbol)
            if entry is None:
                continue
            cfg = entry[0]
            try:
                filled_qty, fill_price, status = self.broker.get_order_fill(
                    marker["order_id"])
            except Exception:
                log.exception("%s: could not query ladder order %s; will retry",
                              symbol, marker["order_id"])
                continue
            if status == "filled":
                # Passive leg fully filled: its OTO stop rode the order. The
                # pending-fill flow logs the fill row — just retire the marker.
                log.info("%s: ladder passive leg filled — supervision active", symbol)
                self.state.clear_ladder_order(symbol)
                continue
            in_window = (session is not None and in_near_close_window(
                now_utc, session[1], EXECUTION.near_close_minutes))
            if not in_window:
                # Window closed with the ladder incomplete (e.g. restart): cancel
                # the stale passive leg and drop the marker. A partial position
                # left without a stop is re-covered by reconcile_equity_stops.
                log.warning("%s: ladder incomplete at window close — cancelling "
                            "passive leg", symbol)
                self.broker.cancel_open_orders(symbol)
                self.state.clear_ladder_order(symbol)
                continue
            # Still in window and unfilled/partial: escalate to a marketable cross
            # for the remaining qty, re-attaching the OTO stop leg. Any partial
            # fill keeps its own OTO stop; the safety-net sweep covers the total.
            remaining = marker["total_qty"] - filled_qty
            self.broker.cancel_open_orders(symbol)
            if filled_qty > 0:
                self._log_ladder_partial(symbol, fill_price, filled_qty)
            if remaining > 0:
                side = marker["side"]
                quote = self._decision_quote(symbol, cfg.asset_class)
                limit_price = equity_entry_limit_price(
                    marker["signal_close"], side, EXECUTION.limit_bps, quote)
                order = self.broker.limit_order(symbol, side, remaining,
                                                cfg.asset_class, limit_price,
                                                stop_loss=marker["stop_price"])
                self._set_pending_fill(symbol, order, side, "entry", quote, limit_price)
                log.info("%s: ladder escalated to marketable cross limit=%.2f qty=%s",
                         symbol, limit_price, remaining)
            self.state.clear_ladder_order(symbol)

    def process_commands(self) -> None:
        return self.ops.process_commands(HALT_FILE)

    def execute_command(self, command) -> str:
        return self.ops.execute_command(command, HALT_FILE)

    def maybe_send_briefings(self) -> None:
        return self.ops.maybe_send_briefings()

    def log_daily_pnl(self) -> None:
        """One daily_pnl.csv row per UTC day, measured against prev_equity — the
        equity stamped the last day a row was written, which may be several days
        back if the bot was down. day_start_equity is not a substitute: it reads
        as None the moment its day passes, and stays unset while the daily loss
        limit is off."""
        today = utc_day()
        if self.state.data.get("prev_day") == today:
            return
        equity = self.broker.account_equity()
        prev_equity = self.state.data.get("prev_equity")
        pnl = round(equity - prev_equity, 2) if prev_equity is not None else 0.0
        if self.state.data.get("prev_day") is not None:
            self.daily_log.record(today, equity, pnl)
        self.state.data["prev_day"] = today
        self.state.data["prev_equity"] = equity
        self.state.save()

    def current_session(self, now_utc: datetime) -> tuple[datetime, datetime] | None:
        """Cached once per ET trading day (session bounds don't move intraday), so
        a transient calendar failure is retried at most once/day, not every cycle.
        A failure degrades to None: equity window gating falls through to allow
        rather than propagating and suspending crypto stop supervision."""
        et_date = now_utc.astimezone(ZoneInfo("America/New_York")).date()
        if self._session_cache is not None and self._session_cache[0] == et_date:
            return self._session_cache[1]
        try:
            session = self.broker.current_session()
        except Exception:
            log.exception("calendar unavailable; equity window gating disabled this cycle")
            return None  # not cached: retry next cycle
        self._session_cache = (et_date, session)
        return session

    def equity_submission_allowed(self, cfg, now_utc: datetime,
                                  session: tuple[datetime, datetime] | None) -> bool:
        """Submission-timing gate for equities. Daily signals submit only inside
        the near-close window; every timeframe defers the opening auction. With
        no session bounds (calendar unavailable) fall through and allow."""
        if session is None:
            return True
        open_utc, close_utc = session
        if in_open_auction_window(now_utc, open_utc, EXECUTION.open_auction_minutes):
            return False
        if cfg.timeframe_minutes >= 1440:
            if not in_near_close_window(now_utc, close_utc, EXECUTION.near_close_minutes):
                return False
            offset = jitter_offset_minutes(
                cfg.symbol,
                now_utc.astimezone(ZoneInfo("America/New_York")).date().isoformat(),
                EXECUTION.submit_jitter_max_minutes, EXECUTION.near_close_minutes,
                EXECUTION.ladder_enabled)
            if offset > 0:
                window_start = close_utc - timedelta(minutes=EXECUTION.near_close_minutes)
                if now_utc < window_start + timedelta(minutes=offset):
                    return False
            return True
        return True

    def run_protective_sweeps(self) -> dict[str, PositionState]:
        """Everything that keeps open positions covered without taking on new
        risk: vanished-position reconciliation (inside positions()), crypto
        software stops, equity stop re-placement, orphan-stop cancellation, and
        pending-fill settlement."""
        positions = self.positions()
        self.supervise_crypto_stops(positions)
        self.reconcile_equity_stops(positions)
        self.cancel_orphan_stops(positions)
        self.reconcile_pending_fills()
        return positions

    def run_cycle(self, force: bool = False) -> None:
        """force=True evaluates every instrument immediately, ignoring bar
        boundaries and market hours — used by dry runs. Only safe because
        dry runs never submit orders."""
        # Before the halt gate, so the one command that lifts a halt is not shut
        # out by the halt itself, and a halt command takes effect this cycle.
        self.process_commands()
        if halt_tripped(HALT_FILE):
            log.warning("HALT file present — not trading. Remove '%s' after review "
                        "to resume.", HALT_FILE)
            # Halted means no new risk, not no protection: stops, fills and
            # vanished positions still reconcile, and the briefings that carry the
            # halt alert still go out — this is the day they matter most.
            try:
                positions = self.run_protective_sweeps()
                # session=None: a resting entry ladder is cancelled, never escalated.
                self.reconcile_equity_ladders(positions, None,
                                              datetime.now(timezone.utc))
            except Exception:
                log.exception("halted: protective sweep failed; briefings still sent")
            self.maybe_send_briefings()
            return
        equity = self.broker.account_equity()
        if self.check_circuit_breaker(equity):
            return
        # Ahead of the daily-loss gate: the day the limit trips is the day whose
        # P&L row matters most, and the gate returns before anything below it.
        self.log_daily_pnl()
        if self.check_daily_loss_limit(equity):
            # Halted, not flattened: the next cycle takes the halt branch and
            # its sweeps keep every open position covered.
            return

        positions = self.run_protective_sweeps()
        self.maybe_send_briefings()

        now = datetime.now(timezone.utc)
        if self.last_heartbeat is None or (now - self.last_heartbeat).total_seconds() >= 3600:
            log.info("heartbeat: equity=%.2f positions=%s",
                     equity, ", ".join(positions) if positions else "none")
            self.last_heartbeat = now
        market_open = self.broker.market_is_open()
        session = self.current_session(now) if market_open else None
        self.reconcile_equity_ladders(positions, session, now)
        for symbol, (cfg, _) in self.strategies.items():
            bar_key = decision_bar_key(cfg, now)
            is_equity = cfg.asset_class == AssetClass.US_EQUITY
            if not force:
                if self.last_boundary.get(symbol) == bar_key:
                    continue
                if is_equity and not market_open:
                    continue
                # Skip WITHOUT marking the bar evaluated, so a later cycle inside
                # the window still sees this bar.
                if is_equity and not self.equity_submission_allowed(cfg, now, session):
                    continue
            elif is_equity:
                allowed = market_open and self.equity_submission_allowed(cfg, now, session)
                log.info("%s: dry run — submission window %s", symbol,
                         "OPEN" if allowed else "CLOSED")
            # Marked only once processing succeeds: a failed exit must retry on
            # the next cycle, not on the next bar boundary (a day later).
            try:
                evaluated = self.process_instrument(symbol, now_utc=now,
                                                    session=session)
            except Exception:
                log.exception("%s: processing failed; continuing with the rest of "
                              "the cycle", symbol)
                continue
            if evaluated is not False:
                self.last_boundary[symbol] = bar_key


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="compute and log signals without submitting orders")
    parser.add_argument("--once", action="store_true",
                        help="run a single cycle and exit (useful for testing)")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    paper = os.getenv("ALPACA_PAPER", "true").lower() != "false"
    if not api_key or not secret:
        print("Missing Alpaca keys: copy .env.example to .env and fill them in.")
        return 1
    if not paper:
        log.warning("LIVE TRADING MODE — real money at risk")

    engine = Engine(
        broker=Broker(api_key, secret, paper=paper),
        data=MarketData(api_key, secret, feed=MARKET_DATA_FEED),
        dry_run=args.dry_run,
    )
    if args.dry_run:
        log.info("dry run: no orders will be submitted")
    engine.check_startup_windows()
    warn_unvalidated_instruments()

    first_cycle = True
    backoff = POLL_SECONDS
    while True:
        try:
            engine.run_cycle(force=args.dry_run and first_cycle)
            first_cycle = False
            backoff = POLL_SECONDS
        except KeyboardInterrupt:
            log.info("stopped by user")
            return 0
        except Exception:
            log.exception("cycle failed; retrying in %ds", backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
            continue
        if args.once:
            return 0
        time.sleep(POLL_SECONDS - datetime.now(timezone.utc).second % POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
