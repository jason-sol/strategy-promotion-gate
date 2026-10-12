"""Daily email briefings.

Composes the morning and evening reports and the alert section they carry.
Sent automatically by the main loop at the configured local hours; can also be
run manually: `python -m bot.briefing morning|evening`. Delivery is bot.mail's.
"""

import csv
import logging
import math
import os
import sys
from datetime import datetime, timedelta, timezone

from bot.data import decision_bars
from bot.halt import halt_tripped, read_halt
from bot.portfolio import (
    SLIPPAGE_MIN_SAMPLE,
    assumed_slippage_bps,
    slippage_summary,
)
from bot.risk_manager import realized_risk_pct, within_risk_budget
from bot.state import parse_utc
from bot.strategies import build_strategy
from bot.strategies.base import PositionState, effective_stop, resolve_stop_and_hwm
from config import DAILY_PNL_CSV, EXECUTION_CSV, PROTECTIONS, RISK, TRADES_CSV

log = logging.getLogger(__name__)


def due_briefing(now_local: datetime, hour: int, last_sent: str | None) -> bool:
    """One send per local day, any time from `hour` onwards."""
    return now_local.hour >= hour and last_sent != now_local.date().isoformat()


def load_trades_since(cutoff: datetime, path: str = TRADES_CSV) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        try:
            ts = datetime.fromisoformat(row["timestamp"])
        except (KeyError, ValueError):
            continue
        if ts >= cutoff:
            out.append(row)
    return out


def latest_daily_pnl(path: str = DAILY_PNL_CSV) -> tuple[float | None, float | None]:
    """(pnl, equity) of the most recent logged day."""
    if not os.path.exists(path):
        return None, None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None
    return float(rows[-1]["pnl"]), float(rows[-1]["equity"])


def _win_rate(trades: list[dict]) -> tuple[int, float]:
    pnls = [float(t["pnl"]) for t in trades if t.get("pnl")]
    if not pnls:
        return 0, 0.0
    return len(pnls), sum(1 for p in pnls if p > 0) / len(pnls)


def _exit_failure_line(symbol: str, failure: dict) -> str:
    return (f"{symbol}: exit FAILED {failure['count']}x since "
            f"{failure['since'][:10]} — {failure['error']}")


def _entry_refusal_line(symbol: str, refusal: dict) -> str:
    return (f"{symbol}: entry REFUSED {refusal['count']}x since "
            f"{refusal['since'][:10]} — not trading this symbol "
            f"({refusal['error']})")


def _cancel_failure_line(symbol: str, failure: dict) -> str:
    return (f"{symbol}: cancel FAILED {failure['count']}x since "
            f"{failure['since'][:10]} — an order may still rest "
            f"({failure['error']})")


def _data_divergence_line(symbol: str, divergence: dict) -> str:
    return (f"{symbol}: decision bar DISTRUSTED {divergence['count']}x "
            f"since {divergence['since'][:10]} — {divergence['detail']}")


def _short_window_line(symbol: str, window: dict) -> str:
    return (f"{symbol}: SHORT bar window {window['count']}x since "
            f"{window['since'][:10]} — {window['detail']}; no decision "
            f"is being taken for this symbol")


# Ordering is the order the alerts are read in, so the most actionable faults
# come first.
_ALERT_CHANNELS = [
    ("exit_failures", _exit_failure_line),
    ("entry_refusals", _entry_refusal_line),
    ("cancel_failures", _cancel_failure_line),
    ("data_divergences", _data_divergence_line),
    ("short_windows", _short_window_line),
]


def collect_alerts(broker, state, positions: dict[str, PositionState] | None = None,
                   halt_path: str | None = None,
                   now: datetime | None = None) -> list[str]:
    """Things that need a human: exits the broker refused, equity positions with
    no resting stop, and an active halt. Each broker probe is independently
    guarded so a single failure never costs the whole section."""
    from config import COMMANDS, HALT_FILE, INSTRUMENTS, AssetClass

    halt_path = halt_path or HALT_FILE
    now = now or datetime.now(timezone.utc)
    alerts = []

    for channel, formatter in _ALERT_CHANNELS:
        records = (state.data.get(channel) or {}) if state is not None else {}
        for symbol, record in sorted(records.items()):
            alerts.append(formatter(symbol, record))

    cooldowns = (state.data.get("cooldowns") or {}) if state is not None else {}
    for symbol in sorted(cooldowns):
        record = state.cooldown(symbol)
        if record is None:
            if state.cooldown_unusable(symbol):
                alerts.append(f"{symbol}: cooldown record unusable — pause NOT "
                              f"enforced; fix or remove it in state.json")
            # Otherwise lapsed by time, which is the expiry working as intended.
            continue
        alerts.append(f"{symbol}: entries PAUSED {record['days']}d after "
                      f"{record['losses']} consecutive losses, until "
                      f"{record['until'][:10]}")

    health = state.command_poll_health() if state is not None else None
    if health is not None:
        since = parse_utc(health.get("since"))
        down_for = (now - since).total_seconds() if since else None
        # An unreadable stamp counts as a long outage rather than none: a kill
        # switch that cannot be reached is the last thing to go unsaid.
        if down_for is None or down_for >= COMMANDS.down_alert_seconds:
            elapsed = f"{down_for / 60:.0f}m" if down_for else "an unknown time"
            alerts.append(
                f"command channel DOWN {elapsed} ({health['count']} failed polls, "
                f"last: {health['error']}) — REMOTE HALT UNAVAILABLE; "
                f"touch {halt_path} on the VPS to stop trading")

    budget = COMMANDS.refusal_replies_per_day
    if state is not None and state.refusal_budget_spent(budget):
        alerts.append(f"command channel: refusal-reply budget spent today "
                      f"({budget}/{budget}) — unsigned CMD mail is being "
                      f"dropped without a reply until the UTC day rolls over")

    if positions is None:
        try:
            positions = broker.get_positions()
        except Exception:
            log.warning("positions unavailable for alerts")
            positions = {}
    equities = {c.symbol for c in INSTRUMENTS
                if c.asset_class == AssetClass.US_EQUITY}
    for symbol in sorted(positions):
        if symbol not in equities:
            continue  # crypto stops are software-managed, not broker-side
        try:
            if not broker.has_open_stop(symbol):
                alerts.append(f"{symbol}: NO resting protective stop at the broker")
        except Exception:
            log.warning("%s: stop check unavailable for alerts", symbol)

    if halt_tripped(halt_path):
        reason = (read_halt(halt_path) or {}).get("reason") or "reason unavailable"
        alerts.append(f"HALT file present — bot is NOT trading ({reason})")
    return alerts


COST_WINDOW_DAYS = 30


def _cost_detail(stat) -> str:
    if not stat.count:
        return f"0 measurable ({stat.dropped} booked, no decision quote)"
    parts = [f"n={stat.count:<4}median {stat.median_bps:+6.1f}bps",
             f"spread {stat.median_spread_bps:6.1f}bps"]
    assumed = assumed_slippage_bps(stat.symbol)
    if assumed is not None:
        parts.append(f"assumed {assumed:.1f}bps")
        if stat.median_bps > assumed:
            # A thin sample's median is noise; flagging it daily trains the
            # reader to ignore the flag.
            parts.append("OVER" if stat.count >= SLIPPAGE_MIN_SAMPLE
                         else f"OVER (n<{SLIPPAGE_MIN_SAMPLE})")
    if stat.dropped:
        parts.append(f"+{stat.dropped} unmeasurable")
    return "  ".join(parts)


def execution_cost_lines(path: str = EXECUTION_CSV) -> list[str]:
    """Measured slippage against the assumed cost over a trailing window, per
    symbol and fill kind. Empty when nothing has filled in the window, so the
    section drops out of the email."""
    since = datetime.now(timezone.utc) - timedelta(days=COST_WINDOW_DAYS)
    stats = slippage_summary(path, since=since)
    if not stats:
        return []
    lines = [f"Execution costs, last {COST_WINDOW_DAYS}d (measured vs assumed):"]
    lines += [f"  {s.symbol:<8}{s.kind:<6}{_cost_detail(s)}" for s in stats]
    return lines


def _alert_lines(alerts: list[str] | None) -> list[str]:
    if not alerts:
        return []
    return ["Alerts:"] + [f"  ! {a}" for a in alerts] + [""]


def _subject_prefix(alerts: list[str] | None) -> str:
    return f"[ALERT x{len(alerts)}] " if alerts else ""


# Each risk flag fires at half its configured breaker, so the warning arrives
# before the halt rather than alongside it. A flag that fires on every red day
# is one the reader learns to skip.
DAY_LOSS_FLAG_FRACTION = 0.5
DRAWDOWN_FLAG_FRACTION = 0.5


def _drawdown_halt_note(max_drawdown: float) -> str:
    """Zero is not a disabled breaker — it halts on any drawdown at all — and
    "halt at 0%" reads as off. A fractional threshold keeps its fraction: a
    rounded percent puts the halt somewhere it is not."""
    if max_drawdown <= 0:
        return "halts on any drawdown"
    return f"halt at {max_drawdown * 100:g}%"


def build_morning_report(equity: float, positions: dict[str, PositionState],
                         marks: dict[str, float], trades_7d: list[dict],
                         yesterday_pnl: float | None, peak_equity: float,
                         halted: bool, alerts: list[str] | None = None,
                         day_start_equity: float | None = None) -> tuple[str, str]:
    lines = _alert_lines(alerts) + [f"Equity: ${equity:,.2f}"]
    if yesterday_pnl is not None:
        lines.append(f"Yesterday's P&L: {yesterday_pnl:+,.2f}")

    if positions:
        lines.append("\nOpen positions:")
        for symbol, pos in sorted(positions.items()):
            mark = marks.get(symbol, pos.entry_price)
            risk = realized_risk_pct(pos.qty, pos.entry_price, pos.stop_price, equity)
            lines.append(f"  {symbol} {pos.side} {pos.qty} @ {pos.entry_price:,.2f} "
                         f"(now {mark:,.2f}, unrealized {pos.pnl_at(mark):+,.2f}, "
                         f"stop {pos.stop_price:,.2f}, risk {risk * 100:.2f}%)")
    else:
        lines.append("\nNo open positions.")

    count, rate = _win_rate(trades_7d)
    lines.append(f"\nLast 7 days: {count} closed trades, win rate {rate:.0%}")

    flags = []
    if peak_equity > 0:
        dd = 1 - equity / peak_equity
        if dd > RISK.max_drawdown * DRAWDOWN_FLAG_FRACTION:
            flags.append(f"drawdown {dd * 100:.1f}% from peak "
                         f"({_drawdown_halt_note(RISK.max_drawdown)})")
    if day_start_equity and PROTECTIONS.daily_loss_limit_pct > 0:
        day_loss = 1 - equity / day_start_equity
        if day_loss > PROTECTIONS.daily_loss_limit_pct * DAY_LOSS_FLAG_FRACTION:
            flags.append(f"down {day_loss * 100:.1f}% on the day "
                         f"(halt at {PROTECTIONS.daily_loss_limit_pct * 100:g}%)")
    # Vol targeting scales per-trade risk by a bounded scalar, so the ceiling no
    # legitimate sizing can clear is the configured fraction times that bound.
    ceiling = RISK.risk_per_trade * RISK.vol_scalar_max
    for symbol, pos in sorted(positions.items()):
        risk = realized_risk_pct(pos.qty, pos.entry_price, pos.stop_price, equity)
        if not within_risk_budget(risk, ceiling):
            flags.append(f"{symbol} risks {risk * 100:.2f}% of equity at its stop "
                         f"(ceiling {ceiling * 100:g}%)")
    if halted:
        flags.append("HALT file present — bot is NOT trading")
    if flags:
        lines.append("\nRisk flags:\n  " + "\n  ".join(flags))

    subject = f"{_subject_prefix(alerts)}Morning briefing — equity ${equity:,.0f}"
    return subject, "\n".join(lines)


def build_evening_report(equity: float, trades_today: list[dict],
                         prev_equity: float | None,
                         alerts: list[str] | None = None,
                         cost_lines: list[str] | None = None) -> tuple[str, str]:
    lines = _alert_lines(alerts) + [f"Equity: ${equity:,.2f}"]
    if prev_equity:
        lines.append(f"Change since yesterday: {equity - prev_equity:+,.2f}")

    if trades_today:
        pnls = [(t, float(t["pnl"])) for t in trades_today if t.get("pnl")]
        total = sum(p for _, p in pnls)
        lines.append(f"\n{len(trades_today)} trade(s) closed today, "
                     f"realized P&L {total:+,.2f}:")
        for t, p in pnls:
            lines.append(f"  {t['instrument']} {t['direction']}: {p:+,.2f}")
        if len(pnls) > 1:
            best = max(pnls, key=lambda x: x[1])
            worst = min(pnls, key=lambda x: x[1])
            lines.append(f"Best: {best[0]['instrument']} {best[1]:+,.2f} | "
                         f"Worst: {worst[0]['instrument']} {worst[1]:+,.2f}")
    else:
        lines.append("\nNo trades closed today.")

    if cost_lines:
        lines.append("\n" + "\n".join(cost_lines))

    subject = f"{_subject_prefix(alerts)}Evening report — equity ${equity:,.0f}"
    return subject, "\n".join(lines)


def _latest_atr(data, cfg) -> float:
    strategy = build_strategy(cfg)
    bars = data.get_recent_bars(cfg.symbol, cfg.asset_class, cfg.timeframe_minutes,
                                bars=strategy.min_bars + 60)
    # Same bar selection the engine decides on, so the briefed stop is the one
    # actually enforced rather than one derived from a half-formed bar.
    bars = decision_bars(bars, cfg.asset_class, cfg.timeframe_minutes,
                         datetime.now(timezone.utc))
    value = float(strategy.prepare(bars)["atr"].iloc[-1])
    if math.isnan(value):
        raise ValueError(f"{cfg.symbol}: ATR is NaN")
    return value


def _overlay_effective_stops(positions: dict[str, PositionState], state, data,
                             cfg_by_symbol: dict) -> None:
    """Replace the broker's placeholder stop with the effective stop the engine
    actually enforces. Any ATR failure degrades to the persisted entry stop so
    the briefing still sends."""
    for symbol, pos in positions.items():
        meta = state.position_meta(symbol) if state is not None else None
        pos.stop_price, pos.high_water_mark = resolve_stop_and_hwm(pos, meta)
        cfg = cfg_by_symbol.get(symbol)
        mult = cfg.params.get("trail_atr_mult") if cfg else None
        if mult is None:
            continue
        try:
            atr_value = _latest_atr(data, cfg)
        except Exception:
            log.warning("%s: ATR unavailable for briefing; showing entry stop", symbol)
            continue
        pos.stop_price = effective_stop(pos, atr_value, mult)


def compose(kind: str, broker, data, peak_equity: float | None = None,
            state=None) -> tuple[str, str]:
    """Build a briefing from live broker state. Shared by the engine loop and
    the manual CLI."""
    from config import HALT_FILE, INSTRUMENTS

    equity = broker.account_equity()
    now = datetime.now(timezone.utc)
    positions = broker.get_positions()
    try:
        alerts = collect_alerts(broker, state, positions, HALT_FILE)
    except Exception:
        log.exception("alert collection failed; briefing sent without alerts")
        alerts = []
    if kind == "morning":
        cfg_by_symbol = {c.symbol: c for c in INSTRUMENTS}
        marks = {s: data.latest_price(s, cfg_by_symbol[s].asset_class)
                 for s in positions if s in cfg_by_symbol}
        _overlay_effective_stops(positions, state, data, cfg_by_symbol)
        yesterday_pnl, _ = latest_daily_pnl()
        return build_morning_report(
            equity, positions, marks, load_trades_since(now - timedelta(days=7)),
            yesterday_pnl, peak_equity or equity, halt_tripped(HALT_FILE), alerts,
            state.day_start_equity() if state is not None else None)
    _, prev_equity = latest_daily_pnl()
    try:
        cost_lines = execution_cost_lines()
    except Exception:
        log.exception("execution cost summary failed; briefing sent without it")
        cost_lines = []
    return build_evening_report(equity, load_trades_since(now - timedelta(days=1)),
                                prev_equity, alerts, cost_lines)


def main(kind: str) -> int:
    from dotenv import load_dotenv
    load_dotenv()
    from bot.broker import Broker
    from bot.data import MarketData
    from bot.mail import send_gmail
    from bot.state import StateStore
    from config import MARKET_DATA_FEED, STATE_FILE

    broker = Broker(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"),
                    paper=os.getenv("ALPACA_PAPER", "true").lower() != "false")
    data = MarketData(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"),
                      feed=MARKET_DATA_FEED)
    store = StateStore(STATE_FILE)
    subject, body = compose(kind, broker, data,
                            store.data.get("peak_equity"), state=store)
    send_gmail(subject, body)
    print(f"sent: {subject}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "evening"))
