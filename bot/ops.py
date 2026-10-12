"""Operator-facing surface of the trading engine.

What belongs here: work an operator or a wall clock drives rather than a bar —
the startup bar-window report, the email command channel, and the
morning/evening briefings. These read the engine's broker, state, data and
strategies, and reach into the trading loop at exactly one point,
Engine.exit_position, for a force-exit.

What does not: anything on the per-bar decision path. A method that has to know
about bar boundaries, signals, sizing or stop placement stays in bot.main.

The halt sentinel path arrives as an argument rather than an import so the
engine stays the single authority on which file it is bound to.
"""

import logging
from datetime import datetime, timezone

from zoneinfo import ZoneInfo

from bot import briefing, commands, mail
from bot.data import decision_bars
from bot.halt import clear_halt, read_halt, trip_halt
from bot.state import parse_utc
from config import BRIEFING, INSTRUMENTS

log = logging.getLogger("bot")

# Bars of headroom over min_bars the live window is expected to carry. Additive
# because the fetch is additive (min_bars + the engine's fetch cushion). It
# catches supply skating close to the per-decision line; whether min_bars is
# itself enough warm-up for the indicators is the drift diagnostic's question,
# not this one's.
STARTUP_WINDOW_CUSHION_BARS = 5


class Ops:
    """Holds the engine rather than its parts, so a broker, data client or
    strategy map rebound after construction is picked up on the next read."""

    def __init__(self, engine):
        self.engine = engine

    def check_startup_windows(self, fetch_extra_bars: int) -> None:
        """Report every instrument whose live bar window sits within the cushion
        of min_bars, once per start. One symbol's data failure must not cost the
        rest of the lineup its check.

        Deliberately log-only, as the first rung of a ladder: a cushion
        shortfall is an operator heads-up in the deploy logs, and the moment
        supply actually breaches min_bars the per-decision path records the
        short_windows alert, which reaches the briefing email.

        Counts completed bars only (no keep_forming), so near an equity close it
        can read one bar under what process_instrument decides on — the
        conservative direction for a warning."""
        now = datetime.now(timezone.utc)
        for symbol, (cfg, strategy) in self.engine.strategies.items():
            try:
                fetched = self.engine.data.get_recent_bars(
                    symbol, cfg.asset_class, cfg.timeframe_minutes,
                    bars=strategy.min_bars + fetch_extra_bars)
                available = len(decision_bars(fetched, cfg.asset_class,
                                              cfg.timeframe_minutes, now))
            except Exception:
                log.exception("%s: bar window could not be checked at startup", symbol)
                continue
            required = strategy.min_bars + STARTUP_WINDOW_CUSHION_BARS
            if available < required:
                log.error("%s: SHORT bar window — %d bars available, %d short of "
                          "the %d wanted (min_bars %d plus %d cushion); indicators "
                          "may be NaN or degraded live where the backtest sees "
                          "full history", symbol, available, required - available,
                          required, strategy.min_bars, STARTUP_WINDOW_CUSHION_BARS)
            else:
                log.info("%s: bar window %d bars (min_bars %d)", symbol, available,
                         strategy.min_bars)

    def process_commands(self, halt_file: str) -> None:
        """Execute the email commands waiting in the mailbox. Every failure mode
        — an unreachable mailbox, a command that blows up, a reply that cannot be
        sent — is contained here: the trading cycle must run whether or not the
        operator has a channel into it."""
        if self.engine.dry_run:
            # A dry run shares the live bot's mailbox; consuming a command here
            # would mark it seen and the live bot would never act on it.
            return
        try:
            pending = commands.poll(self.engine.state)
        except Exception:
            log.exception("command poll failed; no commands this cycle")
            return
        for command in pending:
            try:
                verdict = self.execute_command(command, halt_file)
            except Exception as exc:
                log.exception("command failed: %s", command.subject)
                verdict = f"failed: {exc}"
            try:
                commands.send_reply(command.subject, command.sender, verdict)
            except Exception:
                log.exception("command reply failed: %s", verdict)

    def execute_command(self, command, halt_file: str) -> str:
        """Run one verified command and return the verdict for its reply."""
        if command.verb == "halt":
            reason = command.args or "no reason given"
            trip_halt(f"email command: {reason}", halt_file)
            log.warning("HALT tripped by email command: %s", reason)
            return f"HALT tripped — {reason}"
        if command.verb == "clear-halt":
            if not clear_halt(halt_file):
                return "no HALT was set — nothing to clear"
            log.warning("HALT cleared by email command")
            return "HALT cleared — trading resumes in this cycle"
        if command.verb == "force-exit":
            return self._force_exit(command.args.strip().upper())
        if command.verb == "clear-cooldown":
            return self._clear_cooldown(command.args.strip().upper())
        if command.verb == "status":
            return self._status_report(halt_file)
        # A verb the parser accepts but nothing here implements must refuse
        # rather than fall through to whatever branch happens to be last.
        return f"REFUSED — no handler for verb {command.verb!r}"

    def _force_exit(self, symbol: str) -> str:
        """Close a position on demand through the same path a strategy exit
        takes, so it inherits the rejection handling, stop restore and exit
        alerting. Deliberately ungated by the halt: exits always are."""
        engine = self.engine
        entry = engine.strategies.get(symbol)
        if entry is None:
            return f"REFUSED force-exit {symbol} — not in the configured lineup"
        if engine.exit_in_flight(symbol):
            # A retry carries a fresh signature, so the replay ledger rightly
            # lets it through; the working close is what makes it a no-op.
            return f"force-exit {symbol} already working — not resubmitting"
        cfg = entry[0]
        pos = engine.positions().get(symbol)
        if pos is None:
            return f"REFUSED force-exit {symbol} — no open position"
        price = engine.data.latest_price(symbol, cfg.asset_class)
        if engine.exit_position(cfg, pos, price, "email command: force-exit"):
            return f"force-exit {symbol} submitted at {price:,.2f}"
        return f"force-exit {symbol} FAILED — position still open, see the logs"

    def _clear_cooldown(self, symbol: str) -> str:
        """Lift one symbol's post-losing-streak entry pause. The streak goes
        with the pause: a cooldown lifted on request that re-imposes itself on
        the next single loss is not what was asked for."""
        state = self.engine.state
        if symbol not in self.engine.strategies:
            return f"REFUSED clear-cooldown {symbol} — not in the configured lineup"
        record = state.cooldown(symbol)
        unusable = state.cooldown_unusable(symbol)
        streak = state.losing_streak(symbol)
        removed = state.clear_cooldown(symbol)
        state.set_losing_streak(symbol, 0)
        if not removed and not streak:
            return f"no cooldown on {symbol} — nothing to clear"
        if record is not None:
            until = parse_utc(record["until"])
            done = f"pause until {until.date().isoformat()} lifted"
        elif unusable:
            done = "unusable cooldown record removed (entries were not paused)"
        elif removed:
            done = "expired cooldown record removed (entries were not paused)"
        else:
            done = "no cooldown was in force"
        if streak:
            done += f"; losing streak {streak} reset"
        log.warning("%s: cooldown cleared by email command — %s", symbol, done)
        return f"clear-cooldown {symbol} — {done}"

    def _status_report(self, halt_file: str) -> str:
        engine = self.engine
        lines = [f"Equity: ${engine.broker.account_equity():,.2f}"]
        positions = engine.positions()
        if positions:
            lines.append("Open positions:")
            for symbol, pos in sorted(positions.items()):
                entry = engine.strategies.get(symbol)
                stop = (engine.intended_stop(entry[0], pos) if entry is not None
                        else pos.stop_price)
                lines.append(f"  {symbol} {pos.side} {pos.qty} @ "
                             f"{pos.entry_price:,.2f} stop {stop:,.2f}")
        else:
            lines.append("No open positions.")
        halt = read_halt(halt_file)
        lines.append(f"HALT: {halt.get('reason', 'reason unavailable')}"
                     if halt is not None else "HALT: clear")
        # "Is anything wrong" is the question status is asked to answer, and the
        # briefing already knows how to answer it.
        try:
            alerts = briefing.collect_alerts(engine.broker, engine.state, positions,
                                             halt_file)
        except Exception:
            log.exception("alert collection failed; status sent without alerts")
            alerts = []
        if alerts:
            lines += ["Alerts:"] + [f"  ! {a}" for a in alerts]
        return "\n".join(lines)

    def maybe_send_briefings(self) -> None:
        """Morning/evening email briefings at the configured local hours.
        Silently inactive until Gmail credentials are configured."""
        engine = self.engine
        if not mail.mail_configured():
            return
        now_local = datetime.now(ZoneInfo(BRIEFING.timezone))
        today = now_local.date().isoformat()
        for kind, hour, state_key in (
            ("morning", BRIEFING.morning_hour, "last_morning_briefing"),
            ("evening", BRIEFING.evening_hour, "last_evening_briefing"),
        ):
            if not briefing.due_briefing(now_local, hour,
                                         engine.state.data.get(state_key)):
                continue
            try:
                subject, body = briefing.compose(
                    kind, engine.broker, engine.data,
                    engine.state.data.get("peak_equity"), state=engine.state)
                if not engine.dry_run:
                    mail.send_gmail(subject, body)
                log.info("briefing sent: %s", subject)
            except Exception:
                log.exception("%s briefing failed; will retry next cycle", kind)
                continue
            engine.state.data[state_key] = today
            engine.state.save()


def warn_unvalidated_instruments(instruments=INSTRUMENTS) -> list[str]:
    """Name every enabled instrument with no recorded validation verdict, once
    per start, and return them.

    Deliberately loud and never fatal: running an instrument whose research said
    REJECT — or which the harness never covered — is a decision the owner is
    allowed to make, but not one that should be invisible in the deploy logs. An
    override_reason changes what the warning says rather than silencing it,
    because the missing verdict is the fact worth surfacing either way. Nothing
    reads either field's content; presence is the whole signal."""
    missing = []
    for cfg in instruments:
        if not cfg.enabled or (cfg.validation_ref or "").strip():
            continue
        missing.append(cfg.symbol)
        override = (cfg.override_reason or "").strip()
        if override:
            log.warning("%s: enabled by recorded owner override: %s",
                        cfg.symbol, override)
        else:
            log.warning("%s: enabled without a recorded validation verdict — set "
                        "InstrumentConfig.validation_ref to the validation result "
                        "that justifies it, or override_reason to record running "
                        "without one", cfg.symbol)
    return missing
