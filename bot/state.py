"""Bot-side state Alpaca cannot hold for us: stop prices, trailing high-water
marks, and peak equity. Survives restarts via a JSON file."""

import json
import logging
import os
from datetime import datetime, timezone

log = logging.getLogger(__name__)


def utc_day() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def parse_utc(value) -> datetime | None:
    """Timezone-aware reading of a persisted timestamp, or None when it is
    absent or unparseable. A bare timestamp is read as UTC: everything the bot
    writes is UTC, and a hand-edited one is meant that way too."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


class StateStore:
    def __init__(self, path: str):
        self.path = path
        # In-memory only: reads must not write, so an undatable cooldown is
        # reported once per process instead of being deleted to silence it.
        self._unusable_cooldowns: set[str] = set()
        self.data: dict = {"positions": {}, "resting_orders": {}, "ladder_orders": {},
                           "pending_fills": {}, "exit_failures": {},
                           "data_divergences": {}, "cancel_failures": {},
                           "entry_refusals": {}, "short_windows": {},
                           "last_bars": {}, "processed_commands": [],
                           "refusal_replies": {}, "losing_streaks": {},
                           "command_poll_failure": None,
                           "cooldowns": {}, "day_start_equity": {},
                           "daily_loss_tripped": None,
                           "last_seen_positions": {}, "peak_equity": 0.0,
                           "prev_day": None, "prev_equity": None}
        if os.path.exists(path):
            with open(path) as f:
                self.data.update(json.load(f))

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)

    def position_meta(self, symbol: str) -> dict | None:
        return self.data["positions"].get(symbol)

    def set_position_meta(self, symbol: str, stop_price: float, high_water_mark: float,
                          degraded: bool = False) -> None:
        """degraded marks a stop the engine could not size properly (no ATR); the
        next processed bar refines it."""
        meta = {"stop_price": stop_price, "high_water_mark": high_water_mark}
        if degraded:
            meta["degraded"] = True
        self.data["positions"][symbol] = meta
        self.save()

    def clear_position_meta(self, symbol: str) -> None:
        self.data["positions"].pop(symbol, None)
        self.save()

    def resting_order(self, symbol: str) -> str | None:
        """Bar boundary a resting crypto entry was placed at, or None."""
        return self.data["resting_orders"].get(symbol)

    def set_resting_order(self, symbol: str, boundary_iso: str) -> None:
        self.data["resting_orders"][symbol] = boundary_iso
        self.save()

    def clear_resting_order(self, symbol: str) -> None:
        self.data["resting_orders"].pop(symbol, None)
        self.save()

    def ladder_order(self, symbol: str) -> dict | None:
        """Marker for a resting rest-then-cross ladder leg, or None. Payload holds
        order_id, boundary, stop_price, side, total_qty, signal_close, submitted_at."""
        return self.data["ladder_orders"].get(symbol)

    def set_ladder_order(self, symbol: str, payload: dict) -> None:
        self.data["ladder_orders"][symbol] = payload
        self.save()

    def clear_ladder_order(self, symbol: str) -> None:
        self.data["ladder_orders"].pop(symbol, None)
        self.save()

    def pending_fill(self, symbol: str) -> dict | None:
        """Marker for an order awaiting fill measurement, or None. Payload holds
        order_id, side, event, decision_mid, decision_spread_bps, limit_price,
        submitted_at."""
        return self.data["pending_fills"].get(symbol)

    def set_pending_fill(self, symbol: str, payload: dict) -> None:
        self.data["pending_fills"][symbol] = payload
        self.save()

    def clear_pending_fill(self, symbol: str) -> None:
        self.data["pending_fills"].pop(symbol, None)
        self.save()

    # Alert counters: per-symbol channels the briefing reads, all sharing one
    # payload shape. `since` is the first occurrence and survives every re-record
    # — an alert's age is what tells the operator a fault is stuck rather than
    # transient — and clearing an absent symbol must not write.
    def _get_counter(self, channel: str, symbol: str) -> dict | None:
        return self.data[channel].get(symbol)

    def _record_counter(self, channel: str, symbol: str, detail_key: str,
                        detail: str) -> None:
        existing = self.data[channel].get(symbol)
        self.data[channel][symbol] = {
            detail_key: detail,
            "since": existing["since"] if existing
                     else datetime.now(timezone.utc).isoformat(),
            "count": (existing["count"] if existing else 0) + 1,
        }
        self.save()

    def _clear_counter(self, channel: str, symbol: str) -> None:
        if self.data[channel].pop(symbol, None) is not None:
            self.save()

    def exit_failure(self, symbol: str) -> dict | None:
        """Unresolved exit failure, or None. Payload holds error, since, count."""
        return self._get_counter("exit_failures", symbol)

    def record_exit_failure(self, symbol: str, error: str) -> None:
        self._record_counter("exit_failures", symbol, "error", error)

    def clear_exit_failure(self, symbol: str) -> None:
        self._clear_counter("exit_failures", symbol)

    def entry_refusal(self, symbol: str) -> dict | None:
        """An entry the broker refused outright, or None. Clears the moment an
        entry for that symbol is accepted, so a standing refusal means the bot
        has silently stopped being able to open the position. Payload holds
        error, since, count."""
        return self._get_counter("entry_refusals", symbol)

    def record_entry_refusal(self, symbol: str, error: str) -> None:
        self._record_counter("entry_refusals", symbol, "error", error)

    def clear_entry_refusal(self, symbol: str) -> None:
        self._clear_counter("entry_refusals", symbol)

    def cancel_failure(self, symbol: str) -> dict | None:
        """A resting order the breaker could not cancel, or None. Deliberately
        separate from exit_failures: that channel self-clears once the symbol is
        flat, and a flat symbol is exactly where an uncancelled entry order is
        dangerous. Payload holds error, since, count."""
        return self._get_counter("cancel_failures", symbol)

    def record_cancel_failure(self, symbol: str, error: str) -> None:
        self._record_counter("cancel_failures", symbol, "error", error)

    def clear_cancel_failure(self, symbol: str) -> None:
        self._clear_counter("cancel_failures", symbol)

    def data_divergence(self, symbol: str) -> dict | None:
        """Unresolved disagreement between a decision bar and the live market,
        or None. Payload holds detail, since, count."""
        return self._get_counter("data_divergences", symbol)

    def record_data_divergence(self, symbol: str, detail: str) -> None:
        self._record_counter("data_divergences", symbol, "detail", detail)

    def clear_data_divergence(self, symbol: str) -> None:
        self._clear_counter("data_divergences", symbol)

    def short_window(self, symbol: str) -> dict | None:
        """A bar the strategy could not be evaluated on because fewer bars were
        available than its min_bars, or None. No decision is taken on such a
        bar; self-clears on the next full-window decision. Payload holds detail,
        since, count."""
        return self._get_counter("short_windows", symbol)

    def record_short_window(self, symbol: str, detail: str) -> None:
        self._record_counter("short_windows", symbol, "detail", detail)

    def clear_short_window(self, symbol: str) -> None:
        self._clear_counter("short_windows", symbol)

    def last_bar(self, symbol: str) -> dict | None:
        """Timestamp and close of the last completed bar seen for a symbol, so a
        corporate action restating that same bar is detectable on the next fetch."""
        return self.data["last_bars"].get(symbol)

    def set_last_bar(self, symbol: str, timestamp_iso: str, close: float) -> None:
        snapshot = {"timestamp": timestamp_iso, "close": close}
        if self.data["last_bars"].get(symbol) == snapshot:
            return
        self.data["last_bars"][symbol] = snapshot
        self.save()

    def processed_command(self, signature: str) -> bool:
        """Whether an email command carrying this signature has already run. The
        signature is keyed on rather than the message that delivered it: the
        latter is chosen by the sender, so a replay would simply arrive with a
        fresh one."""
        return signature in self.data["processed_commands"]

    def record_processed_command(self, signature: str, limit: int) -> None:
        ledger = self.data["processed_commands"]
        ledger.append(signature)
        del ledger[:-limit]
        self.save()

    def refusal_replies_today(self) -> int:
        """Refusal replies charged to the current UTC day. A record left from an
        earlier day reads as zero, so the budget refills without anything having
        to clear it on a schedule."""
        record = self.data["refusal_replies"]
        return record["count"] if record.get("date") == utc_day() else 0

    def refusal_budget_spent(self, limit: int) -> bool:
        """Whether the next refusal would go unanswered. Kept beside the claim
        so a reader of the budget and the spender of it cannot disagree about
        where the line falls."""
        return self.refusal_replies_today() >= limit

    def claim_refusal_reply(self, limit: int) -> str:
        """Spend one of today's refusal replies: "send", "exhausted" for the
        first refusal past the limit — the one the caller announces — or
        "suppress" for every one after it. The count saturates one past the
        limit, so a flood costs no further writes."""
        # One clock read for the whole claim: sampling the day twice could
        # straddle midnight and stamp tomorrow with today's spent count.
        today = utc_day()
        record = self.data["refusal_replies"]
        spent = record["count"] if record.get("date") == today else 0
        if spent > limit:
            return "suppress"
        self.data["refusal_replies"] = {"date": today, "count": spent + 1}
        self.save()
        return "send" if spent < limit else "exhausted"

    def command_poll_health(self) -> dict | None:
        """The command channel's unresolved outage, or None when the last poll
        reached the mailbox. Payload holds error, since, last, count.

        Not symbol-keyed like the other counters: there is one channel, and it
        carries the remote kill switch, so its liveness is a property of the bot
        rather than of any instrument."""
        return self.data.get("command_poll_failure")

    def record_command_poll_failure(self, error: str,
                                    now: datetime | None = None) -> None:
        """`since` is the start of the outage and never moves; `last` is this
        attempt, and is what the breaker's backoff is measured from."""
        stamp = (now or datetime.now(timezone.utc)).isoformat()
        existing = self.data.get("command_poll_failure")
        self.data["command_poll_failure"] = {
            "error": error,
            "since": existing["since"] if existing else stamp,
            "last": stamp,
            "count": (existing["count"] if existing else 0) + 1,
        }
        self.save()

    def clear_command_poll_failure(self) -> None:
        if self.data.get("command_poll_failure") is not None:
            self.data["command_poll_failure"] = None
            self.save()

    def losing_streak(self, symbol: str) -> int:
        """Losing exits booked for a symbol since its last winner."""
        return self.data["losing_streaks"].get(symbol, 0)

    def set_losing_streak(self, symbol: str, count: int) -> None:
        if count:
            self.data["losing_streaks"][symbol] = count
            self.save()
        elif self.data["losing_streaks"].pop(symbol, None) is not None:
            self.save()

    def cooldown(self, symbol: str) -> dict | None:
        """An entry pause still in force for a symbol, or None. Expiry is a
        timestamp comparison rather than a scheduled clear, so a record the bot
        was down through simply stops reading as a cooldown once its date
        passes. Payload holds until, losses, days."""
        record = self.data["cooldowns"].get(symbol)
        if record is None:
            return None
        until = parse_utc(record.get("until"))
        if until is None:
            # A hand-edited state.json is one way to lift a cooldown, so a typo
            # lands here — and this is read every cycle. A pause that cannot be
            # dated cannot be enforced, so it lapses; the record is left alone
            # for the operator to correct, and the warning is raised once per
            # process rather than every 60 seconds.
            if symbol not in self._unusable_cooldowns:
                self._unusable_cooldowns.add(symbol)
                log.warning("%s: cooldown 'until' is unusable (%r) — the pause "
                            "cannot be enforced and entries resume; fix or "
                            "remove the record in state.json",
                            symbol, record.get("until"))
            return None
        return record if until > datetime.now(timezone.utc) else None

    def cooldown_unusable(self, symbol: str) -> bool:
        """Whether a cooldown record exists but cannot be dated. Neither this
        nor a lapsed record enforces a pause, but only this one is a broken
        record somebody has to repair — the difference the operator needs."""
        record = self.data["cooldowns"].get(symbol)
        return record is not None and parse_utc(record.get("until")) is None

    def start_cooldown(self, symbol: str, until: datetime, losses: int,
                       days: int) -> None:
        self.data["cooldowns"][symbol] = {"until": until.isoformat(),
                                          "losses": losses, "days": days}
        self.save()

    def clear_cooldown(self, symbol: str) -> bool:
        """Drop a symbol's cooldown record and report whether one was there. A
        record too broken to date counts: that is the other thing an operator
        needs removed, and it is the one the briefing alerts on."""
        if self.data["cooldowns"].pop(symbol, None) is None:
            return False
        # A later broken record deserves its own warning, so forget this one.
        self._unusable_cooldowns.discard(symbol)
        self.save()
        return True

    def day_start_equity(self) -> float | None:
        """Equity the current UTC day opened on, or None before the day has been
        seeded. A record left from an earlier day reads as None, so the
        reference rolls over without anything having to clear it."""
        record = self.data["day_start_equity"]
        return record["equity"] if record.get("date") == utc_day() else None

    def seed_day_start_equity(self, equity: float) -> float:
        """Today's reference, stamping `equity` as it on the first call of the
        day. Writes, unlike day_start_equity(): only the engine may set the
        reference, so the briefing cannot invent one the engine never saw."""
        today = utc_day()
        record = self.data["day_start_equity"]
        if record.get("date") == today:
            return record["equity"]
        self.data["day_start_equity"] = {"date": today, "equity": equity}
        self.save()
        return equity

    def daily_loss_tripped_today(self) -> bool:
        """Whether the daily loss limit has already halted trading today. A
        record from an earlier day reads as False, so the limit re-arms at the
        UTC day roll without anything having to clear it."""
        return self.data["daily_loss_tripped"] == utc_day()

    def record_daily_loss_trip(self) -> None:
        """Spend the day's one daily-loss halt. Called after the sentinel is
        written, never before: a failed write must leave the limit armed rather
        than marked spent."""
        self.data["daily_loss_tripped"] = utc_day()
        self.save()

    def rescale_price_levels(self, symbol: str, factor: float) -> None:
        """Multiply every persisted absolute price for a symbol. Stops and
        high-water marks are raw prices compared against split/dividend-adjusted
        bars; when the bar series is restated they have to follow or the stop
        sits a corporate action away from the market."""
        meta = self.data["positions"].get(symbol)
        if meta is not None:
            meta["stop_price"] *= factor
            meta["high_water_mark"] *= factor
        seen = self.data["last_seen_positions"].get(symbol)
        if seen is not None:
            seen["entry_price"] *= factor
        self.save()

    def last_seen_position(self, symbol: str) -> dict | None:
        """Side/qty/entry_price of the position as last observed at the broker,
        plus the seen_at instant. Survives the position itself so a stop fill can
        be booked as a trade; seen_at bounds which closed order may be that fill,
        so an older one (the entry, or a prior round trip) cannot be booked."""
        return self.data["last_seen_positions"].get(symbol)

    def set_last_seen_position(self, symbol: str, side: str, qty: float,
                               entry_price: float) -> None:
        identity = {"side": side, "qty": qty, "entry_price": entry_price}
        existing = self.data["last_seen_positions"].get(symbol)
        if (existing is not None and existing.get("seen_at")
                and all(existing.get(k) == v for k, v in identity.items())):
            return
        self.data["last_seen_positions"][symbol] = {
            **identity, "seen_at": datetime.now(timezone.utc).isoformat()}
        self.save()

    def clear_last_seen_position(self, symbol: str) -> None:
        if self.data["last_seen_positions"].pop(symbol, None) is not None:
            self.save()
