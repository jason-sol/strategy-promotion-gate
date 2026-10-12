import hashlib
import hmac
import io
import re
import time
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from bot import commands
from bot import state as bot_state
from bot.state import StateStore

SECRET = "s3cret"


def _signature(payload: str, secret: str = SECRET) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def _subject(verb_and_args: str, ts: int | None = None, secret: str = SECRET,
             signature: str | None = None) -> str:
    ts = int(time.time()) if ts is None else ts
    payload = f"{verb_and_args} {ts}"
    return f"CMD {payload} {signature or _signature(payload, secret)}"


def _store(tmp_path) -> StateStore:
    return StateStore(str(tmp_path / "state.json"))


class TestVerifySubject:
    def test_accepts_halt_with_reason(self):
        command, reason = commands.verify_subject(_subject("halt manual stop"), SECRET)
        assert reason == ""
        assert command.verb == "halt"
        assert command.args == "manual stop"

    def test_accepts_verb_without_args(self):
        command, reason = commands.verify_subject(_subject("clear-halt"), SECRET)
        assert reason == ""
        assert command.verb == "clear-halt"
        assert command.args == ""

    def test_accepts_force_exit_symbol(self):
        command, _ = commands.verify_subject(_subject("force-exit SPY"), SECRET)
        assert (command.verb, command.args) == ("force-exit", "SPY")

    def test_accepts_clear_cooldown_symbol(self):
        command, reason = commands.verify_subject(
            _subject("clear-cooldown BTC/USD"), SECRET)
        assert reason == ""
        assert (command.verb, command.args) == ("clear-cooldown", "BTC/USD")

    def test_surrounding_whitespace_is_tolerated(self):
        command, reason = commands.verify_subject(f"  {_subject('status')} ", SECRET)
        assert reason == ""
        assert command.verb == "status"

    def test_wrong_secret_is_refused_as_bad_signature(self):
        command, reason = commands.verify_subject(_subject("halt x", secret="other"),
                                                  SECRET)
        assert command is None
        assert "signature" in reason

    def test_tampered_args_break_the_signature(self):
        subject = _subject("halt manual stop")
        tampered = subject.replace("manual", "forged")
        command, reason = commands.verify_subject(tampered, SECRET)
        assert command is None
        assert "signature" in reason

    # The window tests pin `now` rather than racing the wall clock.
    NOW = 1_756_800_000
    WINDOW = commands.COMMANDS.max_age_seconds

    def test_timestamp_one_second_past_the_window_is_refused(self):
        command, reason = commands.verify_subject(
            _subject("status", ts=self.NOW - self.WINDOW - 1), SECRET, now=self.NOW)
        assert command is None
        assert "stale" in reason

    def test_timestamp_at_the_window_edge_is_accepted(self):
        command, reason = commands.verify_subject(
            _subject("status", ts=self.NOW - self.WINDOW), SECRET, now=self.NOW)
        assert reason == ""
        assert command.verb == "status"

    def test_timestamp_far_in_the_future_is_refused(self):
        command, reason = commands.verify_subject(
            _subject("status", ts=self.NOW + self.WINDOW + 1), SECRET, now=self.NOW)
        assert command is None
        assert "future" in reason

    def test_small_clock_skew_is_accepted(self):
        command, reason = commands.verify_subject(
            _subject("status", ts=self.NOW + 60), SECRET, now=self.NOW)
        assert reason == ""
        assert command.verb == "status"

    def test_non_command_subject_is_malformed(self):
        command, reason = commands.verify_subject("lunch tomorrow?", SECRET)
        assert command is None
        assert "malformed" in reason

    def test_too_few_tokens_is_malformed(self):
        command, reason = commands.verify_subject("CMD halt", SECRET)
        assert command is None
        assert "malformed" in reason

    def test_unparseable_timestamp_is_distinct_from_a_stale_one(self):
        payload = "status never"
        subject = f"CMD {payload} {_signature(payload)}"
        command, reason = commands.verify_subject(subject, SECRET)
        assert command is None
        assert "timestamp" in reason and "stale" not in reason

    def test_unknown_verb_is_refused_even_when_signed(self):
        command, reason = commands.verify_subject(_subject("liquidate-all"), SECRET)
        assert command is None
        assert "unknown verb" in reason

    def test_docstring_one_liner_produces_an_accepted_subject(self, monkeypatch):
        """The operator's documented recipe must round-trip through the parser,
        and must read the secret from the environment rather than the command
        line, which lands in shell history and in `ps`."""
        monkeypatch.setenv("COMMAND_SECRET", SECRET)
        snippet = re.search(r'python -c "(.+?)"', commands.__doc__, re.S).group(1)
        assert "COMMAND_SECRET" in snippet
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exec(snippet, {})
        command, reason = commands.verify_subject(buffer.getvalue(), SECRET)
        assert reason == ""
        assert (command.verb, command.args) == ("halt", "manual stop")


class TestCommandsConfigured:
    def test_inactive_without_secret(self, monkeypatch):
        monkeypatch.setenv("GMAIL_ADDRESS", "bot@example.com")
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "app-password")
        monkeypatch.delenv("COMMAND_SECRET", raising=False)
        assert commands.commands_configured() is False

    def test_inactive_without_gmail_credentials(self, monkeypatch):
        monkeypatch.setenv("COMMAND_SECRET", SECRET)
        monkeypatch.delenv("GMAIL_ADDRESS", raising=False)
        monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
        assert commands.commands_configured() is False

    def test_active_with_all_three(self, monkeypatch):
        monkeypatch.setenv("COMMAND_SECRET", SECRET)
        monkeypatch.setenv("GMAIL_ADDRESS", "bot@example.com")
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "app-password")
        assert commands.commands_configured() is True

    def test_whitespace_secret_is_no_secret(self, monkeypatch):
        monkeypatch.setenv("COMMAND_SECRET", "   ")
        monkeypatch.setenv("GMAIL_ADDRESS", "bot@example.com")
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "app-password")
        assert commands.commands_configured() is False

    def test_secret_reused_from_the_app_password_refuses_to_arm(self, monkeypatch,
                                                                caplog):
        """Mailbox read access must not confer command authority; reusing the
        app password collapses the two into one credential."""
        monkeypatch.setattr(commands, "_shared_secret_warned", False)
        monkeypatch.setenv("COMMAND_SECRET", "app-password")
        monkeypatch.setenv("GMAIL_ADDRESS", "bot@example.com")
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "app-password")

        with caplog.at_level("WARNING"):
            assert commands.commands_configured() is False
            assert commands.commands_configured() is False

        assert len(caplog.records) == 1  # once, not every cycle forever


class FakeIMAP:
    """Stands in for imaplib.IMAP4_SSL: no socket, no Gmail."""

    def __init__(self, messages):
        self.messages = messages
        self.mailbox = None
        self.fetched: list[bytes] = []
        self.stored: list[tuple[bytes, str, str]] = []
        self.seen: set[bytes] = set()

    def login(self, address, password):
        return "OK", [b""]

    def select(self, mailbox):
        self.mailbox = mailbox
        return "OK", [b"1"]

    def search(self, charset, *criteria):
        """Honours UNSEEN and SUBJECT the way a real server does, so a client
        that fails to filter server-side is caught by what it fetches."""
        self.criteria = criteria
        matches = [m for m in self.messages if m["uid"] not in self.seen]
        if "SUBJECT" in criteria:
            term = criteria[criteria.index("SUBJECT") + 1].strip('"').upper()
            matches = [m for m in matches if term in m["subject"].upper()]
        return "OK", [b" ".join(m["uid"] for m in matches)]

    def fetch(self, uid, spec):
        self.fetched.append(uid)
        message = next(m for m in self.messages if m["uid"] == uid)
        if message.get("raises"):
            raise RuntimeError("fetch blew up")
        raw = (f"From: {message['from']}\r\n"
               f"Subject: {message['subject']}\r\n"
               f"Message-ID: {message['message_id']}\r\n\r\n").encode()
        return "OK", [(b"1 (BODY[HEADER] {%d}" % len(raw), raw), b")"]

    def store(self, uid, command, flags):
        message = next(m for m in self.messages if m["uid"] == uid)
        if message.get("store_raises"):
            raise RuntimeError("store blew up")
        self.stored.append((uid, command, flags))
        self.seen.add(uid)
        return "OK", [b""]

    def close(self):
        return "OK", [b""]

    def logout(self):
        return "BYE", [b""]


def _message(subject, uid=b"1", sender="ops@example.com", message_id="<m-1@x>",
             raises=False, store_raises=False):
    return {"uid": uid, "subject": subject, "from": sender,
            "message_id": message_id, "raises": raises,
            "store_raises": store_raises}


@pytest.fixture
def channel(monkeypatch):
    """Configured channel with a fake mailbox and a captured reply transport."""
    monkeypatch.setenv("COMMAND_SECRET", SECRET)
    monkeypatch.setenv("GMAIL_ADDRESS", "bot@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "app-password")
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr("bot.mail.send_gmail",
                        lambda subject, body: sent.append((subject, body)))

    def _install(messages, connect_error=None):
        imap = FakeIMAP(messages)

        def _connect():
            if connect_error is not None:
                raise connect_error
            return imap

        monkeypatch.setattr(commands, "_connect", _connect)
        return imap

    _install.sent = sent
    return _install


class TestPoll:
    def test_inactive_channel_never_connects(self, tmp_path, monkeypatch):
        monkeypatch.delenv("COMMAND_SECRET", raising=False)

        def _boom():
            raise AssertionError("must not connect when unconfigured")

        monkeypatch.setattr(commands, "_connect", _boom)
        assert commands.poll(_store(tmp_path)) == []

    def test_verified_command_is_returned_and_marked_seen(self, tmp_path, channel):
        imap = channel([_message(_subject("halt manual stop"))])

        found = commands.poll(_store(tmp_path))

        assert [(c.verb, c.args) for c in found] == [("halt", "manual stop")]
        assert imap.mailbox == "INBOX"
        assert imap.stored == [(b"1", "+FLAGS", "\\Seen")]

    def test_ordinary_mail_is_left_alone(self, tmp_path, channel):
        imap = channel([_message("Re: your Alpaca statement")])

        assert commands.poll(_store(tmp_path)) == []
        assert imap.stored == []          # a human's mail stays unread
        assert channel.sent == []         # and gets no robot reply

    def test_bad_signature_is_marked_seen_and_answered(self, tmp_path, channel):
        imap = channel([_message(_subject("halt x", secret="wrong"))])

        assert commands.poll(_store(tmp_path)) == []

        assert imap.stored == [(b"1", "+FLAGS", "\\Seen")]  # never retried
        assert len(channel.sent) == 1
        assert "signature" in channel.sent[0][1]

    def test_captured_subject_resent_as_a_fresh_message_is_refused(
            self, tmp_path, channel):
        """The attack the ledger exists to stop: someone who can read the
        mailbox copies a signed subject into a message of their own, which
        carries a Message-Id they chose. Only the signature is not theirs."""
        state = _store(tmp_path)
        subject = _subject("clear-halt")
        channel([_message(subject)])
        assert len(commands.poll(state)) == 1

        channel([_message(subject, uid=b"2", message_id="<attacker-chosen@evil>",
                          sender="evil@example.com")])

        assert commands.poll(state) == []
        assert "already executed" in channel.sent[-1][1]

    def test_ledger_is_bounded(self, tmp_path, channel):
        state = _store(tmp_path)
        now = int(time.time())
        for n in range(commands.COMMANDS.processed_history + 5):
            channel([_message(_subject("status", ts=now - n), uid=str(n).encode(),
                              message_id=f"<m-{n}@x>")])
            commands.poll(state)
        assert (len(state.data["processed_commands"])
                == commands.COMMANDS.processed_history)

    def test_only_command_subjects_are_fetched_from_the_server(self, tmp_path,
                                                               channel):
        """Filtering server-side is what stops ordinary mail — including the
        bot's own briefings — being fetched and answered every cycle."""
        imap = channel([_message("Re: your Alpaca statement", uid=b"9")])

        commands.poll(_store(tmp_path))

        assert "UNSEEN" in imap.criteria
        assert "SUBJECT" in imap.criteria
        assert imap.fetched == []

    def test_flood_is_capped_and_the_overflow_is_left_for_later(self, tmp_path,
                                                                channel, caplog):
        cap = commands.COMMANDS.max_per_cycle
        now = int(time.time())
        junk = [_message(f"CMD junk {n}", uid=str(n).encode(),
                         message_id=f"<j-{n}@x>") for n in range(cap + 5)]
        imap = channel(junk)

        with caplog.at_level("WARNING"):
            commands.poll(_store(tmp_path))

        assert len(imap.fetched) == cap      # the rest is never even fetched
        assert len(channel.sent) == cap      # so the SMTP path is bounded too
        assert sum("deferred" in r.message for r in caplog.records) == 1

    def test_mail_merely_mentioning_the_prefix_cannot_wedge_the_channel(
            self, tmp_path, channel):
        """The server's SUBJECT match is a substring one, so a forward like
        'Re: CMD halt ...' comes back from the search without parsing as a
        command. Left unread it re-matches every cycle and, oldest first, holds
        a cap slot for good — ten of them would silently kill the channel."""
        cap = commands.COMMANDS.max_per_cycle
        wedge = [_message(f"Re: CMD halt {n}", uid=str(n).encode(),
                          message_id=f"<w-{n}@x>") for n in range(cap)]
        real = _message(_subject("clear-halt"), uid=b"99", message_id="<real@x>")
        imap = channel(wedge + [real])
        state = _store(tmp_path)

        assert commands.poll(state) == []   # the cap is full of the junk
        assert channel.sent == []           # which is still never answered
        assert len(imap.stored) == cap      # but is read exactly once

        assert [c.verb for c in commands.poll(state)] == ["clear-halt"]

    def test_the_overflow_drains_on_a_later_cycle(self, tmp_path, channel):
        cap = commands.COMMANDS.max_per_cycle
        junk = [_message(f"CMD junk {n}", uid=str(n).encode(),
                         message_id=f"<j-{n}@x>") for n in range(cap + 5)]
        imap = channel(junk)
        state = _store(tmp_path)

        commands.poll(state)
        commands.poll(state)

        assert len(imap.fetched) == cap + 5

    def test_non_ascii_signature_token_is_refused_not_raised(self, tmp_path, channel):
        channel([_message("CMD halt x 1756800000 dëadbeef")])

        assert commands.poll(_store(tmp_path)) == []
        assert "signature" in channel.sent[-1][1]

    def test_connection_failure_yields_nothing(self, tmp_path, channel):
        channel([], connect_error=OSError("imap unreachable"))
        assert commands.poll(_store(tmp_path)) == []

    def test_a_secret_with_trailing_whitespace_still_verifies(self, tmp_path,
                                                              monkeypatch, channel):
        """An env var edited by hand often keeps a trailing newline. Arming on
        the stripped value but signing with the raw one would refuse every
        command as a bad signature."""
        monkeypatch.setenv("COMMAND_SECRET", f"{SECRET}\n")
        channel([_message(_subject("status"))])

        assert [c.verb for c in commands.poll(_store(tmp_path))] == ["status"]

    def test_one_failing_candidate_does_not_abandon_the_batch(self, tmp_path,
                                                              channel):
        channel([_message(_subject("status"), uid=b"1", message_id="<a@x>"),
                 _message(_subject("halt one"), uid=b"2", message_id="<b@x>",
                          store_raises=True),
                 _message(_subject("clear-halt"), uid=b"3", message_id="<c@x>")])

        found = commands.poll(_store(tmp_path))

        assert [c.verb for c in found] == ["status", "clear-halt"]

    def test_one_unreadable_message_does_not_lose_the_rest(self, tmp_path, channel):
        channel([_message(_subject("status"), uid=b"1", message_id="<a@x>",
                          raises=True),
                 _message(_subject("clear-halt"), uid=b"2", message_id="<b@x>")])

        found = commands.poll(_store(tmp_path))

        assert [c.verb for c in found] == ["clear-halt"]

    def test_sender_is_recorded_for_audit_but_does_not_gate(self, tmp_path, channel):
        channel([_message(_subject("status"), sender="stranger@example.com")])

        found = commands.poll(_store(tmp_path))

        assert found[0].sender == "stranger@example.com"


def _budget(monkeypatch, per_day: int) -> None:
    monkeypatch.setattr(commands, "COMMANDS",
                        replace(commands.COMMANDS, refusal_replies_per_day=per_day))


def _junk(n: int) -> dict:
    return _message(f"CMD junk {n}", uid=str(n).encode(), message_id=f"<j-{n}@x>")


class TestRefusalReplyBudget:
    """Replies to unauthenticated mail are the one part of this channel a
    stranger can drive, and they leave over the same Gmail account that carries
    the alerts and the briefings."""

    def test_each_refusal_reply_spends_one_of_the_day_s_budget(self, tmp_path,
                                                               channel, monkeypatch):
        _budget(monkeypatch, 3)
        state = _store(tmp_path)
        channel([_junk(0), _junk(1)])

        assert commands.poll(state) == []

        assert len(channel.sent) == 2
        assert state.refusal_replies_today() == 2

    def test_accepted_commands_never_touch_the_budget(self, tmp_path, channel,
                                                      monkeypatch):
        _budget(monkeypatch, 1)
        state = _store(tmp_path)
        channel([_message(_subject("status"), uid=b"1", message_id="<a@x>"),
                 _message(_subject("halt now"), uid=b"2", message_id="<b@x>")])

        found = commands.poll(state)

        assert [c.verb for c in found] == ["status", "halt"]
        assert state.refusal_replies_today() == 0

    def test_a_signed_command_still_runs_once_the_budget_is_spent(self, tmp_path,
                                                                  channel,
                                                                  monkeypatch):
        _budget(monkeypatch, 0)
        channel([_message(_subject("status"))])

        assert [c.verb for c in commands.poll(_store(tmp_path))] == ["status"]

    def test_refusals_past_the_budget_are_logged_read_but_not_answered(
            self, tmp_path, channel, monkeypatch, caplog):
        _budget(monkeypatch, 2)
        imap = channel([_junk(n) for n in range(5)])

        with caplog.at_level("WARNING"):
            assert commands.poll(_store(tmp_path)) == []

        assert len(channel.sent) == 2                          # SMTP stops at the cap
        assert len(imap.stored) == 5                           # all still read once
        assert sum("REFUSED" in r.message for r in caplog.records) == 5

    def test_the_exhaustion_warning_is_once_a_day_not_once_a_refusal(
            self, tmp_path, channel, monkeypatch, caplog):
        _budget(monkeypatch, 1)
        state = _store(tmp_path)

        with caplog.at_level("WARNING"):
            for n in range(4):
                channel([_junk(n)])
                commands.poll(state)

        assert len(channel.sent) == 1
        assert sum("budget" in r.message for r in caplog.records) == 1

    def test_the_budget_refills_when_the_utc_day_rolls_over(self, tmp_path, channel,
                                                            monkeypatch):
        _budget(monkeypatch, 1)
        monkeypatch.setattr(bot_state, "utc_day", lambda: "2026-09-02")
        state = _store(tmp_path)
        for n in range(2):
            channel([_junk(n)])
            commands.poll(state)
        assert len(channel.sent) == 1

        monkeypatch.setattr(bot_state, "utc_day", lambda: "2026-09-03")
        channel([_junk(9)])
        commands.poll(state)

        assert len(channel.sent) == 2
        assert state.refusal_replies_today() == 1

    def test_a_restart_does_not_refill_the_budget(self, tmp_path, channel,
                                                  monkeypatch):
        _budget(monkeypatch, 1)
        channel([_junk(0)])
        commands.poll(_store(tmp_path))

        reloaded = _store(tmp_path)
        assert reloaded.refusal_replies_today() == 1

        channel([_junk(1)])
        commands.poll(reloaded)

        assert len(channel.sent) == 1

    def test_the_budget_reads_as_spent_only_once_the_last_reply_is_gone(self, tmp_path):
        state = _store(tmp_path)
        assert not state.refusal_budget_spent(2)

        state.claim_refusal_reply(2)
        assert not state.refusal_budget_spent(2)

        state.claim_refusal_reply(2)
        assert state.refusal_budget_spent(2)

    def test_a_spent_budget_reads_as_unspent_after_the_utc_day_rolls_over(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(bot_state, "utc_day", lambda: "2026-09-02")
        state = _store(tmp_path)
        for _ in range(3):
            state.claim_refusal_reply(2)
        assert state.refusal_budget_spent(2)

        monkeypatch.setattr(bot_state, "utc_day", lambda: "2026-09-03")

        assert not state.refusal_budget_spent(2)


class TestSendReply:
    def test_reply_carries_the_request_and_verdict(self, monkeypatch):
        sent = []
        monkeypatch.setattr("bot.mail.send_gmail",
                            lambda subject, body: sent.append((subject, body)))

        commands.send_reply("CMD status 1 abc", "ops@example.com", "all good")

        subject, body = sent[0]
        assert "all good" in subject
        assert "CMD status 1 abc" in body
        assert "ops@example.com" in body
        assert "all good" in body

    def test_reply_is_not_itself_a_command_candidate(self, monkeypatch):
        """BRIEFING_TO is usually the command inbox: a reply the poller treats as
        a candidate is refused, answered, and loops forever."""
        sent = []
        monkeypatch.setattr("bot.mail.send_gmail",
                            lambda subject, body: sent.append((subject, body)))

        commands.send_reply("CMD status 1 abc", "ops@example.com", "REFUSED — stale")

        assert not commands.looks_like_command(sent[0][0])

    def test_send_failure_is_swallowed(self, monkeypatch):
        def _boom(subject, body):
            raise RuntimeError("smtp down")

        monkeypatch.setattr("bot.mail.send_gmail", _boom)
        commands.send_reply("CMD status 1 abc", "ops@example.com", "all good")


T0 = datetime(2026, 9, 4, 11, 0, tzinfo=timezone.utc)


class TestCommandPollHealth:
    """The channel carries the remote kill switch, so a mailbox that has quietly
    stopped answering has to be a fact the bot holds, not a line in a log."""

    def test_a_channel_that_has_not_failed_reports_nothing(self, tmp_path):
        assert _store(tmp_path).command_poll_health() is None

    def test_first_failure_stamps_since_and_last_together(self, tmp_path):
        store = _store(tmp_path)

        store.record_command_poll_failure("TimeoutError: read timed out", now=T0)

        health = store.command_poll_health()
        assert health["count"] == 1
        assert health["error"] == "TimeoutError: read timed out"
        assert health["since"] == health["last"] == T0.isoformat()

    def test_repeat_failures_advance_last_and_hold_since(self, tmp_path):
        store = _store(tmp_path)
        later = T0 + timedelta(minutes=5)

        store.record_command_poll_failure("handshake timed out", now=T0)
        store.record_command_poll_failure("socket error: EOF", now=later)

        health = store.command_poll_health()
        assert health["count"] == 2
        assert health["error"] == "socket error: EOF"   # newest wins
        assert health["since"] == T0.isoformat()        # but the outage started here
        assert health["last"] == later.isoformat()

    def test_success_clears_the_record(self, tmp_path):
        store = _store(tmp_path)
        store.record_command_poll_failure("boom", now=T0)

        store.clear_command_poll_failure()

        assert store.command_poll_health() is None

    def test_clearing_a_healthy_channel_is_a_no_op(self, tmp_path):
        store = _store(tmp_path)
        store.clear_command_poll_failure()
        assert store.command_poll_health() is None

    def test_an_outage_survives_a_restart(self, tmp_path):
        """A redeploy mid-outage must not reset the count: that is exactly when
        the breaker's backoff and the briefing alert have to stay in force."""
        path = str(tmp_path / "state.json")
        first = StateStore(path)
        first.record_command_poll_failure("boom", now=T0)

        health = StateStore(path).command_poll_health()

        assert health["count"] == 1
        assert health["since"] == T0.isoformat()


class TestPollBreaker:
    """Gmail throttles repeated IMAP logins, so a poll every cycle through an
    outage is what sustains the outage. The breaker stops asking."""

    ERROR = OSError("imap unreachable")
    THRESHOLD = commands.COMMANDS.breaker_threshold
    BASE = commands.COMMANDS.breaker_base_seconds

    def _attempts(self, monkeypatch):
        """Connect attempts the breaker actually let through."""
        calls = []

        def _connect():
            calls.append(1)
            raise self.ERROR

        monkeypatch.setattr(commands, "_connect", _connect)
        return calls

    def _failed(self, tmp_path, count, at=T0):
        store = _store(tmp_path)
        for _ in range(count):
            store.record_command_poll_failure("boom", now=at)
        return store

    def test_failures_below_the_threshold_keep_polling(self, tmp_path, channel,
                                                       monkeypatch):
        attempts = self._attempts(monkeypatch)
        store = self._failed(tmp_path, self.THRESHOLD - 1)

        assert commands.poll(store, now=T0) == []

        # Two sockets, one cycle: a failing connect is retried once immediately.
        assert len(attempts) == 2   # still trying: a blip is not an outage

    def test_the_threshold_shuts_the_mailbox_out(self, tmp_path, channel,
                                                 monkeypatch):
        attempts = self._attempts(monkeypatch)
        store = self._failed(tmp_path, self.THRESHOLD)

        assert commands.poll(store, now=T0 + timedelta(seconds=self.BASE - 1)) == []

        assert attempts == []

    def test_the_wait_expiring_lets_the_next_cycle_through(self, tmp_path, channel,
                                                           monkeypatch):
        attempts = self._attempts(monkeypatch)
        store = self._failed(tmp_path, self.THRESHOLD)

        commands.poll(store, now=T0 + timedelta(seconds=self.BASE))

        assert len(attempts) == 2   # the cycle runs: one try plus its retry

    def test_each_further_failure_doubles_the_wait(self):
        schedule = [commands._breaker_backoff_seconds(self.THRESHOLD + n)
                    for n in range(2)]
        assert schedule == [self.BASE, self.BASE * 2]

    def test_the_doubling_stops_at_the_cap_rather_than_overshooting(self):
        cap = commands.COMMANDS.breaker_max_seconds
        assert all(commands._breaker_backoff_seconds(self.THRESHOLD + n) <= cap
                   for n in range(10))

    def test_below_the_threshold_there_is_no_wait(self):
        assert commands._breaker_backoff_seconds(self.THRESHOLD - 1) == 0

    def test_a_long_outage_settles_at_the_cap(self):
        assert (commands._breaker_backoff_seconds(500)
                == commands.COMMANDS.breaker_max_seconds)

    def test_an_unreadable_stamp_never_wedges_the_channel_shut(self, tmp_path,
                                                               channel, monkeypatch):
        """A hand-edited or corrupt state.json must fail open: losing the kill
        switch to a bad timestamp is far worse than one extra login."""
        attempts = self._attempts(monkeypatch)
        store = self._failed(tmp_path, self.THRESHOLD)
        store.data["command_poll_failure"]["last"] = "not-a-timestamp"

        commands.poll(store, now=T0)

        assert len(attempts) == 2

    def test_the_failure_reason_is_recorded(self, tmp_path, channel, monkeypatch):
        self._attempts(monkeypatch)

        commands.poll(_store(tmp_path), now=T0)

        health = _store(tmp_path).command_poll_health()
        assert health["count"] == 1
        assert "imap unreachable" in health["error"]

    def test_a_reachable_mailbox_closes_the_breaker(self, tmp_path, channel):
        store = self._failed(tmp_path, self.THRESHOLD)
        channel([])   # mailbox answers again

        commands.poll(store, now=T0 + timedelta(seconds=self.BASE))

        assert store.command_poll_health() is None

    def test_recovery_is_announced(self, tmp_path, channel, caplog):
        store = self._failed(tmp_path, self.THRESHOLD)
        channel([])

        with caplog.at_level("INFO"):
            commands.poll(store, now=T0 + timedelta(seconds=self.BASE))

        assert any("reachable again" in r.message for r in caplog.records)

    def test_only_the_first_failure_carries_a_traceback(self, tmp_path, channel,
                                                        monkeypatch, caplog):
        """A throttled Gmail repeats the same three stacks for hours; the
        traceback earns its ~25 log lines once."""
        self._attempts(monkeypatch)
        store = _store(tmp_path)

        with caplog.at_level("WARNING"):
            commands.poll(store, now=T0)
            commands.poll(store, now=T0 + timedelta(seconds=1))

        traced = [r for r in caplog.records if r.exc_info]
        assert len(traced) == 1


class TestConnectRetry:
    """Gmail's LOGIN latency tail is not a down mailbox — the attempt a minute
    later has always succeeded. One immediate retry separates slow from dead."""

    def _attempts(self, monkeypatch, outcomes):
        """Patch _connect to walk `outcomes`, raising exceptions and returning
        anything else. The last entry repeats."""
        calls = []

        def _connect():
            calls.append(1)
            result = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(commands, "_connect", _connect)
        return calls

    def test_a_healthy_connect_is_not_retried(self, tmp_path, channel, monkeypatch):
        calls = self._attempts(monkeypatch, [FakeIMAP([])])

        commands.poll(_store(tmp_path), now=T0)

        assert len(calls) == 1

    def test_a_slow_connect_is_retried_once_and_succeeds(self, tmp_path, channel,
                                                         monkeypatch):
        imap = FakeIMAP([_message(_subject("halt slow login"))])
        calls = self._attempts(monkeypatch,
                               [TimeoutError("read operation timed out"), imap])

        found = commands.poll(_store(tmp_path), now=T0)

        assert len(calls) == 2
        assert [c.verb for c in found] == ["halt"]   # the command still lands

    def test_a_retried_success_leaves_the_channel_healthy(self, tmp_path, channel,
                                                          monkeypatch):
        self._attempts(monkeypatch,
                       [TimeoutError("read operation timed out"), FakeIMAP([])])
        store = _store(tmp_path)

        commands.poll(store, now=T0)

        assert store.command_poll_health() is None

    def test_both_attempts_failing_counts_as_one_failure(self, tmp_path, channel,
                                                         monkeypatch):
        """The breaker counts failed cycles, not failed sockets — otherwise one
        bad minute would spend two of its three lives."""
        calls = self._attempts(monkeypatch, [TimeoutError("read timed out")])
        store = _store(tmp_path)

        commands.poll(store, now=T0)

        assert len(calls) == 2
        assert store.command_poll_health()["count"] == 1

    def test_an_open_breaker_still_skips_both_attempts(self, tmp_path, channel,
                                                       monkeypatch):
        calls = self._attempts(monkeypatch, [TimeoutError("x")])
        store = _store(tmp_path)
        for _ in range(commands.COMMANDS.breaker_threshold):
            store.record_command_poll_failure("boom", now=T0)

        commands.poll(store, now=T0)

        assert calls == []


class TestPartWayFailure:
    """Reaching the mailbox but failing to read it leaves the channel just as
    unusable as never connecting, so it has to count the same."""

    def _select_fails(self, monkeypatch):
        class _Stub(FakeIMAP):
            def __init__(self):
                super().__init__([])
                self.closed = False

            def select(self, mailbox):
                raise TimeoutError("The read operation timed out")

            def close(self):
                self.closed = True

            def logout(self):
                pass

        stub = _Stub()
        monkeypatch.setattr(commands, "_connect", lambda: stub)
        return stub

    def test_a_part_way_failure_is_counted(self, tmp_path, channel, monkeypatch):
        self._select_fails(monkeypatch)
        store = _store(tmp_path)

        assert commands.poll(store, now=T0) == []

        assert store.command_poll_health()["count"] == 1

    def test_a_part_way_failure_does_not_clear_a_running_outage(self, tmp_path,
                                                                channel, monkeypatch):
        """The hole this closes: a SELECT that always timed out read as a
        healthy channel forever, so neither breaker nor alert could ever fire."""
        self._select_fails(monkeypatch)
        store = _store(tmp_path)
        store.record_command_poll_failure("earlier", now=T0)

        commands.poll(store, now=T0 + timedelta(seconds=60))

        assert store.command_poll_health()["count"] == 2

    def test_a_part_way_failure_still_closes_the_connection(self, tmp_path, channel,
                                                            monkeypatch):
        stub = self._select_fails(monkeypatch)

        commands.poll(_store(tmp_path), now=T0)

        assert stub.closed

    def test_a_completed_poll_still_clears_the_outage(self, tmp_path, channel):
        store = _store(tmp_path)
        store.record_command_poll_failure("earlier", now=T0)
        channel([])

        commands.poll(store, now=T0 + timedelta(seconds=60))

        assert store.command_poll_health() is None
