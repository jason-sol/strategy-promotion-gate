"""Remote control over inbound email: halt, clear-halt, force-exit,
clear-cooldown, status.

The bot polls its own Gmail inbox once per cycle for unseen messages whose
subject is a signed command, and the engine executes the verified ones inside
the cycle it is already running — no port, no second process, no new dependency.

Subject format (the body is ignored):

    CMD <verb> [args] <unix-ts> <hex hmac-sha256 over "<verb> [args] <unix-ts>">

Send it as a plain ASCII subject, with COMMAND_SECRET exported in your shell:

    python -c "import hashlib,hmac,os,time; t=int(time.time()); m=f'halt manual stop {t}'; print('CMD',m,hmac.new(os.environ['COMMAND_SECRET'].strip().encode(),m.encode(),hashlib.sha256).hexdigest())"

A subject a mail client encodes (RFC 2047, `=?utf-8?...?=`) is never decoded
here, so it is silently invisible to the bot rather than wrong — keep the
subject plain ASCII, which every signed command already is.

Threat model. Plaintext email means a command is readable in transit and in the
mailbox, so the subject is treated as public: the HMAC is the credential, and
the From header is audit only (trivially spoofable). Forgery needs the secret,
which is deliberately not GMAIL_APP_PASSWORD — mailbox read access must not
confer command authority. That leaves replay, which the timestamp window and
the ledger of executed signatures close from both ends: a captured command can
only be re-sent inside its window, and the ledger refuses it however it is
re-delivered. The ledger keys on the signature and not on the Message-Id
precisely because the Message-Id belongs to whoever composed the message — a
replay pasted into a fresh mail would carry a new one.

Refusals are answered by email, which makes the reply the one thing a stranger
can drive here, over the same Gmail account that carries the alerts and the
briefings. So refusal replies are budgeted per UTC day: past the budget a
refused message is still read, still refused and still logged, but goes
unanswered rather than spending the send quota the alerts depend on. An
accepted command is answered whatever the budget says — it carried the secret.

Only subjects that begin with CMD are ever executed or answered. Mail the
server-side subject filter excludes is never fetched and stays unread; mail
that merely mentions the prefix is marked read once and then ignored.
"""

import hashlib
import hmac
import imaplib
import logging
import os
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from email import message_from_bytes
from email.message import Message

from bot import mail
from bot.state import parse_utc
from config import COMMANDS

log = logging.getLogger(__name__)

PREFIX = "CMD"
VERBS = frozenset({"halt", "clear-halt", "force-exit", "clear-cooldown",
                   "status"})


@dataclass(frozen=True)
class Command:
    verb: str
    args: str = ""
    signature: str = ""
    subject: str = ""
    sender: str = ""
    message_id: str = ""


_shared_secret_warned = False


def _secret() -> str:
    """The signing key. Stripped at every point of use: an env var edited by
    hand keeps a trailing newline, and arming on one value while signing with
    another refuses every command as a bad signature."""
    return (os.getenv("COMMAND_SECRET") or "").strip()


def commands_configured() -> bool:
    global _shared_secret_warned
    secret = _secret()
    if not secret or not mail.mail_configured():
        return False
    if secret == os.getenv("GMAIL_APP_PASSWORD"):
        if not _shared_secret_warned:
            log.warning("COMMAND_SECRET is the Gmail app password — the command "
                        "channel stays OFF: anyone who can read the mailbox could "
                        "then also sign commands. Set a separate secret.")
            _shared_secret_warned = True
        return False
    return True


def looks_like_command(subject: str) -> bool:
    tokens = subject.split()
    return bool(tokens) and tokens[0] == PREFIX


def verify_subject(subject: str, secret: str,
                   now: float | None = None) -> tuple[Command | None, str]:
    """(command, refusal reason). The reason is empty exactly when the command
    is accepted, and is otherwise the audit line for the log and the reply."""
    tokens = subject.split()
    if len(tokens) < 4 or tokens[0] != PREFIX:
        return None, "malformed subject"
    payload = " ".join(tokens[1:-1])
    expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    # Nothing in the subject is interpreted before the signature clears it, and
    # the compare is on bytes: a non-ASCII token must refuse, not raise.
    if not hmac.compare_digest(expected.encode(), tokens[-1].lower().encode()):
        return None, "bad signature"
    try:
        issued_at = int(tokens[-2])
    except ValueError:
        return None, f"unparseable timestamp {tokens[-2]!r}"
    age = (time.time() if now is None else now) - issued_at
    if age > COMMANDS.max_age_seconds:
        return None, (f"stale timestamp ({age:.0f}s old, window "
                      f"{COMMANDS.max_age_seconds}s)")
    if -age > COMMANDS.max_age_seconds:
        return None, f"timestamp is in the future by {-age:.0f}s"
    verb = tokens[1]
    if verb not in VERBS:
        return None, f"unknown verb {verb!r}"
    return Command(verb=verb, args=" ".join(tokens[2:-2]), signature=expected,
                   subject=subject.strip()), ""


def _connect():
    conn = imaplib.IMAP4_SSL(COMMANDS.imap_host, timeout=COMMANDS.timeout_seconds)
    conn.login(os.environ["GMAIL_ADDRESS"], os.environ["GMAIL_APP_PASSWORD"])
    return conn


def _unseen_candidates(conn, limit: int) -> tuple[list[tuple[bytes, Message]], int]:
    """(uid, headers) for at most `limit` unseen command candidates, plus the
    number left behind. The subject filter runs on the server so ordinary mail
    is never fetched, and the limit caps the work — and the replies — one cycle
    can be made to do. Headers only, and PEEKed: what counts as read is decided
    by the caller, not by the fetch.

    A message whose fetch fails is deliberately left unread and re-examined next
    cycle: losing a kill-switch command to a network blip is the worse of the
    two failures."""
    conn.select(COMMANDS.mailbox)
    # PREFIX is a module constant and must stay one: a configurable value
    # interpolated into a SEARCH string is a search-injection hole.
    status, data = conn.search(None, "UNSEEN", "SUBJECT", f'"{PREFIX}"')
    if status != "OK" or not data or not data[0]:
        return [], 0
    uids = data[0].split()
    out = []
    for uid in uids[:limit]:
        try:
            status, payload = conn.fetch(uid, "(BODY.PEEK[HEADER])")
        except Exception:
            log.warning("command mail %s could not be fetched", uid, exc_info=True)
            continue
        raw = next((part[1] for part in payload if isinstance(part, tuple)), b"")
        if status != "OK" or not raw:
            continue
        out.append((uid, message_from_bytes(raw)))
    return out, max(0, len(uids) - limit)


def _close(conn) -> None:
    for step in (conn.close, conn.logout):
        try:
            step()
        except Exception:
            pass


def _process_candidate(conn, uid, headers, secret: str, state) -> Command | None:
    """Verify one candidate, or answer its refusal. Returns the command to
    execute, or None when there is nothing to run."""
    # str(): a header carrying non-ASCII comes back as a Header object.
    subject = str(headers.get("Subject") or "").strip()
    # Read before parsed, and read whatever the search returned. The server
    # matches the prefix anywhere in the subject, so a forward ("Re: CMD ...")
    # arrives here without being a command; skipping it unread would leave it to
    # re-match every cycle and, oldest first, hold a cap slot against real
    # commands for good.
    conn.store(uid, "+FLAGS", "\\Seen")
    if not looks_like_command(subject):
        return None  # read, but never answered: no reply amplifier
    sender = str(headers.get("From") or "unknown")
    message_id = str(headers.get("Message-ID") or "")
    command, reason = verify_subject(subject, secret)
    if command is not None and state.processed_command(command.signature):
        command, reason = None, "already executed (replay)"
    if command is None:
        log.warning("command REFUSED from %s %s (%s): %s", sender, message_id,
                    reason, subject)
        # A refusal is the only reply an unauthenticated stranger can provoke,
        # and it leaves over the same Gmail account the alerts do — so it is
        # budgeted by the day. The refusal itself is unaffected: the message is
        # still read and still never executed, it just goes unanswered.
        claim = state.claim_refusal_reply(COMMANDS.refusal_replies_per_day)
        if claim == "exhausted":
            log.warning("refusal-reply budget for today is spent (%d); further "
                        "refusals are logged but not answered until the UTC day "
                        "rolls over", COMMANDS.refusal_replies_per_day)
        elif claim == "send":
            send_reply(subject, sender, f"REFUSED — {reason}")
        return None
    log.info("command accepted from %s %s: %s", sender, message_id, subject)
    state.record_processed_command(command.signature, COMMANDS.processed_history)
    return replace(command, sender=sender, message_id=message_id)


# A doubling ceiling, so an outage measured in days cannot overflow the shift.
_MAX_BACKOFF_DOUBLINGS = 20


def _breaker_backoff_seconds(count: int) -> float:
    """How long to leave the mailbox alone after `count` consecutive failures.

    Google meters IMAP logins, so polling every cycle through an outage is what
    sustains it. The wait doubles to a cap, taking a locked-out account from
    sixty login attempts an hour down to four."""
    if count < COMMANDS.breaker_threshold:
        return 0.0
    steps = min(count - COMMANDS.breaker_threshold, _MAX_BACKOFF_DOUBLINGS)
    return float(min(COMMANDS.breaker_base_seconds * 2 ** steps,
                     COMMANDS.breaker_max_seconds))


def _breaker_open(health: dict | None, now: datetime) -> bool:
    """True while the mailbox is deliberately being left alone. Fails open on
    anything unreadable: losing the kill switch to a malformed state file is far
    worse than one wasted login."""
    if not health:
        return False
    backoff = _breaker_backoff_seconds(health.get("count", 0))
    last = parse_utc(health.get("last"))
    if not backoff or last is None:
        return False
    return (now - last).total_seconds() < backoff


def _record_poll_failure(state, exc: Exception, now: datetime,
                        summary: str) -> None:
    state.record_command_poll_failure(f"{type(exc).__name__}: {exc}", now=now)
    count = (state.command_poll_health() or {}).get("count", 1)
    backoff = _breaker_backoff_seconds(count)
    waiting = f"; next attempt in {backoff / 60:.0f}m" if backoff else ""
    # The traceback earns its ~25 lines once: a throttled Gmail repeats the same
    # handful of stacks for hours, and the count carries the rest of the story.
    log.warning("%s (consecutive failure %d%s)", summary, count, waiting,
                exc_info=count == 1)


def _connect_with_retry():
    """Gmail's LOGIN latency has a tail longer than the deadline, and a slow
    login is not an unreachable mailbox: the attempt one cycle later has always
    succeeded. Retrying once immediately tells the two apart, so only a channel
    that fails twice in a row is called down.

    The retry stops at the connection. Everything past it marks mail seen, and
    replaying that risks handing back a command already counted as read."""
    try:
        return _connect()
    except Exception:
        log.debug("command mailbox connect failed; retrying once", exc_info=True)
        return _connect()


def _note_poll_recovery(state, now: datetime) -> None:
    """Transitions are logged, not every healthy poll: a line a cycle is 1,440 a
    day, and the outage is the part worth reading."""
    health = state.command_poll_health()
    if health is None:
        return
    since = parse_utc(health.get("since"))
    down = f" after {(now - since).total_seconds() / 60:.0f}m" if since else ""
    log.info("command mailbox reachable again%s (%d failed poll(s))", down,
             health.get("count", 0))
    state.clear_command_poll_failure()


def poll(state, now: datetime | None = None) -> list[Command]:
    """Verified commands waiting in the mailbox, in delivery order. Refused
    candidates are marked seen and answered here; accepted ones are marked seen
    and recorded before they are returned, so a crash mid-execution cannot
    replay them.

    A mailbox that has failed repeatedly is left alone for a widening interval
    instead of being retried every cycle — see _breaker_backoff_seconds."""
    if not commands_configured():
        return []
    secret = _secret()
    now = now or datetime.now(timezone.utc)
    if _breaker_open(state.command_poll_health(), now):
        return []
    try:
        conn = _connect_with_retry()
    except Exception as exc:
        _record_poll_failure(state, exc, now,
                             "command mailbox unreachable; no commands this cycle")
        return []
    accepted = []
    try:
        candidates, deferred = _unseen_candidates(conn, COMMANDS.max_per_cycle)
        if deferred:
            log.warning("command mailbox holds more than one cycle's work: %d "
                        "candidate(s) deferred (cap %d per cycle)", deferred,
                        COMMANDS.max_per_cycle)
        for uid, headers in candidates:
            # One candidate's failure must not cost the rest of the batch the
            # cycle — the same rule the per-symbol sweeps follow.
            try:
                command = _process_candidate(conn, uid, headers, secret, state)
            except Exception:
                log.warning("command mail %s could not be processed", uid,
                            exc_info=True)
                continue
            if command is not None:
                accepted.append(command)
    except Exception as exc:
        # Reaching the mailbox but failing to read it leaves the channel as
        # unusable as never connecting, so it counts the same: otherwise a
        # SELECT that always times out reads as a healthy channel forever and
        # neither the breaker nor the briefing alert can ever fire.
        _record_poll_failure(
            state, exc, now,
            f"command poll failed part-way; {len(accepted)} command(s) survive")
    else:
        _note_poll_recovery(state, now)
    finally:
        _close(conn)
    return accepted


def send_reply(request: str, sender: str, verdict: str) -> None:
    """Answer a command over the same SMTP path the alerts use. Best effort: a
    reply that cannot be sent must never undo a command that already ran.

    The subject deliberately does not start with the command prefix. Replies go
    to BRIEFING_TO, which is usually the command inbox itself: a reply the poller
    would treat as a candidate gets refused, answered, and loops forever."""
    body = "\n".join([f"Request: {request}", f"From: {sender}", "", verdict])
    try:
        mail.send_gmail(f"Bot command — {verdict.splitlines()[0][:60]}", body)
    except Exception:
        log.warning("command reply could not be sent: %s", verdict, exc_info=True)
