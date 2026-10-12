"""The kill switch.

The switch is the *existence* of a sentinel file, not a flag in memory. Three
properties follow from that, and all three are load-bearing:

Existence is authoritative. A sentinel that is empty, truncated, or holds
malformed JSON is still a halt — the payload is attribution for the briefing,
never the decision. A sentinel that is not JSON at all keeps its first line as
the reason, so an old sentinel or a hand-written `echo reason > HALT` still
tells the human why. An upgrade that started reading the payload as the decision
would read every pre-JSON sentinel as "not halted".

Reads fail closed. Absence has to be *proved*, and a stat that fails for any
reason other than "no such file" proves nothing. An unreadable sentinel halts.

Writes are atomic. Same-directory temp file then os.replace, so a crash or a
full disk mid-write can never destroy a sentinel that was already tripped, and
no reader can observe a half-written one.
"""
import json
import os
from datetime import datetime, timezone

from config import HALT_FILE


def _resolve(path: str | None) -> str:
    return path or HALT_FILE


def halt_tripped(path: str | None = None) -> bool:
    """Whether trading is halted. A pure filesystem check with no in-process
    state, so it holds when the run loop is wedged or the process has just
    restarted."""
    try:
        os.stat(_resolve(path))
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return True


def trip_halt(reason: str, path: str | None = None) -> str:
    """Write the sentinel. Idempotent: a fresh trip overwrites the prior one."""
    target = _resolve(path)
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        "tripped_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }
    tmp = os.path.join(parent, f".{os.path.basename(target)}.tmp")
    try:
        with open(tmp, "w") as f:
            f.write(json.dumps(payload))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def read_halt(path: str | None = None) -> dict | None:
    """Attribution for a tripped switch, or None when it is clear. An existing
    but unreadable sentinel returns {} — still halted, attribution just lost."""
    target = _resolve(path)
    if not halt_tripped(target):
        return None
    try:
        raw = open(target).read()
    except OSError:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Pre-JSON sentinels were a bare line of prose, and a human may still
        # trip the switch with `echo reason > HALT`. Keep the line as the reason.
        return _plaintext_attribution(raw)
    return parsed if isinstance(parsed, dict) else _plaintext_attribution(raw)


def _plaintext_attribution(raw: str) -> dict:
    first = raw.strip().splitlines()
    return {"reason": first[0]} if first else {}


def clear_halt(path: str | None = None) -> bool:
    """Remove the sentinel. True when one was there to remove."""
    try:
        os.unlink(_resolve(path))
    except FileNotFoundError:
        return False
    return True
