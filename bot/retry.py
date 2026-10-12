"""Retry transient errors on idempotent reads.

alpaca-py retries only on specific HTTP status codes, and requests' default
adapter retries nothing at the transport layer, so a stale keep-alive socket
closing mid-request surfaces as a raw ConnectionError and aborts the whole
cycle. Server-side 5xx APIErrors are equally transient on a read, so both are
retried here. 4xx APIErrors (auth, bad request) and APIErrors with no HTTP
status propagate immediately.

READS ONLY. A transport error on a write (order submit/cancel) is ambiguous:
the request may have reached Alpaca and taken effect before the socket dropped,
so a blind retry risks duplicate orders. Only HTTP GETs are safe to replay,
so this decorator is applied exclusively to read methods in bot/broker.py and
bot/data.py — never to order-submitting or order-cancelling methods.
"""

import functools
import logging
import time

from alpaca.common.exceptions import APIError
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout

log = logging.getLogger(__name__)

TRANSIENT_ERRORS = (RequestsConnectionError, Timeout)
RETRY_DELAYS = (0.5, 1.0)  # seconds between attempts; total attempts == len + 1


def _is_transient(exc):
    if isinstance(exc, TRANSIENT_ERRORS):
        return True
    if isinstance(exc, APIError):
        status = exc.status_code
        return status is not None and status >= 500
    return False


def retry_transient_reads(fn):
    """Retry `fn` on transient transport errors and 5xx APIErrors, sleeping
    RETRY_DELAYS between attempts, then re-raising the original error.
    Other exceptions (4xx APIError, ValueError, ...) propagate immediately."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        for attempt, delay in enumerate(RETRY_DELAYS, start=1):
            try:
                return fn(*args, **kwargs)
            except TRANSIENT_ERRORS + (APIError,) as exc:
                if not _is_transient(exc):
                    raise
                log.warning(
                    "transient read error in %s (attempt %d/%d): %s; retrying in %.1fs",
                    fn.__name__, attempt, len(RETRY_DELAYS) + 1, exc, delay,
                )
                time.sleep(delay)
        return fn(*args, **kwargs)

    return wrapper
