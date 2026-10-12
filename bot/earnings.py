"""Finnhub earnings client for the PEAD sleeve — SIGNAL-LOGGING-ONLY.

This feeds `bot.strategies.pead_drift.PeadDrift`, which is deliberately NOT in
the live lineup and NOT in research EXPERIMENTS. Reason: the free Finnhub tier
returns CURRENT consensus estimates and only ~4 quarters of surprise history,
neither of which is point-in-time. A backtest built on it would silently use
estimates that did not exist at the historical bar — lookahead that flatters
results. So the sleeve accumulates paper-forward evidence first; only once
enough genuinely-forward observations exist does promotion become honest.

Point-in-time caveats, made explicit for callers:
- `epsEstimate` is the estimate AS OF NOW, not as of the announcement.
- Surprise history is short on the free tier, so `standardized_surprise` (SUE)
  returns NaN until >= 4 quarters are available — expected and acceptable for
  forward logging.
- BMO ("before market open") vs AMC ("after market close") determines the first
  tradeable session: BMO announcements are actionable that day's bar, AMC (and
  unknown/empty) only from the next bar.

Stdlib only (urllib) — no new dependencies.
"""

import json
import math
import os
import statistics
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from urllib.error import URLError

_BASE = "https://finnhub.io/api/v1"
_VALID_HOURS = {"bmo", "amc", "dmh"}


class EarningsError(Exception):
    """Raised on HTTP, transport, or parse failure, and on missing API key."""


@dataclass(frozen=True)
class EarningsEvent:
    date: date
    hour: str  # "bmo" | "amc" | "dmh" | "" (normalized; unknown -> "")
    eps_actual: float | None
    eps_estimate: float | None


@dataclass(frozen=True)
class EarningsSurprise:
    actual: float | None
    estimate: float | None
    surprise: float | None
    period: date


def standardized_surprise(deltas) -> float:
    """SUE = most-recent (actual-estimate) / stdev of the trailing series.

    `deltas` are (actual - estimate) values in chronological order (newest
    LAST). NaN/None entries are dropped. Returns NaN when fewer than four
    usable observations remain (free-tier history is ~4 quarters, so early
    SUE is often NaN) or when the series has zero variance.
    """
    vals = [float(d) for d in deltas if d is not None and not math.isnan(float(d))]
    if len(vals) < 4:
        return float("nan")
    std = statistics.stdev(vals)
    if std == 0:
        return float("nan")
    return vals[-1] / std


def _normalize_hour(raw) -> str:
    if raw is None:
        return ""
    h = str(raw).strip().lower()
    return h if h in _VALID_HOURS else ""


def _to_float(v) -> float | None:
    return None if v is None else float(v)


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


class EarningsData:
    def __init__(self, api_key: str | None = None, *, timeout: float = 10.0,
                 min_interval: float = 1.1, sleep=time.sleep,
                 clock=time.monotonic, opener=urllib.request.urlopen):
        # Alpaca key pattern: python-dotenv is loaded by the entrypoint; read
        # the key straight from the environment here.
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            raise EarningsError("FINNHUB_API_KEY is not set")
        self.timeout = timeout
        # Free tier allows 60 calls/min; 1.1s spacing stays safely under it.
        self._min_interval = min_interval
        self._sleep = sleep
        self._clock = clock
        self._opener = opener
        self._last_request: float | None = None

    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = self._min_interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def _get(self, path: str, params: dict):
        query = dict(params)
        query["token"] = self.api_key
        url = f"{_BASE}/{path}?{urllib.parse.urlencode(query)}"
        self._throttle()
        # One retry: a stale keep-alive socket surfaces as a bare URLError.
        for attempt in range(2):
            try:
                with self._opener(url, timeout=self.timeout) as resp:
                    raw = resp.read()
                break
            except URLError as exc:
                if attempt == 1:
                    raise EarningsError(f"request failed: {url}") from exc
        try:
            return json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise EarningsError(f"malformed JSON from {url}") from exc

    def earnings_calendar(self, symbol: str, start: str, end: str) -> list[EarningsEvent]:
        data = self._get("calendar/earnings",
                         {"from": start, "to": end, "symbol": symbol})
        rows = data.get("earningsCalendar") or []
        return [
            EarningsEvent(
                date=_parse_date(r["date"]),
                hour=_normalize_hour(r.get("hour")),
                eps_actual=_to_float(r.get("epsActual")),
                eps_estimate=_to_float(r.get("epsEstimate")),
            )
            for r in rows
        ]

    def earnings_surprises(self, symbol: str) -> list[EarningsSurprise]:
        data = self._get("stock/earnings", {"symbol": symbol})
        rows = data or []
        return [
            EarningsSurprise(
                actual=_to_float(r.get("actual")),
                estimate=_to_float(r.get("estimate")),
                surprise=_to_float(r.get("surprise")),
                period=_parse_date(r["period"]),
            )
            for r in rows
        ]

    def latest_sue(self, symbol: str) -> float:
        """SUE for the most recent quarter from the surprises endpoint.

        NaN when the free tier returns < 4 quarters (the common early case).
        """
        rows = self.earnings_surprises(symbol)
        chronological = sorted(rows, key=lambda r: r.period)
        deltas = [
            r.actual - r.estimate
            for r in chronological
            if r.actual is not None and r.estimate is not None
        ]
        return standardized_surprise(deltas)
