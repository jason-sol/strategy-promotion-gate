import json
import math
from datetime import date
from unittest.mock import MagicMock
from urllib.error import URLError

import pytest

from bot.earnings import (
    EarningsData,
    EarningsError,
    EarningsEvent,
    EarningsSurprise,
    standardized_surprise,
)

CALENDAR_JSON = {
    "earningsCalendar": [
        {"date": "2024-01-25", "hour": "bmo", "epsActual": 1.20, "epsEstimate": 1.10,
         "symbol": "AAPL"},
        {"date": "2024-04-25", "hour": "amc", "epsActual": None, "epsEstimate": 1.30,
         "symbol": "AAPL"},
        {"date": "2024-07-25", "hour": "", "epsActual": 1.40, "epsEstimate": 1.35,
         "symbol": "AAPL"},
        {"date": "2024-10-25", "hour": "DMH", "epsActual": 1.50, "epsEstimate": 1.45,
         "symbol": "AAPL"},
    ]
}

SURPRISES_JSON = [
    # Finnhub returns most-recent first.
    {"actual": 1.50, "estimate": 1.30, "surprise": 0.20, "period": "2024-09-30", "symbol": "AAPL"},
    {"actual": 1.40, "estimate": 1.35, "surprise": 0.05, "period": "2024-06-30", "symbol": "AAPL"},
    {"actual": 1.20, "estimate": 1.10, "surprise": 0.10, "period": "2024-03-31", "symbol": "AAPL"},
    {"actual": 1.00, "estimate": 1.05, "surprise": -0.05, "period": "2023-12-31", "symbol": "AAPL"},
]


def fake_response(payload):
    """A context-manager stand-in for urllib.request.urlopen's return value."""
    cm = MagicMock()
    cm.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    cm.__exit__.return_value = False
    return cm


def bad_json_response():
    cm = MagicMock()
    cm.__enter__.return_value.read.return_value = b"<html>not json</html>"
    cm.__exit__.return_value = False
    return cm


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_client(payloads, *, min_interval=1.1):
    """EarningsData wired to a scripted opener and injected clock/sleep.

    payloads: list of urlopen return values (or exceptions) consumed in order.
    """
    clock = FakeClock()
    slept = []

    def sleep(dt):
        slept.append(dt)
        clock.t += dt

    opener = MagicMock(side_effect=payloads)
    ed = EarningsData(api_key="test-key", min_interval=min_interval,
                      clock=clock, sleep=sleep, opener=opener)
    return ed, opener, slept, clock


class TestCalendarParsing:
    def test_parses_events(self):
        ed, opener, _, _ = make_client([fake_response(CALENDAR_JSON)])
        events = ed.earnings_calendar("AAPL", "2024-01-01", "2024-12-31")
        assert all(isinstance(e, EarningsEvent) for e in events)
        assert len(events) == 4
        first = events[0]
        assert first.date == date(2024, 1, 25)
        assert first.hour == "bmo"
        assert first.eps_actual == pytest.approx(1.20)
        assert first.eps_estimate == pytest.approx(1.10)

    def test_missing_actual_is_none(self):
        ed, _, _, _ = make_client([fake_response(CALENDAR_JSON)])
        events = ed.earnings_calendar("AAPL", "2024-01-01", "2024-12-31")
        assert events[1].eps_actual is None

    def test_url_carries_symbol_range_and_token(self):
        ed, opener, _, _ = make_client([fake_response(CALENDAR_JSON)])
        ed.earnings_calendar("AAPL", "2024-01-01", "2024-12-31")
        url = opener.call_args[0][0]
        assert "calendar/earnings" in url
        assert "symbol=AAPL" in url
        assert "from=2024-01-01" in url
        assert "to=2024-12-31" in url
        assert "token=test-key" in url


class TestHourMapping:
    @pytest.mark.parametrize("raw,expected", [
        ("bmo", "bmo"), ("amc", "amc"), ("dmh", "dmh"),
        ("BMO", "bmo"), ("DMH", "dmh"), ("", ""), (None, ""), ("garbage", ""),
    ])
    def test_hour_normalization(self, raw, expected):
        payload = {"earningsCalendar": [
            {"date": "2024-01-25", "hour": raw, "epsActual": 1.0, "epsEstimate": 1.0}]}
        ed, _, _, _ = make_client([fake_response(payload)])
        events = ed.earnings_calendar("AAPL", "2024-01-01", "2024-12-31")
        assert events[0].hour == expected


class TestSurprisesParsing:
    def test_parses_surprises(self):
        ed, _, _, _ = make_client([fake_response(SURPRISES_JSON)])
        rows = ed.earnings_surprises("AAPL")
        assert all(isinstance(r, EarningsSurprise) for r in rows)
        assert len(rows) == 4
        assert rows[0].actual == pytest.approx(1.50)
        assert rows[0].estimate == pytest.approx(1.30)
        assert rows[0].surprise == pytest.approx(0.20)
        assert rows[0].period == date(2024, 9, 30)


class TestSUE:
    def test_nan_when_history_short(self):
        assert math.isnan(standardized_surprise([0.1, 0.05, -0.05]))

    def test_computed_when_enough_history(self):
        deltas = [-0.05, 0.10, 0.05, 0.20]  # oldest -> newest, newest last
        import statistics
        expected = deltas[-1] / statistics.stdev(deltas)
        assert standardized_surprise(deltas) == pytest.approx(expected)

    def test_nan_when_zero_variance(self):
        assert math.isnan(standardized_surprise([0.1, 0.1, 0.1, 0.1]))

    def test_nan_values_dropped_then_short(self):
        assert math.isnan(standardized_surprise([0.1, float("nan"), 0.05, None]))

    def test_latest_sue_from_endpoint(self):
        ed, _, _, _ = make_client([fake_response(SURPRISES_JSON)])
        sue = ed.latest_sue("AAPL")
        # 4 quarters present; newest delta 0.20 over stdev of chronological deltas.
        deltas = [-0.05, 0.10, 0.05, 0.20]
        import statistics
        assert sue == pytest.approx(0.20 / statistics.stdev(deltas))


class TestRateLimit:
    def test_spacing_uses_injected_sleep(self):
        ed, _, slept, _ = make_client(
            [fake_response(CALENDAR_JSON), fake_response(SURPRISES_JSON)],
            min_interval=1.1)
        ed.earnings_calendar("AAPL", "2024-01-01", "2024-12-31")  # first: no wait
        assert slept == []
        ed.earnings_surprises("AAPL")  # immediate second call: spaced
        assert slept == [pytest.approx(1.1)]


class TestErrors:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        with pytest.raises(EarningsError):
            EarningsData(api_key=None)

    def test_malformed_json_raises_typed(self):
        ed, _, _, _ = make_client([bad_json_response()])
        with pytest.raises(EarningsError):
            ed.earnings_calendar("AAPL", "2024-01-01", "2024-12-31")

    def test_single_retry_on_urlerror_then_success(self):
        ed, opener, _, _ = make_client(
            [URLError("stale socket"), fake_response(CALENDAR_JSON)])
        events = ed.earnings_calendar("AAPL", "2024-01-01", "2024-12-31")
        assert len(events) == 4
        assert opener.call_count == 2

    def test_urlerror_twice_raises_typed(self):
        ed, _, _, _ = make_client([URLError("down"), URLError("down")])
        with pytest.raises(EarningsError):
            ed.earnings_calendar("AAPL", "2024-01-01", "2024-12-31")
