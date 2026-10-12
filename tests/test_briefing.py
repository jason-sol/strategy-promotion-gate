from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd

from bot import briefing
from bot.halt import trip_halt
from bot.briefing import (
    build_evening_report,
    build_morning_report,
    collect_alerts,
    compose,
    due_briefing,
    execution_cost_lines,
)
from bot.portfolio import ExecutionLog
from bot.state import StateStore
from config import COMMANDS as CommandsConfig
from config import ProtectionsConfig, RiskConfig
from bot.strategies.base import PositionState

SYD = ZoneInfo("Australia/Sydney")


def briefing_broker(positions=None, equity=100_000.0, has_stop=True):
    """Broker stub carrying the three attributes bot.briefing reads — equity,
    positions, and whether a symbol has a resting stop. Used by the compose and
    alert suites below."""
    broker = MagicMock()
    broker.account_equity.return_value = equity
    broker.get_positions.return_value = dict(positions or {})
    broker.has_open_stop.return_value = has_stop
    return broker


def spy_long():
    return PositionState(symbol="SPY", side="long", qty=100, entry_price=560.0,
                         stop_price=550.0, high_water_mark=565.0)


class TestDueBriefing:
    def test_due_after_hour_when_not_sent_today(self):
        now = datetime(2026, 7, 22, 7, 30, tzinfo=SYD)
        assert due_briefing(now, hour=7, last_sent="2026-07-21")

    def test_not_due_before_hour(self):
        now = datetime(2026, 7, 22, 6, 59, tzinfo=SYD)
        assert not due_briefing(now, hour=7, last_sent="2026-07-21")

    def test_not_due_twice_same_day(self):
        now = datetime(2026, 7, 22, 8, 0, tzinfo=SYD)
        assert not due_briefing(now, hour=7, last_sent="2026-07-22")

    def test_due_when_never_sent(self):
        now = datetime(2026, 7, 22, 9, 0, tzinfo=SYD)
        assert due_briefing(now, hour=7, last_sent=None)


class TestMorningReport:
    def test_contains_positions_equity_and_winrate(self):
        subject, body = build_morning_report(
            equity=100_038.60,
            positions={"SPY": spy_long()},
            marks={"SPY": 563.0},
            trades_7d=[{"pnl": "120.0"}, {"pnl": "-60.0"}, {"pnl": "40.0"}],
            yesterday_pnl=38.60,
            peak_equity=100_100.0,
            halted=False,
        )
        assert "Morning briefing" in subject
        assert "100,038.60" in body
        assert "SPY" in body and "long" in body
        assert "+300.00" in body            # (563-560)*100 unrealized
        assert "67%" in body                # 2 of 3 wins
        assert "HALT" not in body

    def test_flags_halt_and_drawdown(self):
        _, body = build_morning_report(
            equity=91_000.0, positions={}, marks={}, trades_7d=[],
            yesterday_pnl=-500.0, peak_equity=100_000.0, halted=True,
        )
        assert "HALT" in body
        assert "9.0%" in body               # drawdown from peak


class TestMorningPositionRisk:
    """Vol targeting lets a sized entry risk up to risk_per_trade x
    vol_scalar_max; anything above that ceiling is a sizing fault, not a regime."""

    def _report(self, qty, equity=100_000.0):
        pos = PositionState(symbol="SPY", side="long", qty=qty, entry_price=560.0,
                            stop_price=550.0, high_water_mark=565.0)
        return build_morning_report(
            equity=equity, positions={"SPY": pos}, marks={"SPY": 563.0},
            trades_7d=[], yesterday_pnl=None, peak_equity=equity, halted=False)[1]

    def test_position_line_carries_risk_at_the_stop(self):
        # 100 shares x $10 to the stop, of $100k equity
        assert "risk 1.00%" in self._report(qty=100)

    def test_risk_at_the_scalar_ceiling_is_not_flagged(self):
        body = self._report(qty=150)            # 1.5% = risk_per_trade x 1.5
        assert "risk 1.50%" in body
        assert "Risk flags" not in body

    def test_risk_above_the_scalar_ceiling_is_flagged(self):
        body = self._report(qty=200)            # 2.0%
        assert "Risk flags" in body
        assert "SPY risks 2.00% of equity at its stop (ceiling 1.5%)" in body

    def test_ceiling_follows_the_configured_risk(self, monkeypatch):
        monkeypatch.setattr(briefing, "RISK",
                            RiskConfig(risk_per_trade=0.02, vol_scalar_max=2.0))
        body = self._report(qty=200)
        assert "risk 2.00%" in body
        assert "Risk flags" not in body


class TestEveningReport:
    def test_summarizes_todays_trades(self):
        trades = [
            {"instrument": "SPY", "direction": "long", "pnl": "150.0"},
            {"instrument": "USO", "direction": "long", "pnl": "-80.0"},
        ]
        subject, body = build_evening_report(
            equity=100_070.0, trades_today=trades, prev_equity=100_000.0)
        assert "Evening report" in subject
        assert "2 trade" in body
        assert "SPY" in body and "USO" in body
        assert "+70.00" in body             # equity change

    def test_quiet_day(self):
        _, body = build_evening_report(equity=100_000.0, trades_today=[],
                                       prev_equity=100_000.0)
        assert "No trades" in body


class TestComposeMorningStops:
    def _uso_long(self):
        return PositionState(symbol="USO", side="long", qty=10, entry_price=100.0,
                             stop_price=0.0, high_water_mark=100.0)

    def _data(self):
        data = MagicMock()
        data.latest_price.return_value = 110.0
        return data

    def _quiet_files(self, monkeypatch):
        monkeypatch.setattr(briefing, "latest_daily_pnl", lambda *a, **k: (None, None))
        monkeypatch.setattr(briefing, "load_trades_since", lambda *a, **k: [])

    def test_compose_morning_shows_trail_stop(self, tmp_path, monkeypatch):
        self._quiet_files(monkeypatch)
        monkeypatch.setattr(briefing, "_latest_atr", lambda data, cfg: 5.0)
        store = StateStore(str(tmp_path / "state.json"))
        store.set_position_meta("USO", 95.0, 120.0)
        _, body = compose("morning", briefing_broker({"USO": self._uso_long()}),
                          self._data(), state=store)
        assert "stop 105.00" in body           # max(95, 120 - 3*5)
        assert "stop 0.00" not in body

    def test_compose_morning_atr_failure_falls_back_to_entry_stop(self, tmp_path, monkeypatch):
        self._quiet_files(monkeypatch)

        def boom(data, cfg):
            raise RuntimeError("no bars")

        monkeypatch.setattr(briefing, "_latest_atr", boom)
        store = StateStore(str(tmp_path / "state.json"))
        store.set_position_meta("USO", 95.0, 120.0)
        _, body = compose("morning", briefing_broker({"USO": self._uso_long()}),
                          self._data(), state=store)
        assert "stop 95.00" in body

    def test_compose_morning_no_state_meta_uses_fallback_not_zero(self, tmp_path, monkeypatch):
        self._quiet_files(monkeypatch)

        def boom(data, cfg):
            raise RuntimeError("no bars")

        monkeypatch.setattr(briefing, "_latest_atr", boom)
        store = StateStore(str(tmp_path / "state.json"))  # empty: no meta for USO
        _, body = compose("morning", briefing_broker({"USO": self._uso_long()}),
                          self._data(), state=store)
        assert "stop 99.00" in body            # entry 100 * 0.99
        assert "stop 0.00" not in body

    def test_compose_morning_non_trailing_instrument_shows_entry_stop(self, tmp_path, monkeypatch):
        self._quiet_files(monkeypatch)
        atr = MagicMock(side_effect=AssertionError("_latest_atr must not be called"))
        monkeypatch.setattr(briefing, "_latest_atr", atr)
        spy = PositionState(symbol="SPY", side="long", qty=100, entry_price=560.0,
                            stop_price=0.0, high_water_mark=560.0)
        store = StateStore(str(tmp_path / "state.json"))
        store.set_position_meta("SPY", 550.0, 565.0)
        _, body = compose("morning", briefing_broker({"SPY": spy}), self._data(), state=store)
        assert "stop 550.00" in body
        atr.assert_not_called()


class TestAlerts:
    def _store(self, tmp_path):
        return StateStore(str(tmp_path / "state.json"))

    def test_failed_exits_listed_with_age_and_reason(self, tmp_path):
        store = self._store(tmp_path)
        store.record_exit_failure("SPY", "insufficient qty available")
        alerts = collect_alerts(briefing_broker(), store, positions={},
                                halt_path=str(tmp_path / "HALT"))
        assert len(alerts) == 1
        assert "SPY" in alerts[0] and "insufficient qty available" in alerts[0]

    def test_position_without_resting_stop_listed(self, tmp_path):
        alerts = collect_alerts(briefing_broker(has_stop=False), self._store(tmp_path),
                                positions={"SPY": spy_long()},
                                halt_path=str(tmp_path / "HALT"))
        assert any("SPY" in a and "stop" in a.lower() for a in alerts)

    def test_crypto_position_not_flagged_for_missing_stop(self, tmp_path):
        btc = PositionState(symbol="BTC/USD", side="long", qty=0.3,
                            entry_price=50_000.0, stop_price=47_000.0,
                            high_water_mark=50_000.0)
        alerts = collect_alerts(briefing_broker(has_stop=False), self._store(tmp_path),
                                positions={"BTC/USD": btc},
                                halt_path=str(tmp_path / "HALT"))
        assert alerts == []  # crypto stops are software-managed

    def test_halt_file_listed_with_reason(self, tmp_path):
        halt = tmp_path / "HALT"
        halt.write_text("drawdown halt at 2026-08-30T12:00:00+00:00 equity=85000.00\n")
        alerts = collect_alerts(briefing_broker(), self._store(tmp_path), positions={},
                                halt_path=str(halt))
        assert any("HALT" in a and "drawdown halt" in a for a in alerts)

    def test_json_sentinel_reason_is_listed(self, tmp_path):
        halt = tmp_path / "HALT"
        trip_halt("drawdown halt: equity=85000.00 peak=100000.00", str(halt))
        alerts = collect_alerts(briefing_broker(), self._store(tmp_path), positions={},
                                halt_path=str(halt))
        alert = next(a for a in alerts if "HALT" in a)
        assert "drawdown halt: equity=85000.00 peak=100000.00" in alert
        assert "tripped_at" not in alert  # the raw payload must not reach the email

    def test_unreadable_sentinel_still_alerts(self, tmp_path):
        halt = tmp_path / "HALT"
        halt.write_text("")
        alerts = collect_alerts(briefing_broker(), self._store(tmp_path), positions={},
                                halt_path=str(halt))
        assert any("HALT" in a for a in alerts)

    def test_cancel_failure_is_listed(self, tmp_path):
        store = self._store(tmp_path)
        store.record_cancel_failure("BTC/USD", "api down")
        alerts = collect_alerts(briefing_broker(), store, positions={},
                                halt_path=str(tmp_path / "HALT"))
        assert any("BTC/USD" in a and "cancel" in a.lower() for a in alerts)

    def test_entry_refusal_is_listed(self, tmp_path):
        store = self._store(tmp_path)
        store.record_entry_refusal("SPY", "order status is rejected")
        alerts = collect_alerts(briefing_broker(), store, positions={},
                                halt_path=str(tmp_path / "HALT"))
        assert any("SPY" in a and "entry" in a.lower() for a in alerts)

    def test_clean_state_has_no_alerts(self, tmp_path):
        alerts = collect_alerts(briefing_broker(), self._store(tmp_path),
                                positions={"SPY": spy_long()},
                                halt_path=str(tmp_path / "HALT"))
        assert alerts == []

    def test_stop_check_failure_does_not_break_alerts(self, tmp_path):
        broker = MagicMock()
        broker.has_open_stop.side_effect = RuntimeError("api down")
        alerts = collect_alerts(broker, self._store(tmp_path),
                                positions={"SPY": spy_long()},
                                halt_path=str(tmp_path / "HALT"))
        assert alerts == []


class TestReportsCarryAlerts:
    def test_morning_report_surfaces_alerts_and_flags_subject(self):
        subject, body = build_morning_report(
            equity=100_000.0, positions={}, marks={}, trades_7d=[],
            yesterday_pnl=0.0, peak_equity=100_000.0, halted=False,
            alerts=["SPY: exit FAILED 31x since 2026-07-30 — insufficient qty"],
        )
        assert "ALERT" in subject
        assert "Alerts:" in body
        assert "exit FAILED 31x" in body

    def test_evening_report_surfaces_alerts(self):
        subject, body = build_evening_report(
            equity=100_000.0, trades_today=[], prev_equity=100_000.0,
            alerts=["SPY: no resting protective stop at the broker"])
        assert "ALERT" in subject
        assert "no resting protective stop" in body

    def test_no_alerts_leaves_reports_unchanged(self):
        subject, body = build_morning_report(
            equity=100_000.0, positions={}, marks={}, trades_7d=[],
            yesterday_pnl=0.0, peak_equity=100_000.0, halted=False, alerts=[])
        assert "ALERT" not in subject
        assert "Alerts:" not in body


class TestComposeAlerts:
    def test_compose_morning_includes_alerts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(briefing, "latest_daily_pnl", lambda *a, **k: (None, None))
        monkeypatch.setattr(briefing, "load_trades_since", lambda *a, **k: [])
        monkeypatch.setattr(briefing, "_latest_atr", lambda data, cfg: 5.0)
        store = StateStore(str(tmp_path / "state.json"))
        store.record_exit_failure("SPY", "insufficient qty available")
        data = MagicMock()

        subject, body = compose("morning", briefing_broker(), data, state=store)

        assert "ALERT" in subject
        assert "insufficient qty available" in body

    def test_compose_survives_alert_collection_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(briefing, "latest_daily_pnl", lambda *a, **k: (None, None))
        monkeypatch.setattr(briefing, "load_trades_since", lambda *a, **k: [])

        def boom(*a, **k):
            raise RuntimeError("state unreadable")

        monkeypatch.setattr(briefing, "collect_alerts", boom)
        subject, _ = compose("evening", briefing_broker(), MagicMock())
        assert "Evening report" in subject


# ---------------------------------------------------------------------------
# Task 8b/8a — divergence alerts and forming-bar-free ATR
# ---------------------------------------------------------------------------

class TestDivergenceAlerts:
    def test_distrusted_bar_listed_with_age_and_detail(self, tmp_path):
        store = StateStore(str(tmp_path / "state.json"))
        store.record_data_divergence("USO", "close 120.46 is 6.8% from the mid 129.22")
        broker = MagicMock()
        broker.has_open_stop.return_value = True
        alerts = collect_alerts(broker, store, positions={},
                                halt_path=str(tmp_path / "HALT"))
        assert len(alerts) == 1
        assert "USO" in alerts[0] and "120.46" in alerts[0]


class TestLatestAtrBarSelection:
    def test_forming_daily_bar_is_excluded(self, monkeypatch):
        from config import INSTRUMENTS

        cfg = next(c for c in INSTRUMENTS if c.symbol == "USO")
        now = datetime.now(timezone.utc)
        et_midnight = (now.astimezone(ZoneInfo("America/New_York"))
                       .replace(hour=0, minute=0, second=0, microsecond=0))
        idx = pd.DatetimeIndex(
            [et_midnight - timedelta(days=d) for d in range(60, 0, -1)]
            + [et_midnight])
        frame = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                              "close": 100.0, "volume": 1_000.0}, index=idx)
        frame.iloc[-1, frame.columns.get_loc("high")] = 900.0  # forming-bar spike
        frame.iloc[-1, frame.columns.get_loc("low")] = 1.0
        data = MagicMock()
        data.get_recent_bars.return_value = frame

        # The forming bar's absurd range must not reach ATR.
        assert briefing._latest_atr(data, cfg) < 10.0


class TestComposeHaltFlag:
    """The morning body's halt banner must come from the same fail-closed read
    as the trading gate, or the email says 'trading' while the bot is halted."""

    def test_morning_body_flags_an_unreadable_sentinel_as_halted(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.HALT_FILE", str(tmp_path / "HALT"))
        monkeypatch.setattr("bot.briefing.halt_tripped", lambda path=None: True)
        monkeypatch.setattr("bot.briefing.latest_daily_pnl", lambda: (0.0, 100_000.0))
        monkeypatch.setattr("bot.briefing.load_trades_since", lambda since: [])

        _, body = compose("morning", briefing_broker(), MagicMock(),
                          peak_equity=100_000.0, state=StateStore(str(tmp_path / "s.json")))

        assert "HALT" in body

    def test_morning_body_is_clean_when_the_switch_is_clear(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.HALT_FILE", str(tmp_path / "HALT"))
        monkeypatch.setattr("bot.briefing.latest_daily_pnl", lambda: (0.0, 100_000.0))
        monkeypatch.setattr("bot.briefing.load_trades_since", lambda since: [])

        _, body = compose("morning", briefing_broker(), MagicMock(),
                          peak_equity=100_000.0, state=StateStore(str(tmp_path / "s.json")))

        assert "HALT" not in body


class TestShortWindowAlerts:
    def test_short_window_listed_with_age_and_detail(self, tmp_path):
        store = StateStore(str(tmp_path / "state.json"))
        store.record_short_window("VOO", "21 bars available, 25 needed")
        broker = MagicMock()
        broker.has_open_stop.return_value = True
        alerts = collect_alerts(broker, store, positions={},
                                halt_path=str(tmp_path / "HALT"))
        assert len(alerts) == 1
        assert "VOO" in alerts[0] and "21 bars available" in alerts[0]

    def test_no_short_window_no_alert(self, tmp_path):
        store = StateStore(str(tmp_path / "state.json"))
        broker = MagicMock()
        broker.has_open_stop.return_value = True
        assert collect_alerts(broker, store, positions={},
                              halt_path=str(tmp_path / "HALT")) == []


class TestExecutionCostLines:
    def _fills(self, tmp_path, rows):
        """rows: (symbol, event, side, slip) or (symbol, event, side, slip, age_days)."""
        path = tmp_path / "fills.csv"
        log = ExecutionLog(str(path))
        for row in rows:
            symbol, event, side, slip = row[:4]
            age = row[4] if len(row) > 4 else 0
            spread = row[5] if len(row) > 5 else 2.0
            mid = 100.0 if slip != "" else "nan"
            when = datetime.now(timezone.utc) - timedelta(days=age)
            log.record(when.isoformat(), symbol, side, event, mid, spread,
                       100.0, 100.0, 10, slip)
        return str(path)

    def test_no_log_renders_nothing(self, tmp_path):
        assert execution_cost_lines(str(tmp_path / "absent.csv")) == []

    def test_log_without_fills_renders_nothing(self, tmp_path):
        assert execution_cost_lines(self._fills(tmp_path, [])) == []

    def test_entries_and_exits_are_listed_with_counts_and_medians(self, tmp_path):
        path = self._fills(tmp_path, [("SPY", "entry", "buy", 2.0),
                                      ("SPY", "entry", "buy", 4.0),
                                      ("SPY", "exit", "sell", -1.0)])
        body = "\n".join(execution_cost_lines(path))
        assert "Execution costs" in body
        entry = next(ln for ln in body.splitlines() if "entry" in ln)
        exit_ = next(ln for ln in body.splitlines() if "exit" in ln)
        assert "n=2" in entry and "+3.0bps" in entry
        assert "n=1" in exit_ and "-1.0bps" in exit_

    def test_the_header_names_the_trailing_window(self, tmp_path):
        path = self._fills(tmp_path, [("SPY", "entry", "buy", 2.0)])
        assert "last 30d" in execution_cost_lines(path)[0]

    def test_fills_older_than_the_window_are_excluded(self, tmp_path):
        path = self._fills(tmp_path, [("SPY", "entry", "buy", 2.0, 60)])
        assert execution_cost_lines(path) == []

    def test_a_fill_inside_the_window_survives(self, tmp_path):
        path = self._fills(tmp_path, [("SPY", "entry", "buy", 2.0, 29)])
        assert execution_cost_lines(path) != []

    def test_median_spread_is_shown_beside_the_measurement(self, tmp_path):
        """The spread at decision is what explains an OVER."""
        path = self._fills(tmp_path, [("SPY", "entry", "buy", 9.0, 0, 7.5)])
        line = execution_cost_lines(path)[1]
        assert "spread" in line and "7.5" in line

    def test_lines_are_padded_for_monospace_alignment(self, tmp_path):
        path = self._fills(tmp_path, [("SPY", "entry", "buy", 2.0),
                                      ("BTC/USD", "entry", "buy", 2.0)])
        lines = [ln for ln in execution_cost_lines(path) if "median" in ln]
        assert len(lines) == 2
        assert len({ln.index("median") for ln in lines}) == 1

    def test_assumed_cost_is_the_configured_fraction_in_bps(self, tmp_path):
        # SPY is configured at slippage_pct=0.0001, i.e. 1 bp.
        path = self._fills(tmp_path, [("SPY", "entry", "buy", 0.5)])
        assert "assumed 1.0bps" in "\n".join(execution_cost_lines(path))

    def test_a_thin_sample_over_assumed_is_qualified_not_shouted(self, tmp_path):
        """One bad fill is not evidence: the daily email must not cry wolf."""
        path = self._fills(tmp_path, [("SPY", "entry", "buy", 9.0)])
        assert "OVER (n<20)" in "\n".join(execution_cost_lines(path))

    def test_a_full_sample_over_assumed_is_flagged_plainly(self, tmp_path):
        path = self._fills(tmp_path, [("SPY", "entry", "buy", 9.0)] * 20)
        body = "\n".join(execution_cost_lines(path))
        assert "OVER" in body
        assert "n<20" not in body

    def test_measured_below_assumed_is_not_flagged(self, tmp_path):
        path = self._fills(tmp_path, [("SPY", "entry", "buy", 0.2)])
        assert "OVER" not in "\n".join(execution_cost_lines(path))

    def test_a_symbol_outside_the_lineup_still_reports_its_measurement(self, tmp_path):
        path = self._fills(tmp_path, [("TLT", "entry", "buy", 3.0)])
        body = "\n".join(execution_cost_lines(path))
        assert "TLT" in body and "+3.0bps" in body
        assert "OVER" not in body

    def test_unmeasurable_stop_fills_are_named_rather_than_hidden(self, tmp_path):
        """Broker stops book with no decision quote. Reporting only the
        measurable exits would understate exactly the costliest fills."""
        path = self._fills(tmp_path, [("VOO", "stop_fill", "sell", "")] * 7)
        line = next(ln for ln in execution_cost_lines(path) if "VOO" in ln)
        assert "stop" in line
        assert "0 measurable (7 booked, no decision quote)" in line

    def test_a_partly_measurable_kind_still_names_the_gap(self, tmp_path):
        path = self._fills(tmp_path, [("SPY", "entry", "buy", 2.0),
                                      ("SPY", "entry", "buy", "")])
        line = next(ln for ln in execution_cost_lines(path) if "SPY" in ln)
        assert "n=1" in line
        assert "1 unmeasurable" in line


class TestEveningReportExecutionCosts:
    def test_section_is_rendered_when_lines_are_supplied(self):
        _, body = build_evening_report(
            equity=100_000.0, trades_today=[], prev_equity=None,
            cost_lines=["Execution costs:", "  SPY entry: n=2 median +3.0bps"])
        assert "Execution costs:" in body
        assert "SPY entry: n=2 median +3.0bps" in body

    def test_section_is_absent_without_fills(self):
        _, body = build_evening_report(equity=100_000.0, trades_today=[],
                                       prev_equity=None, cost_lines=[])
        assert "Execution costs" not in body

    def test_omitting_the_argument_leaves_the_report_unchanged(self):
        _, with_none = build_evening_report(100_000.0, [], None)
        _, with_empty = build_evening_report(100_000.0, [], None, cost_lines=[])
        assert with_none == with_empty


class TestComposeExecutionCosts:
    def test_evening_body_carries_the_costs_section(self, monkeypatch):
        monkeypatch.setattr(briefing, "latest_daily_pnl", lambda *a, **k: (None, None))
        monkeypatch.setattr(briefing, "load_trades_since", lambda *a, **k: [])
        monkeypatch.setattr(briefing, "execution_cost_lines",
                            lambda *a, **k: ["Execution costs:", "  SPY entry: n=1"])
        _, body = compose("evening", briefing_broker(), MagicMock())
        assert "SPY entry: n=1" in body

    def test_a_failing_costs_probe_never_kills_the_email(self, monkeypatch):
        monkeypatch.setattr(briefing, "latest_daily_pnl", lambda *a, **k: (None, None))
        monkeypatch.setattr(briefing, "load_trades_since", lambda *a, **k: [])

        def boom(*a, **k):
            raise RuntimeError("fills.csv is a directory")

        monkeypatch.setattr(briefing, "execution_cost_lines", boom)
        subject, body = compose("evening", briefing_broker(), MagicMock())
        assert "Evening report" in subject
        assert "Equity" in body

    def test_the_morning_briefing_does_not_carry_the_section(self, monkeypatch):
        monkeypatch.setattr(briefing, "latest_daily_pnl", lambda *a, **k: (None, None))
        monkeypatch.setattr(briefing, "load_trades_since", lambda *a, **k: [])
        monkeypatch.setattr(briefing, "execution_cost_lines",
                            lambda *a, **k: ["Execution costs:", "  SPY entry: n=1"])
        _, body = compose("morning", briefing_broker(), MagicMock())
        assert "Execution costs" not in body


class TestCooldownAlerts:
    def _store(self, tmp_path):
        return StateStore(str(tmp_path / "state.json"))

    def _alerts(self, tmp_path, store):
        broker = MagicMock()
        broker.has_open_stop.return_value = True
        return collect_alerts(broker, store, positions={},
                              halt_path=str(tmp_path / "HALT"))

    def test_active_cooldown_is_listed_with_its_expiry(self, tmp_path):
        store = self._store(tmp_path)
        until = datetime.now(timezone.utc) + timedelta(days=5)
        store.start_cooldown("USO", until, 3, 5)

        alerts = self._alerts(tmp_path, store)

        assert len(alerts) == 1
        assert "USO" in alerts[0]
        assert "PAUSED 5d" in alerts[0]
        assert "3 consecutive losses" in alerts[0]
        assert until.date().isoformat() in alerts[0]

    def test_expired_cooldown_is_not_listed(self, tmp_path):
        store = self._store(tmp_path)
        store.start_cooldown("USO", datetime.now(timezone.utc) - timedelta(days=1), 3, 5)
        assert self._alerts(tmp_path, store) == []

    def test_an_unusable_record_is_flagged_for_repair(self, tmp_path):
        store = self._store(tmp_path)
        store.data["cooldowns"]["USO"] = {"losses": 3, "days": 5,
                                         "until": "next tuesday"}

        alerts = self._alerts(tmp_path, store)

        assert len(alerts) == 1
        assert "USO" in alerts[0] and "unusable" in alerts[0]
        assert "state.json" in alerts[0]

    def test_a_record_with_no_until_is_flagged_for_repair(self, tmp_path):
        store = self._store(tmp_path)
        store.data["cooldowns"]["USO"] = {"losses": 3, "days": 5}
        assert "unusable" in self._alerts(tmp_path, store)[0]

    def test_a_cooldown_lapsed_by_time_is_not_flagged_as_unusable(self, tmp_path):
        store = self._store(tmp_path)
        store.start_cooldown("USO", datetime.now(timezone.utc) - timedelta(days=1),
                             3, 5)
        assert "unusable" not in "".join(self._alerts(tmp_path, store))

    def test_an_active_cooldown_is_not_flagged_as_unusable(self, tmp_path):
        store = self._store(tmp_path)
        store.start_cooldown("USO", datetime.now(timezone.utc) + timedelta(days=5),
                             3, 5)
        assert "unusable" not in "".join(self._alerts(tmp_path, store))


class TestDrawdownFlag:
    def _body(self, equity, peak):
        return build_morning_report(
            equity=equity, positions={}, marks={}, trades_7d=[], yesterday_pnl=None,
            peak_equity=peak, halted=False)[1]

    def test_a_drawdown_past_the_floor_is_flagged_with_the_configured_halt(self):
        assert ("drawdown 6.0% from peak (halt at 10%)"
                in self._body(94_000.0, 100_000.0))

    def test_a_drawdown_under_the_floor_is_not_flagged(self):
        assert "from peak" not in self._body(97_000.0, 100_000.0)

    def test_the_floor_tracks_the_breaker_threshold(self, monkeypatch):
        """Halfway to the forced flatten, wherever the breaker is set."""
        monkeypatch.setattr(briefing, "RISK", RiskConfig(max_drawdown=0.20))
        assert "from peak" not in self._body(92_000.0, 100_000.0)
        assert ("drawdown 12.0% from peak (halt at 20%)"
                in self._body(88_000.0, 100_000.0))

    def test_a_drawdown_exactly_at_the_floor_is_not_flagged(self, monkeypatch):
        """The floor is a threshold to exceed. 87_500/100_000 lands on 12.5%
        exactly in binary, so this pins the comparison and not a rounding."""
        monkeypatch.setattr(briefing, "RISK", RiskConfig(max_drawdown=0.25))
        assert "from peak" not in self._body(87_500.0, 100_000.0)

    def test_a_fractional_threshold_is_not_rounded_away(self, monkeypatch):
        """Rounding the halt to a whole percent tells the reader the breaker
        sits somewhere it does not."""
        monkeypatch.setattr(briefing, "RISK", RiskConfig(max_drawdown=0.125))
        assert "(halt at 12.5%)" in self._body(90_000.0, 100_000.0)
        monkeypatch.setattr(briefing, "RISK", RiskConfig(max_drawdown=0.175))
        assert "(halt at 17.5%)" in self._body(85_000.0, 100_000.0)

    def test_a_zero_threshold_still_arms_the_flag(self, monkeypatch):
        """No `> 0` guard here on purpose: unlike the daily-loss limit, zero
        max_drawdown halts on any drawdown at all rather than disabling the
        breaker, so the flag has to stay armed."""
        monkeypatch.setattr(briefing, "RISK", RiskConfig(max_drawdown=0.0))
        assert "from peak" in self._body(99_999.0, 100_000.0)

    def test_a_zero_threshold_does_not_read_as_disabled(self, monkeypatch):
        """"halt at 0%" reads as off in this codebase's idiom; the breaker is
        instead maximally sensitive."""
        monkeypatch.setattr(briefing, "RISK", RiskConfig(max_drawdown=0.0))
        body = self._body(99_999.0, 100_000.0)
        assert "halts on any drawdown" in body
        assert "halt at 0%" not in body


class TestDailyLossFlag:
    def _body(self, equity, day_start):
        return build_morning_report(
            equity=equity, positions={}, marks={}, trades_7d=[], yesterday_pnl=None,
            peak_equity=100_000.0, halted=False, day_start_equity=day_start)[1]

    def test_down_on_the_day_is_flagged_with_the_configured_threshold(self):
        assert "down 2.0% on the day (halt at 3%)" in self._body(98_000.0, 100_000.0)

    def test_up_on_the_day_is_not_flagged(self):
        assert "on the day" not in self._body(101_000.0, 100_000.0)

    def test_a_loss_under_the_noise_floor_is_not_flagged(self):
        """The flag warns that the halt is being approached, so it has to stay
        quiet on an ordinary red day: firing on any loss at all trains the
        reader to skip the section that carries the real warnings."""
        assert "on the day" not in self._body(99_000.0, 100_000.0)

    def test_the_floor_tracks_the_halt_threshold(self, monkeypatch):
        """Same early-warning ratio as the drawdown flag: halfway to the halt,
        wherever the halt is set."""
        monkeypatch.setattr(briefing, "PROTECTIONS",
                            ProtectionsConfig(daily_loss_limit_pct=0.10))
        assert "on the day" not in self._body(96_000.0, 100_000.0)
        assert "down 6.0% on the day" in self._body(94_000.0, 100_000.0)

    def test_a_fractional_threshold_is_not_rounded_away(self, monkeypatch):
        """Rounding the halt to a whole percent tells the reader the breaker
        sits somewhere it does not."""
        monkeypatch.setattr(briefing, "PROTECTIONS",
                            ProtectionsConfig(daily_loss_limit_pct=0.035))
        assert "(halt at 3.5%)" in self._body(97_000.0, 100_000.0)
        monkeypatch.setattr(briefing, "PROTECTIONS",
                            ProtectionsConfig(daily_loss_limit_pct=0.125))
        assert "(halt at 12.5%)" in self._body(90_000.0, 100_000.0)

    def test_a_disabled_limit_is_not_flagged(self, monkeypatch):
        """Reachable through the documented remedy: turn the knob off and
        redeploy on a day the engine has already seeded a reference for."""
        monkeypatch.setattr(briefing, "PROTECTIONS",
                            ProtectionsConfig(daily_loss_limit_pct=0.0))
        assert "on the day" not in self._body(98_000.0, 100_000.0)

    def test_no_reference_yet_is_not_flagged(self):
        assert "on the day" not in self._body(98_000.0, None)

    def _compose_morning(self, monkeypatch, store, equity):
        monkeypatch.setattr(briefing, "latest_daily_pnl", lambda *a, **k: (None, None))
        monkeypatch.setattr(briefing, "load_trades_since", lambda *a, **k: [])
        return compose("morning", briefing_broker(equity=equity), MagicMock(),
                       state=store)[1]

    def test_compose_reads_todays_persisted_reference(self, tmp_path, monkeypatch):
        store = StateStore(str(tmp_path / "state.json"))
        store.data["day_start_equity"] = {
            "date": datetime.now(timezone.utc).date().isoformat(),
            "equity": 100_000.0}

        assert "down 2.0% on the day" in self._compose_morning(monkeypatch, store,
                                                               98_000.0)

    def test_a_stale_reference_is_ignored(self, tmp_path, monkeypatch):
        store = StateStore(str(tmp_path / "state.json"))
        store.data["day_start_equity"] = {"date": "2020-01-01", "equity": 100_000.0}

        assert "on the day" not in self._compose_morning(monkeypatch, store, 98_000.0)

    def test_composing_a_briefing_never_seeds_the_reference(self, tmp_path,
                                                            monkeypatch):
        """The briefing also runs from the CLI, against the live bot's state
        file: reading it must not stamp a day-start the engine never saw."""
        store = StateStore(str(tmp_path / "state.json"))

        self._compose_morning(monkeypatch, store, 98_000.0)

        assert store.data["day_start_equity"] == {}


class TestAlertChannelContract:
    """The briefing reads count, since and one detail field off every alert
    channel, so the five channels have to record identically. A channel that
    drifts to its own implementation changes what its alert line means."""

    # channel, record, clear, read, detail field
    TRIPLES = [
        ("exit_failures", "record_exit_failure", "clear_exit_failure",
         "exit_failure", "error"),
        ("entry_refusals", "record_entry_refusal", "clear_entry_refusal",
         "entry_refusal", "error"),
        ("cancel_failures", "record_cancel_failure", "clear_cancel_failure",
         "cancel_failure", "error"),
        ("data_divergences", "record_data_divergence", "clear_data_divergence",
         "data_divergence", "detail"),
        ("short_windows", "record_short_window", "clear_short_window",
         "short_window", "detail"),
    ]

    def test_every_channel_ages_counts_and_clears_alike(self, tmp_path):
        store = StateStore(str(tmp_path / "state.json"))
        for channel, record, clear, read, detail in self.TRIPLES:
            getattr(store, record)("SPY", "first")
            first = getattr(store, read)("SPY")
            # A counter only survives the restart the operator reads it after if
            # recording persists it.
            with patch.object(store, "save", wraps=store.save) as save:
                getattr(store, record)("SPY", "second")
            assert save.called, channel
            payload = getattr(store, read)("SPY")

            assert list(payload) == [detail, "since", "count"], channel
            assert payload["since"] == first["since"], channel
            assert payload["count"] == 2, channel
            assert payload[detail] == "second", channel

            getattr(store, clear)("SPY")
            assert getattr(store, read)("SPY") is None, channel

    def test_no_channel_writes_when_clearing_an_absent_symbol(self, tmp_path):
        store = StateStore(str(tmp_path / "state.json"))
        for channel, _, clear, _, _ in self.TRIPLES:
            with patch.object(store, "save") as save:
                getattr(store, clear)("SPY")
            assert not save.called, channel


class TestCommandChannelAlert:
    """The email channel carries the remote kill switch. An outage that only
    ever reaches a log line is an outage nobody finds out about."""

    DOWN_AT = datetime(2026, 9, 4, 11, 0, tzinfo=timezone.utc)
    THRESHOLD = CommandsConfig.down_alert_seconds
    BUDGET = CommandsConfig.refusal_replies_per_day

    def _store(self, tmp_path):
        return StateStore(str(tmp_path / "state.json"))

    def _down_since(self, tmp_path, count=3):
        store = self._store(tmp_path)
        for _ in range(count):
            store.record_command_poll_failure("TimeoutError: read timed out",
                                              now=self.DOWN_AT)
        return store

    def _alerts(self, store, tmp_path, now):
        return collect_alerts(briefing_broker(), store, positions={},
                              halt_path=str(tmp_path / "HALT"), now=now)

    def _spent_replies(self, tmp_path, replies):
        store = self._store(tmp_path)
        for _ in range(replies):
            store.claim_refusal_reply(self.BUDGET)
        return store

    def test_a_healthy_channel_is_not_mentioned(self, tmp_path):
        alerts = self._alerts(self._store(tmp_path), tmp_path, self.DOWN_AT)
        assert alerts == []

    def test_a_passing_throttle_stays_quiet(self, tmp_path):
        """Gmail throttles for minutes at a time; alerting on that would train
        the reader to ignore the line that matters."""
        store = self._down_since(tmp_path)
        now = self.DOWN_AT + timedelta(seconds=self.THRESHOLD - 1)

        assert self._alerts(store, tmp_path, now) == []

    def test_a_sustained_outage_is_alerted(self, tmp_path):
        store = self._down_since(tmp_path, count=7)
        now = self.DOWN_AT + timedelta(seconds=self.THRESHOLD)

        alert = next(a for a in self._alerts(store, tmp_path, now)
                     if "command channel" in a.lower())

        assert "7" in alert                          # how many polls failed
        assert "read timed out" in alert             # and why

    def test_the_alert_names_the_out_of_band_route(self, tmp_path):
        """Remote halt is exactly what is unavailable, so the alert has to point
        at the route that still works."""
        store = self._down_since(tmp_path)
        now = self.DOWN_AT + timedelta(hours=3)

        alert = next(a for a in self._alerts(store, tmp_path, now)
                     if "command channel" in a.lower())

        assert "HALT" in alert

    def test_an_unreadable_since_stamp_still_alerts(self, tmp_path):
        """Fail loud: a corrupt stamp must not be the reason a dead kill switch
        goes unreported."""
        store = self._down_since(tmp_path)
        store.data["command_poll_failure"]["since"] = "not-a-timestamp"

        alerts = self._alerts(store, tmp_path, self.DOWN_AT)

        assert any("command channel" in a.lower() for a in alerts)

    def test_a_spent_refusal_budget_is_alerted(self, tmp_path):
        """Past the budget, junk CMD mail is dropped without a reply; the drop
        is the only sign an authorised sender's mail may also be going
        unanswered."""
        store = self._spent_replies(tmp_path, self.BUDGET)

        alert = next(a for a in self._alerts(store, tmp_path, self.DOWN_AT)
                     if "refusal-reply" in a)

        assert str(self.BUDGET) in alert
        assert "dropped" in alert

    def test_a_budget_with_a_reply_left_is_not_mentioned(self, tmp_path):
        store = self._spent_replies(tmp_path, self.BUDGET - 1)

        assert self._alerts(store, tmp_path, self.DOWN_AT) == []
