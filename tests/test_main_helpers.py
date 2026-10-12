import csv
import logging
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import config
import pandas as pd
import pytest
from helpers import Q102, patch_main_paths, trades_rows

from bot import backtest
from bot import main as bot_main
from bot.data import LatestQuote
from bot.halt import clear_halt, read_halt, trip_halt
from bot.main import (
    Engine,
    bar_boundary,
    crypto_maker_limit_price,
    decision_bar_key,
    equity_entry_limit_price,
    in_near_close_window,
    in_open_auction_window,
    jitter_offset_minutes,
    marketable_limit_price,
    passive_limit_price,
    quote_anchored_limit_price,
)
from bot.portfolio import TradeLog
from bot.state import StateStore
from bot.strategies.base import Action, PositionState, Signal
from config import (
    INSTRUMENTS,
    MARKET_DATA_FEED,
    PROTECTIONS,
    AssetClass,
    ExecutionConfig,
    InstrumentConfig,
    ProtectionsConfig,
    StrategyName,
)


class TestBarBoundary:
    def test_15_min_floor(self):
        now = datetime(2026, 7, 21, 14, 47, 12, tzinfo=timezone.utc)
        assert bar_boundary(now, 15) == datetime(2026, 7, 21, 14, 45, tzinfo=timezone.utc)

    def test_1_hour_floor(self):
        now = datetime(2026, 7, 21, 14, 47, tzinfo=timezone.utc)
        assert bar_boundary(now, 60) == datetime(2026, 7, 21, 14, 0, tzinfo=timezone.utc)

    def test_4_hour_floor(self):
        now = datetime(2026, 7, 21, 14, 47, tzinfo=timezone.utc)
        assert bar_boundary(now, 240) == datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)


class TestMarketableLimitPrice:
    def test_buy_reaches_above_close(self):
        assert marketable_limit_price(100.00, "buy", 10.0) == 100.10

    def test_sell_reaches_below_close(self):
        assert marketable_limit_price(100.00, "sell", 10.0) == 99.90

    def test_rounds_to_penny(self):
        # 543.21 * (1 + 10/10000) = 543.753... -> 543.75
        assert marketable_limit_price(543.21, "buy", 10.0) == 543.75


class TestNearCloseWindow:
    CLOSE = datetime(2026, 7, 15, 20, 0, tzinfo=timezone.utc)  # 16:00 EDT

    def test_inside_window(self):
        now = datetime(2026, 7, 15, 19, 55, tzinfo=timezone.utc)  # 5 min out
        assert in_near_close_window(now, self.CLOSE, 10)

    def test_outside_window(self):
        now = datetime(2026, 7, 15, 19, 30, tzinfo=timezone.utc)  # 30 min out
        assert not in_near_close_window(now, self.CLOSE, 10)

    def test_start_boundary_inclusive(self):
        now = datetime(2026, 7, 15, 19, 50, tzinfo=timezone.utc)  # exactly 10 min
        assert in_near_close_window(now, self.CLOSE, 10)

    def test_at_close_excluded(self):
        assert not in_near_close_window(self.CLOSE, self.CLOSE, 10)

    def test_after_close_excluded(self):
        now = datetime(2026, 7, 15, 20, 1, tzinfo=timezone.utc)
        assert not in_near_close_window(now, self.CLOSE, 10)

    def test_dst_shift_is_pure_utc_arithmetic(self):
        # Same "9 min before close" holds regardless of EDT/EST: the winter close
        # is 21:00 UTC (16:00 EST), the summer close 20:00 UTC (16:00 EDT).
        winter_close = datetime(2026, 1, 15, 21, 0, tzinfo=timezone.utc)
        winter_now = datetime(2026, 1, 15, 20, 51, tzinfo=timezone.utc)
        summer_now = datetime(2026, 7, 15, 19, 51, tzinfo=timezone.utc)
        assert in_near_close_window(winter_now, winter_close, 10)
        assert in_near_close_window(summer_now, self.CLOSE, 10)


class TestOpenAuctionWindow:
    OPEN = datetime(2026, 7, 15, 13, 30, tzinfo=timezone.utc)  # 09:30 EDT

    def test_at_open_inside(self):
        assert in_open_auction_window(self.OPEN, self.OPEN, 5)

    def test_within_five_minutes(self):
        now = datetime(2026, 7, 15, 13, 33, tzinfo=timezone.utc)
        assert in_open_auction_window(now, self.OPEN, 5)

    def test_end_boundary_excluded(self):
        now = datetime(2026, 7, 15, 13, 35, tzinfo=timezone.utc)  # exactly 5 min
        assert not in_open_auction_window(now, self.OPEN, 5)

    def test_after_window(self):
        now = datetime(2026, 7, 15, 14, 0, tzinfo=timezone.utc)
        assert not in_open_auction_window(now, self.OPEN, 5)

    def test_dst_shift_is_pure_utc_arithmetic(self):
        winter_open = datetime(2026, 1, 15, 14, 30, tzinfo=timezone.utc)  # 09:30 EST
        assert in_open_auction_window(winter_open, winter_open, 5)
        one_min_in = datetime(2026, 1, 15, 14, 31, tzinfo=timezone.utc)
        assert in_open_auction_window(one_min_in, winter_open, 5)


def _make_engine(tmp_path, monkeypatch, session):
    """Engine with a fully mocked broker, an open market, and process_instrument
    stubbed so we can observe which instruments the cycle evaluates and marks."""
    patch_main_paths(monkeypatch, tmp_path)
    broker = MagicMock()
    broker.account_equity.return_value = 100_000.0
    broker.get_positions.return_value = {}
    broker.market_is_open.return_value = True
    broker.current_session.return_value = session
    engine = Engine(broker, MagicMock(), dry_run=False)
    engine.maybe_send_briefings = lambda: None
    engine.process_instrument = MagicMock()
    return engine


def _evaluated(engine, symbol):
    return any(c.args == (symbol,) for c in engine.process_instrument.call_args_list)


def _intraday_equity_cfg(symbol="XGLD"):
    """A 240-minute equity instrument. The lineup has carried none since GLD
    moved to daily, but the intraday branches in decision_bars/keeps_forming_bar
    are still live code, so they are exercised against a synthetic config rather
    than against whichever instrument happens to be intraday this month."""
    return InstrumentConfig(
        symbol=symbol, asset_class=AssetClass.US_EQUITY,
        strategy=StrategyName.TREND_FOLLOWING, timeframe_minutes=240,
        params={"fast_period": 20, "slow_period": 100, "trail_atr_mult": 3.0,
                "allow_short": False},
        slippage_pct=0.0001)


class TestSubmissionWindowDedup:
    def test_daily_equity_outside_window_not_evaluated_or_marked(self, tmp_path, monkeypatch):
        now = datetime.now(timezone.utc)
        session = (now - timedelta(hours=6), now + timedelta(hours=1))  # close far off
        engine = _make_engine(tmp_path, monkeypatch, session)
        engine.run_cycle()
        assert not _evaluated(engine, "SPY")
        assert "SPY" not in engine.last_boundary  # bar preserved for a later cycle

    def test_daily_equity_inside_window_evaluated_and_marked(self, tmp_path, monkeypatch):
        now = datetime.now(timezone.utc)
        session = (now - timedelta(hours=6), now + timedelta(minutes=5))  # near close
        engine = _make_engine(tmp_path, monkeypatch, session)
        engine.run_cycle()
        assert _evaluated(engine, "SPY")
        assert "SPY" in engine.last_boundary

    def test_intraday_equity_in_open_auction_deferred_not_marked(self, tmp_path, monkeypatch):
        now = datetime.now(timezone.utc)
        session = (now, now + timedelta(hours=6))  # just opened
        monkeypatch.setattr(config, "INSTRUMENTS", [_intraday_equity_cfg()])
        engine = _make_engine(tmp_path, monkeypatch, session)
        engine.run_cycle()
        assert not _evaluated(engine, "XGLD")
        assert "XGLD" not in engine.last_boundary

    def test_intraday_equity_after_auction_evaluated(self, tmp_path, monkeypatch):
        now = datetime.now(timezone.utc)
        session = (now - timedelta(minutes=30), now + timedelta(hours=6))
        monkeypatch.setattr(config, "INSTRUMENTS", [_intraday_equity_cfg()])
        engine = _make_engine(tmp_path, monkeypatch, session)
        engine.run_cycle()
        assert _evaluated(engine, "XGLD")
        assert "XGLD" in engine.last_boundary

    def test_calendar_failure_degrades_to_none_and_keeps_crypto(self, tmp_path, monkeypatch):
        now = datetime.now(timezone.utc)
        engine = _make_engine(tmp_path, monkeypatch, session=None)
        engine.broker.current_session.side_effect = RuntimeError("calendar down")
        engine.run_cycle()  # must not raise
        assert _evaluated(engine, "BTC/USD")  # crypto supervision unaffected

    def test_session_fetched_once_per_trading_day(self, tmp_path, monkeypatch):
        now = datetime.now(timezone.utc)
        session = (now - timedelta(hours=6), now + timedelta(hours=1))
        engine = _make_engine(tmp_path, monkeypatch, session)
        engine.run_cycle()
        engine.last_boundary.clear()  # force re-evaluation next cycle
        engine.run_cycle()
        assert engine.broker.current_session.call_count == 1

    def test_dry_run_force_evaluates_regardless_of_window(self, tmp_path, monkeypatch):
        now = datetime.now(timezone.utc)
        session = (now - timedelta(hours=6), now + timedelta(hours=1))  # outside window
        engine = _make_engine(tmp_path, monkeypatch, session)
        engine.run_cycle(force=True)
        assert _evaluated(engine, "SPY")


class TestCryptoMakerLimitPrice:
    def test_rests_below_close(self):
        assert crypto_maker_limit_price(50_000.0, 5.0) == 49_975.0

    def test_floors_to_whole_dollar_by_default(self):
        # BTC/USD's price increment is 1: sub-dollar limits are rejected. Floor
        # keeps the resting buy at or below the offset target.
        # 50100 * (1 - 5/10000) = 50074.95 -> 50074
        assert crypto_maker_limit_price(50_100.0, 5.0) == 50_074.0

    def test_respects_finer_increment(self):
        # 50100 * 0.9995 = 50074.95 -> floored to 0.5 -> 50074.5
        assert crypto_maker_limit_price(50_100.0, 5.0, price_increment=0.5) == 50_074.5


def _crypto_engine(tmp_path, monkeypatch, dry_run=False):
    """Engine with a mocked broker and a real StateStore on a tmp file, so the
    resting-order state transitions are exercised end to end."""
    patch_main_paths(monkeypatch, tmp_path)
    broker = MagicMock()
    broker.account_equity.return_value = 100_000.0
    broker.get_positions.return_value = {}
    broker.has_open_order.return_value = False
    broker.has_open_stop.return_value = False
    broker.limit_order.return_value = MagicMock(id="order-1")
    broker.close_position.return_value = MagicMock(id="close-1")
    data = MagicMock()
    # No default quote: the decision-price guardrail compares the bar close
    # against the quote mid, so a symbol-agnostic stub would distrust every bar.
    # Tests that care about the quote set it explicitly.
    data.latest_quote.return_value = None
    return Engine(broker, data, dry_run=dry_run)


D0 = datetime(2026, 7, 22, tzinfo=timezone.utc)
D1 = datetime(2026, 7, 23, tzinfo=timezone.utc)


class TestCryptoRestingEntry:
    def test_entry_places_resting_gtc_limit_no_stop(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["BTC/USD"]
        engine.enter_position(cfg, "long", 50_000.0, 1_000.0, {}, D0)
        engine.broker.limit_order.assert_called_once()
        args, kwargs = engine.broker.limit_order.call_args
        assert args[0] == "BTC/USD"
        assert args[3] == AssetClass.CRYPTO       # broker translates crypto -> GTC
        assert args[4] == pytest.approx(49_975.0)  # close x (1 - 5bps)
        assert "stop_loss" not in kwargs           # crypto has no broker stop leg
        assert engine.state.resting_order("BTC/USD") == D0.isoformat()

    def test_intended_stop_persisted_before_submission(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["BTC/USD"]

        def assert_stop_already_persisted(*a, **k):
            # Both the stop and the resting marker must exist before submission so
            # a crash before the broker ack still leaves a recoverable stop.
            assert engine.state.position_meta("BTC/USD") is not None
            assert engine.state.resting_order("BTC/USD") == D0.isoformat()
            return MagicMock(id="order-1")

        engine.broker.limit_order.side_effect = assert_stop_already_persisted
        engine.enter_position(cfg, "long", 50_000.0, 1_000.0, {}, D0)
        # trail_atr_mult 3.0 * atr 1000 -> stop 3000 below close
        assert engine.state.position_meta("BTC/USD")["stop_price"] == pytest.approx(47_000.0)

    def test_signal_fires_places_resting_order(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["BTC/USD"]
        strat = MagicMock()
        strat.min_bars = 2
        strat.stop_atr_mult = 3.0
        strat.prepare.return_value = pd.DataFrame(
            {"close": [49_000.0, 49_500.0, 50_000.0], "atr": [1_000.0, 1_000.0, 1_000.0]})
        strat.signal.return_value = Signal(Action.LONG, "golden cross")
        engine.strategies["BTC/USD"] = (cfg, strat)
        idx = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
        engine.data.get_recent_bars.return_value = pd.DataFrame(
            {"close": [1, 2, 3, 4, 5]}, index=idx)

        engine.process_instrument("BTC/USD")

        engine.broker.limit_order.assert_called_once()
        assert engine.broker.limit_order.call_args.args[4] == pytest.approx(49_975.0)
        assert engine.state.resting_order("BTC/USD") is not None

    def test_unfilled_at_next_boundary_cancelled_then_reevaluates(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_position_meta("BTC/USD", 47_000.0, 50_000.0)
        engine.state.set_resting_order("BTC/USD", D0.isoformat())
        engine.broker.has_open_order.return_value = True

        blocked, cancelled = engine.reconcile_crypto_resting("BTC/USD", D1, None)

        assert blocked is False  # entry may be re-evaluated
        assert cancelled is True  # this cycle cancelled a stale resting order
        engine.broker.cancel_open_orders.assert_called_once_with("BTC/USD")
        assert engine.state.resting_order("BTC/USD") is None
        assert engine.state.position_meta("BTC/USD") is None

    def test_filled_cancels_remainder_and_keeps_stop(self, tmp_path, monkeypatch):
        # Alpaca crypto limits can partially fill; the unfilled remainder must be
        # cancelled so it cannot rest GTC and grow the position past sized qty.
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_position_meta("BTC/USD", 47_000.0, 50_000.0)
        engine.state.set_resting_order("BTC/USD", D0.isoformat())
        engine.broker.has_open_order.return_value = True  # remainder still open
        pos = PositionState("BTC/USD", "long", 0.3, 50_000.0, 47_000.0, 50_000.0)

        blocked, cancelled = engine.reconcile_crypto_resting("BTC/USD", D1, pos)

        assert blocked is False
        assert cancelled is False  # the remainder of a fill, not a stale order
        engine.broker.cancel_open_orders.assert_called_once_with("BTC/USD")
        assert engine.state.resting_order("BTC/USD") is None
        assert engine.state.position_meta("BTC/USD") == {
            "stop_price": 47_000.0, "high_water_mark": 50_000.0}

    def test_no_double_entry_while_order_rests_same_boundary(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_resting_order("BTC/USD", D0.isoformat())
        engine.broker.has_open_order.return_value = True

        blocked, cancelled = engine.reconcile_crypto_resting("BTC/USD", D0, None)

        assert blocked is True
        assert cancelled is False
        engine.broker.cancel_open_orders.assert_not_called()

    def test_order_gone_no_position_never_wipes_stop(self, tmp_path, monkeypatch):
        # Race: the order filled between the cycle's position snapshot and this
        # check, so pos is stale-None but no order is open. Must not wipe the
        # intended stop — positions() reconciles it (keep on fill, drop on none).
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_position_meta("BTC/USD", 47_000.0, 50_000.0)
        engine.state.set_resting_order("BTC/USD", D0.isoformat())
        engine.broker.has_open_order.return_value = False

        blocked, cancelled = engine.reconcile_crypto_resting("BTC/USD", D0, None)

        assert blocked is False
        assert cancelled is False  # nothing was cancelled here
        engine.broker.cancel_open_orders.assert_not_called()
        assert engine.state.resting_order("BTC/USD") is None
        assert engine.state.position_meta("BTC/USD") is not None  # stop preserved

    def test_positions_preserves_stop_while_order_rests(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.get_positions.return_value = {}
        engine.state.set_position_meta("BTC/USD", 47_000.0, 50_000.0)
        engine.state.set_resting_order("BTC/USD", D0.isoformat())

        engine.positions()

        assert engine.state.position_meta("BTC/USD") is not None

    def test_positions_clears_stop_without_resting_order(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.get_positions.return_value = {}
        engine.state.set_position_meta("BTC/USD", 47_000.0, 50_000.0)

        engine.positions()

        assert engine.state.position_meta("BTC/USD") is None

    def test_dry_run_places_nothing(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch, dry_run=True)
        cfg, _ = engine.strategies["BTC/USD"]
        engine.enter_position(cfg, "long", 50_000.0, 1_000.0, {}, D0)
        engine.broker.limit_order.assert_not_called()
        assert engine.state.resting_order("BTC/USD") is None

    def test_dry_run_reconcile_unfilled_preserves_state_and_no_cancel(self, tmp_path, monkeypatch):
        # A dry-run must never mutate the shared state.json a real run persisted.
        engine = _crypto_engine(tmp_path, monkeypatch, dry_run=True)
        engine.state.set_position_meta("BTC/USD", 47_000.0, 50_000.0)
        engine.state.set_resting_order("BTC/USD", D0.isoformat())
        engine.broker.has_open_order.return_value = True

        assert engine.reconcile_crypto_resting("BTC/USD", D1, None) == (False, False)

        engine.broker.cancel_open_orders.assert_not_called()
        assert engine.state.resting_order("BTC/USD") == D0.isoformat()
        assert engine.state.position_meta("BTC/USD") is not None

    def test_dry_run_reconcile_filled_preserves_state(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch, dry_run=True)
        engine.state.set_position_meta("BTC/USD", 47_000.0, 50_000.0)
        engine.state.set_resting_order("BTC/USD", D0.isoformat())
        pos = PositionState("BTC/USD", "long", 0.3, 50_000.0, 47_000.0, 50_000.0)

        engine.reconcile_crypto_resting("BTC/USD", D1, pos)

        engine.broker.cancel_open_orders.assert_not_called()
        assert engine.state.resting_order("BTC/USD") == D0.isoformat()
        assert engine.state.position_meta("BTC/USD") is not None


class TestStateStore:
    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "state.json")
        store = StateStore(path)
        store.set_position_meta("BTC/USD", stop_price=48_000.0, high_water_mark=52_000.0)
        store.data["peak_equity"] = 105_000.0
        store.save()

        reloaded = StateStore(path)
        meta = reloaded.position_meta("BTC/USD")
        assert meta == {"stop_price": 48_000.0, "high_water_mark": 52_000.0}
        assert reloaded.data["peak_equity"] == 105_000.0

    def test_clear(self, tmp_path):
        path = str(tmp_path / "state.json")
        store = StateStore(path)
        store.set_position_meta("SPY", 98.0, 100.0)
        store.clear_position_meta("SPY")
        assert StateStore(path).position_meta("SPY") is None

    def test_pending_fill_round_trip(self, tmp_path):
        path = str(tmp_path / "state.json")
        store = StateStore(path)
        payload = {"order_id": "oid-1", "side": "buy", "event": "entry",
                   "decision_mid": 500.0, "decision_spread_bps": 4.0,
                   "limit_price": 500.5, "submitted_at": "2026-07-22T00:00:00+00:00"}
        store.set_pending_fill("SPY", payload)
        assert StateStore(path).pending_fill("SPY") == payload

    def test_pending_fill_absent_is_none(self, tmp_path):
        assert StateStore(str(tmp_path / "state.json")).pending_fill("SPY") is None

    def test_pending_fill_clear(self, tmp_path):
        path = str(tmp_path / "state.json")
        store = StateStore(path)
        store.set_pending_fill("SPY", {"order_id": "oid-1"})
        store.clear_pending_fill("SPY")
        assert StateStore(path).pending_fill("SPY") is None
        store.clear_pending_fill("SPY")  # idempotent


def _fills_rows(tmp_path):
    return list(csv.reader((tmp_path / "fills.csv").open()))


def _pending(event="entry", side="buy", mid=500.0, order_id="oid-1", when=None):
    return {"order_id": order_id, "side": side, "event": event,
            "decision_mid": mid, "decision_spread_bps": 4.0, "limit_price": 500.5,
            "submitted_at": (when or datetime.now(timezone.utc)).isoformat()}


class TestPendingFillEntryExit:
    def test_equity_entry_persists_marker(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        engine.risk.approve_entry = MagicMock(
            return_value=MagicMock(qty=100, stop_price=490.0))
        quote = LatestQuote(499.9, 500.1, 500.0, 4.0)
        engine.enter_position(cfg, "long", 500.0, 5.0, {}, D0, quote=quote)
        engine.broker.limit_order.assert_called_once()
        marker = engine.state.pending_fill("SPY")
        assert marker["event"] == "entry"
        assert marker["side"] == "buy"
        assert marker["order_id"] == "order-1"
        assert marker["decision_mid"] == pytest.approx(500.0)

    def test_equity_entry_dry_run_persists_no_marker(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch, dry_run=True)
        cfg, _ = engine.strategies["SPY"]
        engine.risk.approve_entry = MagicMock(
            return_value=MagicMock(qty=100, stop_price=490.0))
        engine.enter_position(cfg, "long", 500.0, 5.0, {}, D0,
                              quote=LatestQuote(499.9, 500.1, 500.0, 4.0))
        engine.broker.limit_order.assert_not_called()
        assert engine.state.pending_fill("SPY") is None

    def test_exit_persists_marker(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        pos = PositionState("SPY", "long", 100, 500.0, 490.0, 500.0)
        engine.exit_position(cfg, pos, 505.0, "exit signal")
        marker = engine.state.pending_fill("SPY")
        assert marker["event"] == "exit"
        assert marker["side"] == "sell"
        assert marker["order_id"] == "close-1"

    def test_entry_with_no_quote_records_nan_mid(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        engine.risk.approve_entry = MagicMock(
            return_value=MagicMock(qty=100, stop_price=490.0))
        engine.enter_position(cfg, "long", 500.0, 5.0, {}, D0, quote=None)
        marker = engine.state.pending_fill("SPY")
        assert marker["decision_mid"] != marker["decision_mid"]  # NaN


class TestReconcilePendingFills:
    def test_fill_writes_one_row_then_clears(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_pending_fill("SPY", _pending())
        engine.broker.get_order_fill.return_value = (100.0, 500.4, "filled")
        engine.reconcile_pending_fills()
        engine.reconcile_pending_fills()  # marker gone: must not double-log
        rows = _fills_rows(tmp_path)
        assert len(rows) == 2  # header + one fill
        assert rows[1][3] == "entry"
        assert float(rows[1][9]) == pytest.approx(8.0)  # buy 500.4 vs mid 500.0
        assert engine.state.pending_fill("SPY") is None

    def test_unfilled_keeps_marker_no_row(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_pending_fill("SPY", _pending())
        engine.broker.get_order_fill.return_value = (0.0, None, "new")
        engine.reconcile_pending_fills()
        assert engine.state.pending_fill("SPY") is not None
        assert len(_fills_rows(tmp_path)) == 1  # header only

    def test_stale_marker_dropped_without_row(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        old = datetime.now(timezone.utc) - timedelta(days=4)
        engine.state.set_pending_fill("SPY", _pending(when=old))
        engine.reconcile_pending_fills()
        assert engine.state.pending_fill("SPY") is None
        engine.broker.get_order_fill.assert_not_called()
        assert len(_fills_rows(tmp_path)) == 1  # header only

    def test_dry_run_does_not_query_or_log(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch, dry_run=True)
        engine.state.set_pending_fill("SPY", _pending())
        engine.reconcile_pending_fills()
        engine.broker.get_order_fill.assert_not_called()
        assert engine.state.pending_fill("SPY") is not None

    def test_crypto_maker_marker_skipped_by_generic_reconcile(self, tmp_path, monkeypatch):
        # crypto maker fills are reconciled in reconcile_crypto_resting, not here.
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_pending_fill("BTC/USD", _pending(event="crypto_maker_placed"))
        engine.reconcile_pending_fills()
        engine.broker.get_order_fill.assert_not_called()
        assert engine.state.pending_fill("BTC/USD") is not None


def _long_signal_engine(tmp_path, monkeypatch, symbol, quote):
    engine = _crypto_engine(tmp_path, monkeypatch)
    cfg, _ = engine.strategies[symbol]
    strat = MagicMock()
    strat.min_bars = 2
    strat.stop_atr_mult = 3.0
    strat.prepare.return_value = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0], "atr": [1.0, 1.0, 1.0]})
    strat.signal.return_value = Signal(Action.LONG, "entry")
    engine.strategies[symbol] = (cfg, strat)
    idx = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
    engine.data.get_recent_bars.return_value = pd.DataFrame(
        {"close": [1, 2, 3, 4, 5]}, index=idx)
    engine.data.latest_quote.return_value = quote
    return engine


class TestSpreadGate:
    def test_disabled_allows(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)  # max_spread_bps None
        assert engine._within_spread_limit("SPY", LatestQuote(1, 1, 1, 99.0)) is True

    def test_under_threshold_allows(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bot.main.EXECUTION", ExecutionConfig(max_spread_bps=10.0))
        engine = _crypto_engine(tmp_path, monkeypatch)
        assert engine._within_spread_limit("SPY", LatestQuote(1, 1, 1, 4.0)) is True

    def test_over_threshold_skips(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bot.main.EXECUTION", ExecutionConfig(max_spread_bps=10.0))
        engine = _crypto_engine(tmp_path, monkeypatch)
        assert engine._within_spread_limit("SPY", LatestQuote(1, 1, 1, 25.0)) is False

    def test_missing_quote_fails_open(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bot.main.EXECUTION", ExecutionConfig(max_spread_bps=10.0))
        engine = _crypto_engine(tmp_path, monkeypatch)
        assert engine._within_spread_limit("SPY", None) is True

    def test_nan_spread_fails_open(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bot.main.EXECUTION", ExecutionConfig(max_spread_bps=10.0))
        engine = _crypto_engine(tmp_path, monkeypatch)
        assert engine._within_spread_limit("SPY", LatestQuote(0, 0, 0, float("nan"))) is True

    def test_process_instrument_skips_wide_spread_without_marking(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bot.main.EXECUTION", ExecutionConfig(max_spread_bps=5.0))
        engine = _long_signal_engine(tmp_path, monkeypatch, "SPY",
                                     LatestQuote(101.0, 103.0, 102.0, 200.0))
        # False == bar not evaluated, so run_cycle leaves it unmarked and a later
        # cycle retries once the spread narrows.
        assert engine.process_instrument("SPY") is False
        engine.broker.limit_order.assert_not_called()

    def test_process_instrument_enters_under_threshold(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bot.main.EXECUTION", ExecutionConfig(max_spread_bps=50.0))
        engine = _long_signal_engine(tmp_path, monkeypatch, "SPY", Q102)
        engine.risk.approve_entry = MagicMock(
            return_value=MagicMock(qty=100, stop_price=98.0))
        engine.process_instrument("SPY")
        engine.broker.limit_order.assert_called_once()


class TestCryptoMakerEvents:
    def test_placed_logged_on_entry(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["BTC/USD"]
        engine.enter_position(cfg, "long", 50_000.0, 1_000.0, {}, D0,
                              quote=LatestQuote(49_990.0, 50_010.0, 50_000.0, 4.0))
        events = [r[3] for r in _fills_rows(tmp_path)[1:]]
        assert events == ["crypto_maker_placed"]

    def test_filled_logged_with_negative_slippage_and_clears_marker(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_position_meta("BTC/USD", 47_000.0, 50_000.0)
        engine.state.set_resting_order("BTC/USD", D0.isoformat())
        engine.state.set_pending_fill("BTC/USD", _pending(
            event="crypto_maker_placed", mid=50_000.0, order_id="o"))
        engine.broker.has_open_order.return_value = True
        pos = PositionState("BTC/USD", "long", 0.3, 49_975.0, 47_000.0, 50_000.0)

        engine.reconcile_crypto_resting("BTC/USD", D1, pos)

        filled = [r for r in _fills_rows(tmp_path)[1:] if r[3] == "crypto_maker_filled"]
        assert len(filled) == 1
        assert float(filled[0][9]) < 0  # bought below mid: captured the spread
        assert engine.state.pending_fill("BTC/USD") is None

    def test_cancelled_logged_and_clears_marker(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_position_meta("BTC/USD", 47_000.0, 50_000.0)
        engine.state.set_resting_order("BTC/USD", D0.isoformat())
        engine.state.set_pending_fill("BTC/USD", _pending(
            event="crypto_maker_placed", order_id="o"))
        engine.broker.has_open_order.return_value = True

        engine.reconcile_crypto_resting("BTC/USD", D1, None)  # boundary advanced, unfilled

        cancelled = [r for r in _fills_rows(tmp_path)[1:] if r[3] == "crypto_maker_cancelled"]
        assert len(cancelled) == 1
        assert engine.state.pending_fill("BTC/USD") is None

    def test_dry_run_logs_nothing(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch, dry_run=True)
        engine.state.set_resting_order("BTC/USD", D0.isoformat())
        engine.state.set_pending_fill("BTC/USD", _pending(event="crypto_maker_placed"))
        engine.broker.has_open_order.return_value = True
        pos = PositionState("BTC/USD", "long", 0.3, 49_975.0, 47_000.0, 50_000.0)

        engine.reconcile_crypto_resting("BTC/USD", D1, pos)

        assert len(_fills_rows(tmp_path)) == 1  # header only


# ---------------------------------------------------------------------------
# Task 1c — quote-anchored equity limit + opt-in rest-then-cross ladder
# ---------------------------------------------------------------------------

Q = LatestQuote(bid=99.90, ask=100.10, mid=100.0, spread_bps=20.0)


class TestQuoteAnchoredLimitPrice:
    def test_buy_crosses_up_from_ask(self):
        # 100.10 * (1 + 10/10000) = 100.2001 -> 100.20
        assert quote_anchored_limit_price(Q, "buy", 10.0) == 100.20

    def test_sell_crosses_down_from_bid(self):
        # 99.90 * (1 - 10/10000) = 99.8001 -> 99.80
        assert quote_anchored_limit_price(Q, "sell", 10.0) == 99.80


class TestEquityEntryLimitPrice:
    def test_prefers_live_ask_over_stale_close(self):
        # Anchors to the ask (100.10), not the stale signal close (105.0).
        assert equity_entry_limit_price(105.0, "buy", 10.0, Q) == 100.20

    def test_prefers_live_bid_for_sell(self):
        assert equity_entry_limit_price(105.0, "sell", 10.0, Q) == 99.80

    def test_falls_back_to_close_when_no_quote(self):
        assert equity_entry_limit_price(100.0, "buy", 10.0, None) == 100.10

    def test_falls_back_when_anchor_nan(self):
        nan_q = LatestQuote(float("nan"), float("nan"), float("nan"), float("nan"))
        assert equity_entry_limit_price(100.0, "buy", 10.0, nan_q) == 100.10


class TestPassiveLimitPrice:
    def test_buy_rests_below_mid(self):
        # 100.0 * (1 - 5/10000) = 99.95
        assert passive_limit_price(Q, "buy", 5.0) == 99.95

    def test_sell_rests_above_mid(self):
        assert passive_limit_price(Q, "sell", 5.0) == 100.05

    def test_zero_offset_is_mid(self):
        assert passive_limit_price(Q, "buy", 0.0) == 100.0


def _ladder_marker(order_id="passive-1", total_qty=100, stop_price=490.0,
                   side="buy", signal_close=500.0, boundary=None):
    return {"order_id": order_id, "boundary": (boundary or D0).isoformat(),
            "stop_price": stop_price, "side": side, "total_qty": total_qty,
            "signal_close": signal_close,
            "submitted_at": datetime.now(timezone.utc).isoformat()}


def _in_window_session(now):
    return (now - timedelta(hours=6), now + timedelta(minutes=5))


SPY_QUOTE = LatestQuote(bid=499.90, ask=500.10, mid=500.0, spread_bps=4.0)


class TestEquityLadderPlacement:
    def test_placement_posts_passive_at_mid_with_oto_stop(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bot.main.EXECUTION",
                            ExecutionConfig(ladder_enabled=True, ladder_passive_bps=2.0))
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        engine.risk.approve_entry = MagicMock(
            return_value=MagicMock(qty=100, stop_price=490.0))
        engine.enter_position(cfg, "long", 500.0, 5.0, {}, D0, quote=SPY_QUOTE)

        args, kwargs = engine.broker.limit_order.call_args
        assert args[4] == pytest.approx(499.90)   # mid 500 * (1 - 2bps)
        assert kwargs["stop_loss"] == 490.0       # OTO stop rides the passive leg
        marker = engine.state.ladder_order("SPY")
        assert marker["order_id"] == "order-1"
        assert marker["total_qty"] == 100
        assert marker["stop_price"] == 490.0
        assert marker["signal_close"] == 500.0

    def test_no_quote_falls_back_to_marketable_cross_no_ladder(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bot.main.EXECUTION",
                            ExecutionConfig(ladder_enabled=True, ladder_passive_bps=2.0))
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        engine.risk.approve_entry = MagicMock(
            return_value=MagicMock(qty=100, stop_price=490.0))
        engine.enter_position(cfg, "long", 500.0, 5.0, {}, D0, quote=None)

        args, kwargs = engine.broker.limit_order.call_args
        assert args[4] == pytest.approx(500.5)    # close-anchored marketable
        assert kwargs["stop_loss"] == 490.0
        assert engine.state.ladder_order("SPY") is None

    def test_ladder_disabled_uses_quote_anchored_cross(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)  # default: ladder off
        cfg, _ = engine.strategies["SPY"]
        engine.risk.approve_entry = MagicMock(
            return_value=MagicMock(qty=100, stop_price=490.0))
        engine.enter_position(cfg, "long", 505.0, 5.0, {}, D0, quote=SPY_QUOTE)

        args, kwargs = engine.broker.limit_order.call_args
        assert args[4] == pytest.approx(500.60)   # ask 500.10 * (1 + 10bps)
        assert kwargs["stop_loss"] == 490.0
        assert engine.state.ladder_order("SPY") is None


class TestEquityLadderReconcile:
    def _engine(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bot.main.EXECUTION",
                            ExecutionConfig(ladder_enabled=True, ladder_passive_bps=2.0))
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.data.latest_quote.return_value = SPY_QUOTE
        return engine

    def test_passive_fully_filled_retires_marker_and_keeps_stop(self, tmp_path, monkeypatch):
        # INVARIANT: passive filled -> the OTO stop attached at placement survives.
        engine = self._engine(tmp_path, monkeypatch)
        engine.state.set_ladder_order("SPY", _ladder_marker())
        engine.state.set_pending_fill("SPY", _pending(mid=500.0))
        engine.broker.get_order_fill.return_value = (100.0, 499.90, "filled")
        now = datetime.now(timezone.utc)
        pos = PositionState("SPY", "long", 100, 499.90, 490.0, 499.90)

        engine.reconcile_equity_ladders({"SPY": pos}, _in_window_session(now), now)

        engine.broker.cancel_open_orders.assert_not_called()  # OTO stop preserved
        engine.broker.limit_order.assert_not_called()         # no cross
        assert engine.state.ladder_order("SPY") is None

    def test_escalation_crosses_with_stop_reattached(self, tmp_path, monkeypatch):
        # INVARIANT: escalation path -> the cross carries the OTO stop.
        engine = self._engine(tmp_path, monkeypatch)
        engine.state.set_ladder_order("SPY", _ladder_marker())
        engine.state.set_pending_fill("SPY", _pending(mid=500.0))
        engine.broker.get_order_fill.return_value = (0.0, None, "new")
        now = datetime.now(timezone.utc)

        engine.reconcile_equity_ladders({}, _in_window_session(now), now)

        engine.broker.cancel_open_orders.assert_called_once_with("SPY")
        args, kwargs = engine.broker.limit_order.call_args
        assert args[2] == 100                      # remaining qty == total
        assert args[4] == pytest.approx(500.60)    # quote-anchored cross (ask + 10bps)
        assert kwargs["stop_loss"] == 490.0
        assert engine.state.ladder_order("SPY") is None

    def test_partial_fill_cross_covers_remaining_stop_covers_total(self, tmp_path, monkeypatch):
        # INVARIANT: partial passive + cross -> stop qty sums to the total position.
        engine = self._engine(tmp_path, monkeypatch)
        engine.state.set_ladder_order("SPY", _ladder_marker(total_qty=100, stop_price=490.0))
        engine.state.set_pending_fill("SPY", _pending(mid=500.0))
        engine.broker.get_order_fill.return_value = (40.0, 499.90, "partially_filled")
        now = datetime.now(timezone.utc)
        pos = PositionState("SPY", "long", 40, 499.90, 490.0, 499.90)

        engine.reconcile_equity_ladders({"SPY": pos}, _in_window_session(now), now)

        engine.broker.cancel_open_orders.assert_called_once_with("SPY")
        cross_args, cross_kwargs = engine.broker.limit_order.call_args
        assert cross_args[2] == 60                 # remaining = total - filled
        assert cross_kwargs["stop_loss"] == 490.0  # OTO covers the 60-share cross

        # The 40 already-filled shares are covered by the safety-net sweep.
        engine.broker.has_open_stop.return_value = False
        engine.reconcile_equity_stops({"SPY": pos})
        engine.broker.stop_order.assert_called_once_with(
            "SPY", "sell", 40, 490.0, AssetClass.US_EQUITY)
        # 40 (stop_order) + 60 (cross OTO) == 100 total qty.

    def test_window_closed_cancels_stale_passive_and_clears_marker(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        engine.state.set_ladder_order("SPY", _ladder_marker())
        engine.broker.get_order_fill.return_value = (0.0, None, "new")
        now = datetime.now(timezone.utc)

        engine.reconcile_equity_ladders({}, None, now)  # session None -> window closed

        engine.broker.cancel_open_orders.assert_called_once_with("SPY")
        engine.broker.limit_order.assert_not_called()   # no escalation outside window
        assert engine.state.ladder_order("SPY") is None

    def test_crash_between_cancel_and_resubmit_reconcile_replaces_stop(self, tmp_path, monkeypatch):
        # INVARIANT: marker present, no open order, position exists, no broker stop
        # -> reconcile_equity_stops re-places the intended stop for the full qty.
        engine = self._engine(tmp_path, monkeypatch)
        engine.state.set_position_meta("SPY", 490.0, 500.0)  # intended stop preserved
        engine.state.set_ladder_order("SPY", _ladder_marker(total_qty=40))
        engine.broker.get_positions.return_value = {
            "SPY": PositionState("SPY", "long", 40, 499.90, 0.0, 499.90)}

        positions = engine.positions()  # overlays the preserved meta stop (490.0)
        engine.broker.has_open_stop.return_value = False
        engine.reconcile_equity_stops(positions)

        engine.broker.stop_order.assert_called_once_with(
            "SPY", "sell", 40, 490.0, AssetClass.US_EQUITY)

    def test_dry_run_reconcile_mutates_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bot.main.EXECUTION",
                            ExecutionConfig(ladder_enabled=True, ladder_passive_bps=2.0))
        engine = _crypto_engine(tmp_path, monkeypatch, dry_run=True)
        engine.data.latest_quote.return_value = SPY_QUOTE
        engine.state.set_ladder_order("SPY", _ladder_marker())
        now = datetime.now(timezone.utc)

        engine.reconcile_equity_ladders({}, _in_window_session(now), now)

        engine.broker.cancel_open_orders.assert_not_called()
        engine.broker.limit_order.assert_not_called()
        assert engine.state.ladder_order("SPY") is not None


class TestLadderMetaExemption:
    def test_positions_preserves_stop_while_passive_rests(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.get_positions.return_value = {}
        engine.state.set_position_meta("SPY", 490.0, 500.0)
        engine.state.set_ladder_order("SPY", _ladder_marker())

        engine.positions()

        assert engine.state.position_meta("SPY") is not None

    def test_positions_clears_stop_without_ladder_or_resting(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.get_positions.return_value = {}
        engine.state.set_position_meta("SPY", 490.0, 500.0)

        engine.positions()

        assert engine.state.position_meta("SPY") is None


class TestLadderStateStore:
    def test_ladder_round_trip(self, tmp_path):
        path = str(tmp_path / "state.json")
        store = StateStore(path)
        payload = _ladder_marker()
        store.set_ladder_order("SPY", payload)
        assert StateStore(path).ladder_order("SPY") == payload

    def test_ladder_absent_is_none(self, tmp_path):
        assert StateStore(str(tmp_path / "state.json")).ladder_order("SPY") is None

    def test_ladder_clear_idempotent(self, tmp_path):
        path = str(tmp_path / "state.json")
        store = StateStore(path)
        store.set_ladder_order("SPY", _ladder_marker())
        store.clear_ladder_order("SPY")
        assert StateStore(path).ladder_order("SPY") is None
        store.clear_ladder_order("SPY")  # idempotent


# ---------------------------------------------------------------------------
# Task 1e — submission-time jitter inside the near-close window
# ---------------------------------------------------------------------------

class TestJitterOffset:
    def test_zero_max_is_no_offset(self):
        assert jitter_offset_minutes("SPY", "2026-07-15", 0.0, 10.0, False) == 0.0

    def test_within_window_and_leaves_escalation_room(self):
        # ladder on: cap = min(5, window 10 - 2 poll cycles) = 5, so the target
        # lands >= 2 min before the close.
        off = jitter_offset_minutes("SPY", "2026-07-15", 5.0, 10.0, ladder_enabled=True)
        assert 0.0 <= off <= 5.0
        assert off <= 10.0 - 2.0

    def test_deterministic_for_symbol_date(self):
        a = jitter_offset_minutes("SPY", "2026-07-15", 5.0, 10.0, False)
        b = jitter_offset_minutes("SPY", "2026-07-15", 5.0, 10.0, False)
        assert a == b

    def test_differs_across_symbols(self):
        a = jitter_offset_minutes("SPY", "2026-07-15", 5.0, 10.0, False)
        b = jitter_offset_minutes("QQQ", "2026-07-15", 5.0, 10.0, False)
        assert a != b

    def test_differs_across_dates(self):
        a = jitter_offset_minutes("SPY", "2026-07-15", 5.0, 10.0, False)
        b = jitter_offset_minutes("SPY", "2026-07-16", 5.0, 10.0, False)
        assert a != b

    def test_short_window_ladder_safety_yields_no_offset(self):
        # window 1 min can't fit the 2-min ladder safety margin -> no jitter.
        assert jitter_offset_minutes("SPY", "2026-07-15", 5.0, 1.0, True) == 0.0


class TestJitterSubmissionGate:
    CLOSE = datetime(2026, 7, 15, 20, 0, tzinfo=timezone.utc)  # 16:00 EDT
    SESSION = (datetime(2026, 7, 15, 13, 30, tzinfo=timezone.utc), CLOSE)

    def test_defers_before_target_then_allows(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "bot.main.EXECUTION",
            ExecutionConfig(submit_jitter_max_minutes=5.0, near_close_minutes=10))
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]  # daily equity
        off = jitter_offset_minutes("SPY", "2026-07-15", 5.0, 10.0, False)
        window_start = self.CLOSE - timedelta(minutes=10)
        target = window_start + timedelta(minutes=off)

        assert engine.equity_submission_allowed(
            cfg, target - timedelta(seconds=30), self.SESSION) is False
        assert engine.equity_submission_allowed(
            cfg, target + timedelta(seconds=30), self.SESSION) is True

    def test_zero_jitter_allows_at_window_start(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)  # default jitter 0
        cfg, _ = engine.strategies["SPY"]
        at_start = self.CLOSE - timedelta(minutes=10)
        assert engine.equity_submission_allowed(cfg, at_start, self.SESSION) is True

# ---------------------------------------------------------------------------
# Exit / stop reliability: failure isolation, same-day retry, stop correctness,
# broker-side stop-fill reconciliation
# ---------------------------------------------------------------------------


def _hold_engine(tmp_path, monkeypatch, symbol, pos, atr=5.0, stop_atr_mult=3.0):
    """Engine whose strategy always HOLDs, so process_instrument exercises only
    the position-maintenance path."""
    engine = _crypto_engine(tmp_path, monkeypatch)
    cfg, _ = engine.strategies[symbol]
    strat = MagicMock()
    strat.min_bars = 2
    strat.stop_atr_mult = stop_atr_mult
    strat.prepare.return_value = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0], "atr": [atr, atr, atr]})
    strat.signal.return_value = Signal(Action.HOLD, "hold")
    engine.strategies[symbol] = (cfg, strat)
    idx = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
    engine.data.get_recent_bars.return_value = pd.DataFrame(
        {"close": [1, 2, 3, 4, 5]}, index=idx)
    engine.broker.get_positions.return_value = {symbol: pos}
    return engine


class TestExitFailureIsolation:
    def test_failure_preserves_state_and_records_it(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        engine.state.set_position_meta("SPY", 490.0, 505.0)
        engine.broker.close_position.side_effect = RuntimeError(
            "insufficient qty available")
        pos = PositionState("SPY", "long", 100, 500.0, 490.0, 505.0)

        assert engine.exit_position(cfg, pos, 505.0, "snap-back") is False

        assert engine.state.position_meta("SPY") is not None  # still ours to exit
        failure = engine.state.exit_failure("SPY")
        assert failure["count"] == 1
        assert "insufficient qty" in failure["error"]
        assert engine.state.pending_fill("SPY") is None
        assert trades_rows(tmp_path) == [TradeLog.header]  # no phantom trade

    def test_failure_restores_protective_stop_at_effective_level(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["USO"]  # trend following, trail 3 ATR
        engine.broker.close_position.side_effect = RuntimeError("403 held_for_orders")
        pos = PositionState("USO", "long", 10, 100.0, 95.0, 120.0)

        engine.exit_position(cfg, pos, 118.0, "death cross", atr_value=5.0)

        engine.broker.stop_order.assert_called_once_with(
            "USO", "sell", 10, 105.0, AssetClass.US_EQUITY)  # max(95, 120 - 3*5)

    def test_repeated_failures_keep_first_seen_and_count_up(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        engine.broker.close_position.side_effect = RuntimeError("403")
        pos = PositionState("SPY", "long", 100, 500.0, 490.0, 505.0)

        engine.exit_position(cfg, pos, 505.0, "snap-back")
        first_seen = engine.state.exit_failure("SPY")["since"]
        engine.exit_position(cfg, pos, 506.0, "snap-back")

        failure = engine.state.exit_failure("SPY")
        assert failure["count"] == 2
        assert failure["since"] == first_seen

    def test_success_clears_failure_marker(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        engine.state.record_exit_failure("SPY", "403")
        pos = PositionState("SPY", "long", 100, 500.0, 490.0, 505.0)

        assert engine.exit_position(cfg, pos, 505.0, "snap-back") is True

        assert engine.state.exit_failure("SPY") is None
        assert engine.state.position_meta("SPY") is None

    def test_crypto_failure_places_no_broker_stop(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["BTC/USD"]
        engine.broker.close_position.side_effect = RuntimeError("boom")
        pos = PositionState("BTC/USD", "long", 0.3, 50_000.0, 47_000.0, 50_000.0)

        assert engine.exit_position(cfg, pos, 46_000.0, "software hard stop") is False

        engine.broker.stop_order.assert_not_called()  # crypto stops are software

    def test_exit_signal_failure_makes_process_instrument_report_unevaluated(
            self, tmp_path, monkeypatch):
        pos = PositionState("SPY", "long", 100, 500.0, 490.0, 505.0)
        engine = _hold_engine(tmp_path, monkeypatch, "SPY", pos)
        engine.strategies["SPY"][1].signal.return_value = Signal(Action.EXIT, "snap-back")
        engine.state.set_position_meta("SPY", 490.0, 505.0)
        engine.broker.close_position.side_effect = RuntimeError("403")

        assert engine.process_instrument("SPY") is False


class TestBarRetryAndCycleIsolation:
    def test_boundary_unmarked_when_processing_reports_unevaluated(
            self, tmp_path, monkeypatch):
        now = datetime.now(timezone.utc)
        engine = _make_engine(tmp_path, monkeypatch, _in_window_session(now))
        engine.process_instrument = MagicMock(return_value=False)
        engine.run_cycle()
        assert "SPY" not in engine.last_boundary  # retried next cycle, not next day

    def test_one_instrument_raising_does_not_abort_the_cycle(self, tmp_path, monkeypatch):
        now = datetime.now(timezone.utc)
        engine = _make_engine(tmp_path, monkeypatch, _in_window_session(now))

        def boom(symbol, **kwargs):
            if symbol == "SPY":
                raise RuntimeError("exit blew up")
            return True

        engine.process_instrument = MagicMock(side_effect=boom)
        engine.run_cycle()

        assert _evaluated(engine, "BTC/USD")   # loop continued
        assert "SPY" not in engine.last_boundary
        assert "BTC/USD" in engine.last_boundary

    def test_equity_entry_suppressed_while_an_order_is_outstanding(
            self, tmp_path, monkeypatch):
        engine = _long_signal_engine(tmp_path, monkeypatch, "SPY", Q102)
        engine.risk.approve_entry = MagicMock(
            return_value=MagicMock(qty=100, stop_price=98.0))
        engine.broker.has_open_order.return_value = True

        assert engine.process_instrument("SPY") is True  # bar handled, no retry
        engine.broker.limit_order.assert_not_called()


class TestCircuitBreakerFlatten:
    def _engine(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.get_positions.return_value = {
            "SPY": PositionState("SPY", "long", 100, 500.0, 0.0, 500.0),
            "USO": PositionState("USO", "long", 10, 100.0, 0.0, 100.0),
        }
        engine.state.set_position_meta("SPY", 490.0, 505.0)
        engine.state.set_position_meta("USO", 95.0, 100.0)
        engine.state.data["peak_equity"] = 100_000.0
        return engine

    def test_halt_written_and_every_symbol_attempted_when_a_close_fails(
            self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        engine.broker.close_position.side_effect = RuntimeError("403 held_for_orders")

        assert engine.check_circuit_breaker(85_000.0) is True

        assert (tmp_path / "HALT").exists()  # halted despite the failures
        assert engine.broker.close_position.call_count == 2
        assert engine.state.exit_failure("SPY") is not None
        assert engine.state.exit_failure("USO") is not None

    def test_clean_flatten_halts_without_failures(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        assert engine.check_circuit_breaker(85_000.0) is True
        assert (tmp_path / "HALT").exists()
        assert engine.state.exit_failure("SPY") is None

    def test_no_halt_above_threshold(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        assert engine.check_circuit_breaker(99_000.0) is False
        assert not (tmp_path / "HALT").exists()

    def test_the_halt_log_names_the_configured_threshold(
            self, tmp_path, monkeypatch, caplog):
        """The line the operator reads must be the knob that fired, not a
        number typed beside it."""
        engine = self._engine(tmp_path, monkeypatch)
        engine.risk.risk = replace(engine.risk.risk, max_drawdown=0.25)

        with caplog.at_level(logging.ERROR):
            assert engine.check_circuit_breaker(70_000.0) is True

        assert "25.0%" in caplog.text
        assert "10%" not in caplog.text


class TestStopReplacementLevel:
    def test_trailing_strategy_replaced_at_effective_stop(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        monkeypatch.setattr(engine, "_latest_atr", lambda symbol: 5.0)
        engine.broker.has_open_stop.return_value = False
        pos = PositionState("USO", "long", 10, 100.0, 95.0, 120.0)

        engine.reconcile_equity_stops({"USO": pos})

        engine.broker.stop_order.assert_called_once_with(
            "USO", "sell", 10, 105.0, AssetClass.US_EQUITY)  # trail binds, not 95

    def test_non_trailing_strategy_keeps_entry_stop(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        atr = MagicMock(side_effect=AssertionError("no trail: ATR must not be needed"))
        monkeypatch.setattr(engine, "_latest_atr", atr)
        engine.broker.has_open_stop.return_value = False
        pos = PositionState("SPY", "long", 100, 741.0, 704.0, 740.0)

        engine.reconcile_equity_stops({"SPY": pos})

        engine.broker.stop_order.assert_called_once_with(
            "SPY", "sell", 100, 704.0, AssetClass.US_EQUITY)

    def test_atr_failure_degrades_to_entry_stop(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        monkeypatch.setattr(engine, "_latest_atr", lambda symbol: None)
        engine.broker.has_open_stop.return_value = False
        pos = PositionState("USO", "long", 10, 100.0, 95.0, 120.0)

        engine.reconcile_equity_stops({"USO": pos})

        engine.broker.stop_order.assert_called_once_with(
            "USO", "sell", 10, 95.0, AssetClass.US_EQUITY)


class TestMissingStopRecovery:
    def test_recomputes_atr_sized_stop_not_one_percent(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.get_positions.return_value = {
            "SPY": PositionState("SPY", "long", 100, 741.0, 0.0, 741.0)}
        monkeypatch.setattr(engine, "_latest_atr", lambda symbol: 12.0)

        positions = engine.positions()

        # stop_mult 3.0 x ATR 12 = 36 below entry, not the 1% (733.59) fallback.
        assert positions["SPY"].stop_price == pytest.approx(705.0)
        meta = engine.state.position_meta("SPY")
        assert meta["stop_price"] == pytest.approx(705.0)
        assert "degraded" not in meta

    def test_short_recovers_stop_above_entry(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.get_positions.return_value = {
            "SPY": PositionState("SPY", "short", 100, 741.0, 0.0, 741.0)}
        monkeypatch.setattr(engine, "_latest_atr", lambda symbol: 12.0)

        positions = engine.positions()

        assert positions["SPY"].stop_price == pytest.approx(777.0)

    def test_atr_unavailable_marks_meta_degraded(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.get_positions.return_value = {
            "SPY": PositionState("SPY", "long", 100, 741.0, 0.0, 741.0)}
        monkeypatch.setattr(engine, "_latest_atr", lambda symbol: None)

        positions = engine.positions()

        assert positions["SPY"].stop_price == pytest.approx(733.59)  # 1% fallback
        assert engine.state.position_meta("SPY")["degraded"] is True

    def test_degraded_stop_refined_on_next_processed_bar(self, tmp_path, monkeypatch):
        pos = PositionState("SPY", "long", 100, 741.0, 0.0, 741.0)
        engine = _hold_engine(tmp_path, monkeypatch, "SPY", pos, atr=12.0)
        engine.state.set_position_meta("SPY", 733.59, 741.0, degraded=True)

        engine.process_instrument("SPY")

        meta = engine.state.position_meta("SPY")
        assert meta["stop_price"] == pytest.approx(705.0)
        assert "degraded" not in meta

    def test_healthy_meta_is_not_recomputed(self, tmp_path, monkeypatch):
        pos = PositionState("SPY", "long", 100, 741.0, 0.0, 741.0)
        engine = _hold_engine(tmp_path, monkeypatch, "SPY", pos, atr=12.0)
        engine.state.set_position_meta("SPY", 704.0, 741.0)

        engine.process_instrument("SPY")

        assert engine.state.position_meta("SPY")["stop_price"] == pytest.approx(704.0)

    def test_latest_atr_reads_the_strategy_frame(self, tmp_path, monkeypatch):
        pos = PositionState("SPY", "long", 100, 741.0, 0.0, 741.0)
        engine = _hold_engine(tmp_path, monkeypatch, "SPY", pos, atr=12.0)
        assert engine._latest_atr("SPY") == pytest.approx(12.0)

    def test_latest_atr_none_when_bars_unavailable(self, tmp_path, monkeypatch):
        pos = PositionState("SPY", "long", 100, 741.0, 0.0, 741.0)
        engine = _hold_engine(tmp_path, monkeypatch, "SPY", pos)
        engine.data.get_recent_bars.side_effect = RuntimeError("no bars")
        assert engine._latest_atr("SPY") is None


def _filled_order(qty="100", price="704.0", order_type="OrderType.STOP",
                  status="filled", filled_at=None):
    """filled_at defaults to "just now" because the vanish sweep only books an
    order that filled after the position was last seen. Pass False for an order
    the broker reports with no fill timestamp."""
    if filled_at is None:
        filled_at = datetime.now(timezone.utc)
    return MagicMock(filled_qty=qty, filled_avg_price=price, order_type=order_type,
                     status=status, filled_at=filled_at or None)


class TestStopFillReconciliation:
    def _vanished(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_position_meta("SPY", 704.0, 741.0)
        engine.state.set_last_seen_position("SPY", "long", 100, 741.0)
        engine.broker.get_positions.return_value = {}
        return engine

    def test_positions_records_last_seen_snapshot(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_position_meta("SPY", 704.0, 741.0)
        engine.broker.get_positions.return_value = {
            "SPY": PositionState("SPY", "long", 100, 741.0, 0.0, 741.0)}

        engine.positions()

        seen = engine.state.last_seen_position("SPY")
        assert (seen["side"], seen["qty"], seen["entry_price"]) == ("long", 100, 741.0)

    def test_broker_stop_fill_logged_to_trades_and_fills(self, tmp_path, monkeypatch):
        engine = self._vanished(tmp_path, monkeypatch)
        engine.broker.closed_orders.return_value = [_filled_order()]

        engine.positions()

        trades = trades_rows(tmp_path)
        assert len(trades) == 2
        assert trades[1][1] == "SPY" and trades[1][2] == "long"
        assert float(trades[1][3]) == pytest.approx(741.0)
        assert float(trades[1][4]) == pytest.approx(704.0)
        assert float(trades[1][5]) == pytest.approx(-3700.0)
        fills = _fills_rows(tmp_path)
        assert fills[1][3] == "stop_fill"
        assert float(fills[1][7]) == pytest.approx(704.0)
        assert engine.state.position_meta("SPY") is None
        assert engine.state.last_seen_position("SPY") is None

    def test_unfilled_closed_orders_ignored(self, tmp_path, monkeypatch):
        engine = self._vanished(tmp_path, monkeypatch)
        engine.broker.closed_orders.return_value = [
            _filled_order(status="canceled", price=None), _filled_order()]

        engine.positions()

        assert len(trades_rows(tmp_path)) == 2

    def test_missing_closed_order_still_clears_state(self, tmp_path, monkeypatch):
        engine = self._vanished(tmp_path, monkeypatch)
        engine.broker.closed_orders.return_value = []

        engine.positions()

        assert len(trades_rows(tmp_path)) == 1  # header only
        assert engine.state.position_meta("SPY") is None

    def test_broker_query_failure_still_clears_state(self, tmp_path, monkeypatch):
        engine = self._vanished(tmp_path, monkeypatch)
        engine.broker.closed_orders.side_effect = RuntimeError("api down")

        engine.positions()

        assert engine.state.position_meta("SPY") is None

    def test_dry_run_logs_nothing(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch, dry_run=True)
        engine.state.set_position_meta("SPY", 704.0, 741.0)
        engine.state.set_last_seen_position("SPY", "long", 100, 741.0)
        engine.broker.get_positions.return_value = {}
        engine.broker.closed_orders.return_value = [_filled_order()]

        engine.positions()

        assert len(trades_rows(tmp_path)) == 1
        assert len(_fills_rows(tmp_path)) == 1

    def test_engine_exit_does_not_double_log(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        engine.state.set_position_meta("SPY", 704.0, 741.0)
        engine.state.set_last_seen_position("SPY", "long", 100, 741.0)
        pos = PositionState("SPY", "long", 100, 741.0, 704.0, 741.0)

        engine.exit_position(cfg, pos, 750.0, "snap-back")
        engine.broker.get_positions.return_value = {}
        engine.positions()
        engine.broker.get_order_fill.return_value = (100.0, 751.0, "filled")
        engine.reconcile_pending_fills()

        assert len(trades_rows(tmp_path)) == 2  # only the reconciled exit row
        engine.broker.closed_orders.assert_not_called()


class TestExitFailureState:
    def test_round_trip_and_clear(self, tmp_path):
        path = str(tmp_path / "state.json")
        store = StateStore(path)
        store.record_exit_failure("SPY", "403")
        assert StateStore(path).exit_failure("SPY")["error"] == "403"
        store.clear_exit_failure("SPY")
        assert StateStore(path).exit_failure("SPY") is None
        store.clear_exit_failure("SPY")  # idempotent

    def test_last_seen_round_trip(self, tmp_path):
        path = str(tmp_path / "state.json")
        store = StateStore(path)
        store.set_last_seen_position("SPY", "long", 100, 741.0)
        assert StateStore(path).last_seen_position("SPY")["entry_price"] == 741.0
        store.clear_last_seen_position("SPY")
        assert StateStore(path).last_seen_position("SPY") is None
        store.clear_last_seen_position("SPY")  # idempotent


class TestExitInFlight:
    def _exiting(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        pos = PositionState("SPY", "long", 100, 741.0, 704.0, 741.0)
        engine.state.set_position_meta("SPY", 704.0, 741.0)
        engine.state.set_last_seen_position("SPY", "long", 100, 741.0)
        engine.exit_position(cfg, pos, 750.0, "snap-back")  # leaves an exit marker
        return engine, pos

    def test_no_stop_re_armed_under_a_working_close(self, tmp_path, monkeypatch):
        engine, pos = self._exiting(tmp_path, monkeypatch)
        engine.broker.has_open_stop.return_value = False

        engine.reconcile_equity_stops({"SPY": pos})

        engine.broker.stop_order.assert_not_called()

    def test_meta_not_resurrected_while_close_is_working(self, tmp_path, monkeypatch):
        engine, pos = self._exiting(tmp_path, monkeypatch)
        # The close is submitted but unfilled, so the broker still reports it.
        engine.broker.get_positions.return_value = {
            "SPY": PositionState("SPY", "long", 100, 741.0, 0.0, 741.0)}

        engine.positions()

        assert engine.state.position_meta("SPY") is None
        assert engine.state.last_seen_position("SPY") is None

    def test_settled_close_books_exactly_one_trade(self, tmp_path, monkeypatch):
        engine, pos = self._exiting(tmp_path, monkeypatch)
        engine.broker.get_positions.return_value = {
            "SPY": PositionState("SPY", "long", 100, 741.0, 0.0, 741.0)}
        engine.positions()                                   # close still working
        engine.broker.get_positions.return_value = {}
        engine.broker.closed_orders.return_value = [
            _filled_order(price="750.0", order_type="OrderType.MARKET")]
        engine.broker.get_order_fill.return_value = (100.0, 750.0, "filled")

        engine.positions()                                   # close settled
        engine.reconcile_pending_fills()

        assert len(trades_rows(tmp_path)) == 2  # only the reconciled exit row


class TestTerminalUnfilledOrders:
    def test_rejected_close_clears_marker_and_records_failure(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_pending_fill("SPY", _pending(event="exit", side="sell"))
        engine.broker.get_order_fill.return_value = (0.0, None, "rejected")

        engine.reconcile_pending_fills()

        assert engine.state.pending_fill("SPY") is None  # stop supervision resumes
        assert engine.state.exit_failure("SPY")["error"].endswith("rejected")
        assert len(_fills_rows(tmp_path)) == 1  # header only: nothing was filled

    def test_expired_entry_clears_marker_without_failure(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_pending_fill("SPY", _pending(event="entry"))
        engine.broker.get_order_fill.return_value = (0.0, None, "expired")

        engine.reconcile_pending_fills()

        assert engine.state.pending_fill("SPY") is None
        assert engine.state.exit_failure("SPY") is None

    def test_working_order_keeps_marker(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_pending_fill("SPY", _pending(event="exit"))
        engine.broker.get_order_fill.return_value = (0.0, None, "accepted")

        engine.reconcile_pending_fills()

        assert engine.state.pending_fill("SPY") is not None


class TestExitFailureSelfHeals:
    def test_failure_cleared_once_the_position_is_flat(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.record_exit_failure("SPY", "403 held_for_orders")
        engine.broker.get_positions.return_value = {}
        engine.broker.closed_orders.return_value = []

        engine.positions()

        assert engine.state.exit_failure("SPY") is None

    def test_failure_kept_while_the_position_is_still_open(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.record_exit_failure("SPY", "403 held_for_orders")
        engine.state.set_position_meta("SPY", 704.0, 741.0)
        engine.broker.get_positions.return_value = {
            "SPY": PositionState("SPY", "long", 100, 741.0, 0.0, 741.0)}

        engine.positions()

        assert engine.state.exit_failure("SPY") is not None


# ---------------------------------------------------------------------------
# Task 8a — signal bar == fill bar for daily equities in the near-close window
# ---------------------------------------------------------------------------

# 2026-08-31 is EDT: daily equity bars are stamped 00:00 ET == 04:00Z, and the
# session closes 20:00Z. 19:50Z is the first instant of the near-close window.
NOW_IN_WINDOW = datetime(2026, 8, 31, 19, 52, tzinfo=timezone.utc)
NOW_MIDSESSION = datetime(2026, 8, 31, 17, 0, tzinfo=timezone.utc)
SESSION_TODAY = (datetime(2026, 8, 31, 13, 30, tzinfo=timezone.utc),
                 datetime(2026, 8, 31, 20, 0, tzinfo=timezone.utc))
YESTERDAY_CLOSE = 120.46
TODAY_CLOSE = 129.22


def _daily_equity_engine(tmp_path, monkeypatch, symbol="USO"):
    """Engine fed a daily equity frame whose last bar is the still-forming
    session bar, with a pass-through strategy so the bar actually consumed is
    observable in the signal call."""
    engine = _crypto_engine(tmp_path, monkeypatch)
    cfg, _ = engine.strategies[symbol]
    strat = MagicMock()
    strat.min_bars = 2
    strat.stop_atr_mult = 3.0
    strat.prepare.side_effect = lambda bars: bars.assign(atr=1.0)
    strat.signal.return_value = Signal(Action.HOLD, "hold")
    engine.strategies[symbol] = (cfg, strat)
    idx = pd.DatetimeIndex([pd.Timestamp("2026-08-27T04:00:00+00:00"),
                            pd.Timestamp("2026-08-28T04:00:00+00:00"),
                            pd.Timestamp("2026-08-31T04:00:00+00:00")])
    engine.data.get_recent_bars.return_value = pd.DataFrame(
        {"close": [118.0, YESTERDAY_CLOSE, TODAY_CLOSE]}, index=idx)
    return engine, strat


def _decided_close(strat):
    df, i, _ = strat.signal.call_args.args
    return float(df["close"].iloc[i])


class TestDecisionBarKey:
    def test_daily_equity_keys_off_the_et_session_date(self):
        assert decision_bar_key(_cfg("USO"), NOW_IN_WINDOW) == "2026-08-31"

    def test_daily_equity_key_stable_across_the_session(self):
        assert (decision_bar_key(_cfg("USO"), NOW_MIDSESSION)
                == decision_bar_key(_cfg("USO"), NOW_IN_WINDOW))

    def test_daily_crypto_keys_off_the_utc_boundary(self):
        key = decision_bar_key(_cfg("BTC/USD"), NOW_IN_WINDOW)
        assert key == datetime(2026, 8, 31, tzinfo=timezone.utc).isoformat()

    def test_intraday_equity_keys_off_the_utc_boundary(self):
        key = decision_bar_key(_intraday_equity_cfg(), NOW_IN_WINDOW)
        assert key == datetime(2026, 8, 31, 16, 0, tzinfo=timezone.utc).isoformat()


def _cfg(symbol):
    return next(c for c in INSTRUMENTS if c.symbol == symbol)


class TestFormingBarDecision:
    def test_daily_equity_in_window_decides_on_todays_bar(self, tmp_path, monkeypatch):
        engine, strat = _daily_equity_engine(tmp_path, monkeypatch)
        engine.process_instrument("USO", now_utc=NOW_IN_WINDOW, session=SESSION_TODAY)
        assert _decided_close(strat) == pytest.approx(TODAY_CLOSE)

    def test_daily_equity_outside_window_decides_on_the_completed_bar(
            self, tmp_path, monkeypatch):
        engine, strat = _daily_equity_engine(tmp_path, monkeypatch)
        engine.process_instrument("USO", now_utc=NOW_MIDSESSION, session=SESSION_TODAY)
        assert _decided_close(strat) == pytest.approx(YESTERDAY_CLOSE)

    def test_no_session_falls_back_to_completed_bars(self, tmp_path, monkeypatch):
        engine, strat = _daily_equity_engine(tmp_path, monkeypatch)
        engine.process_instrument("USO", now_utc=NOW_IN_WINDOW, session=None)
        assert _decided_close(strat) == pytest.approx(YESTERDAY_CLOSE)

    def test_intraday_equity_never_keeps_the_forming_bar(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        assert engine.keeps_forming_bar(_intraday_equity_cfg(), NOW_IN_WINDOW,
                                        SESSION_TODAY) is False

    def test_crypto_never_keeps_the_forming_bar(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        assert engine.keeps_forming_bar(_cfg("BTC/USD"), NOW_IN_WINDOW,
                                        SESSION_TODAY) is False

    def test_daily_equity_keeps_forming_bar_only_in_window(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        assert engine.keeps_forming_bar(_cfg("USO"), NOW_IN_WINDOW, SESSION_TODAY) is True
        assert engine.keeps_forming_bar(_cfg("USO"), NOW_MIDSESSION,
                                        SESSION_TODAY) is False


class TestSessionDedupe:
    def test_same_session_processed_once(self, tmp_path, monkeypatch):
        now = datetime.now(timezone.utc)
        engine = _make_engine(tmp_path, monkeypatch, _in_window_session(now))
        engine.run_cycle()
        engine.run_cycle()
        spy_calls = [c for c in engine.process_instrument.call_args_list
                     if c.args[0] == "SPY"]
        assert len(spy_calls) == 1

    def test_dedupe_key_is_the_et_session_date_for_daily_equities(
            self, tmp_path, monkeypatch):
        now = datetime.now(timezone.utc)
        engine = _make_engine(tmp_path, monkeypatch, _in_window_session(now))
        engine.run_cycle()
        assert engine.last_boundary["SPY"] == decision_bar_key(_cfg("SPY"), now)

    def test_failed_bar_still_retried_within_the_same_session(self, tmp_path, monkeypatch):
        now = datetime.now(timezone.utc)
        engine = _make_engine(tmp_path, monkeypatch, _in_window_session(now))
        engine.process_instrument = MagicMock(return_value=False)
        engine.run_cycle()
        engine.run_cycle()
        spy_calls = [c for c in engine.process_instrument.call_args_list
                     if c.args[0] == "SPY"]
        assert len(spy_calls) == 2


# ---------------------------------------------------------------------------
# Task 8b — decision-price guardrail and fill-priced trades.csv
# ---------------------------------------------------------------------------

class TestDecisionPriceGuard:
    def _spy(self):
        return PositionState("SPY", "long", 100, 500.0, 490.0, 505.0)

    def test_exit_skipped_when_the_bar_disagrees_with_the_market(
            self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        engine.data.latest_quote.return_value = LatestQuote(539.9, 540.1, 540.0, 4.0)

        assert engine.exit_position(cfg, self._spy(), 505.0, "trail stop") is False

        engine.broker.close_position.assert_not_called()
        engine.broker.stop_order.assert_not_called()  # nothing cancelled to restore
        assert trades_rows(tmp_path) == [TradeLog.header]
        divergence = engine.state.data_divergence("SPY")
        assert divergence["count"] == 1
        assert "505.00" in divergence["detail"] and "540.00" in divergence["detail"]

    def test_exit_proceeds_within_tolerance_and_clears_the_alert(
            self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        engine.state.record_data_divergence("SPY", "stale")
        engine.data.latest_quote.return_value = LatestQuote(504.9, 505.1, 505.0, 4.0)

        assert engine.exit_position(cfg, self._spy(), 505.0, "trail stop") is True

        engine.broker.close_position.assert_called_once_with("SPY")
        assert engine.state.data_divergence("SPY") is None

    def test_missing_quote_fails_open(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        engine.data.latest_quote.return_value = None
        assert engine.exit_position(cfg, self._spy(), 505.0, "trail stop") is True

    def test_repeated_divergence_counts_up_keeping_first_seen(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        engine.data.latest_quote.return_value = LatestQuote(539.9, 540.1, 540.0, 4.0)
        engine.exit_position(cfg, self._spy(), 505.0, "trail stop")
        since = engine.state.data_divergence("SPY")["since"]
        engine.exit_position(cfg, self._spy(), 505.0, "trail stop")
        assert engine.state.data_divergence("SPY")["count"] == 2
        assert engine.state.data_divergence("SPY")["since"] == since

    def test_entry_skipped_without_marking_the_bar(self, tmp_path, monkeypatch):
        engine = _long_signal_engine(tmp_path, monkeypatch, "SPY",
                                     LatestQuote(199.9, 200.1, 200.0, 4.0))
        engine.risk.approve_entry = MagicMock(
            return_value=MagicMock(qty=100, stop_price=98.0))

        assert engine.process_instrument("SPY") is False

        engine.broker.limit_order.assert_not_called()
        assert engine.state.data_divergence("SPY") is not None

    def test_entry_proceeds_when_the_quote_agrees(self, tmp_path, monkeypatch):
        engine = _long_signal_engine(tmp_path, monkeypatch, "SPY", Q102)
        engine.risk.approve_entry = MagicMock(
            return_value=MagicMock(qty=100, stop_price=98.0))

        assert engine.process_instrument("SPY") is True

        engine.broker.limit_order.assert_called_once()
        assert engine.state.data_divergence("SPY") is None


class TestReconciledExitTrade:
    def _exited(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        pos = PositionState("SPY", "long", 100, 500.0, 490.0, 505.0)
        engine.exit_position(cfg, pos, 505.0, "exit signal")
        return engine

    def test_submit_books_no_trade_row(self, tmp_path, monkeypatch):
        engine = self._exited(tmp_path, monkeypatch)
        assert trades_rows(tmp_path) == [TradeLog.header]
        marker = engine.state.pending_fill("SPY")
        assert marker["entry_price"] == pytest.approx(500.0)
        assert marker["position_side"] == "long"
        assert marker["qty"] == 100

    def test_fill_books_the_trade_at_the_fill_price(self, tmp_path, monkeypatch):
        engine = self._exited(tmp_path, monkeypatch)
        engine.broker.get_order_fill.return_value = (100.0, 511.0, "filled")

        engine.reconcile_pending_fills()

        rows = trades_rows(tmp_path)
        assert len(rows) == 2
        assert float(rows[1][3]) == pytest.approx(500.0)   # entry
        assert float(rows[1][4]) == pytest.approx(511.0)   # the FILL, not 505.0
        assert float(rows[1][5]) == pytest.approx(1100.0)  # 100 x (511 - 500)
        assert float(rows[1][6]) == pytest.approx(100.0)

    def test_short_exit_pnl_sign(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        pos = PositionState("SPY", "short", 10, 500.0, 510.0, 495.0)
        engine.exit_position(cfg, pos, 495.0, "exit signal")
        engine.broker.get_order_fill.return_value = (10.0, 496.0, "filled")

        engine.reconcile_pending_fills()

        assert float(trades_rows(tmp_path)[1][5]) == pytest.approx(40.0)

    def test_partial_fill_books_the_filled_quantity(self, tmp_path, monkeypatch):
        engine = self._exited(tmp_path, monkeypatch)
        engine.broker.get_order_fill.return_value = (60.0, 511.0, "filled")

        engine.reconcile_pending_fills()

        assert float(trades_rows(tmp_path)[1][6]) == pytest.approx(60.0)

    def test_unfilled_close_books_nothing(self, tmp_path, monkeypatch):
        engine = self._exited(tmp_path, monkeypatch)
        engine.broker.get_order_fill.return_value = (0.0, None, "rejected")

        engine.reconcile_pending_fills()

        assert trades_rows(tmp_path) == [TradeLog.header]

    def test_entry_fill_books_no_trade_row(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_pending_fill("SPY", _pending(event="entry"))
        engine.broker.get_order_fill.return_value = (100.0, 500.4, "filled")

        engine.reconcile_pending_fills()

        assert trades_rows(tmp_path) == [TradeLog.header]


# ---------------------------------------------------------------------------
# Task 8d — corporate-action rescale of persisted raw price levels
# ---------------------------------------------------------------------------

def _uso_frame(prior_close, latest_close):
    idx = pd.DatetimeIndex([pd.Timestamp("2020-04-27T04:00:00+00:00"),
                            pd.Timestamp("2020-04-28T04:00:00+00:00")])
    return pd.DataFrame({"close": [prior_close, latest_close]}, index=idx)


PRIOR_TS = "2020-04-27T04:00:00+00:00"


class TestCorporateActionRescale:
    def _engine_with_levels(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_position_meta("USO", 7.0, 9.0)
        engine.state.set_last_seen_position("USO", "long", 800, 8.5)
        engine.state.set_last_bar("USO", PRIOR_TS, 8.0)
        return engine

    def test_reverse_split_rescales_stop_hwm_and_entry(self, tmp_path, monkeypatch):
        # USO's 2020 1:8 reverse split: adjusted history restates every pre-split
        # print x8, so persisted raw levels must move with it.
        engine = self._engine_with_levels(tmp_path, monkeypatch)

        engine.apply_corporate_action_rescale("USO", _uso_frame(64.0, 60.0))

        meta = engine.state.position_meta("USO")
        assert meta["stop_price"] == pytest.approx(56.0)
        assert meta["high_water_mark"] == pytest.approx(72.0)
        assert engine.state.last_seen_position("USO")["entry_price"] == pytest.approx(68.0)

    def test_stores_the_latest_completed_bar(self, tmp_path, monkeypatch):
        engine = self._engine_with_levels(tmp_path, monkeypatch)
        engine.apply_corporate_action_rescale("USO", _uso_frame(64.0, 60.0))
        stored = engine.state.last_bar("USO")
        assert stored["close"] == pytest.approx(60.0)
        assert stored["timestamp"] == "2020-04-28T04:00:00+00:00"

    def test_ordinary_dividend_drift_leaves_levels_alone(self, tmp_path, monkeypatch):
        engine = self._engine_with_levels(tmp_path, monkeypatch)
        engine.apply_corporate_action_rescale("USO", _uso_frame(8.05, 8.10))
        assert engine.state.position_meta("USO")["stop_price"] == pytest.approx(7.0)

    def test_unknown_prior_bar_is_a_no_op(self, tmp_path, monkeypatch):
        engine = self._engine_with_levels(tmp_path, monkeypatch)
        idx = pd.DatetimeIndex([pd.Timestamp("2021-01-04T05:00:00+00:00")])
        engine.apply_corporate_action_rescale(
            "USO", pd.DataFrame({"close": [64.0]}, index=idx))
        assert engine.state.position_meta("USO")["stop_price"] == pytest.approx(7.0)

    def test_first_sighting_only_records(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_position_meta("USO", 7.0, 9.0)
        engine.apply_corporate_action_rescale("USO", _uso_frame(8.0, 8.2))
        assert engine.state.position_meta("USO")["stop_price"] == pytest.approx(7.0)
        assert engine.state.last_bar("USO")["close"] == pytest.approx(8.2)

    def test_empty_frame_is_a_no_op(self, tmp_path, monkeypatch):
        engine = self._engine_with_levels(tmp_path, monkeypatch)
        engine.apply_corporate_action_rescale("USO", pd.DataFrame(columns=["close"]))
        assert engine.state.last_bar("USO")["close"] == pytest.approx(8.0)

    def test_process_instrument_records_the_completed_bar_not_the_forming_one(
            self, tmp_path, monkeypatch):
        engine, _ = _daily_equity_engine(tmp_path, monkeypatch)
        engine.process_instrument("USO", now_utc=NOW_IN_WINDOW, session=SESSION_TODAY)
        # The forming bar's close moves every minute; only a completed bar is a
        # stable reference for detecting a restatement.
        assert engine.state.last_bar("USO")["close"] == pytest.approx(YESTERDAY_CLOSE)


class TestLastBarState:
    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "state.json")
        store = StateStore(path)
        store.set_last_bar("USO", PRIOR_TS, 8.0)
        assert StateStore(path).last_bar("USO") == {"timestamp": PRIOR_TS, "close": 8.0}

    def test_absent_is_none(self, tmp_path):
        assert StateStore(str(tmp_path / "state.json")).last_bar("USO") is None


class TestDataDivergenceState:
    def test_round_trip_and_clear(self, tmp_path):
        path = str(tmp_path / "state.json")
        store = StateStore(path)
        store.record_data_divergence("USO", "close 120.46 vs mid 129.22")
        assert "120.46" in StateStore(path).data_divergence("USO")["detail"]
        store.clear_data_divergence("USO")
        assert StateStore(path).data_divergence("USO") is None
        store.clear_data_divergence("USO")  # idempotent


# ---------------------------------------------------------------------------
# Task 8c — one feed for live decisions and backtests
# ---------------------------------------------------------------------------

class TestFeedParity:
    def test_live_engine_uses_the_shared_feed(self, monkeypatch):
        captured = {}
        monkeypatch.setattr("bot.main.MarketData",
                            lambda k, s, feed=None: captured.setdefault("feed", feed))
        monkeypatch.setattr("bot.main.Broker", lambda *a, **k: MagicMock())
        monkeypatch.setattr("bot.main.Engine", lambda **k: MagicMock())
        monkeypatch.setattr(sys, "argv", ["bot", "--once"])
        monkeypatch.setenv("ALPACA_API_KEY", "key")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")

        assert bot_main.main() == 0
        assert captured["feed"] == MARKET_DATA_FEED

    def test_backtest_uses_the_shared_feed(self, monkeypatch):
        captured = {}
        monkeypatch.setattr("bot.backtest.MarketData",
                            lambda k, s, feed=None: captured.setdefault("feed", feed))
        monkeypatch.setattr("bot.backtest.load_instruments", lambda *a, **k: [])

        assert backtest.main() == 1  # nothing to backtest, but the feed is set
        assert captured["feed"] == MARKET_DATA_FEED


class TestFormingBarThroughRunCycle:
    def test_run_cycle_hands_the_session_bar_to_a_daily_equity(self, tmp_path, monkeypatch):
        # End-to-end wiring: run_cycle's session/now must reach process_instrument,
        # or the near-close window is computed but never used.
        engine, strat = _daily_equity_engine(tmp_path, monkeypatch)
        now = datetime.now(timezone.utc)
        et_midnight = (now.astimezone(ZoneInfo("America/New_York"))
                       .replace(hour=0, minute=0, second=0, microsecond=0))
        idx = pd.DatetimeIndex([et_midnight - timedelta(days=3),
                                et_midnight - timedelta(days=1), et_midnight])
        engine.data.get_recent_bars.return_value = pd.DataFrame(
            {"close": [118.0, YESTERDAY_CLOSE, TODAY_CLOSE]}, index=idx)
        engine.broker.current_session.return_value = _in_window_session(now)
        engine.broker.market_is_open.return_value = True
        engine.strategies = {"USO": engine.strategies["USO"]}

        engine.run_cycle()

        assert _decided_close(strat) == pytest.approx(TODAY_CLOSE)
        assert engine.last_boundary["USO"] == decision_bar_key(_cfg("USO"), now)


# ---------------------------------------------------------------------------
# Review fixes: halted-but-protecting cycles, breaker recovery, vanish-sweep
# recency, per-symbol stop isolation, partial-close booking
# ---------------------------------------------------------------------------


def _halted_engine(tmp_path, monkeypatch):
    engine = _crypto_engine(tmp_path, monkeypatch)
    (tmp_path / "HALT").write_text("drawdown halt at 2026-08-31T00:00:00+00:00\n")
    engine.maybe_send_briefings = MagicMock()
    engine.broker.has_open_stop.return_value = True
    return engine


class TestHaltedCycleStillProtects:
    def test_briefings_still_sent_under_halt(self, tmp_path, monkeypatch):
        engine = _halted_engine(tmp_path, monkeypatch)

        engine.run_cycle()

        engine.maybe_send_briefings.assert_called_once()

    def test_halt_takes_no_new_positions(self, tmp_path, monkeypatch):
        engine = _halted_engine(tmp_path, monkeypatch)
        engine.process_instrument = MagicMock()

        engine.run_cycle()

        engine.process_instrument.assert_not_called()
        engine.broker.account_equity.assert_not_called()  # breaker not re-run

    def test_missing_equity_stop_is_replaced_under_halt(self, tmp_path, monkeypatch):
        engine = _halted_engine(tmp_path, monkeypatch)
        engine.broker.has_open_stop.return_value = False
        engine.state.set_position_meta("SPY", 704.0, 741.0)
        engine.broker.get_positions.return_value = {
            "SPY": PositionState("SPY", "long", 100, 741.0, 0.0, 741.0)}

        engine.run_cycle()

        engine.broker.stop_order.assert_called_once_with(
            "SPY", "sell", 100, 704.0, AssetClass.US_EQUITY)

    def test_crypto_software_stop_supervised_under_halt(self, tmp_path, monkeypatch):
        engine = _halted_engine(tmp_path, monkeypatch)
        engine.state.set_position_meta("BTC/USD", 47_000.0, 50_000.0)
        engine.broker.get_positions.return_value = {
            "BTC/USD": PositionState("BTC/USD", "long", 0.3, 50_000.0, 0.0, 50_000.0)}
        engine.data.latest_price.return_value = 46_000.0

        engine.run_cycle()

        engine.broker.close_position.assert_called_once_with("BTC/USD")

    def test_pending_fills_reconciled_under_halt(self, tmp_path, monkeypatch):
        engine = _halted_engine(tmp_path, monkeypatch)
        engine.state.set_pending_fill("SPY", _pending())
        engine.broker.get_order_fill.return_value = (100.0, 500.4, "filled")

        engine.run_cycle()

        assert engine.state.pending_fill("SPY") is None
        assert len(_fills_rows(tmp_path)) == 2  # header + the measured fill

    def test_vanished_position_reconciled_under_halt(self, tmp_path, monkeypatch):
        engine = _halted_engine(tmp_path, monkeypatch)
        engine.state.set_position_meta("SPY", 704.0, 741.0)
        engine.state.set_last_seen_position("SPY", "long", 100, 741.0)
        engine.broker.get_positions.return_value = {}
        engine.broker.closed_orders.return_value = [_filled_order()]

        engine.run_cycle()

        assert len(trades_rows(tmp_path)) == 2  # the stop fill reached trades.csv

    def test_sweep_failure_does_not_silence_the_briefing(self, tmp_path, monkeypatch):
        engine = _halted_engine(tmp_path, monkeypatch)
        engine.broker.get_positions.side_effect = RuntimeError("api down")

        engine.run_cycle()  # must not raise

        engine.maybe_send_briefings.assert_called_once()

    def test_resting_entry_ladder_is_cancelled_not_escalated_under_halt(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr("bot.main.EXECUTION",
                            ExecutionConfig(ladder_enabled=True, ladder_passive_bps=2.0))
        engine = _halted_engine(tmp_path, monkeypatch)
        # Flat everywhere: no symbol has a stop resting, so the only cancel in
        # this cycle can be the ladder's, not the orphan sweep's.
        engine.broker.has_open_stop.return_value = False
        engine.state.set_ladder_order("SPY", _ladder_marker())
        engine.broker.get_order_fill.return_value = (0.0, None, "new")

        engine.run_cycle()

        engine.broker.cancel_open_orders.assert_called_once_with("SPY")
        engine.broker.limit_order.assert_not_called()  # no new risk while halted
        assert engine.state.ladder_order("SPY") is None

    def test_unhalted_cycle_still_sweeps_and_briefs(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.maybe_send_briefings = MagicMock()
        engine.state.set_position_meta("SPY", 704.0, 741.0)
        engine.broker.get_positions.return_value = {
            "SPY": PositionState("SPY", "long", 100, 741.0, 0.0, 741.0)}
        engine.broker.market_is_open.return_value = False
        engine.process_instrument = MagicMock()

        engine.run_cycle()

        engine.maybe_send_briefings.assert_called_once()
        engine.broker.stop_order.assert_called_once_with(
            "SPY", "sell", 100, 704.0, AssetClass.US_EQUITY)


class TestBreakerRecoveryPaths:
    def _engine(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.get_positions.return_value = {
            "SPY": PositionState("SPY", "long", 100, 500.0, 0.0, 500.0)}
        engine.state.set_position_meta("SPY", 490.0, 505.0)
        engine.state.data["peak_equity"] = 100_000.0
        engine.broker.has_open_stop.return_value = False
        monkeypatch.setattr(engine, "_latest_atr", lambda symbol: None)
        return engine

    def test_failed_flatten_puts_the_protective_stop_back(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        engine.broker.close_position.side_effect = RuntimeError("403 held_for_orders")

        assert engine.check_circuit_breaker(85_000.0) is True

        engine.broker.stop_order.assert_called_once_with(
            "SPY", "sell", 100, 490.0, AssetClass.US_EQUITY)

    def test_successful_flatten_places_no_stop(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)

        assert engine.check_circuit_breaker(85_000.0) is True

        engine.broker.stop_order.assert_not_called()

    def test_unreadable_positions_still_halt_and_defer_the_flatten(
            self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        engine.broker.get_positions.side_effect = RuntimeError("api down")

        assert engine.check_circuit_breaker(85_000.0) is True  # must not raise

        assert (tmp_path / "HALT").exists()
        engine.broker.close_position.assert_not_called()


class TestRestoreStopSkipsWhenAlreadyProtected:
    def test_timeout_leaves_the_resting_stop_alone(self, tmp_path, monkeypatch, caplog):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        engine.broker.close_position.side_effect = TimeoutError(
            "SPY: open orders still pending cancellation after 10s; not closing")
        engine.broker.has_open_stop.return_value = True
        pos = PositionState("SPY", "long", 100, 741.0, 704.0, 741.0)

        with caplog.at_level("WARNING"):
            assert engine.exit_position(cfg, pos, 750.0, "snap-back") is False

        engine.broker.stop_order.assert_not_called()  # a duplicate would be rejected
        assert "NAKED" not in caplog.text

    def test_no_resting_stop_still_restores(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        engine.broker.close_position.side_effect = RuntimeError("403")
        engine.broker.has_open_stop.return_value = False
        pos = PositionState("SPY", "long", 100, 741.0, 704.0, 741.0)

        engine.exit_position(cfg, pos, 750.0, "snap-back")

        engine.broker.stop_order.assert_called_once_with(
            "SPY", "sell", 100, 704.0, AssetClass.US_EQUITY)


class TestEquityStopSweepIsolation:
    def test_one_rejection_does_not_starve_the_other_symbols(
            self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.has_open_stop.return_value = False
        monkeypatch.setattr(engine, "_latest_atr", lambda symbol: None)
        engine.broker.stop_order.side_effect = (
            lambda symbol, *a, **k: (_ for _ in ()).throw(
                RuntimeError("insufficient qty available")) if symbol == "SPY" else None)
        positions = {"SPY": PositionState("SPY", "long", 100, 741.0, 704.0, 741.0),
                     "USO": PositionState("USO", "long", 10, 100.0, 95.0, 100.0)}

        engine.reconcile_equity_stops(positions)  # must not raise

        assert [c.args[0] for c in engine.broker.stop_order.call_args_list] == \
            ["SPY", "USO"]

    def test_a_failed_stop_lookup_is_isolated_too(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        monkeypatch.setattr(engine, "_latest_atr", lambda symbol: None)
        engine.broker.has_open_stop.side_effect = (
            lambda symbol: (_ for _ in ()).throw(RuntimeError("api down"))
            if symbol == "SPY" else False)
        positions = {"SPY": PositionState("SPY", "long", 100, 741.0, 704.0, 741.0),
                     "USO": PositionState("USO", "long", 10, 100.0, 95.0, 100.0)}

        engine.reconcile_equity_stops(positions)  # must not raise

        engine.broker.stop_order.assert_called_once_with(
            "USO", "sell", 10, 95.0, AssetClass.US_EQUITY)


class TestVanishSweepGuards:
    def test_pending_entry_fill_exempts_the_symbol(self, tmp_path, monkeypatch):
        # The entry limit is working: no position yet, but its stop must survive.
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.get_positions.return_value = {}
        engine.state.set_position_meta("SPY", 490.0, 500.0)
        engine.state.set_pending_fill("SPY", _pending(event="entry"))

        engine.positions()

        assert engine.state.position_meta("SPY") is not None
        engine.broker.closed_orders.assert_not_called()

    def test_stale_filled_order_is_not_booked(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_position_meta("SPY", 704.0, 741.0)
        engine.state.set_last_seen_position("SPY", "long", 100, 741.0)
        engine.broker.get_positions.return_value = {}
        engine.broker.closed_orders.return_value = [
            _filled_order(filled_at=datetime.now(timezone.utc) - timedelta(days=7))]

        engine.positions()

        assert trades_rows(tmp_path) == [TradeLog.header]  # no phantom trade
        assert len(_fills_rows(tmp_path)) == 1              # header only
        assert engine.state.position_meta("SPY") is None    # state still cleared

    def test_order_without_a_fill_timestamp_is_not_booked(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_position_meta("SPY", 704.0, 741.0)
        engine.state.set_last_seen_position("SPY", "long", 100, 741.0)
        engine.broker.get_positions.return_value = {}
        engine.broker.closed_orders.return_value = [_filled_order(filled_at=False)]

        engine.positions()

        assert trades_rows(tmp_path) == [TradeLog.header]

    def test_no_last_seen_snapshot_books_nothing(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_position_meta("SPY", 704.0, 741.0)
        engine.broker.get_positions.return_value = {}
        engine.broker.closed_orders.return_value = [_filled_order()]

        engine.positions()

        assert trades_rows(tmp_path) == [TradeLog.header]
        assert len(_fills_rows(tmp_path)) == 1

    def test_last_seen_snapshot_carries_a_timestamp(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_position_meta("SPY", 704.0, 741.0)
        engine.broker.get_positions.return_value = {
            "SPY": PositionState("SPY", "long", 100, 741.0, 0.0, 741.0)}

        engine.positions()

        seen = engine.state.last_seen_position("SPY")
        assert datetime.fromisoformat(seen["seen_at"]).tzinfo is not None


def _orphan_engine(tmp_path, monkeypatch, symbol="SPY", dry_run=False):
    """Engine where exactly one symbol has a stop resting at the broker and no
    position anywhere — the orphan the sweep exists for."""
    engine = _crypto_engine(tmp_path, monkeypatch, dry_run=dry_run)
    engine.broker.get_positions.return_value = {}
    engine.broker.has_open_stop.side_effect = lambda s: s == symbol
    engine.broker.has_open_order.side_effect = lambda s: s == symbol
    return engine


class TestOrphanStopCancellation:
    def test_orphan_stop_is_cancelled(self, tmp_path, monkeypatch, caplog):
        engine = _orphan_engine(tmp_path, monkeypatch)

        with caplog.at_level("WARNING"):
            engine.cancel_orphan_stops({})

        engine.broker.cancel_open_orders.assert_called_once_with("SPY")
        assert "orphan stop for SPY" in caplog.text

    def test_live_position_keeps_its_stop(self, tmp_path, monkeypatch):
        engine = _orphan_engine(tmp_path, monkeypatch)
        pos = PositionState("SPY", "long", 100, 741.0, 704.0, 741.0)

        engine.cancel_orphan_stops({"SPY": pos})

        engine.broker.cancel_open_orders.assert_not_called()

    def test_working_entry_keeps_its_oto_stop(self, tmp_path, monkeypatch):
        engine = _orphan_engine(tmp_path, monkeypatch)
        engine.state.set_pending_fill("SPY", _pending(event="entry"))

        engine.cancel_orphan_stops({})

        engine.broker.cancel_open_orders.assert_not_called()

    def test_resting_ladder_keeps_its_oto_stop(self, tmp_path, monkeypatch):
        engine = _orphan_engine(tmp_path, monkeypatch)
        engine.state.set_ladder_order("SPY", _ladder_marker())

        engine.cancel_orphan_stops({})

        engine.broker.cancel_open_orders.assert_not_called()

    def test_exit_in_flight_keeps_its_stop(self, tmp_path, monkeypatch):
        engine = _orphan_engine(tmp_path, monkeypatch)
        engine.state.set_pending_fill("SPY", _pending(event="exit", side="sell"))

        engine.cancel_orphan_stops({})

        engine.broker.cancel_open_orders.assert_not_called()

    def test_kept_position_meta_keeps_its_stop(self, tmp_path, monkeypatch):
        # The position is still tracked (e.g. a snapshot the broker has not
        # caught up with): its stop is doing its job, not orphaned.
        engine = _orphan_engine(tmp_path, monkeypatch)
        engine.state.set_position_meta("SPY", 704.0, 741.0)

        engine.cancel_orphan_stops({})

        engine.broker.cancel_open_orders.assert_not_called()

    def test_crypto_resting_maker_entry_is_never_touched(self, tmp_path, monkeypatch):
        # Crypto has no broker-side stop, so a resting BTC order is always the
        # maker entry — cancelling it would silently kill the entry.
        engine = _orphan_engine(tmp_path, monkeypatch, symbol="BTC/USD")
        engine.state.set_resting_order("BTC/USD", D0.isoformat())

        engine.cancel_orphan_stops({})

        engine.broker.cancel_open_orders.assert_not_called()

    def test_a_plain_open_order_without_a_stop_is_left_alone(self, tmp_path, monkeypatch):
        engine = _orphan_engine(tmp_path, monkeypatch)
        engine.broker.has_open_stop.side_effect = lambda s: False

        engine.cancel_orphan_stops({})

        engine.broker.cancel_open_orders.assert_not_called()

    def test_dry_run_cancels_nothing(self, tmp_path, monkeypatch):
        engine = _orphan_engine(tmp_path, monkeypatch, dry_run=True)

        engine.cancel_orphan_stops({})

        engine.broker.cancel_open_orders.assert_not_called()

    def test_one_failure_does_not_starve_the_other_symbols(self, tmp_path, monkeypatch):
        engine = _orphan_engine(tmp_path, monkeypatch)
        engine.broker.has_open_stop.side_effect = lambda s: True
        engine.broker.cancel_open_orders.side_effect = (
            lambda s: (_ for _ in ()).throw(RuntimeError("api down"))
            if s == "SPY" else None)

        engine.cancel_orphan_stops({})  # must not raise

        cancelled = [c.args[0] for c in engine.broker.cancel_open_orders.call_args_list]
        assert "SPY" in cancelled and len(cancelled) > 1

    def test_entry_is_no_longer_blocked_once_the_orphan_is_gone(
            self, tmp_path, monkeypatch):
        engine = _orphan_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        open_orders = {"SPY"}
        engine.broker.has_open_stop.side_effect = lambda s: s in open_orders
        engine.broker.has_open_order.side_effect = lambda s: s in open_orders
        engine.broker.cancel_open_orders.side_effect = open_orders.discard
        assert engine._entry_order_outstanding(cfg) is True

        engine.cancel_orphan_stops({})

        assert engine._entry_order_outstanding(cfg) is False

    def test_vanished_position_has_its_leftover_stop_cancelled(
            self, tmp_path, monkeypatch):
        # Full path: the exit is booked by the vanish sweep, then the stop that
        # outlived the position is cancelled in the same cycle.
        engine = _orphan_engine(tmp_path, monkeypatch)
        engine.state.set_position_meta("SPY", 704.0, 741.0)
        engine.state.set_last_seen_position("SPY", "long", 100, 741.0)
        engine.broker.closed_orders.return_value = [_filled_order()]

        engine.run_protective_sweeps()

        assert len(trades_rows(tmp_path)) == 2  # the exit still reaches trades.csv
        engine.broker.cancel_open_orders.assert_called_once_with("SPY")

    def test_orphan_is_swept_under_halt(self, tmp_path, monkeypatch):
        engine = _halted_engine(tmp_path, monkeypatch)
        engine.broker.get_positions.return_value = {}
        engine.broker.has_open_stop.side_effect = lambda s: s == "SPY"

        engine.run_cycle()

        engine.broker.cancel_open_orders.assert_called_once_with("SPY")


class TestPartialCloseBooking:
    def _exit_marker(self):
        payload = _pending(event="exit", side="sell", order_id="close-1")
        payload.update({"position_side": "long", "entry_price": 741.0, "qty": 100})
        return payload

    def test_cancelled_close_with_a_partial_fill_is_booked(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_pending_fill("SPY", self._exit_marker())
        engine.broker.get_order_fill.return_value = (60.0, 750.0, "canceled")

        engine.reconcile_pending_fills()

        trades = trades_rows(tmp_path)[1:]
        assert len(trades) == 1
        assert float(trades[0][6]) == pytest.approx(60.0)
        assert float(trades[0][5]) == pytest.approx(540.0)  # 60 * (750 - 741)
        fills = _fills_rows(tmp_path)[1:]
        assert len(fills) == 1 and float(fills[0][8]) == pytest.approx(60.0)
        assert engine.state.exit_failure("SPY") is not None
        assert engine.state.pending_fill("SPY") is None

    def test_cancelled_close_with_no_fill_books_nothing(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_pending_fill("SPY", self._exit_marker())
        engine.broker.get_order_fill.return_value = (0.0, None, "canceled")

        engine.reconcile_pending_fills()

        assert trades_rows(tmp_path) == [TradeLog.header]
        assert len(_fills_rows(tmp_path)) == 1

    def test_retried_close_books_each_filled_slice_exactly_once(
            self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.has_open_stop.return_value = False
        cfg, _ = engine.strategies["SPY"]
        engine.state.set_position_meta("SPY", 704.0, 741.0)
        engine.state.set_last_seen_position("SPY", "long", 100, 741.0)
        pos = PositionState("SPY", "long", 100, 741.0, 704.0, 741.0)

        engine.broker.close_position.return_value = MagicMock(id="close-1")
        assert engine.exit_position(cfg, pos, 750.0, "snap-back") is True

        # 60 of 100 filled, then the close is retried: close_position cancels the
        # partially-filled order, so its 60 shares must be booked before the new
        # marker overwrites the old one.
        engine.broker.get_order_fill.return_value = (60.0, 750.0, "partially_filled")
        engine.broker.close_position.return_value = MagicMock(id="close-2")
        remaining = PositionState("SPY", "long", 40, 741.0, 704.0, 741.0)
        assert engine.exit_position(cfg, remaining, 749.0, "snap-back") is True
        assert engine.state.pending_fill("SPY")["order_id"] == "close-2"

        engine.broker.get_order_fill.return_value = (40.0, 748.0, "filled")
        engine.reconcile_pending_fills()

        trades = trades_rows(tmp_path)[1:]
        assert [float(r[6]) for r in trades] == [60.0, 40.0]  # 100 total, no phantom
        assert [float(r[4]) for r in trades] == [750.0, 748.0]
        assert len(_fills_rows(tmp_path)) == 3  # header + one row per slice

    def test_full_fill_still_books_exactly_one_row(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_pending_fill("SPY", self._exit_marker())
        engine.broker.get_order_fill.return_value = (100.0, 750.0, "filled")

        engine.reconcile_pending_fills()

        assert len(trades_rows(tmp_path)) == 2
        assert len(_fills_rows(tmp_path)) == 2


class TestHaltGateBeforeEveryOrder:
    """A cycle takes minutes and submits for six instruments. Checking the switch
    once at the top of the cycle lets every order after the trip through."""

    def test_entry_is_refused_once_the_switch_trips_mid_cycle(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["BTC/USD"]

        engine.enter_position(cfg, "long", 50_000.0, 1_000.0, {}, D0)
        assert engine.broker.limit_order.call_count == 1

        trip_halt("tripped by hand mid-cycle", str(tmp_path / "HALT"))
        engine.enter_position(cfg, "long", 50_000.0, 1_000.0, {}, D1)

        assert engine.broker.limit_order.call_count == 1  # the second never went

    def test_refused_entry_persists_no_stop_meta(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["BTC/USD"]
        trip_halt("halted", str(tmp_path / "HALT"))

        engine.enter_position(cfg, "long", 50_000.0, 1_000.0, {}, D0)

        assert engine.state.position_meta("BTC/USD") is None
        assert engine.state.resting_order("BTC/USD") is None

    def test_an_unreadable_switch_refuses_the_entry(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["BTC/USD"]
        monkeypatch.setattr("bot.main.halt_tripped", lambda path=None: True)

        engine.enter_position(cfg, "long", 50_000.0, 1_000.0, {}, D0)

        engine.broker.limit_order.assert_not_called()

    def test_exits_still_run_under_a_halt(self, tmp_path, monkeypatch):
        # The breaker trips the switch and then flattens. Gating exits on the
        # switch would deadlock it against its own halt, and would strand every
        # position the strategy wants out of.
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["BTC/USD"]
        pos = PositionState("BTC/USD", "long", 1.0, 50_000.0, 0.0, 49_000.0)
        trip_halt("halted", str(tmp_path / "HALT"))

        assert engine.exit_position(cfg, pos, 51_000.0, "signal") is True
        engine.broker.close_position.assert_called_once()


class TestCancelBeforeFlatten:
    """Cancel every resting order first, then close. A resting entry left live
    while we flatten can fill and re-open the exposure the breaker just closed."""

    def _engine(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.get_positions.return_value = {
            "SPY": PositionState("SPY", "long", 100, 500.0, 0.0, 490.0),
        }
        engine.state.set_position_meta("SPY", 490.0, 505.0)
        engine.state.data["peak_equity"] = 100_000.0
        return engine

    def test_resting_order_on_a_flat_symbol_is_cancelled(self, tmp_path, monkeypatch):
        # BTC/USD holds no position, so the close loop never visits it — but its
        # resting maker entry is still live and can fill after the halt.
        engine = self._engine(tmp_path, monkeypatch)
        engine.state.set_resting_order("BTC/USD", D0.isoformat())

        engine.check_circuit_breaker(85_000.0)

        cancelled = {c.args[0] for c in engine.broker.cancel_open_orders.call_args_list}
        assert "BTC/USD" in cancelled

    def test_every_configured_symbol_is_cancelled(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        engine.check_circuit_breaker(85_000.0)
        cancelled = {c.args[0] for c in engine.broker.cancel_open_orders.call_args_list}
        assert cancelled == set(engine.strategies)

    def test_cancels_precede_the_first_close(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        order = []
        engine.broker.cancel_open_orders.side_effect = lambda s: order.append(("cancel", s))
        engine.broker.close_position.side_effect = lambda s: order.append(("close", s))

        engine.check_circuit_breaker(85_000.0)

        first_close = next(i for i, (kind, _) in enumerate(order) if kind == "close")
        assert all(kind == "cancel" for kind, _ in order[:first_close])
        assert len([k for k, _ in order if k == "cancel"]) == len(engine.strategies)

    def test_a_failed_cancel_does_not_stop_the_flatten(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        engine.broker.cancel_open_orders.side_effect = RuntimeError("api down")

        assert engine.check_circuit_breaker(85_000.0) is True

        engine.broker.close_position.assert_called_once_with("SPY")

    def test_a_failed_cancel_is_surfaced_as_an_alert(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        engine.broker.cancel_open_orders.side_effect = RuntimeError("api down")
        engine.check_circuit_breaker(85_000.0)
        assert engine.state.cancel_failure("BTC/USD") is not None

    def test_a_cancel_failure_on_a_flat_symbol_survives_the_sweep(
            self, tmp_path, monkeypatch):
        # An exit failure self-clears once the symbol is flat, which is right for
        # an exit and wrong for a cancel: a flat symbol is exactly where an
        # uncancelled resting entry is dangerous. The two cannot share a channel.
        engine = self._engine(tmp_path, monkeypatch)
        engine.broker.cancel_open_orders.side_effect = RuntimeError("api down")
        engine.check_circuit_breaker(85_000.0)

        engine.positions()

        assert engine.state.cancel_failure("BTC/USD") is not None

    def test_a_clean_cancel_records_nothing(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        engine.check_circuit_breaker(85_000.0)
        assert engine.state.cancel_failure("BTC/USD") is None


class TestRejectedOrderIsNotAcceptance:
    """A broker that returns an Order whose status is 'rejected' has refused the
    request. Reading only the absence of an exception books it as done."""

    def _pos(self):
        return PositionState("SPY", "long", 100, 500.0, 0.0, 490.0)

    def _equity_cfg(self, engine):
        return engine.strategies["SPY"][0]

    def test_rejected_close_returns_false(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.close_position.return_value = MagicMock(id="c-1", status="rejected")
        monkeypatch.setattr(engine, "_decision_price_trusted", lambda *a, **k: True)

        assert engine.exit_position(self._equity_cfg(engine), self._pos(),
                                    505.0, "signal") is False

    def test_rejected_close_keeps_the_position_meta(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_position_meta("SPY", 490.0, 505.0)
        engine.broker.close_position.return_value = MagicMock(id="c-1", status="rejected")
        monkeypatch.setattr(engine, "_decision_price_trusted", lambda *a, **k: True)

        engine.exit_position(self._equity_cfg(engine), self._pos(), 505.0, "signal")

        assert engine.state.position_meta("SPY") is not None

    def test_rejected_close_records_an_exit_failure(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.close_position.return_value = MagicMock(id="c-1", status="rejected")
        monkeypatch.setattr(engine, "_decision_price_trusted", lambda *a, **k: True)

        engine.exit_position(self._equity_cfg(engine), self._pos(), 505.0, "signal")

        assert engine.state.exit_failure("SPY") is not None

    def test_rejected_close_restores_the_protective_stop(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.close_position.return_value = MagicMock(id="c-1", status="rejected")
        monkeypatch.setattr(engine, "_decision_price_trusted", lambda *a, **k: True)

        engine.exit_position(self._equity_cfg(engine), self._pos(), 505.0, "signal")

        engine.broker.stop_order.assert_called_once()

    def test_a_filled_close_is_still_an_acceptance(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.close_position.return_value = MagicMock(id="c-1", status="filled")
        monkeypatch.setattr(engine, "_decision_price_trusted", lambda *a, **k: True)

        assert engine.exit_position(self._equity_cfg(engine), self._pos(),
                                    505.0, "signal") is True

    def test_an_order_with_no_status_attribute_is_an_acceptance(self, tmp_path, monkeypatch):
        # Not every broker response carries a status; absence is not a rejection.
        engine = _crypto_engine(tmp_path, monkeypatch)
        order = MagicMock(id="c-1")
        del order.status
        engine.broker.close_position.return_value = order
        monkeypatch.setattr(engine, "_decision_price_trusted", lambda *a, **k: True)

        assert engine.exit_position(self._equity_cfg(engine), self._pos(),
                                    505.0, "signal") is True

    def test_a_none_response_is_a_refusal(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.close_position.return_value = None
        monkeypatch.setattr(engine, "_decision_price_trusted", lambda *a, **k: True)

        assert engine.exit_position(self._equity_cfg(engine), self._pos(),
                                    505.0, "signal") is False

    def test_a_rejected_stop_replacement_is_not_reported_as_restored(
            self, tmp_path, monkeypatch, caplog):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.has_open_stop.return_value = False
        engine.broker.stop_order.return_value = MagicMock(id="s-1", status="rejected")

        with caplog.at_level(logging.ERROR):
            engine._restore_protective_stop(self._equity_cfg(engine), self._pos(), 5.0)

        assert "NAKED" in caplog.text

    def test_an_accepted_stop_replacement_is_reported_as_restored(
            self, tmp_path, monkeypatch, caplog):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.has_open_stop.return_value = False
        engine.broker.stop_order.return_value = MagicMock(id="s-1", status="new")

        with caplog.at_level(logging.WARNING):
            engine._restore_protective_stop(self._equity_cfg(engine), self._pos(), 5.0)

        assert "NAKED" not in caplog.text
        assert "restored" in caplog.text


class TestRejectedStopReplacementInTheSweep:
    def _pos(self):
        return PositionState("SPY", "long", 100, 500.0, 0.0, 490.0)

    def test_a_refused_replacement_is_logged_as_still_missing(
            self, tmp_path, monkeypatch, caplog):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.has_open_stop.return_value = False
        engine.broker.stop_order.return_value = MagicMock(id="s-1", status="rejected")
        engine.state.set_position_meta("SPY", 490.0, 505.0)

        with caplog.at_level(logging.ERROR):
            engine.reconcile_equity_stops({"SPY": self._pos()})

        assert "REFUSED" in caplog.text

    def test_an_accepted_replacement_logs_no_error(self, tmp_path, monkeypatch, caplog):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.has_open_stop.return_value = False
        engine.broker.stop_order.return_value = MagicMock(id="s-1", status="new")
        engine.state.set_position_meta("SPY", 490.0, 505.0)

        with caplog.at_level(logging.ERROR):
            engine.reconcile_equity_stops({"SPY": self._pos()})

        assert caplog.text == ""


class TestRefusedEntryAtSubmitSite:
    """The stop meta and the order marker are both written BEFORE submitting, on
    the assumption the order will rest. A refusal returns an Order rather than
    raising, so nothing rolled them back and the markers outlived the order."""

    def _engine(self, tmp_path, monkeypatch, status="rejected"):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.limit_order.return_value = MagicMock(id="order-1", status=status)
        engine.risk.approve_entry = MagicMock(
            return_value=MagicMock(qty=100, stop_price=490.0))
        return engine

    # --- plain equity limit ------------------------------------------------
    def test_equity_refusal_leaves_no_stop_meta(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        engine.enter_position(cfg, "long", 500.0, 5.0, {}, D0, quote=SPY_QUOTE)
        assert engine.state.position_meta("SPY") is None

    def test_equity_refusal_leaves_no_pending_fill(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        engine.enter_position(cfg, "long", 500.0, 5.0, {}, D0, quote=SPY_QUOTE)
        assert engine.state.pending_fill("SPY") is None

    def test_equity_refusal_is_logged_as_refused(self, tmp_path, monkeypatch, caplog):
        engine = self._engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        with caplog.at_level(logging.ERROR):
            engine.enter_position(cfg, "long", 500.0, 5.0, {}, D0, quote=SPY_QUOTE)
        assert "REFUSED" in caplog.text

    def test_an_accepted_equity_entry_still_records_its_state(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch, status="new")
        cfg, _ = engine.strategies["SPY"]
        engine.enter_position(cfg, "long", 500.0, 5.0, {}, D0, quote=SPY_QUOTE)
        assert engine.state.position_meta("SPY") is not None
        assert engine.state.pending_fill("SPY") is not None

    # --- ladder leg --------------------------------------------------------
    def test_ladder_refusal_leaves_no_marker(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bot.main.EXECUTION",
                            ExecutionConfig(ladder_enabled=True, ladder_passive_bps=2.0))
        engine = self._engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]

        engine.enter_position(cfg, "long", 500.0, 5.0, {}, D0, quote=SPY_QUOTE)

        # A marker carrying a dead order id sends the next escalation sweep
        # chasing an order that never existed.
        assert engine.state.ladder_order("SPY") is None
        assert engine.state.position_meta("SPY") is None

    def test_ladder_refusal_does_not_claim_the_leg_was_placed(
            self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr("bot.main.EXECUTION",
                            ExecutionConfig(ladder_enabled=True, ladder_passive_bps=2.0))
        engine = self._engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        with caplog.at_level(logging.INFO):
            engine.enter_position(cfg, "long", 500.0, 5.0, {}, D0, quote=SPY_QUOTE)
        assert "passive entry placed" not in caplog.text

    # --- crypto maker ------------------------------------------------------
    def test_crypto_refusal_leaves_no_resting_marker(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["BTC/USD"]

        engine.enter_position(cfg, "long", 50_000.0, 1_000.0, {}, D0)

        # A stale resting marker suppresses the next bar's entry for a symbol
        # that has no order resting at all.
        assert engine.state.resting_order("BTC/USD") is None
        assert engine.state.position_meta("BTC/USD") is None

    def test_crypto_refusal_does_not_claim_the_order_rests(
            self, tmp_path, monkeypatch, caplog):
        engine = self._engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["BTC/USD"]
        with caplog.at_level(logging.INFO):
            engine.enter_position(cfg, "long", 50_000.0, 1_000.0, {}, D0)
        assert "maker entry placed" not in caplog.text

    def test_a_none_response_is_also_a_refusal(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        engine.broker.limit_order.return_value = None
        cfg, _ = engine.strategies["BTC/USD"]
        engine.enter_position(cfg, "long", 50_000.0, 1_000.0, {}, D0)
        assert engine.state.resting_order("BTC/USD") is None


class TestRefusedEntryAlert:
    """A refused entry risks nothing on its own — but a standing refusal (no
    buying power, a wash-trade block, a PDT freeze) is the bot silently ceasing
    to trade, which is the failure mode that went a month unnoticed."""

    def _engine(self, tmp_path, monkeypatch, status="rejected"):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.limit_order.return_value = MagicMock(id="order-1", status=status)
        engine.risk.approve_entry = MagicMock(
            return_value=MagicMock(qty=100, stop_price=490.0))
        return engine

    def test_refusal_is_recorded(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        engine.enter_position(cfg, "long", 500.0, 5.0, {}, D0, quote=SPY_QUOTE)
        assert engine.state.entry_refusal("SPY") is not None

    def test_repeat_refusals_accumulate_against_one_since(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        engine.enter_position(cfg, "long", 500.0, 5.0, {}, D0, quote=SPY_QUOTE)
        engine.enter_position(cfg, "long", 500.0, 5.0, {}, D1, quote=SPY_QUOTE)
        assert engine.state.entry_refusal("SPY")["count"] == 2

    def test_an_accepted_entry_clears_the_refusal(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        engine.enter_position(cfg, "long", 500.0, 5.0, {}, D0, quote=SPY_QUOTE)

        engine.broker.limit_order.return_value = MagicMock(id="order-2", status="new")
        engine.enter_position(cfg, "long", 500.0, 5.0, {}, D1, quote=SPY_QUOTE)

        assert engine.state.entry_refusal("SPY") is None

    def test_a_clean_entry_records_nothing(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch, status="new")
        cfg, _ = engine.strategies["SPY"]
        engine.enter_position(cfg, "long", 500.0, 5.0, {}, D0, quote=SPY_QUOTE)
        assert engine.state.entry_refusal("SPY") is None


class TestEntryRiskBudget:
    """Every entry reports what it actually risks against the budget its sizing
    intended. Observation only: the order goes out either way."""

    def test_a_sized_entry_reports_realized_risk_against_its_budget(
            self, tmp_path, monkeypatch, caplog):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        with caplog.at_level(logging.INFO):
            engine.enter_position(cfg, "long", 500.0, 5.0, {}, D0, quote=SPY_QUOTE)
        # 66 shares x 3 ATR ($15) of $100k equity, against a neutral 1% budget
        assert "SPY: risk 0.990% of equity vs budget 1.000%" in caplog.text
        assert "within budget" in caplog.text

    def test_vol_scaled_entry_is_measured_against_the_scaled_budget(
            self, tmp_path, monkeypatch, caplog):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        with caplog.at_level(logging.INFO):
            engine.enter_position(cfg, "long", 500.0, 5.0, {}, D0,
                                  realized_vol=0.05, quote=SPY_QUOTE)
        # Scalar clipped at 1.5: 100 shares risk 1.5% of equity — half again
        # the flat fraction, and exactly what the overlay budgets for.
        assert "SPY: risk 1.500% of equity vs budget 1.500%" in caplog.text
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_an_oversized_entry_is_warned_about_but_still_submitted(
            self, tmp_path, monkeypatch, caplog):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        sized = engine.risk.approve_entry("SPY", "long", 100_000.0, 500.0, 5.0,
                                          cfg.asset_class, {}, 3.0)
        engine.risk.approve_entry = MagicMock(
            return_value=MagicMock(qty=sized.qty * 2, stop_price=sized.stop_price))
        with caplog.at_level(logging.INFO):
            engine.enter_position(cfg, "long", 500.0, 5.0, {}, D0, quote=SPY_QUOTE)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("over budget" in r.getMessage() for r in warnings)
        engine.broker.limit_order.assert_called_once()
        assert engine.broker.limit_order.call_args.args[2] == sized.qty * 2


class TestEntryStopLegExpiry:
    """Alpaca refuses to change an OTO leg's time-in-force, so the stop attached
    at entry dies at that session's close and the sweep re-places it GTC. That
    expiry is the design, and must not read like a stop that was lost."""

    def _pos(self):
        return PositionState("SPY", "long", 100, 741.0, 704.0, 741.0)

    def _entered(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        cfg, _ = engine.strategies["SPY"]
        engine.enter_position(cfg, "long", 500.0, 5.0, {}, D0, quote=SPY_QUOTE)
        engine.broker.reset_mock()
        engine.broker.has_open_stop.return_value = False
        return engine

    def test_a_resting_stop_is_never_rewritten(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.has_open_stop.return_value = True

        engine.reconcile_equity_stops({"SPY": self._pos()})

        assert [c[0] for c in engine.broker.method_calls] == ["has_open_stop"]

    def test_the_entry_legs_expiry_is_re_placed_without_a_warning(
            self, tmp_path, monkeypatch, caplog):
        engine = self._entered(tmp_path, monkeypatch)
        engine.broker.market_is_open.return_value = False

        with caplog.at_level(logging.INFO):
            engine.reconcile_equity_stops({"SPY": self._pos()})

        engine.broker.stop_order.assert_called_once()
        assert "entry stop expired at the close" in caplog.text
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_a_stop_lost_during_the_session_still_warns(
            self, tmp_path, monkeypatch, caplog):
        engine = self._entered(tmp_path, monkeypatch)
        engine.broker.market_is_open.return_value = True

        with caplog.at_level(logging.WARNING):
            engine.reconcile_equity_stops({"SPY": self._pos()})

        engine.broker.stop_order.assert_called_once()
        assert "protective stop missing" in caplog.text

    def test_a_missing_stop_with_no_entry_this_session_still_warns(
            self, tmp_path, monkeypatch, caplog):
        # A restart, or a position adopted from the broker: nothing says this
        # stop was an entry leg, so its absence is treated as a loss.
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.broker.market_is_open.return_value = False

        with caplog.at_level(logging.WARNING):
            engine.reconcile_equity_stops({"SPY": self._pos()})

        engine.broker.stop_order.assert_called_once()
        assert "protective stop missing" in caplog.text

    def test_a_market_status_outage_never_delays_the_re_placement(
            self, tmp_path, monkeypatch):
        engine = self._entered(tmp_path, monkeypatch)
        engine.broker.market_is_open.side_effect = RuntimeError("clock down")

        engine.reconcile_equity_stops({"SPY": self._pos()})

        engine.broker.stop_order.assert_called_once()

    def test_the_expected_expiry_is_excused_only_once(
            self, tmp_path, monkeypatch, caplog):
        # The GTC replacement does not expire; if it too goes missing, that is
        # a real loss.
        engine = self._entered(tmp_path, monkeypatch)
        engine.broker.market_is_open.return_value = False
        engine.reconcile_equity_stops({"SPY": self._pos()})

        with caplog.at_level(logging.WARNING):
            engine.reconcile_equity_stops({"SPY": self._pos()})

        assert "protective stop missing" in caplog.text


# ---------------------------------------------------------------------------
# Startup and per-decision bar-window guard
# ---------------------------------------------------------------------------

def _window_engine(tmp_path, monkeypatch, symbol, min_bars, available):
    """Engine narrowed to one instrument whose fetched window is `available`
    bars long, so the warm-up guard can be observed in isolation."""
    engine = _crypto_engine(tmp_path, monkeypatch)
    cfg, _ = engine.strategies[symbol]
    strat = MagicMock()
    strat.min_bars = min_bars
    strat.stop_atr_mult = 3.0
    strat.prepare.side_effect = lambda bars: bars.assign(atr=1.0)
    strat.signal.return_value = Signal(Action.HOLD, "hold")
    engine.strategies = {symbol: (cfg, strat)}
    idx = pd.date_range("2020-01-01", periods=available, freq="D", tz="UTC")
    engine.data.get_recent_bars.return_value = pd.DataFrame(
        {"close": [100.0] * available}, index=idx)
    return engine


class TestStartupWindowCheck:
    """A backtest sees the whole history; live sees a fetched window. A window
    that barely clears min_bars is one bad fetch from the per-decision line, so
    the shortfall is named at startup."""

    def test_window_inside_the_cushion_is_logged_as_an_error(
            self, tmp_path, monkeypatch, caplog):
        engine = _window_engine(tmp_path, monkeypatch, "SPY", min_bars=100,
                                available=102)
        with caplog.at_level(logging.ERROR):
            engine.check_startup_windows()
        assert "SPY" in caplog.text
        assert "102" in caplog.text and "105" in caplog.text  # have vs wanted

    def test_window_at_the_cushion_logs_no_error(self, tmp_path, monkeypatch, caplog):
        engine = _window_engine(tmp_path, monkeypatch, "SPY", min_bars=100,
                                available=105)
        with caplog.at_level(logging.ERROR):
            engine.check_startup_windows()
        assert caplog.text == ""

    def test_the_engines_own_fetch_satisfies_the_cushion(self, tmp_path, monkeypatch,
                                                         caplog):
        """The cushion has to be reachable by the window the engine asks for, or
        every start reports a shortfall nobody can act on and the ERROR stops
        being read."""
        engine = _crypto_engine(tmp_path, monkeypatch)
        delivered = {symbol: strategy.min_bars + bot_main.BAR_WINDOW_EXTRA
                     for symbol, (_, strategy) in engine.strategies.items()}
        engine.data.get_recent_bars.side_effect = lambda symbol, *a, **k: pd.DataFrame(
            {"close": [100.0] * delivered[symbol]},
            index=pd.date_range("2015-01-01", periods=delivered[symbol], freq="D",
                                tz="UTC"))

        with caplog.at_level(logging.ERROR):
            engine.check_startup_windows()

        assert caplog.text == ""

    def test_a_fetch_failure_never_stops_the_engine(self, tmp_path, monkeypatch):
        engine = _window_engine(tmp_path, monkeypatch, "SPY", min_bars=100,
                                available=200)
        engine.data.get_recent_bars.side_effect = RuntimeError("data down")
        engine.check_startup_windows()  # must not raise

    def test_one_symbols_failure_does_not_starve_another(self, tmp_path, monkeypatch,
                                                         caplog):
        engine = _window_engine(tmp_path, monkeypatch, "SPY", min_bars=100,
                                available=102)
        cfg, strat = engine.strategies["SPY"]
        engine.strategies["USO"] = (_cfg("USO"), strat)
        engine.data.get_recent_bars.side_effect = (
            lambda symbol, *a, **k: (_ for _ in ()).throw(RuntimeError("down"))
            if symbol == "SPY" else pd.DataFrame(
                {"close": [100.0] * 102},
                index=pd.date_range("2020-01-01", periods=102, freq="D", tz="UTC")))
        with caplog.at_level(logging.ERROR):
            engine.check_startup_windows()
        assert "USO" in caplog.text


class TestShortDecisionWindow:
    def test_short_window_records_the_alert(self, tmp_path, monkeypatch, caplog):
        engine = _window_engine(tmp_path, monkeypatch, "SPY", min_bars=10,
                                available=5)
        with caplog.at_level(logging.ERROR):
            engine.process_instrument("SPY")
        assert "SPY" in caplog.text
        alert = engine.state.short_window("SPY")
        assert alert["count"] == 1
        assert "5" in alert["detail"] and "10" in alert["detail"]

    def test_short_window_suppresses_the_entry(self, tmp_path, monkeypatch):
        engine = _long_signal_engine(tmp_path, monkeypatch, "SPY", Q102)
        cfg, strat = engine.strategies["SPY"]
        strat.min_bars = 10  # the 5 fetched bars are not enough
        engine.risk.approve_entry = MagicMock(
            return_value=MagicMock(qty=100, stop_price=98.0))

        engine.process_instrument("SPY")

        engine.broker.limit_order.assert_not_called()

    def test_repeated_short_windows_count_up_keeping_first_seen(self, tmp_path,
                                                                monkeypatch):
        engine = _window_engine(tmp_path, monkeypatch, "SPY", min_bars=10,
                                available=5)
        engine.process_instrument("SPY")
        since = engine.state.short_window("SPY")["since"]
        engine.process_instrument("SPY")
        assert engine.state.short_window("SPY")["count"] == 2
        assert engine.state.short_window("SPY")["since"] == since

    def test_a_full_window_clears_the_alert(self, tmp_path, monkeypatch):
        engine = _window_engine(tmp_path, monkeypatch, "SPY", min_bars=10,
                                available=5)
        engine.process_instrument("SPY")

        idx = pd.date_range("2020-01-01", periods=20, freq="D", tz="UTC")
        engine.data.get_recent_bars.return_value = pd.DataFrame(
            {"close": [100.0] * 20}, index=idx)
        engine.process_instrument("SPY")

        assert engine.state.short_window("SPY") is None

    def test_protective_sweeps_still_run(self, tmp_path, monkeypatch):
        """Loud but non-fatal: no decision is taken on a short window, and
        broker-side stops and the protective sweeps are unaffected."""
        engine = _window_engine(tmp_path, monkeypatch, "SPY", min_bars=10,
                                available=5)
        engine.broker.market_is_open.return_value = True
        engine.broker.current_session.return_value = None
        engine.maybe_send_briefings = lambda: None
        engine.reconcile_equity_stops = MagicMock()

        engine.run_cycle()

        engine.reconcile_equity_stops.assert_called_once()


class TestShortWindowState:
    def test_record_then_clear(self, tmp_path):
        store = StateStore(str(tmp_path / "state.json"))
        store.record_short_window("SPY", "5 bars, 10 needed")
        assert store.short_window("SPY")["count"] == 1
        store.clear_short_window("SPY")
        assert store.short_window("SPY") is None

    def test_survives_a_reload(self, tmp_path):
        path = str(tmp_path / "state.json")
        StateStore(path).record_short_window("SPY", "5 bars, 10 needed")
        assert StateStore(path).short_window("SPY")["detail"] == "5 bars, 10 needed"


def _command(verb, args=""):
    from bot.commands import Command
    return Command(verb=verb, args=args, subject=f"CMD {verb} {args}".strip(),
                   sender="ops@example.com", message_id="<m-1@x>")


def _command_engine(tmp_path, monkeypatch, pending, replies=None):
    """Engine whose command channel yields `pending` and records its replies."""
    engine = _crypto_engine(tmp_path, monkeypatch)
    engine.maybe_send_briefings = lambda: None
    monkeypatch.setattr("bot.main.commands.poll", lambda state: list(pending))
    monkeypatch.setattr(
        "bot.main.commands.send_reply",
        lambda request, sender, verdict: (replies if replies is not None else []
                                          ).append(verdict))
    return engine


class TestEmailCommandExecution:
    def test_halt_trips_the_sentinel_with_an_audit_prefix(self, tmp_path, monkeypatch):
        replies = []
        engine = _command_engine(tmp_path, monkeypatch,
                                 [_command("halt", "manual stop")], replies)

        engine.process_commands()

        from bot.halt import read_halt
        reason = read_halt(str(tmp_path / "HALT"))["reason"]
        assert "email command" in reason and "manual stop" in reason
        assert "HALT" in replies[0]

    def test_clear_halt_removes_the_sentinel(self, tmp_path, monkeypatch):
        replies = []
        engine = _command_engine(tmp_path, monkeypatch, [_command("clear-halt")],
                                 replies)
        trip_halt("drawdown", str(tmp_path / "HALT"))

        engine.process_commands()

        assert not (tmp_path / "HALT").exists()
        assert "cleared" in replies[0]

    def test_clear_halt_on_a_clear_switch_says_so(self, tmp_path, monkeypatch):
        replies = []
        engine = _command_engine(tmp_path, monkeypatch, [_command("clear-halt")],
                                 replies)

        engine.process_commands()

        assert "no HALT" in replies[0]

    def test_force_exit_routes_through_the_exit_path(self, tmp_path, monkeypatch):
        replies = []
        engine = _command_engine(tmp_path, monkeypatch,
                                 [_command("force-exit", "SPY")], replies)
        engine.state.set_position_meta("SPY", 490.0, 505.0)
        engine.broker.get_positions.return_value = {
            "SPY": PositionState("SPY", "long", 100, 500.0, 490.0, 505.0)}
        engine.data.latest_price.return_value = 505.0

        engine.process_commands()

        engine.broker.close_position.assert_called_once_with("SPY")
        assert engine.state.pending_fill("SPY")["event"] == "exit"
        assert "force-exit" in replies[0]

    def test_force_exit_works_while_halted(self, tmp_path, monkeypatch):
        engine = _command_engine(tmp_path, monkeypatch,
                                 [_command("force-exit", "SPY")])
        trip_halt("drawdown", str(tmp_path / "HALT"))
        engine.broker.get_positions.return_value = {
            "SPY": PositionState("SPY", "long", 100, 500.0, 490.0, 505.0)}
        engine.data.latest_price.return_value = 505.0

        engine.process_commands()

        engine.broker.close_position.assert_called_once_with("SPY")

    def test_force_exit_refuses_a_symbol_outside_the_lineup(self, tmp_path,
                                                            monkeypatch):
        replies = []
        engine = _command_engine(tmp_path, monkeypatch,
                                 [_command("force-exit", "TSLA")], replies)

        engine.process_commands()

        engine.broker.close_position.assert_not_called()
        assert "REFUSED" in replies[0] and "lineup" in replies[0]

    def test_force_exit_refuses_a_flat_symbol(self, tmp_path, monkeypatch):
        replies = []
        engine = _command_engine(tmp_path, monkeypatch,
                                 [_command("force-exit", "SPY")], replies)
        engine.broker.get_positions.return_value = {}

        engine.process_commands()

        engine.broker.close_position.assert_not_called()
        assert "REFUSED" in replies[0] and "no open position" in replies[0]

    def test_failed_force_exit_reports_the_position_is_still_open(
            self, tmp_path, monkeypatch):
        replies = []
        engine = _command_engine(tmp_path, monkeypatch,
                                 [_command("force-exit", "SPY")], replies)
        engine.broker.get_positions.return_value = {
            "SPY": PositionState("SPY", "long", 100, 500.0, 490.0, 505.0)}
        engine.data.latest_price.return_value = 505.0
        engine.broker.close_position.side_effect = RuntimeError("403 held_for_orders")

        engine.process_commands()

        assert "FAILED" in replies[0]

    def test_status_reports_equity_positions_and_halt_and_submits_nothing(
            self, tmp_path, monkeypatch):
        replies = []
        engine = _command_engine(tmp_path, monkeypatch, [_command("status")], replies)
        engine.state.set_position_meta("USO", 95.0, 120.0)
        engine.broker.get_positions.return_value = {
            "USO": PositionState("USO", "long", 10, 100.0, 95.0, 120.0)}
        trip_halt("drawdown halt", str(tmp_path / "HALT"))

        engine.process_commands()

        body = replies[0]
        assert "100,000" in body
        assert "USO" in body
        assert "HALT" in body
        engine.broker.close_position.assert_not_called()
        assert (tmp_path / "HALT").exists()

    def test_status_shows_the_effective_trailing_stop(self, tmp_path, monkeypatch):
        replies = []
        engine = _command_engine(tmp_path, monkeypatch, [_command("status")], replies)
        engine.state.set_position_meta("USO", 95.0, 120.0)
        engine.broker.get_positions.return_value = {
            "USO": PositionState("USO", "long", 10, 100.0, 95.0, 120.0)}
        monkeypatch.setattr(Engine, "_latest_atr", lambda self, symbol: 5.0)

        engine.process_commands()

        assert "105.00" in replies[0]  # max(95, 120 - 3 x 5), not the entry stop

    def test_a_verb_with_no_handler_is_refused_not_treated_as_status(
            self, tmp_path, monkeypatch):
        """A verb added to commands.VERBS but not wired here must refuse. Falling
        through to the status report would answer a command nobody implemented."""
        replies = []
        engine = _command_engine(tmp_path, monkeypatch, [_command("liquidate-all")],
                                 replies)

        engine.process_commands()

        assert "REFUSED" in replies[0] and "liquidate-all" in replies[0]
        engine.broker.account_equity.assert_not_called()

    def test_status_carries_the_briefing_alerts(self, tmp_path, monkeypatch):
        replies = []
        engine = _command_engine(tmp_path, monkeypatch, [_command("status")], replies)
        engine.broker.get_positions.return_value = {
            "SPY": PositionState("SPY", "long", 100, 500.0, 490.0, 505.0)}
        engine.state.record_exit_failure("SPY", "403 held_for_orders")

        engine.process_commands()

        assert "exit FAILED" in replies[0]
        assert "403 held_for_orders" in replies[0]

    def test_every_command_is_answered(self, tmp_path, monkeypatch):
        replies = []
        engine = _command_engine(
            tmp_path, monkeypatch,
            [_command("status"), _command("force-exit", "TSLA")], replies)

        engine.process_commands()

        assert len(replies) == 2


class TestClearCooldown:
    """Lifting a per-symbol entry pause from a phone. The losing streak goes
    with it, so the pause cannot re-impose itself on the next single loss."""

    def _engine(self, tmp_path, monkeypatch, replies=None, symbol="BTC/USD"):
        return _command_engine(tmp_path, monkeypatch,
                               [_command("clear-cooldown", symbol)], replies)

    def _pause(self, engine, symbol="BTC/USD", losses=3):
        engine.state.start_cooldown(
            symbol, datetime.now(timezone.utc) + timedelta(days=5), losses, 5)

    def test_an_active_pause_and_its_streak_both_go(self, tmp_path, monkeypatch):
        replies = []
        engine = self._engine(tmp_path, monkeypatch, replies)
        self._pause(engine)
        engine.state.set_losing_streak("BTC/USD", 2)

        engine.process_commands()

        assert engine.state.cooldown("BTC/USD") is None
        assert engine.state.losing_streak("BTC/USD") == 0
        assert not engine.cooldown_blocked("BTC/USD")
        assert "BTC/USD" in replies[0] and "losing streak 2" in replies[0]

    def test_the_clearance_survives_a_restart(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        self._pause(engine)

        engine.process_commands()

        assert StateStore(engine.state.path).cooldown("BTC/USD") is None

    def test_only_the_named_symbol_is_lifted(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        self._pause(engine, "BTC/USD")
        self._pause(engine, "SPY")

        engine.process_commands()

        assert engine.state.cooldown("SPY") is not None

    def test_a_symbol_outside_the_lineup_is_refused(self, tmp_path, monkeypatch):
        replies = []
        engine = self._engine(tmp_path, monkeypatch, replies, symbol="TSLA")

        engine.process_commands()

        assert "REFUSED" in replies[0] and "lineup" in replies[0]

    def test_nothing_to_clear_says_so_rather_than_claiming_a_lift(self, tmp_path,
                                                                 monkeypatch):
        replies = []
        engine = self._engine(tmp_path, monkeypatch, replies)

        engine.process_commands()

        assert "nothing to clear" in replies[0]
        assert "cooldowns" in engine.state.data and not engine.state.data["cooldowns"]

    def test_a_standing_streak_short_of_the_pause_is_still_reset(self, tmp_path,
                                                                 monkeypatch):
        replies = []
        engine = self._engine(tmp_path, monkeypatch, replies)
        engine.state.set_losing_streak("BTC/USD", 2)

        engine.process_commands()

        assert engine.state.losing_streak("BTC/USD") == 0
        assert "losing streak 2" in replies[0]

    def test_an_undatable_record_is_removable_remotely(self, tmp_path, monkeypatch):
        """The one cooldown fault the briefing alerts on: a record that cannot
        be dated enforces nothing and has to be repaired by hand."""
        replies = []
        engine = self._engine(tmp_path, monkeypatch, replies)
        engine.state.data["cooldowns"]["BTC/USD"] = {"until": "whenever",
                                                     "losses": 3, "days": 5}

        engine.process_commands()

        assert not engine.state.cooldown_unusable("BTC/USD")
        assert "unusable" in replies[0]


class TestEmailCommandIsolation:
    def test_poll_failure_never_reaches_the_cycle(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.maybe_send_briefings = lambda: None
        engine.broker.market_is_open.return_value = False
        engine.process_instrument = MagicMock()

        def _boom(state):
            raise RuntimeError("imap exploded")

        monkeypatch.setattr("bot.main.commands.poll", _boom)

        engine.run_cycle()  # must not raise

        engine.process_instrument.assert_called()  # trading cycle unaffected

    def test_one_command_blowing_up_still_answers_and_continues(
            self, tmp_path, monkeypatch):
        replies = []
        engine = _command_engine(
            tmp_path, monkeypatch,
            [_command("force-exit", "SPY"), _command("status")], replies)
        engine.broker.get_positions.side_effect = RuntimeError("api down")

        engine.process_commands()

        assert len(replies) == 2
        assert "failed" in replies[0].lower()

    def test_reply_failure_does_not_raise(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        monkeypatch.setattr("bot.main.commands.poll",
                            lambda state: [_command("status")])

        def _boom(request, sender, verdict):
            raise RuntimeError("smtp down")

        monkeypatch.setattr("bot.main.commands.send_reply", _boom)

        engine.process_commands()  # must not raise


class TestEmailCommandWiring:
    def test_trading_cycle_polls_once(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.maybe_send_briefings = lambda: None
        engine.broker.market_is_open.return_value = False
        engine.process_instrument = MagicMock()
        polls = []
        monkeypatch.setattr("bot.main.commands.poll",
                            lambda state: polls.append(1) or [])

        engine.run_cycle()

        assert len(polls) == 1

    def test_dry_run_leaves_the_mailbox_to_the_live_bot(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch, dry_run=True)
        polls = []
        monkeypatch.setattr("bot.main.commands.poll",
                            lambda state: polls.append(1) or [])

        engine.process_commands()

        assert polls == []  # a dry run shares the mailbox and must not read it

    def test_halted_cycle_still_accepts_commands(self, tmp_path, monkeypatch):
        engine = _halted_engine(tmp_path, monkeypatch)
        engine.broker.market_is_open.return_value = False
        engine.process_instrument = MagicMock()
        monkeypatch.setattr("bot.main.commands.poll",
                            lambda state: [_command("clear-halt")])
        monkeypatch.setattr("bot.main.commands.send_reply",
                            lambda request, sender, verdict: None)

        engine.run_cycle()

        assert not (tmp_path / "HALT").exists()  # the channel outlives the halt


class TestExitInFlightBlocksASecondClose:
    def test_force_exit_does_not_double_close_crypto_in_the_same_cycle(
            self, tmp_path, monkeypatch):
        """force-exit drops the position meta and marks the exit in flight; the
        sweep that follows in the SAME cycle sees a position on a placeholder
        stop with the market below it, and must not close it again."""
        engine = _command_engine(tmp_path, monkeypatch,
                                 [_command("force-exit", "BTC/USD")])
        engine.state.set_position_meta("BTC/USD", 47_000.0, 50_000.0)
        engine.broker.get_positions.return_value = {
            "BTC/USD": PositionState("BTC/USD", "long", 0.3, 50_000.0, 47_000.0,
                                     50_000.0)}
        engine.data.latest_price.return_value = 46_000.0

        engine.process_commands()
        engine.run_protective_sweeps()

        engine.broker.close_position.assert_called_once_with("BTC/USD")

    def test_crypto_software_stop_defers_while_a_close_is_working(
            self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_pending_fill("BTC/USD", _pending(event="exit", side="sell"))
        engine.broker.get_positions.return_value = {
            "BTC/USD": PositionState("BTC/USD", "long", 0.3, 50_000.0, 47_000.0,
                                     50_000.0)}
        engine.data.latest_price.return_value = 46_000.0

        engine.supervise_crypto_stops(engine.positions())

        engine.broker.close_position.assert_not_called()

    def test_strategy_exit_is_not_resubmitted_while_one_is_in_flight(
            self, tmp_path, monkeypatch):
        pos = PositionState("SPY", "long", 100, 500.0, 490.0, 505.0)
        engine = _hold_engine(tmp_path, monkeypatch, "SPY", pos)
        engine.strategies["SPY"][1].signal.return_value = Signal(Action.EXIT,
                                                                 "snap-back")
        engine.state.set_pending_fill("SPY", _pending(event="exit", side="sell"))

        assert engine.process_instrument("SPY") is True  # bar consumed, not retried

        engine.broker.close_position.assert_not_called()


class TestForceExitIsNotResubmitted:
    def test_a_retried_force_exit_does_not_stack_a_second_close(
            self, tmp_path, monkeypatch):
        """A fresh signature legitimately clears the replay ledger, so the
        operator retrying 60s later reaches the engine twice. The close already
        working is what must stop the second one."""
        replies = []
        engine = _command_engine(tmp_path, monkeypatch,
                                 [_command("force-exit", "SPY")], replies)
        engine.broker.get_positions.return_value = {
            "SPY": PositionState("SPY", "long", 100, 500.0, 490.0, 505.0)}
        engine.data.latest_price.return_value = 505.0

        engine.process_commands()
        engine.process_commands()

        engine.broker.close_position.assert_called_once_with("SPY")
        assert "already working" in replies[1]


def _exit_payload(entry_price=500.0, qty=100.0, side="long"):
    return {"order_id": "oid-1", "side": "sell", "event": "exit",
            "decision_mid": float("nan"), "decision_spread_bps": float("nan"),
            "limit_price": None,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "position_side": side, "entry_price": entry_price, "qty": qty}


def _vanished_engine(tmp_path, monkeypatch, entry_price=741.0):
    engine = _crypto_engine(tmp_path, monkeypatch)
    engine.state.set_position_meta("SPY", 704.0, entry_price)
    engine.state.set_last_seen_position("SPY", "long", 100, entry_price)
    engine.broker.get_positions.return_value = {}
    return engine


class TestLosingStreak:
    def test_losing_engine_close_increments_the_streak(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine._book_reconciled_exit("SPY", _exit_payload(), 490.0, 100.0)
        assert engine.state.losing_streak("SPY") == 1

    def test_winning_engine_close_resets_the_streak(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_losing_streak("SPY", 2)
        engine._book_reconciled_exit("SPY", _exit_payload(), 510.0, 100.0)
        assert engine.state.losing_streak("SPY") == 0

    def test_breakeven_exit_resets_the_streak(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_losing_streak("SPY", 2)
        engine._book_reconciled_exit("SPY", _exit_payload(), 500.0, 100.0)
        assert engine.state.losing_streak("SPY") == 0

    def test_losing_stop_fill_increments_the_streak(self, tmp_path, monkeypatch):
        engine = _vanished_engine(tmp_path, monkeypatch)
        engine.broker.closed_orders.return_value = [_filled_order(price="704.0")]
        engine.positions()
        assert engine.state.losing_streak("SPY") == 1

    def test_winning_stop_fill_resets_the_streak(self, tmp_path, monkeypatch):
        engine = _vanished_engine(tmp_path, monkeypatch)
        engine.state.set_losing_streak("SPY", 2)
        engine.broker.closed_orders.return_value = [_filled_order(price="780.0")]
        engine.positions()
        assert engine.state.losing_streak("SPY") == 0

    def test_reaching_the_limit_starts_a_cooldown_and_clears_the_streak(
            self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_losing_streak("SPY", 2)  # PROTECTIONS default is 3

        engine._book_reconciled_exit("SPY", _exit_payload(), 490.0, 100.0)

        cooldown = engine.state.cooldown("SPY")
        assert cooldown["losses"] == 3
        assert cooldown["days"] == 5
        assert engine.state.losing_streak("SPY") == 0

    def test_a_streak_below_the_limit_starts_no_cooldown(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_losing_streak("SPY", 1)
        engine._book_reconciled_exit("SPY", _exit_payload(), 490.0, 100.0)
        assert engine.state.cooldown("SPY") is None

    def test_a_split_exit_increments_once_per_booking(self, tmp_path, monkeypatch):
        """trades.csv books a row per exit fill and the streak follows it, so a
        split exit counts twice. Pinned: a booking-path refactor must say so."""
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine._book_reconciled_exit("SPY", _exit_payload(qty=40.0), 490.0, 40.0)
        engine._book_reconciled_exit("SPY", _exit_payload(qty=60.0), 490.0, 60.0)
        assert engine.state.losing_streak("SPY") == 2

    def test_zero_after_losses_freezes_an_existing_streak(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bot.main.PROTECTIONS",
                            ProtectionsConfig(cooldown_after_losses=0))
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_losing_streak("SPY", 2)
        engine._book_reconciled_exit("SPY", _exit_payload(), 510.0, 100.0)  # a winner
        assert engine.state.losing_streak("SPY") == 2  # frozen, not reset

    def test_zero_after_losses_disables_the_protection(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bot.main.PROTECTIONS",
                            ProtectionsConfig(cooldown_after_losses=0))
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.set_losing_streak("SPY", 9)
        engine._book_reconciled_exit("SPY", _exit_payload(), 490.0, 100.0)
        assert engine.state.cooldown("SPY") is None


def _cooldown_until(days):
    return datetime.now(timezone.utc) + timedelta(days=days)


class TestCooldownGate:
    def test_active_cooldown_refuses_the_entry(self, tmp_path, monkeypatch):
        engine = _long_signal_engine(tmp_path, monkeypatch, "SPY", Q102)
        engine.state.start_cooldown("SPY", _cooldown_until(5), 3, 5)

        # True: the bar is consumed, because a later cycle on the same bar would
        # find the same cooldown.
        assert engine.process_instrument("SPY") is True
        engine.broker.limit_order.assert_not_called()

    def test_expired_cooldown_allows_the_entry(self, tmp_path, monkeypatch):
        engine = _long_signal_engine(tmp_path, monkeypatch, "SPY", Q102)
        engine.state.start_cooldown("SPY", _cooldown_until(-1), 3, 5)
        engine.risk.approve_entry = MagicMock(
            return_value=MagicMock(qty=100, stop_price=98.0))

        engine.process_instrument("SPY")

        engine.broker.limit_order.assert_called_once()

    def test_a_cooldown_on_one_symbol_leaves_another_trading(self, tmp_path,
                                                             monkeypatch):
        engine = _long_signal_engine(tmp_path, monkeypatch, "SPY", Q102)
        engine.state.start_cooldown("USO", _cooldown_until(5), 3, 5)
        engine.risk.approve_entry = MagicMock(
            return_value=MagicMock(qty=100, stop_price=98.0))

        engine.process_instrument("SPY")

        engine.broker.limit_order.assert_called_once()

    def test_cooldown_survives_a_restart(self, tmp_path):
        path = str(tmp_path / "state.json")
        StateStore(path).start_cooldown("SPY", _cooldown_until(5), 3, 5)
        assert StateStore(path).cooldown("SPY")["losses"] == 3

    def test_cooldown_does_not_block_an_exit(self, tmp_path, monkeypatch):
        pos = PositionState("SPY", "long", 100, 500.0, 490.0, 505.0)
        engine = _hold_engine(tmp_path, monkeypatch, "SPY", pos)
        engine.strategies["SPY"][1].signal.return_value = Signal(Action.EXIT, "snap-back")
        engine.state.start_cooldown("SPY", _cooldown_until(5), 3, 5)
        engine.data.latest_quote.return_value = None

        engine.process_instrument("SPY")

        engine.broker.close_position.assert_called_once_with("SPY")

    def test_cooldown_does_not_block_the_protective_sweep(self, tmp_path, monkeypatch):
        engine = _crypto_engine(tmp_path, monkeypatch)
        engine.state.start_cooldown("SPY", _cooldown_until(5), 3, 5)
        engine.state.set_position_meta("SPY", 490.0, 505.0)
        engine.broker.get_positions.return_value = {
            "SPY": PositionState("SPY", "long", 100, 500.0, 490.0, 505.0)}

        engine.run_protective_sweeps()

        engine.broker.stop_order.assert_called_once()


class TestDailyLossLimit:
    def _engine(self, tmp_path, monkeypatch, day_start=100_000.0, **kwargs):
        if kwargs:
            monkeypatch.setattr("bot.main.PROTECTIONS", ProtectionsConfig(**kwargs))
        engine = _crypto_engine(tmp_path, monkeypatch)
        if day_start is not None:
            engine.state.data["day_start_equity"] = {
                "date": datetime.now(timezone.utc).date().isoformat(),
                "equity": day_start}
        return engine

    def test_loss_past_the_limit_trips_halt(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        assert engine.check_daily_loss_limit(96_999.0) is True
        assert (tmp_path / "HALT").exists()

    def test_loss_exactly_at_the_limit_does_not_trip(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        assert engine.check_daily_loss_limit(97_000.0) is False
        assert not (tmp_path / "HALT").exists()

    def test_halt_carries_the_reason(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        engine.check_daily_loss_limit(90_000.0)
        assert "daily loss limit" in read_halt(str(tmp_path / "HALT"))["reason"]

    def test_trip_does_not_flatten(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        engine.broker.get_positions.return_value = {
            "SPY": PositionState("SPY", "long", 100, 500.0, 490.0, 505.0)}

        engine.check_daily_loss_limit(90_000.0)

        engine.broker.close_position.assert_not_called()
        engine.broker.cancel_open_orders.assert_not_called()

    def test_zero_limit_disables_the_protection(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch, daily_loss_limit_pct=0.0)
        assert engine.check_daily_loss_limit(50_000.0) is False
        assert not (tmp_path / "HALT").exists()

    def test_first_read_of_the_day_seeds_the_reference(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch, day_start=None)
        assert engine.check_daily_loss_limit(80_000.0) is False
        assert engine.state.data["day_start_equity"]["equity"] == 80_000.0

    def test_a_new_utc_day_reseeds_from_the_current_equity(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch, day_start=None)
        engine.state.data["day_start_equity"] = {"date": "2020-01-01",
                                                 "equity": 100_000.0}

        assert engine.check_daily_loss_limit(80_000.0) is False

        today = datetime.now(timezone.utc).date().isoformat()
        assert engine.state.data["day_start_equity"] == {"date": today,
                                                         "equity": 80_000.0}

    def test_run_cycle_stops_before_the_instruments(self, tmp_path, monkeypatch):
        now = datetime.now(timezone.utc)
        session = (now - timedelta(hours=6), now + timedelta(minutes=5))
        engine = _make_engine(tmp_path, monkeypatch, session)
        engine.state.data["day_start_equity"] = {
            "date": now.date().isoformat(), "equity": 100_000.0}
        engine.state.data["peak_equity"] = 90_000.0  # breaker stays clear
        engine.broker.account_equity.return_value = 90_000.0

        engine.run_cycle()

        assert (tmp_path / "HALT").exists()
        assert not _evaluated(engine, "SPY")

    def test_the_day_is_still_booked_on_the_day_the_limit_trips(self, tmp_path,
                                                               monkeypatch):
        now = datetime.now(timezone.utc)
        session = (now - timedelta(hours=6), now + timedelta(minutes=5))
        engine = _make_engine(tmp_path, monkeypatch, session)
        engine.state.data["day_start_equity"] = {
            "date": now.date().isoformat(), "equity": 100_000.0}
        engine.state.data["peak_equity"] = 90_000.0
        engine.state.data["prev_day"] = "2020-01-01"
        engine.state.data["prev_equity"] = 100_000.0
        engine.broker.account_equity.return_value = 90_000.0

        engine.run_cycle()

        assert (tmp_path / "HALT").exists()
        rows = list(csv.reader((tmp_path / "pnl.csv").open()))
        assert rows[-1][0] == now.date().isoformat()

    def test_the_limit_fires_only_once_per_utc_day(self, tmp_path, monkeypatch):
        """Owner decision 2026-09-03: one-shot per UTC day. Clearing the halt is
        the operator deciding to trade on, so re-tripping while still underwater
        would overrule that decision every 60 seconds and make the remote
        clear-halt inert for the rest of the day. The 10% drawdown breaker still
        sits underneath."""
        engine = self._engine(tmp_path, monkeypatch)

        assert engine.check_daily_loss_limit(90_000.0) is True
        clear_halt(str(tmp_path / "HALT"))

        assert engine.check_daily_loss_limit(89_000.0) is False  # deeper, still no
        assert not (tmp_path / "HALT").exists()

    def test_the_trip_records_the_day_it_spent(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        engine.check_daily_loss_limit(90_000.0)
        assert engine.state.daily_loss_tripped_today() is True

    def test_a_new_utc_day_rearms_the_limit(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch, day_start=None)
        engine.state.data["daily_loss_tripped"] = "2020-01-01"
        engine.state.data["day_start_equity"] = {
            "date": datetime.now(timezone.utc).date().isoformat(),
            "equity": 90_000.0}  # today's own fresh reference

        assert engine.check_daily_loss_limit(85_000.0) is True
        assert (tmp_path / "HALT").exists()

    def test_the_drawdown_breaker_still_fires_while_the_limit_is_spent(
            self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        engine.state.data["daily_loss_tripped"] = \
            datetime.now(timezone.utc).date().isoformat()
        engine.state.data["peak_equity"] = 100_000.0

        assert engine.check_daily_loss_limit(85_000.0) is False
        assert engine.check_circuit_breaker(85_000.0) is True
        assert (tmp_path / "HALT").exists()

    def test_run_cycle_resumes_after_a_clear_on_a_spent_day(self, tmp_path,
                                                           monkeypatch):
        """The operator-visible payoff: clear-halt genuinely resumes trading for
        the day rather than being undone by the next cycle."""
        now = datetime.now(timezone.utc)
        session = (now - timedelta(hours=6), now + timedelta(minutes=5))
        engine = _make_engine(tmp_path, monkeypatch, session)
        engine.state.data["day_start_equity"] = {
            "date": now.date().isoformat(), "equity": 100_000.0}
        engine.state.data["daily_loss_tripped"] = now.date().isoformat()
        engine.state.data["peak_equity"] = 90_000.0  # breaker stays clear
        engine.broker.account_equity.return_value = 90_000.0

        engine.run_cycle()

        assert not (tmp_path / "HALT").exists()
        assert _evaluated(engine, "SPY")

    def test_a_nonpositive_day_start_never_trips(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch, day_start=0.0)
        assert engine.check_daily_loss_limit(-5_000.0) is False
        assert not (tmp_path / "HALT").exists()


class TestCooldownRecordHardening:
    """state.json is the operator's only cooldown override, so a hand-edited
    record must never raise into the 60-second cycle."""

    def _store(self, tmp_path, until):
        store = StateStore(str(tmp_path / "state.json"))
        record = {"losses": 3, "days": 5}
        if until is not None:
            record["until"] = until
        store.data["cooldowns"]["SPY"] = record
        return store

    def test_a_naive_until_is_read_as_utc(self, tmp_path):
        future = datetime.now(timezone.utc) + timedelta(days=5)
        store = self._store(tmp_path, future.replace(tzinfo=None).isoformat())
        assert store.cooldown("SPY")["losses"] == 3

    def test_a_naive_expired_until_lapses(self, tmp_path):
        past = datetime.now(timezone.utc) - timedelta(days=1)
        store = self._store(tmp_path, past.replace(tzinfo=None).isoformat())
        assert store.cooldown("SPY") is None

    def test_a_missing_until_lapses_with_a_warning(self, tmp_path, caplog):
        store = self._store(tmp_path, None)
        with caplog.at_level("WARNING"):
            assert store.cooldown("SPY") is None
        assert any("SPY" in r.getMessage() for r in caplog.records)

    def test_an_unparseable_until_lapses_with_a_warning(self, tmp_path, caplog):
        store = self._store(tmp_path, "next tuesday")
        with caplog.at_level("WARNING"):
            assert store.cooldown("SPY") is None
        assert any("SPY" in r.getMessage() for r in caplog.records)

    def test_a_malformed_record_warns_only_once(self, tmp_path, caplog):
        store = self._store(tmp_path, "next tuesday")
        with caplog.at_level("WARNING"):
            store.cooldown("SPY")
            store.cooldown("SPY")
        assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1

    def test_reading_a_malformed_record_leaves_state_untouched(self, tmp_path):
        """Reads must not write: the briefing reads cooldowns too, from a second
        process, against the live bot's state file."""
        store = self._store(tmp_path, "next tuesday")

        store.cooldown("SPY")

        assert store.data["cooldowns"]["SPY"] == {"losses": 3, "days": 5,
                                                  "until": "next tuesday"}

    def test_clearing_rearms_the_warning_for_the_next_broken_record(self, tmp_path,
                                                                    caplog):
        """The once-per-process suppression is keyed on the record that was
        there. Once an operator has cleared it, a record broken again is news."""
        store = self._store(tmp_path, "next tuesday")
        with caplog.at_level("WARNING"):
            store.cooldown("SPY")
            store.clear_cooldown("SPY")
            store.data["cooldowns"]["SPY"] = {"losses": 3, "days": 5,
                                              "until": "next wednesday"}
            store.cooldown("SPY")
        assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 2

    def test_clearing_reports_that_a_record_was_there(self, tmp_path):
        store = self._store(tmp_path, "next tuesday")
        assert store.clear_cooldown("SPY") is True
        assert "SPY" not in store.data["cooldowns"]

    def test_clearing_nothing_reports_nothing_was_there(self, tmp_path):
        store = self._store(tmp_path, "next tuesday")
        assert store.clear_cooldown("QQQ") is False

    def test_a_cleared_cooldown_is_persisted(self, tmp_path):
        """The override has to survive the restart the operator is about to
        trigger, so clearing writes rather than only mutating memory."""
        path = str(tmp_path / "state.json")
        store = StateStore(path)
        store.data["cooldowns"]["SPY"] = {"losses": 3, "days": 5}
        store.save()

        assert store.clear_cooldown("SPY") is True

        assert StateStore(path).data["cooldowns"] == {}

    def test_a_no_op_clear_does_not_rewrite_state(self, tmp_path):
        """Nothing to clear is not a state change, and the briefing reads this
        file from another process."""
        store = self._store(tmp_path, "next tuesday")
        store.save = MagicMock()
        assert store.clear_cooldown("QQQ") is False
        store.save.assert_not_called()

    def test_the_entry_gate_survives_a_malformed_record(self, tmp_path, monkeypatch):
        engine = _long_signal_engine(tmp_path, monkeypatch, "SPY", Q102)
        engine.state.data["cooldowns"]["SPY"] = {"losses": 3, "days": 5}
        engine.risk.approve_entry = MagicMock(
            return_value=MagicMock(qty=100, stop_price=98.0))

        engine.process_instrument("SPY")  # must not raise

        engine.broker.limit_order.assert_called_once()


class TestProtectionsDefaults:
    def test_both_protections_ship_on(self):
        assert (PROTECTIONS.cooldown_after_losses, PROTECTIONS.cooldown_days,
                PROTECTIONS.daily_loss_limit_pct) == (3, 5, 0.03)


class TestUnvalidatedInstrumentWarning:
    """Enabling an instrument is a promotion decision. Where no validation
    verdict was recorded, startup says so out loud rather than letting the
    omission pass as evidence."""

    def _cfg(self, symbol, *, ref=None, override=None, enabled=True):
        return InstrumentConfig(
            symbol=symbol, asset_class=AssetClass.US_EQUITY,
            strategy=StrategyName.TREND_FOLLOWING, timeframe_minutes=1440,
            validation_ref=ref, override_reason=override, enabled=enabled)

    def test_enabled_without_a_ref_warns_once_per_symbol(self, caplog):
        with caplog.at_level(logging.WARNING):
            missing = bot_main.warn_unvalidated_instruments(
                [self._cfg("GLD"), self._cfg("USO")])
        assert missing == ["GLD", "USO"]
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 2
        assert "GLD" in caplog.text and "USO" in caplog.text
        assert "without a recorded validation verdict" in caplog.text

    def test_a_referenced_instrument_is_silent(self, caplog):
        with caplog.at_level(logging.WARNING):
            missing = bot_main.warn_unvalidated_instruments(
                [self._cfg("VOO", ref="validation-2026-09-02.json: VOO TOM PROMOTE")])
        assert missing == []
        assert caplog.text == ""

    def test_a_disabled_instrument_never_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            missing = bot_main.warn_unvalidated_instruments(
                [self._cfg("SPY", enabled=False)])
        assert missing == []
        assert caplog.text == ""

    def test_an_empty_ref_counts_as_no_verdict(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert bot_main.warn_unvalidated_instruments(
                [self._cfg("SPY", ref="  ")]) == ["SPY"]

    def test_it_defaults_to_the_live_lineup(self, caplog):
        """The lineup running without a recorded verdict today. QQQ and VOO are
        the only instruments with one; changing either list means a promotion
        decision was made and should be reviewed, not silently re-baselined."""
        with caplog.at_level(logging.WARNING):
            missing = bot_main.warn_unvalidated_instruments()
        assert missing == ["SPY", "BTC/USD", "GLD", "USO"]
        assert [c.symbol for c in INSTRUMENTS if c.validation_ref] == ["QQQ", "VOO"]

    def test_an_override_changes_the_warning_rather_than_silencing_it(self, caplog):
        """A recorded decision to run without a verdict is still a warning: the
        artifact says nothing about this instrument, and that stays visible."""
        with caplog.at_level(logging.WARNING):
            missing = bot_main.warn_unvalidated_instruments(
                [self._cfg("GLD", override="owner decision 2026-09-01: keep it on")])
        assert missing == ["GLD"]
        assert "enabled by recorded owner override" in caplog.text
        assert "owner decision 2026-09-01: keep it on" in caplog.text
        assert "without a recorded validation verdict" not in caplog.text

    def test_a_verdict_beats_an_override_when_both_are_set(self, caplog):
        with caplog.at_level(logging.WARNING):
            missing = bot_main.warn_unvalidated_instruments(
                [self._cfg("VOO", ref="VOO TOM PROMOTE", override="belt and braces")])
        assert missing == [] and caplog.text == ""

    def test_it_is_never_fatal(self, caplog):
        with caplog.at_level(logging.WARNING):
            bot_main.warn_unvalidated_instruments([self._cfg("GLD")])  # no raise
        assert "GLD: enabled without a recorded validation verdict" in caplog.text


class TestEngineReadsTheLineupThroughConfig:
    """One route to the lineup. A second copy of the enabled-filter would make
    patching config.INSTRUMENTS a silent no-op here."""

    def test_the_engine_builds_exactly_the_enabled_instruments(self, tmp_path,
                                                               monkeypatch):
        template = next(c for c in INSTRUMENTS if c.symbol == "VOO")
        monkeypatch.setattr(config, "INSTRUMENTS",
                            [replace(template, symbol="ON", enabled=True),
                             replace(template, symbol="OFF", enabled=False)])

        engine = _crypto_engine(tmp_path, monkeypatch)

        assert list(engine.strategies) == ["ON"]
