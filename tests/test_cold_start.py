"""Cold start: what the first cycle does with an empty (or foreign) state.json
against a broker that already holds positions, stops and resting orders.

These are characterization tests for the restart invariant recorded in
bot.main's module docstring — broker state is authoritative, state.json is an
accelerator. Every scenario starts from a state file the engine has never
written, so the only trading truth available is the broker's.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest
from helpers import Q102, patch_main_paths, trades_rows

from bot import briefing
from bot.main import Engine
from bot.portfolio import TradeLog
from bot.strategies.base import Action, PositionState, Signal
from config import AssetClass


def _cold_start_engine(tmp_path, monkeypatch, *, positions=None, stops=(),
                       open_orders=(), state=None, market_open=False,
                       session=None, equity=100_000.0):
    """Engine whose state.json does not exist unless `state` is written for it —
    the restart the invariant is about. `stops` and `open_orders` are the symbols
    the broker reports something resting for; `positions` is broker truth."""
    if state is not None:
        (tmp_path / "state.json").write_text(json.dumps(state))
    patch_main_paths(monkeypatch, tmp_path)
    broker = MagicMock()
    broker.account_equity.return_value = equity
    broker.get_positions.return_value = dict(positions or {})
    broker.has_open_stop.side_effect = lambda s: s in stops
    broker.has_open_order.side_effect = lambda s: s in open_orders
    broker.closed_orders.return_value = []
    broker.market_is_open.return_value = market_open
    broker.current_session.return_value = session
    broker.limit_order.return_value = MagicMock(id="order-1")
    broker.stop_order.return_value = MagicMock(id="stop-1")
    broker.close_position.return_value = MagicMock(id="close-1")
    data = MagicMock()
    data.latest_quote.return_value = None
    engine = Engine(broker, data, dry_run=False)
    # A cold start has no mailbox and no Gmail credentials in the test process;
    # neither is part of the invariant under test.
    engine.process_commands = lambda: None
    engine.maybe_send_briefings = lambda: None
    return engine


def _sweep_only(engine):
    """Stub the decision path so an assertion about the cycle's sweeps cannot be
    satisfied (or broken) by a strategy signal."""
    engine.process_instrument = MagicMock(return_value=True)
    return engine


def _stub_strategy(engine, symbol, action, *, atr=12.0, min_bars=2,
                   stop_atr_mult=3.0, closes=(100.0, 101.0, 102.0)):
    cfg, _ = engine.strategies[symbol]
    strat = MagicMock()
    strat.min_bars = min_bars
    strat.stop_atr_mult = stop_atr_mult
    strat.prepare.return_value = pd.DataFrame(
        {"close": list(closes), "atr": [atr] * len(closes)})
    strat.signal.return_value = Signal(action, action.value)
    engine.strategies[symbol] = (cfg, strat)
    idx = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
    engine.data.get_recent_bars.return_value = pd.DataFrame(
        {"close": [1, 2, 3, 4, 5]}, index=idx)
    return engine


SPY_LONG = PositionState("SPY", "long", 100, 741.0, 0.0, 741.0)
BTC_LONG = PositionState("BTC/USD", "long", 0.3, 50_000.0, 0.0, 50_000.0)


class TestEquityPositionWithARestingStop:
    """Scenario (a): a position and its broker-side stop both survived the
    restart; only the bot's memory of them did not. Adopting the resting stop is
    what keeps it from being orphaned or duplicated."""

    def _engine(self, tmp_path, monkeypatch):
        engine = _cold_start_engine(tmp_path, monkeypatch,
                                    positions={"SPY": SPY_LONG}, stops={"SPY"})
        monkeypatch.setattr(engine, "_latest_atr", lambda symbol: 12.0)
        return _sweep_only(engine)

    def test_state_file_absent_before_the_first_cycle(self, tmp_path, monkeypatch):
        self._engine(tmp_path, monkeypatch)
        assert not (tmp_path / "state.json").exists()

    def test_resting_stop_is_adopted_not_duplicated(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)

        engine.run_cycle()

        engine.broker.stop_order.assert_not_called()
        engine.broker.cancel_open_orders.assert_not_called()

    def test_position_is_not_closed(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)

        engine.run_cycle()

        engine.broker.close_position.assert_not_called()
        assert trades_rows(tmp_path) == [TradeLog.header]  # no phantom exit

    def test_stop_meta_is_reconstructed_from_the_strategy_atr(
            self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)

        engine.run_cycle()

        meta = engine.state.position_meta("SPY")
        assert meta["stop_price"] == pytest.approx(705.0)  # 741 - 3 x ATR 12
        assert "degraded" not in meta

    def test_position_identity_is_recorded_for_the_next_vanish_sweep(
            self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)

        engine.run_cycle()

        seen = engine.state.last_seen_position("SPY")
        assert (seen["side"], seen["qty"], seen["entry_price"]) == ("long", 100, 741.0)
        assert seen["seen_at"]  # dated, so a stop fill can be booked as a trade


class TestEquityPositionWithNoRestingStop:
    """Scenario (b): the position survived, the stop did not. The one thing that
    must never happen is closing the position to resolve the disagreement."""

    def test_stop_is_re_placed_at_the_recomputed_atr_level(self, tmp_path, monkeypatch):
        engine = _cold_start_engine(tmp_path, monkeypatch,
                                    positions={"SPY": SPY_LONG})
        monkeypatch.setattr(engine, "_latest_atr", lambda symbol: 12.0)
        _sweep_only(engine)

        engine.run_cycle()

        engine.broker.stop_order.assert_called_once_with(
            "SPY", "sell", 100, 705.0, AssetClass.US_EQUITY)
        engine.broker.close_position.assert_not_called()

    def test_atr_outage_places_the_degraded_fallback_stop_and_flags_it(
            self, tmp_path, monkeypatch):
        engine = _cold_start_engine(tmp_path, monkeypatch,
                                    positions={"SPY": SPY_LONG})
        monkeypatch.setattr(engine, "_latest_atr", lambda symbol: None)
        _sweep_only(engine)

        engine.run_cycle()

        engine.broker.stop_order.assert_called_once_with(
            "SPY", "sell", 100, pytest.approx(733.59), AssetClass.US_EQUITY)
        assert engine.state.position_meta("SPY")["degraded"] is True
        engine.broker.close_position.assert_not_called()

    def test_degraded_stop_is_refined_on_the_first_processed_bar(
            self, tmp_path, monkeypatch):
        engine = _cold_start_engine(tmp_path, monkeypatch,
                                    positions={"SPY": SPY_LONG})
        monkeypatch.setattr(engine, "_latest_atr", lambda symbol: None)
        _stub_strategy(engine, "SPY", Action.HOLD, atr=12.0)

        engine.run_cycle()                     # recovers the degraded stop
        engine.process_instrument("SPY")       # first bar the strategy sees

        meta = engine.state.position_meta("SPY")
        assert meta["stop_price"] == pytest.approx(705.0)
        assert "degraded" not in meta

    def test_a_short_recovers_its_stop_above_entry(self, tmp_path, monkeypatch):
        short = PositionState("SPY", "short", 100, 741.0, 0.0, 741.0)
        engine = _cold_start_engine(tmp_path, monkeypatch, positions={"SPY": short})
        monkeypatch.setattr(engine, "_latest_atr", lambda symbol: 12.0)
        _sweep_only(engine)

        engine.run_cycle()

        engine.broker.stop_order.assert_called_once_with(
            "SPY", "buy", 100, 777.0, AssetClass.US_EQUITY)


class TestCryptoPositionSupervisionResumes:
    """Scenario (c): crypto carries no broker-side stop, so the whole protection
    is the reconstructed software stop."""

    def _engine(self, tmp_path, monkeypatch, price):
        engine = _cold_start_engine(tmp_path, monkeypatch,
                                    positions={"BTC/USD": BTC_LONG})
        monkeypatch.setattr(engine, "_latest_atr", lambda symbol: 1_000.0)
        engine.data.latest_price.return_value = price
        return _sweep_only(engine)

    def test_price_above_the_recovered_stop_holds_the_position(
            self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch, price=50_500.0)

        engine.run_cycle()

        engine.broker.close_position.assert_not_called()
        engine.broker.stop_order.assert_not_called()  # never a broker stop leg
        assert engine.state.position_meta("BTC/USD")["stop_price"] < 50_000.0

    def test_price_through_the_recovered_stop_closes_on_the_first_cycle(
            self, tmp_path, monkeypatch):
        # Supervision resumes at full strength: the reconstructed stop is
        # enforced from the first poll, not from the first bar boundary.
        engine = self._engine(tmp_path, monkeypatch, price=40_000.0)

        engine.run_cycle()

        engine.broker.close_position.assert_called_once_with("BTC/USD")

    def test_orphan_sweep_never_touches_a_crypto_position(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch, price=50_500.0)
        engine.broker.has_open_stop.side_effect = lambda s: True

        engine.run_cycle()

        cancelled = [c.args[0] for c in engine.broker.cancel_open_orders.call_args_list]
        assert "BTC/USD" not in cancelled


class TestOrphanStopOnAFlatSymbol:
    """Scenario (d): a stop resting with no position behind it. Nothing expires
    it, and while it rests it blocks every future entry for that symbol."""

    def _engine(self, tmp_path, monkeypatch):
        engine = _cold_start_engine(tmp_path, monkeypatch, positions={},
                                    stops={"SPY"}, open_orders={"SPY"})
        return _sweep_only(engine)

    def test_orphan_stop_is_cancelled(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)

        engine.run_cycle()

        engine.broker.cancel_open_orders.assert_called_once_with("SPY")

    def test_orphan_stop_is_not_mistaken_for_a_position(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)

        engine.run_cycle()

        assert engine.state.position_meta("SPY") is None
        assert engine.state.last_seen_position("SPY") is None
        engine.broker.close_position.assert_not_called()

    def test_no_phantom_trade_is_booked(self, tmp_path, monkeypatch):
        # The vanish sweep reads state, and a cold start has no position to
        # reconcile — so there is nothing for it to book against.
        engine = self._engine(tmp_path, monkeypatch)

        engine.run_cycle()

        assert trades_rows(tmp_path) == [TradeLog.header]
        engine.broker.closed_orders.assert_not_called()


class TestUnknownRestingEntryOrder:
    """Scenario (e): a resting entry order on a flat symbol that state.json knows
    nothing about — the shape a wiped volume leaves behind, because the resting
    marker is written to state.json but the order lives at the broker."""

    def _blocked(self, tmp_path, monkeypatch):
        """Flat crypto symbol with a long signal and a resting order state.json
        knows nothing about."""
        engine = _cold_start_engine(tmp_path, monkeypatch, positions={},
                                    open_orders={"BTC/USD"})
        _stub_strategy(engine, "BTC/USD", Action.LONG,
                       closes=(50_000.0, 50_100.0, 50_200.0), atr=1_000.0)
        engine.risk.approve_entry = MagicMock(
            return_value=MagicMock(qty=0.3, stop_price=47_000.0))
        return engine

    def test_equity_entry_is_deduped_by_the_broker_order_check(
            self, tmp_path, monkeypatch):
        # The dedupe asks the broker, not state.json, so the lost marker costs
        # nothing. The bar IS consumed here: an equity entry has no cancel of
        # its own in flight, so a retry would find the same standing order.
        engine = _cold_start_engine(tmp_path, monkeypatch, positions={},
                                    open_orders={"SPY"})
        _stub_strategy(engine, "SPY", Action.LONG)
        engine.data.latest_quote.return_value = Q102
        engine.risk.approve_entry = MagicMock(
            return_value=MagicMock(qty=100, stop_price=98.0))

        assert engine.process_instrument("SPY") is True  # bar handled, no retry

        engine.broker.limit_order.assert_not_called()

    def test_unknown_equity_entry_order_is_left_resting(self, tmp_path, monkeypatch):
        """HAZARD (pinned, not fixed): the orphan sweep only cancels resting
        STOPS, so an entry order nobody remembers keeps resting, blocking the
        symbol's entries until it fills or expires. A fill is safe — the
        position is adopted and protected like any other, and its exit is booked
        to trades.csv. What is lost is the entry's execution-quality row in
        fills.csv, and risk-sizing consent for the quantity: nobody approved
        that size against current equity."""
        engine = _cold_start_engine(tmp_path, monkeypatch, positions={},
                                    open_orders={"SPY"})
        _sweep_only(engine)

        engine.run_cycle()

        engine.broker.cancel_open_orders.assert_not_called()
        assert engine.state.pending_fill("SPY") is None

    def test_crypto_entry_is_not_stacked_on_an_unknown_resting_order(
            self, tmp_path, monkeypatch):
        """A crypto maker entry rests GTC at the broker while its only marker
        lives in state.json. With the marker lost, the broker's open-order check
        is the one thing standing between a restart and twice the sized
        position — both orders can fill on one signal's risk budget.

        HAZARD (pinned, not fixed): the suppression is unbounded for an order
        this engine did not place. Nothing cancels a foreign crypto order — the
        orphan sweep is equity-only, since a resting crypto order is normally
        the maker entry — so the bar is never consumed, the symbol takes no new
        entry until the order fills or is cancelled by hand, and the only
        automatic escalation is one WARNING per poll plus the entry-refusal
        marker the briefing carries."""
        engine = self._blocked(tmp_path, monkeypatch)

        assert engine.process_instrument("BTC/USD") is False  # bar retried

        engine.broker.limit_order.assert_not_called()

    def test_suppression_is_recorded_as_an_entry_refusal(self, tmp_path, monkeypatch):
        # A symbol silently taking no entries is the failure mode that hid for a
        # month last time. The suppression is loud in the logs; the marker is
        # what carries it into the morning email.
        engine = self._blocked(tmp_path, monkeypatch)

        engine.process_instrument("BTC/USD")

        refusal = engine.state.entry_refusal("BTC/USD")
        assert "resting order" in refusal["error"]
        assert refusal["count"] == 1

    def test_repeated_polls_count_up_against_the_same_marker(
            self, tmp_path, monkeypatch):
        engine = self._blocked(tmp_path, monkeypatch)

        engine.process_instrument("BTC/USD")
        first_seen = engine.state.entry_refusal("BTC/USD")["since"]
        engine.process_instrument("BTC/USD")

        refusal = engine.state.entry_refusal("BTC/USD")
        assert refusal["count"] == 2
        assert refusal["since"] == first_seen  # how long it has been blocked

    def test_an_accepted_entry_clears_the_marker(self, tmp_path, monkeypatch):
        engine = self._blocked(tmp_path, monkeypatch)
        engine.process_instrument("BTC/USD")
        assert engine.state.entry_refusal("BTC/USD") is not None

        engine.broker.has_open_order.side_effect = lambda s: False
        engine.process_instrument("BTC/USD")

        engine.broker.limit_order.assert_called_once()
        assert engine.state.entry_refusal("BTC/USD") is None

    def test_the_marker_reaches_the_briefing_alerts(self, tmp_path, monkeypatch):
        engine = self._blocked(tmp_path, monkeypatch)

        engine.process_instrument("BTC/USD")

        alerts = briefing.collect_alerts(engine.broker, engine.state, positions={},
                                         halt_path=str(tmp_path / "HALT"))
        assert any("BTC/USD" in a and "resting order" in a for a in alerts)


class TestPendingCancelIsStillAnOutstandingOrder:
    """The mirror image of a lost marker: state says the stale maker order was
    cancelled, the broker still reports it open. Alpaca cancels asynchronously —
    close_position polls has_open_order precisely to wait one out — so the
    boundary-advance path can look at an order it has already cancelled."""

    def _engine(self, tmp_path, monkeypatch):
        engine = _cold_start_engine(tmp_path, monkeypatch, positions={},
                                    open_orders={"BTC/USD"})
        # A marker from the previous bar: the boundary has advanced, so
        # reconcile_crypto_resting cancels the stale order and allows the entry.
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)
                     ).replace(hour=0, minute=0, second=0, microsecond=0)
        engine.state.set_resting_order("BTC/USD", yesterday.isoformat())
        _stub_strategy(engine, "BTC/USD", Action.LONG,
                       closes=(50_000.0, 50_100.0, 50_200.0), atr=1_000.0)
        engine.risk.approve_entry = MagicMock(
            return_value=MagicMock(qty=0.3, stop_price=47_000.0))
        return engine

    def test_no_second_order_is_stacked_on_a_pending_cancel(
            self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)

        engine.process_instrument("BTC/USD")

        engine.broker.cancel_open_orders.assert_called_once_with("BTC/USD")
        engine.broker.limit_order.assert_not_called()

    def test_the_bar_is_retried_rather_than_consumed(self, tmp_path, monkeypatch):
        # Suppressing on an order that is mid-cancellation must not cost the bar:
        # for a daily crypto strategy that would drop the signal for a day.
        engine = self._engine(tmp_path, monkeypatch)

        assert engine.process_instrument("BTC/USD") is False

    def test_the_entry_lands_once_the_cancel_settles(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, monkeypatch)
        engine.process_instrument("BTC/USD")           # suppressed, bar retried
        engine.broker.has_open_order.side_effect = lambda s: False

        engine.process_instrument("BTC/USD")           # next poll, cancel settled

        engine.broker.limit_order.assert_called_once()

    def test_a_pending_cancel_is_not_an_entry_refusal(self, tmp_path, monkeypatch):
        # The engine cancelled this order itself moments ago, so the suppression
        # is the async cancel working as designed — not a symbol that has
        # silently stopped taking entries.
        engine = self._engine(tmp_path, monkeypatch)

        engine.process_instrument("BTC/USD")

        assert engine.state.entry_refusal("BTC/USD") is None

    def test_a_hold_after_the_cancel_leaves_no_standing_alert(
            self, tmp_path, monkeypatch):
        # Only an accepted entry clears the marker, so a marker recorded on a
        # routine cancel stands in the briefing for as long as the signal says
        # hold.
        engine = self._engine(tmp_path, monkeypatch)
        engine.process_instrument("BTC/USD")           # cancel issued, entry retried
        engine.broker.has_open_order.side_effect = lambda s: False
        _, strat = engine.strategies["BTC/USD"]
        strat.signal.return_value = Signal(Action.HOLD, "no signal")

        engine.process_instrument("BTC/USD")           # no entry to clear anything

        assert engine.state.entry_refusal("BTC/USD") is None
        alerts = briefing.collect_alerts(engine.broker, engine.state, positions={},
                                         halt_path=str(tmp_path / "HALT"))
        assert not any("BTC/USD" in a for a in alerts)

    def test_a_standing_refusal_survives_a_routine_cancel(self, tmp_path, monkeypatch):
        # Clearing stays the accepted entry's job: a symbol the broker is
        # refusing must not be silenced by a cycle that only cancelled a stale
        # order.
        engine = self._engine(tmp_path, monkeypatch)
        engine.state.record_entry_refusal("BTC/USD", "order status is rejected")

        engine.process_instrument("BTC/USD")

        refusal = engine.state.entry_refusal("BTC/USD")
        assert refusal["error"] == "order status is rejected"
        assert refusal["count"] == 1

    def test_a_cancel_that_never_settles_is_recorded(self, tmp_path, monkeypatch):
        # Routine covers only the cycle that issued the cancel. An order still
        # resting a poll later is a cancel that did not take or an order nobody
        # placed, and either way the symbol takes no entries until it clears.
        engine = self._engine(tmp_path, monkeypatch)
        engine.process_instrument("BTC/USD")           # cancel issued

        engine.process_instrument("BTC/USD")           # next poll, still resting

        assert engine.state.entry_refusal("BTC/USD") is not None


class TestPeakEquityLost:
    """Scenario (f): peak_equity is the one number in state.json the drawdown
    breaker reads, and nothing at the broker can reconstruct it."""

    def test_breaker_reseeds_from_current_equity_without_tripping(
            self, tmp_path, monkeypatch):
        engine = _cold_start_engine(tmp_path, monkeypatch)

        assert engine.check_circuit_breaker(100_000.0) is False

        assert engine.state.data["peak_equity"] == pytest.approx(100_000.0)
        assert not (tmp_path / "HALT").exists()

    def test_a_real_drawdown_is_forgiven_because_the_peak_is_gone(
            self, tmp_path, monkeypatch):
        """The cost of losing this accelerator, pinned: an account 20% below its
        true peak restarts with the breaker's reference re-seeded at today's
        equity, so the drawdown halt it had already earned never fires. The
        protection is weakened, never inverted — the re-seed can only fail to
        halt, it can never halt or flatten spuriously."""
        engine = _cold_start_engine(tmp_path, monkeypatch, equity=80_000.0)
        _sweep_only(engine)

        engine.run_cycle()

        assert not (tmp_path / "HALT").exists()
        engine.broker.close_position.assert_not_called()

    def test_peak_survives_the_first_cycle_for_the_next_restart(
            self, tmp_path, monkeypatch):
        engine = _cold_start_engine(tmp_path, monkeypatch)
        _sweep_only(engine)

        engine.run_cycle()

        assert json.loads((tmp_path / "state.json").read_text())["peak_equity"] \
            == pytest.approx(100_000.0)


class TestProtectionAcceleratorsLost:
    """Scenario (g): cooldowns, losing streaks and the day's starting equity are
    all accelerators. Losing them weakens the stand-downs; it must never make the
    engine act."""

    def test_no_cooldown_and_no_streak_on_a_fresh_store(self, tmp_path, monkeypatch):
        engine = _cold_start_engine(tmp_path, monkeypatch)

        assert engine.cooldown_blocked("SPY") is False
        assert engine.state.losing_streak("SPY") == 0
        assert engine.state.cooldown_unusable("SPY") is False

    def test_lost_streak_restarts_the_count_rather_than_pausing(
            self, tmp_path, monkeypatch):
        """Pinned reading: two of the three losses that would have paused this
        symbol are forgotten, so it needs three fresh ones. The failure direction
        is more trading, not a wrong action — and every position still carries
        its stop."""
        engine = _cold_start_engine(tmp_path, monkeypatch)

        engine._record_exit_outcome("SPY", -100.0)

        assert engine.state.losing_streak("SPY") == 1
        assert engine.state.cooldown("SPY") is None

    def test_daily_loss_limit_seeds_the_day_from_current_equity(
            self, tmp_path, monkeypatch):
        engine = _cold_start_engine(tmp_path, monkeypatch)

        assert engine.check_daily_loss_limit(80_000.0) is False

        assert engine.state.day_start_equity() == pytest.approx(80_000.0)
        assert not (tmp_path / "HALT").exists()

    def test_first_cycle_writes_no_halt_and_takes_no_action(
            self, tmp_path, monkeypatch):
        engine = _cold_start_engine(tmp_path, monkeypatch,
                                    positions={"SPY": SPY_LONG}, stops={"SPY"})
        monkeypatch.setattr(engine, "_latest_atr", lambda symbol: 12.0)
        _sweep_only(engine)

        engine.run_cycle()

        assert not (tmp_path / "HALT").exists()
        engine.broker.close_position.assert_not_called()
        engine.broker.cancel_open_orders.assert_not_called()

    def test_a_halt_sentinel_is_the_one_file_a_restart_must_obey(
            self, tmp_path, monkeypatch):
        # Unlike state.json, HALT is not an accelerator: it is the only record
        # that the operator (or a rail) stopped trading, and a restart that
        # ignored it would resume on its own.
        engine = _cold_start_engine(tmp_path, monkeypatch)
        (tmp_path / "HALT").write_text("daily loss limit\n")
        engine.process_instrument = MagicMock()

        engine.run_cycle()

        engine.process_instrument.assert_not_called()


class TestStaleVolumeState:
    """Scenario (h): a state.json restored from a stale volume — present and
    readable, but describing an account state that has since moved on, so every
    persisted level disagrees with the live broker and quotes."""

    def _stale(self, **overrides):
        state = {
            "positions": {"SPY": {"stop_price": 500.0, "high_water_mark": 520.0}},
            "peak_equity": 96_000.0,
            "losing_streaks": {"SPY": 2},
        }
        state.update(overrides)
        return state

    def test_stale_stop_is_honoured_not_acted_on(self, tmp_path, monkeypatch):
        # The persisted stop is a whole regime away from the live position. It
        # re-places a wide stop (protection weakened) and closes nothing.
        engine = _cold_start_engine(tmp_path, monkeypatch,
                                    positions={"SPY": SPY_LONG},
                                    state=self._stale())
        _sweep_only(engine)

        engine.run_cycle()

        engine.broker.stop_order.assert_called_once_with(
            "SPY", "sell", 100, 500.0, AssetClass.US_EQUITY)
        engine.broker.close_position.assert_not_called()
        assert trades_rows(tmp_path) == [TradeLog.header]

    def test_stale_peak_below_current_equity_reseeds_upward(
            self, tmp_path, monkeypatch):
        engine = _cold_start_engine(tmp_path, monkeypatch,
                                    positions={"SPY": SPY_LONG},
                                    state=self._stale())
        _sweep_only(engine)

        engine.run_cycle()

        assert engine.state.data["peak_equity"] == pytest.approx(100_000.0)
        assert not (tmp_path / "HALT").exists()

    def test_stale_peak_above_current_equity_does_not_flatten_inside_the_band(
            self, tmp_path, monkeypatch):
        engine = _cold_start_engine(tmp_path, monkeypatch,
                                    positions={"SPY": SPY_LONG},
                                    state=self._stale(peak_equity=105_000.0))
        _sweep_only(engine)

        engine.run_cycle()

        assert not (tmp_path / "HALT").exists()
        engine.broker.close_position.assert_not_called()
        assert engine.state.data["peak_equity"] == pytest.approx(105_000.0)

    def test_stale_positions_the_broker_is_flat_on_are_dropped_unbooked(
            self, tmp_path, monkeypatch):
        # The foreign volume claims a position that never existed on this
        # account. It is dropped, and with no dated last-seen snapshot nothing
        # is booked against it.
        engine = _cold_start_engine(tmp_path, monkeypatch, positions={},
                                    state=self._stale())
        _sweep_only(engine)

        engine.run_cycle()

        assert engine.state.position_meta("SPY") is None
        assert trades_rows(tmp_path) == [TradeLog.header]

    def test_stale_price_levels_converge_through_the_restatement_path(
            self, tmp_path, monkeypatch):
        # A stale file's last_bar is the anchor the rescale path needs: the same
        # bar coming back at a different price restates every persisted level,
        # so a stop recorded pre-split follows the series instead of sitting a
        # corporate action away from the market.
        prior_ts = "2020-04-27T04:00:00+00:00"
        engine = _cold_start_engine(
            tmp_path, monkeypatch,
            state=self._stale(
                positions={"USO": {"stop_price": 7.0, "high_water_mark": 9.0}},
                last_bars={"USO": {"timestamp": prior_ts, "close": 8.0}}))
        idx = pd.DatetimeIndex([pd.Timestamp(prior_ts),
                                pd.Timestamp("2020-04-28T04:00:00+00:00")])

        engine.apply_corporate_action_rescale(
            "USO", pd.DataFrame({"close": [64.0, 60.0]}, index=idx))

        assert engine.state.position_meta("USO")["stop_price"] == pytest.approx(56.0)

    def test_stale_crypto_stop_above_the_market_closes_on_the_first_cycle(
            self, tmp_path, monkeypatch):
        """The one place a state disagreement does drive an action, pinned:
        crypto stops are software-enforced, so a persisted stop the live price is
        already through closes the position on the first poll. It cannot be told
        apart from a real stop, and the direction is de-risking — one unwanted
        exit, never unwanted exposure."""
        engine = _cold_start_engine(
            tmp_path, monkeypatch, positions={"BTC/USD": BTC_LONG},
            state=self._stale(positions={
                "BTC/USD": {"stop_price": 60_000.0, "high_water_mark": 61_000.0}}))
        engine.data.latest_price.return_value = 50_500.0
        _sweep_only(engine)

        engine.run_cycle()

        engine.broker.close_position.assert_called_once_with("BTC/USD")

    def test_stale_cooldown_from_a_past_day_lapses_without_blocking(
            self, tmp_path, monkeypatch):
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        engine = _cold_start_engine(
            tmp_path, monkeypatch,
            state=self._stale(cooldowns={"SPY": {"until": yesterday, "losses": 3,
                                                 "days": 5}}))

        assert engine.cooldown_blocked("SPY") is False
        assert engine.state.cooldown_unusable("SPY") is False

    def test_stale_day_start_equity_from_a_past_day_reseeds(
            self, tmp_path, monkeypatch):
        engine = _cold_start_engine(
            tmp_path, monkeypatch,
            state=self._stale(day_start_equity={"date": "2020-01-01",
                                                "equity": 500_000.0}))

        assert engine.check_daily_loss_limit(100_000.0) is False

        assert engine.state.day_start_equity() == pytest.approx(100_000.0)
        assert not (tmp_path / "HALT").exists()
