from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from alpaca.data.timeframe import TimeFrameUnit
from alpaca.trading.enums import OrderStatus, QueryOrderStatus, TimeInForce

from bot.broker import Broker, normalize_symbol
from bot.data import (
    MarketData,
    calendar_days_for_bars,
    decision_bars,
    timeframe_from_minutes,
)
from config import AssetClass


class TestTimeframes:
    def test_15_minutes(self):
        tf = timeframe_from_minutes(15)
        assert tf.amount == 15 and tf.unit == TimeFrameUnit.Minute

    def test_1_hour(self):
        tf = timeframe_from_minutes(60)
        assert tf.amount == 1 and tf.unit == TimeFrameUnit.Hour

    def test_4_hour(self):
        tf = timeframe_from_minutes(240)
        assert tf.amount == 4 and tf.unit == TimeFrameUnit.Hour

    def test_daily(self):
        tf = timeframe_from_minutes(1440)
        assert tf.amount == 1 and tf.unit == TimeFrameUnit.Day


class TestCalendarDays:
    def test_crypto_close_to_wall_clock(self):
        # 24 one-hour bars ~ 1 day
        assert 3 <= calendar_days_for_bars(24, 60, AssetClass.CRYPTO) <= 5

    def test_equity_accounts_for_market_hours(self):
        # 260 15-min bars ~ 10 trading days -> needs well over 10 calendar days
        assert calendar_days_for_bars(260, 15, AssetClass.US_EQUITY) >= 15


class TestNormalizeSymbol:
    def test_crypto_slash_restored(self):
        assert normalize_symbol("BTCUSD") == "BTC/USD"

    def test_equity_unchanged(self):
        assert normalize_symbol("SPY") == "SPY"


@pytest.fixture
def broker():
    with patch("bot.broker.TradingClient") as mock_cls:
        b = Broker("key", "secret", paper=True)
        b.client = mock_cls.return_value
        yield b


class TestBroker:
    def test_crypto_market_order_uses_gtc(self, broker):
        broker.market_order("BTC/USD", "buy", 0.5, AssetClass.CRYPTO)
        req = broker.client.submit_order.call_args.kwargs["order_data"]
        assert req.time_in_force == TimeInForce.GTC

    def test_equity_market_order_uses_day(self, broker):
        broker.market_order("SPY", "buy", 100, AssetClass.US_EQUITY)
        req = broker.client.submit_order.call_args.kwargs["order_data"]
        assert req.time_in_force == TimeInForce.DAY

    def test_crypto_short_rejected(self, broker):
        with pytest.raises(ValueError):
            broker.market_order("BTC/USD", "sell_short", 0.5, AssetClass.CRYPTO)

    def test_crypto_stop_order_rejected(self, broker):
        with pytest.raises(ValueError):
            broker.stop_order("BTC/USD", "sell", 0.5, 50_000.0, AssetClass.CRYPTO)

    def test_equity_entry_attaches_oto_stop(self, broker):
        broker.market_order("SPY", "buy", 100, AssetClass.US_EQUITY, stop_loss=98.126)
        req = broker.client.submit_order.call_args.kwargs["order_data"]
        assert str(req.order_class).lower().endswith("oto")
        assert float(req.stop_loss.stop_price) == pytest.approx(98.13)

    def test_limit_order_sets_limit_price_and_day_tif(self, broker):
        broker.limit_order("SPY", "buy", 100, AssetClass.US_EQUITY, limit_price=500.05)
        req = broker.client.submit_order.call_args.kwargs["order_data"]
        assert float(req.limit_price) == pytest.approx(500.05)
        assert req.time_in_force == TimeInForce.DAY

    def test_limit_order_attaches_oto_stop(self, broker):
        broker.limit_order("SPY", "buy", 100, AssetClass.US_EQUITY,
                           limit_price=500.05, stop_loss=490.126)
        req = broker.client.submit_order.call_args.kwargs["order_data"]
        assert str(req.order_class).lower().endswith("oto")
        assert float(req.stop_loss.stop_price) == pytest.approx(490.13)

    def test_limit_order_crypto_stop_rejected(self, broker):
        with pytest.raises(ValueError):
            broker.limit_order("BTC/USD", "buy", 0.5, AssetClass.CRYPTO,
                               limit_price=50_000.0, stop_loss=48_000.0)

    def test_current_session_converts_eastern_to_utc(self, broker):
        entry = MagicMock(open=datetime(2026, 7, 15, 9, 30),
                          close=datetime(2026, 7, 15, 16, 0))
        broker.client.get_calendar.return_value = [entry]
        open_utc, close_utc = broker.current_session()
        # 2026-07-15 is EDT (UTC-4): 09:30 ET -> 13:30 UTC, 16:00 ET -> 20:00 UTC.
        assert open_utc == datetime(2026, 7, 15, 13, 30, tzinfo=timezone.utc)
        assert close_utc == datetime(2026, 7, 15, 20, 0, tzinfo=timezone.utc)

    def test_current_session_none_on_non_trading_day(self, broker):
        broker.client.get_calendar.return_value = []
        assert broker.current_session() is None

    def test_crypto_entry_with_stop_loss_rejected(self, broker):
        with pytest.raises(ValueError):
            broker.market_order("BTC/USD", "buy", 0.5, AssetClass.CRYPTO, stop_loss=48_000.0)

    def test_has_open_stop_detects_stop_orders(self, broker):
        broker.client.get_orders.return_value = [MagicMock(order_type="OrderType.STOP")]
        assert broker.has_open_stop("SPY")
        broker.client.get_orders.return_value = [MagicMock(order_type="OrderType.MARKET")]
        assert not broker.has_open_stop("SPY")

    def test_has_open_order_queries_open_orders_for_symbol(self, broker):
        broker.client.get_orders.return_value = [MagicMock()]
        assert broker.has_open_order("BTC/USD")
        req = broker.client.get_orders.call_args.kwargs["filter"]
        assert req.status == QueryOrderStatus.OPEN
        assert req.symbols == ["BTC/USD"]
        broker.client.get_orders.return_value = []
        assert not broker.has_open_order("BTC/USD")

    def test_crypto_limit_order_uses_gtc_and_no_stop(self, broker):
        broker.limit_order("BTC/USD", "buy", 0.5, AssetClass.CRYPTO, limit_price=49_975.0)
        req = broker.client.submit_order.call_args.kwargs["order_data"]
        assert req.time_in_force == TimeInForce.GTC
        assert float(req.limit_price) == pytest.approx(49_975.0)
        assert req.order_class is None

    def test_equity_stop_order_submitted(self, broker):
        broker.stop_order("SPY", "sell", 100, 98.123, AssetClass.US_EQUITY)
        req = broker.client.submit_order.call_args.kwargs["order_data"]
        assert float(req.stop_price) == pytest.approx(98.12)

    def test_close_position_strips_slash_and_cancels_orders(self, broker, monkeypatch):
        monkeypatch.setattr("bot.broker.CLOSE_POLL_SECONDS", 0.0)
        broker.client.get_orders.side_effect = [[MagicMock(id="oid-1")], []]
        broker.close_position("BTC/USD")
        broker.client.cancel_order_by_id.assert_called_once_with("oid-1")
        broker.client.close_position.assert_called_once_with("BTCUSD")

    def test_closed_orders_queries_closed_status_for_symbol(self, broker):
        broker.client.get_orders.return_value = [MagicMock()]
        assert broker.closed_orders("SPY")
        req = broker.client.get_orders.call_args.kwargs["filter"]
        assert req.status == QueryOrderStatus.CLOSED
        assert req.symbols == ["SPY"]


class TestClosePositionWaitsForCancel:
    """Alpaca cancels asynchronously: shares stay held_for_orders for a moment
    after the cancel is accepted, and a close in that window is rejected with
    'insufficient qty available'."""

    def test_waits_until_orders_released_then_closes(self, broker, monkeypatch):
        monkeypatch.setattr("bot.broker.CLOSE_POLL_SECONDS", 0.0)
        resting = MagicMock(id="oid-1")
        # cancel sweep, then two polls: still held, then released.
        broker.client.get_orders.side_effect = [[resting], [resting], []]
        broker.close_position("SPY")
        assert broker.client.get_orders.call_count == 3
        broker.client.close_position.assert_called_once_with("SPY")

    def test_closes_immediately_when_nothing_rests(self, broker):
        broker.client.get_orders.return_value = []
        broker.close_position("SPY")
        broker.client.cancel_order_by_id.assert_not_called()
        broker.client.close_position.assert_called_once_with("SPY")

    def test_raises_without_closing_when_never_released(self, broker, monkeypatch):
        monkeypatch.setattr("bot.broker.CLOSE_POLL_SECONDS", 0.0)
        monkeypatch.setattr("bot.broker.CLOSE_CANCEL_TIMEOUT_SECONDS", 0.0)
        broker.client.get_orders.return_value = [MagicMock(id="oid-1")]
        with pytest.raises(TimeoutError):
            broker.close_position("SPY")
        broker.client.close_position.assert_not_called()

    def test_positions_normalized(self, broker):
        p = MagicMock(symbol="BTCUSD", qty="0.5", avg_entry_price="50000", side="PositionSide.LONG")
        broker.client.get_all_positions.return_value = [p]
        positions = broker.get_positions()
        assert "BTC/USD" in positions
        assert positions["BTC/USD"].side == "long"
        assert positions["BTC/USD"].qty == 0.5

    def test_get_order_fill_maps_filled_order(self, broker):
        broker.client.get_order_by_id.return_value = MagicMock(
            filled_qty="100", filled_avg_price="500.42", status=OrderStatus.FILLED)
        qty, price, status = broker.get_order_fill("oid-1")
        broker.client.get_order_by_id.assert_called_once_with("oid-1")
        assert qty == pytest.approx(100.0)
        assert price == pytest.approx(500.42)
        assert status == "filled"

    def test_get_order_fill_unfilled_has_none_price(self, broker):
        broker.client.get_order_by_id.return_value = MagicMock(
            filled_qty="0", filled_avg_price=None, status=OrderStatus.NEW)
        qty, price, status = broker.get_order_fill("oid-2")
        assert qty == pytest.approx(0.0)
        assert price is None
        assert status == "new"


@pytest.fixture
def market_data():
    md = MarketData("key", "secret", feed="iex")
    md._stock = MagicMock()
    md._crypto = MagicMock()
    return md


def _quote(bid, ask):
    return MagicMock(bid_price=bid, ask_price=ask)


class TestLatestQuote:
    def test_equity_quote_field_mapping(self, market_data):
        market_data._stock.get_stock_latest_quote.return_value = {"SPY": _quote(499.90, 500.10)}
        q = market_data.latest_quote("SPY", AssetClass.US_EQUITY)
        assert q.bid == pytest.approx(499.90)
        assert q.ask == pytest.approx(500.10)
        assert q.mid == pytest.approx(500.0)
        # spread 0.20 on a 500.0 mid = 4 bps
        assert q.spread_bps == pytest.approx(4.0)

    def test_equity_quote_uses_configured_feed(self, market_data):
        market_data._stock.get_stock_latest_quote.return_value = {"SPY": _quote(499.90, 500.10)}
        market_data.latest_quote("SPY", AssetClass.US_EQUITY)
        req = market_data._stock.get_stock_latest_quote.call_args.args[0]
        assert req.feed == market_data.feed

    def test_crypto_quote_field_mapping(self, market_data):
        market_data._crypto.get_crypto_latest_quote.return_value = {
            "BTC/USD": _quote(49_990.0, 50_010.0)}
        q = market_data.latest_quote("BTC/USD", AssetClass.CRYPTO)
        assert q.bid == pytest.approx(49_990.0)
        assert q.ask == pytest.approx(50_010.0)
        assert q.mid == pytest.approx(50_000.0)
        assert q.spread_bps == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Task 8a — decision-bar selection (Alpaca stamps daily equity bars at ET midnight)
# ---------------------------------------------------------------------------

def _daily_equity_frame(*iso_stamps):
    idx = pd.DatetimeIndex([pd.Timestamp(s) for s in iso_stamps])
    return pd.DataFrame({"close": list(range(1, len(iso_stamps) + 1))}, index=idx)


class TestDecisionBars:
    # 2026-08-31 is EDT: the session bar is stamped 00:00 ET == 04:00Z.
    YESTERDAY = "2026-08-28T04:00:00+00:00"
    TODAY = "2026-08-31T04:00:00+00:00"
    NOW = datetime(2026, 8, 31, 19, 50, tzinfo=timezone.utc)  # 15:50 ET

    def test_daily_equity_drops_the_forming_session_bar_by_default(self):
        bars = _daily_equity_frame(self.YESTERDAY, self.TODAY)
        out = decision_bars(bars, AssetClass.US_EQUITY, 1440, self.NOW)
        assert list(out.index.astype(str)) == [str(pd.Timestamp(self.YESTERDAY))]

    def test_daily_equity_keeps_the_forming_session_bar_when_asked(self):
        bars = _daily_equity_frame(self.YESTERDAY, self.TODAY)
        out = decision_bars(bars, AssetClass.US_EQUITY, 1440,
                            self.NOW, keep_forming=True)
        assert len(out) == 2

    def test_daily_equity_after_the_close_still_excludes_the_session_bar(self):
        # 21:00Z is 17:00 ET: the UTC bar boundary (00:00Z) would now KEEP the
        # session bar, which is exactly the stale/fresh flip-flop being fixed.
        after_close = datetime(2026, 8, 31, 21, 0, tzinfo=timezone.utc)
        bars = _daily_equity_frame(self.YESTERDAY, self.TODAY)
        out = decision_bars(bars, AssetClass.US_EQUITY, 1440, after_close)
        assert len(out) == 1

    def test_daily_equity_handles_est(self):
        # January: ET midnight is 05:00Z.
        now = datetime(2026, 1, 15, 20, 50, tzinfo=timezone.utc)
        bars = _daily_equity_frame("2026-01-14T05:00:00+00:00",
                                   "2026-01-15T05:00:00+00:00")
        assert len(decision_bars(bars, AssetClass.US_EQUITY, 1440, now)) == 1
        assert len(decision_bars(bars, AssetClass.US_EQUITY, 1440, now,
                                 keep_forming=True)) == 2

    def test_daily_crypto_uses_the_utc_boundary(self):
        bars = _daily_equity_frame("2026-08-30T00:00:00+00:00",
                                   "2026-08-31T00:00:00+00:00")
        out = decision_bars(bars, AssetClass.CRYPTO, 1440, self.NOW)
        assert len(out) == 1  # today's 00:00Z bar is still forming

    def test_intraday_equity_ignores_keep_forming(self):
        bars = _daily_equity_frame("2026-08-31T12:00:00+00:00",
                                   "2026-08-31T16:00:00+00:00")
        now = datetime(2026, 8, 31, 19, 50, tzinfo=timezone.utc)
        for keep in (False, True):
            out = decision_bars(bars, AssetClass.US_EQUITY, 240, now,
                                keep_forming=keep)
            assert len(out) == 1  # only 240m daily equities opt in

    def test_empty_frame_passes_through(self):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        assert decision_bars(empty, AssetClass.US_EQUITY, 1440, self.NOW).empty
