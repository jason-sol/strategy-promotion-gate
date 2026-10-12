"""Thin, asset-class-aware wrapper over alpaca-py's TradingClient.

Crypto constraints enforced here: no shorting, no broker-side stop orders
(the engine manages crypto stops in software), GTC time-in-force.
"""

import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetCalendarRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    StopOrderRequest,
)

from bot.retry import retry_transient_reads
from bot.strategies.base import PositionState
from config import INSTRUMENTS, AssetClass


# Alpaca has no per-symbol atomic "cancel orders and close" (close_all_positions
# takes cancel_orders, close_position does not), so a close must wait for the
# cancel to be processed before the shares are released.
CLOSE_CANCEL_TIMEOUT_SECONDS = 10.0
CLOSE_POLL_SECONDS = 0.5


def normalize_symbol(raw: str) -> str:
    """Alpaca reports crypto positions without the slash (BTCUSD -> BTC/USD)."""
    for cfg in INSTRUMENTS:
        if cfg.symbol == raw or cfg.symbol.replace("/", "") == raw:
            return cfg.symbol
    return raw


#: Statuses a freshly submitted order comes back with when the broker refused
#: it. The HTTP call succeeded and an Order was returned, so nothing raises —
#: the refusal is only visible in the payload.
_REFUSED_STATUSES = frozenset({
    "rejected", "canceled", "cancelled", "expired", "suspended",
})


def rejection_reason(order) -> str | None:
    """Why the broker refused this order, or None if it accepted it.

    A 2xx response carrying an Order is not proof of acceptance: Alpaca returns
    a fully-formed order with status='rejected' when it declines one. Treating
    the absence of an exception as success books a close that never happened and
    leaves the position naked, its stop already cancelled.

    A missing status is an acceptance — not every response carries one, and
    absence is not refusal.
    """
    if order is None:
        return "broker returned no order"
    raw = getattr(order, "status", None)
    if raw is None:
        return None
    text = str(getattr(raw, "value", raw)).lower()
    return f"order status is {text}" if text in _REFUSED_STATUSES else None


class Broker:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.client = TradingClient(api_key, secret_key, paper=paper)

    @retry_transient_reads
    def account_equity(self) -> float:
        return float(self.client.get_account().equity)

    @retry_transient_reads
    def get_positions(self) -> dict[str, PositionState]:
        """Positions as known to the broker. stop_price/high_water_mark are not
        broker state; the engine overlays them from its persisted state file."""
        result = {}
        for p in self.client.get_all_positions():
            symbol = normalize_symbol(p.symbol)
            side = "long" if str(p.side).lower().endswith("long") else "short"
            qty = abs(float(p.qty))
            entry = float(p.avg_entry_price)
            result[symbol] = PositionState(
                symbol=symbol, side=side, qty=qty, entry_price=entry,
                stop_price=0.0, high_water_mark=entry,
            )
        return result

    def market_order(self, symbol: str, side: str, qty: float, asset_class: AssetClass,
                     stop_loss: float | None = None):
        """stop_loss attaches a protective stop as an OTO leg (equities only) so
        entry and stop cannot exist without each other."""
        if asset_class == AssetClass.CRYPTO and side == "sell_short":
            raise ValueError("crypto cannot be shorted on Alpaca")
        tif = TimeInForce.GTC if asset_class == AssetClass.CRYPTO else TimeInForce.DAY
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        kwargs = {}
        if stop_loss is not None:
            if asset_class == AssetClass.CRYPTO:
                raise ValueError("crypto stops are software-managed, not broker-side")
            kwargs = {
                "order_class": OrderClass.OTO,
                "stop_loss": StopLossRequest(stop_price=round(stop_loss, 2)),
            }
        req = MarketOrderRequest(symbol=symbol, qty=qty, side=order_side,
                                 time_in_force=tif, **kwargs)
        return self.client.submit_order(order_data=req)

    def limit_order(self, symbol: str, side: str, qty: float, asset_class: AssetClass,
                    limit_price: float, stop_loss: float | None = None):
        """Marketable-limit entry. stop_loss attaches a protective stop as an OTO
        leg (equities only), identical to market_order."""
        tif = TimeInForce.GTC if asset_class == AssetClass.CRYPTO else TimeInForce.DAY
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        kwargs = {}
        if stop_loss is not None:
            if asset_class == AssetClass.CRYPTO:
                raise ValueError("crypto stops are software-managed, not broker-side")
            kwargs = {
                "order_class": OrderClass.OTO,
                "stop_loss": StopLossRequest(stop_price=round(stop_loss, 2)),
            }
        req = LimitOrderRequest(symbol=symbol, qty=qty, side=order_side,
                                time_in_force=tif, limit_price=limit_price, **kwargs)
        return self.client.submit_order(order_data=req)

    def stop_order(self, symbol: str, side: str, qty: float, stop_price: float,
                   asset_class: AssetClass):
        if asset_class == AssetClass.CRYPTO:
            raise ValueError("crypto stops are software-managed, not broker-side")
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        req = StopOrderRequest(symbol=symbol, qty=qty, side=order_side,
                               time_in_force=TimeInForce.GTC, stop_price=round(stop_price, 2))
        return self.client.submit_order(order_data=req)

    @retry_transient_reads
    def get_order_fill(self, order_id) -> tuple[float, float | None, str]:
        """(filled_qty, filled_avg_price, status) for a submitted order. Price is
        None until at least one share fills; status is the lowercased order state
        (e.g. "filled", "new", "canceled")."""
        order = self.client.get_order_by_id(order_id)
        filled_qty = float(order.filled_qty)
        avg = order.filled_avg_price
        filled_avg_price = float(avg) if avg is not None else None
        status = getattr(order.status, "value", str(order.status)).lower()
        return filled_qty, filled_avg_price, status

    @retry_transient_reads
    def has_open_stop(self, symbol: str) -> bool:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        return any("stop" in str(o.order_type).lower()
                   for o in self.client.get_orders(filter=req))

    @retry_transient_reads
    def has_open_order(self, symbol: str) -> bool:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        return bool(self.client.get_orders(filter=req))

    @retry_transient_reads
    def closed_orders(self, symbol: str, limit: int = 20) -> list:
        """Most recent terminal orders for a symbol (filled, cancelled, expired),
        newest first — how a broker-side stop fill is discovered after the fact."""
        req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[symbol],
                               limit=limit)
        return list(self.client.get_orders(filter=req))

    def cancel_open_orders(self, symbol: str) -> None:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        for order in self.client.get_orders(filter=req):
            self.client.cancel_order_by_id(order.id)

    def close_position(self, symbol: str):
        """Alpaca processes cancels asynchronously: the shares stay
        held_for_orders for a moment after the cancel is accepted, and a close
        submitted in that window is rejected with 'insufficient qty available'.
        Wait for the release rather than racing it."""
        self.cancel_open_orders(symbol)
        self._await_orders_released(symbol)
        return self.client.close_position(symbol.replace("/", ""))

    def _await_orders_released(self, symbol: str) -> None:
        deadline = time.monotonic() + CLOSE_CANCEL_TIMEOUT_SECONDS
        while self.has_open_order(symbol):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{symbol}: open orders still pending cancellation after "
                    f"{CLOSE_CANCEL_TIMEOUT_SECONDS:.0f}s; not closing")
            time.sleep(CLOSE_POLL_SECONDS)

    @retry_transient_reads
    def market_is_open(self) -> bool:
        return self.client.get_clock().is_open

    @retry_transient_reads
    def current_session(self) -> tuple[datetime, datetime] | None:
        """Today's regular-session open/close as UTC timestamps, or None on a
        non-trading day. The calendar reports US/Eastern times; ZoneInfo makes
        the ET->UTC conversion DST-correct (the clock only exposes the next
        session's open once today's session is already underway)."""
        et = ZoneInfo("America/New_York")
        today = datetime.now(timezone.utc).astimezone(et).date()
        entries = self.client.get_calendar(GetCalendarRequest(start=today, end=today))
        if not entries:
            return None
        entry = entries[0]
        open_utc = entry.open.replace(tzinfo=et).astimezone(timezone.utc)
        close_utc = entry.close.replace(tzinfo=et).astimezone(timezone.utc)
        return open_utc, close_utc
