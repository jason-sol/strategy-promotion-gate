"""Historical bar fetching for stocks and crypto via alpaca-py.

Free-tier constraint: stock historical queries must end >=15 minutes in the
past; crypto data has no such restriction.

latest_quote caveat: on the free IEX feed a quote reflects only IEX's own book,
which is thinner than the consolidated NBBO. The measured spread is therefore
biased wide, so any spread-based gating built on it is conservative (it skips
more, never fewer, than a full-market feed would).
"""

import math
from collections import namedtuple
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import (
    CryptoBarsRequest,
    CryptoLatestQuoteRequest,
    CryptoLatestTradeRequest,
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockLatestTradeRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from bot.retry import retry_transient_reads
from config import AssetClass

STOCK_DATA_DELAY = timedelta(minutes=16)
EQUITY_TRADING_MINUTES_PER_DAY = 390
ET = ZoneInfo("America/New_York")

LatestQuote = namedtuple("LatestQuote", ["bid", "ask", "mid", "spread_bps"])


def bar_boundary(now: datetime, timeframe_minutes: int) -> datetime:
    """Start of the bar `now` falls in, floored against the UTC epoch."""
    epoch_minutes = int(now.timestamp() // 60)
    floored = epoch_minutes - epoch_minutes % timeframe_minutes
    return datetime.fromtimestamp(floored * 60, tz=timezone.utc)


def is_daily_equity(asset_class: AssetClass, timeframe_minutes: int) -> bool:
    return asset_class == AssetClass.US_EQUITY and timeframe_minutes >= 1440


def et_session_date(now_utc: datetime) -> date:
    return now_utc.astimezone(ET).date()


def decision_bars(bars: pd.DataFrame, asset_class: AssetClass,
                  timeframe_minutes: int, now_utc: datetime,
                  keep_forming: bool = False) -> pd.DataFrame:
    """Bars a decision may be taken on.

    Alpaca stamps daily equity bars at 00:00 America/New_York, not 00:00 UTC, so
    the UTC bar boundary is the wrong cutoff for them: it drops the session's own
    bar while the session runs and adopts it again once ET midnight has passed in
    UTC terms. Everything else floors to the UTC boundary as before.

    keep_forming retains the current session's incomplete bar and applies only to
    daily equities — see Engine.keeps_forming_bar for when that is legitimate.
    """
    if bars.empty:
        return bars
    if is_daily_equity(asset_class, timeframe_minutes):
        cutoff_date = et_session_date(now_utc) + timedelta(days=1 if keep_forming else 0)
        cutoff = datetime.combine(cutoff_date, time(0, 0), tzinfo=ET).astimezone(timezone.utc)
    else:
        cutoff = bar_boundary(now_utc, timeframe_minutes)
    return bars[bars.index < cutoff]


def timeframe_from_minutes(minutes: int) -> TimeFrame:
    if minutes % 1440 == 0:
        return TimeFrame(minutes // 1440, TimeFrameUnit.Day)
    if minutes % 60 == 0:
        return TimeFrame(minutes // 60, TimeFrameUnit.Hour)
    return TimeFrame(minutes, TimeFrameUnit.Minute)


def calendar_days_for_bars(bars: int, timeframe_minutes: int, asset_class: AssetClass) -> int:
    """Calendar days of history needed to yield roughly `bars` bars."""
    if asset_class == AssetClass.CRYPTO:
        return math.ceil(bars * timeframe_minutes / 1440 * 1.1) + 2
    trading_days = bars * timeframe_minutes / EQUITY_TRADING_MINUTES_PER_DAY
    return math.ceil(trading_days * 1.6) + 5


class MarketData:
    """feed="sip": full-market bars, free only when the query ends >=15 min ago
    (right for backtests). feed="iex": free real-time bars from the IEX exchange
    (right for live signals, which need the just-closed bar immediately)."""

    def __init__(self, api_key: str | None, secret_key: str | None, feed: str = "sip"):
        # Crypto data requires no keys; stock data does.
        self._stock = StockHistoricalDataClient(api_key, secret_key) if api_key else None
        self._crypto = CryptoHistoricalDataClient()
        self.feed = DataFeed.IEX if feed == "iex" else DataFeed.SIP

    @retry_transient_reads
    def get_bars(self, symbol: str, asset_class: AssetClass, timeframe_minutes: int,
                 start: datetime, end: datetime | None = None) -> pd.DataFrame:
        now = datetime.now(timezone.utc)
        tf = timeframe_from_minutes(timeframe_minutes)

        if asset_class == AssetClass.CRYPTO:
            req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=tf,
                                    start=start, end=end or now)
            bars = self._crypto.get_crypto_bars(req)
        else:
            if self._stock is None:
                raise RuntimeError(
                    "Stock data requires Alpaca API keys. Copy .env.example to .env "
                    "and add your paper trading keys from https://app.alpaca.markets"
                )
            capped_end = end or now
            if self.feed == DataFeed.SIP:
                capped_end = min(capped_end, now - STOCK_DATA_DELAY)
            # Split/dividend-adjusted: raw bars fake huge moves across splits
            # (e.g. USO's 2020 8:1 reverse split).
            req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf,
                                   start=start, end=capped_end, feed=self.feed,
                                   adjustment=Adjustment.ALL)
            bars = self._stock.get_stock_bars(req)

        df = bars.df
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel("symbol")
        return df[["open", "high", "low", "close", "volume"]].sort_index()

    def get_recent_bars(self, symbol: str, asset_class: AssetClass,
                        timeframe_minutes: int, bars: int) -> pd.DataFrame:
        days = calendar_days_for_bars(bars, timeframe_minutes, asset_class)
        start = datetime.now(timezone.utc) - timedelta(days=days)
        return self.get_bars(symbol, asset_class, timeframe_minutes, start)

    @retry_transient_reads
    def latest_price(self, symbol: str, asset_class: AssetClass) -> float:
        if asset_class == AssetClass.CRYPTO:
            req = CryptoLatestTradeRequest(symbol_or_symbols=symbol)
            return float(self._crypto.get_crypto_latest_trade(req)[symbol].price)
        req = StockLatestTradeRequest(symbol_or_symbols=symbol, feed=self.feed)
        return float(self._stock.get_stock_latest_trade(req)[symbol].price)

    @retry_transient_reads
    def latest_quote(self, symbol: str, asset_class: AssetClass) -> LatestQuote:
        """Top-of-book bid/ask with derived mid and spread in bps. See the module
        docstring: on IEX the spread is biased wide."""
        if asset_class == AssetClass.CRYPTO:
            req = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
            q = self._crypto.get_crypto_latest_quote(req)[symbol]
        else:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=self.feed)
            q = self._stock.get_stock_latest_quote(req)[symbol]
        bid = float(q.bid_price)
        ask = float(q.ask_price)
        mid = (bid + ask) / 2
        spread_bps = (ask - bid) / mid * 10_000 if mid > 0 else float("nan")
        return LatestQuote(bid=bid, ask=ask, mid=mid, spread_bps=spread_bps)
