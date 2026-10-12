"""Pure pandas indicator functions shared by live trading and backtesting."""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def rolling_std(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).std()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    # Wilder smoothing
    return true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    # loss == 0 with gains -> rs = inf -> RSI 100; 0/0 (flat) -> NaN.
    return 100 - 100 / (1 + gain / loss)


def rolling_high(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).max()


def rolling_low(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).min()


def realized_vol(close: pd.Series, period: int) -> pd.Series:
    """Annualized realized volatility of daily returns over a rolling window."""
    return close.pct_change().rolling(period).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def add_vol_features(df: pd.DataFrame, lookback: int, rank_window: int) -> pd.DataFrame:
    """Add realized vol and its rolling percentile rank for a stand-aside gate.

    vol_pct is the percentile of the current realized vol within the trailing
    rank_window window (1.0 == most volatile in the window). NaN through the
    warmup (first lookback + rank_window - 1 bars) where the window isn't full.
    """
    df = df.copy()
    df["realized_vol"] = realized_vol(df["close"], lookback)
    df["vol_pct"] = df["realized_vol"].rolling(rank_window).rank(pct=True)
    return df


def realized_vol_at(close: pd.Series, i: int, period: int) -> float:
    """Realized vol as of bar i, computed only from closes up to and including i.

    Shared by the live engine and the backtester so the two cannot drift; the
    slice bounds the window at i to rule out lookahead beyond the current bar.
    """
    return float(realized_vol(close.iloc[: i + 1], period).iloc[-1])
