"""Bar-by-bar backtest over the same strategy and risk code the live bot runs.

Usage: uv run python -m bot.backtest
"""

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from bot.data import MarketData, calendar_days_for_bars
from bot.metrics import TradeStats, max_drawdown, sharpe_ratio, summarize_trades
from bot.portfolio import SimAccount
from bot.risk_manager import RiskManager
from bot.strategies import build_strategy
from bot.strategies.base import Action, Strategy
from config import (
    BACKTEST_INITIAL_EQUITY as INITIAL_EQUITY,
    BACKTEST_MONTHS,
    CRYPTO_TAKER_FEE_PCT,
    INSTRUMENTS,
    MARKET_DATA_FEED,
    RISK,
    AssetClass,
    InstrumentConfig,
)


@dataclass
class TradeRecord:
    timestamp: pd.Timestamp
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    qty: float
    reason: str


@dataclass
class BacktestReport:
    trades: list[TradeRecord]
    equity_curve: pd.Series           # portfolio equity indexed by timestamp
    symbol_pnl: dict[str, pd.Series]  # cumulative net P&L per symbol
    halted_at: pd.Timestamp | None

    def instrument_stats(self, symbol: str) -> TradeStats:
        return summarize_trades([t.pnl for t in self.trades if t.symbol == symbol])

    def _daily_curve(self, symbol: str) -> pd.Series:
        return INITIAL_EQUITY + self.symbol_pnl[symbol].resample("1D").last().dropna()

    def instrument_sharpe(self, symbol: str) -> float:
        return sharpe_ratio(self._daily_curve(symbol).pct_change())

    def instrument_max_drawdown(self, symbol: str) -> float:
        return max_drawdown(self._daily_curve(symbol))


def fill_price(close: float, buying: bool, slippage_pct: float) -> float:
    return close * (1 + slippage_pct) if buying else close * (1 - slippage_pct)


def trade_fee(notional: float, asset_class: AssetClass) -> float:
    return notional * CRYPTO_TAKER_FEE_PCT if asset_class == AssetClass.CRYPTO else 0.0


class Backtester:
    def __init__(self, initial_equity: float = INITIAL_EQUITY, halt_enabled: bool = True):
        self.acct = SimAccount(cash=initial_equity)
        self.risk = RiskManager(RISK)
        self.peak_equity = initial_equity
        # halt_enabled=False is for diagnostics only: shows full-window strategy
        # behaviour instead of truncating at the first 10% drawdown.
        self.halt_enabled = halt_enabled

    def run(self, instruments: list[tuple[InstrumentConfig, Strategy, pd.DataFrame]],
            sim_start: pd.Timestamp) -> BacktestReport:
        events = sorted(
            (ts, k, i)
            for k, (_, _, df) in enumerate(instruments)
            for i, ts in enumerate(df.index)
        )
        cfg_by_symbol = {cfg.symbol: cfg for cfg, _, _ in instruments}
        marks: dict[str, float] = {}
        trades: list[TradeRecord] = []
        realized: dict[str, float] = {cfg.symbol: 0.0 for cfg, _, _ in instruments}
        sym_pnl: dict[str, list] = {cfg.symbol: [] for cfg, _, _ in instruments}
        curve: list = []
        halted_at = None

        def close_position(symbol, inst_cfg, price, ts, reason):
            pos = self.acct.positions[symbol]
            fp = fill_price(price, buying=pos.side == "short",
                            slippage_pct=inst_cfg.slippage_pct)
            pnl = self.acct.close_position(
                symbol, fp, fee=trade_fee(pos.qty * fp, inst_cfg.asset_class))
            realized[symbol] += pnl
            trades.append(TradeRecord(ts, symbol, pos.side, pos.entry_price, fp,
                                      pnl, pos.qty, reason))

        for ts, k, i in events:
            cfg, strategy, df = instruments[k]
            symbol = cfg.symbol
            row = df.iloc[i]
            marks[symbol] = row["close"]

            pos = self.acct.positions.get(symbol)
            if pos is not None:
                pos.update_high_water_mark(row["close"])
                stop_hit = (pos.side == "long" and row["low"] <= pos.stop_price) or \
                           (pos.side == "short" and row["high"] >= pos.stop_price)
                if stop_hit:
                    close_position(symbol, cfg, pos.stop_price, ts, "hard stop")
                    pos = None

            if ts >= sim_start and halted_at is None:
                signal = strategy.signal(df, i, pos)
                if signal.action == Action.EXIT and pos is not None:
                    close_position(symbol, cfg, row["close"], ts, signal.reason)
                elif signal.action in (Action.LONG, Action.SHORT) and pos is None:
                    side = "long" if signal.action == Action.LONG else "short"
                    rvol = self.risk.realized_vol_for(df["close"], i, cfg)
                    sizing = self.risk.approve_entry(
                        symbol, side, self.acct.equity(marks), row["close"], row["atr"],
                        cfg.asset_class, self.acct.positions, strategy.stop_atr_mult,
                        realized_vol=rvol)
                    if sizing is not None:
                        fp = fill_price(row["close"], buying=side == "long",
                                        slippage_pct=cfg.slippage_pct)
                        self.acct.open_position(
                            symbol, side, sizing.qty, fp, sizing.stop_price,
                            fee=trade_fee(sizing.qty * fp, cfg.asset_class))

            equity = self.acct.equity(marks)
            self.peak_equity = max(self.peak_equity, equity)
            if halted_at is None and self.halt_enabled and ts >= sim_start and \
                    self.risk.drawdown_halted(equity, self.peak_equity):
                for sym_open in list(self.acct.positions):
                    close_position(sym_open, cfg_by_symbol[sym_open],
                                   marks[sym_open], ts, "circuit breaker")
                halted_at = ts

            if ts >= sim_start:
                curve.append((ts, self.acct.equity(marks)))
                open_pos = self.acct.positions.get(symbol)
                unrealized = open_pos.pnl_at(row["close"]) if open_pos is not None else 0.0
                sym_pnl[symbol].append((ts, realized[symbol] + unrealized))

        equity_curve = pd.Series(dict(curve)).sort_index()
        symbol_pnl = {s: pd.Series(dict(v)).sort_index() for s, v in sym_pnl.items()}
        return BacktestReport(trades, equity_curve, symbol_pnl, halted_at)


def load_instruments(md: MarketData, sim_start: datetime, has_keys: bool):
    instruments = []
    for cfg in INSTRUMENTS:
        if not cfg.enabled:
            print(f"  SKIP {cfg.symbol}: disabled in config.py")
            continue
        if cfg.asset_class == AssetClass.US_EQUITY and not has_keys:
            print(f"  SKIP {cfg.symbol}: equity data needs Alpaca API keys (see .env.example)")
            continue
        strategy = build_strategy(cfg)
        warmup_bars = strategy.min_bars + RISK.atr_period + 10
        warmup_days = calendar_days_for_bars(warmup_bars, cfg.timeframe_minutes, cfg.asset_class)
        start = sim_start - timedelta(days=warmup_days)
        df = md.get_bars(cfg.symbol, cfg.asset_class, cfg.timeframe_minutes, start)
        if len(df) < strategy.min_bars + 5:
            print(f"  SKIP {cfg.symbol}: only {len(df)} bars, need {strategy.min_bars + 5}")
            continue
        print(f"  {cfg.symbol}: {len(df)} bars ({cfg.timeframe_minutes}min)")
        instruments.append((cfg, strategy, strategy.prepare(df)))
    return instruments


def print_report(report: BacktestReport, instruments) -> bool:
    header = (f"{'Symbol':<9}{'Trades':>7}{'Win%':>7}{'AvgWin':>10}{'AvgLoss':>10}"
              f"{'PF':>7}{'MaxDD':>8}{'Sharpe':>8}{'P&L':>12}")
    print("\n" + header)
    print("-" * len(header))
    any_negative_sharpe = False
    for cfg, _, _ in instruments:
        s = report.instrument_stats(cfg.symbol)
        sharpe = report.instrument_sharpe(cfg.symbol)
        dd = report.instrument_max_drawdown(cfg.symbol)
        flag = ""
        if sharpe < 0:
            any_negative_sharpe = True
            flag = "  <-- NEGATIVE SHARPE: adjust parameters before trading this"
        pf = f"{s.profit_factor:.2f}" if s.profit_factor != float("inf") else "inf"
        print(f"{cfg.symbol:<9}{s.total_trades:>7}{s.win_rate * 100:>6.1f}%"
              f"{s.avg_win:>10.2f}{s.avg_loss:>10.2f}{pf:>7}{dd * 100:>7.2f}%"
              f"{sharpe:>8.2f}{s.total_pnl:>12.2f}{flag}")

    daily = report.equity_curve.resample("1D").last().dropna()
    total_return = daily.iloc[-1] / INITIAL_EQUITY - 1
    print("-" * len(header))
    print(f"{'PORTFOLIO':<9} return {total_return * 100:+.2f}%"
          f"  maxDD {max_drawdown(daily) * 100:.2f}%"
          f"  Sharpe {sharpe_ratio(daily.pct_change()):.2f}"
          f"  final equity {daily.iloc[-1]:,.0f}")
    if report.halted_at is not None:
        print(f"CIRCUIT BREAKER tripped at {report.halted_at} — trading halted for rest of test")
    return any_negative_sharpe


def plot_report(report: BacktestReport, path: str = "backtest_results.png") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    daily = report.equity_curve.resample("1D").last().dropna()
    ax1.plot(daily.index, daily.values)
    ax1.set_title("Portfolio equity")
    ax1.grid(True, alpha=0.3)
    for symbol, series in report.symbol_pnl.items():
        d = series.resample("1D").last().dropna()
        ax2.plot(d.index, d.values, label=symbol)
    ax2.set_title("Cumulative P&L per instrument")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Equity curve saved to {path}")


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY") or None
    secret = os.getenv("ALPACA_SECRET_KEY") or None
    if not api_key:
        print("No Alpaca API keys found: running crypto-only backtest.")
        print("For the full 5-instrument test, copy .env.example to .env and add paper keys.\n")

    sim_start = datetime.now(timezone.utc) - timedelta(days=BACKTEST_MONTHS * 30)
    print(f"Backtest window: {sim_start.date()} to today. Loading data...")
    md = MarketData(api_key, secret, feed=MARKET_DATA_FEED)
    instruments = load_instruments(md, sim_start, has_keys=bool(api_key))
    if not instruments:
        print("Nothing to backtest.")
        return 1

    report = Backtester().run(instruments, pd.Timestamp(sim_start))
    any_negative = print_report(report, instruments)
    plot_report(report)
    return 2 if any_negative else 0


if __name__ == "__main__":
    sys.exit(main())
