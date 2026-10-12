import math

import pandas as pd
import pytest

from bot.backtest import Backtester
from bot.metrics import max_drawdown, sharpe_ratio, summarize_trades
from bot.strategies import build_strategy
from config import AssetClass, InstrumentConfig, StrategyName


def spy_config(vol_target=False):
    return InstrumentConfig(
        symbol="SPY", asset_class=AssetClass.US_EQUITY,
        strategy=StrategyName.MEAN_REVERSION, timeframe_minutes=15,
        params={"sma_period": 20, "entry_z": 1.5, "allow_short": True,
                "vol_target": vol_target},
    )


def with_timestamps(df):
    df.index = pd.date_range("2026-01-05 14:30", periods=len(df), freq="15min", tz="UTC")
    return df


def run_single(closes, make_bars, vol_target=False):
    cfg = spy_config(vol_target=vol_target)
    strategy = build_strategy(cfg)
    df = strategy.prepare(with_timestamps(make_bars(closes)))
    bt = Backtester(initial_equity=100_000)
    report = bt.run([(cfg, strategy, df)], sim_start=df.index[0])
    return bt, report


class TestBacktesterRoundTrip:
    def test_dip_and_revert_produces_one_winning_trade(self, make_bars):
        closes = [100.0] * 30 + [90.0] + [100.0] * 5
        bt, report = run_single(closes, make_bars)
        assert len(report.trades) == 1
        trade = report.trades[0]
        assert trade.direction == "long"
        assert trade.pnl > 0
        assert trade.entry_price == pytest.approx(90.0 * 1.0005)
        assert trade.exit_price == pytest.approx(100.0 * 0.9995)
        assert bt.acct.cash == pytest.approx(100_000 + trade.pnl)
        assert report.equity_curve.iloc[-1] == pytest.approx(100_000 + trade.pnl)

    def test_hard_stop_caps_loss_near_one_percent(self, make_bars):
        # Entry long ~90, then a crash bar whose low pierces the ATR stop
        closes = [100.0] * 30 + [90.0] + [80.0] + [80.0] * 3
        bt, report = run_single(closes, make_bars)
        stop_trades = [t for t in report.trades if t.reason == "hard stop"]
        assert len(stop_trades) == 1
        trade = stop_trades[0]
        assert trade.pnl < 0
        # 1% of 100k = $1000 risk; slippage makes it slightly worse but same order
        assert abs(trade.pnl) < 1_500
        # filled at the stop, not at the bar close of 80
        assert trade.exit_price > 80.0 * 1.01

    def test_no_trades_on_flat_series(self, make_bars):
        closes = [100.0] * 40
        _, report = run_single(closes, make_bars)
        assert report.trades == []


class TestVolTargetWiring:
    # A dip to 90 on a flat baseline: realized vol at entry (~0.35) sits well
    # above the 0.15 target, so the vol scalar pins to its 0.5 floor.
    DIP = [100.0] * 30 + [90.0] + [100.0] * 5

    def test_opt_in_halves_entry_qty_vs_opt_out(self, make_bars):
        # Same bars, same ATR sizing — the only difference is the vol scalar.
        _, off = run_single(self.DIP, make_bars, vol_target=False)
        _, on = run_single(self.DIP, make_bars, vol_target=True)
        assert off.trades and on.trades
        assert on.trades[0].qty < off.trades[0].qty
        assert on.trades[0].qty == pytest.approx(off.trades[0].qty * 0.5, abs=1)

    def test_high_vol_run_sizes_smaller_than_low_vol_run(self, make_bars):
        deep = [100.0] * 30 + [90.0] + [100.0] * 5   # ~10% dip, high realized vol
        shallow = [100.0] * 30 + [99.0] + [100.0] * 5  # ~1% dip, low realized vol
        _, high = run_single(deep, make_bars, vol_target=True)
        _, low = run_single(shallow, make_bars, vol_target=True)
        assert high.trades and low.trades
        assert high.trades[0].qty < low.trades[0].qty


class TestMetrics:
    def test_max_drawdown(self):
        equity = pd.Series([100.0, 110.0, 99.0, 105.0])
        assert max_drawdown(equity) == pytest.approx(0.1)

    def test_max_drawdown_monotonic_rise_is_zero(self):
        assert max_drawdown(pd.Series([100.0, 101.0, 102.0])) == 0.0

    def test_sharpe_positive_for_steady_gains(self):
        returns = pd.Series([0.01, 0.012, 0.009, 0.011])
        assert sharpe_ratio(returns) > 0

    def test_sharpe_zero_for_flat_returns(self):
        assert sharpe_ratio(pd.Series([0.0, 0.0, 0.0])) == 0.0

    def test_trade_summary(self):
        stats = summarize_trades([100.0, -50.0, 200.0, -50.0])
        assert stats.total_trades == 4
        assert stats.win_rate == pytest.approx(0.5)
        assert stats.avg_win == pytest.approx(150.0)
        assert stats.avg_loss == pytest.approx(-50.0)
        assert stats.profit_factor == pytest.approx(3.0)
        assert stats.total_pnl == pytest.approx(200.0)
