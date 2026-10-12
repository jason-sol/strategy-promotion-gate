import math

import pandas as pd
import pytest

from bot.indicators import realized_vol_at
from bot.risk_manager import RiskManager, realized_risk_pct, within_risk_budget
from bot.strategies.base import PositionState
from config import AssetClass, InstrumentConfig, RiskConfig, StrategyName


def make_position(symbol, side="long"):
    return PositionState(
        symbol=symbol, side=side, qty=10, entry_price=100.0,
        stop_price=99.0, high_water_mark=100.0,
    )


@pytest.fixture
def rm():
    return RiskManager(RiskConfig())


class TestPositionSizing:
    def test_one_atr_move_equals_one_percent_equity(self, rm):
        res = rm.position_size(equity=100_000, price=100.0, atr_value=2.0,
                               side="long", asset_class=AssetClass.US_EQUITY)
        # $1000 risk / $2 stop distance = 500 shares
        assert res.qty == 500
        assert res.stop_price == pytest.approx(98.0)

    def test_volatile_instrument_gets_smaller_position(self, rm):
        quiet = rm.position_size(100_000, 100.0, 2.0, "long", AssetClass.US_EQUITY)
        volatile = rm.position_size(100_000, 100.0, 10.0, "long", AssetClass.US_EQUITY)
        assert volatile.qty < quiet.qty
        assert volatile.qty == 100

    def test_notional_capped_at_pct_of_equity(self, rm):
        res = rm.position_size(100_000, 400.0, 0.5, "long", AssetClass.US_EQUITY)
        assert res.qty * 400.0 <= 95_000

    def test_short_stop_above_entry(self, rm):
        res = rm.position_size(100_000, 100.0, 2.0, "short", AssetClass.US_EQUITY)
        assert res.stop_price == pytest.approx(102.0)

    def test_equity_qty_is_whole_shares(self, rm):
        res = rm.position_size(100_000, 333.0, 3.7, "long", AssetClass.US_EQUITY)
        assert res.qty == int(res.qty)

    def test_crypto_qty_is_fractional(self, rm):
        res = rm.position_size(100_000, 60_000.0, 1_500.0, "long", AssetClass.CRYPTO)
        assert res.qty == pytest.approx(0.6667, abs=1e-4)

    def test_zero_atr_yields_zero_qty(self, rm):
        assert rm.position_size(100_000, 100.0, 0.0, "long", AssetClass.US_EQUITY).qty == 0

    def test_wider_stop_mult_shrinks_position_same_dollar_risk(self, rm):
        res = rm.position_size(100_000, 100.0, 2.0, "long", AssetClass.US_EQUITY,
                               stop_atr_mult=2.0)
        # $1000 risk / (2 x $2 ATR) = 250 shares, stop 2 ATR below entry
        assert res.qty == 250
        assert res.stop_price == pytest.approx(96.0)


class TestVolTargetScalar:
    def test_scalar_above_one_when_realized_below_target(self, rm):
        # target 0.15, realized 0.12 -> 1.25
        assert rm.vol_target_scalar(0.12) == pytest.approx(1.25)

    def test_scalar_below_one_when_realized_above_target(self, rm):
        # target 0.15, realized 0.30 -> 0.5 (floor)
        assert rm.vol_target_scalar(0.30) < 1.0

    def test_clipped_at_upper_bound(self, rm):
        assert rm.vol_target_scalar(0.001) == pytest.approx(1.5)

    def test_clipped_at_lower_bound(self, rm):
        assert rm.vol_target_scalar(10.0) == pytest.approx(0.5)

    def test_neutral_on_nan(self, rm):
        assert rm.vol_target_scalar(float("nan")) == 1.0

    def test_neutral_on_zero(self, rm):
        assert rm.vol_target_scalar(0.0) == 1.0

    def test_neutral_on_negative(self, rm):
        assert rm.vol_target_scalar(-0.2) == 1.0

    def test_neutral_on_none(self, rm):
        assert rm.vol_target_scalar(None) == 1.0


def make_cfg(vol_target: bool):
    params = {"vol_target": True} if vol_target else {}
    return InstrumentConfig(
        symbol="SPY", asset_class=AssetClass.US_EQUITY,
        strategy=StrategyName.RSI_MEAN_REVERSION, timeframe_minutes=1440,
        params=params,
    )


class TestRealizedVolFor:
    CLOSE = pd.Series([100.0 + (i % 5) for i in range(40)], dtype=float)

    def test_disabled_returns_none(self, rm):
        assert rm.realized_vol_for(self.CLOSE, 30, make_cfg(vol_target=False)) is None

    def test_enabled_matches_indicator_helper(self, rm):
        expected = realized_vol_at(self.CLOSE, 30, RiskConfig().vol_lookback)
        assert rm.realized_vol_for(self.CLOSE, 30, make_cfg(vol_target=True)) == pytest.approx(expected)


class TestPositionSizingWithVolTarget:
    REP_CASES = [
        (100_000, 100.0, 2.0, "long", AssetClass.US_EQUITY, None),
        (100_000, 100.0, 10.0, "long", AssetClass.US_EQUITY, None),
        (100_000, 400.0, 0.5, "long", AssetClass.US_EQUITY, None),
        (100_000, 60_000.0, 1_500.0, "long", AssetClass.CRYPTO, None),
        (100_000, 100.0, 2.0, "short", AssetClass.US_EQUITY, 2.0),
    ]

    @pytest.mark.parametrize("equity,price,atr,side,ac,mult", REP_CASES)
    def test_none_is_byte_identical_to_current(self, rm, equity, price, atr, side, ac, mult):
        base = rm.position_size(equity, price, atr, side, ac, mult)
        with_none = rm.position_size(equity, price, atr, side, ac, mult, realized_vol=None)
        assert with_none.qty == base.qty
        assert with_none.stop_price == base.stop_price

    def test_high_vol_shrinks_qty(self, rm):
        base = rm.position_size(100_000, 100.0, 2.0, "long", AssetClass.US_EQUITY)
        shrunk = rm.position_size(100_000, 100.0, 2.0, "long", AssetClass.US_EQUITY,
                                  realized_vol=0.30)
        assert shrunk.qty < base.qty
        assert shrunk.qty == math.floor(base.qty * 0.5)

    def test_low_vol_grows_qty(self, rm):
        base = rm.position_size(100_000, 100.0, 2.0, "long", AssetClass.US_EQUITY)
        grown = rm.position_size(100_000, 100.0, 2.0, "long", AssetClass.US_EQUITY,
                                 realized_vol=0.10)
        assert grown.qty > base.qty

    def test_scalar_applied_before_notional_cap(self, rm):
        # base qty 800 (< 950-share cap); scalar 1.5 -> 1200, capped back to 950.
        # Proves the multiply happens before the cap, not after.
        no_scalar = rm.position_size(100_000, 100.0, 1.25, "long", AssetClass.US_EQUITY)
        assert no_scalar.qty == 800
        scaled = rm.position_size(100_000, 100.0, 1.25, "long", AssetClass.US_EQUITY,
                                  realized_vol=0.10)  # scalar 1.5
        assert scaled.qty == 950


class TestApproveEntry:
    def test_blocked_by_correlation_returns_none(self, rm):
        positions = {"SPY": make_position("SPY"), "QQQ": make_position("QQQ")}
        assert rm.approve_entry("BTC/USD", "long", 100_000, 50_000.0, 1_000.0,
                                AssetClass.CRYPTO, positions) is None

    def test_zero_size_returns_none(self, rm):
        assert rm.approve_entry("SPY", "long", 100_000, 100.0, 0.0,
                                AssetClass.US_EQUITY, {}) is None

    def test_approved_returns_sizing(self, rm):
        res = rm.approve_entry("SPY", "long", 100_000, 100.0, 2.0,
                               AssetClass.US_EQUITY, {})
        assert res is not None and res.qty == 500


class TestCorrelationFilter:
    def test_blocks_btc_long_when_spy_and_qqq_long(self, rm):
        positions = {"SPY": make_position("SPY"), "QQQ": make_position("QQQ")}
        assert rm.correlation_blocked("BTC/USD", "long", positions)

    def test_allows_btc_long_when_only_spy_long(self, rm):
        positions = {"SPY": make_position("SPY")}
        assert not rm.correlation_blocked("BTC/USD", "long", positions)

    def test_allows_btc_long_when_qqq_short(self, rm):
        positions = {"SPY": make_position("SPY"), "QQQ": make_position("QQQ", side="short")}
        assert not rm.correlation_blocked("BTC/USD", "long", positions)

    def test_does_not_block_other_symbols(self, rm):
        positions = {"SPY": make_position("SPY"), "QQQ": make_position("QQQ")}
        assert not rm.correlation_blocked("GLD", "long", positions)


class TestDrawdownCircuitBreaker:
    def test_halts_beyond_10_percent_drawdown(self, rm):
        assert rm.drawdown_halted(equity=89_999, peak_equity=100_000)

    def test_no_halt_within_10_percent(self, rm):
        assert not rm.drawdown_halted(equity=90_001, peak_equity=100_000)


class TestRealizedRiskPct:
    def test_one_atr_stop_risks_one_percent(self):
        assert realized_risk_pct(qty=500, price=100.0, stop_price=98.0,
                                 equity=100_000) == pytest.approx(0.01)

    def test_short_stop_above_entry_risks_the_same(self):
        assert realized_risk_pct(qty=500, price=100.0, stop_price=102.0,
                                 equity=100_000) == pytest.approx(0.01)

    def test_no_position_risks_nothing(self):
        assert realized_risk_pct(0, 100.0, 98.0, 100_000) == 0.0

    def test_non_positive_equity_risks_nothing(self):
        assert realized_risk_pct(500, 100.0, 98.0, 0) == 0.0


class TestRiskBudget:
    def test_budget_is_the_flat_fraction_without_vol_targeting(self, rm):
        assert rm.risk_budget_pct(None) == pytest.approx(RiskConfig().risk_per_trade)

    def test_budget_carries_the_vol_scalar(self, rm):
        # realized 0.12 against a 0.15 target scales the budget by 1.25
        assert rm.risk_budget_pct(0.12) == pytest.approx(0.0125)

    def test_budget_at_the_scalar_cap_is_the_hard_ceiling(self, rm):
        cfg = RiskConfig()
        assert rm.risk_budget_pct(0.05) == pytest.approx(
            cfg.risk_per_trade * cfg.vol_scalar_max)

    def test_sized_entry_is_within_its_budget(self, rm):
        sizing = rm.position_size(100_000, 100.0, 2.0, "long", AssetClass.US_EQUITY)
        check = rm.check_risk_budget(sizing.qty, 100.0, sizing.stop_price, 100_000)
        assert check.realized_pct == pytest.approx(0.01)
        assert check.budget_pct == pytest.approx(0.01)
        assert check.within_budget

    def test_vol_scaled_entry_risks_more_than_the_flat_fraction(self, rm):
        sizing = rm.position_size(100_000, 100.0, 2.0, "long", AssetClass.US_EQUITY,
                                  realized_vol=0.05)  # scalar clipped at 1.5
        check = rm.check_risk_budget(sizing.qty, 100.0, sizing.stop_price, 100_000,
                                     realized_vol=0.05)
        assert check.realized_pct == pytest.approx(0.015)
        assert check.realized_pct > RiskConfig().risk_per_trade
        assert check.budget_pct == pytest.approx(0.015)
        assert check.within_budget

    def test_double_the_sized_qty_is_over_budget(self, rm):
        sizing = rm.position_size(100_000, 100.0, 2.0, "long", AssetClass.US_EQUITY,
                                  realized_vol=0.05)
        check = rm.check_risk_budget(sizing.qty * 2, 100.0, sizing.stop_price,
                                     100_000, realized_vol=0.05)
        assert check.realized_pct == pytest.approx(0.03)
        assert check.budget_pct == pytest.approx(0.015)
        assert not check.within_budget
        assert "over budget" in check.note

    def test_rounding_noise_is_not_over_budget(self):
        assert within_risk_budget(0.01 + 1e-12, 0.01)

    def test_a_real_overshoot_is_over_budget(self):
        assert not within_risk_budget(0.0101, 0.01)
