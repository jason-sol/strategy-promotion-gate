import pytest

from bot.strategies import STRATEGY_BUILDERS, build_strategy
from bot.strategies.base import (
    Action,
    PositionState,
    effective_stop,
    resolve_stop_and_hwm,
    trailing_stop_hit,
    trailing_stop_level,
)
from bot.strategies.mean_reversion import MeanReversion
from bot.strategies.momentum_breakout import MomentumBreakout
from bot.strategies.trend_following import TrendFollowing
from config import AssetClass, InstrumentConfig, StrategyName, enabled_instruments


def long_position(entry=100.0, hwm=None, stop=99.0):
    return PositionState(
        symbol="X", side="long", qty=10, entry_price=entry,
        stop_price=stop, high_water_mark=hwm if hwm is not None else entry,
    )


def short_position(entry=100.0, hwm=None, stop=101.0):
    return PositionState(
        symbol="X", side="short", qty=10, entry_price=entry,
        stop_price=stop, high_water_mark=hwm if hwm is not None else entry,
    )


def past_warmup(pattern, bars=24, level=100.0):
    """Flat bars ahead of `pattern` so the pattern's bars clear TrendFollowing's
    warm-up (4x the slow EMA span; 24 at slow_period=6). A no-op for the
    indicators: ewm(adjust=False) seeds on the first close, and the fixture's
    fixed high/low spread pins ATR whatever the closes do.
    """
    return [level] * bars + list(pattern)


class TestTrailingStopHelper:
    def test_long_triggers_below_hwm_minus_mult_atr(self):
        pos = long_position(entry=100.0, hwm=110.0)
        assert trailing_stop_hit(pos, close=105.9, atr=2.0, mult=2.0)
        assert not trailing_stop_hit(pos, close=106.1, atr=2.0, mult=2.0)

    def test_short_triggers_above_hwm_plus_mult_atr(self):
        pos = short_position(entry=100.0, hwm=90.0)
        assert trailing_stop_hit(pos, close=96.1, atr=2.0, mult=3.0)
        assert not trailing_stop_hit(pos, close=95.9, atr=2.0, mult=3.0)

    def test_pnl_at_signs(self):
        assert long_position(entry=100.0).pnl_at(105.0) == pytest.approx(50.0)
        assert short_position(entry=100.0).pnl_at(105.0) == pytest.approx(-50.0)

    def test_hwm_update_direction(self):
        pos = long_position(hwm=105.0)
        pos.update_high_water_mark(110.0)
        pos.update_high_water_mark(100.0)
        assert pos.high_water_mark == 110.0
        pos = short_position(hwm=95.0)
        pos.update_high_water_mark(90.0)
        pos.update_high_water_mark(99.0)
        assert pos.high_water_mark == 90.0


class TestStopHelpers:
    def test_trailing_stop_level_long(self):
        pos = long_position(hwm=120.0)
        assert trailing_stop_level(pos, atr=5.0, mult=3.0) == pytest.approx(105.0)

    def test_trailing_stop_level_short(self):
        pos = short_position(hwm=80.0)
        assert trailing_stop_level(pos, atr=5.0, mult=3.0) == pytest.approx(95.0)

    def test_effective_stop_long_trail_binds(self):
        # trail 120-3*5=105 above entry stop 95 -> trail binds
        pos = long_position(entry=100.0, hwm=120.0, stop=95.0)
        assert effective_stop(pos, atr=5.0, mult=3.0) == pytest.approx(105.0)

    def test_effective_stop_long_entry_binds(self):
        # trail 110-3*5=95 below entry stop 108 -> entry stop binds
        pos = long_position(entry=100.0, hwm=110.0, stop=108.0)
        assert effective_stop(pos, atr=5.0, mult=3.0) == pytest.approx(108.0)

    def test_effective_stop_short_trail_binds(self):
        # trail 80+3*5=95 below entry stop 105 -> trail binds (min)
        pos = short_position(entry=100.0, hwm=80.0, stop=105.0)
        assert effective_stop(pos, atr=5.0, mult=3.0) == pytest.approx(95.0)

    def test_effective_stop_short_entry_binds(self):
        # trail 98+3*5=113 above entry stop 101 -> entry stop binds (min)
        pos = short_position(entry=100.0, hwm=98.0, stop=101.0)
        assert effective_stop(pos, atr=5.0, mult=3.0) == pytest.approx(101.0)

    def test_resolve_meta_present(self):
        pos = long_position(entry=100.0)
        assert resolve_stop_and_hwm(
            pos, {"stop_price": 95.0, "high_water_mark": 120.0}) == (95.0, 120.0)

    def test_resolve_none_long_fallback(self):
        assert resolve_stop_and_hwm(long_position(entry=100.0), None) == pytest.approx((99.0, 100.0))

    def test_resolve_none_short_fallback(self):
        assert resolve_stop_and_hwm(short_position(entry=100.0), None) == pytest.approx((101.0, 100.0))

    def test_trailing_stop_hit_boundary_long(self):
        pos = long_position(hwm=120.0)  # level = 120 - 3*5 = 105
        assert trailing_stop_hit(pos, close=105.0, atr=5.0, mult=3.0)       # exactly at level -> hit
        assert not trailing_stop_hit(pos, close=105.01, atr=5.0, mult=3.0)  # just above -> not hit

    def test_trailing_stop_hit_boundary_short(self):
        pos = short_position(hwm=80.0)  # level = 80 + 3*5 = 95
        assert trailing_stop_hit(pos, close=95.0, atr=5.0, mult=3.0)        # exactly at level -> hit
        assert not trailing_stop_hit(pos, close=94.99, atr=5.0, mult=3.0)   # just below -> not hit


class TestMeanReversion:
    def strat(self, entry_z=1.5, allow_short=True):
        return MeanReversion(sma_period=20, entry_z=entry_z, allow_short=allow_short, atr_period=14)

    def test_long_entry_on_deep_dip(self, make_bars):
        df = self.strat().prepare(make_bars([100.0] * 24 + [90.0]))
        sig = self.strat().signal(df, len(df) - 1, None)
        assert sig.action == Action.LONG

    def test_short_entry_on_spike(self, make_bars):
        df = self.strat().prepare(make_bars([100.0] * 24 + [110.0]))
        sig = self.strat().signal(df, len(df) - 1, None)
        assert sig.action == Action.SHORT

    def test_no_short_when_disallowed(self, make_bars):
        strat = self.strat(allow_short=False)
        df = strat.prepare(make_bars([100.0] * 24 + [110.0]))
        assert strat.signal(df, len(df) - 1, None).action == Action.HOLD

    def test_small_deviation_holds(self, make_bars):
        df = self.strat().prepare(make_bars([100.0, 101.0] * 12 + [100.5]))
        assert self.strat().signal(df, len(df) - 1, None).action == Action.HOLD

    def test_wider_threshold_holds_where_tighter_enters(self, make_bars):
        bars = make_bars([100.0, 101.0] * 12 + [98.9])
        tight = self.strat(entry_z=1.5)
        wide = self.strat(entry_z=5.0)
        assert tight.signal(tight.prepare(bars), len(bars) - 1, None).action == Action.LONG
        assert wide.signal(wide.prepare(bars), len(bars) - 1, None).action == Action.HOLD

    def test_exit_long_when_price_reverts_to_mean(self, make_bars):
        df = self.strat().prepare(make_bars([100.0] * 24 + [100.0]))
        sig = self.strat().signal(df, len(df) - 1, long_position(entry=95.0))
        assert sig.action == Action.EXIT

    def test_exit_short_when_price_reverts_to_mean(self, make_bars):
        df = self.strat().prepare(make_bars([100.0] * 24 + [99.9]))
        sig = self.strat().signal(df, len(df) - 1, short_position(entry=105.0))
        assert sig.action == Action.EXIT

    def test_holds_long_before_reversion_completes(self, make_bars):
        df = self.strat().prepare(make_bars([100.0] * 24 + [96.0]))
        sig = self.strat().signal(df, len(df) - 1, long_position(entry=95.0))
        assert sig.action == Action.HOLD

    def test_insufficient_history_holds(self, make_bars):
        df = self.strat().prepare(make_bars([100.0] * 5 + [50.0]))
        assert self.strat().signal(df, len(df) - 1, None).action == Action.HOLD


class TestMomentumBreakout:
    def strat(self):
        return MomentumBreakout(channel_period=20, volume_mult=1.5, trail_atr_mult=2.0, atr_period=14)

    def base_closes(self):
        # 30 bars oscillating 99-101, prior 20-bar high (of highs, spread=1) = 102
        return [100.0, 101.0, 99.0] * 10

    def test_breakout_with_volume_goes_long(self, make_bars):
        closes = self.base_closes() + [105.0]
        vols = [1000.0] * 30 + [2000.0]
        df = self.strat().prepare(make_bars(closes, vols))
        assert self.strat().signal(df, len(df) - 1, None).action == Action.LONG

    def test_breakout_without_volume_holds(self, make_bars):
        closes = self.base_closes() + [105.0]
        vols = [1000.0] * 30 + [1100.0]
        df = self.strat().prepare(make_bars(closes, vols))
        assert self.strat().signal(df, len(df) - 1, None).action == Action.HOLD

    def test_no_breakout_holds(self, make_bars):
        closes = self.base_closes() + [100.0]
        vols = [1000.0] * 30 + [2000.0]
        df = self.strat().prepare(make_bars(closes, vols))
        assert self.strat().signal(df, len(df) - 1, None).action == Action.HOLD

    def test_breakdown_never_shorts(self, make_bars):
        closes = self.base_closes() + [90.0]
        vols = [1000.0] * 30 + [3000.0]
        df = self.strat().prepare(make_bars(closes, vols))
        assert self.strat().signal(df, len(df) - 1, None).action == Action.HOLD

    def test_exit_on_channel_breakdown(self, make_bars):
        closes = self.base_closes() + [90.0]
        df = self.strat().prepare(make_bars(closes))
        sig = self.strat().signal(df, len(df) - 1, long_position(entry=102.0, hwm=105.0))
        assert sig.action == Action.EXIT

    def test_exit_on_trailing_stop(self, make_bars):
        # ATR ~2 (constant spread=1 range); hwm 110, close 100 -> drop 10 > 2*ATR
        closes = self.base_closes() + [100.0]
        df = self.strat().prepare(make_bars(closes))
        sig = self.strat().signal(df, len(df) - 1, long_position(entry=105.0, hwm=110.0))
        assert sig.action == Action.EXIT

    def test_holds_position_inside_channel_and_trail(self, make_bars):
        closes = self.base_closes() + [101.0]
        df = self.strat().prepare(make_bars(closes))
        sig = self.strat().signal(df, len(df) - 1, long_position(entry=100.0, hwm=101.5))
        assert sig.action == Action.HOLD

    def test_hard_stop_distance_matches_trail(self):
        assert self.strat().stop_atr_mult == 2.0


class TestTrendFollowing:
    def strat(self, allow_short=False):
        return TrendFollowing(
            fast_period=2, slow_period=6, trail_atr_mult=3.0, atr_period=14, allow_short=allow_short
        )

    def test_golden_cross_goes_long(self, make_bars):
        # Downtrend then sharp rally: fast EMA crosses above slow on the last bar
        closes = past_warmup([100.0 - i for i in range(10)] + [95.0, 100.0, 106.0])
        df = self.strat().prepare(make_bars(closes))
        actions = [self.strat().signal(df, i, None).action for i in range(8, len(df))]
        assert Action.LONG in actions

    def test_no_entry_without_cross(self, make_bars):
        # Steady uptrend: fast already above slow, no cross at the last bar
        closes = [100.0 + i for i in range(30)]
        df = self.strat().prepare(make_bars(closes))
        assert self.strat().signal(df, len(df) - 1, None).action == Action.HOLD

    def test_death_cross_exits_long(self, make_bars):
        closes = past_warmup([100.0 + i for i in range(10)] + [108.0, 102.0, 95.0])
        df = self.strat().prepare(make_bars(closes))
        actions = [
            self.strat().signal(df, i, long_position(entry=100.0, hwm=110.0)).action
            for i in range(8, len(df))
        ]
        assert Action.EXIT in actions

    def test_trailing_stop_exits_long(self, make_bars):
        # Uptrend keeps fast > slow, then a plunge deeper than 3*ATR from hwm
        closes = past_warmup([100.0 + i for i in range(20)] + [119.5, 119.0, 105.0])
        df = self.strat().prepare(make_bars(closes))
        sig = self.strat().signal(df, len(df) - 1, long_position(entry=110.0, hwm=119.0))
        assert sig.action == Action.EXIT

    def test_death_cross_when_flat_holds_by_default(self, make_bars):
        closes = past_warmup([100.0 + i for i in range(10)] + [108.0, 102.0, 95.0])
        df = self.strat().prepare(make_bars(closes))
        actions = [self.strat().signal(df, i, None).action for i in range(8, len(df))]
        assert Action.SHORT not in actions

    def test_death_cross_when_flat_shorts_if_allowed(self, make_bars):
        strat = self.strat(allow_short=True)
        closes = past_warmup([100.0 + i for i in range(10)] + [108.0, 102.0, 95.0])
        df = strat.prepare(make_bars(closes))
        actions = [strat.signal(df, i, None).action for i in range(8, len(df))]
        assert Action.SHORT in actions

    def test_hard_stop_distance_matches_trail(self):
        assert self.strat().stop_atr_mult == 3.0


class TestTrendFollowingVolumeGate:
    # Volume confirmation on entries only: a cross on thin volume is the
    # fakeout this is meant to stand aside from. Real volume series here rather
    # than a single-cell override, so prepare() and signal() are proved together.
    def gated(self, volume_gate_mult=1.5, lookback=3):
        return TrendFollowing(fast_period=2, slow_period=6, trail_atr_mult=3.0,
                              atr_period=14, allow_short=True,
                              volume_gate_mult=volume_gate_mult,
                              volume_gate_lookback=lookback)

    def _closes(self):
        return past_warmup([100.0 - i for i in range(10)] + [95.0, 100.0, 106.0])

    def _entry_bar(self, make_bars, closes):
        ref = TrendFollowing(fast_period=2, slow_period=6, trail_atr_mult=3.0,
                             atr_period=14, allow_short=True)
        rdf = ref.prepare(make_bars(closes))
        return next(i for i in range(len(rdf)) if ref.signal(rdf, i, None).action == Action.LONG)

    def _volumes(self, closes, entry_i, entry_volume):
        vols = [1000.0] * len(closes)
        vols[entry_i] = entry_volume
        return vols

    def test_entry_suppressed_when_volume_does_not_confirm(self, make_bars):
        closes = self._closes()
        i = self._entry_bar(make_bars, closes)
        s = self.gated()
        df = s.prepare(make_bars(closes, self._volumes(closes, i, 500.0)))
        assert s.signal(df, i, None).action == Action.HOLD

    def test_entry_allowed_when_volume_confirms_same_trigger(self, make_bars):
        closes = self._closes()
        i = self._entry_bar(make_bars, closes)
        s = self.gated()
        df = s.prepare(make_bars(closes, self._volumes(closes, i, 2000.0)))
        assert s.signal(df, i, None).action == Action.LONG

    def test_gate_is_off_by_default(self, make_bars):
        closes = self._closes()
        i = self._entry_bar(make_bars, closes)
        s = TrendFollowing(fast_period=2, slow_period=6, trail_atr_mult=3.0,
                           atr_period=14, allow_short=True)
        df = s.prepare(make_bars(closes, self._volumes(closes, i, 1.0)))
        assert s.signal(df, i, None).action == Action.LONG

    def test_warmup_nan_average_does_not_suppress(self, make_bars):
        closes = self._closes()
        i = self._entry_bar(make_bars, closes)
        s = self.gated()
        df = s.prepare(make_bars(closes, self._volumes(closes, i, 1.0)))
        df.loc[i, "volume_avg"] = float("nan")
        assert s.signal(df, i, None).action == Action.LONG

    def test_exit_ignores_the_gate(self, make_bars):
        # A death cross on thin volume must still close the position: the gate
        # stands aside from NEW risk, it never traps open risk.
        closes = past_warmup([100.0 + i for i in range(10)] + [110.0, 104.0, 98.0])
        s = self.gated()
        vols = [1000.0] * len(closes)
        vols[-1] = 1.0
        df = s.prepare(make_bars(closes, vols))
        assert s.signal(df, len(closes) - 1, long_position()).action == Action.EXIT

    def test_min_bars_covers_the_volume_lookback(self):
        # The average is shifted a bar, so the lookback needs lookback+1 bars.
        s = self.gated(lookback=100)
        assert s.min_bars >= 101


class TestTrendFollowingVolGate:
    # Discrete stand-aside on entries only; vol_pct is overridden at the entry
    # bar (documented single-cell technique) to isolate the gate from price.
    def gated(self, vol_gate_percentile=0.8):
        return TrendFollowing(fast_period=2, slow_period=6, trail_atr_mult=3.0,
                              atr_period=14, allow_short=True,
                              vol_gate_percentile=vol_gate_percentile,
                              vol_gate_lookback=3, vol_gate_rank_window=5)

    def _entry_bar(self, make_bars, closes):
        ref = TrendFollowing(fast_period=2, slow_period=6, trail_atr_mult=3.0,
                             atr_period=14, allow_short=True)
        rdf = ref.prepare(make_bars(closes))
        return next(i for i in range(len(rdf)) if ref.signal(rdf, i, None).action == Action.LONG)

    def test_entry_suppressed_when_vol_pct_above_threshold(self, make_bars):
        closes = past_warmup([100.0 - i for i in range(10)] + [95.0, 100.0, 106.0])
        s = self.gated()
        df = s.prepare(make_bars(closes))
        i = self._entry_bar(make_bars, closes)
        df.loc[i, "vol_pct"] = 0.95
        assert s.signal(df, i, None).action == Action.HOLD

    def test_entry_allowed_in_calm_regime_same_trigger(self, make_bars):
        closes = past_warmup([100.0 - i for i in range(10)] + [95.0, 100.0, 106.0])
        s = self.gated()
        df = s.prepare(make_bars(closes))
        i = self._entry_bar(make_bars, closes)
        df.loc[i, "vol_pct"] = 0.5
        assert s.signal(df, i, None).action == Action.LONG

    def test_nan_vol_pct_does_not_suppress(self, make_bars):
        closes = past_warmup([100.0 - i for i in range(10)] + [95.0, 100.0, 106.0])
        s = self.gated()
        df = s.prepare(make_bars(closes))
        i = self._entry_bar(make_bars, closes)
        df.loc[i, "vol_pct"] = float("nan")
        assert s.signal(df, i, None).action == Action.LONG

    def test_short_entry_suppressed_when_vol_pct_above_threshold(self, make_bars):
        # Gate sits above BOTH entry branches: the short leg is blocked too.
        closes = past_warmup([100.0 + i for i in range(10)] + [108.0, 102.0, 95.0])
        s = self.gated()
        df = s.prepare(make_bars(closes))
        ref = TrendFollowing(fast_period=2, slow_period=6, trail_atr_mult=3.0,
                             atr_period=14, allow_short=True)
        rdf = ref.prepare(make_bars(closes))
        i = next(j for j in range(len(rdf)) if ref.signal(rdf, j, None).action == Action.SHORT)
        df.loc[i, "vol_pct"] = 0.95
        assert s.signal(df, i, None).action == Action.HOLD

    def test_exit_fires_during_high_vol(self, make_bars):
        # Trailing-stop exit must ignore the gate entirely.
        closes = past_warmup([100.0 + i for i in range(20)] + [119.5, 119.0, 105.0])
        s = self.gated()
        df = s.prepare(make_bars(closes))
        i = len(df) - 1
        df.loc[i, "vol_pct"] = 0.99
        sig = s.signal(df, i, long_position(entry=110.0, hwm=119.0))
        assert sig.action == Action.EXIT

    def test_gate_none_adds_no_columns_and_matches_ungated(self, make_bars):
        closes = past_warmup([100.0 - i for i in range(10)] + [95.0, 100.0, 106.0])
        plain = TrendFollowing(fast_period=2, slow_period=6, trail_atr_mult=3.0,
                               atr_period=14, allow_short=True)
        df = plain.prepare(make_bars(closes))
        assert "vol_pct" not in df.columns
        assert "realized_vol" not in df.columns
        i = self._entry_bar(make_bars, closes)
        assert plain.signal(df, i, None).action == Action.LONG


class TestWarmupOutlivesTheIndicatorSeed:
    """Live only ever sees min_bars of history, and a recursive smoother keeps
    a real share of its seed until several multiples of its span have passed.
    Warm up for less and the same bar reads differently live than in the
    backtest that validated it — worst on ATR, which sets both the stop
    distance and the position size.
    """

    # Periods of the recursive smoothers a strategy can carry: EMA spans and
    # the Wilder-smoothed ATR and RSI. Rolling windows (SMA, Donchian channel,
    # momentum lookbacks) are deliberately absent — a finite window is either
    # exactly warm or NaN, never seeded.
    RECURSIVE_PERIOD_ATTRS = ("fast_period", "slow_period", "atr_period", "rsi_period")
    SEED_FREE_SPANS = 4

    @pytest.mark.parametrize("cfg", enabled_instruments(), ids=lambda c: c.symbol)
    def test_enabled_instrument_warms_every_recursive_indicator(self, cfg):
        strategy = build_strategy(cfg)
        spans = {attr: getattr(strategy, attr) for attr in self.RECURSIVE_PERIOD_ATTRS
                 if hasattr(strategy, attr)}
        assert spans, (
            f"{type(strategy).__name__} exposes none of {self.RECURSIVE_PERIOD_ATTRS}: "
            "either it runs no recursive indicator, or this guard has to learn "
            "the name of the period it does run")
        attr, longest = max(spans.items(), key=lambda item: item[1])
        assert strategy.min_bars >= self.SEED_FREE_SPANS * longest, (
            f"{cfg.symbol}: min_bars {strategy.min_bars} warms {attr}={longest} for "
            f"{strategy.min_bars / longest:.1f} spans, under the "
            f"{self.SEED_FREE_SPANS} this floor requires (EMA sheds its seed by then; "
            f"Wilder is slower — strategies with Wilder spans should warm longer, "
            f"as TurnOfMonth does at 7x)")


class TestStrategyRegistry:
    def test_every_strategy_name_has_a_builder(self):
        assert set(STRATEGY_BUILDERS) == set(StrategyName)

    def test_build_strategy_dispatches_via_registry(self):
        cfg = InstrumentConfig(
            symbol="TEST", asset_class=AssetClass.US_EQUITY,
            strategy=StrategyName.TREND_FOLLOWING, timeframe_minutes=1440,
            params={"fast_period": 20, "slow_period": 100,
                    "trail_atr_mult": 3.0, "allow_short": False},
        )
        assert isinstance(build_strategy(cfg), TrendFollowing)

    def test_unregistered_strategy_raises(self, monkeypatch):
        monkeypatch.delitem(STRATEGY_BUILDERS, StrategyName.TREND_FOLLOWING)
        cfg = InstrumentConfig(
            symbol="TEST", asset_class=AssetClass.US_EQUITY,
            strategy=StrategyName.TREND_FOLLOWING, timeframe_minutes=1440,
            params={},
        )
        with pytest.raises(ValueError, match="unknown strategy"):
            build_strategy(cfg)
