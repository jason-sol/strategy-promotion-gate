import math

import pytest

from bot.strategies import build_strategy
from bot.strategies.base import Action, PositionState
from bot.strategies.ts_momentum import TsMomentum
from config import AssetClass, InstrumentConfig, StrategyName


def strat(lookbacks=(5, 10), skip_days=3, atr_period=14, stop_atr_mult=3.0, **kw):
    # Small lookbacks keep fixtures short; the logic is period-agnostic.
    return TsMomentum(lookbacks=list(lookbacks), skip_days=skip_days,
                      atr_period=atr_period, stop_atr_mult=stop_atr_mult, **kw)


def long_position():
    return PositionState(symbol="X", side="long", qty=10, entry_price=100.0,
                         stop_price=90.0, high_water_mark=100.0)


class TestTsMomentum:
    def test_sustained_uptrend_scores_positive_and_goes_long(self, make_bars):
        closes = [100.0 + i for i in range(40)]
        s = strat()
        df = s.prepare(make_bars(closes))
        assert df["ts_score"].iloc[-1] == 1.0
        assert s.signal(df, len(df) - 1, None).action == Action.LONG

    def test_sustained_downtrend_no_entry_when_flat(self, make_bars):
        closes = [200.0 - i for i in range(40)]
        s = strat()
        df = s.prepare(make_bars(closes))
        assert df["ts_score"].iloc[-1] == -1.0
        assert s.signal(df, len(df) - 1, None).action == Action.HOLD

    def test_sustained_downtrend_exits_position(self, make_bars):
        closes = [200.0 - i for i in range(40)]
        s = strat()
        df = s.prepare(make_bars(closes))
        assert s.signal(df, len(df) - 1, long_position()).action == Action.EXIT

    def test_warmup_score_is_nan_and_holds(self, make_bars):
        # Fewer bars than max(lookback)+skip: the long lookback never warms.
        closes = [100.0 + i for i in range(10)]
        s = strat()
        df = s.prepare(make_bars(closes))
        assert math.isnan(df["ts_score"].iloc[-1])
        assert s.signal(df, len(df) - 1, None).action == Action.HOLD

    def test_disagreeing_lookbacks_average_to_net_flat_no_entry(self, make_bars):
        # Recent 5-bar window up (+1) but 10-bar window down (-1): net vote 0.
        s = strat(lookbacks=(5, 10), skip_days=1)
        closes = [120.0] * 22 + [115, 110, 105, 100, 90, 92, 94, 96, 98, 100, 100]
        df = s.prepare(make_bars(closes))
        score = df["ts_score"].iloc[-1]
        assert score == 0.0
        assert s.signal(df, len(df) - 1, None).action == Action.HOLD

    def test_net_flat_score_exits_held_position(self, make_bars):
        # score == 0 is the <= 0 exit boundary: a held long must exit on it.
        s = strat(lookbacks=(5, 10), skip_days=1)
        closes = [120.0] * 22 + [115, 110, 105, 100, 90, 92, 94, 96, 98, 100, 100]
        df = s.prepare(make_bars(closes))
        assert df["ts_score"].iloc[-1] == 0.0
        assert s.signal(df, len(df) - 1, long_position()).action == Action.EXIT

    def test_partial_vote_is_fractional(self, make_bars):
        # Two lookbacks up, one down -> score strictly between 0 and 1.
        s = strat(lookbacks=(5, 10, 15), skip_days=1)
        closes = [130.0] * 20 + [115, 105, 100, 97, 95, 93, 91, 90, 90, 90, 92, 94, 96, 98, 100, 100]
        df = s.prepare(make_bars(closes))
        score = df["ts_score"].iloc[-1]
        assert 0 < abs(score) < 1
        assert s.signal(df, len(df) - 1, None).action == Action.LONG

    def test_12_1_skip_ignores_rally_inside_skip_window(self, make_bars):
        # Prior window gently declining; a huge rally lives ONLY in the last
        # skip_days bars. Correct shift arithmetic must NOT see the rally.
        s = strat(lookbacks=(5,), skip_days=3)
        closes = [130.0 - i for i in range(31)] + [200.0, 200.0, 200.0]
        df = s.prepare(make_bars(closes))
        score = df["ts_score"].iloc[-1]
        assert score == -1.0  # measured window ends before the rally, and is down
        assert s.signal(df, len(df) - 1, None).action == Action.HOLD

    def test_hard_stop_distance_default(self):
        assert strat().stop_atr_mult == 3.0

    def test_hard_stop_distance_configurable(self):
        assert strat(stop_atr_mult=2.0).stop_atr_mult == 2.0

    def test_empty_lookbacks_fails_loudly(self, make_bars):
        # A misconfigured sweep must raise, not silently run the default ensemble.
        s = TsMomentum(lookbacks=[])
        with pytest.raises(ValueError):
            _ = s.min_bars


class TestTsMomentumVolGate:
    def gated(self, vol_gate_percentile=0.8):
        return strat(lookbacks=(5, 10), skip_days=3, vol_gate_percentile=vol_gate_percentile,
                     vol_gate_lookback=3, vol_gate_rank_window=5)

    def _entry_bar(self, make_bars, closes):
        ref = strat(lookbacks=(5, 10), skip_days=3)
        rdf = ref.prepare(make_bars(closes))
        return next(i for i in range(len(rdf)) if ref.signal(rdf, i, None).action == Action.LONG)

    def test_entry_suppressed_when_vol_pct_above_threshold(self, make_bars):
        closes = [100.0 + i for i in range(40)]
        s = self.gated()
        df = s.prepare(make_bars(closes))
        i = self._entry_bar(make_bars, closes)
        df.loc[i, "vol_pct"] = 0.95
        assert s.signal(df, i, None).action == Action.HOLD

    def test_entry_allowed_in_calm_regime_same_trigger(self, make_bars):
        closes = [100.0 + i for i in range(40)]
        s = self.gated()
        df = s.prepare(make_bars(closes))
        i = self._entry_bar(make_bars, closes)
        df.loc[i, "vol_pct"] = 0.5
        assert s.signal(df, i, None).action == Action.LONG

    def test_nan_vol_pct_does_not_suppress(self, make_bars):
        closes = [100.0 + i for i in range(40)]
        s = self.gated()
        df = s.prepare(make_bars(closes))
        i = self._entry_bar(make_bars, closes)
        df.loc[i, "vol_pct"] = float("nan")
        assert s.signal(df, i, None).action == Action.LONG

    def test_exit_fires_during_high_vol(self, make_bars):
        # Exits ignore the gate entirely.
        closes = [200.0 - i for i in range(40)]
        s = self.gated()
        df = s.prepare(make_bars(closes))
        i = len(df) - 1
        df.loc[i, "vol_pct"] = 0.99
        assert s.signal(df, i, long_position()).action == Action.EXIT

    def test_gate_none_adds_no_columns_and_matches_ungated(self, make_bars):
        closes = [100.0 + i for i in range(40)]
        plain = strat(lookbacks=(5, 10), skip_days=3)
        df = plain.prepare(make_bars(closes))
        assert "vol_pct" not in df.columns
        assert "realized_vol" not in df.columns
        i = self._entry_bar(make_bars, closes)
        assert plain.signal(df, i, None).action == Action.LONG


class TestBuildStrategy:
    def cfg(self, params):
        return InstrumentConfig(
            symbol="AAA", asset_class=AssetClass.US_EQUITY,
            strategy=StrategyName.TS_MOMENTUM, timeframe_minutes=1440, params=params,
        )

    def test_builds_with_defaults(self):
        s = build_strategy(self.cfg({}))
        assert isinstance(s, TsMomentum)
        assert s.lookbacks == [126, 147, 168, 189, 210, 231, 252]
        assert s.skip_days == 21
        assert s.stop_atr_mult == 3.0

    def test_builds_with_custom_params(self):
        s = build_strategy(self.cfg({
            "lookbacks": [60, 120], "skip_days": 10, "stop_atr_mult": 2.5,
            "vol_gate_percentile": 0.9,
        }))
        assert s.lookbacks == [60, 120]
        assert s.skip_days == 10
        assert s.stop_atr_mult == 2.5
        assert s.vol_gate_percentile == 0.9
