from bot.strategies import build_strategy
from bot.strategies.base import Action, PositionState
from bot.strategies.rsi_mean_reversion import RsiMeanReversion
from config import AssetClass, StrategyName


def strat(regime_period=10, confirm_days=1):
    # Small periods so fixtures stay short; logic is period-agnostic.
    return RsiMeanReversion(rsi_period=2, entry_below=10.0, regime_period=regime_period,
                            exit_period=3, stop_mult=3.0, atr_period=5,
                            confirm_days=confirm_days)


def long_position():
    return PositionState(symbol="SPY", side="long", qty=10, entry_price=100.0,
                         stop_price=90.0, high_water_mark=100.0)


class TestRsiMeanReversion:
    def test_buys_sharp_dip_in_uptrend(self, make_bars):
        # Old low prices hold the regime SMA well under price (like a real
        # 200-SMA), so a two-day plunge crushes RSI(2) without breaking regime
        closes = [50.0] * 20 + [100.0 + 2 * i for i in range(12)] + [112.0, 105.0]
        s = strat(regime_period=20)
        df = s.prepare(make_bars(closes))
        assert s.signal(df, len(df) - 1, None).action == Action.LONG

    def test_no_buy_below_regime_sma(self, make_bars):
        # Same plunge shape but under the regime SMA: filter blocks it
        closes = [200.0] * 20 + [100.0 - 2 * i for i in range(12)] + [70.0, 65.0]
        s = strat(regime_period=20)
        df = s.prepare(make_bars(closes))
        assert s.signal(df, len(df) - 1, None).action == Action.HOLD

    def test_no_buy_without_oversold_rsi(self, make_bars):
        closes = [100.0 + i for i in range(22)]
        df = strat().prepare(make_bars(closes))
        assert strat().signal(df, len(df) - 1, None).action == Action.HOLD

    def test_exits_on_snapback_above_exit_sma(self, make_bars):
        # Dip then strong bounce well above the 3-SMA
        closes = [100.0 + i for i in range(18)] + [112.0, 110.0, 118.0]
        df = strat().prepare(make_bars(closes))
        sig = strat().signal(df, len(df) - 1, long_position())
        assert sig.action == Action.EXIT

    def test_holds_position_before_snapback(self, make_bars):
        closes = [100.0 + i for i in range(18)] + [113.0, 110.0, 108.0]
        df = strat().prepare(make_bars(closes))
        sig = strat().signal(df, len(df) - 1, long_position())
        assert sig.action == Action.HOLD


class TestRegimeConfirmation:
    # A genuine RSI(2)<10 dip that keeps close above the SMA can only happen in
    # a sustained uptrend (the SMA sits well below price). The dip needed to
    # crush RSI drags close back below its own rolling mean, so a "close just
    # crossed above the SMA" bar and an "RSI oversold" bar cannot coincide on
    # real bars. The signal-level tests therefore build the regime_ok column
    # from real prepare()-computed prices and override only the single rsi value
    # to isolate the confirmation gate from the price dynamics.

    def test_confirm_days_one_matches_raw_comparison(self, make_bars):
        # Regression guard: with confirm_days=1, regime_ok must be exactly the
        # old `close > regime_sma` comparison (NaN warmup falsy), so every
        # signal is unchanged from the pre-confirmation behaviour.
        closes = [50.0] * 20 + [100.0 + 2 * i for i in range(12)] + [112.0, 105.0]
        df = strat(regime_period=20, confirm_days=1).prepare(make_bars(closes))
        raw = df["close"] > df["regime_sma"]
        assert df["regime_ok"].tolist() == raw.tolist()
        assert not df["regime_ok"].iloc[:20].any()  # warmup evaluates falsy

    def test_regime_ok_requires_consecutive_bars(self, make_bars):
        # Close crosses above the regime SMA on the last two bars only.
        closes = [50.0] * 19 + [80.0, 81.0]
        d1 = strat(regime_period=10, confirm_days=1).prepare(make_bars(closes))
        d3 = strat(regime_period=10, confirm_days=3).prepare(make_bars(closes))
        above = d1["close"] > d1["regime_sma"]
        assert above.iloc[-1] and above.iloc[-2] and not above.iloc[-3]
        assert d1["regime_ok"].iloc[-1]
        assert not d3["regime_ok"].iloc[-1]  # only two consecutive bars above

    def test_regime_ok_resets_after_dip_below(self, make_bars):
        # A single bar back below the SMA restarts the consecutive count.
        closes = [50.0] * 10 + [80.0, 81.0, 82.0, 55.0, 90.0, 91.0, 92.0]
        d3 = strat(regime_period=10, confirm_days=3).prepare(make_bars(closes))
        above = d3["close"] > d3["regime_sma"]
        assert not above.iloc[13]  # the dip bar is below the SMA
        assert not d3["regime_ok"].iloc[14]  # 1 bar back above -> not confirmed
        assert not d3["regime_ok"].iloc[15]  # 2 bars back above -> not confirmed
        assert d3["regime_ok"].iloc[16]      # 3 sustained bars -> confirmed

    def test_dip_after_recent_cross_longs_at_one_holds_at_three(self, make_bars):
        # Close is above the regime SMA for exactly the last two bars.
        closes = [50.0] * 19 + [80.0, 81.0]
        i = len(closes) - 1
        d1 = strat(regime_period=10, confirm_days=1).prepare(make_bars(closes))
        d3 = strat(regime_period=10, confirm_days=3).prepare(make_bars(closes))
        d1.loc[i, "rsi"] = 5.0  # isolate the regime gate from price dynamics
        d3.loc[i, "rsi"] = 5.0
        assert strat(regime_period=10, confirm_days=1).signal(d1, i, None).action == Action.LONG
        assert strat(regime_period=10, confirm_days=3).signal(d3, i, None).action == Action.HOLD

    def test_dip_after_sustained_bars_longs_at_three(self, make_bars):
        # Close held above the regime SMA for well over three bars.
        closes = [50.0] * 15 + [80.0, 81.0, 82.0, 83.0, 84.0, 85.0]
        i = len(closes) - 1
        d3 = strat(regime_period=10, confirm_days=3).prepare(make_bars(closes))
        assert d3["regime_ok"].iloc[i]
        d3.loc[i, "rsi"] = 5.0
        assert strat(regime_period=10, confirm_days=3).signal(d3, i, None).action == Action.LONG


def gated_strat(vol_gate_percentile=0.8):
    # Small vol-gate windows so short fixtures still produce a non-NaN vol_pct.
    return RsiMeanReversion(rsi_period=2, entry_below=10.0, regime_period=20,
                            exit_period=3, stop_mult=3.0, atr_period=5, confirm_days=1,
                            vol_gate_percentile=vol_gate_percentile,
                            vol_gate_lookback=3, vol_gate_rank_window=5)


# A LONG-triggering dip-in-uptrend fixture reused across gate tests.
_ENTRY_CLOSES = [50.0] * 20 + [100.0 + 2 * i for i in range(12)] + [112.0, 105.0]


class TestVolGate:
    # The gate is a discrete stand-aside on entries only. vol_pct is overridden
    # at the signal bar (documented single-cell technique) to isolate the gate
    # from the price dynamics that drive the entry trigger.
    def test_entry_suppressed_when_vol_pct_above_threshold(self, make_bars):
        s = gated_strat(vol_gate_percentile=0.8)
        df = s.prepare(make_bars(_ENTRY_CLOSES))
        i = len(df) - 1
        df.loc[i, "vol_pct"] = 0.95
        assert s.signal(df, i, None).action == Action.HOLD

    def test_entry_allowed_in_calm_regime_same_trigger(self, make_bars):
        s = gated_strat(vol_gate_percentile=0.8)
        df = s.prepare(make_bars(_ENTRY_CLOSES))
        i = len(df) - 1
        df.loc[i, "vol_pct"] = 0.5
        assert s.signal(df, i, None).action == Action.LONG

    def test_nan_vol_pct_does_not_suppress(self, make_bars):
        s = gated_strat(vol_gate_percentile=0.8)
        df = s.prepare(make_bars(_ENTRY_CLOSES))
        i = len(df) - 1
        df.loc[i, "vol_pct"] = float("nan")
        assert s.signal(df, i, None).action == Action.LONG

    def test_exit_fires_during_high_vol(self, make_bars):
        # Exits must ignore the gate entirely. Uptrend keeps close above the
        # exit SMA; length clears min_bars (regime_period 20 -> 30).
        s = gated_strat(vol_gate_percentile=0.8)
        closes = [100.0 + i for i in range(34)]
        df = s.prepare(make_bars(closes))
        i = len(df) - 1
        df.loc[i, "vol_pct"] = 0.99
        assert s.signal(df, i, long_position()).action == Action.EXIT

    def test_gate_none_matches_ungated_and_adds_no_columns(self, make_bars):
        df_plain = strat(regime_period=20).prepare(make_bars(_ENTRY_CLOSES))
        assert "vol_pct" not in df_plain.columns
        assert "realized_vol" not in df_plain.columns
        i = len(df_plain) - 1
        assert strat(regime_period=20).signal(df_plain, i, None).action == Action.LONG


class TestQqqLineupConfig:
    def test_qqq_lineup_config_builds_promoted_variant(self):
        # QQQ runs the promoted RSI2<10 200sma variant, which the harness
        # measured without the vol-target overlay.
        from config import INSTRUMENTS
        qqq = next(c for c in INSTRUMENTS if c.symbol == "QQQ")
        assert qqq.strategy == StrategyName.RSI_MEAN_REVERSION
        assert qqq.asset_class == AssetClass.US_EQUITY
        assert qqq.timeframe_minutes == 1440
        assert qqq.slippage_pct == 0.0001
        assert "vol_target" not in qqq.params
        s = build_strategy(qqq)
        assert isinstance(s, RsiMeanReversion)
        assert (s.rsi_period, s.entry_below, s.regime_period) == (2, 10, 200)
        assert (s.exit_period, s.stop_mult, s.confirm_days) == (5, 3.0, 1)
        assert s.vol_gate_percentile is None
