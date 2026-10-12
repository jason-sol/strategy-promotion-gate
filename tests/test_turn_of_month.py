import numpy as np
import pandas as pd
import pytest

from bot.strategies import build_strategy
from bot.strategies.base import Action, PositionState
from bot.strategies.turn_of_month import TurnOfMonth
from config import AssetClass, InstrumentConfig, StrategyName


def strat(entry_window_bdays=2, exit_day_of_month=3, atr_period=5,
          stop_atr_mult=3.0, **kw):
    return TurnOfMonth(entry_window_bdays=entry_window_bdays,
                       exit_day_of_month=exit_day_of_month, atr_period=atr_period,
                       stop_atr_mult=stop_atr_mult, **kw)


def long_position():
    return PositionState(symbol="X", side="long", qty=10, entry_price=100.0,
                         stop_price=90.0, high_water_mark=100.0)


def dated_bars(make_bars, start="2023-09-01", periods=180):
    # Real business-day DatetimeIndex, mirroring what Backtester feeds prepare().
    # Starts months before the Feb 2024 dates the window tests probe so those
    # bars sit past min_bars (a 100-bar ATR warm-up) and reach signal() at all.
    closes = [100.0 + i for i in range(periods)]
    df = make_bars(closes)
    df.index = pd.bdate_range(start, periods=periods)
    return df


def loc(df, date):
    return df.index.get_loc(pd.Timestamp(date))


class TestTurnOfMonthWindow:
    def test_last_business_day_is_in_window_and_goes_long(self, make_bars):
        s = strat()
        df = s.prepare(dated_bars(make_bars))
        i = loc(df, "2024-02-29")  # last business day of Feb 2024
        assert df["in_tom_window"].iloc[i]
        assert s.signal(df, i, None).action == Action.LONG

    def test_second_trading_day_in_window_fourth_out(self, make_bars):
        s = strat()
        df = s.prepare(dated_bars(make_bars))
        assert df["in_tom_window"].iloc[loc(df, "2024-02-02")]   # 2nd bday <= 3
        assert not df["in_tom_window"].iloc[loc(df, "2024-02-06")]  # 4th bday > 3

    def test_entry_side_boundary_last_two_business_days(self, make_bars):
        # entry_window_bdays=N means the last N business days are in-window:
        # at N=2, 2024-02-28 (remaining=1) is IN, 2024-02-27 (remaining=2) is OUT.
        s = strat()
        df = s.prepare(dated_bars(make_bars))
        assert df["in_tom_window"].iloc[loc(df, "2024-02-28")]
        assert not df["in_tom_window"].iloc[loc(df, "2024-02-27")]

    def test_tz_aware_index_matches_tz_naive(self, make_bars):
        # Live bars are tz-aware UTC; the calendar window must be identical to the
        # tz-naive case (guards the df.index.tz branch in prepare).
        s = strat()
        naive = dated_bars(make_bars)
        aware = naive.copy()
        aware.index = naive.index.tz_localize("UTC")
        assert list(s.prepare(naive)["in_tom_window"]) == list(s.prepare(aware)["in_tom_window"])

    def test_mid_month_flat_stays_flat(self, make_bars):
        s = strat()
        df = s.prepare(dated_bars(make_bars))
        i = loc(df, "2024-02-15")
        assert not df["in_tom_window"].iloc[i]
        assert s.signal(df, i, None).action == Action.HOLD

    def test_held_position_exits_first_bar_outside_window(self, make_bars):
        s = strat()
        df = s.prepare(dated_bars(make_bars))
        i = loc(df, "2024-02-15")
        assert s.signal(df, i, long_position()).action == Action.EXIT

    def test_held_position_holds_inside_window(self, make_bars):
        s = strat()
        df = s.prepare(dated_bars(make_bars))
        i = loc(df, "2024-02-29")
        assert s.signal(df, i, long_position()).action == Action.HOLD

    def test_no_lookahead_window_stable_when_future_bars_appended(self, make_bars):
        # The last-business-day mark is calendar-derived, not read off future
        # bars: truncating everything after a boundary bar must not change it.
        s = strat()
        full = dated_bars(make_bars)
        for date in ("2024-02-29", "2024-02-02"):
            k = loc(full, date) + 1
            trunc = s.prepare(full.iloc[:k])
            whole = s.prepare(full)
            assert trunc["in_tom_window"].iloc[k - 1] == whole["in_tom_window"].iloc[k - 1]

    def test_prepare_adds_atr(self, make_bars):
        s = strat()
        df = s.prepare(dated_bars(make_bars))
        assert "atr" in df.columns


def _old_window(index, entry_days_before_eom, exit_day_of_month):
    # Pinned pre-rename semantics: entry side used bdays_remaining <= N, so N
    # actually admitted the last N+1 business days. Kept here purely as the
    # equivalence oracle; production now uses bdays_remaining < entry_window_bdays.
    idx = index.normalize().tz_localize(None) if index.tz else index.normalize()
    month_ends = idx + pd.offsets.MonthEnd(0)
    bdays_remaining = np.array([len(pd.bdate_range(d, e)) - 1 for d, e in zip(idx, month_ends)])
    tbday = pd.Series(0, index=idx).groupby(idx.to_period("M")).cumcount().to_numpy() + 1
    return (bdays_remaining <= entry_days_before_eom) | (tbday <= exit_day_of_month)


class TestEntryWindowSemantics:
    def multi_month(self, make_bars, periods=180):
        closes = [100.0 + i for i in range(periods)]
        df = make_bars(closes)
        df.index = pd.bdate_range("2023-11-01", periods=periods)
        return df

    @pytest.mark.parametrize("new_n,exit_dom", [(2, 3), (3, 3), (2, 5)])
    def test_new_n_matches_old_n_minus_one(self, make_bars, new_n, exit_dom):
        # Promoted variants must be behaviour-identical after the rename:
        # new (2,3)==old (1,3), new (3,3)==old (2,3), new (2,5)==old (1,5).
        df = self.multi_month(make_bars)
        s = strat(entry_window_bdays=new_n, exit_day_of_month=exit_dom)
        got = s.prepare(df)["in_tom_window"].to_numpy()
        expected = _old_window(df.index, new_n - 1, exit_dom)
        assert list(got) == list(expected)

    def test_n_equals_literal_last_n_business_days(self, make_bars):
        # N literally names the count of month-end business days in the window.
        df = self.multi_month(make_bars, periods=90)
        for n in (1, 2, 3):
            got = strat(entry_window_bdays=n, exit_day_of_month=0).prepare(df)["in_tom_window"]
            # Feb 2024 last business days, most-recent first.
            eom_bdays = ["2024-02-29", "2024-02-28", "2024-02-27"]
            for k, d in enumerate(eom_bdays):
                assert got.iloc[loc(df, d)] == (k < n)


class TestHolidayMonthEnd:
    def test_window_shrinks_around_holiday_month_end(self, make_bars):
        # pd.bdate_range is weekend-only, not holiday-aware. Good Friday
        # 2018-03-30 was March's last WEEKDAY but the market's last TRADING day
        # was 2018-03-29. With that holiday absent from the bar index, the
        # last-business-day math treats 2018-03-30 as still ahead, so with
        # entry_window_bdays=2 only 2018-03-29 lands in-window (one day, not two).
        # Accepted trade-off: the window can shrink by up to a day around holiday
        # month-ends rather than add an exchange-calendar dependency.
        idx = pd.bdate_range("2018-03-01", "2018-03-31")
        idx = idx[idx != pd.Timestamp("2018-03-30")]  # market closed Good Friday
        closes = [100.0 + i for i in range(len(idx))]
        df = make_bars(closes)
        df.index = idx
        got = strat(entry_window_bdays=2, exit_day_of_month=0).prepare(df)["in_tom_window"]
        assert got.iloc[loc(df, "2018-03-29")]        # only the true last bday
        assert not got.iloc[loc(df, "2018-03-28")]    # window lost its second day


class TestTurnOfMonthIndexRequirement:
    def test_non_datetime_index_raises(self, make_bars):
        # Backtester supplies a DatetimeIndex; a RangeIndex cannot yield a valid
        # calendar window and must fail loudly rather than silently misfire.
        s = strat()
        with pytest.raises((TypeError, ValueError)):
            s.prepare(make_bars([100.0 + i for i in range(30)]))


class TestTurnOfMonthVolGate:
    def gated(self, vol_gate_percentile=0.8):
        return strat(vol_gate_percentile=vol_gate_percentile,
                     vol_gate_lookback=3, vol_gate_rank_window=5)

    def test_entry_suppressed_when_vol_pct_above_threshold(self, make_bars):
        s = self.gated()
        df = s.prepare(dated_bars(make_bars))
        i = loc(df, "2024-02-29")
        df.loc[df.index[i], "vol_pct"] = 0.95
        assert s.signal(df, i, None).action == Action.HOLD

    def test_entry_allowed_in_calm_regime(self, make_bars):
        s = self.gated()
        df = s.prepare(dated_bars(make_bars))
        i = loc(df, "2024-02-29")
        df.loc[df.index[i], "vol_pct"] = 0.5
        assert s.signal(df, i, None).action == Action.LONG

    def test_nan_vol_pct_does_not_suppress(self, make_bars):
        s = self.gated()
        df = s.prepare(dated_bars(make_bars))
        i = loc(df, "2024-02-29")
        df.loc[df.index[i], "vol_pct"] = float("nan")
        assert s.signal(df, i, None).action == Action.LONG

    def test_exit_ignores_gate(self, make_bars):
        s = self.gated()
        df = s.prepare(dated_bars(make_bars))
        i = loc(df, "2024-02-15")  # outside window -> exit regardless of vol
        df.loc[df.index[i], "vol_pct"] = 0.99
        assert s.signal(df, i, long_position()).action == Action.EXIT

    def test_gate_none_adds_no_columns(self, make_bars):
        s = strat()
        df = s.prepare(dated_bars(make_bars))
        assert "vol_pct" not in df.columns
        assert "realized_vol" not in df.columns


class TestBuildStrategy:
    def cfg(self, params):
        return InstrumentConfig(
            symbol="VOO", asset_class=AssetClass.US_EQUITY,
            strategy=StrategyName.TURN_OF_MONTH, timeframe_minutes=1440, params=params,
        )

    def test_builds_with_defaults(self):
        s = build_strategy(self.cfg({}))
        assert isinstance(s, TurnOfMonth)
        assert s.entry_window_bdays == 2
        assert s.exit_day_of_month == 3
        assert s.stop_atr_mult == 3.0

    def test_builds_with_custom_params(self):
        s = build_strategy(self.cfg({
            "entry_window_bdays": 3, "exit_day_of_month": 4,
            "stop_atr_mult": 2.5, "vol_gate_percentile": 0.9,
        }))
        assert s.entry_window_bdays == 3
        assert s.exit_day_of_month == 4
        assert s.stop_atr_mult == 2.5
        assert s.vol_gate_percentile == 0.9

    def test_voo_lineup_config_builds_promoted_variant(self):
        # VOO joins the paper lineup under the promoted TOM 2/5 variant.
        from config import INSTRUMENTS
        voo = next(c for c in INSTRUMENTS if c.symbol == "VOO")
        assert voo.strategy == StrategyName.TURN_OF_MONTH
        assert voo.asset_class == AssetClass.US_EQUITY
        assert voo.timeframe_minutes == 1440
        assert voo.slippage_pct == 0.0001
        assert "vol_target" not in voo.params
        s = build_strategy(voo)
        assert isinstance(s, TurnOfMonth)
        assert s.entry_window_bdays == 2
        assert s.exit_day_of_month == 5
