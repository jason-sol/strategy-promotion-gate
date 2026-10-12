from datetime import date

import pandas as pd
import pytest

from bot.strategies import build_strategy
from bot.strategies.base import Action, PositionState
from bot.strategies.pead_drift import PeadDrift
from config import AssetClass, InstrumentConfig, StrategyName


def dated_bars(make_bars, start="2024-01-01", periods=40):
    closes = [100.0 + i for i in range(periods)]
    df = make_bars(closes)
    df.index = pd.bdate_range(start, periods=periods)
    return df


def events_df(rows):
    return pd.DataFrame(rows, columns=["announce_date", "hour", "sue"])


def strat(events, sue_min=1.0, drift_window=10, atr_period=5, stop_atr_mult=3.0):
    return PeadDrift(events=events, sue_min=sue_min, drift_window=drift_window,
                     atr_period=atr_period, stop_atr_mult=stop_atr_mult)


def long_position():
    return PositionState(symbol="X", side="long", qty=10, entry_price=100.0,
                         stop_price=90.0, high_water_mark=100.0)


def loc(df, d):
    return df.index.get_loc(pd.Timestamp(d))


class TestPrepare:
    def test_adds_atr_and_feature_columns(self, make_bars):
        s = strat(events_df([[date(2024, 1, 15), "bmo", 2.0]]))
        df = s.prepare(dated_bars(make_bars))
        for col in ("atr", "days_since_announcement", "standardized_surprise", "event_active"):
            assert col in df.columns

    def test_non_datetime_index_raises(self, make_bars):
        s = strat(events_df([[date(2024, 1, 15), "bmo", 2.0]]))
        with pytest.raises((TypeError, ValueError)):
            s.prepare(make_bars([100.0 + i for i in range(30)]))

    def test_bmo_maps_to_announce_bar(self, make_bars):
        s = strat(events_df([[date(2024, 1, 15), "bmo", 2.0]]))
        df = s.prepare(dated_bars(make_bars))
        i = loc(df, "2024-01-15")
        assert df["event_active"].iloc[i]
        assert df["days_since_announcement"].iloc[i] == 0
        assert df["standardized_surprise"].iloc[i] == pytest.approx(2.0)

    def test_amc_shifts_to_next_bar(self, make_bars):
        s = strat(events_df([[date(2024, 1, 15), "amc", 2.0]]))
        df = s.prepare(dated_bars(make_bars))
        assert not df["event_active"].iloc[loc(df, "2024-01-15")]
        j = loc(df, "2024-01-16")
        assert df["event_active"].iloc[j]
        assert df["days_since_announcement"].iloc[j] == 0

    def test_unknown_hour_shifts_to_next_bar(self, make_bars):
        s = strat(events_df([[date(2024, 1, 15), "", 2.0]]))
        df = s.prepare(dated_bars(make_bars))
        assert not df["event_active"].iloc[loc(df, "2024-01-15")]
        assert df["days_since_announcement"].iloc[loc(df, "2024-01-16")] == 0

    def test_nan_sue_never_activates(self, make_bars):
        s = strat(events_df([[date(2024, 1, 15), "bmo", float("nan")]]))
        df = s.prepare(dated_bars(make_bars))
        assert not df["event_active"].any()

    def test_window_closes_after_drift_window(self, make_bars):
        s = strat(events_df([[date(2024, 1, 15), "bmo", 2.0]]), drift_window=3)
        df = s.prepare(dated_bars(make_bars))
        assert df["event_active"].iloc[loc(df, "2024-01-18")]       # day 3
        assert not df["event_active"].iloc[loc(df, "2024-01-19")]   # day 4 > window

    def test_overlapping_events_most_recent_wins(self, make_bars):
        s = strat(events_df([
            [date(2024, 1, 15), "bmo", 2.0],
            [date(2024, 1, 17), "bmo", 3.0],
        ]))
        df = s.prepare(dated_bars(make_bars))
        i = loc(df, "2024-01-17")
        assert df["standardized_surprise"].iloc[i] == pytest.approx(3.0)
        assert df["days_since_announcement"].iloc[i] == 0


class TestSignal:
    def test_long_fires_in_window_when_sue_qualifies(self, make_bars):
        s = strat(events_df([[date(2024, 1, 15), "bmo", 2.0]]))
        df = s.prepare(dated_bars(make_bars))
        i = loc(df, "2024-01-16")  # day 1
        assert s.signal(df, i, None).action == Action.LONG

    def test_no_entry_on_day_zero(self, make_bars):
        s = strat(events_df([[date(2024, 1, 15), "bmo", 2.0]]))
        df = s.prepare(dated_bars(make_bars))
        i = loc(df, "2024-01-15")  # day 0 = announcement-reaction bar
        assert s.signal(df, i, None).action == Action.HOLD

    def test_no_entry_below_sue_min(self, make_bars):
        s = strat(events_df([[date(2024, 1, 15), "bmo", 0.5]]), sue_min=1.0)
        df = s.prepare(dated_bars(make_bars))
        i = loc(df, "2024-01-16")
        assert s.signal(df, i, None).action == Action.HOLD

    def test_entry_at_sue_min_boundary(self, make_bars):
        s = strat(events_df([[date(2024, 1, 15), "bmo", 1.0]]), sue_min=1.0)
        df = s.prepare(dated_bars(make_bars))
        i = loc(df, "2024-01-16")
        assert s.signal(df, i, None).action == Action.LONG

    def test_holds_position_inside_window(self, make_bars):
        s = strat(events_df([[date(2024, 1, 15), "bmo", 2.0]]), drift_window=5)
        df = s.prepare(dated_bars(make_bars))
        i = loc(df, "2024-01-18")  # day 3, still active
        assert s.signal(df, i, long_position()).action == Action.HOLD

    def test_exits_past_drift_window(self, make_bars):
        s = strat(events_df([[date(2024, 1, 15), "bmo", 2.0]]), drift_window=3)
        df = s.prepare(dated_bars(make_bars))
        i = loc(df, "2024-01-19")  # day 4 > window
        assert s.signal(df, i, long_position()).action == Action.EXIT

    def test_never_shorts_on_negative_sue(self, make_bars):
        s = strat(events_df([[date(2024, 1, 15), "bmo", -3.0]]))
        df = s.prepare(dated_bars(make_bars))
        actions = {s.signal(df, i, None).action for i in range(len(df))}
        assert Action.SHORT not in actions

    def test_neutral_with_no_events(self, make_bars):
        s = strat(events_df([]))
        df = s.prepare(dated_bars(make_bars))
        actions = {s.signal(df, i, None).action for i in range(len(df))}
        assert actions == {Action.HOLD}

    def test_neutral_on_nan_sue(self, make_bars):
        s = strat(events_df([[date(2024, 1, 15), "bmo", float("nan")]]))
        df = s.prepare(dated_bars(make_bars))
        actions = {s.signal(df, i, None).action for i in range(len(df))}
        assert actions == {Action.HOLD}

    def test_stop_atr_mult_is_wide_disaster_stop(self, make_bars):
        assert strat(events_df([]), stop_atr_mult=3.0).stop_atr_mult == 3.0


class TestBuildStrategy:
    def cfg(self, params):
        return InstrumentConfig(
            symbol="AAPL", asset_class=AssetClass.US_EQUITY,
            strategy=StrategyName.PEAD_DRIFT, timeframe_minutes=1440, params=params)

    def test_builds_from_params_events(self):
        ev = events_df([[date(2024, 1, 15), "bmo", 2.0]])
        s = build_strategy(self.cfg({"events": ev, "sue_min": 1.5, "drift_window": 8}))
        assert isinstance(s, PeadDrift)
        assert s.sue_min == 1.5
        assert s.drift_window == 8

    def test_raises_without_events(self):
        with pytest.raises(ValueError):
            build_strategy(self.cfg({}))
