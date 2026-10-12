import re
import sys
from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd
import pytest

import config
from bot import diagnose
from bot.diagnose import (
    indicator_drift,
    lookahead_indices,
    pooled_lookahead_probe,
    probe_frame,
    probe_lookahead,
    report_lookahead,
    report_slippage,
    report_startup_windows,
    sweep_history_start,
    synthetic_bars,
)
from bot.indicators import atr
from bot.portfolio import ExecutionLog
from bot.strategies import build_strategy
from bot.strategies.base import HOLD, Action, PositionState, Signal, Strategy
from config import BACKTEST_MONTHS, INSTRUMENTS, enabled_instruments


class _WarmupHungry(Strategy):
    """min_bars understates what the indicators need: sma_50 cannot warm up on
    the last 20 bars alone, while doubled_close is bar-local and cannot drift."""

    @property
    def min_bars(self) -> int:
        return 20

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        df["atr"] = atr(df, 14)
        df["sma_50"] = df["close"].rolling(50).mean()
        df["doubled_close"] = df["close"] * 2
        return df

    def signal(self, df: pd.DataFrame, i: int,
               position: PositionState | None) -> Signal:
        return HOLD


def _bars(n: int) -> pd.DataFrame:
    # The constant 2.0 high-low range is load-bearing: Wilder ATR converges to
    # exactly 2.0 from any seed, so atr never drifts and the NaN case can assert
    # on an exact column list.
    closes = [100.0 + i for i in range(n)]
    return pd.DataFrame(
        {"open": closes, "high": [c + 1 for c in closes],
         "low": [c - 1 for c in closes], "close": closes,
         "volume": [1000.0] * n},
        index=pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC"))


class _ScaledDrift(_WarmupHungry):
    """`scaled` differs by exactly `delta` (relative) between the full frame and
    the min_bars tail, so the tolerance threshold itself can be pinned."""

    def __init__(self, delta: float):
        self.delta = delta

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        df["atr"] = atr(df, 14)
        df["scaled"] = df["close"] * (1 + self.delta if len(df) > self.min_bars else 1.0)
        return df


class TestIndicatorDrift:
    def test_column_nan_only_on_the_truncated_window_is_flagged(self):
        drifted = indicator_drift(_WarmupHungry(), _bars(200))
        assert [d.column for d in drifted] == ["sma_50"]

    def test_a_bar_local_column_never_drifts(self):
        drifted = indicator_drift(_WarmupHungry(), _bars(200))
        assert "doubled_close" not in [d.column for d in drifted]

    def test_raw_price_columns_are_not_reported(self):
        drifted = indicator_drift(_WarmupHungry(), _bars(200))
        assert "close" not in [d.column for d in drifted]

    def test_a_window_long_enough_for_every_indicator_is_clean(self):
        class _Warm(_WarmupHungry):
            @property
            def min_bars(self) -> int:
                return 120

        assert indicator_drift(_Warm(), _bars(200)) == []

    def test_a_recursive_indicator_drifts_in_value_not_nan(self):
        class _Recursive(_WarmupHungry):
            def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
                df = bars.copy()
                df["atr"] = atr(df, 14)
                # Wilder smoothing never fully forgets its seed: a short window
                # gives a different value, not a NaN.
                df["ema_15"] = df["close"].ewm(span=15, adjust=False).mean()
                return df

        drifted = indicator_drift(_Recursive(), _bars(200))
        assert "ema_15" in [d.column for d in drifted]

    def test_a_sub_tolerance_drift_is_suppressed(self):
        assert indicator_drift(_ScaledDrift(1e-7), _bars(200)) == []

    def test_an_above_tolerance_drift_is_reported(self):
        drifted = indicator_drift(_ScaledDrift(1e-5), _bars(200))
        assert [d.column for d in drifted] == ["scaled"]

    def test_a_frame_shorter_than_min_bars_is_rejected(self):
        """Truncating to min_bars would hand back the same frame twice and
        report a clean bill of health nobody measured."""
        with pytest.raises(ValueError):
            indicator_drift(_WarmupHungry(), _bars(10))

    def test_a_frame_of_exactly_min_bars_is_rejected(self):
        with pytest.raises(ValueError):
            indicator_drift(_WarmupHungry(), _bars(20))


class TestSweepHistoryStart:
    """The drift baseline is the history the diagnostic sweep backtests on, so
    'full history' means the same series the validated numbers came from."""

    NOW = datetime(2026, 9, 2, tzinfo=timezone.utc)

    def _cfg(self, symbol):
        return next(c for c in INSTRUMENTS if c.symbol == symbol)

    def test_baseline_covers_the_simulated_window_plus_warmup(self):
        cfg = self._cfg("VOO")
        start = sweep_history_start(cfg, build_strategy(cfg).min_bars, self.NOW)
        assert (self.NOW - start).days > BACKTEST_MONTHS * 30

    def test_short_min_bars_still_gets_the_full_simulated_window(self):
        """A 19-bar min_bars must not shrink the baseline to a few dozen bars:
        the sweep it mirrors runs on months of history regardless."""
        voo = self._cfg("VOO")
        spans = {s: (self.NOW - sweep_history_start(voo, s, self.NOW)).days
                 for s in (19, 400)}
        assert spans[19] >= BACKTEST_MONTHS * 30
        assert spans[400] > spans[19]


class TestSlippageReport:
    def _fills(self, tmp_path, rows):
        path = tmp_path / "fills.csv"
        log = ExecutionLog(str(path))
        for symbol, event, side, spread, slip in rows:
            mid = "nan" if slip == "" else 100.0
            log.record("2026-07-01T10:00:00+00:00", symbol, side, event,
                       mid, spread, 100.0, 100.0, 10, slip)
        return str(path)

    def test_empty_log_says_so_and_still_succeeds(self, tmp_path, capsys):
        assert report_slippage(str(tmp_path / "absent.csv")) == 0
        assert "no measured fills" in capsys.readouterr().out.lower()

    def test_row_carries_count_mean_median_and_worst(self, tmp_path, capsys):
        path = self._fills(tmp_path, [("SPY", "entry", "buy", 2.0, 1.0),
                                      ("SPY", "entry", "buy", 2.0, 2.0),
                                      ("SPY", "entry", "buy", 2.0, 9.0)])
        report_slippage(path)
        line = next(ln for ln in capsys.readouterr().out.splitlines() if "SPY" in ln)
        assert "entry" in line
        for value in ("3", "4.00", "2.00", "9.00"):
            assert value in line

    def test_spread_at_decision_is_reported(self, tmp_path, capsys):
        path = self._fills(tmp_path, [("SPY", "entry", "buy", 7.5, 1.0)])
        report_slippage(path)
        line = next(ln for ln in capsys.readouterr().out.splitlines() if "SPY" in ln)
        assert "7.50" in line

    def test_assumed_column_converts_the_configured_fraction_to_bps(self, tmp_path, capsys):
        # SPY is configured at slippage_pct=0.0001 — one basis point.
        path = self._fills(tmp_path, [("SPY", "entry", "buy", 2.0, 0.5)])
        report_slippage(path)
        line = next(ln for ln in capsys.readouterr().out.splitlines() if "SPY" in ln)
        assert "1.00" in line
        assert "OVER" not in line

    def test_a_median_above_assumed_is_flagged_over(self, tmp_path, capsys):
        path = self._fills(tmp_path, [("SPY", "entry", "buy", 2.0, 9.0)])
        report_slippage(path)
        assert "OVER" in capsys.readouterr().out

    def test_a_median_below_assumed_reads_under(self, tmp_path, capsys):
        path = self._fills(tmp_path, [("SPY", "entry", "buy", 2.0, 0.2)])
        report_slippage(path)
        line = next(ln for ln in capsys.readouterr().out.splitlines() if "SPY" in ln)
        assert "under" in line.lower()

    def test_a_symbol_outside_the_lineup_has_no_assumed_cost(self, tmp_path, capsys):
        path = self._fills(tmp_path, [("TLT", "entry", "buy", 2.0, 3.0)])
        report_slippage(path)
        line = next(ln for ln in capsys.readouterr().out.splitlines() if "TLT" in ln)
        assert "OVER" not in line

    def test_each_fill_kind_gets_its_own_row(self, tmp_path, capsys):
        path = self._fills(tmp_path, [("SPY", "entry", "buy", 2.0, 1.0),
                                      ("SPY", "exit", "sell", 2.0, 2.0),
                                      ("SPY", "stop_fill", "sell", 2.0, "")])
        report_slippage(path)
        rows = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("SPY")]
        assert [r.split()[1] for r in rows] == ["entry", "exit", "stop"]

    def test_unmeasurable_stops_are_counted_not_dropped_silently(self, tmp_path, capsys):
        """The go-live gate reads this table: a stop bucket with nothing
        measurable must read as a gap, not as an absence of stops."""
        path = self._fills(tmp_path, [("VOO", "stop_fill", "sell", 2.0, "")] * 7)
        report_slippage(path)
        line = next(ln for ln in capsys.readouterr().out.splitlines()
                    if ln.startswith("VOO"))
        assert "stop" in line
        assert "7" in line
        assert "no decision quote" in line.lower() or "unmeasurable" in line.lower()

    def test_a_partly_measurable_kind_shows_both_counts(self, tmp_path, capsys):
        path = self._fills(tmp_path, [("SPY", "entry", "buy", 2.0, 1.0),
                                      ("SPY", "entry", "buy", 2.0, "")])
        report_slippage(path)
        line = next(ln for ln in capsys.readouterr().out.splitlines()
                    if ln.startswith("SPY"))
        fields = line.split()
        assert fields[2] == "1"   # measured
        assert fields[3] == "1"   # dropped

    def test_output_is_labelled_since_inception(self, tmp_path, capsys):
        """The email reports a trailing window; this one must not be mistaken
        for the same number."""
        path = self._fills(tmp_path, [("SPY", "entry", "buy", 2.0, 1.0)])
        report_slippage(path)
        assert "inception" in capsys.readouterr().out.lower()

    def test_note_gates_config_changes_on_sample_size(self, tmp_path, capsys):
        report_slippage(self._fills(tmp_path, [("SPY", "entry", "buy", 2.0, 1.0)]))
        out = capsys.readouterr().out
        assert "20" in out
        assert "slippage_pct" in out
        # The bot must never rewrite the assumption for itself.
        assert "by hand" in out.lower() or "manual" in out.lower()


class TestSlippageCli:
    def test_the_two_report_modes_are_mutually_exclusive(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["diagnose", "--slippage", "--startup-window"])
        with pytest.raises(SystemExit):
            diagnose.main()


    def test_the_flag_reports_without_touching_market_data(self, tmp_path, monkeypatch, capsys):
        """--slippage reads a CSV; needing API keys for it would make the
        diagnostic unusable exactly when execution is being investigated."""
        def boom(*a, **k):
            raise AssertionError("MarketData must not be constructed")

        monkeypatch.setattr(diagnose, "MarketData", boom)
        monkeypatch.setattr(diagnose, "report_slippage", lambda *a, **k: 7)
        monkeypatch.setattr(sys, "argv", ["diagnose", "--slippage"])
        assert diagnose.main() == 7


class _BackwardLooking(Strategy):
    """Every column at bar i is computed from bars up to i, so withholding the
    later bars cannot change a decision."""

    @property
    def min_bars(self) -> int:
        return 30

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        df["atr"] = atr(df, 14)
        df["sma_10"] = df["close"].rolling(10).mean()
        return df

    def signal(self, df: pd.DataFrame, i: int,
               position: PositionState | None) -> Signal:
        if i + 1 < self.min_bars:
            return HOLD
        row = df.iloc[i]
        if position is not None:
            if row["close"] < row["sma_10"]:
                return Signal(Action.EXIT, "lost the sma")
            return HOLD
        if row["close"] > row["sma_10"]:
            return Signal(Action.LONG, "above the sma")
        return HOLD


class _PeeksOnEntry(_BackwardLooking):
    """The df.iloc[-1] accident: the entry gate compares against the final
    close of whatever frame it was handed, which in the truncated frame is the
    probed bar itself."""

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(bars)
        df["final_close"] = df["close"].iloc[-1]
        return df

    def signal(self, df: pd.DataFrame, i: int,
               position: PositionState | None) -> Signal:
        if i + 1 < self.min_bars or position is not None:
            return HOLD
        row = df.iloc[i]
        if row["close"] >= row["final_close"]:
            return Signal(Action.LONG, "nothing higher to come")
        return HOLD


class _PeeksOnExit(_BackwardLooking):
    """Exits on tomorrow's close — visible only when the probe holds a
    position, and NaN once the future bar is withheld."""

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(bars)
        df["next_close"] = df["close"].shift(-1)
        return df

    def signal(self, df: pd.DataFrame, i: int,
               position: PositionState | None) -> Signal:
        if i + 1 < self.min_bars or position is None:
            return HOLD
        if df.iloc[i]["close"] < df.iloc[i]["next_close"]:
            return Signal(Action.EXIT, "tomorrow is higher")
        return HOLD


class _PeeksInSignal(_BackwardLooking):
    """The same peek, resident in signal() rather than prepare(): a clean
    prepared frame is no guarantee if the decision reads the frame's tail."""

    def signal(self, df: pd.DataFrame, i: int,
               position: PositionState | None) -> Signal:
        if i + 1 < self.min_bars or position is not None:
            return HOLD
        if df["close"].iloc[i] >= df["close"].iloc[-1]:
            return Signal(Action.LONG, "nothing higher to come")
        return HOLD


class _DropsRows(_BackwardLooking):
    """prepare() drops the warm-up rows, so index i no longer names the same
    bar in the two frames."""

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        return super().prepare(bars).dropna()


class _MutatesPosition(_BackwardLooking):
    """Decides, then writes to the position it was handed. Sharing one
    PositionState between the two frames would let this manufacture a leak."""

    def signal(self, df: pd.DataFrame, i: int,
               position: PositionState | None) -> Signal:
        if i + 1 < self.min_bars or position is None:
            return HOLD
        row = df.iloc[i]
        hit = row["close"] <= position.high_water_mark - row["atr"]
        position.update_high_water_mark(row["close"] * 1.10)
        if hit:
            return Signal(Action.EXIT, "trailing stop")
        return HOLD


class _NeverEnters(_BackwardLooking):
    """Decides exits, never entries: one exercised leg cannot vouch for the
    other."""

    def signal(self, df: pd.DataFrame, i: int,
               position: PositionState | None) -> Signal:
        if position is None:
            return HOLD
        return super().signal(df, i, position)


class _AlwaysDecides(_BackwardLooking):
    """Decides on every probed bar, so the coverage counts the report prints are
    countable by hand from the probe's own sampling."""

    def signal(self, df: pd.DataFrame, i: int,
               position: PositionState | None) -> Signal:
        if position is not None:
            return Signal(Action.EXIT, "always exits")
        return Signal(Action.LONG, "always enters")


class _EntersOnce(_AlwaysDecides):
    """One decisive entry across the whole probe against a fully exercised exit
    leg: thin on one leg only."""

    def signal(self, df: pd.DataFrame, i: int,
               position: PositionState | None) -> Signal:
        if position is None and i != self.min_bars - 1:
            return HOLD
        return super().signal(df, i, position)


class _ShortsAndCovers(_BackwardLooking):
    """Enters short and covers on its own rule. The long leg decides on every
    probed bar, so a probe without a held-short leg reports CLEAN having run
    the cover rule zero times."""

    def signal(self, df: pd.DataFrame, i: int,
               position: PositionState | None) -> Signal:
        if position is None:
            return Signal(Action.SHORT, "always shorts")
        if position.side == "short":
            return Signal(Action.EXIT, "covered")
        return Signal(Action.EXIT, "always exits")


class _ShortsButNeverCovers(_ShortsAndCovers):
    """Can enter short and has no rule that ever leaves one."""

    def signal(self, df: pd.DataFrame, i: int,
               position: PositionState | None) -> Signal:
        if position is not None and position.side == "short":
            return HOLD
        return super().signal(df, i, position)


class _PeeksOnShortCover(_BackwardLooking):
    """Covers on tomorrow's close, and only when short: invisible to a probe
    whose only held leg is long."""

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(bars)
        df["next_close"] = df["close"].shift(-1)
        return df

    def signal(self, df: pd.DataFrame, i: int,
               position: PositionState | None) -> Signal:
        if i + 1 < self.min_bars:
            return HOLD
        if position is None:
            return Signal(Action.SHORT, "always shorts")
        if position.side != "short":
            return HOLD
        if df.iloc[i]["close"] < df.iloc[i]["next_close"]:
            return Signal(Action.EXIT, "tomorrow goes against the short")
        return HOLD


class _PeeksOnAnUnreachableCover(_AlwaysDecides):
    """Long-only, carrying a leaking cover rule the engine can never reach: it
    has no path into a short position to leak out of."""

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(bars)
        df["next_close"] = df["close"].shift(-1)
        return df

    def signal(self, df: pd.DataFrame, i: int,
               position: PositionState | None) -> Signal:
        if position is not None and position.side == "short":
            if df.iloc[i]["close"] < df.iloc[i]["next_close"]:
                return Signal(Action.EXIT, "tomorrow goes against the short")
            return HOLD
        return super().signal(df, i, position)


class TestProbeLookahead:
    def test_a_backward_looking_strategy_is_clean(self):
        assert probe_lookahead(_BackwardLooking(), _bars(60)).leaks == []

    def test_a_peek_inside_signal_is_caught_too(self):
        leaks = probe_lookahead(_PeeksInSignal(), _bars(60)).leaks
        assert leaks
        assert {lk.leg for lk in leaks} == {"flat"}

    def test_a_strategy_that_mutates_the_position_stays_clean(self):
        """Each frame is decided with its own PositionState; hoisting one out of
        the loop would let this strategy's own write look like leakage."""
        assert probe_lookahead(_MutatesPosition(), _bars(60)).leaks == []

    def test_a_prepare_that_drops_rows_is_rejected(self):
        """Row positions have to survive prepare() or index i names two
        different bars and the comparison means nothing."""
        with pytest.raises(ValueError, match="_DropsRows"):
            probe_lookahead(_DropsRows(), _bars(60))

    def test_an_entry_that_reads_a_whole_series_statistic_is_caught(self):
        leaks = probe_lookahead(_PeeksOnEntry(), _bars(60)).leaks
        assert leaks
        assert {(lk.leg, lk.with_future_action, lk.without_future_action)
                for lk in leaks} == {("flat", Action.HOLD, Action.LONG)}

    def test_an_exit_leak_is_caught_by_the_held_position_probe(self):
        """A flat-only probe would call this strategy clean: the leaking branch
        never runs without a position. It reads no side, so both held legs
        catch it."""
        leaks = probe_lookahead(_PeeksOnExit(), _bars(60)).leaks
        assert {(lk.leg, lk.with_future_action, lk.without_future_action)
                for lk in leaks} == {("long", Action.EXIT, Action.HOLD),
                                     ("short", Action.EXIT, Action.HOLD)}

    def test_every_checkable_index_is_probed(self):
        leaks = probe_lookahead(_PeeksOnEntry(), _bars(60)).leaks
        assert [lk.bar_index for lk in leaks] == list(range(29, 59))

    def test_the_last_bar_is_not_checkable(self):
        """Nothing can be withheld from the final bar, so a decision there
        cannot be compared against anything."""
        assert 59 not in lookahead_indices(_PeeksOnEntry(), _bars(60))

    def test_warmup_bars_are_not_probed(self):
        assert min(lookahead_indices(_BackwardLooking(), _bars(60))) == 29

    def test_sample_every_thins_the_probes(self):
        leaks = probe_lookahead(_PeeksOnEntry(), _bars(60), sample_every=10).leaks
        assert [lk.bar_index for lk in leaks] == [29, 39, 49]

    def test_one_bar_past_the_warmup_leaves_exactly_one_index(self):
        leaks = probe_lookahead(_PeeksOnEntry(), _bars(31)).leaks
        assert [lk.bar_index for lk in leaks] == [29]

    def test_a_frame_with_no_future_bar_to_withhold_is_rejected(self):
        with pytest.raises(ValueError):
            probe_lookahead(_BackwardLooking(), _bars(30))

    def test_a_sample_stride_below_one_is_rejected(self):
        with pytest.raises(ValueError):
            probe_lookahead(_BackwardLooking(), _bars(60), sample_every=0)


class TestTheProbeReportsItsOwnCoverage:
    """The leaks and the evidence that the probe compared something come from
    one pass, so the two answers cannot describe different runs."""

    def test_a_clean_probe_still_reports_what_it_decided(self):
        probe = probe_lookahead(_BackwardLooking(), _bars(60))
        assert probe.leaks == []
        assert Action.LONG in probe.flat_actions

    def test_both_legs_report_a_decision_for_every_probed_bar(self):
        probe = probe_lookahead(_BackwardLooking(), _bars(60))
        probed = len(lookahead_indices(_BackwardLooking(), _bars(60)))
        assert len(probe.flat_actions) == len(probe.held_actions) == probed > 0

    def test_the_two_legs_are_kept_apart(self):
        """An exit branch only runs holding a position; pooling the legs would
        let a flat HOLD stand in as exit coverage."""
        probe = probe_lookahead(_PeeksOnExit(), _bars(60))
        assert Action.EXIT in probe.held_actions
        assert Action.EXIT not in probe.flat_actions

    def test_coverage_records_the_decision_made_with_the_future(self):
        """The truncated frame is the counterfactual; coverage has to describe
        the frame the strategy would really be handed."""
        probe = probe_lookahead(_PeeksOnEntry(), _bars(60))
        assert probe.leaks
        assert Action.LONG not in probe.flat_actions

    def test_a_strategy_the_probe_cannot_examine_reports_no_coverage_at_all(self):
        """Coverage rides in the probe's return, so there is no path that
        vouches for a strategy the probe could not examine."""
        with pytest.raises(ValueError, match="_DropsRows"):
            probe_lookahead(_DropsRows(), _bars(60))
        with pytest.raises(ValueError, match="_DropsRows"):
            pooled_lookahead_probe(_DropsRows(), seeds=1)

    def test_the_probe_pools_coverage_across_its_seeds(self):
        one = pooled_lookahead_probe(_BackwardLooking(), seeds=1, samples=5)
        three = pooled_lookahead_probe(_BackwardLooking(), seeds=3, samples=5)
        assert len(three.flat_actions) == 3 * len(one.flat_actions) > 0
        assert len(three.held_actions) == len(three.flat_actions)


class TestTheProbeExercisesTheShortLeg:
    """A cover rule only runs holding a short. With no held-short leg it is
    structurally unprobed, and the probe reports CLEAN having never run it."""

    def test_the_short_leg_decides_every_probed_bar(self):
        strategy = _ShortsAndCovers()
        probe = probe_lookahead(strategy, _bars(60))
        probed = len(lookahead_indices(strategy, _bars(60)))
        assert len(probe.short_actions) == probed > 0

    def test_the_cover_rule_is_reached_only_by_the_short_leg(self):
        probe = probe_lookahead(_ShortsAndCovers(), _bars(60))
        assert Action.EXIT in probe.short_actions
        assert Action.EXIT not in probe.flat_actions

    def test_a_leak_in_the_cover_rule_is_caught(self):
        leaks = probe_lookahead(_PeeksOnShortCover(), _bars(60)).leaks
        assert leaks
        assert {(lk.leg, lk.with_future_action, lk.without_future_action)
                for lk in leaks} == {("short", Action.EXIT, Action.HOLD)}

    def test_the_pooled_probe_aggregates_all_three_legs(self):
        one = pooled_lookahead_probe(_ShortsAndCovers(), seeds=1, samples=5)
        three = pooled_lookahead_probe(_ShortsAndCovers(), seeds=3, samples=5)
        assert len(three.short_actions) == 3 * len(one.short_actions) > 0
        assert (len(three.short_actions) == len(three.held_actions)
                == len(three.flat_actions))


class TestSyntheticBars:
    def test_the_same_seed_gives_the_same_bars(self):
        assert synthetic_bars(50).equals(synthetic_bars(50))

    def test_a_different_seed_gives_different_bars(self):
        assert not synthetic_bars(50, seed=1).equals(synthetic_bars(50, seed=2))

    def test_bars_are_date_indexed_for_calendar_strategies(self):
        index = synthetic_bars(50).index
        assert isinstance(index, pd.DatetimeIndex)
        assert index.is_monotonic_increasing

    def test_the_index_spans_a_month_turn_for_calendar_strategies(self):
        assert len({(d.year, d.month) for d in synthetic_bars(60).index}) >= 2

    def test_high_and_low_bracket_the_close(self):
        bars = synthetic_bars(50)
        assert (bars["high"] >= bars["close"]).all()
        assert (bars["low"] <= bars["close"]).all()


class TestPooledLookaheadProbe:
    def test_the_probe_sizes_its_own_bars_from_min_bars(self):
        assert pooled_lookahead_probe(_PeeksOnEntry(), seeds=1).leaks

    def test_the_probe_keeps_the_sample_count_bounded_per_seed(self):
        leaks = pooled_lookahead_probe(_PeeksOnEntry(), seeds=1, samples=5).leaks
        assert 0 < len(leaks) <= 5

    def test_more_seeds_probe_more_bars(self):
        """One random walk triggers few entry gates; the probe's reach comes
        from unioning several."""
        one = pooled_lookahead_probe(_PeeksOnEntry(), seeds=1, samples=5).leaks
        three = pooled_lookahead_probe(_PeeksOnEntry(), seeds=3, samples=5).leaks
        assert len(three) > len(one)

    def test_the_default_probes_more_than_one_seed(self):
        assert len(pooled_lookahead_probe(_PeeksOnEntry(), samples=5).leaks) > 5

    def test_a_clean_strategy_probes_clean(self):
        assert pooled_lookahead_probe(_BackwardLooking(), seeds=1).leaks == []

    def test_a_pooled_leak_names_the_seed_that_produced_it(self):
        """Leaks from every seed land in one list, so a leak that does not say
        which walk it came from cannot be traced back to any. All three walks
        leak here, so the pooled seeds have to name all three."""
        seeds = {lk.seed for lk in pooled_lookahead_probe(_PeeksOnEntry(), seeds=3).leaks}
        assert seeds == {0, 1, 2}

    def test_every_pooled_leak_replays_from_the_seed_it_names(self):
        """Seed plus bar index is the whole reproduction recipe. Each walk
        leaks at bars the others never reach, so a leak carrying the wrong seed
        does not appear in that walk's own probe at all."""
        leaks = pooled_lookahead_probe(_PeeksOnEntry(), seeds=3).leaks
        replays = {}
        for seed in {lk.seed for lk in leaks}:
            bars, stride = probe_frame(_PeeksOnEntry(), seed)
            replays[seed] = probe_lookahead(_PeeksOnEntry(), bars,
                                            sample_every=stride, seed=seed).leaks

        assert leaks
        for leak in leaks:
            assert leak in replays[leak.seed]


REAL_LINEUP = tuple(cfg.symbol for cfg in enabled_instruments())


@pytest.fixture(scope="module")
def lineup_probes():
    """One 20-seed probe per enabled instrument, shared by every gate below —
    the probe is the suite's most expensive check and it answers both the leak
    and the coverage question at once.

    Built lazily and cached for the module, so under a different test order it
    could first materialize inside a test that has patched the lineup. The
    keys are pinned to the lineup collected at import time, which fails loudly
    rather than vouching for a substituted one."""
    probes = {cfg.symbol: pooled_lookahead_probe(build_strategy(cfg))
              for cfg in enabled_instruments()}
    assert tuple(probes) == REAL_LINEUP
    return probes


class TestLiveLineupHasNoLookahead:
    """CI gate: a strategy in the live lineup that re-decides a bar once the
    future is withheld is reading data the engine will not have."""

    @pytest.mark.parametrize(
        "cfg", enabled_instruments(), ids=lambda c: c.symbol)
    def test_no_enabled_instrument_leaks(self, cfg, lineup_probes):
        assert lineup_probes[cfg.symbol].leaks == []

    @pytest.mark.parametrize(
        "cfg", enabled_instruments(), ids=lambda c: c.symbol)
    def test_every_enabled_instrument_was_actually_probed(self, cfg, lineup_probes):
        """A lineup entry the probe never reached would read as CLEAN while
        nothing about it was examined."""
        assert cfg.symbol in lineup_probes
        assert lineup_probes[cfg.symbol].flat_actions


class TestTheProbeIsNotVacuous:
    """A clean probe is only evidence if it actually decided something.
    All-HOLD coverage would pass the gate while testing nothing, so the
    branches the probes reach are pinned here."""

    @pytest.mark.parametrize("cfg", enabled_instruments(), ids=lambda c: c.symbol)
    def test_every_instrument_reaches_a_real_exit_decision(self, cfg, lineup_probes):
        held = lineup_probes[cfg.symbol].held_actions
        assert any(action is not Action.HOLD for action in held)

    @pytest.mark.parametrize("cfg", enabled_instruments(), ids=lambda c: c.symbol)
    def test_every_instrument_reaches_a_real_entry_decision(self, cfg, lineup_probes):
        """No single random walk triggers an RSI(2)<10 dip or an EMA cross on a
        sampled bar for most of the lineup; the seed union does."""
        assert Action.LONG in lineup_probes[cfg.symbol].flat_actions

    def test_a_single_seed_reaches_far_fewer_entries(self):
        """Why the union exists: pinning it stops a future 'simplification'
        back to one seed from quietly gutting entry coverage."""
        reached = [Action.LONG in pooled_lookahead_probe(build_strategy(cfg),
                                                         seeds=1).flat_actions
                   for cfg in enabled_instruments()]
        assert sum(reached) < len(reached)


class TestEnabledInstruments:
    def test_a_disabled_instrument_is_left_out(self, monkeypatch):
        voo = next(c for c in INSTRUMENTS if c.symbol == "VOO")
        monkeypatch.setattr(config, "INSTRUMENTS",
                            [replace(voo, enabled=False), replace(voo, symbol="ONE")])
        assert [c.symbol for c in enabled_instruments()] == ["ONE"]


class TestLookaheadCli:
    def test_the_flag_reports_without_touching_market_data(self, monkeypatch):
        """Synthetic bars only: the check has to run keyless in CI."""
        def boom(*a, **k):
            raise AssertionError("MarketData must not be constructed")

        monkeypatch.setattr(diagnose, "MarketData", boom)
        monkeypatch.setattr(diagnose, "report_lookahead", lambda *a, **k: 7)
        monkeypatch.setattr(sys, "argv", ["diagnose", "--lookahead"])
        assert diagnose.main() == 7

    def test_it_is_mutually_exclusive_with_the_other_reports(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["diagnose", "--lookahead", "--slippage"])
        with pytest.raises(SystemExit):
            diagnose.main()

    def test_a_clean_lineup_names_every_enabled_instrument(self, capsys):
        """Runs the operator's own invocation: only the full seed union reaches
        the entry gates, so a thinned run here would report most of the lineup
        as vacuous — truthfully, and pinning nothing about a clean report.

        The trend followers reach a single-digit entry count even over the union
        and so read WEAK, which is coverage commentary and not a failure."""
        assert report_lookahead() == 0
        out = capsys.readouterr().out
        for cfg in enabled_instruments():
            assert cfg.symbol in out
        assert "CLEAN" in out
        assert "VACUOUS" not in out

    def test_the_header_records_what_was_probed(self, capsys):
        """The report is only reproducible if it says which seeds and how many
        bars produced it."""
        report_lookahead(seeds=1)
        out = capsys.readouterr().out
        assert "seed" in out.lower()
        assert str(diagnose.LOOKAHEAD_SAMPLES) in out

    def test_an_all_hold_probe_is_not_reported_clean(self, monkeypatch, capsys):
        """A probe whose every decision is HOLD compared nothing worth
        comparing, and CLEAN there vouches for a strategy nobody examined."""
        monkeypatch.setattr(config, "INSTRUMENTS", [self._voo()])
        monkeypatch.setattr(diagnose, "build_strategy", lambda cfg: _WarmupHungry())
        assert report_lookahead(seeds=1) != 0
        out = capsys.readouterr().out
        assert "VACUOUS" in out
        assert "CLEAN" not in out
        assert "probed nothing" in out

    def test_a_vacuous_line_names_the_coverage_of_each_leg(self, monkeypatch, capsys):
        """Entry and exit rules leak independently, so the line has to say
        which leg went unexercised."""
        monkeypatch.setattr(config, "INSTRUMENTS", [self._voo()])
        monkeypatch.setattr(diagnose, "build_strategy", lambda cfg: _NeverEnters())
        assert report_lookahead(seeds=1) != 0
        out = capsys.readouterr().out
        assert re.search(r"\b0 entry\b", out)
        assert not re.search(r"\b0 exit\b", out)

    def test_the_clean_line_carries_the_coverage_behind_it(self, monkeypatch, capsys):
        """CLEAN means as much as the decisions it was drawn from, so the
        operator gets to see them."""
        monkeypatch.setattr(config, "INSTRUMENTS", [self._voo()])
        monkeypatch.setattr(diagnose, "build_strategy", lambda cfg: _AlwaysDecides())
        assert report_lookahead(seeds=1) == 0
        out = capsys.readouterr().out
        assert "CLEAN" in out
        assert "entry" in out and "exit" in out
        assert not re.search(r"\b0 (entry|exit)\b", out)

    def test_the_clean_line_counts_the_decisions_actually_made(
            self, monkeypatch, capsys):
        """The counts have to come from the probe: a constant pair would read
        the same on a report drawn from a single bar."""
        strategy = _AlwaysDecides()
        bars, stride = probe_frame(strategy)
        probed = len(lookahead_indices(strategy, bars, stride))
        assert probed > diagnose.DECISIVE_FLOOR

        monkeypatch.setattr(config, "INSTRUMENTS", [self._voo()])
        monkeypatch.setattr(diagnose, "build_strategy", lambda cfg: strategy)
        assert report_lookahead(seeds=1) == 0
        assert (f"CLEAN ({probed} entry, {probed} exit decision(s) probed)"
                in capsys.readouterr().out)

    def test_a_barely_decisive_leg_is_reported_weak_without_failing(
            self, monkeypatch, capsys):
        """One entry is not zero, so it is not vacuous — but it is one
        knob-trim away, and the operator has to see that before it lands."""
        monkeypatch.setattr(config, "INSTRUMENTS", [self._voo()])
        monkeypatch.setattr(diagnose, "build_strategy", lambda cfg: _EntersOnce())
        assert report_lookahead(seeds=1) == 0
        out = capsys.readouterr().out
        assert "WEAK" in out
        assert re.search(r"\b1 entry\b", out)
        assert "CLEAN" not in out
        assert "VACUOUS" not in out

    def test_a_leaking_instrument_is_reported_and_fails(self, monkeypatch, capsys):
        monkeypatch.setattr(config, "INSTRUMENTS", [self._voo()])
        monkeypatch.setattr(diagnose, "build_strategy", lambda cfg: _PeeksOnEntry())
        assert report_lookahead(seeds=1) == 1
        out = capsys.readouterr().out
        assert "VOO" in out
        assert "long" in out and "hold" in out
        assert "flat" in out
        assert "could not be checked" not in out

    def test_a_leak_line_names_the_seed_it_came_from(self, monkeypatch, capsys):
        """The line has to carry the recipe: a bar index alone does not say
        which of the twenty walks to regenerate."""
        monkeypatch.setattr(config, "INSTRUMENTS", [self._voo()])
        monkeypatch.setattr(diagnose, "build_strategy", lambda cfg: _PeeksOnEntry())
        report_lookahead(seeds=1)
        assert "seed 0 bar" in capsys.readouterr().out

    def test_an_exit_leak_is_reported_as_the_holding_leg(self, monkeypatch, capsys):
        monkeypatch.setattr(config, "INSTRUMENTS", [self._voo()])
        monkeypatch.setattr(diagnose, "build_strategy", lambda cfg: _PeeksOnExit())
        assert report_lookahead(seeds=1) == 1
        assert "holding" in capsys.readouterr().out

    def test_a_short_capable_strategy_carries_its_cover_coverage(
            self, monkeypatch, capsys):
        """The long leg alone would print CLEAN here with the cover rule run
        zero times, so the count that vouches for it has to be on the line."""
        monkeypatch.setattr(config, "INSTRUMENTS", [self._voo()])
        monkeypatch.setattr(diagnose, "build_strategy", lambda cfg: _ShortsAndCovers())
        assert report_lookahead(seeds=1) == 0
        line = next(ln for ln in capsys.readouterr().out.splitlines()
                    if ln.startswith("VOO"))
        assert "CLEAN" in line
        assert "short-exit" in line
        assert not re.search(r"\b0 short-exit\b", line)

    def test_an_unexercised_cover_rule_cannot_print_clean(self, monkeypatch, capsys):
        monkeypatch.setattr(config, "INSTRUMENTS", [self._voo()])
        monkeypatch.setattr(diagnose, "build_strategy",
                            lambda cfg: _ShortsButNeverCovers())
        assert report_lookahead(seeds=1) != 0
        out = capsys.readouterr().out
        assert "VACUOUS" in out
        assert re.search(r"\b0 short-exit\b", out)
        assert "CLEAN" not in out

    def test_a_cover_leak_is_reported_as_the_short_leg(self, monkeypatch, capsys):
        monkeypatch.setattr(config, "INSTRUMENTS", [self._voo()])
        monkeypatch.setattr(diagnose, "build_strategy", lambda cfg: _PeeksOnShortCover())
        assert report_lookahead(seeds=1) == 1
        assert "holding short" in capsys.readouterr().out

    def test_a_long_only_strategy_reports_no_short_leg(self, monkeypatch, capsys):
        """A strategy that never decides SHORT has no path into a short
        position, so short-leg coverage would describe a branch the engine
        cannot reach."""
        monkeypatch.setattr(config, "INSTRUMENTS", [self._voo()])
        monkeypatch.setattr(diagnose, "build_strategy", lambda cfg: _AlwaysDecides())
        assert report_lookahead(seeds=1) == 0
        line = next(ln for ln in capsys.readouterr().out.splitlines()
                    if ln.startswith("VOO"))
        assert "CLEAN" in line
        assert "short" not in line

    def test_an_unreachable_cover_leak_is_not_a_failure(self, monkeypatch, capsys):
        """Same reasoning from the other end: dead code cannot leak into a
        trade, and failing on it would make the gate unfixable by any means
        except deleting a branch nobody runs."""
        strategy = _PeeksOnAnUnreachableCover()
        probe = pooled_lookahead_probe(strategy, seeds=1)
        assert {lk.leg for lk in probe.leaks} == {"short"}

        monkeypatch.setattr(config, "INSTRUMENTS", [self._voo()])
        monkeypatch.setattr(diagnose, "build_strategy", lambda cfg: strategy)
        assert report_lookahead(seeds=1) == 0
        assert "leaking" not in capsys.readouterr().out

    def test_the_header_says_what_puts_the_short_leg_in_scope(self, capsys):
        report_lookahead(seeds=1)
        assert "enter short" in capsys.readouterr().out.lower()

    def test_a_prepare_that_drops_rows_is_skipped_not_called_clean(
            self, monkeypatch, capsys):
        monkeypatch.setattr(config, "INSTRUMENTS", [self._voo()])
        monkeypatch.setattr(diagnose, "build_strategy", lambda cfg: _DropsRows())
        assert report_lookahead(seeds=1) != 0
        out = capsys.readouterr().out
        assert "SKIP" in out
        assert "could not be checked" in out

    def test_an_unbuildable_strategy_is_skipped_and_still_fails(
            self, monkeypatch, capsys):
        """A lineup entry nobody can build is a gap in the check, not a pass:
        an exit code of 0 here would report a clean bill of health for a
        strategy that was never examined."""
        monkeypatch.setattr(config, "INSTRUMENTS", [self._voo()])

        def unbuildable(cfg):
            raise ValueError("needs params nobody supplied")

        monkeypatch.setattr(diagnose, "build_strategy", unbuildable)
        assert report_lookahead() != 0
        out = capsys.readouterr().out
        assert "SKIP" in out
        assert "could not be checked" in out

    def test_a_probe_that_raises_is_not_reported_as_clean(self, monkeypatch, capsys):
        """A strategy reading df.iloc[i+1] raises on the truncated frame. That
        is the loudest possible leak, so it must not exit 0."""
        monkeypatch.setattr(config, "INSTRUMENTS", [self._voo()])

        def raises(*a, **k):
            raise IndexError("single positional indexer is out-of-bounds")

        monkeypatch.setattr(diagnose, "pooled_lookahead_probe", raises)
        assert report_lookahead() != 0
        out = capsys.readouterr().out
        assert "out-of-bounds" in out
        assert "could not be checked" in out

    def test_the_two_kinds_of_skip_read_differently(self, monkeypatch, capsys):
        """Whether the strategy could not be built or the probe blew up on it
        points at different faults, so the line has to say which."""
        monkeypatch.setattr(config, "INSTRUMENTS", [self._voo()])

        def raises(*a, **k):
            raise IndexError("out-of-bounds")

        monkeypatch.setattr(diagnose, "pooled_lookahead_probe", raises)
        report_lookahead()
        assert "probe" in capsys.readouterr().out.lower()

        def unbuildable(cfg):
            raise ValueError("no params")

        monkeypatch.setattr(diagnose, "build_strategy", unbuildable)
        report_lookahead()
        assert "build" in capsys.readouterr().out.lower()

    def _voo(self):
        return next(c for c in INSTRUMENTS if c.symbol == "VOO")


class TestSweepCoversEveryConfiguredSymbol:
    def test_a_symbol_with_no_variants_is_skipped_not_a_traceback(
            self, monkeypatch, capsys):
        """VOO is in the lineup with no sweep variants of its own. The headline
        `python -m bot.diagnose` invocation has to survive that: a sweep the
        lineup can outgrow is a diagnostic nobody can run."""
        voo = next(c for c in INSTRUMENTS if c.symbol == "VOO")
        assert voo.symbol not in diagnose.VARIANTS

        def unusable_market_data(*args, **kwargs):
            raise AssertionError("no bars should be fetched for a skipped symbol")

        monkeypatch.setattr(config, "INSTRUMENTS", [voo])
        monkeypatch.setattr(diagnose, "MarketData", lambda *a, **k: None)
        monkeypatch.setattr(diagnose, "sweep_history_start", unusable_market_data)
        monkeypatch.setenv("ALPACA_API_KEY", "key")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
        monkeypatch.setattr(sys, "argv", ["diagnose"])

        assert diagnose.main() == 0

        out = capsys.readouterr().out
        assert "VOO" in out
        assert "SKIP" in out


class _StubMarketData:
    """Answers every fetch with one frame, or raises what it was handed."""

    def __init__(self, bars=None, error=None):
        self.bars, self.error = bars, error

    def get_bars(self, *args, **kwargs):
        if self.error is not None:
            raise self.error
        return self.bars


class _SkipsOneSymbol:
    """Answers every fetch but one, so a partial gap in the report can be told
    apart from a total one."""

    def __init__(self, bars, failing):
        self.bars, self.failing = bars, failing

    def get_bars(self, symbol, *args, **kwargs):
        if symbol == self.failing:
            raise RuntimeError("no bars for this symbol")
        return self.bars


class TestStartupWindowReport:
    def _voo(self):
        return next(c for c in INSTRUMENTS if c.symbol == "VOO")

    def test_one_instrument_skipping_among_many_is_not_clean(
            self, monkeypatch, capsys):
        """Measuring all but one is a gap in the check, and exiting 0 hands
        whatever runs this the same answer as no drift anywhere."""
        voo = self._voo()
        monkeypatch.setattr(config, "INSTRUMENTS", [voo, replace(voo, symbol="ONE")])
        monkeypatch.setattr(diagnose, "build_strategy", lambda cfg: _WarmupHungry())

        assert report_startup_windows(_SkipsOneSymbol(_bars(200), "ONE")) != 0

        out = capsys.readouterr().out
        assert "sma_50" in out
        assert "SKIP" in out
        assert "1 instrument(s) could not be measured." in out
        assert "No instrument could be checked" not in out

    def test_an_empty_lineup_is_not_a_clean_report(self, monkeypatch, capsys):
        """Nothing measured is not the same answer as no drift, and exiting 0
        makes the two indistinguishable to whatever runs this."""
        monkeypatch.setattr(config, "INSTRUMENTS", [])
        assert report_startup_windows(_StubMarketData()) != 0
        assert "No instrument could be checked" in capsys.readouterr().out

    def test_a_lineup_that_skips_every_instrument_is_not_clean(
            self, monkeypatch, capsys):
        monkeypatch.setattr(config, "INSTRUMENTS", [self._voo()])
        monkeypatch.setattr(diagnose, "build_strategy", lambda cfg: _WarmupHungry())
        md = _StubMarketData(error=RuntimeError("no bars for this symbol"))

        assert report_startup_windows(md) != 0

        out = capsys.readouterr().out
        assert "SKIP" in out
        assert "No instrument could be checked" in out

    def test_an_unbuildable_strategy_is_skipped_not_a_traceback(
            self, monkeypatch, capsys):
        """One lineup entry nobody can build must cost that row, not the drift
        every other instrument would have reported."""
        voo = self._voo()
        monkeypatch.setattr(config, "INSTRUMENTS", [voo, replace(voo, symbol="ONE")])

        def unbuildable(cfg):
            if cfg.symbol == "ONE":
                raise ValueError("needs params nobody supplied")
            return _WarmupHungry()

        monkeypatch.setattr(diagnose, "build_strategy", unbuildable)

        assert report_startup_windows(_StubMarketData(_bars(200))) != 0

        out = capsys.readouterr().out
        assert "SKIP: cannot build the strategy: needs params nobody supplied" in out
        assert "sma_50" in out

    def test_the_two_kinds_of_skip_read_differently(self, monkeypatch, capsys):
        """A config typo raises a bare KeyError whose message is just the key
        name; without a staged prefix that is indistinguishable from a data
        outage, and the two point at different faults."""
        monkeypatch.setattr(config, "INSTRUMENTS", [self._voo()])

        def unbuildable(cfg):
            raise KeyError("fast_period")

        monkeypatch.setattr(diagnose, "build_strategy", unbuildable)
        report_startup_windows(_StubMarketData(_bars(200)))
        assert "build" in capsys.readouterr().out.lower()

        monkeypatch.setattr(diagnose, "build_strategy", lambda cfg: _WarmupHungry())
        report_startup_windows(_StubMarketData(error=RuntimeError("no bars")))
        out = capsys.readouterr().out
        assert "no bars" in out
        assert "build" not in out.lower()

    def test_a_measured_instrument_exits_clean_and_reports_its_drift(
            self, monkeypatch, capsys):
        monkeypatch.setattr(config, "INSTRUMENTS", [self._voo()])
        monkeypatch.setattr(diagnose, "build_strategy", lambda cfg: _WarmupHungry())

        assert report_startup_windows(_StubMarketData(_bars(200))) == 0

        out = capsys.readouterr().out
        assert "sma_50" in out
        assert "No instrument could be checked" not in out


class TestStartupWindowDriftThreshold:
    """Opt-in gate: the coverage-only exit is right for a human reading the
    table and wrong for cron, which sees 0 and learns nothing about the drift."""

    def _lineup(self, monkeypatch, strategy):
        voo = next(c for c in INSTRUMENTS if c.symbol == "VOO")
        monkeypatch.setattr(config, "INSTRUMENTS", [voo])
        monkeypatch.setattr(diagnose, "build_strategy", lambda cfg: strategy)

    def test_a_drift_under_the_threshold_exits_clean(self, monkeypatch):
        # _ScaledDrift(0.01) drifts 0.01/1.01 relative: just under 1%.
        self._lineup(monkeypatch, _ScaledDrift(0.01))
        md = _StubMarketData(_bars(200))
        assert report_startup_windows(md, fail_on_drift_pct=5.0) == 0

    def test_a_drift_past_the_threshold_fails(self, monkeypatch):
        self._lineup(monkeypatch, _ScaledDrift(0.01))
        md = _StubMarketData(_bars(200))
        assert report_startup_windows(md, fail_on_drift_pct=0.1) != 0

    def test_the_failing_line_names_the_column_its_drift_and_the_threshold(
            self, monkeypatch, capsys):
        """A nonzero exit nobody can explain sends the operator back to the
        table to work out which number tripped it."""
        self._lineup(monkeypatch, _ScaledDrift(0.01))
        report_startup_windows(_StubMarketData(_bars(200)), fail_on_drift_pct=0.1)
        line = capsys.readouterr().out.strip().splitlines()[-1]
        assert "scaled" in line
        assert "VOO" in line
        assert "0.990" in line
        assert "0.1%" in line

    def test_the_same_drift_without_the_flag_still_exits_clean(self, monkeypatch):
        """The gate is opt-in: the report a human runs by hand must not start
        failing on a finding it has always just printed."""
        self._lineup(monkeypatch, _ScaledDrift(0.01))
        assert report_startup_windows(_StubMarketData(_bars(200))) == 0

    def test_a_threshold_nothing_breaches_leaves_the_report_untouched(
            self, monkeypatch, capsys):
        self._lineup(monkeypatch, _ScaledDrift(0.01))
        report_startup_windows(_StubMarketData(_bars(200)))
        plain = capsys.readouterr().out
        report_startup_windows(_StubMarketData(_bars(200)), fail_on_drift_pct=5.0)
        assert capsys.readouterr().out == plain

    def test_a_threshold_of_zero_fails_on_any_drift_the_report_lists(
            self, monkeypatch):
        """0 is not 'off': it is the strictest setting, and a report listing a
        drifted column at all has something to fail on."""
        self._lineup(monkeypatch, _ScaledDrift(1e-5))
        assert report_startup_windows(_StubMarketData(_bars(200)),
                                      fail_on_drift_pct=0.0) != 0

    def test_a_drift_below_the_report_tolerance_passes_a_zero_threshold(
            self, monkeypatch):
        """Nothing is listed, so there is nothing to fail on: the gate cannot
        be stricter than the measurement behind it."""
        self._lineup(monkeypatch, _ScaledDrift(1e-7))
        assert report_startup_windows(_StubMarketData(_bars(200)),
                                      fail_on_drift_pct=0.0) == 0

    def test_a_column_absent_from_the_min_bars_window_fails_any_threshold(
            self, monkeypatch, capsys):
        """sma_50 has no value at all on 20 bars. That is not a small drift to
        be waved through by a generous threshold — the indicator is not there."""
        self._lineup(monkeypatch, _WarmupHungry())
        assert report_startup_windows(_StubMarketData(_bars(200)),
                                      fail_on_drift_pct=1e9) != 0
        assert "sma_50" in capsys.readouterr().out.strip().splitlines()[-1]

    def test_a_coverage_gap_still_fails_under_a_generous_threshold(
            self, monkeypatch):
        """The drift gate is added to the coverage one, not swapped for it."""
        voo = next(c for c in INSTRUMENTS if c.symbol == "VOO")
        monkeypatch.setattr(config, "INSTRUMENTS", [voo])
        monkeypatch.setattr(diagnose, "build_strategy", lambda cfg: _WarmupHungry())
        md = _StubMarketData(error=RuntimeError("no bars"))
        assert report_startup_windows(md, fail_on_drift_pct=1e9) != 0


class TestStartupWindowCli:
    def test_the_threshold_reaches_the_report(self, monkeypatch):
        seen = {}

        def fake(md, fail_on_drift_pct=None):
            seen["pct"] = fail_on_drift_pct
            return 3

        monkeypatch.setattr(diagnose, "MarketData", lambda *a, **k: None)
        monkeypatch.setattr(diagnose, "report_startup_windows", fake)
        monkeypatch.setattr(sys, "argv",
                            ["diagnose", "--startup-window", "--fail-on-drift", "0.5"])
        assert diagnose.main() == 3
        assert seen["pct"] == 0.5

    def test_the_flagless_run_gates_on_nothing(self, monkeypatch):
        seen = {}

        def fake(md, fail_on_drift_pct=None):
            seen["pct"] = fail_on_drift_pct
            return 0

        monkeypatch.setattr(diagnose, "MarketData", lambda *a, **k: None)
        monkeypatch.setattr(diagnose, "report_startup_windows", fake)
        monkeypatch.setattr(sys, "argv", ["diagnose", "--startup-window"])
        assert diagnose.main() == 0
        assert seen["pct"] is None

    def test_a_threshold_without_the_report_it_gates_is_rejected(self, monkeypatch):
        """Accepting it silently would give cron a threshold that never runs."""
        monkeypatch.setattr(sys, "argv", ["diagnose", "--fail-on-drift", "0.5"])
        with pytest.raises(SystemExit):
            diagnose.main()

    def test_a_negative_threshold_is_rejected(self, monkeypatch):
        monkeypatch.setattr(sys, "argv",
                            ["diagnose", "--startup-window", "--fail-on-drift", "-1"])
        with pytest.raises(SystemExit):
            diagnose.main()
