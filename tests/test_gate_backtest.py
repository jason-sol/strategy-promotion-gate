import json
import math
import os
from dataclasses import replace
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from bot import gate_backtest
from bot.gate_backtest import (
    CeilingRow,
    Calibration,
    GateBacktest,
    GateRow,
    SkippedCell,
    Split,
)
from bot.research import EXPERIMENTS, make_cfg, validate_symbol
from bot.strategies import build_strategy
from bot.strategies.base import HOLD, Action, Signal, Strategy
from bot.validation import Candidate, build_return_matrix
from config import MARKET_DATA_FEED, VALIDATION, AssetClass, InstrumentConfig, StrategyName
from helpers import daily_bars

# A short two-variant field and a business-day frame: what is under test is the
# replay's shape, not a strategy's edge, and the full shortlist over a decade of
# bars is a minutes-long run.
N_BLOCKS = 4
N_FOLDS = 2
CUTOFF = date(2020, 12, 31)
SYMBOL = "AAA"


def mr_cfg(symbol: str, entry_z: float,
           sma_period: int = 10) -> InstrumentConfig:
    return InstrumentConfig(
        symbol=symbol, asset_class=AssetClass.US_EQUITY,
        strategy=StrategyName.MEAN_REVERSION, timeframe_minutes=1440,
        params={"sma_period": sma_period, "entry_z": entry_z,
                "allow_short": False},
        slippage_pct=0.0,
    )


def mr_variant(label: str, sma_period: int):
    """A variant whose min_bars is sma_period + 1, so a field can be built
    around a chosen warm-up."""
    return (label, lambda: build_strategy(mr_cfg(SYMBOL, 1.0, sma_period)))


def choppy_uptrend(n: int) -> list[float]:
    t = np.arange(n)
    return list(100.0 + 0.05 * t + 4.0 * np.sin(t / 5.0) + 1.5 * np.sin(t / 1.7))


VARIANTS = [
    ("z1.0", lambda: build_strategy(mr_cfg(SYMBOL, 1.0))),
    ("z2.0", lambda: build_strategy(mr_cfg(SYMBOL, 2.0))),
]


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    # 1000 business days from 2019-01-01: ~2019-01 -> ~2022-10, which carries
    # the 2020-12-31 cutoff with a full year of history and a full year forward.
    return daily_bars(choppy_uptrend(1000), start="2019-01-01")


@pytest.fixture(scope="module")
def long_bars() -> pd.DataFrame:
    # 1700 business days from 2015-01-01 (-> 2021-07). The 2019-09-30 cutoff
    # leaves 1238 history bars, so walk-forward's first train block is 412 over
    # two folds: room for a 210-bar warm-up and not for an 800-bar one.
    return daily_bars(choppy_uptrend(1700), start="2015-01-01")


LONG_CUTOFF = date(2019, 9, 30)
LONG_FIRST_TRAIN_BARS = 412
MIXED_VARIANTS = [mr_variant("slow-800", 799), mr_variant("fast-210", 209)]
ALL_FLAT_VARIANTS = [mr_variant("slow-800", 799), mr_variant("slow-900", 899)]


@pytest.fixture(scope="module")
def cell(bars) -> GateRow:
    return gate_backtest.evaluate_cell(SYMBOL, AssetClass.US_EQUITY, 0.0,
                                       VARIANTS, bars, CUTOFF,
                                       n_blocks=N_BLOCKS, n_folds=N_FOLDS)


class FirstRowAnchor(Strategy):
    """Every decision is measured against the close of the FIRST row of whatever
    frame prepare() was handed, so its returns change when the frame starts
    somewhere else — which is exactly what a warm-up carve-out must preserve."""

    @property
    def min_bars(self) -> int:
        return 2

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        df["atr"] = 1.0
        df["anchor"] = float(bars["close"].iloc[0])
        return df

    def signal(self, df, i, position):
        row = df.iloc[i]
        if position is None:
            return (Signal(Action.LONG, "under anchor")
                    if row["close"] < row["anchor"] else HOLD)
        return (Signal(Action.EXIT, "over anchor")
                if row["close"] >= row["anchor"] else HOLD)


class FixedScheduleStrategy(Strategy):
    """Base for the trade-schedule fixtures: a flat ATR and a hard stop far
    enough away that it never fires, so what the schedule says happens is what
    happens."""

    @property
    def min_bars(self) -> int:
        return 2

    @property
    def stop_atr_mult(self) -> float:
        return 50.0

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        df["atr"] = 1.0
        return df


class EveryTenthBar(FixedScheduleStrategy):
    """Enters long on every tenth bar of whatever frame it is handed and exits
    on the next one, so a window of known length holds a known number of round
    trips."""

    PERIOD = 10

    def signal(self, df, i, position):
        if position is None:
            return (Signal(Action.LONG, "cycle") if i % self.PERIOD == 0
                    else HOLD)
        return Signal(Action.EXIT, "cycle")


class EntersOnceNeverExits(FixedScheduleStrategy):
    """Enters on its first chance and holds to the end of the window: real
    exposure with no completed round trip behind it."""

    def __init__(self):
        self.entered = False

    def signal(self, df, i, position):
        if position is not None or self.entered:
            return HOLD
        self.entered = True
        return Signal(Action.LONG, "once")


class NeverTrades(FixedScheduleStrategy):
    """Holds through everything: a flat forward curve with nothing behind it,
    which is the case the exposure count exists to tell apart."""

    def signal(self, df, i, position):
        return HOLD


class EntersOnLastBar(FixedScheduleStrategy):
    """Enters on the final bar of the frame: no round trip closes and no bar is
    carried into a later decision, yet the entry's slippage lands in the last
    return."""

    def signal(self, df, i, position):
        if position is None and i == len(df) - 1:
            return Signal(Action.LONG, "last bar")
        return HOLD


class TestSplitBars:
    def test_history_stops_at_the_cutoff_and_forward_starts_after(self, bars):
        history, forward = gate_backtest.split_bars(bars, CUTOFF)
        assert history.index[-1].date() <= CUTOFF
        assert forward.index[0].date() > CUTOFF
        assert len(history) + len(forward) < len(bars)  # trailing bars withheld

    def test_forward_window_ends_forward_days_after_the_cutoff(self, bars):
        _, forward = gate_backtest.split_bars(bars, CUTOFF, forward_days=365)
        assert forward.index[-1].date() <= date(2021, 12, 31)
        assert forward.index[-1].date() > date(2021, 12, 24)

    def test_forward_days_is_honoured(self, bars):
        _, short = gate_backtest.split_bars(bars, CUTOFF, forward_days=90)
        _, long_ = gate_backtest.split_bars(bars, CUTOFF, forward_days=365)
        assert len(short) < len(long_)


class TestFeasibility:
    def test_short_history_is_skipped_not_scored(self, bars):
        # A month of history cannot warm the shortest variant in the field
        # before the first test fold, so nothing in it can trade.
        result = gate_backtest.evaluate_cell(
            SYMBOL, AssetClass.US_EQUITY, 0.0, VARIANTS, bars,
            date(2019, 1, 31), n_blocks=N_BLOCKS, n_folds=N_FOLDS)
        assert isinstance(result, SkippedCell)
        assert result.cutoff == date(2019, 1, 31)
        assert "warm-up" in result.reason
        assert "15" in result.reason and "7-bar" in result.reason

    def test_short_forward_window_is_skipped_not_scored(self, bars):
        # The frame ends ~2022-10, so a 2022-06-30 cutoff leaves under 200
        # forward bars.
        result = gate_backtest.evaluate_cell(
            SYMBOL, AssetClass.US_EQUITY, 0.0, VARIANTS, bars,
            date(2022, 6, 30), n_blocks=N_BLOCKS, n_folds=N_FOLDS)
        assert isinstance(result, SkippedCell)
        assert "forward" in result.reason

    def test_crypto_needs_more_forward_bars_than_equities(self, bars):
        # ~261 business-day forward bars: enough for an equity row, short of the
        # crypto bound, which counts calendar days.
        equity = gate_backtest.evaluate_cell(
            SYMBOL, AssetClass.US_EQUITY, 0.0, VARIANTS, bars, CUTOFF,
            n_blocks=N_BLOCKS, n_folds=N_FOLDS)
        crypto = gate_backtest.evaluate_cell(
            SYMBOL, AssetClass.CRYPTO, 0.0, VARIANTS, bars, CUTOFF,
            n_blocks=N_BLOCKS, n_folds=N_FOLDS)
        assert isinstance(equity, GateRow)
        assert isinstance(crypto, SkippedCell)
        assert str(gate_backtest.MIN_FORWARD_BARS_CRYPTO) in crypto.reason

    def test_cscv_block_floor_is_enforced(self, bars):
        assert gate_backtest.history_shortfall(bars.iloc[:7], VARIANTS,
                                               n_blocks=N_BLOCKS,
                                               n_folds=N_FOLDS) is not None
        assert gate_backtest.history_shortfall(bars, VARIANTS,
                                               n_blocks=N_BLOCKS,
                                               n_folds=N_FOLDS) is None


class TestWarmupDisclosure:
    def test_a_variant_that_cannot_warm_is_disclosed_not_skipped(self, long_bars):
        # research runs these fields as they are, so the replay has to score
        # them too: the 800-bar variant reads flat in the earliest train folds
        # and is reported, not used as grounds to drop the cell.
        result = gate_backtest.evaluate_cell(
            SYMBOL, AssetClass.US_EQUITY, 0.0, MIXED_VARIANTS, long_bars,
            LONG_CUTOFF, n_blocks=N_BLOCKS, n_folds=N_FOLDS)
        assert isinstance(result, GateRow)
        assert result.first_train_bars == LONG_FIRST_TRAIN_BARS
        assert result.warmup_flat_variants == ("slow-800",)

    def test_a_field_with_one_feasible_variant_is_not_a_shortfall(self, long_bars):
        history, _ = gate_backtest.split_bars(long_bars, LONG_CUTOFF)
        assert gate_backtest.history_shortfall(
            history, MIXED_VARIANTS, n_blocks=N_BLOCKS,
            n_folds=N_FOLDS) is None

    def test_a_field_where_nothing_can_warm_is_skipped(self, long_bars):
        result = gate_backtest.evaluate_cell(
            SYMBOL, AssetClass.US_EQUITY, 0.0, ALL_FLAT_VARIANTS, long_bars,
            LONG_CUTOFF, n_blocks=N_BLOCKS, n_folds=N_FOLDS)
        assert isinstance(result, SkippedCell)
        assert "800" in result.reason
        assert str(LONG_FIRST_TRAIN_BARS) in result.reason

    def test_first_train_bars_follows_walk_forwards_own_fold_split(self, long_bars):
        history, _ = gate_backtest.split_bars(long_bars, LONG_CUTOFF)
        assert gate_backtest.first_train_bars(history, N_FOLDS) == \
            len(history) // (N_FOLDS + 1)

    def test_flat_labels_are_the_ones_outrunning_the_first_train_block(self):
        assert gate_backtest.warmup_flat_variants(MIXED_VARIANTS, 412) == \
            ("slow-800",)
        assert gate_backtest.warmup_flat_variants(MIXED_VARIANTS, 900) == ()
        assert gate_backtest.warmup_flat_variants(MIXED_VARIANTS, 100) == \
            ("slow-800", "fast-210")


class TestGateLeg:
    def test_the_gate_sees_only_bars_at_or_before_the_cutoff(self, bars, monkeypatch):
        seen = {}
        real = gate_backtest.validate_symbol

        def spy(symbol, asset_class, slippage, variants, frame, **kwargs):
            seen["frame"] = frame
            return real(symbol, asset_class, slippage, variants, frame, **kwargs)

        monkeypatch.setattr(gate_backtest, "validate_symbol", spy)
        gate_backtest.evaluate_cell(SYMBOL, AssetClass.US_EQUITY, 0.0, VARIANTS,
                                    bars, CUTOFF, n_blocks=N_BLOCKS,
                                    n_folds=N_FOLDS)
        history, _ = gate_backtest.split_bars(bars, CUTOFF)
        assert seen["frame"].equals(history)
        assert seen["frame"].index[-1].date() <= CUTOFF

    def test_row_carries_the_gate_inputs_the_harness_computed(self, bars, cell):
        history, forward = gate_backtest.split_bars(bars, CUTOFF)
        expected = validate_symbol(SYMBOL, AssetClass.US_EQUITY, 0.0, VARIANTS,
                                   history, n_blocks=N_BLOCKS, n_folds=N_FOLDS)
        assert cell.variant == expected.best_label
        assert cell.dsr == pytest.approx(expected.best_dsr, nan_ok=True)
        assert cell.p_oos_loss == pytest.approx(expected.p_oos_loss, nan_ok=True)
        assert cell.pbo == pytest.approx(expected.pbo, nan_ok=True)
        assert cell.oos_sharpe == pytest.approx(expected.oos_sharpe, nan_ok=True)
        assert cell.promoted == expected.promoted
        assert cell.history_bars == len(history)
        assert cell.forward_bars == len(forward)


class TestForwardLeg:
    def test_forward_run_warms_indicators_on_the_pre_cutoff_history(self, bars):
        cand = Candidate("anchor", make_cfg(SYMBOL, AssetClass.US_EQUITY, 0.0),
                         FirstRowAnchor())
        _, forward = gate_backtest.split_bars(bars, CUTOFF)
        warmed = gate_backtest.forward_run(cand, bars, forward).returns
        cold = gate_backtest.forward_run(cand, forward, forward).returns
        assert list(warmed.index) == list(cold.index)
        assert not warmed.equals(cold)

    def test_forward_run_trades_only_after_the_cutoff(self, bars):
        cand = Candidate("z1.0", make_cfg(SYMBOL, AssetClass.US_EQUITY, 0.0),
                         build_strategy(mr_cfg(SYMBOL, 1.0)))
        _, forward = gate_backtest.split_bars(bars, CUTOFF)
        returns = gate_backtest.forward_run(cand, bars, forward).returns
        assert returns.index[0] == forward.index[0]
        assert returns.index[0].date() > CUTOFF
        assert returns.index[-1].date() <= forward.index[-1].date()

    def test_forward_run_uses_the_selected_variant(self, bars, monkeypatch):
        seen = {}
        real = gate_backtest.forward_run

        def spy(candidate, frame, forward):
            seen["label"] = candidate.label
            seen["frame"] = frame
            seen["forward"] = forward
            return real(candidate, frame, forward)

        monkeypatch.setattr(gate_backtest, "forward_run", spy)
        row = gate_backtest.evaluate_cell(SYMBOL, AssetClass.US_EQUITY, 0.0,
                                          VARIANTS, bars, CUTOFF,
                                          n_blocks=N_BLOCKS, n_folds=N_FOLDS)
        _, forward = gate_backtest.split_bars(bars, CUTOFF)
        assert seen["label"] == row.variant
        assert seen["forward"].index[0] == forward.index[0]
        # The forward frame starts at the very first bar, not at the cutoff.
        assert seen["frame"].index[0] == bars.index[0]

    def test_row_records_forward_sharpe_return_and_buy_and_hold(self, cell):
        assert math.isfinite(cell.forward_sharpe)
        assert math.isfinite(cell.forward_return)
        assert math.isfinite(cell.buy_and_hold_sharpe)
        assert cell.forward_return > -1.0

    def test_forward_run_counts_the_round_trips_the_variant_made(self, bars):
        cand = Candidate("cycle", make_cfg(SYMBOL, AssetClass.US_EQUITY, 0.0),
                         EveryTenthBar())
        _, forward = gate_backtest.split_bars(bars, CUTOFF)
        frame = bars.loc[:forward.index[-1]]
        # Entries land on every tenth bar of the prepared frame and close on
        # the next one, so one opened on the final bar never round-trips.
        first_tradable = len(frame) - len(forward)
        expected = sum(1 for i in range(first_tradable, len(frame) - 1)
                       if i % EveryTenthBar.PERIOD == 0)
        run = gate_backtest.forward_run(cand, bars, forward)
        assert expected > 0
        assert run.trades == expected
        # Each round trip carries exactly one bar into its own exit decision.
        assert run.exposure_bars == expected

    def test_a_position_still_open_at_the_end_is_exposure_not_a_round_trip(
            self, bars):
        cand = Candidate("hold", make_cfg(SYMBOL, AssetClass.US_EQUITY, 0.0),
                         EntersOnceNeverExits())
        _, forward = gate_backtest.split_bars(bars, CUTOFF)
        run = gate_backtest.forward_run(cand, bars, forward)
        assert run.trades == 0
        assert run.exposure_bars > 0
        assert gate_backtest.sharpe_ratio(run.returns) != 0.0

    def test_a_variant_that_never_fires_reads_as_zero_trades(self, bars):
        cand = Candidate("flat", make_cfg(SYMBOL, AssetClass.US_EQUITY, 0.0),
                         NeverTrades())
        _, forward = gate_backtest.split_bars(bars, CUTOFF)
        run = gate_backtest.forward_run(cand, bars, forward)
        assert run.trades == 0
        assert run.exposure_bars == 0
        assert run.entry_signals == 0
        assert gate_backtest.sharpe_ratio(run.returns) == 0.0

    def test_an_entry_on_the_final_bar_is_counted_but_not_a_measurement(
            self, bars):
        # Slippage so the entry moves the last day's equity: the Sharpe that
        # falls out is that one tick, which is why the count is disclosure and
        # not evidence the forward window was measured.
        cand = Candidate("last", make_cfg(SYMBOL, AssetClass.US_EQUITY, 0.01),
                         EntersOnLastBar())
        _, forward = gate_backtest.split_bars(bars, CUTOFF)
        run = gate_backtest.forward_run(cand, bars, forward)
        assert run.trades == 0
        assert run.exposure_bars == 0
        assert run.entry_signals == 1
        assert gate_backtest.sharpe_ratio(run.returns) != 0.0
        assert row(forward_trades=0, forward_exposure_bars=0,
                   forward_entry_signals=1).traded is False

    def test_the_probe_does_not_change_what_the_variant_does(self, bars):
        cand = Candidate("z1.0", make_cfg(SYMBOL, AssetClass.US_EQUITY, 0.0),
                         build_strategy(mr_cfg(SYMBOL, 1.0)))
        _, forward = gate_backtest.split_bars(bars, CUTOFF)
        frame = bars.loc[:forward.index[-1]]
        run = gate_backtest.forward_run(cand, bars, forward)
        matrix = build_return_matrix([cand], {SYMBOL: frame},
                                     sim_start=forward.index[0])
        assert run.returns.equals(matrix["z1.0"])

    def test_row_discloses_how_many_times_the_selected_variant_traded(self, cell):
        assert cell.forward_trades > 0
        assert cell.forward_exposure_bars > 0
        assert cell.forward_entry_signals > 0
        assert cell.traded

    def test_the_row_takes_each_count_from_its_own_field_of_the_run(
            self, bars, monkeypatch):
        # Three distinct numbers, so a field wired to the wrong one shows up.
        run = gate_backtest.ForwardRun(
            returns=pd.Series([0.01, -0.005, 0.02]),
            trades=3, exposure_bars=17, entry_signals=5)
        monkeypatch.setattr(gate_backtest, "forward_run",
                            lambda *a, **k: run)
        result = gate_backtest.evaluate_cell(
            SYMBOL, AssetClass.US_EQUITY, 0.0, VARIANTS, bars, CUTOFF,
            n_blocks=N_BLOCKS, n_folds=N_FOLDS)
        assert (result.forward_trades, result.forward_exposure_bars,
                result.forward_entry_signals) == (3, 17, 5)


class TestRunGateBacktest:
    def test_scores_the_feasible_cells_and_records_the_rest_as_skipped(self, bars):
        grouped = {SYMBOL: (AssetClass.US_EQUITY, 0.0, VARIANTS)}
        result = gate_backtest.run_gate_backtest(
            grouped, {SYMBOL: bars}, [date(2019, 1, 31), CUTOFF],
            n_blocks=N_BLOCKS, n_folds=N_FOLDS)
        assert [r.cutoff for r in result.rows] == [CUTOFF]
        assert [s.cutoff for s in result.skipped] == [date(2019, 1, 31)]
        assert result.calibration.n_rows == 1
        assert result.forward_days == gate_backtest.FORWARD_DAYS

    def test_a_symbol_without_bars_is_not_invented(self, bars):
        grouped = {SYMBOL: (AssetClass.US_EQUITY, 0.0, VARIANTS),
                   "BBB": (AssetClass.US_EQUITY, 0.0, VARIANTS)}
        result = gate_backtest.run_gate_backtest(
            grouped, {SYMBOL: bars}, [CUTOFF], n_blocks=N_BLOCKS,
            n_folds=N_FOLDS)
        assert {r.symbol for r in result.rows} == {SYMBOL}
        assert all(s.symbol != "BBB" for s in result.skipped)


class TestSpearman:
    def test_monotone_series_correlate_perfectly(self):
        assert gate_backtest.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
        assert gate_backtest.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0

    def test_rank_not_value_is_what_is_correlated(self):
        assert gate_backtest.spearman([1, 2, 3, 4], [1, 2, 3, 400]) == 1.0

    def test_ties_share_their_average_rank(self):
        # Ranks 1, 2.5, 2.5, 4 against 1, 2, 3, 4.
        rho = gate_backtest.spearman([1, 2, 2, 3], [1, 2, 3, 4])
        assert rho == pytest.approx(0.9486832980505138)

    def test_a_constant_series_has_no_ranking_to_correlate(self):
        assert math.isnan(gate_backtest.spearman([1, 1, 1], [1, 2, 3]))

    def test_fewer_than_two_pairs_is_not_computable(self):
        assert math.isnan(gate_backtest.spearman([1.0], [2.0]))
        assert math.isnan(gate_backtest.spearman([], []))

    def test_nan_pairs_are_dropped_not_ranked(self):
        nan = float("nan")
        assert gate_backtest.spearman([1, 2, nan, 3], [1, 2, 5, 3]) == 1.0


def row(cutoff_year=2020, symbol="SPY", dsr=0.97, p_oos_loss=0.2,
        pbo=0.4, oos_sharpe=0.5, forward_sharpe=1.0, forward_trades=12,
        forward_exposure_bars=120, forward_entry_signals=12,
        warmup_flat=("ema50/200",)) -> GateRow:
    """A row with no backtest behind it: the calibration layer reads numbers."""
    return GateRow(cutoff=date(cutoff_year, 12, 31), symbol=symbol,
                   variant="v", dsr=dsr, p_oos_loss=p_oos_loss, pbo=pbo,
                   oos_sharpe=oos_sharpe, forward_sharpe=forward_sharpe,
                   forward_return=0.1, forward_trades=forward_trades,
                   forward_exposure_bars=forward_exposure_bars,
                   forward_entry_signals=forward_entry_signals,
                   buy_and_hold_sharpe=0.3,
                   history_bars=500, forward_bars=250, first_train_bars=100,
                   warmup_flat_variants=tuple(warmup_flat))


class TestSplit:
    def test_an_empty_bucket_is_nan_not_zero(self):
        split = gate_backtest.split_rows([row(dsr=0.99)], lambda r: True)
        assert split.n_pass == 1 and split.n_fail == 0
        assert split.mean_pass == 1.0
        assert math.isnan(split.mean_fail)
        assert math.isnan(split.separation)

    def test_separation_is_the_gap_between_the_buckets(self):
        rows = [row(p_oos_loss=0.1, forward_sharpe=2.0),
                row(p_oos_loss=0.9, forward_sharpe=0.0)]
        split = gate_backtest.split_rows(rows, lambda r: r.p_oos_loss <= 0.5)
        assert (split.n_pass, split.n_fail) == (1, 1)
        assert split.separation == pytest.approx(2.0)


class TestCalibration:
    def test_ceiling_table_counts_move_with_the_ceiling(self):
        rows = [row(p_oos_loss=p, forward_sharpe=1.0 - p)
                for p in (0.02, 0.15, 0.25, 0.45, 0.48)]
        cal = gate_backtest.calibrate(rows)
        by_ceiling = {c.ceiling: c.full for c in cal.ceilings}
        assert by_ceiling[0.05].n_pass == 1
        assert by_ceiling[0.2].n_pass == 2
        assert by_ceiling[0.3].n_pass == 3
        assert by_ceiling[0.5].n_pass == 5
        assert all(c.full.n_pass + c.full.n_fail == 5 for c in cal.ceilings)

    def test_changed_counts_verdicts_that_flip_against_the_live_ceiling(self):
        rows = [row(p_oos_loss=p) for p in (0.02, 0.15, 0.25, 0.45, 0.48)]
        cal = gate_backtest.calibrate(rows)
        changed = {c.ceiling: c.changed for c in cal.ceilings}
        assert changed[0.5] == 0
        assert changed[0.3] == 2
        assert changed[0.05] == 4

    def test_a_row_failing_on_dsr_never_passes_at_any_ceiling(self):
        rows = [row(dsr=0.1, p_oos_loss=0.0)]
        cal = gate_backtest.calibrate(rows)
        assert all(c.full.n_pass == 0 for c in cal.ceilings)
        # The loss-probability criterion alone still passes it, which is the
        # point of reporting the two apart.
        assert all(c.p_oos_loss_only.n_pass == 1 for c in cal.ceilings)

    def test_dsr_only_split_is_ceiling_independent(self):
        rows = [row(dsr=0.99, forward_sharpe=2.0), row(dsr=0.5, forward_sharpe=0.0)]
        cal = gate_backtest.calibrate(rows)
        assert (cal.dsr_only.n_pass, cal.dsr_only.n_fail) == (1, 1)
        assert cal.dsr_only.separation == pytest.approx(2.0)

    def test_rank_correlations_cover_all_three_panels(self):
        rows = [row(p_oos_loss=p, dsr=1.0 - p, pbo=p, forward_sharpe=1.0 - p)
                for p in (0.1, 0.2, 0.3, 0.4)]
        cal = gate_backtest.calibrate(rows)
        assert cal.spearman_p_oos_loss == pytest.approx(-1.0)
        assert cal.spearman_dsr == pytest.approx(1.0)
        assert cal.spearman_pbo == pytest.approx(-1.0)
        assert cal.n_rows == 4

    def test_an_empty_run_is_all_nan_not_zero(self):
        cal = gate_backtest.calibrate([])
        assert cal.n_rows == 0
        assert math.isnan(cal.spearman_p_oos_loss)
        assert math.isnan(cal.spearman_dsr)
        assert math.isnan(cal.spearman_pbo)
        assert all(math.isnan(c.full.mean_pass) for c in cal.ceilings)
        assert cal.best_ceiling is None

    def test_strongest_predictor_is_the_largest_magnitude_ranking(self):
        rows = [row(p_oos_loss=p, dsr=0.97, pbo=0.4, forward_sharpe=1.0 - p)
                for p in (0.1, 0.2, 0.3, 0.4)]
        cal = gate_backtest.calibrate(rows)
        name, rho = cal.strongest_predictor
        assert name == "P(OOS loss)"
        assert rho == pytest.approx(-1.0)

    def test_nan_p_oos_loss_passes_the_ceiling_like_the_live_gate(self):
        rows = [row(p_oos_loss=float("nan"))]
        cal = gate_backtest.calibrate(rows)
        assert all(c.full.n_pass == 1 for c in cal.ceilings)


class TestTradedOnlyCalibration:
    def test_a_row_that_never_held_a_position_is_not_a_measured_zero(self):
        assert row(forward_trades=0, forward_exposure_bars=0,
                   forward_entry_signals=0).traded is False
        assert row(forward_trades=0, forward_exposure_bars=1).traded is True

    def test_an_entry_with_no_bar_after_it_is_disclosed_not_measured(self):
        # Entered on the window's final bar: nothing closed and no bar was
        # held, so the forward Sharpe is one slippage tick rather than a year.
        # The count stays on the row as disclosure.
        entry_only = row(forward_trades=0, forward_exposure_bars=0,
                         forward_entry_signals=1)
        assert entry_only.traded is False
        assert entry_only.forward_entry_signals == 1

    def test_a_round_trip_with_no_bar_carried_still_counts_as_traded(self):
        # Entered and stopped out before it carried a bar into its own next
        # decision: a trade happened, so the row is a measurement.
        assert row(forward_trades=1, forward_exposure_bars=0,
                   forward_entry_signals=1).traded is True

    def test_no_trade_rows_are_calibrated_apart(self):
        traded = [row(p_oos_loss=p, forward_sharpe=1.0 - p)
                  for p in (0.1, 0.2, 0.3, 0.4)]
        # Two rows whose variant was never in the market: a 0.00 Sharpe that is
        # absent evidence, at the two ends of the ranking so it moves it.
        flat = [row(p_oos_loss=0.05, forward_sharpe=0.0, forward_trades=0,
                    forward_exposure_bars=0, forward_entry_signals=0),
                row(p_oos_loss=0.45, forward_sharpe=0.0, forward_trades=0,
                    forward_exposure_bars=0, forward_entry_signals=0)]
        cal = gate_backtest.calibrate(traded + flat)
        assert cal.n_rows == 6
        assert cal.traded_only is not None
        assert cal.traded_only.n_rows == 4
        assert cal.traded_only.spearman_p_oos_loss == pytest.approx(-1.0)
        assert cal.spearman_p_oos_loss != pytest.approx(
            cal.traded_only.spearman_p_oos_loss)
        assert cal.traded_only.dsr_only.n_pass == 4

    def test_a_run_where_every_variant_traded_has_no_second_calibration(self):
        cal = gate_backtest.calibrate([row(p_oos_loss=p) for p in (0.1, 0.2)])
        assert cal.traded_only is None
        assert cal.traded is cal

    def test_n_no_trade_counts_the_rows_the_traded_read_drops(self):
        cal = gate_backtest.calibrate(
            [row(), row(), row(forward_trades=0, forward_exposure_bars=0,
                               forward_entry_signals=0)])
        assert cal.n_rows == 3
        assert cal.traded.n_rows == 2
        assert cal.n_no_trade == 1
        assert gate_backtest.calibrate([row(), row()]).n_no_trade == 0

    def test_n_traded_is_recorded_whether_or_not_the_split_was_taken(self):
        rows = [row(), row(forward_trades=0, forward_exposure_bars=0,
                           forward_entry_signals=0)]
        split = gate_backtest.calibrate(rows)
        unsplit = gate_backtest.calibrate(rows, split_no_trade=False)
        assert (split.n_rows, split.n_traded, split.n_no_trade) == (2, 1, 1)
        assert (unsplit.n_rows, unsplit.n_traded, unsplit.n_no_trade) == (2, 1, 1)

    def test_a_headline_without_the_split_does_not_claim_every_row_traded(self):
        # Entry-only rows: no position was held, so none of them is a
        # measurement — and split_no_trade=False means the traded-only read was
        # never computed, which is not the same as nothing being excluded.
        rows = [row(p_oos_loss=p, forward_sharpe=s, forward_trades=0,
                    forward_exposure_bars=0, forward_entry_signals=1)
                for p, s in ((0.1, -0.98), (0.2, 0.42), (0.3, 0.07))]
        cal = gate_backtest.calibrate(rows, split_no_trade=False)
        assert (cal.n_rows, cal.n_traded) == (3, 0)
        assert cal.traded_only is None
        assert ("every row's selected variant was in the market"
                not in gate_backtest.headline(cal))

    def test_the_traded_only_calibration_does_not_nest_again(self):
        cal = gate_backtest.calibrate(
            [row(), row(forward_trades=0, forward_exposure_bars=0,
                       forward_entry_signals=0)])
        assert cal.traded.n_rows == 1
        assert cal.traded.traded_only is None


def fake_result(rows=None, skipped=None) -> GateBacktest:
    """A GateBacktest with no backtests behind it: the reporting layer only
    reads the numbers."""
    rows = [row(2020, "SPY", p_oos_loss=0.1, forward_sharpe=1.5),
            row(2021, "QQQ", p_oos_loss=0.6, forward_sharpe=float("nan"),
                dsr=float("nan"))] if rows is None else rows
    skipped = [SkippedCell(date(2019, 12, 31), "BTC/USD", "142 forward bars, "
                           "needs 300")] if skipped is None else skipped
    return gate_backtest.GateBacktest(
        rows=rows, skipped=skipped,
        cutoffs=[date(2019, 12, 31), date(2020, 12, 31), date(2021, 12, 31)],
        forward_days=365, n_blocks=16, n_folds=4,
        calibration=gate_backtest.calibrate(rows))


class TestReport:
    def test_report_records_the_rows_the_skips_and_the_run_context(self):
        at = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
        report = gate_backtest.build_report(fake_result(), generated_at=at)
        assert report["generated_at"] == at.isoformat()
        assert report["market_data_feed"] == MARKET_DATA_FEED
        assert report["forward_days"] == 365
        assert report["cutoffs"] == ["2019-12-31", "2020-12-31", "2021-12-31"]
        assert [r["symbol"] for r in report["rows"]] == ["SPY", "QQQ"]
        assert report["rows"][0]["cutoff"] == "2020-12-31"
        assert report["rows"][0]["p_oos_loss"] == 0.1
        assert report["skipped"][0]["symbol"] == "BTC/USD"
        assert "forward bars" in report["skipped"][0]["reason"]
        assert report["gate"]["dsr_min"] == VALIDATION.dsr_min
        assert report["gate"]["p_oos_loss_max"] == VALIDATION.p_oos_loss_max

    def test_report_carries_a_schema_version(self):
        report = gate_backtest.build_report(fake_result())
        assert gate_backtest.SCHEMA_VERSION == 1
        assert report["schema_version"] == gate_backtest.SCHEMA_VERSION
        # The version-1 calibration shape.
        assert set(report["calibration"]) == {"all_rows", "traded_only"}

    def test_report_says_per_row_whether_the_variant_was_in_the_market(self):
        rows = [row(2020, "SPY", forward_trades=7, forward_exposure_bars=44,
                    forward_entry_signals=7),
                row(2021, "QQQ", forward_trades=0, forward_exposure_bars=210,
                    forward_entry_signals=1),
                row(2022, "GLD", forward_trades=0, forward_exposure_bars=0,
                    forward_entry_signals=0)]
        report = gate_backtest.build_report(fake_result(rows=rows))
        assert [r["traded"] for r in report["rows"]] == [True, True, False]
        assert [r["forward_entry_signals"] for r in report["rows"]] == [7, 1, 0]

    def test_the_traded_only_read_repeats_all_rows_when_nothing_was_excluded(
            self):
        rows = [row(2020, "SPY", p_oos_loss=0.1, forward_sharpe=1.5),
                row(2021, "QQQ", p_oos_loss=0.2, forward_sharpe=0.5)]
        cal = gate_backtest.build_report(fake_result(rows=rows))["calibration"]
        assert cal["traded_only"] == cal["all_rows"]

    def test_report_discloses_the_warm_up_flat_variants_per_row(self):
        report = gate_backtest.build_report(fake_result())
        entry = report["rows"][0]
        assert entry["first_train_bars"] == 100
        assert entry["warmup_flat_variants"] == ["ema50/200"]

    def test_calibration_is_recorded_panel_by_panel(self):
        cal = gate_backtest.build_report(fake_result())["calibration"]["all_rows"]
        assert set(cal["spearman"]) == {"p_oos_loss", "dsr", "pbo"}
        assert cal["n_rows"] == 2
        assert [c["ceiling"] for c in cal["ceilings"]] == list(gate_backtest.CEILINGS)
        first = cal["ceilings"][0]
        assert set(first) >= {"ceiling", "full", "p_oos_loss_only", "changed"}
        assert set(first["full"]) == {"n_pass", "n_fail", "mean_pass",
                                      "mean_fail", "separation"}
        assert "dsr_only" in cal

    def test_report_records_the_trade_and_exposure_counts_per_row(self):
        rows = [row(2020, "SPY", forward_trades=7, forward_exposure_bars=44),
                row(2021, "QQQ", forward_sharpe=1.4, forward_trades=0,
                    forward_exposure_bars=210),
                row(2022, "GLD", forward_sharpe=0.0, forward_trades=0,
                    forward_exposure_bars=0, forward_entry_signals=0)]
        report = gate_backtest.build_report(fake_result(rows=rows))
        assert [r["forward_trades"] for r in report["rows"]] == [7, 0, 0]
        assert [r["forward_exposure_bars"] for r in report["rows"]] == \
            [44, 210, 0]

    def test_calibration_is_recorded_with_and_without_the_no_trade_rows(self):
        rows = [row(2020, "SPY", p_oos_loss=0.1, forward_sharpe=1.5),
                row(2021, "QQQ", p_oos_loss=0.2, forward_sharpe=0.8),
                row(2022, "GLD", p_oos_loss=0.4, forward_sharpe=0.0,
                    forward_trades=0, forward_exposure_bars=0,
                    forward_entry_signals=0)]
        cal = gate_backtest.build_report(fake_result(rows=rows))["calibration"]
        assert cal["all_rows"]["n_rows"] == 3
        assert cal["traded_only"]["n_rows"] == 2
        assert set(cal["traded_only"]["spearman"]) == set(
            cal["all_rows"]["spearman"])
        assert cal["all_rows"]["base_ceiling"] == VALIDATION.p_oos_loss_max

    def test_report_round_trips_with_nan_written_as_null(self, tmp_path):
        at = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
        report = gate_backtest.build_report(fake_result(), generated_at=at)
        path = gate_backtest.write_report(report, at, directory=str(tmp_path))
        assert os.path.basename(path) == "gate-backtest-2026-09-16.json"
        with open(path) as fh:
            loaded = json.load(fh)
        assert loaded["rows"][1]["forward_sharpe"] is None
        assert loaded["rows"][1]["dsr"] is None
        assert loaded["rows"][0]["forward_trades"] == 12
        assert set(loaded["calibration"]) == {"all_rows", "traded_only"}
        assert loaded == report

    def test_a_record_is_never_overwritten(self, tmp_path):
        at = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
        report = gate_backtest.build_report(fake_result(), generated_at=at)
        gate_backtest.write_report(report, at, directory=str(tmp_path))
        with pytest.raises(FileExistsError):
            gate_backtest.write_report(report, at, directory=str(tmp_path))

    def test_record_paths_are_date_stamped_siblings(self, tmp_path):
        at = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
        json_path, md_path = gate_backtest.record_paths(at, str(tmp_path))
        assert json_path.name == "gate-backtest-2026-09-16.json"
        assert md_path.name == "gate-backtest-2026-09-16.md"


class TestMarkdown:
    def test_markdown_leads_with_the_headline_and_the_tables(self):
        at = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
        rows = [row(2020, "SPY", p_oos_loss=0.1, forward_sharpe=1.5),
                row(2021, "QQQ", p_oos_loss=0.2, forward_sharpe=1.0),
                row(2022, "GLD", p_oos_loss=0.4, forward_sharpe=0.2)]
        text = gate_backtest.build_markdown(fake_result(rows=rows), at)
        assert text.startswith("# Promotion-gate calibration — 2026-09-16")
        assert "P(OOS loss)" in text and "PBO" in text and "DSR" in text
        # Row table and ceiling table both present.
        assert "| cutoff | symbol |" in text
        assert "| ceiling |" in text
        assert "SPY" in text and "0.05" in text

    def test_markdown_carries_the_small_n_caveat(self):
        at = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
        text = gate_backtest.build_markdown(fake_result(), at)
        assert "calibration read, not a statistical proof" in text
        assert "overlapping" in text
        assert "experiment record, not a promotion artifact" in text

    def test_markdown_shows_the_warm_up_flat_column_and_says_what_it_means(self):
        at = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
        rows = [row(2020, "SPY", warmup_flat=("ema50/200", "tsmom")),
                row(2021, "QQQ", warmup_flat=())]
        text = gate_backtest.build_markdown(fake_result(rows=rows), at)
        assert "warm-up flat" in text
        header, spy, qqq = (line for line in text.splitlines()
                            if line.startswith("| cutoff |")
                            or " SPY " in line or " QQQ " in line)
        assert header.count("|") == spy.count("|") == qqq.count("|")
        assert spy.endswith(" 2 |") and qqq.endswith(" 0 |")
        assert ("rows with warm-up-flat variants replay research's own "
                "behaviour" in text.lower())
        assert "cannot be selected in the earliest training folds" in text

    def test_markdown_shows_trades_over_exposure_and_both_calibrations(self):
        at = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
        rows = [row(2020, "SPY", p_oos_loss=0.1, forward_sharpe=1.5,
                    forward_trades=9, forward_exposure_bars=88),
                row(2021, "QQQ", p_oos_loss=0.2, forward_sharpe=0.5,
                    forward_trades=0, forward_exposure_bars=210),
                row(2022, "GLD", p_oos_loss=0.4, forward_sharpe=0.0,
                    forward_trades=0, forward_exposure_bars=0,
                    forward_entry_signals=0)]
        text = gate_backtest.build_markdown(fake_result(rows=rows), at)
        assert "fwd trades / bars held" in text
        header, spy, qqq, gld = (line for line in text.splitlines()
                                 if line.startswith("| cutoff |")
                                 or " SPY " in line or " QQQ " in line
                                 or " GLD " in line)
        assert header.count("|") == spy.count("|") == qqq.count("|") == \
            gld.count("|")
        assert "| 9 / 88 |" in spy
        # Held all year without closing a round trip: still a measured row.
        assert "| 0 / 210 |" in qqq
        assert "| 0 / 0 |" in gld
        # The headline reads the ranking both ways round: only GLD is dropped.
        assert "rows that traded" in text
        assert "2 rows" in text
        # A no-trade row's 0.00 Sharpe is named, not left implicit.
        assert "never held a position" in text
        assert "bars held" in text.split("## Caveat")[1]

    def test_markdown_says_so_when_every_variant_traded(self):
        at = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
        rows = [row(2020, "SPY", p_oos_loss=0.1, forward_sharpe=1.5),
                row(2021, "QQQ", p_oos_loss=0.2, forward_sharpe=0.5)]
        text = gate_backtest.build_markdown(fake_result(rows=rows), at)
        # The branch's own sentence, not merely "every row": the headline
        # carries that phrase whether or not the branch ran.
        assert ("Every row's selected variant held a position in its forward "
                "window, so the traded-only calibration is this same read."
                in text)
        assert text.count("| ceiling | pass | fail |") == 1
        assert "rows that traded" not in text

    def test_the_trade_column_discloses_an_entry_with_nothing_behind_it(self):
        at = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
        rows = [row(2020, "SPY", forward_trades=0, forward_exposure_bars=0,
                    forward_entry_signals=1),
                row(2021, "QQQ", forward_trades=0, forward_exposure_bars=0,
                    forward_entry_signals=2)]
        text = gate_backtest.build_markdown(fake_result(rows=rows), at)
        assert "| 0 / 0 / 1 entry |" in text
        assert "| 0 / 0 / 2 entries |" in text

    def test_markdown_does_not_tabulate_a_calibration_over_no_rows(self):
        at = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
        rows = [row(2020, "SPY", p_oos_loss=0.1, forward_sharpe=0.0,
                    forward_trades=0, forward_exposure_bars=0,
                    forward_entry_signals=0),
                row(2021, "QQQ", p_oos_loss=0.2, forward_sharpe=0.0,
                    forward_trades=0, forward_exposure_bars=0,
                    forward_entry_signals=0)]
        text = gate_backtest.build_markdown(fake_result(rows=rows), at)
        assert text.count("| ceiling | pass | fail |") == 1
        assert "Over the 0 rows" not in text

    def test_markdown_names_the_feed_the_run_used_not_the_environment(self):
        at = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
        feed = "sip" if MARKET_DATA_FEED != "sip" else "iex"
        result = replace(fake_result(), market_data_feed=feed)
        text = gate_backtest.build_markdown(result, at)
        assert f"the {feed} feed" in text
        assert f"the {MARKET_DATA_FEED} feed" not in text
        assert gate_backtest.build_report(result, at)["market_data_feed"] == feed

    def test_markdown_names_the_skipped_cells(self):
        at = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
        text = gate_backtest.build_markdown(fake_result(), at)
        assert "BTC/USD" in text
        assert "skipped" in text.lower()

    def test_markdown_survives_an_empty_run(self):
        at = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
        text = gate_backtest.build_markdown(fake_result(rows=[]), at)
        assert "no rows" in text.lower()
        # Nothing was scored, so nothing held a position either.
        assert "held a position in its forward window" not in text
        assert text.count("| ceiling | pass | fail |") == 1

    def test_markdown_reads_the_traded_only_calibration_out_in_full(self):
        at = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
        rows = [row(2020, "SPY", p_oos_loss=0.1, forward_sharpe=1.5),
                row(2021, "QQQ", p_oos_loss=0.2, forward_sharpe=0.8),
                row(2022, "GLD", p_oos_loss=0.4, forward_sharpe=0.0,
                    forward_trades=0, forward_exposure_bars=0,
                    forward_entry_signals=0),
                row(2023, "IWM", p_oos_loss=0.45, forward_sharpe=0.0,
                    forward_trades=0, forward_exposure_bars=0,
                    forward_entry_signals=0)]
        text = gate_backtest.build_markdown(fake_result(rows=rows), at)
        # One ceiling table per calibration read.
        assert text.count("| ceiling | pass | fail |") == 2
        assert ("The same read with the 2 rows whose variant never held a "
                "position dropped." in text)
        # A retained row can have zero closed trades, so "no-trade rows" names
        # the wrong set.
        assert "no-trade rows" not in text
        assert "Over the 2 rows whose selected variant was in the market" in text

    def test_markdown_names_the_conditioning_the_traded_only_read_buys(self):
        at = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
        text = gate_backtest.build_markdown(fake_result(), at)
        assert ("conditions on a forward-window outcome" in text
                and "not an unbiased estimate" in text)

    def test_the_headline_sentence_is_printed_once(self):
        at = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
        rows = [row(2020, "SPY", p_oos_loss=0.1, forward_sharpe=1.5),
                row(2021, "QQQ", p_oos_loss=0.2, forward_sharpe=0.8),
                row(2022, "GLD", p_oos_loss=0.4, forward_sharpe=0.2)]
        result = fake_result(rows=rows)
        text = gate_backtest.build_markdown(result, at)
        assert gate_backtest.headline(result.calibration) in text
        assert text.count(
            "Over 3 rows the gate inputs rank forward Sharpe at ") == 1


class TestConsoleTable:
    def test_the_console_spells_the_trade_column_the_markdown_way(self, capsys):
        at = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
        rows = [row(2020, "SPY", forward_trades=9, forward_exposure_bars=88),
                row(2021, "QQQ", forward_trades=0, forward_exposure_bars=0,
                    forward_entry_signals=1)]
        result = fake_result(rows=rows)
        gate_backtest.print_rows(result)
        out = capsys.readouterr().out
        text = gate_backtest.build_markdown(result, at)
        for cell in ("9 / 88", "0 / 0 / 1 entry"):
            assert cell in out
            assert f"| {cell} |" in text

    def test_the_console_names_the_skipped_cells_too(self, capsys):
        gate_backtest.print_rows(fake_result())
        out = capsys.readouterr().out
        assert "BTC/USD" in out and "skipped" in out


class TestCli:
    def test_cutoffs_parse_as_iso_dates(self):
        args = gate_backtest.build_parser().parse_args(
            ["--cutoffs", "2020-12-31,2021-12-31"])
        assert args.cutoffs == [date(2020, 12, 31), date(2021, 12, 31)]

    def test_default_cutoffs_are_the_year_ends_2019_to_2024(self):
        assert gate_backtest.default_cutoffs() == [
            date(y, 12, 31) for y in range(2019, 2025)]
        assert gate_backtest.build_parser().parse_args([]).cutoffs == \
            gate_backtest.default_cutoffs()

    def test_an_unparsable_date_is_rejected(self, capsys):
        with pytest.raises(SystemExit) as exc:
            gate_backtest.build_parser().parse_args(["--cutoffs", "last-tuesday"])
        assert exc.value.code == 2
        assert "last-tuesday" in capsys.readouterr().err

    def test_parse_cutoffs_rejects_an_empty_list(self):
        with pytest.raises(ValueError):
            gate_backtest.parse_cutoffs(" , ")

    def test_help_prints_usage_without_touching_the_network(self, capsys):
        with pytest.raises(SystemExit) as exc:
            gate_backtest.main(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "usage:" in out
        assert "--cutoffs" in out


class TestShortlists:
    def test_shortlists_are_research_experiments_grouped_by_symbol(self):
        grouped = gate_backtest.shortlists()
        assert set(grouped) == {row[0] for row in EXPERIMENTS}
        for symbol, (asset_class, slippage, variants) in grouped.items():
            expected = [r[3] for r in EXPERIMENTS if r[0] == symbol]
            assert [label for label, _f in variants] == expected
            assert asset_class == next(r[1] for r in EXPERIMENTS if r[0] == symbol)
            assert slippage == next(r[2] for r in EXPERIMENTS if r[0] == symbol)
