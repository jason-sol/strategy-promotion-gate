import dataclasses
import json
import math
from functools import partial

import numpy as np
import pandas as pd
import pytest

from bot import validation
from bot.diagnose import decisive, probe_lookahead
from bot.metrics import sharpe_ratio
from bot.strategies import build_strategy
from bot.strategies.base import HOLD, Action, Signal, Strategy
from bot.validation import (
    Candidate,
    CscvReport,
    _run_returns,
    build_return_matrix,
    cscv_pbo,
    cscv_report,
    walk_forward,
)
from config import AssetClass, InstrumentConfig, StrategyName
from helpers import daily_bars

# Contiguous calendar days: no weekend gap, so resampling to 1D is the identity.
calendar_bars = partial(daily_bars, freq="1D")


def mr_cfg(symbol: str, entry_z: float) -> InstrumentConfig:
    return InstrumentConfig(
        symbol=symbol, asset_class=AssetClass.US_EQUITY,
        strategy=StrategyName.MEAN_REVERSION, timeframe_minutes=1440,
        params={"sma_period": 10, "entry_z": entry_z, "allow_short": False},
        slippage_pct=0.0,
    )


def candidate(label: str, symbol: str, entry_z: float) -> Candidate:
    cfg = mr_cfg(symbol, entry_z)
    return Candidate(label, cfg, build_strategy(cfg))


def choppy_uptrend(n: int) -> list[float]:
    t = np.arange(n)
    return list(100.0 + 0.5 * t + 4.0 * np.sin(t / 2.0))


def sawtooth(n: int) -> list[float]:
    """Trendless oscillation, so a level read off one row bisects the series
    instead of sitting above or below all of it."""
    t = np.arange(n)
    return list(100.0 + 20.0 * np.sin(t / 7.0) + 5.0 * np.sin(t / 2.0))


FAMILY_VOL = 0.01
FAMILY_CORR = 0.97


def dated_matrix(columns: dict[str, np.ndarray]) -> pd.DataFrame:
    length = len(next(iter(columns.values())))
    index = pd.date_range("2020-01-01", periods=length, freq="B", tz="UTC")
    return pd.DataFrame(columns, index=index)


def near_duplicate_family(rng: np.random.Generator, annual_sharpe: float,
                          t: int = 600, n_cols: int = 10) -> pd.DataFrame:
    """One shared signal plus per-column jitter, pairwise correlation
    FAMILY_CORR. Every member has near-identical true skill, so only the jitter
    breaks the in-sample tie — the shape an overlapping parameter window
    produces."""
    jitter = FAMILY_VOL * math.sqrt((1.0 - FAMILY_CORR) / FAMILY_CORR)
    base = rng.normal(annual_sharpe * FAMILY_VOL / math.sqrt(252.0), FAMILY_VOL, t)
    columns = base[:, None] + rng.normal(0.0, jitter, (t, n_cols))
    return dated_matrix({f"v{i}": columns[:, i] for i in range(n_cols)})


def drifting_field(drift: float, n_cols: int = 6, t: int = 240,
                   seed: int = 5) -> pd.DataFrame:
    """Every column drifts alike at twenty times its own noise, so the sign of
    any column's Sharpe is the sign of the drift on every split."""
    rng = np.random.default_rng(seed)
    values = drift + 0.001 * rng.standard_normal((t, n_cols))
    return dated_matrix({f"c{i}": values[:, i] for i in range(n_cols)})


def one_bump_per_block(n_blocks: int = 8, block_size: int = 10,
                       seed: int = 0) -> pd.DataFrame:
    """Column j earns its whole edge inside block j and bleeds everywhere else,
    so the in-sample winner is always a column whose bump the in-sample legs
    just consumed and whose out-of-sample legs are all losses."""
    rng = np.random.default_rng(seed)
    t = n_blocks * block_size
    columns = {}
    for j in range(n_blocks):
        values = -0.01 + 0.001 * rng.standard_normal(t)
        values[j * block_size:(j + 1) * block_size] += 0.06
        columns[f"blk{j}"] = values
    return dated_matrix(columns)


def scale_paired_blocks(shapes: dict[str, list[float]], n_blocks: int = 4,
                        scale: float = 2.0) -> pd.DataFrame:
    """Odd blocks repeat the even blocks scaled up. Sharpe ignores scale, so
    every split's out-of-sample Sharpe reproduces its in-sample Sharpe; a scale
    of 1.0 makes every split score identically."""
    columns = {}
    for label, shape in shapes.items():
        block = np.asarray(shape, dtype=float)
        columns[label] = np.concatenate(
            [block if b % 2 == 0 else scale * block for b in range(n_blocks)]
        )
    return dated_matrix(columns)


SCALE_PAIRED_SHAPES = {
    "a": [0.01, -0.005, 0.02, 0.0, 0.015, -0.01, 0.005, 0.01],
    "b": [-0.01, 0.02, 0.005, 0.01, -0.005, 0.03, 0.0, -0.02],
    "c": [0.005, 0.005, -0.02, 0.01, 0.02, -0.015, 0.01, 0.0],
}


class LastRowAnchor(Strategy):
    """Every decision is measured against the close of the LAST row of whatever
    frame prepare() was handed, so its returns change with the frame's end."""

    @property
    def min_bars(self) -> int:
        return 2

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        df["atr"] = 1.0
        df["anchor"] = float(bars["close"].iloc[-1])
        return df

    def signal(self, df, i, position):
        row = df.iloc[i]
        if position is None:
            return Signal(Action.LONG, "under anchor") if row["close"] < row["anchor"] else HOLD
        return Signal(Action.EXIT, "over anchor") if row["close"] >= row["anchor"] else HOLD


class NextBarPeeker(Strategy):
    """Deliberately non-causal: prepare() shifts close by -1, so the decision at
    bar i reads bar i+1's close."""

    @property
    def min_bars(self) -> int:
        return 3

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        df["atr"] = 1.0
        df["next_close"] = bars["close"].shift(-1)
        return df

    def signal(self, df, i, position):
        row = df.iloc[i]
        if pd.isna(row["next_close"]):
            return HOLD
        rises = row["next_close"] > row["close"]
        if position is None:
            return Signal(Action.LONG, "rises next bar") if rises else HOLD
        return HOLD if rises else Signal(Action.EXIT, "falls next bar")


class AlwaysLong(Strategy):
    """Enters on the first tradable bar and never exits, so equity moves on
    every bar of the window."""

    @property
    def min_bars(self) -> int:
        return 2

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        df["atr"] = 1.0
        return df

    def signal(self, df, i, position):
        return HOLD if position is not None else Signal(Action.LONG, "always long")


def custom_candidate(label: str, strategy: Strategy) -> Candidate:
    return Candidate(label, mr_cfg("AAA", 1.0), strategy)


def fold_bounds(n: int, n_folds: int) -> list[tuple[int, int, int]]:
    """(train_start, test_start, test_end) positions walk_forward derives."""
    block = n // (n_folds + 1)
    return [(i * block, (i + 1) * block,
             n if i == n_folds - 1 else (i + 2) * block)
            for i in range(n_folds)]


def record_run_returns(monkeypatch) -> list[tuple[pd.DataFrame, pd.Timestamp, pd.Series]]:
    """Capture the (frame, sim_start, returns) of every backtest walk_forward runs."""
    calls: list[tuple[pd.DataFrame, pd.Timestamp, pd.Series]] = []

    def spy(candidate, bars, sim_start):
        returns = _run_returns(candidate, bars, sim_start)
        calls.append((bars, sim_start, returns))
        return returns

    monkeypatch.setattr(validation, "_run_returns", spy)
    return calls


_EXPECTED_OOS_SHARPE = 7.204511757971705  # honest select-IS-best-then-trade, locked


class TestCscvPbo:
    def test_pure_noise_gives_pbo_near_half(self):
        # A single noise matrix is a high-variance PBO estimate; average a few
        # independent draws to test the underlying property (PBO of noise ~ 0.5).
        rng = np.random.default_rng(42)
        pbos = [cscv_pbo(pd.DataFrame(rng.standard_normal((250, 20))), n_blocks=10)
                for _ in range(8)]
        assert 0.3 <= float(np.mean(pbos)) <= 0.7

    def test_dominant_column_gives_low_pbo(self):
        rng = np.random.default_rng(7)
        data = rng.standard_normal((250, 20))
        data[:, 0] += 0.5  # same vol, clearly higher mean
        matrix = pd.DataFrame(data)
        pbo = cscv_pbo(matrix, n_blocks=10)
        assert pbo < 0.2

    def test_single_column_is_not_computable(self):
        matrix = pd.DataFrame(np.random.default_rng(0).standard_normal((250, 1)))
        assert math.isnan(cscv_pbo(matrix))

    def test_too_few_rows_is_not_computable(self):
        matrix = pd.DataFrame(np.random.default_rng(0).standard_normal((5, 4)))
        assert math.isnan(cscv_pbo(matrix, n_blocks=16))

    def test_rows_below_two_per_block_is_not_computable(self):
        # 2*n_blocks - 1 rows: a symmetric split would leave a single-row group.
        matrix = pd.DataFrame(np.random.default_rng(0).standard_normal((7, 4)))
        assert math.isnan(cscv_pbo(matrix, n_blocks=4))

    def test_odd_n_blocks_rejected(self):
        matrix = pd.DataFrame(np.random.default_rng(0).standard_normal((250, 4)))
        with pytest.raises(ValueError):
            cscv_pbo(matrix, n_blocks=15)


class TestCscvReport:
    @pytest.mark.parametrize("n_blocks", [4, 8, 10])
    def test_pbo_panel_is_what_the_scalar_wrapper_returns(self, n_blocks):
        matrix = pd.DataFrame(np.random.default_rng(3).standard_normal((240, 6)))
        assert cscv_report(matrix, n_blocks).pbo == cscv_pbo(matrix, n_blocks)

    def test_p_oos_loss_is_one_when_the_is_best_always_loses_out_of_sample(self):
        assert cscv_report(one_bump_per_block(), n_blocks=8).p_oos_loss == 1.0

    def test_p_oos_loss_is_zero_when_the_field_wins_on_every_leg(self):
        assert cscv_report(drifting_field(0.02), n_blocks=8).p_oos_loss == 0.0

    def test_p_oos_loss_is_one_when_the_field_loses_on_every_leg(self):
        assert cscv_report(drifting_field(-0.02), n_blocks=8).p_oos_loss == 1.0

    def test_median_logit_is_positive_when_the_is_best_ranks_top_half_oos(self):
        values = np.random.default_rng(7).standard_normal((250, 20))
        values[:, 0] += 0.5  # same vol, clearly higher mean
        result = cscv_report(pd.DataFrame(values), n_blocks=10)

        assert result.pbo == 0.0
        assert result.median_logit > 0.0

    def test_median_logit_is_negative_when_the_is_best_ranks_bottom_half_oos(self):
        result = cscv_report(one_bump_per_block(), n_blocks=8)

        assert result.pbo == 1.0
        assert result.median_logit < 0.0

    def test_degradation_slope_is_one_when_oos_reproduces_is(self):
        matrix = scale_paired_blocks(SCALE_PAIRED_SHAPES)
        assert cscv_report(matrix, n_blocks=4).degradation_slope == pytest.approx(1.0)

    def test_degradation_slope_is_not_computable_when_every_split_scores_alike(self):
        matrix = scale_paired_blocks(SCALE_PAIRED_SHAPES, scale=1.0)
        assert math.isnan(cscv_report(matrix, n_blocks=4).degradation_slope)

    def test_logits_carry_one_value_per_split(self):
        result = cscv_report(drifting_field(0.02), n_blocks=8)
        assert len(result.logits) == math.comb(8, 4)

    def test_per_candidate_panels_key_every_column(self):
        matrix = drifting_field(0.02)
        result = cscv_report(matrix, n_blocks=8)

        labels = [str(col) for col in matrix.columns]
        assert list(result.oos_rank_mean) == labels
        assert list(result.is_best_share) == labels

    def test_is_best_share_is_a_distribution_over_the_columns(self):
        result = cscv_report(one_bump_per_block(), n_blocks=8)

        assert sum(result.is_best_share.values()) == pytest.approx(1.0)
        assert all(0.0 <= share <= 1.0 for share in result.is_best_share.values())

    def test_oos_rank_mean_stays_in_the_unit_interval(self):
        matrix = pd.DataFrame(np.random.default_rng(3).standard_normal((240, 6)))
        result = cscv_report(matrix, n_blocks=8)
        assert all(0.0 <= rank <= 1.0 for rank in result.oos_rank_mean.values())

    def test_oos_rank_mean_puts_a_dominant_column_above_the_whole_field(self):
        values = np.random.default_rng(7).standard_normal((250, 20))
        values[:, 0] += 0.5
        ranks = cscv_report(pd.DataFrame(values), n_blocks=10).oos_rank_mean

        assert ranks["0"] == pytest.approx(19 / 20)
        assert max(ranks, key=ranks.get) == "0"

    @pytest.mark.parametrize("rows,cols", [(250, 1), (5, 4), (7, 4)])
    def test_degenerate_input_is_all_nan_with_empty_panels(self, rows, cols):
        matrix = pd.DataFrame(np.random.default_rng(0).standard_normal((rows, cols)))
        result = cscv_report(matrix, n_blocks=4 if rows == 7 else 16)

        assert math.isnan(result.pbo)
        assert math.isnan(result.p_oos_loss)
        assert math.isnan(result.median_logit)
        assert math.isnan(result.degradation_slope)
        assert result.logits == ()
        assert result.oos_rank_mean == {}
        assert result.is_best_share == {}

    def test_degenerate_pbo_matches_the_scalar_wrapper(self):
        matrix = pd.DataFrame(np.random.default_rng(0).standard_normal((250, 1)))
        assert math.isnan(cscv_report(matrix).pbo)
        assert math.isnan(cscv_pbo(matrix))

    def test_odd_n_blocks_cannot_split_symmetrically(self):
        with pytest.raises(ValueError, match="must be even"):
            cscv_report(drifting_field(0.02), n_blocks=15)

    def test_report_round_trips_through_json(self):
        result = cscv_report(drifting_field(0.02), n_blocks=8)
        restored = json.loads(json.dumps(dataclasses.asdict(result)))

        assert isinstance(result, CscvReport)
        assert restored["pbo"] == result.pbo
        assert restored["is_best_share"] == result.is_best_share
        assert restored["oos_rank_mean"] == result.oos_rank_mean
        assert restored["logits"] == list(result.logits)


class TestPboVersusOosLossOnNearDuplicateFamilies:
    """PBO reads the same for a near-duplicate family with a shared real edge as
    for one with no edge: the members' true skill is near-identical either way,
    so only the jitter breaks the in-sample tie and the rank statistic is a coin
    flip. P(OOS loss) reads the shared signal instead and separates the two."""

    def test_pbo_cannot_separate_a_shared_edge_from_noise_while_p_oos_loss_can(self):
        edged, flat = [], []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            edged.append(cscv_report(near_duplicate_family(rng, 3.0), n_blocks=8))
            flat.append(cscv_report(near_duplicate_family(rng, 0.0), n_blocks=8))

        pbo_edged = float(np.mean([r.pbo for r in edged]))
        pbo_flat = float(np.mean([r.pbo for r in flat]))
        loss_edged = float(np.mean([r.p_oos_loss for r in edged]))
        loss_flat = float(np.mean([r.p_oos_loss for r in flat]))

        assert 0.35 <= pbo_edged <= 0.65
        assert 0.35 <= pbo_flat <= 0.65
        assert abs(pbo_edged - pbo_flat) < 0.2
        assert loss_edged < 0.1
        assert loss_flat > 0.25


class TestBuildReturnMatrix:
    def test_two_candidates_align_into_clean_matrix(self):
        bars_by_symbol = {
            "AAA": calendar_bars(choppy_uptrend(60)),
            "BBB": calendar_bars(choppy_uptrend(80)),
        }
        candidates = [
            candidate("aaa-mr", "AAA", 1.0),
            candidate("bbb-mr", "BBB", 1.0),
        ]
        matrix = build_return_matrix(candidates, bars_by_symbol)
        assert list(matrix.columns) == ["aaa-mr", "bbb-mr"]
        assert matrix.isna().sum().sum() == 0
        assert matrix.index.is_monotonic_increasing
        assert matrix.shape[0] == 60  # common observed span, not the 80-row union

    def test_equal_histories_are_untouched_by_the_trim(self):
        bars = calendar_bars(choppy_uptrend(80))
        candidates = [candidate("z1.0", "AAA", 1.0), candidate("z2.0", "AAA", 2.0)]
        expected = pd.DataFrame(
            {c.label: _run_returns(c, bars, bars.index[0]) for c in candidates}
        ).sort_index().fillna(0.0)
        assert build_return_matrix(candidates, {"AAA": bars}).equals(expected)

    def test_late_starting_candidate_trims_the_head_instead_of_padding_it(self):
        full = daily_bars(choppy_uptrend(80))
        late = full.iloc[20:]
        candidates = [candidate("aaa-mr", "AAA", 1.0), candidate("bbb-mr", "BBB", 1.0)]
        matrix = build_return_matrix(candidates, {"AAA": full, "BBB": late})
        assert list(matrix.index) == list(late.index)
        assert matrix.loc[: full.index[19]].empty  # the long candidate loses its head

    def test_disjoint_histories_are_rejected(self):
        full = daily_bars(choppy_uptrend(80))
        candidates = [candidate("aaa-mr", "AAA", 1.0), candidate("bbb-mr", "BBB", 1.0)]
        with pytest.raises(ValueError, match="do not overlap"):
            build_return_matrix(candidates,
                                {"AAA": full.iloc[:30], "BBB": full.iloc[40:]})


class TestWalkForward:
    def test_dominant_candidate_chosen_every_fold_with_positive_oos(self):
        bars = calendar_bars(choppy_uptrend(260))
        dominant = candidate("dip-buyer", "AAA", 1.0)
        donothing = candidate("inert", "AAA", 6.0)
        result = walk_forward([donothing, dominant], bars, n_folds=4)
        assert len(result.folds) == 4
        assert all(f.chosen_label == "dip-buyer" for f in result.folds)
        assert result.oos_sharpe > 0

    def test_oos_windows_are_contiguous_and_non_overlapping(self):
        bars = calendar_bars(choppy_uptrend(260))
        result = walk_forward([candidate("dip-buyer", "AAA", 1.0)], bars, n_folds=4)
        pos = {ts: i for i, ts in enumerate(bars.index)}
        for a, b in zip(result.folds, result.folds[1:]):
            assert pos[a.test_start] <= pos[a.test_end]           # window well-formed
            assert pos[b.test_start] == pos[a.test_end] + 1       # adjacent, no gap/overlap

    def test_empty_candidates_returns_empty_result(self):
        result = walk_forward([], calendar_bars(choppy_uptrend(260)), n_folds=4)
        assert result.folds == []
        assert result.oos_returns.empty
        assert math.isnan(result.oos_sharpe)
        assert result.variant_oos_sharpe == {}

    def test_too_short_for_folds_returns_empty_result(self):
        result = walk_forward([candidate("dip-buyer", "AAA", 1.0)],
                              calendar_bars(choppy_uptrend(3)), n_folds=4)
        assert result.folds == []
        assert result.oos_returns.empty
        assert math.isnan(result.oos_sharpe)
        assert result.variant_oos_sharpe == {}

    def test_variant_oos_sharpe_keys_are_every_candidate_label(self):
        bars = calendar_bars(choppy_uptrend(260))
        cands = [candidate("dip-buyer", "AAA", 1.0), candidate("inert", "AAA", 6.0)]
        result = walk_forward(cands, bars, n_folds=4)
        assert set(result.variant_oos_sharpe) == {"dip-buyer", "inert"}

    def test_winner_oos_sharpe_matches_aggregate_when_one_candidate_wins_all_folds(self):
        # A single dominant candidate is the fold winner every fold, so the
        # honest select-then-trade oos_sharpe equals that variant's concatenated
        # per-variant OOS Sharpe.
        bars = calendar_bars(choppy_uptrend(260))
        dominant = candidate("dip-buyer", "AAA", 1.0)
        donothing = candidate("inert", "AAA", 6.0)
        result = walk_forward([donothing, dominant], bars, n_folds=4)
        assert all(f.chosen_label == "dip-buyer" for f in result.folds)
        assert result.variant_oos_sharpe["dip-buyer"] == pytest.approx(result.oos_sharpe)

    def test_oos_sharpe_unchanged_for_known_fixture(self):
        bars = calendar_bars(choppy_uptrend(260))
        cands = [candidate("dip-buyer", "AAA", 1.0), candidate("inert", "AAA", 6.0)]
        assert walk_forward(cands, bars, n_folds=4).oos_sharpe == pytest.approx(
            _EXPECTED_OOS_SHARPE
        )


class TestDailyReturnSeam:
    def test_non_trading_gap_keeps_the_return_across_it(self):
        # Business-day bars: the weekend has no bar, so resampling to calendar
        # days inserts empty Saturdays and Sundays. Differencing before those
        # are dropped strands every Monday against an empty Sunday.
        bars = daily_bars(sawtooth(120))
        returns = _run_returns(custom_candidate("anchor", LastRowAnchor()),
                               bars, bars.index[0]).dropna()
        assert list(returns.index) == list(bars.index[1:])

    def test_oos_covers_every_test_bar_but_the_first_across_non_trading_gaps(self):
        bars = daily_bars(sawtooth(260))
        result = walk_forward([custom_candidate("anchor", LastRowAnchor())],
                              bars, n_folds=4)
        expected = [ts for _, test_start, test_end in fold_bounds(len(bars), 4)
                    for ts in bars.index[test_start + 1:test_end]]
        assert list(result.oos_returns.index) == expected


class TestFoldBoundary:
    """LastRowAnchor reads the final row of the frame it is prepared on, so its
    returns move whenever that frame ends somewhere else. Every pin here would
    break if walk_forward ever handed a leg more bars than its window."""

    def test_anchor_returns_move_with_the_end_of_the_frame(self):
        bars = calendar_bars(sawtooth(260))
        cand = custom_candidate("anchor", LastRowAnchor())
        for train_start, test_start, _ in fold_bounds(len(bars), 4):
            truncated = _run_returns(cand, bars.iloc[:test_start],
                                     bars.index[train_start])
            on_full = _run_returns(cand, bars,
                                   bars.index[train_start]).reindex(truncated.index)
            assert not truncated.equals(on_full)

    def test_train_leg_is_handed_no_bar_at_or_after_test_start(self, monkeypatch):
        bars = calendar_bars(sawtooth(260))
        cand = custom_candidate("anchor", LastRowAnchor())
        calls = record_run_returns(monkeypatch)
        walk_forward([cand], bars, n_folds=4)
        train_calls = calls[0::2]  # one train then one test call per fold
        bounds = fold_bounds(len(bars), 4)
        assert len(train_calls) == len(bounds)
        for (train_start, test_start, _), (frame, sim_start, returns) in zip(
                bounds, train_calls):
            assert frame.index[-1] < bars.index[test_start]
            assert frame.equals(bars.iloc[:test_start])
            assert sim_start == bars.index[train_start]
            assert returns.equals(_run_returns(cand, bars.iloc[:test_start],
                                               bars.index[train_start]))

    def test_test_leg_is_handed_no_bar_after_test_end(self, monkeypatch):
        bars = calendar_bars(sawtooth(260))
        cand = custom_candidate("anchor", LastRowAnchor())
        calls = record_run_returns(monkeypatch)
        walk_forward([cand], bars, n_folds=4)
        test_calls = calls[1::2]
        bounds = fold_bounds(len(bars), 4)
        for (_, test_start, test_end), (frame, sim_start, returns) in zip(
                bounds, test_calls):
            assert frame.equals(bars.iloc[:test_end])
            assert sim_start == bars.index[test_start]
            assert returns.equals(_run_returns(cand, bars.iloc[:test_end],
                                               bars.index[test_start]))
        # The last fold's window is the whole frame, so only the earlier folds
        # can show that trailing bars were withheld.
        for _, test_start, test_end in bounds[:-1]:
            truncated = _run_returns(cand, bars.iloc[:test_end], bars.index[test_start])
            on_full = _run_returns(cand, bars,
                                   bars.index[test_start]).reindex(truncated.index)
            assert not truncated.equals(on_full)

    def test_walk_forward_trusts_prepare_and_the_probe_is_what_checks_it(self):
        # Division of labor: walk_forward's truncation stops a strategy seeing
        # bars outside its window, but a prepare() that reads forward inside the
        # window is invisible to it. The diagnose probe is the enforcement.
        bars = calendar_bars(sawtooth(260))
        peeker = NextBarPeeker()
        result = walk_forward([custom_candidate("peeker", peeker)], bars, n_folds=4)
        assert len(result.folds) == 4
        assert result.oos_sharpe > 0
        probe = probe_lookahead(peeker, bars, sample_every=13)
        assert probe.leaks
        assert decisive(probe.flat_actions) and decisive(probe.held_actions)


class TestOosCoverage:
    def test_every_moving_day_lands_in_exactly_one_fold(self):
        bars = calendar_bars(sawtooth(260))
        result = walk_forward([custom_candidate("long", AlwaysLong())], bars, n_folds=4)
        bounds = fold_bounds(len(bars), 4)
        expected = [ts for _, test_start, test_end in bounds
                    for ts in bars.index[test_start + 1:test_end]]
        assert (result.oos_returns != 0).all()  # equity really moves every day
        assert result.oos_returns.index.is_unique
        assert list(result.oos_returns.index) == expected

    def test_each_folds_first_test_bar_carries_no_return(self):
        bars = calendar_bars(sawtooth(260))
        result = walk_forward([custom_candidate("long", AlwaysLong())], bars, n_folds=4)
        bounds = fold_bounds(len(bars), 4)
        covered = set(result.oos_returns.index)
        # The equity curve starts at sim_start, so the first bar of a window has
        # no prior close to difference against and drops out of OOS entirely.
        assert [ts for ts in bars.index[bounds[0][1]:] if ts not in covered] == [
            bars.index[test_start] for _, test_start, _ in bounds]

    def test_intraday_boundary_day_is_counted_once_not_twice(self):
        bars = daily_bars(sawtooth(260), freq="8h", start="2020-01-01 00:00")
        bounds = fold_bounds(len(bars), 4)
        straddled = [(k, bars.index[test_start].normalize())
                     for k, (_, test_start, _) in enumerate(bounds)
                     if k > 0 and bars.index[test_start - 1].date()
                     == bars.index[test_start].date()]
        assert straddled  # the seam has bite: a boundary falls mid-day
        result = walk_forward([custom_candidate("long", AlwaysLong())], bars, n_folds=4)
        index = list(result.oos_returns.index)
        assert result.oos_returns.index.is_unique
        assert index == sorted(index)
        # A day split across a boundary is measured by the earlier fold; the
        # later fold's copy of it is the one with no prior close.
        for _, day in straddled:
            assert index.count(day) == 1

    def test_train_path_leading_nan_leaves_the_is_ranking_intact(self):
        bars = calendar_bars(sawtooth(260))
        cand = custom_candidate("anchor", LastRowAnchor())
        for train_start, test_start, _ in fold_bounds(len(bars), 4):
            train = _run_returns(cand, bars.iloc[:test_start], bars.index[train_start])
            assert math.isnan(train.iloc[0])
            assert sharpe_ratio(train) == sharpe_ratio(train.dropna()) != 0.0
