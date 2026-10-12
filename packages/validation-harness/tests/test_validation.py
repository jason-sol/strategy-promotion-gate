import dataclasses
import json
import math

import numpy as np
import pandas as pd
import pytest
from harness_fixtures import (
    SCALE_PAIRED_SHAPES,
    bars,
    compound,
    curve,
    drift,
    drifting_field,
    empty,
    near_duplicate_family,
    noisy,
    one_bump_per_block,
    return_matrix,
    scale_paired_blocks,
)

from validation_harness import (
    Candidate,
    CscvReport,
    build_return_matrix,
    cscv_pbo,
    cscv_report,
    deflated_sharpes,
    deflated_sharpes_from_matrix,
    probabilistic_sharpe_ratio,
    walk_forward,
)
from validation_harness.validation import _daily_returns


class TestDailyReturns:
    def test_weekend_gap_lands_on_the_monday(self):
        """Resampling a business-day curve to calendar days invents empty
        weekend rows; dropping them before differencing keeps the Friday-to-
        Monday move instead of stranding it against an empty Sunday."""
        index = bars(10).index
        returns = _daily_returns(curve(index, 0.01))

        assert len(returns) == len(index)
        assert not returns.iloc[1:].isna().any()
        assert returns.iloc[1:].to_numpy() == pytest.approx(0.01)

    def test_only_the_first_row_prices_no_return(self):
        returns = _daily_returns(curve(bars(5, freq="1D").index, 0.02))
        assert math.isnan(returns.iloc[0])
        assert len(returns.dropna()) == 4


class TestBuildReturnMatrix:
    def test_one_column_per_candidate_over_the_shared_span(self):
        frame = bars(30)
        matrix = build_return_matrix([drift("a"), drift("b", step=0.02)], {"A": frame})

        assert list(matrix.columns) == ["a", "b"]
        assert matrix.index.min() == frame.index.min()
        assert matrix.index.max() == frame.index.max()

    def test_no_candidates_yields_an_empty_frame(self):
        assert build_return_matrix([], {}).empty

    def test_offset_history_is_head_trimmed_not_zero_padded(self):
        """Zero-vol padding over a short-history candidate's missing head would
        flatter its Sharpe and DSR, so the non-overlap is cut instead."""
        early = bars(40, start="2020-01-01")
        late = bars(40, start="2020-02-03")
        matrix = build_return_matrix(
            [drift("early", symbol="A"), drift("late", symbol="B", step=0.02)],
            {"A": early, "B": late},
        )

        assert matrix.index.min() == late.index.min()
        assert matrix.index.max() == early.index.max()
        assert len(matrix) < len(early)

    def test_interior_gap_is_filled_flat(self):
        """A date the shared span covers but one candidate never priced is a
        genuine flat day for that candidate. Only the leading row, which prices
        no return for anyone, is flat for the rest."""
        frame = bars(30)
        matrix = build_return_matrix([drift("a"), drift("b", skip=10)], {"A": frame})

        assert not matrix.isna().to_numpy().any()
        assert (matrix.iloc[0] == 0.0).all()
        assert (matrix["a"] == 0.0).sum() == 1
        assert (matrix["b"] == 0.0).sum() == 2

    def test_disjoint_histories_raise(self):
        with pytest.raises(ValueError, match="do not overlap"):
            build_return_matrix(
                [drift("jan", symbol="A"), drift("mar", symbol="B")],
                {"A": bars(20, start="2020-01-01"), "B": bars(20, start="2020-03-02")},
            )

    def test_a_candidate_with_no_returns_raises(self):
        with pytest.raises(ValueError, match="produced no returns"):
            build_return_matrix([drift("a"), empty("silent")], {"A": bars(20)})

    def test_sim_start_carves_out_a_warmup_window(self):
        frame = bars(40)
        full = build_return_matrix([drift("a")], {"A": frame})
        late = build_return_matrix([drift("a")], {"A": frame}, sim_start=frame.index[10])

        assert len(late) < len(full)
        assert late.index.min() == frame.index[10]


class TestDeflatedSharpes:
    def test_a_lone_column_has_no_dispersion_so_the_dsr_is_an_undeflated_psr(self):
        matrix = build_return_matrix([noisy("only", drift_per_bar=0.002)], {"A": bars(60)})
        assert deflated_sharpes_from_matrix(matrix, 1) == {
            "only": pytest.approx(probabilistic_sharpe_ratio(matrix["only"]))
        }

    def test_more_trials_deflate_every_column(self):
        matrix = build_return_matrix(
            [noisy("a", seed=1, drift_per_bar=0.002), noisy("b", seed=2, drift_per_bar=0.001)],
            {"A": bars(80)},
        )
        few = deflated_sharpes_from_matrix(matrix, 2)
        many = deflated_sharpes_from_matrix(matrix, 500)

        assert set(few) == {"a", "b"}
        assert all(many[label] <= few[label] for label in few)

    def test_candidate_entry_point_matches_the_matrix_entry_point(self):
        candidates = [noisy("a", seed=1, drift_per_bar=0.002), noisy("b", seed=2)]
        bars_by_symbol = {"A": bars(80)}
        staged = deflated_sharpes_from_matrix(
            build_return_matrix(candidates, bars_by_symbol), len(candidates)
        )

        assert deflated_sharpes(candidates, bars_by_symbol) == staged


class TestCscvPbo:
    def test_odd_block_counts_cannot_split_symmetrically(self):
        matrix = return_matrix({"a": [0.01] * 64, "b": [0.02] * 64})
        with pytest.raises(ValueError, match="must be even"):
            cscv_pbo(matrix, n_blocks=7)

    def test_not_computable_with_a_single_column(self):
        matrix = return_matrix({"a": list(np.linspace(-0.01, 0.01, 64))})
        assert math.isnan(cscv_pbo(matrix, n_blocks=4))

    def test_not_computable_when_a_block_would_be_a_single_row(self):
        matrix = return_matrix({"a": [0.01] * 7, "b": [0.02] * 7})
        assert math.isnan(cscv_pbo(matrix, n_blocks=4))

    def test_a_uniformly_dominant_column_is_never_overfit(self):
        rng = np.random.default_rng(7)
        noise = rng.standard_normal(160) * 0.01
        matrix = return_matrix({"weak": list(noise), "strong": list(noise + 0.05)})
        assert cscv_pbo(matrix, n_blocks=6) == 0.0

    def test_a_column_that_only_wins_in_sample_is_flagged(self):
        """Strong over the first half, weak over the second: whenever the early
        blocks are in-sample the winner is bottom-half out-of-sample."""
        rng = np.random.default_rng(11)
        noise = rng.standard_normal(160) * 0.01
        mined = list(noise[:80] + 0.05) + list(noise[80:] - 0.05)
        steady = list(rng.standard_normal(160) * 0.01 + 0.001)
        assert cscv_pbo(return_matrix({"mined": mined, "steady": steady}), n_blocks=4) > 0.0

    def test_probability_stays_in_the_unit_interval(self):
        rng = np.random.default_rng(3)
        matrix = return_matrix(
            {name: list(rng.standard_normal(200) * 0.01) for name in ("a", "b", "c", "d")}
        )
        assert 0.0 <= cscv_pbo(matrix, n_blocks=6) <= 1.0


def noise_matrix(n_cols: int = 6, t: int = 240, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((t, n_cols)) * 0.01
    return return_matrix({f"c{i}": values[:, i] for i in range(n_cols)})


def dominant_first_column(n_cols: int = 20, t: int = 250, seed: int = 7) -> pd.DataFrame:
    """Column c0 carries the same vol as the rest and a clearly higher mean."""
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((t, n_cols))
    values[:, 0] += 0.5
    return return_matrix({f"c{i}": values[:, i] for i in range(n_cols)})


class TestCscvReport:
    @pytest.mark.parametrize("n_blocks", [4, 8, 10])
    def test_pbo_panel_is_what_the_scalar_wrapper_returns(self, n_blocks):
        matrix = noise_matrix()
        assert cscv_report(matrix, n_blocks).pbo == cscv_pbo(matrix, n_blocks)

    def test_p_oos_loss_is_one_when_the_is_best_always_loses_out_of_sample(self):
        assert cscv_report(one_bump_per_block(), n_blocks=8).p_oos_loss == 1.0

    def test_p_oos_loss_is_zero_when_the_field_wins_on_every_leg(self):
        assert cscv_report(drifting_field(0.02), n_blocks=8).p_oos_loss == 0.0

    def test_p_oos_loss_is_one_when_the_field_loses_on_every_leg(self):
        assert cscv_report(drifting_field(-0.02), n_blocks=8).p_oos_loss == 1.0

    def test_median_logit_is_positive_when_the_is_best_ranks_top_half_oos(self):
        result = cscv_report(dominant_first_column(), n_blocks=10)

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

        labels = list(matrix.columns)
        assert list(result.oos_rank_mean) == labels
        assert list(result.is_best_share) == labels

    def test_is_best_share_is_a_distribution_over_the_columns(self):
        result = cscv_report(one_bump_per_block(), n_blocks=8)

        assert sum(result.is_best_share.values()) == pytest.approx(1.0)
        assert all(0.0 <= share <= 1.0 for share in result.is_best_share.values())

    def test_oos_rank_mean_stays_in_the_unit_interval(self):
        result = cscv_report(noise_matrix(), n_blocks=8)
        assert all(0.0 <= rank <= 1.0 for rank in result.oos_rank_mean.values())

    def test_oos_rank_mean_puts_a_dominant_column_above_the_whole_field(self):
        ranks = cscv_report(dominant_first_column(), n_blocks=10).oos_rank_mean

        assert ranks["c0"] == pytest.approx(19 / 20)
        assert max(ranks, key=ranks.get) == "c0"

    @pytest.mark.parametrize("n_cols,t,n_blocks", [(1, 250, 16), (4, 5, 16), (4, 7, 4)])
    def test_degenerate_input_is_all_nan_with_empty_panels(self, n_cols, t, n_blocks):
        result = cscv_report(noise_matrix(n_cols, t), n_blocks)

        assert math.isnan(result.pbo)
        assert math.isnan(result.p_oos_loss)
        assert math.isnan(result.median_logit)
        assert math.isnan(result.degradation_slope)
        assert result.logits == ()
        assert result.oos_rank_mean == {}
        assert result.is_best_share == {}

    def test_degenerate_pbo_matches_the_scalar_wrapper(self):
        matrix = noise_matrix(n_cols=1)
        assert math.isnan(cscv_report(matrix).pbo)
        assert math.isnan(cscv_pbo(matrix))

    def test_odd_block_counts_cannot_split_symmetrically(self):
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


def rising(label: str, seed: int) -> Candidate:
    return noisy(label, seed=seed, drift_per_bar=0.01)


def falling(label: str, seed: int) -> Candidate:
    return noisy(label, seed=seed, drift_per_bar=-0.01)


class TestWalkForward:
    def test_folds_tile_the_window(self):
        # 53 does not divide into n_folds + 1 blocks of 10, so the last fold has
        # to absorb the ragged tail rather than stopping a block short.
        frame = bars(53)
        result = walk_forward([rising("up", 1), falling("down", 2)], frame, n_folds=4)

        assert len(result.folds) == 4
        assert [f.train_start for f in result.folds] == [frame.index[i * 10] for i in range(4)]
        assert [f.test_start for f in result.folds] == [frame.index[(i + 1) * 10] for i in range(4)]
        assert result.folds[-1].test_end == frame.index[-1]
        for fold in result.folds:
            assert fold.train_start < fold.test_start <= fold.test_end

    def test_oos_returns_follow_the_selected_candidate_only(self):
        frame = bars(53)
        result = walk_forward([rising("up", 1), falling("down", 2)], frame, n_folds=4)

        assert {f.chosen_label for f in result.folds} == {"up"}
        assert set(result.variant_oos_sharpe) == {"up", "down"}
        assert result.variant_oos_sharpe["down"] < 0.0 < result.variant_oos_sharpe["up"]
        assert result.oos_sharpe == pytest.approx(result.variant_oos_sharpe["up"])
        assert not result.oos_returns.empty

    def test_selection_is_rerun_every_fold(self):
        """Two candidates whose edge swaps mid-window: a harness that fixed the
        winner at the first fold would report one label throughout."""
        frame = bars(60)
        mid = frame.index[30]

        def alternating(label: str, early: float, late: float) -> Candidate:
            def run(f: pd.DataFrame, sim_start: pd.Timestamp) -> pd.Series:
                rng = np.random.default_rng(5)
                noise = 0.002 * rng.standard_normal(len(f))
                steps = pd.Series(
                    np.where(f.index < mid, early, late) + noise, index=f.index
                )
                return compound(steps[steps.index >= sim_start])

            return Candidate(label, "A", run)

        result = walk_forward(
            [alternating("first", 0.02, -0.02), alternating("second", -0.02, 0.02)],
            frame,
            n_folds=4,
        )
        assert len({f.chosen_label for f in result.folds}) == 2

    def test_too_few_bars_for_a_block_yields_an_empty_result(self):
        result = walk_forward([rising("up", 1)], bars(3), n_folds=4)

        assert result.folds == []
        assert result.oos_returns.empty
        assert math.isnan(result.oos_sharpe)
        assert result.variant_oos_sharpe == {}

    def test_no_candidates_yields_an_empty_result(self):
        result = walk_forward([], bars(50), n_folds=4)

        assert result.folds == []
        assert math.isnan(result.oos_sharpe)
