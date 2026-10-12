import dataclasses
import json
import math
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bot import optimize, search
from bot.strategies.trend_following import TrendFollowing
from bot.validation import CscvReport
from config import MARKET_DATA_FEED, VALIDATION, AssetClass
from helpers import daily_bars

# A tiny field and a short frame: what is under test is the search's shape, and
# the real 130-cell run over a decade of bars takes minutes. Twenty-four
# feasible genomes and a budget two generations of four cannot spend keeps "did
# the search find the front" a real question rather than an exhaustive sweep in
# disguise, and makes the generation ceiling this fixture's stop; the budget
# stop has its own tests.
SPACE = search.GeneSpace(fast=(5, 10, 15), slow=(20, 30), trail=(2.0, 3.0),
                         volume=(None, 1.5))
N_BLOCKS = 4
N_FOLDS = 2
POPULATION = 4
GENERATIONS = 2
BUDGET = 12


def choppy_uptrend(n: int) -> list[float]:
    t = np.arange(n)
    return list(100.0 + 0.05 * t + 4.0 * np.sin(t / 5.0) + 1.5 * np.sin(t / 1.7))


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return daily_bars(choppy_uptrend(500), start="2019-01-01")  # block 166 -> slow <= 31


def search_kwargs(**overrides) -> dict:
    kwargs = dict(space=SPACE, population=POPULATION, generations=GENERATIONS,
                  budget=BUDGET, n_blocks=N_BLOCKS, n_folds=N_FOLDS)
    return kwargs | overrides


@pytest.fixture(scope="module")
def result(bars) -> "search.SearchResult":
    return search.run_search(bars, **search_kwargs(seed=0))


@pytest.fixture(scope="module")
def field(bars) -> "optimize.GridResult":
    return optimize.score_field(
        optimize.trend_candidates(fasts=SPACE.fast, slows=SPACE.slow,
                                  trails=SPACE.trail, volume_mults=SPACE.volume),
        bars, symbol=optimize.TREND_SYMBOL, spec=optimize.TREND_FIELD,
        n_blocks=N_BLOCKS, n_folds=N_FOLDS)


@pytest.fixture(scope="module")
def comparison(result, field) -> "search.Comparison":
    return search.compare(result, field)


# --- the volume gene --------------------------------------------------------


def test_mutate_rewrites_the_volume_gene():
    # The trap this test exists for: crossover and mutate drew rng.random(3)
    # against a three-gene genome. A fourth gene nobody draws for is silently
    # frozen at its seed value while the search still reports a full run and
    # writes a plausible artifact.
    space = search.GeneSpace(fast=(5,), slow=(40,), trail=(3.0,), volume=(1.5,))
    genome = search.Genome(fast=5, slow=40, trail=3.0, volume=None)
    mutated = search.mutate(genome, np.random.default_rng(0), space, rate=1.0)
    assert mutated.volume == 1.5


def test_crossover_takes_the_volume_gene_from_either_parent():
    a = search.Genome(fast=5, slow=40, trail=3.0, volume=None)
    b = search.Genome(fast=5, slow=40, trail=3.0, volume=1.5)
    seen = {search.crossover(a, b, np.random.default_rng(s)).volume
            for s in range(30)}
    assert seen == {None, 1.5}


def test_repair_snaps_the_volume_gene_into_the_space():
    space = search.GeneSpace(fast=(5,), slow=(40,), trail=(3.0,), volume=(1.5,))
    stray = search.Genome(fast=5, slow=40, trail=3.0, volume=9.9)
    assert search.repair(stray, space, np.random.default_rng(0)).volume == 1.5


def test_repair_preserves_a_volume_gene_that_is_already_legal():
    # repair() rebuilds the genome field by field, so a gene it forgets to pass
    # through is silently reset to the dataclass default for every genome the
    # search ever constructs — random_genome routes through here.
    space = search.GeneSpace(fast=(5,), slow=(40,), trail=(3.0,), volume=(None, 1.2, 1.5))
    legal = search.Genome(fast=5, slow=40, trail=3.0, volume=1.2)
    assert search.repair(legal, space, np.random.default_rng(0)).volume == 1.2


def test_candidate_for_carries_the_volume_gene_into_the_strategy():
    genome = search.Genome(fast=5, slow=40, trail=3.0, volume=1.5)
    candidate = search.candidate_for(genome, "USO", AssetClass.US_EQUITY, 0.0005)
    assert candidate.strategy.volume_gate_mult == 1.5


# --- gene space and feasibility ---------------------------------------------


def test_gene_space_defaults_to_the_exhaustive_field():
    space = search.GeneSpace()
    assert space.fast == optimize.TREND_FAST
    assert space.slow == optimize.TREND_SLOW
    assert space.trail == optimize.TREND_TRAIL
    assert space.volume == optimize.TREND_VOLUME_MULT
    assert space.size == 390  # the same field the exhaustive grid enumerates


def test_limiting_the_space_drops_infeasible_slow_periods():
    space = search.GeneSpace().limited(60)
    assert space.slow == (40, 60)
    assert space.fast == optimize.TREND_FAST
    assert search.GeneSpace().limited(100).slow == optimize.TREND_SLOW
    with pytest.raises(ValueError):
        search.GeneSpace().limited(4)  # nothing feasible left to search


def test_random_genomes_are_feasible_and_seed_reproducible():
    rng = np.random.default_rng(7)
    drawn = [search.random_genome(rng, SPACE) for _ in range(20)]
    again = [search.random_genome(np.random.default_rng(7), SPACE)]
    assert drawn[0] == again[0]
    assert drawn != [search.random_genome(np.random.default_rng(8), SPACE)] * 20
    for g in drawn:
        assert g.fast in SPACE.fast and g.slow in SPACE.slow
        assert g.trail in SPACE.trail
        assert g.fast < g.slow


def test_the_initial_population_is_distinct_so_no_budget_is_wasted():
    rng = np.random.default_rng(0)
    pop = search.initial_population(rng, SPACE, 8)
    assert len(pop) == len(set(pop)) == 8
    # A population larger than the space cannot be distinct; it is capped, not
    # padded with duplicates that would each cost a backtest.
    small = search.GeneSpace(fast=(5,), slow=(20,), trail=(2.0, 3.0),
                             volume=(None,))
    assert len(search.initial_population(rng, small, 8)) == small.size == 2


def test_repair_snaps_genes_into_the_space_and_uninverts_the_pair():
    rng = np.random.default_rng(0)
    repaired = search.repair(search.Genome(fast=99, slow=21, trail=2.4),
                             SPACE, rng)
    assert repaired.slow == 20        # 21 snaps to the nearest allowed value
    assert repaired.trail == 2.0
    assert repaired.fast < repaired.slow
    assert repaired.fast in SPACE.fast
    assert search.repair(search.Genome(5, 30, 3.0), SPACE, rng) == \
        search.Genome(5, 30, 3.0)     # already feasible, left alone


# --- NSGA-II-lite primitives ------------------------------------------------


def test_non_dominated_sort_layers_a_known_set():
    # Minimize both columns. (0,0) dominates everything; (1,2) and (2,1) are
    # mutually non-dominated; (3,3) is dominated by both of those.
    fronts = search.non_dominated_sort(np.array([[0.0, 0.0], [1.0, 2.0],
                                                 [2.0, 1.0], [3.0, 3.0]]))
    assert fronts == [[0], [1, 2], [3]]


def test_non_dominated_sort_sends_unscorable_points_to_the_last_front():
    fronts = search.non_dominated_sort(np.array([[1.0, 1.0],
                                                 [np.inf, np.inf]]))
    assert fronts == [[0], [1]]
    assert search.non_dominated_sort(np.empty((0, 2))) == []


def test_the_objective_row_negates_dsr_and_keeps_the_drawdown():
    assert search.objective_row(0.8, 0.12, 1.4) == [-0.8, 0.12]


def test_a_cell_at_or_under_the_sharpe_floor_takes_the_violation_row():
    base = search.INFEASIBLE_BASE
    assert search.objective_row(0.8, 0.012, -0.20) == [base + 0.20, base + 0.20]
    assert search.objective_row(0.8, 0.012, search.SHARPE_FLOOR) == [base, base]
    assert search.objective_row(0.8, 0.012, float("nan")) == [np.inf, np.inf]


def test_a_cell_without_a_number_takes_the_violation_row():
    base = search.INFEASIBLE_BASE
    assert search.objective_row(float("nan"), 0.12, 1.4) == [base, base]
    assert search.objective_row(0.8, float("nan"), 1.4) == [base, base]
    assert search.objective_row(float("nan"), 0.12, float("nan")) == [np.inf,
                                                                      np.inf]


def test_a_cell_missing_a_number_but_not_losing_sits_on_the_base():
    base = search.INFEASIBLE_BASE
    assert search.objective_row(float("nan"), 0.12, 1.4) == [base, base]
    fronts = search.non_dominated_sort(np.array([
        search.objective_row(float("nan"), 0.12, 1.4),
        search.objective_row(0.8, 0.30, 0.0)]))
    assert fronts == [[0, 1]]   # tied on the base, neither dominates the other


def test_a_less_losing_infeasible_cell_dominates_a_more_losing_one():
    fronts = search.non_dominated_sort(np.array([
        search.objective_row(0.8, 0.30, -0.3),
        search.objective_row(0.8, 0.30, -0.1)]))
    assert fronts == [[1], [0]]


def test_any_feasible_cell_dominates_any_infeasible_one():
    fronts = search.non_dominated_sort(np.array([
        search.objective_row(0.01, 0.99, -0.1),   # barely under the floor
        search.objective_row(0.01, 0.99, 0.01)]))  # worst feasible cell there is
    assert fronts == [[1], [0]]


def test_the_floor_count_and_the_objective_row_share_one_feasible_set():
    cells = [plain_cell("T 10/40/2.0", 1.4, 0.8, float("nan")),
             plain_cell("T 30/80/2.5", 1.4, float("nan"), 0.12),
             plain_cell("T 50/100/4.0", 1.4, 0.8, 0.12)]
    assert search._below_floor(cells) == 2


def test_crowding_distance_protects_the_boundaries():
    objs = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
    dist = search.crowding_distance(objs)
    assert math.isinf(dist[0]) and math.isinf(dist[3])
    assert math.isfinite(dist[1]) and math.isfinite(dist[2])
    assert search.crowding_distance(objs[:2]).tolist() == [np.inf, np.inf]


def test_crowding_distance_over_a_front_with_no_width_stays_quiet():
    # Under the floor a whole front can be infeasible, and inf minus inf is a
    # warning rather than a span.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dist = search.crowding_distance(np.full((3, 2), np.inf))
    assert dist.tolist() == [np.inf, 0.0, np.inf]  # as for any front of ties


def test_tournament_prefers_lower_rank_then_more_room():
    rank = np.array([0, 0, 1])
    crowd = np.array([0.5, 2.0, np.inf])
    assert search.tournament_winner(2, 0, rank, crowd) == 0  # rank beats crowding
    assert search.tournament_winner(0, 1, rank, crowd) == 1  # tie -> less crowded
    rng = np.random.default_rng(3)
    picks = search.binary_tournament(rng, rank, crowd, 40)
    assert len(picks) == 40
    assert set(picks) <= {0, 1, 2}
    assert np.mean(rank[picks]) < np.mean(rank)  # selection pressure exists


def test_uniform_crossover_takes_every_gene_from_one_parent_or_the_other():
    a, b = search.Genome(5, 20, 2.0), search.Genome(15, 30, 3.0)
    rng = np.random.default_rng(1)
    children = [search.crossover(a, b, rng) for _ in range(30)]
    for child in children:
        assert child.fast in (a.fast, b.fast)
        assert child.slow in (a.slow, b.slow)
        assert child.trail in (a.trail, b.trail)
    assert len(set(children)) > 1  # genes really are mixed, not copied wholesale


def test_mutation_stays_inside_the_space_and_respects_its_rate():
    rng = np.random.default_rng(2)
    genome = search.Genome(5, 20, 2.0)
    assert search.mutate(genome, rng, SPACE, 0.0) == genome
    mutated = [search.mutate(genome, rng, SPACE, 1.0) for _ in range(30)]
    assert any(m != genome for m in mutated)
    for m in mutated:
        assert m.fast in SPACE.fast and m.slow in SPACE.slow
        assert m.trail in SPACE.trail


# --- the search loop --------------------------------------------------------


def test_search_is_reproducible_from_its_seed_alone(bars, result):
    repeat = search.run_search(bars, **search_kwargs(seed=0))
    assert repeat.seed == result.seed == 0
    assert [c.label for c in repeat.evaluated.cells] == \
        [c.label for c in result.evaluated.cells]
    assert repeat.front == result.front
    assert [c.dsr for c in repeat.evaluated.cells] == \
        [c.dsr for c in result.evaluated.cells]
    assert [(t.generation, t.n_evaluated, t.front_size, t.n_trials)
            for t in repeat.trace] == \
        [(t.generation, t.n_evaluated, t.front_size, t.n_trials)
         for t in result.trace]


def test_a_different_seed_walks_a_different_path(bars, result):
    other = search.run_search(bars, **search_kwargs(seed=1))
    labels = {c.label for c in result.evaluated.cells}
    assert {c.label for c in other.evaluated.cells} != labels


def test_search_respects_its_budget_and_backtests_nothing_twice(bars, monkeypatch):
    calls = []
    real = search.build_return_matrix

    def spy(candidates, bars_by_symbol, sim_start=None):
        calls.append([c.label for c in candidates])
        return real(candidates, bars_by_symbol, sim_start)

    monkeypatch.setattr(search, "build_return_matrix", spy)
    run = search.run_search(bars, **search_kwargs(seed=0))
    evaluated = [c.label for c in run.evaluated.cells]
    assert len(evaluated) == len(set(evaluated))  # genome dedupe
    assert len(evaluated) <= BUDGET
    scored = [label for call in calls for label in call]
    assert sorted(scored) == sorted(evaluated)  # one backtest per unique genome


def test_deflation_tightens_as_the_search_spends_trials(result):
    trials = [t.n_trials for t in result.trace]
    assert trials == sorted(trials)
    assert [t.n_trials for t in result.trace] == [t.n_evaluated for t in result.trace]
    assert result.trace[-1].n_trials == len(result.evaluated.cells)
    assert result.evaluated.n_trials == len(result.evaluated.cells)
    assert [t.generation for t in result.trace] == [1, 2]
    assert all(t.front_size >= 1 for t in result.trace)


def test_walk_forward_runs_once_over_the_front_not_per_genome(bars, monkeypatch):
    calls = []
    real = search.walk_forward

    def spy(candidates, frame, n_folds=4):
        calls.append([c.label for c in candidates])
        return real(candidates, frame, n_folds=n_folds)

    monkeypatch.setattr(search, "walk_forward", spy)
    run = search.run_search(bars, **search_kwargs(seed=0))
    assert len(calls) == 1                # selection never sees an OOS number
    assert sorted(calls[0]) == sorted(run.front)


def test_only_front_cells_carry_an_oos_number(result):
    for cell in result.evaluated.cells:
        if cell.label in result.front:
            assert math.isfinite(cell.oos_sharpe)
        else:
            assert math.isnan(cell.oos_sharpe)
            assert cell.passes is False  # no OOS evidence is not a pass


def test_the_front_is_pareto_optimal_over_dsr_and_drawdown(result):
    cells = {c.label: c for c in result.evaluated.cells}
    assert result.front
    assert set(result.front) <= set(cells)
    feasible = [c for c in cells.values() if c.sharpe > search.SHARPE_FLOOR]
    assert feasible
    for label in result.front:
        front_cell = cells[label]
        assert front_cell.sharpe > search.SHARPE_FLOOR
        for other in feasible:
            dominated = (other.dsr >= front_cell.dsr
                         and other.max_dd <= front_cell.max_dd
                         and (other.dsr > front_cell.dsr
                              or other.max_dd < front_cell.max_dd))
            assert not dominated, f"{other.label} dominates front cell {label}"


def test_the_evaluated_population_is_scored_as_its_own_field(result):
    assert result.evaluated.spec is optimize.TREND_FIELD
    assert result.evaluated.symbol == optimize.TREND_SYMBOL
    assert result.evaluated.n_blocks == N_BLOCKS
    assert result.evaluated.n_folds == N_FOLDS
    assert all(0.0 <= c.dsr <= 1.0 for c in result.evaluated.cells)
    assert all(0.0 <= c.max_dd <= 1.0 for c in result.evaluated.cells)
    assert all(c.params["allow_short"] is False for c in result.evaluated.cells)
    pbo = result.evaluated.pbo
    assert math.isnan(pbo) or 0.0 <= pbo <= 1.0
    p_oos_loss = result.evaluated.p_oos_loss
    assert math.isnan(p_oos_loss) or 0.0 <= p_oos_loss <= 1.0
    corr = result.evaluated.field_correlation
    assert math.isnan(corr.mean) or -1.0 <= corr.min <= corr.mean <= 1.0
    assert all(c.passes == VALIDATION.passes_validation(
        c.dsr, p_oos_loss, c.oos_sharpe) for c in result.evaluated.cells)


def test_search_cells_record_whether_they_traded(bars):
    starved = search.run_search(bars, **search_kwargs(
        seed=0, space=search.GeneSpace(fast=(5,), slow=(20,), trail=(2.0,),
                                       volume=(None, 1e9))))
    by_label = {c.label: c for c in starved.evaluated.cells}
    assert by_label["T 5/20/2.0"].traded is True
    assert by_label["T 5/20/2.0/1000000000.0"].traded is False
    assert starved.evaluated.degenerate_labels == ("T 5/20/2.0/1000000000.0",)


@pytest.fixture(scope="module")
def budget_stopped(bars) -> "search.SearchResult":
    return search.run_search(bars, **search_kwargs(seed=0, generations=20,
                                                   budget=8))


def test_the_budget_stops_the_search(budget_stopped):
    # The budget is the stop: once it is spent there is nothing left to buy, so
    # the remaining generations are not run and leave no trace entry.
    assert budget_stopped.unique_evals == 8
    assert budget_stopped.generations_run < 20
    assert budget_stopped.trace[-1].n_evaluated == 8


def test_the_ceiling_stops_a_search_that_cannot_spend_its_budget(bars):
    # A budget wider than the field can never be spent, so the ceiling is what
    # ends the run.
    run = search.run_search(bars, **search_kwargs(seed=0, generations=3,
                                                  budget=SPACE.size + 10))
    assert run.generations_run == 3
    assert run.unique_evals <= SPACE.size


def test_a_search_with_nothing_to_spend_is_refused(bars):
    # A zero budget or population would leave the final front empty, and an
    # empty front reported as a result is worse than a refusal.
    with pytest.raises(ValueError):
        search.run_search(bars, **search_kwargs(budget=0))
    with pytest.raises(ValueError):
        search.run_search(bars, **search_kwargs(population=0))


def test_the_search_never_evaluates_an_infeasible_genome(bars):
    # A slow period the frame cannot warm up would score a silent 0.0 and win
    # folds on it, so the clamp is applied to whatever space a caller passes.
    greedy = search.GeneSpace(fast=(5, 10), slow=(20, 30, 200), trail=(2.0,))
    run = search.run_search(bars, **search_kwargs(seed=0, space=greedy))
    bound = optimize.feasible_slow_max(len(bars), N_FOLDS)
    assert run.space.slow == (20, 30)
    for cell in run.evaluated.cells:
        slow = cell.params["slow_period"]
        assert slow <= bound
        assert TrendFollowing(1, slow, 2.0, 14, False).min_bars <= len(bars)


# --- B4: the measurement ----------------------------------------------------


def test_pareto_front_of_the_exhaustive_field(field):
    front = search.pareto_front(field)
    cells = {c.label: c for c in field.cells}
    assert front and set(front) <= set(cells)
    feasible = [c for c in field.cells if c.sharpe > search.SHARPE_FLOOR]
    assert feasible
    for label in front:
        best = cells[label]
        assert best.sharpe > search.SHARPE_FLOOR
        assert not any(o.dsr >= best.dsr and o.max_dd <= best.max_dd
                       and (o.dsr > best.dsr or o.max_dd < best.max_dd)
                       for o in feasible)


def test_comparison_measures_front_coverage_against_the_ground_truth(comparison,
                                                                     result, field):
    assert comparison.field_front == search.pareto_front(field)
    assert comparison.search_front == result.front
    assert set(comparison.matched) == set(comparison.search_front) & \
        set(comparison.field_front)
    assert comparison.coverage == pytest.approx(
        len(comparison.matched) / len(comparison.field_front))
    assert comparison.precision == pytest.approx(
        len(comparison.matched) / len(comparison.search_front))
    assert 0.0 <= comparison.coverage <= 1.0
    assert 0.0 <= comparison.precision <= 1.0


def test_comparison_reports_the_dsr_regret_of_the_search_pick(comparison, result,
                                                              field):
    assert comparison.field_best == field.dsr_best.label
    assert comparison.field_best_dsr == field.dsr_best.dsr
    assert comparison.search_pick == result.evaluated.dsr_best.label
    # The same cell, deflated twice: once at the search's own trial count and
    # once at the whole field's. Regret is measured in the field's currency.
    assert comparison.search_pick_dsr_own == result.evaluated.dsr_best.dsr
    assert comparison.search_pick_dsr_field == \
        field.cell(comparison.search_pick).dsr
    assert comparison.dsr_regret == pytest.approx(
        comparison.field_best_dsr - comparison.search_pick_dsr_field)
    assert comparison.dsr_regret >= -1e-12
    assert comparison.evals == len(result.evaluated.cells)
    assert comparison.field_size == len(field.cells)
    assert comparison.evals < comparison.field_size


def test_every_front_cell_is_deflated_twice(comparison, result, field):
    # The B4 headline, per cell rather than only for the pick: the same backtest
    # scored against the trials the search paid for and against the whole field.
    own = {c.label: c.dsr for c in result.evaluated.cells}
    assert [row.label for row in comparison.front_cells] == result.front
    for row in comparison.front_cells:
        assert row.dsr_own == own[row.label]
        assert row.dsr_field == field.cell(row.label).dsr
        assert row.on_field_front == (row.label in comparison.field_front)
        assert row.deflation_cost == pytest.approx(row.dsr_own - row.dsr_field)


def test_comparison_reports_the_front_haircut_and_both_pbos(comparison, result,
                                                            field):
    cells = {c.label: c for c in result.evaluated.cells}
    expected = np.mean([cells[label].sharpe - cells[label].oos_sharpe
                        for label in result.front])
    assert comparison.front_haircut == pytest.approx(expected)
    assert comparison.search_pbo is result.evaluated.pbo or \
        comparison.search_pbo == result.evaluated.pbo
    assert comparison.field_pbo is field.pbo or comparison.field_pbo == field.pbo
    assert comparison.search_p_oos_loss == result.evaluated.p_oos_loss or \
        math.isnan(comparison.search_p_oos_loss)
    assert comparison.field_p_oos_loss == field.p_oos_loss or \
        math.isnan(comparison.field_p_oos_loss)


def fake_field(with_live=True) -> optimize.GridResult:
    """An exhaustive field with no backtests behind it: the measurement layer
    only reads the numbers."""
    def cell(label, fast, slow, trail, sharpe, dsr, oos, dd, passes=False):
        return optimize.CellResult(
            label, {"fast_period": fast, "slow_period": slow,
                    "trail_atr_mult": trail}, sharpe, dsr, oos, passes, dd)

    cells = [cell("T 10/40/2.0", 10, 40, 2.0, 1.10, 0.99, 0.90, 0.20, True),
             cell("T 30/80/2.5", 30, 80, 2.5, 0.70, 0.80, 0.30, 0.08),
             cell("T 50/100/4.0", 50, 100, 4.0, 0.40, 0.55, -0.10, 0.31)]
    if with_live:
        cells.append(cell("T 20/100/3.0", 20, 100, 3.0, 0.62, 0.74, 0.44, 0.15))
    return field_of(cells)


def field_of(cells) -> optimize.GridResult:
    return optimize.GridResult(
        "USO", cells,
        CscvReport(pbo=0.44, p_oos_loss=0.31, median_logit=0.6,
                   degradation_slope=-0.9, logits=(0.2, 0.4),
                   oos_rank_mean={c.label: 0.5 for c in cells},
                   is_best_share={c.label: 1.0 / len(cells) for c in cells}),
        0.55, len(cells), N_BLOCKS, N_FOLDS, spec=optimize.TREND_FIELD,
        field_correlation=optimize.FieldCorrelation(0.88, 0.62))


def research_field(name: str) -> tuple[optimize.GridResult, dict]:
    """A recorded field rebuilt from its artifact, so a change to the front rule
    is measured against the front that shipped."""
    path = Path(__file__).resolve().parents[1] / "docs/research" / name
    report = json.loads(path.read_text())
    return _grid_from(report["exhaustive"]["cells"]), report


def _grid_from(cell_entries) -> optimize.GridResult:
    return field_of([optimize.CellResult(
        c["label"], c["params"], _number(c["sharpe"]), _number(c["dsr"]),
        _number(c["oos_sharpe"]), c["passes"], _number(c["max_dd"]),
        c.get("traded", True) is not False)
        for c in cell_entries])


def _number(x) -> float:
    return float("nan") if x is None else float(x)


def plain_cell(label, sharpe, dsr, dd) -> optimize.CellResult:
    return optimize.CellResult(label, {}, sharpe, dsr, float("nan"), False, dd)


def test_live_cell_standing_is_read_out_of_the_exhaustive_field():
    live = search.live_standing(fake_field())
    assert live.present is True
    assert live.label == "T 20/100/3.0"
    assert live.field_size == 4
    assert live.dsr == 0.74
    assert live.dsr_rank == 3           # behind 0.99 and 0.80
    assert live.oos_rank == 2           # behind 0.90, ahead of 0.30 and -0.10
    assert live.max_dd == 0.15
    assert live.passes is False
    # T 30/80/2.5 beats it on both objectives (DSR 0.80, drawdown 8%), so the
    # live cell is dominated and off the front.
    assert live.on_front is False
    front = search.pareto_front(fake_field())
    assert "T 30/80/2.5" in front
    cells = {c.label: c for c in fake_field().cells}
    assert all(cells[label].sharpe > search.SHARPE_FLOOR for label in front)


def test_a_losing_cell_with_the_smallest_drawdown_is_not_on_the_front():
    front = search.pareto_front(field_of([
        plain_cell("T 10/40/2.0", -0.20, 0.90, 0.01),
        plain_cell("T 30/80/2.5", 0.70, 0.40, 0.22)]))
    assert front == ["T 30/80/2.5"]


def test_a_field_of_only_infeasible_cells_fronts_its_least_losing_cells():
    # Nothing clears the floor, so the front is the cells that miss it by the
    # least rather than the whole field.
    labels = ["T 10/40/2.0", "T 30/80/2.5", "T 50/100/4.0"]
    front = search.pareto_front(field_of(
        [plain_cell(labels[0], -0.20, 0.90, 0.01),
         plain_cell(labels[1], 0.0, 0.40, 0.22),
         plain_cell(labels[2], float("nan"), 0.10, 0.30)]))
    assert front == [labels[1]]


def test_the_09_07_field_front_is_unchanged_under_the_floor():
    field, report = research_field("trend-search-2026-09-07.json")
    assert set(search.pareto_front(field)) == set(report["comparison"]["field_front"])


def test_the_09_14_field_front_loses_exactly_its_non_positive_sharpe_cells():
    field, report = research_field("trend-search-2026-09-14.json")
    recorded = set(report["comparison"]["field_front"])
    under = {c.label for c in field.cells if not c.sharpe > search.SHARPE_FLOOR}
    assert under                      # the field that motivated the floor
    assert set(search.pareto_front(field)) == recorded - under
    assert "T 15/80/4.0/1.2" in search.pareto_front(field)  # 0.025 clears it


def test_the_09_14_search_front_loses_exactly_its_non_positive_sharpe_cells():
    _, report = research_field("trend-search-2026-09-14.json")
    entries = report["search"]["evaluated"]["cells"]
    evaluated = _grid_from(entries)
    counted = sum(1 for c in entries if not c["sharpe"] > search.SHARPE_FLOOR)
    assert search._below_floor(evaluated.cells) == counted
    assert counted == 61            # a drift in the record should be visible
    under = {c.label for c in evaluated.cells
             if not search.feasible(c.dsr, c.max_dd, c.sharpe)}
    assert set(search.pareto_front(evaluated)) == \
        set(report["comparison"]["search_front"]) - under


def test_live_cell_absent_from_a_field_that_excludes_it():
    live = search.live_standing(fake_field(with_live=False))
    assert live.present is False
    assert live.oos_rank is None and live.dsr_rank is None
    assert math.isnan(live.dsr)


# --- B5: the artifact -------------------------------------------------------


@pytest.fixture(scope="module")
def report(result, field, comparison) -> dict:
    at = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)
    return search.build_search_report(field, result, comparison, generated_at=at)


def test_report_records_both_runs_and_the_run_context(report, field, result):
    assert report["generated_at"] == "2026-09-05T12:00:00+00:00"
    assert report["symbol"] == optimize.TREND_SYMBOL
    assert report["market_data_feed"] == MARKET_DATA_FEED
    assert report["n_blocks"] == N_BLOCKS and report["n_folds"] == N_FOLDS
    assert len(report["exhaustive"]["cells"]) == len(field.cells)
    assert report["exhaustive"]["summary"]["pbo"] == field.pbo
    assert report["exhaustive"]["summary"]["p_oos_loss"] == field.p_oos_loss
    assert set(report["exhaustive"]["summary"]["field_correlation"]) == \
        {"mean", "min"}
    assert len(report["search"]["evaluated"]["cells"]) == len(result.evaluated.cells)


def test_report_records_the_seed_and_the_whole_search_trace(report, result):
    block = report["search"]
    assert block["seed"] == 0
    assert block["budget"] == BUDGET
    assert block["population"] == POPULATION
    assert block["generations"] == GENERATIONS
    assert block["unique_evals"] == len(result.evaluated.cells)
    assert block["front"] == result.front
    assert block["gene_space"] == {"fast": list(SPACE.fast),
                                   "slow": list(SPACE.slow),
                                   "trail": list(SPACE.trail),
                                   "volume": list(SPACE.volume)}
    assert [g["generation"] for g in block["trace"]] == [1, 2]
    assert [g["n_trials"] for g in block["trace"]] == \
        [t.n_trials for t in result.trace]


def test_report_carries_the_measurement(report, comparison):
    block = report["comparison"]
    assert block["search_front"] == comparison.search_front
    assert block["field_front"] == comparison.field_front
    assert block["coverage"] == pytest.approx(comparison.coverage)
    assert block["dsr_regret"] == pytest.approx(comparison.dsr_regret)
    assert block["search_pick"]["dsr_own"] == comparison.search_pick_dsr_own
    assert block["search_pick"]["dsr_field"] == comparison.search_pick_dsr_field
    assert [row["label"] for row in block["front_cells"]] == \
        [row.label for row in comparison.front_cells]
    assert block["front_cells"][0]["dsr_field"] == \
        comparison.front_cells[0].dsr_field
    assert block["front_haircut"] == pytest.approx(comparison.front_haircut)
    assert block["search_pbo"] == comparison.search_pbo
    assert block["field_pbo"] == comparison.field_pbo
    assert block["search_p_oos_loss"] == comparison.search_p_oos_loss
    assert block["field_p_oos_loss"] == comparison.field_p_oos_loss
    assert block["live_cell"]["label"] == optimize.TREND_FIELD.headline_label
    assert block["field_below_floor"] == comparison.field_below_floor
    assert block["search_below_floor"] == comparison.search_below_floor


def test_the_record_states_the_floor_the_fronts_were_built_under(report,
                                                                comparison,
                                                                field, result):
    assert report["search"]["sharpe_floor"] == search.SHARPE_FLOOR
    assert comparison.field_below_floor == sum(
        1 for c in field.cells if not c.sharpe > search.SHARPE_FLOOR)
    assert comparison.search_below_floor == sum(
        1 for c in result.evaluated.cells if not c.sharpe > search.SHARPE_FLOOR)


def test_report_is_strict_json(report):
    assert json.loads(json.dumps(report, allow_nan=False))["symbol"] == "USO"


def test_the_search_replays_from_the_artifact_alone(bars, report, result):
    # The reproducibility claim, enforced rather than asserted in prose: every
    # knob the run turned is recorded, and replaying them reproduces the front.
    assert report["search"]["generations_run"] == result.generations_run
    kwargs = search.search_kwargs_from_report(report)
    assert "generations_run" not in kwargs  # a reading, not a knob
    assert "sharpe_floor" not in kwargs     # the rule it was scored under
    replay = search.run_search(bars, **kwargs)
    assert replay.front == result.front
    assert [c.label for c in replay.evaluated.cells] == \
        [c.label for c in result.evaluated.cells]


def test_markdown_leads_with_the_decision(result, field, comparison):
    at = datetime(2026, 9, 5, tzinfo=timezone.utc)
    text = search.build_search_markdown(field, result, comparison, at)
    assert text.startswith("# USO ")
    assert "seed 0" in text
    assert f"{len(result.evaluated.cells)} unique" in text
    assert f"{len(field.cells)} cells" in text
    assert optimize.TREND_FIELD.headline_label in text
    assert MARKET_DATA_FEED in text
    assert "docs/validation/" in text  # this record promotes nothing
    # The front, with both deflations side by side: the reader should not have
    # to open the JSON to see what the honest trial count costs.
    assert (f"| DSR @ {len(result.evaluated.cells)} trials "
            f"| DSR @ {len(field.cells)} trials |") in text
    for row in comparison.front_cells:
        assert f"| {row.label} |" in text


def test_markdown_says_both_fronts_are_built_over_the_same_floor(result, field,
                                                                comparison):
    at = datetime(2026, 9, 5, tzinfo=timezone.utc)
    text = search.build_search_markdown(field, result, comparison, at)
    assert (f"Both fronts are built over cells with Sharpe > 0; "
            f"{comparison.field_below_floor} field cells and "
            f"{comparison.search_below_floor} evaluated cells fall under the "
            f"floor.") in text


def test_markdown_does_not_claim_a_front_over_feasible_cells_when_there_are_none(
        result, field, comparison):
    at = datetime(2026, 9, 5, tzinfo=timezone.utc)
    starved = dataclasses.replace(comparison,
                                  field_below_floor=comparison.field_size,
                                  search_below_floor=comparison.evals)
    text = search.build_search_markdown(field, result, starved, at)
    assert "Both fronts are built over" not in text
    assert "No cell clears Sharpe > 0" in text
    assert "least-losing" in text
    assert (f"{comparison.field_size} field cells and {comparison.evals} "
            f"evaluated cells fall under the floor.") in text


def test_markdown_only_claims_a_deflation_gap_that_is_there(result, field,
                                                            comparison):
    at = datetime(2026, 9, 5, tzinfo=timezone.utc)
    text = search.build_search_markdown(field, result, comparison, at)
    gap = comparison.search_pick_dsr_own - comparison.search_pick_dsr_field
    if gap > 0.0005:
        assert "cheap search reports a bigger number" in text
    else:
        assert "cheap search reports a bigger number" not in text


def test_artifact_lands_under_docs_research_with_the_trend_prefix(tmp_path, report,
                                                                  result, field,
                                                                  comparison):
    at = datetime(2026, 9, 5, tzinfo=timezone.utc)
    directory = str(tmp_path / "research")
    json_path = search.write_search_report(report, at, directory=directory)
    md_path = search.write_search_markdown(field, result, comparison, at,
                                           directory=directory)
    assert os.path.basename(json_path) == "trend-search-2026-09-05.json"
    assert os.path.basename(md_path) == "trend-search-2026-09-05.md"
    with open(json_path) as fh:
        assert json.load(fh)["search"]["seed"] == 0
    assert search.RESEARCH_REPORT_DIR == optimize.RESEARCH_REPORT_DIR
    assert "validation" not in search.RESEARCH_REPORT_DIR


def test_the_record_paths_are_what_the_writers_write(tmp_path, report, result,
                                                     field, comparison):
    at = datetime(2026, 9, 5, tzinfo=timezone.utc)
    json_path, md_path = search.search_record_paths(at, str(tmp_path))
    assert str(json_path) == search.write_search_report(report, at,
                                                        directory=str(tmp_path))
    assert str(md_path) == search.write_search_markdown(
        field, result, comparison, at, directory=str(tmp_path))


def test_a_writer_refuses_to_overwrite_an_existing_record(tmp_path, report,
                                                          result, field,
                                                          comparison):
    # Two runs on one UTC day share a filename, and the second one silently
    # taking the first one's place loses an experiment.
    at = datetime(2026, 9, 5, tzinfo=timezone.utc)
    directory = str(tmp_path)
    search.write_search_report(report, at, directory=directory)
    search.write_search_markdown(field, result, comparison, at,
                                 directory=directory)
    with pytest.raises(FileExistsError):
        search.write_search_report(report, at, directory=directory)
    with pytest.raises(FileExistsError):
        search.write_search_markdown(field, result, comparison, at,
                                     directory=directory)


def test_print_search_shows_the_front_the_seed_and_the_live_cell(capsys, result,
                                                                 field,
                                                                 comparison):
    search.print_search(field, result, comparison)
    out = capsys.readouterr().out
    assert "seed 0" in out
    for label in result.front:
        assert label in out
    assert f"{len(result.evaluated.cells)}" in out
    assert "coverage" in out.lower()
    assert search._floor_sentence(comparison) in out


@pytest.mark.parametrize("which", ["result", "budget_stopped"])
def test_the_prose_reports_which_stop_ended_the_run(capsys, field, request,
                                                    which):
    run = request.getfixturevalue(which)
    comparison = search.compare(run, field)
    at = datetime(2026, 9, 5, tzinfo=timezone.utc)
    text = search.build_search_markdown(field, run, comparison, at)
    search.print_search(field, run, comparison)
    out = capsys.readouterr().out
    for rendered in (text, out):
        if run.unique_evals >= run.budget:
            assert f"budget {run.budget} spent in " \
                   f"{run.generations_run} generations" in rendered
            assert f"ceiling {run.generations}" in rendered
        else:
            assert f"{run.generations}-generation ceiling" in rendered
            assert f"{run.unique_evals} of the {run.budget} budget" in rendered


def test_markdown_names_p_oos_loss_as_the_binding_criterion(result, field,
                                                            comparison):
    text = search.build_search_markdown(field, result, comparison,
                                        datetime(2026, 9, 5, tzinfo=timezone.utc))
    assert "P(OOS loss)" in text
    assert "advisory" in text.lower()


def test_search_promotes_nothing():
    assert not hasattr(search, "write_validation_report")
    assert not hasattr(search, "ValidationArtifactDegraded")


# --- CLI --------------------------------------------------------------------


def test_default_invocation_still_runs_the_tom_grid(monkeypatch):
    calls = []
    monkeypatch.setattr(optimize, "run_tom_experiment",
                        lambda: calls.append("tom") or 0)
    monkeypatch.setattr(search, "main", lambda argv=(): calls.append("trend") or 0)
    assert optimize.main([]) == 0
    assert calls == ["tom"]


def test_the_trend_subcommand_runs_the_search(monkeypatch):
    calls = []
    monkeypatch.setattr(optimize, "run_tom_experiment",
                        lambda: calls.append("tom") or 0)
    monkeypatch.setattr(search, "main", lambda argv=(): calls.append("trend") or 0)
    assert optimize.main(["trend"]) == 0
    assert calls == ["trend"]


def test_an_unknown_subcommand_is_refused(monkeypatch, capsys):
    monkeypatch.setattr(optimize, "run_tom_experiment", lambda: 0)
    assert optimize.main(["grid"]) == 2
    assert "trend" in capsys.readouterr().err


def test_the_search_cli_stops_without_market_data(monkeypatch, capsys):
    monkeypatch.setattr(search, "load_env", lambda: None)
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    assert search.main([]) == 1
    assert "Alpaca" in capsys.readouterr().err


def test_the_search_cli_stops_when_the_fetch_fails(monkeypatch, tmp_path,
                                                   capsys):
    monkeypatch.setattr(search, "load_env", lambda: None)
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setattr(search, "RESEARCH_REPORT_DIR", str(tmp_path))

    class Broken:
        def __init__(self, *a, **k):
            pass

        def get_bars(self, *a, **k):
            raise RuntimeError("no entitlement")

    monkeypatch.setattr(search, "MarketData", Broken)
    assert search.main([]) == 1
    err = capsys.readouterr().err
    assert "no entitlement" in err  # never a fabricated field


def test_the_search_cli_refuses_before_it_fetches_anything(monkeypatch,
                                                          tmp_path, capsys):
    monkeypatch.setattr(search, "load_env", lambda: None)
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setattr(search, "RESEARCH_REPORT_DIR", str(tmp_path))

    class MustNotFetch:
        def __init__(self, *a, **k):
            raise AssertionError("must not fetch")

    monkeypatch.setattr(search, "MarketData", MustNotFetch)
    existing = search.search_record_paths(datetime.now(timezone.utc),
                                          str(tmp_path))[1]
    existing.write_text("# an experiment already recorded today")
    assert search.main([]) == 1
    assert str(existing) in capsys.readouterr().err


def test_the_defaults_are_the_declared_budget():
    assert search.DEFAULT_SEED == 0
    assert search.DEFAULT_POPULATION == 36
    assert search.DEFAULT_GENERATIONS == 40
    assert search.DEFAULT_BUDGET == 180
    assert 0.0 < search.DEFAULT_MUTATION_RATE < 1.0
    assert search.SEARCH_FOLDS == VALIDATION.wf_folds


def test_the_budget_holds_its_share_of_the_field():
    # "Did the search find the front" only means the same thing run to run if the
    # budget stays the same FRACTION of the field. Widening the field without
    # rescaling the budget silently swaps the question for a harder one and makes
    # the new numbers look worse than the recorded ones for no real reason. The
    # ratio only says something because the loop can actually reach the budget;
    # test_the_ceiling_is_not_the_normal_stop is what keeps that true.
    coverage = search.DEFAULT_BUDGET / search.GeneSpace().size
    assert 0.44 <= coverage <= 0.48


def test_the_ceiling_is_not_the_normal_stop():
    # The ceiling is there for a population that has converged and yields no new
    # genomes. If it is the usual stop, the budget is a number the search never
    # spends and every claim about the share of the field it covers is false.
    capacity = search.DEFAULT_POPULATION * (search.DEFAULT_GENERATIONS + 1)
    assert capacity >= 4 * search.DEFAULT_BUDGET
