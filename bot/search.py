"""Seeded multi-objective parameter search, measured against exhaustive truth.

bot.optimize enumerates a whole field and pays the honest deflation. This asks
the next question: if a search spends a fraction of that budget, does it find
the same answer, and what does it cost to believe it did?

The experiment runs two things over the same USO bars and compares them:

- the exhaustive EMA-crossover trend field (the ground truth), and
- an NSGA-II-lite search over the same field, seeded and budget-capped,
  maximizing Deflated Sharpe while minimizing max drawdown.

Three disciplines make the comparison readable rather than flattering:

- The search's DSR is recomputed every generation at n_trials = the cumulative
  count of unique genomes evaluated so far, so the deflation TIGHTENS as the
  search spends trials. A search cannot buy a better number by looking harder.
- Walk-forward runs once, at the end, over the final Pareto front. Selecting on
  an out-of-sample number would be leakage, and paying for it per genome would
  cost more than the search saves.
- The genome's values are the exhaustive field's own value sets. A search able
  to land between grid points would beat the grid by construction and "regret
  against the exhaustive best" would stop meaning anything.

Output is an experiment record under docs/research/, deliberately not the
docs/validation/ promotion artifact: nothing here promotes anything. Replaying
a record reuses its search kwargs but ranks under the current SHARPE_FLOOR,
which is why the floor is recorded and not replayed.

Usage: MARKET_DATA_FEED=sip op run --env-file .env -- \
       uv run python -m bot.optimize trend
"""

import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bot import optimize
from bot.data import MarketData
from bot.metrics import sharpe_ratio
from bot.optimize import (
    RESEARCH_REPORT_DIR,
    TREND_FIELD,
    TREND_SLIPPAGE_PCT,
    TREND_SYMBOL,
    CellResult,
    GridResult,
    build_grid_report,
    drawdown_of,
    feasible_slow_max,
    json_number,
    refuse_existing_record,
    traded_in,
    trend_candidates,
    trend_cell_label,
)
from bot.research import DAILY, EQUITY_START, strategy_params
from bot.validation import (
    Candidate,
    build_return_matrix,
    cscv_report,
    deflated_sharpes_from_matrix,
    walk_forward,
)
from config import MARKET_DATA_FEED, VALIDATION, AssetClass

log = logging.getLogger(__name__)

DEFAULT_SEED = 0
DEFAULT_POPULATION = 36
DEFAULT_GENERATIONS = 40
DEFAULT_BUDGET = 180  # unique backtests; ~46% of the 390-cell field
# The budget is what the search spends, and what holds it at a fixed SHARE
# (~46%) of the exhaustive field it is scored against — widen the field and the
# budget has to be rescaled with it. `generations` is only a ceiling: it binds
# when the population has converged and offspring stop being new genomes, which
# is the one way a run ends with the budget unspent.
DEFAULT_MUTATION_RATE = 0.25
# Minimizing drawdown alone makes "barely trade" Pareto-optimal: a cell that
# lost money must not dominate one that made money on its drawdown.
SHARPE_FLOOR = 0.0
# Finite and above any reachable objective (drawdown <= 1, -DSR >= -1), so every
# feasible cell dominates every infeasible one while a less-losing cell still
# dominates a more-losing one — a gradient out of a losing region.
INFEASIBLE_BASE = 1e6
SEARCH_FOLDS = VALIDATION.wf_folds


@dataclass(frozen=True)
class Genome:
    """Deliberately not order=True: the volume gene's off state is None, which
    has no ordering against the floats beside it, so a generated total order
    would raise exactly when two genomes differ only in that gene. Sort on
    `sort_key` instead."""
    fast: int
    slow: int
    trail: float
    volume: float | None = None

    @property
    def sort_key(self) -> tuple:
        return (self.fast, self.slow, self.trail,
                self.volume is not None, self.volume or 0.0)


@dataclass(frozen=True)
class GeneSpace:
    """The values each gene may take.

    Value sets rather than continuous ranges, and by default the exhaustive
    field's own: the search is scored against that field, and a genome landing
    between its points would make the comparison meaningless. Periods stay
    integers and the trail stays on its 0.5 grid by construction."""
    fast: tuple[int, ...] = optimize.TREND_FAST
    slow: tuple[int, ...] = optimize.TREND_SLOW
    trail: tuple[float, ...] = optimize.TREND_TRAIL
    volume: tuple[float | None, ...] = optimize.TREND_VOLUME_MULT

    def __post_init__(self) -> None:
        if not (self.fast and self.slow and self.trail and self.volume):
            raise ValueError("every gene needs at least one allowed value")
        if not self.feasible_pairs:
            raise ValueError(f"no fast < slow pair in fast={self.fast} "
                             f"slow={self.slow}: nothing feasible to search")

    def limited(self, max_slow: int) -> "GeneSpace":
        """The space with slow periods that cannot warm up removed."""
        return replace(self, slow=tuple(s for s in self.slow if s <= max_slow))

    @property
    def feasible_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple((f, s) for s in self.slow for f in self.fast if f < s)

    @property
    def size(self) -> int:
        """Unique feasible genomes: the field the search samples from."""
        return len(self.feasible_pairs) * len(self.trail) * len(self.volume)


def _pick(rng: np.random.Generator, values: Sequence) -> object:
    return values[int(rng.integers(len(values)))]


def _snap(value: float, allowed: Sequence):
    """Nearest allowed value, ties to the smaller."""
    return min(allowed, key=lambda v: (abs(v - value), v))


def _snap_optional(value: float | None, allowed: Sequence):
    """Nearest allowed value on an axis whose off state is None. None is its own
    category rather than a number, so it snaps to itself when allowed and to the
    axis head otherwise."""
    if value in allowed:
        return value
    numeric = [v for v in allowed if v is not None]
    if value is None or not numeric:
        return allowed[0]
    return min(numeric, key=lambda v: (abs(v - value), v))


def repair(genome: Genome, space: GeneSpace,
           rng: np.random.Generator) -> Genome:
    """The genome as the space's nearest legal member: every gene snapped to an
    allowed value and fast resampled below slow when the pair came out
    inverted. Crossover and mutation are free to produce nonsense; nothing
    downstream ever sees it."""
    fast = _snap(genome.fast, space.fast)
    slow = _snap(genome.slow, space.slow)
    trail = _snap(genome.trail, space.trail)
    volume = _snap_optional(genome.volume, space.volume)
    if fast >= slow:
        below = [f for f in space.fast if f < slow]
        if not below:
            slow = min(s for s in space.slow if s > min(space.fast))
            below = [f for f in space.fast if f < slow]
        fast = _pick(rng, below)
    return Genome(int(fast), int(slow), float(trail), volume)


def random_genome(rng: np.random.Generator, space: GeneSpace) -> Genome:
    return repair(Genome(_pick(rng, space.fast), _pick(rng, space.slow),
                         _pick(rng, space.trail), _pick(rng, space.volume)),
                  space, rng)


def initial_population(rng: np.random.Generator, space: GeneSpace,
                       size: int) -> list[Genome]:
    """`size` distinct genomes, or the whole space when it is smaller. Every
    duplicate in the seed population would cost a backtest and buy nothing, so
    they are resampled rather than deduped away afterwards."""
    wanted = min(size, space.size)
    pop: dict[Genome, None] = {}
    attempts = 0
    while len(pop) < wanted and attempts < 100 * wanted:
        pop[random_genome(rng, space)] = None
        attempts += 1
    return list(pop)


def crossover(a: Genome, b: Genome, rng: np.random.Generator) -> Genome:
    take = rng.random(4) < 0.5
    return Genome(a.fast if take[0] else b.fast,
                  a.slow if take[1] else b.slow,
                  a.trail if take[2] else b.trail,
                  a.volume if take[3] else b.volume)


def mutate(genome: Genome, rng: np.random.Generator, space: GeneSpace,
           rate: float) -> Genome:
    draws = rng.random(4)
    return Genome(
        _pick(rng, space.fast) if draws[0] < rate else genome.fast,
        _pick(rng, space.slow) if draws[1] < rate else genome.slow,
        _pick(rng, space.trail) if draws[2] < rate else genome.trail,
        _pick(rng, space.volume) if draws[3] < rate else genome.volume,
    )


def non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """Pareto fronts, best first, over objectives that are all minimized.

    A point with a non-finite objective dominates nothing and is dominated by
    everything finite, so an unscorable genome sinks to the last front instead
    of riding along as "non-dominated"."""
    objs = np.asarray(objectives, dtype=float)
    n = len(objs)
    if n == 0:
        return []
    weakly = (objs[:, None, :] <= objs[None, :, :]).all(axis=2)
    strictly = (objs[:, None, :] < objs[None, :, :]).any(axis=2)
    dominates = weakly & strictly
    dominated_by = dominates.sum(axis=0)

    fronts: list[list[int]] = []
    remaining = np.ones(n, dtype=bool)
    while remaining.any():
        current = np.flatnonzero(remaining & (dominated_by == 0))
        if current.size == 0:  # only reachable when every point is unscorable
            current = np.flatnonzero(remaining)
        fronts.append([int(i) for i in current])
        remaining[current] = False
        dominated_by = dominated_by - dominates[current].sum(axis=0)
    return fronts


def crowding_distance(objectives: np.ndarray) -> np.ndarray:
    """Per-point room within one front: the normalized objective span between
    its neighbours. Boundary points are infinite so the extremes survive
    truncation and the front keeps its width; a front with no finite width
    shares out no distance, leaving the boundaries infinite and the interior
    zero."""
    objs = np.asarray(objectives, dtype=float)
    n = len(objs)
    if n <= 2:
        return np.full(n, np.inf)
    dist = np.zeros(n)
    for k in range(objs.shape[1]):
        order = np.argsort(objs[:, k], kind="stable")
        values = objs[order, k]
        dist[order[0]] = dist[order[-1]] = np.inf
        if not np.isfinite(values[[0, -1]]).all():
            continue  # a front of infeasible points has no width to share out
        span = values[-1] - values[0]
        if span > 0:
            dist[order[1:-1]] += (values[2:] - values[:-2]) / span
    return dist


def tournament_winner(i: int, j: int, rank: np.ndarray,
                      crowd: np.ndarray) -> int:
    """Lower front wins; within a front, the one with more room around it."""
    if rank[i] != rank[j]:
        return i if rank[i] < rank[j] else j
    if crowd[i] != crowd[j]:
        return i if crowd[i] > crowd[j] else j
    return i


def binary_tournament(rng: np.random.Generator, rank: np.ndarray,
                      crowd: np.ndarray, n: int) -> np.ndarray:
    picks = rng.integers(0, len(rank), size=(n, 2))
    return np.array([tournament_winner(int(a), int(b), rank, crowd)
                     for a, b in picks], dtype=int)


@dataclass(frozen=True)
class GenerationTrace:
    generation: int
    n_evaluated: int  # cumulative unique genomes, and so the DSR trial count
    new_evals: int
    front_size: int
    n_trials: int
    best_dsr: float


@dataclass(frozen=True)
class SearchResult:
    evaluated: GridResult  # every genome the search paid for, scored as a field
    front: list[str]       # Pareto front over all evaluated genomes
    seed: int
    population: int
    generations: int
    budget: int
    mutation_rate: float
    space: GeneSpace
    trace: list[GenerationTrace]

    @property
    def unique_evals(self) -> int:
        return len(self.evaluated.cells)

    @property
    def generations_run(self) -> int:
        return len(self.trace)


def genome_label(genome: Genome) -> str:
    return trend_cell_label(genome.fast, genome.slow, genome.trail, genome.volume)


def candidate_for(genome: Genome, symbol: str, asset_class: AssetClass,
                  slippage: float) -> Candidate:
    """One genome as a validation Candidate, through the same constructor the
    exhaustive field uses, so a cell means the same thing in both runs."""
    built = trend_candidates(symbol, asset_class, slippage, (genome.fast,),
                             (genome.slow,), (genome.trail,), (genome.volume,))
    if not built:
        raise ValueError(f"{genome} is not a crossover: fast >= slow")
    return built[0]


def _matrix(returns: dict[Genome, pd.Series]) -> pd.DataFrame:
    """The evaluated population as a return matrix, assembled the way
    build_return_matrix assembles one."""
    return pd.DataFrame({genome_label(g): s for g, s in returns.items()}
                        ).sort_index().fillna(0.0)


def _deflated(returns: dict[Genome, pd.Series]) -> dict[str, float]:
    """DSR per evaluated genome at n_trials = the count of unique genomes
    evaluated so far, with the benchmark's dispersion taken across all of
    them."""
    matrix = _matrix(returns)
    return deflated_sharpes_from_matrix(matrix, len(matrix.columns))


def feasible(dsr: float, max_dd: float, sharpe: float) -> bool:
    return sharpe > SHARPE_FLOOR and dsr == dsr and max_dd == max_dd


def objective_row(dsr: float, max_dd: float, sharpe: float) -> list[float]:
    """The minimize-both row: -DSR and max drawdown. A cell under the Sharpe
    floor, or missing a number, is ranked instead by how far it misses the
    floor, so it sinks below every feasible cell without losing its ordering. A
    cell infeasible only for a missing number misses the floor by nothing and
    sits on the base itself."""
    if feasible(dsr, max_dd, sharpe):
        return [-dsr, max_dd]
    violation = max(SHARPE_FLOOR - sharpe, 0.0) if sharpe == sharpe else np.inf
    return [INFEASIBLE_BASE + violation, INFEASIBLE_BASE + violation]


def _objectives(genomes: Sequence[Genome], dsr: dict[str, float],
                returns: dict[Genome, pd.Series]) -> np.ndarray:
    rows = []
    for genome in genomes:
        label = genome_label(genome)
        series = returns.get(genome)
        rows.append(objective_row(
            dsr.get(label, float("nan")),
            drawdown_of(series) if series is not None else float("nan"),
            sharpe_ratio(series) if series is not None else float("nan")))
    return np.array(rows, dtype=float).reshape(len(genomes), 2)


def _rank_and_crowding(objs: np.ndarray,
                       fronts: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    rank = np.zeros(len(objs), dtype=int)
    crowd = np.zeros(len(objs))
    for index, front in enumerate(fronts):
        rank[front] = index
        crowd[front] = crowding_distance(objs[front])
    return rank, crowd


def _truncate(genomes: Sequence[Genome], objs: np.ndarray,
              keep: int) -> list[Genome]:
    """Front-by-front survivor selection, the overflowing front cut by
    crowding distance so the survivors stay spread out."""
    chosen: list[int] = []
    for front in non_dominated_sort(objs):
        if len(chosen) + len(front) <= keep:
            chosen.extend(front)
            continue
        room = keep - len(chosen)
        order = np.argsort(-crowding_distance(objs[front]), kind="stable")
        chosen.extend(front[int(i)] for i in order[:room])
        break
    return [genomes[i] for i in chosen]


def _dedupe(genomes: Sequence[Genome]) -> list[Genome]:
    return list(dict.fromkeys(genomes))


def run_search(bars: pd.DataFrame, symbol: str = TREND_SYMBOL,
               asset_class: AssetClass = AssetClass.US_EQUITY,
               slippage: float = TREND_SLIPPAGE_PCT,
               space: GeneSpace | None = None,
               seed: int = DEFAULT_SEED,
               population: int = DEFAULT_POPULATION,
               generations: int = DEFAULT_GENERATIONS,
               budget: int = DEFAULT_BUDGET,
               mutation_rate: float = DEFAULT_MUTATION_RATE,
               n_blocks: int = VALIDATION.n_cscv_blocks,
               n_folds: int = SEARCH_FOLDS) -> SearchResult:
    """NSGA-II-lite over the trend genome, seeded and budget-capped. Pure and
    network-free: callers fetch bars, this searches them.

    The feasibility clamp is applied to whatever space a caller passes: a slow
    period the frame cannot warm up produces a flat curve whose Sharpe is a
    silent 0.0, and a 0.0 wins a max-by-Sharpe pick against every losing
    candidate. Infeasibility is excluded, never scored.
    """
    if budget < 1 or population < 1:
        raise ValueError(f"a search needs something to spend: budget={budget}, "
                         f"population={population}")
    space = (space or GeneSpace()).limited(feasible_slow_max(len(bars), n_folds))
    rng = np.random.default_rng(seed)
    returns: dict[Genome, pd.Series] = {}  # the cache: one backtest per genome

    def evaluate(genomes: Sequence[Genome]) -> None:
        for genome in genomes:
            if genome in returns or len(returns) >= budget:
                continue
            candidate = candidate_for(genome, symbol, asset_class, slippage)
            matrix = build_return_matrix([candidate], {symbol: bars})
            returns[genome] = matrix[candidate.label]

    pop = initial_population(rng, space, population)
    evaluate(pop)
    pop = [g for g in pop if g in returns]

    trace: list[GenerationTrace] = []
    for generation in range(1, generations + 1):
        if len(returns) >= budget:
            break
        before = len(returns)
        dsr = _deflated(returns)
        objs = _objectives(pop, dsr, returns)
        rank, crowd = _rank_and_crowding(objs, non_dominated_sort(objs))
        parents = binary_tournament(rng, rank, crowd, len(pop))

        offspring = []
        for i in range(len(parents)):
            a, b = int(parents[i]), int(parents[(i + 1) % len(parents)])
            child = crossover(pop[a], pop[b], rng)
            offspring.append(repair(mutate(child, rng, space, mutation_rate),
                                    space, rng))
        evaluate(offspring)

        pooled = [g for g in _dedupe(list(pop) + offspring) if g in returns]
        dsr = _deflated(returns)
        pooled_objs = _objectives(pooled, dsr, returns)
        pop = _truncate(pooled, pooled_objs, population)
        trace.append(GenerationTrace(
            generation=generation, n_evaluated=len(returns),
            new_evals=len(returns) - before,
            front_size=len(non_dominated_sort(pooled_objs)[0]),
            n_trials=len(returns), best_dsr=max(dsr.values())))

    return _finalize(returns, bars, symbol, asset_class, slippage, space, seed,
                     population, generations, budget, mutation_rate, n_blocks,
                     n_folds, trace)


def _finalize(returns, bars, symbol, asset_class, slippage, space, seed,
              population, generations, budget, mutation_rate, n_blocks,
              n_folds, trace) -> SearchResult:
    """Score the evaluated population once the search has spent its budget.

    Walk-forward lands here and only here: it is the expensive measurement, and
    running it inside the loop would let an out-of-sample number influence
    selection, which is the leak this whole experiment exists to avoid."""
    genomes = list(returns)
    dsr = _deflated(returns)
    matrix = _matrix(returns)
    cscv = cscv_report(matrix, n_blocks=n_blocks)

    objs = _objectives(genomes, dsr, returns)
    front_genomes = [genomes[i] for i in non_dominated_sort(objs)[0]]
    front = [genome_label(g) for g in front_genomes]
    wf = walk_forward([candidate_for(g, symbol, asset_class, slippage)
                       for g in front_genomes], bars, n_folds=n_folds)

    cells = []
    for genome in genomes:
        label = genome_label(genome)
        cell_dsr = dsr[label]
        oos = wf.variant_oos_sharpe.get(label, float("nan"))
        cells.append(CellResult(
            label=label,
            params=strategy_params(candidate_for(genome, symbol, asset_class,
                                                 slippage).strategy),
            sharpe=sharpe_ratio(matrix[label]),
            dsr=cell_dsr,
            oos_sharpe=oos,
            passes=VALIDATION.passes_validation(cell_dsr, cscv.p_oos_loss, oos),
            max_dd=drawdown_of(matrix[label]),
            traded=traded_in(matrix[label]),
        ))
    evaluated = GridResult(symbol=symbol, cells=cells, cscv=cscv,
                           select_oos_sharpe=wf.oos_sharpe, n_trials=len(cells),
                           n_blocks=n_blocks, n_folds=n_folds, spec=TREND_FIELD,
                           field_correlation=optimize.field_correlation(matrix))
    return SearchResult(evaluated=evaluated, front=front, seed=seed,
                        population=population, generations=generations,
                        budget=budget, mutation_rate=mutation_rate, space=space,
                        trace=trace)


# --- the measurement --------------------------------------------------------


def pareto_front(field: GridResult) -> list[str]:
    """Labels on the field's Pareto front over (DSR, max drawdown), among the
    cells that clear the Sharpe floor — or, if none do, the cells that miss it
    by the least."""
    cells = field.cells
    objs = np.array([objective_row(c.dsr, c.max_dd, c.sharpe) for c in cells],
                    dtype=float).reshape(len(cells), 2)
    return [cells[i].label for i in non_dominated_sort(objs)[0]]


def _dsr_rank(field: GridResult, label: str) -> int:
    """1-based rank of a cell among the field's DSRs; NaN sorts last."""
    ordered = sorted(field.cells,
                     key=lambda c: c.dsr if c.dsr == c.dsr else float("-inf"),
                     reverse=True)
    return [c.label for c in ordered].index(label) + 1


@dataclass(frozen=True)
class LiveStanding:
    """Where the cell the live config trades sits in the exhaustive field —
    computed like every other cell, which is the whole point."""
    label: str
    present: bool
    field_size: int
    sharpe: float
    dsr: float
    oos_sharpe: float
    max_dd: float
    dsr_rank: int | None
    oos_rank: int | None
    passes: bool
    on_front: bool


def live_standing(field: GridResult) -> LiveStanding:
    label = field.spec.headline_label
    cell = field.cell(label)
    nan = float("nan")
    if cell is None:
        return LiveStanding(label, False, len(field.cells), nan, nan, nan, nan,
                            None, None, False, False)
    return LiveStanding(
        label=label, present=True, field_size=len(field.cells),
        sharpe=cell.sharpe, dsr=cell.dsr, oos_sharpe=cell.oos_sharpe,
        max_dd=cell.max_dd,
        dsr_rank=_dsr_rank(field, label),
        oos_rank=field.oos_rank(label),
        passes=cell.passes, on_front=label in pareto_front(field))


@dataclass(frozen=True)
class FrontCell:
    """One search-front cell scored against both trial counts. deflation_cost is
    what the honest count takes off a number the cheap search reported."""
    label: str
    dsr_own: float    # deflated at the search's cumulative trial count
    dsr_field: float  # the same returns deflated against the whole field
    deflation_cost: float
    sharpe: float
    oos_sharpe: float
    max_dd: float
    on_field_front: bool


@dataclass(frozen=True)
class Comparison:
    front_cells: list[FrontCell]
    field_front: list[str]
    search_front: list[str]
    matched: list[str]   # search-front cells that are on the exhaustive front
    coverage: float      # share of the true front the search found
    precision: float     # share of the search's front that is really on it
    search_pick: str     # the search's DSR-best evaluated cell
    search_pick_dsr_own: float    # deflated at the search's own trial count
    search_pick_dsr_field: float  # the same cell under the field's deflation
    field_best: str
    field_best_dsr: float
    dsr_regret: float    # what picking the search's cell costs, in field DSR
    front_haircut: float  # mean full-window minus OOS Sharpe over the front
    search_p_oos_loss: float
    field_p_oos_loss: float
    search_pbo: float    # advisory alongside p_oos_loss, not a gate input
    field_pbo: float
    evals: int
    field_size: int
    field_below_floor: int   # cells the Sharpe floor makes infeasible
    search_below_floor: int
    live: LiveStanding


def _below_floor(cells: Sequence[CellResult]) -> int:
    return sum(1 for c in cells if not feasible(c.dsr, c.max_dd, c.sharpe))


def compare(search: SearchResult, field: GridResult) -> Comparison:
    """The search against the ground truth it was hiding from."""
    field_front = pareto_front(field)
    matched = [label for label in search.front if label in field_front]
    pick = search.evaluated.dsr_best
    in_field = field.cell(pick.label)
    field_best = field.dsr_best
    pick_field_dsr = in_field.dsr if in_field else float("nan")

    cells = {c.label: c for c in search.evaluated.cells}
    haircuts = [cells[label].sharpe - cells[label].oos_sharpe
                for label in search.front
                if cells[label].oos_sharpe == cells[label].oos_sharpe]

    front_cells = []
    for label in search.front:
        cell = cells[label]
        in_field_cell = field.cell(label)
        field_dsr = in_field_cell.dsr if in_field_cell else float("nan")
        front_cells.append(FrontCell(
            label=label, dsr_own=cell.dsr, dsr_field=field_dsr,
            deflation_cost=cell.dsr - field_dsr, sharpe=cell.sharpe,
            oos_sharpe=cell.oos_sharpe, max_dd=cell.max_dd,
            on_field_front=label in field_front))

    return Comparison(
        front_cells=front_cells,
        field_front=field_front,
        search_front=list(search.front),
        matched=matched,
        coverage=len(matched) / len(field_front) if field_front else float("nan"),
        precision=(len(matched) / len(search.front) if search.front
                   else float("nan")),
        search_pick=pick.label,
        search_pick_dsr_own=pick.dsr,
        search_pick_dsr_field=pick_field_dsr,
        field_best=field_best.label,
        field_best_dsr=field_best.dsr,
        dsr_regret=field_best.dsr - pick_field_dsr,
        front_haircut=float(np.mean(haircuts)) if haircuts else float("nan"),
        search_p_oos_loss=search.evaluated.p_oos_loss,
        field_p_oos_loss=field.p_oos_loss,
        search_pbo=search.evaluated.pbo,
        field_pbo=field.pbo,
        evals=search.unique_evals,
        field_size=len(field.cells),
        field_below_floor=_below_floor(field.cells),
        search_below_floor=_below_floor(search.evaluated.cells),
        live=live_standing(field),
    )


# --- the artifact -----------------------------------------------------------


def _trace_entry(step: GenerationTrace) -> dict:
    return {"generation": step.generation, "n_evaluated": step.n_evaluated,
            "new_evals": step.new_evals, "front_size": step.front_size,
            "n_trials": step.n_trials, "best_dsr": json_number(step.best_dsr)}


def _live_entry(live: LiveStanding) -> dict:
    return {"label": live.label, "present": live.present,
            "field_size": live.field_size, "sharpe": json_number(live.sharpe),
            "dsr": json_number(live.dsr),
            "oos_sharpe": json_number(live.oos_sharpe),
            "max_dd": json_number(live.max_dd), "dsr_rank": live.dsr_rank,
            "oos_rank": live.oos_rank, "passes": live.passes,
            "on_front": live.on_front}


def build_search_report(field: GridResult, search: SearchResult,
                        comparison: Comparison,
                        generated_at: datetime | None = None) -> dict:
    """Both runs and the measurement as one record. The search block carries
    every knob the run turned, so the search replays from this file alone —
    see search_kwargs_from_report."""
    at = generated_at or datetime.now(timezone.utc)
    space = search.space
    return {
        "generated_at": at.isoformat(),
        "symbol": field.symbol,
        "asset_class": AssetClass.US_EQUITY.name,
        "slippage_pct": TREND_SLIPPAGE_PCT,
        "market_data_feed": MARKET_DATA_FEED,
        "equity_start": EQUITY_START.date().isoformat(),
        "n_blocks": field.n_blocks,
        "n_folds": field.n_folds,
        "exhaustive": build_grid_report(field, at),
        "search": {
            "seed": search.seed,
            "population": search.population,
            "generations": search.generations,
            "generations_run": search.generations_run,
            "budget": search.budget,
            "mutation_rate": search.mutation_rate,
            "sharpe_floor": SHARPE_FLOOR,
            "gene_space": {"fast": list(space.fast), "slow": list(space.slow),
                           "trail": list(space.trail), "volume": list(space.volume)},
            "space_size": space.size,
            "unique_evals": search.unique_evals,
            "front": list(search.front),
            "trace": [_trace_entry(t) for t in search.trace],
            "evaluated": build_grid_report(search.evaluated, at),
        },
        "comparison": {
            "front_cells": [
                {"label": row.label, "dsr_own": json_number(row.dsr_own),
                 "dsr_field": json_number(row.dsr_field),
                 "deflation_cost": json_number(row.deflation_cost),
                 "sharpe": json_number(row.sharpe),
                 "oos_sharpe": json_number(row.oos_sharpe),
                 "max_dd": json_number(row.max_dd),
                 "on_field_front": row.on_field_front}
                for row in comparison.front_cells],
            "field_front": comparison.field_front,
            "search_front": comparison.search_front,
            "matched": comparison.matched,
            "coverage": json_number(comparison.coverage),
            "precision": json_number(comparison.precision),
            "search_pick": {
                "label": comparison.search_pick,
                "dsr_own": json_number(comparison.search_pick_dsr_own),
                "dsr_field": json_number(comparison.search_pick_dsr_field),
            },
            "field_best": {"label": comparison.field_best,
                           "dsr": json_number(comparison.field_best_dsr)},
            "dsr_regret": json_number(comparison.dsr_regret),
            "front_haircut": json_number(comparison.front_haircut),
            "search_p_oos_loss": json_number(comparison.search_p_oos_loss),
            "field_p_oos_loss": json_number(comparison.field_p_oos_loss),
            "search_pbo": json_number(comparison.search_pbo),
            "field_pbo": json_number(comparison.field_pbo),
            "evals": comparison.evals,
            "field_size": comparison.field_size,
            "field_below_floor": comparison.field_below_floor,
            "search_below_floor": comparison.search_below_floor,
            "live_cell": _live_entry(comparison.live),
        },
    }


def search_kwargs_from_report(report: dict) -> dict:
    """The run_search call the artifact records, so the reproducibility claim is
    a property of the file rather than a sentence in it."""
    block = report["search"]
    space = block["gene_space"]
    return {
        "symbol": report["symbol"],
        "asset_class": AssetClass[report["asset_class"]],
        "slippage": report["slippage_pct"],
        "space": GeneSpace(fast=tuple(space["fast"]), slow=tuple(space["slow"]),
                           trail=tuple(space["trail"]),
                           volume=tuple(space["volume"])),
        "seed": block["seed"],
        "population": block["population"],
        "generations": block["generations"],
        "budget": block["budget"],
        "mutation_rate": block["mutation_rate"],
        "n_blocks": report["n_blocks"],
        "n_folds": report["n_folds"],
    }


def _fmt(x: float, spec: str = ".3f") -> str:
    return "n/a" if x != x else format(x, spec)  # x != x is True only for NaN


def _front_table(comparison: Comparison) -> str:
    """The search front with both deflations side by side — the measurement a
    reader came for, without opening the JSON."""
    own, whole = comparison.evals, comparison.field_size
    rows = [f"| Cell | Sharpe | DSR @ {own} trials | DSR @ {whole} trials | "
            f"maxDD | OOS Sharpe | On true front |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    rows += [f"| {r.label} | {_fmt(r.sharpe, '.2f')} | {_fmt(r.dsr_own)} | "
             f"{_fmt(r.dsr_field)} | {_fmt(r.max_dd, '.1%')} | "
             f"{_fmt(r.oos_sharpe, '.2f')} | "
             f"{'yes' if r.on_field_front else 'no'} |"
             for r in comparison.front_cells]
    return "\n".join(rows)


def _stop_phrase(search: SearchResult) -> str:
    """Which stop ended the run: the budget it spent, or the ceiling it reached
    with the budget still unspent."""
    if search.unique_evals >= search.budget:
        return (f"budget {search.budget} spent in {search.generations_run} "
                f"generations (ceiling {search.generations})")
    return (f"stopped at the {search.generations}-generation ceiling with "
            f"{search.unique_evals} of the {search.budget} budget")


def _floor_sentence(comparison: Comparison) -> str:
    starved = [side for side, under, total in (
        ("field", comparison.field_below_floor, comparison.field_size),
        ("evaluated", comparison.search_below_floor, comparison.evals))
        if under == total]
    counts = (f"{comparison.field_below_floor} field cells and "
              f"{comparison.search_below_floor} evaluated cells fall under the "
              f"floor.")
    if starved:
        return (f"No cell clears Sharpe > {SHARPE_FLOOR:g} on the "
                f"{' or '.join(starved)} side, so that front is its "
                "least-losing cells rather than a front over feasible ones; "
                + counts)
    return (f"Both fronts are built over cells with Sharpe > {SHARPE_FLOOR:g}; "
            + counts)


def build_search_markdown(field: GridResult, search: SearchResult,
                          comparison: Comparison, generated_at: datetime) -> str:
    """The findings in prose, decision first."""
    live = comparison.live
    n_field = len(field.cells)
    lines = [
        f"# {field.symbol} {field.spec.name} search vs exhaustive truth — "
        f"{generated_at.date().isoformat()}",
        "",
        f"An NSGA-II-lite search (seed {search.seed}, population "
        f"{search.population}, {_stop_phrase(search)}) covered "
        f"{search.unique_evals} unique backtests over the "
        f"same genome space the exhaustive grid enumerates in {n_field} cells "
        f"({optimize.axis_phrase(field)}), on daily {field.symbol} bars from "
        f"{EQUITY_START.date().isoformat()} on the {MARKET_DATA_FEED} feed. "
        f"Objectives: maximize Deflated Sharpe, minimize max drawdown. Every "
        f"search DSR is deflated at the search's OWN cumulative trial count, so "
        f"the deflation tightens as it spends the budget; the exhaustive field "
        f"deflates at {field.n_trials}.",
        "",
        "## Did the search find the truth",
        "",
        f"The exhaustive Pareto front has {len(comparison.field_front)} cells; "
        f"the search's front has {len(comparison.search_front)}, of which "
        f"{len(comparison.matched)} are exactly on the true front — coverage "
        f"{_fmt(comparison.coverage, '.0%')} of the true front, precision "
        f"{_fmt(comparison.precision, '.0%')} of the search's. The search's "
        f"DSR-best cell is {comparison.search_pick} at DSR "
        f"{_fmt(comparison.search_pick_dsr_own)} on its own "
        f"{comparison.evals} trials; the same cell deflated against the whole "
        f"{comparison.field_size}-cell field is "
        f"{_fmt(comparison.search_pick_dsr_field)}, against the field's best "
        f"{comparison.field_best} at {_fmt(comparison.field_best_dsr)} — a DSR "
        f"regret of {_fmt(comparison.dsr_regret)}."
        + (" The cheap search reports a bigger number for the same backtest; "
           "the trial count is the only thing that makes the two comparable."
           if comparison.search_pick_dsr_own
           - comparison.search_pick_dsr_field > 0.0005
           else " Over this field the two trial counts land the same cell in "
                "the same place, so the search's own deflation was not "
                "flattering it.")
        + " " + _floor_sentence(comparison),
        "",
        _front_table(comparison),
        "",
        "Walk-forward over the search front "
        + ("gives back" if comparison.front_haircut > 0 else "picks up")
        + f" a mean {_fmt(abs(comparison.front_haircut), '.2f')} Sharpe from "
        f"full-window to out-of-sample, and the select-then-trade path over the "
        f"front returns "
        f"OOS Sharpe {_fmt(search.evaluated.select_oos_sharpe, '.2f')}. The "
        f"gate's binding criterion, P(OOS loss), is "
        f"{_fmt(comparison.search_p_oos_loss)} over the {comparison.evals} "
        f"genomes the search evaluated and {_fmt(comparison.field_p_oos_loss)} "
        f"over the whole field (gate: <= {VALIDATION.p_oos_loss_max}) — the "
        f"search's population is a selected sample, so its number is not the "
        f"field's and neither licenses the other. PBO is advisory here: "
        f"{_fmt(comparison.search_pbo)} evaluated, "
        f"{_fmt(comparison.field_pbo)} over the field, with pairwise return "
        f"correlation across the field of mean "
        f"{_fmt(field.field_correlation.mean, '.2f')} / min "
        f"{_fmt(field.field_correlation.min, '.2f')}. "
        + (field.spec.pbo_note or ""),
        "",
        "## The live cell",
        "",
    ]
    if live.present:
        lines.append(
            f"{live.label} is what the live config trades. In the exhaustive "
            f"field it ranks {live.dsr_rank} of {live.field_size} on DSR "
            f"({_fmt(live.dsr)}) and {live.oos_rank} of {live.field_size} on "
            f"walk-forward OOS Sharpe ({_fmt(live.oos_sharpe, '.2f')}), with "
            f"full-window Sharpe {_fmt(live.sharpe, '.2f')} and max drawdown "
            f"{_fmt(live.max_dd, '.1%')}. It is "
            + ("ON" if live.on_front else "NOT on")
            + " the exhaustive Pareto front and it "
            + ("CLEARS" if live.passes else "FAILS")
            + f" the gate as one trial among {field.n_trials}."
            + ("" if live.passes else
               " That is a finding about the evidence behind the cell, not a "
               "signal to trade differently today."))
    else:
        lines.append(f"{live.label} is not in this field.")
    lines += [
        "",
        f"Both runs, the seed and the per-generation trace: "
        f"`{field.spec.report_prefix}-{generated_at.date().isoformat()}.json`. "
        f"The search replays from that file alone. This is an experiment "
        f"record, not a promotion artifact — the docs/validation/ verdicts are "
        f"unaffected.",
        "",
    ]
    return "\n".join(lines)


def print_search(field: GridResult, search: SearchResult,
                 comparison: Comparison) -> None:
    header = (f"{'Front':<7}{'Variant':<17}{'Sharpe':>8}{'DSR@own':>9}"
              f"{'DSR@field':>11}{'maxDD':>8}{'OOS':>8}{'Gate':>7}")
    print(header)
    print("-" * len(header))
    cells = {c.label: c for c in search.evaluated.cells}
    for row in comparison.front_cells:
        cell = cells[row.label]
        marks = []
        if row.on_field_front:
            marks.append("<-- on true front")
        if row.label == comparison.search_pick:
            marks.append("<-- DSR-best")
        if row.label == field.spec.headline_label:
            marks.append("<-- live")
        print(f"{'search':<7}{row.label:<17}{_fmt(row.sharpe, '.2f'):>8}"
              f"{_fmt(row.dsr_own):>9}{_fmt(row.dsr_field):>11}"
              f"{_fmt(row.max_dd, '.1%'):>8}"
              f"{_fmt(row.oos_sharpe, '.2f'):>8}"
              f"{'pass' if cell.passes else '-':>7}  "
              f"{'  '.join(marks)}".rstrip())

    print(f"\n{'':<7}-> seed {search.seed}, population {search.population}, "
          f"{_stop_phrase(search)}, {comparison.evals} unique evals of "
          f"a {comparison.field_size}-cell field")
    print(f"{'':<7}-> front coverage {_fmt(comparison.coverage, '.0%')} of the "
          f"{len(comparison.field_front)}-cell true front, precision "
          f"{_fmt(comparison.precision, '.0%')}, DSR regret "
          f"{_fmt(comparison.dsr_regret)}")
    print(f"{'':<7}-> {_floor_sentence(comparison)}")
    print(f"{'':<7}-> search pick {comparison.search_pick}: DSR "
          f"{_fmt(comparison.search_pick_dsr_own)} at {comparison.evals} trials "
          f"-> {_fmt(comparison.search_pick_dsr_field)} at "
          f"{comparison.field_size}; field best {comparison.field_best} "
          f"{_fmt(comparison.field_best_dsr)}")
    print(f"{'':<7}-> P(OOS loss) {_fmt(comparison.search_p_oos_loss)} "
          f"(evaluated) vs {_fmt(comparison.field_p_oos_loss)} (field); front "
          f"haircut {_fmt(comparison.front_haircut, '.2f')} Sharpe")
    print(f"{'':<7}-> advisory PBO {_fmt(comparison.search_pbo)} (evaluated) vs "
          f"{_fmt(comparison.field_pbo)} (field)")
    live = comparison.live
    if live.present:
        print(f"{'':<7}-> live {live.label}: DSR {_fmt(live.dsr)} (rank "
              f"{live.dsr_rank}/{live.field_size}), OOS "
              f"{_fmt(live.oos_sharpe, '.2f')} (rank {live.oos_rank}/"
              f"{live.field_size}), maxDD {_fmt(live.max_dd, '.1%')}, gate "
              f"{'PASS' if live.passes else 'FAIL'}"
              f"{', on the true front' if live.on_front else ''}")
    else:
        print(f"{'':<7}-> live {live.label} is not in this field")


def search_record_paths(generated_at: datetime,
                        directory: str = RESEARCH_REPORT_DIR
                        ) -> tuple[Path, Path]:
    return optimize.grid_record_paths(generated_at, directory,
                                      TREND_FIELD.report_prefix)


def write_search_report(report: dict, generated_at: datetime,
                        directory: str = RESEARCH_REPORT_DIR) -> str:
    return optimize.write_grid_report(report, generated_at, directory=directory,
                                      prefix=TREND_FIELD.report_prefix)


def write_search_markdown(field: GridResult, search: SearchResult,
                          comparison: Comparison, generated_at: datetime,
                          directory: str = RESEARCH_REPORT_DIR) -> str:
    path = search_record_paths(generated_at, directory)[1]
    os.makedirs(directory, exist_ok=True)
    with open(path, "x") as fh:
        fh.write(build_search_markdown(field, search, comparison, generated_at))
    return str(path)


def load_env() -> None:
    from dotenv import load_dotenv
    load_dotenv()


def main(argv: Sequence[str] = ()) -> int:
    # One timestamp for the whole run: the record's name is fixed before the
    # work starts, so the run cannot discover at the end that it has nowhere to
    # land.
    generated_at = datetime.now(timezone.utc)
    load_env()
    api_key = os.getenv("ALPACA_API_KEY") or None
    if not api_key:
        print("The trend search experiment needs Alpaca API keys.",
              file=sys.stderr)
        return 1
    if refuse_existing_record(search_record_paths(generated_at,
                                                  RESEARCH_REPORT_DIR)):
        return 1
    md = MarketData(api_key, os.getenv("ALPACA_SECRET_KEY") or None,
                    feed=MARKET_DATA_FEED)
    try:
        bars = md.get_bars(TREND_SYMBOL, AssetClass.US_EQUITY, DAILY,
                           EQUITY_START)
    except Exception as exc:
        print(f"{TREND_SYMBOL}: bars could not be fetched "
              f"({type(exc).__name__}: {exc})", file=sys.stderr)
        return 1

    max_slow = feasible_slow_max(len(bars), SEARCH_FOLDS)
    space = GeneSpace().limited(max_slow)
    candidates = trend_candidates(max_slow=max_slow)
    print(f"{TREND_SYMBOL}: {len(bars)} daily bars, {bars.index[0].date()} -> "
          f"{bars.index[-1].date()} ({MARKET_DATA_FEED})")
    print(f"\nFeasibility bound: slow_period <= {max_slow} (warm-up "
          f"{optimize.TREND_WARMUP_MULT}x slow within "
          f"{optimize.WARMUP_BLOCK_FRACTION:.0%} of the first of "
          f"{SEARCH_FOLDS + 1} walk-forward blocks). Exhaustive field: "
          f"{len(candidates)} cells; search space {space.size} genomes.\n")

    # One block count for both runs: the artifact records the field's, and the
    # two DSR/CSCV numbers are only comparable if they were computed alike.
    n_blocks = VALIDATION.n_cscv_blocks
    field = optimize.score_field(candidates, bars, TREND_SYMBOL, TREND_FIELD,
                                 n_blocks=n_blocks, n_folds=SEARCH_FOLDS)
    print("Exhaustive ground truth:")
    optimize.print_grid(field)

    result = run_search(bars, space=space, n_blocks=n_blocks,
                        n_folds=SEARCH_FOLDS)
    comparison = compare(result, field)
    print("\nSeeded multi-objective search:")
    print_search(field, result, comparison)

    report = build_search_report(field, result, comparison, generated_at)
    json_path = write_search_report(report, generated_at, RESEARCH_REPORT_DIR)
    md_path = write_search_markdown(field, result, comparison, generated_at,
                                    RESEARCH_REPORT_DIR)
    print(f"\nExperiment recorded in {json_path} and {md_path} — an experiment "
          f"record, not a promotion artifact.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
