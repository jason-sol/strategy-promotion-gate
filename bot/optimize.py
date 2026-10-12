"""Exhaustive parameter-grid experiment: how much edge does the search itself
manufacture?

The harness in bot.validation judges a handful of hand-picked variants. This
runs the WHOLE TurnOfMonth grid on one symbol and pays the deflation the
hand-picked run never paid: every cell's Deflated Sharpe is computed at
n_trials = the full grid size, so the number answers "is this cell good given
that 35 were tried?" rather than "given that 3 were shown".

What it measures (the induced-overfitting numbers):
- where the naive in-sample winner lands in the field's out-of-sample ranking,
  and what its Sharpe gives back from full-window to out-of-sample,
- whether deflation picks a different cell than the naive Sharpe does,
- the field's CSCV panels, including how often the in-sample winner loses
  money out of sample,
- how near-duplicate the field's cells are, as a correlation summary,
- how many cells clear the promotion gate on their own numbers,
- the walk-forward select-then-trade Sharpe, the honest path a live system
  would have followed.

Output is an experiment record under docs/research/, deliberately not the
docs/validation/ promotion artifact: nothing here promotes anything.

Usage: MARKET_DATA_FEED=sip op run --env-file .env -- uv run python -m bot.optimize
"""

import json
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bot.data import MarketData
from bot.metrics import max_drawdown, sharpe_ratio
from bot.research import DAILY, EQUITY_START, make_cfg, strategy_params, tom
from bot.strategies.trend_following import TrendFollowing
from bot.validation import (
    Candidate,
    CscvReport,
    build_return_matrix,
    cscv_report,
    deflated_sharpes_from_matrix,
    walk_forward,
)
from config import MARKET_DATA_FEED, RISK, VALIDATION, AssetClass

log = logging.getLogger(__name__)

SYMBOL = "VOO"
SLIPPAGE_PCT = 0.0001
ENTRY_BDAYS: tuple[int, ...] = (1, 2, 3, 4, 5)
EXIT_DAYS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
# The cell the live config trades, and the reason this experiment exists: its
# recorded DSR was deflated against a handful of variants, not the grid.
HEADLINE_LABEL = "TOM 2/5"
RESEARCH_REPORT_DIR = os.path.join("docs", "research")


@dataclass(frozen=True)
class FieldSpec:
    """What distinguishes one field from another, so everything below it is
    strategy-agnostic: the field's name in prose, the cell the live config
    trades, the artifact filename prefix, the parameters the field varies, and
    any field-specific reading of the field's advisory PBO."""
    name: str
    headline_label: str
    report_prefix: str
    axes: tuple[str, ...]
    pbo_note: str = ""


TOM_FIELD = FieldSpec(
    name="turn-of-month",
    headline_label=HEADLINE_LABEL,
    report_prefix="tom-grid",
    axes=("entry_window_bdays", "exit_day_of_month"),
    pbo_note=("Read it as a property of the field's correlation structure: "
              "overlapping calendar windows make the cells near-duplicates, so "
              "CSCV's in-sample winner pick is close to random and PBO tends "
              "toward 0.5 and above whatever the underlying edge. A dense grid "
              "is the wrong shape of field for this statistic; the hand-picked "
              "three-variant run's low PBO was not measuring 35 independent "
              "bets either."),
)

TREND_SYMBOL = "USO"
TREND_SLIPPAGE_PCT = 0.0005  # matches USO's research.EXPERIMENTS row
TREND_FAST: tuple[int, ...] = (5, 10, 15, 20, 30, 40, 50)
TREND_SLOW: tuple[int, ...] = (40, 60, 80, 100)
TREND_TRAIL: tuple[float, ...] = (2.0, 2.5, 3.0, 3.5, 4.0)
# None is a cell, not an absence: the ungated variant has to be IN the field
# so the gate is measured against its own alternative rather than assumed.
TREND_VOLUME_MULT: tuple[float | None, ...] = (None, 1.2, 1.5)
TREND_FIELD = FieldSpec(
    name="EMA-crossover trend",
    headline_label="T 20/100/3.0",
    report_prefix="trend-search",
    axes=("fast_period", "slow_period", "trail_atr_mult", "volume_gate_mult"),
    pbo_note=("Neighbouring EMA pairs trade the same crosses a few bars apart, "
              "so the cells are near-duplicates rather than independent bets "
              "and CSCV's in-sample winner pick is closer to random than the "
              "cell count suggests."),
)

# TrendFollowing.min_bars is slow_period * 4, and it is only a signal()
# self-guard: a strategy whose warm-up outruns its data never fires, so its
# curve is flat and its Sharpe is a silent 0.0 that BEATS every negative
# candidate in a max-by-Sharpe pick. Feasibility therefore has to be a
# construction-time filter, and a cell is feasible only if its warm-up fits
# inside this fraction of the first walk-forward train block.
TREND_WARMUP_MULT = 4
WARMUP_BLOCK_FRACTION = 0.75


def feasible_slow_max(n_bars: int, n_folds: int,
                      fraction: float = WARMUP_BLOCK_FRACTION) -> int:
    """Largest TrendFollowing slow_period whose warm-up leaves the first
    walk-forward train block room to actually trade."""
    block = n_bars // (n_folds + 1)
    return int(block * fraction) // TREND_WARMUP_MULT


def cell_label(entry_window_bdays: int, exit_day_of_month: int) -> str:
    """Grid label, within print_validation's 16-char variant column."""
    return f"TOM {entry_window_bdays}/{exit_day_of_month}"


def trend_cell_label(fast: int, slow: int, trail: float,
                     volume_mult: float | None = None) -> str:
    """Grid label, within print_validation's 16-char variant column.

    An ungated cell keeps the three-part name it had before the volume axis
    existed, so the 130 EMA cells stay comparable to the runs already recorded
    under those labels.
    """
    base = f"T {fast}/{slow}/{trail:.1f}"
    return base if volume_mult is None else f"{base}/{volume_mult:.1f}"


def grid_candidates(symbol: str = SYMBOL,
                    asset_class: AssetClass = AssetClass.US_EQUITY,
                    slippage: float = SLIPPAGE_PCT,
                    entry_bdays: Sequence[int] = ENTRY_BDAYS,
                    exit_days: Sequence[int] = EXIT_DAYS) -> list[Candidate]:
    """The grid as validation Candidates, built through research.tom/make_cfg so
    the cells are the same objects the harness would judge."""
    cfg = make_cfg(symbol, asset_class, slippage)
    return [Candidate(cell_label(entry, exit), cfg, tom(entry, exit))
            for entry in entry_bdays for exit in exit_days]


def trend_candidates(symbol: str = TREND_SYMBOL,
                     asset_class: AssetClass = AssetClass.US_EQUITY,
                     slippage: float = TREND_SLIPPAGE_PCT,
                     fasts: Sequence[int] = TREND_FAST,
                     slows: Sequence[int] = TREND_SLOW,
                     trails: Sequence[float] = TREND_TRAIL,
                     volume_mults: Sequence[float | None] = TREND_VOLUME_MULT,
                     max_slow: int | None = None) -> list[Candidate]:
    """The trend field as validation Candidates, long-only like the live config.

    TrendFollowing is constructed here rather than through research.trend()
    because that helper fixes trail_atr_mult at 3.0 and the trail is one of the
    axes. Cells with fast >= slow are not a crossover, and cells above max_slow
    cannot warm up inside the first train block (see feasible_slow_max)."""
    cfg = make_cfg(symbol, asset_class, slippage)
    return [
        Candidate(trend_cell_label(fast, slow, trail, volume), cfg,
                  TrendFollowing(fast_period=fast, slow_period=slow,
                                 trail_atr_mult=trail,
                                 atr_period=RISK.atr_period, allow_short=False,
                                 volume_gate_mult=volume))
        for slow in slows if max_slow is None or slow <= max_slow
        for fast in fasts if fast < slow
        for trail in trails
        for volume in volume_mults
    ]


@dataclass(frozen=True)
class CellResult:
    label: str
    params: dict       # the cell's genome, harvested like a validation artifact's
    sharpe: float      # full-window, undeflated: what a naive search maximizes
    dsr: float         # deflated against the full grid
    oos_sharpe: float  # walk-forward, this cell's own concatenated OOS returns
    passes: bool       # the promotion gate on this cell's own numbers
    max_dd: float = float("nan")  # full-window max drawdown, the second objective
    # False when the cell never opened a position. sharpe_ratio scores a flat
    # curve 0.0 and walk_forward selects by max(Sharpe), so a cell that never
    # trades outranks every losing cell. The warm-up filter cannot catch a gate
    # that starves the strategy — that depends on the bars, not the parameters —
    # so it is recorded per cell and counted in the report instead.
    traded: bool = True


@dataclass(frozen=True)
class FieldCorrelation:
    """How much of the field is the same bet twice: the mean and the minimum of
    the off-diagonal pairwise Pearson correlations of the cells' return columns.
    A field whose minimum pair still correlates near 1 carries far fewer
    independent trials than its cell count claims, which is what makes CSCV's
    in-sample winner pick near-random over it."""
    mean: float = float("nan")
    min: float = float("nan")


def field_correlation(matrix: pd.DataFrame) -> FieldCorrelation:
    """Not computable, and so all-nan, below two columns or when every pair is
    degenerate (a flat column has no correlation with anything)."""
    values = matrix.to_numpy(dtype=float)
    if values.shape[1] < 2:
        return FieldCorrelation()
    corr = np.corrcoef(values, rowvar=False)
    off_diagonal = corr[np.triu_indices_from(corr, k=1)]
    finite = off_diagonal[np.isfinite(off_diagonal)]
    if finite.size == 0:
        return FieldCorrelation()
    return FieldCorrelation(mean=float(finite.mean()), min=float(finite.min()))


@dataclass(frozen=True)
class GridResult:
    symbol: str
    cells: list[CellResult]
    cscv: CscvReport          # every CSCV panel over the whole field
    select_oos_sharpe: float  # walk-forward select-then-trade
    n_trials: int
    n_blocks: int
    n_folds: int
    spec: FieldSpec = TOM_FIELD
    field_correlation: FieldCorrelation = FieldCorrelation()

    @property
    def pbo(self) -> float:
        return self.cscv.pbo

    @property
    def p_oos_loss(self) -> float:
        return self.cscv.p_oos_loss

    def cell(self, label: str) -> CellResult | None:
        return next((c for c in self.cells if c.label == label), None)

    @property
    def is_best(self) -> CellResult:
        """The cell a naive search would ship: highest full-window Sharpe."""
        return max(self.cells, key=lambda c: _finite(c.sharpe))

    @property
    def dsr_best(self) -> CellResult:
        return max(self.cells, key=lambda c: _finite(c.dsr))

    @property
    def disagrees(self) -> bool:
        return self.dsr_best.label != self.is_best.label

    def oos_rank(self, label: str) -> int:
        """1-based rank of a cell among the field's walk-forward OOS Sharpes."""
        ordered = sorted(self.cells, key=lambda c: _finite(c.oos_sharpe), reverse=True)
        return [c.label for c in ordered].index(label) + 1

    @property
    def is_best_oos_rank(self) -> int:
        return self.oos_rank(self.is_best.label)

    @property
    def is_best_haircut(self) -> float:
        """Sharpe the naive winner gives back from full-window to OOS."""
        return self.is_best.sharpe - self.is_best.oos_sharpe

    @property
    def n_passing(self) -> int:
        return sum(1 for c in self.cells if c.passes)

    @property
    def n_dsr_passing(self) -> int:
        """Cells clearing the DSR bar on their own evidence. Reported apart from
        n_passing because the field's P(OOS loss) is a single scalar: when it
        exceeds the gate it closes every cell at once, which would otherwise read
        as "no cell has evidence"."""
        return sum(1 for c in self.cells if c.dsr >= VALIDATION.dsr_min)

    @property
    def oos_loss_blocks_gate(self) -> bool:
        return (self.p_oos_loss == self.p_oos_loss
                and self.p_oos_loss > VALIDATION.p_oos_loss_max)

    @property
    def headline(self) -> CellResult | None:
        return self.cell(self.spec.headline_label)

    @property
    def degenerate_labels(self) -> tuple[str, ...]:
        """Cells that never opened a position. Their 0.0 Sharpe is an artefact
        of standing aside, not a result, and it outranks every losing cell in a
        max-by-Sharpe selection — so a front built on them is a flat front."""
        return tuple(c.label for c in self.cells if not c.traded)

    @property
    def axes(self) -> dict[str, list]:
        """The field's dimensions and the values it actually enumerated."""
        # An axis whose "off" setting is None still needs a total order, and
        # off belongs at the head of it: the gate's own absence is the baseline
        # every other value on the axis is read against.
        return {axis: sorted({c.params[axis] for c in self.cells
                              if axis in c.params},
                             key=lambda v: (v is not None, v))
                for axis in self.spec.axes}


def _finite(x: float) -> float:
    """NaN sorts last rather than poisoning a max()."""
    return x if x == x else float("-inf")  # x == x is False only for NaN


def drawdown_of(returns: pd.Series) -> float:
    """Max drawdown of the equity curve a daily-return column implies."""
    return max_drawdown((1.0 + returns.fillna(0.0)).cumprod())


def traded_in(returns: pd.Series) -> bool:
    """Whether a daily-return column ever moved, which is the only evidence a
    cell opened a position."""
    return bool((returns.fillna(0.0) != 0.0).any())


def score_field(candidates: list[Candidate], bars: pd.DataFrame, symbol: str,
                spec: FieldSpec,
                n_blocks: int = VALIDATION.n_cscv_blocks,
                n_folds: int = VALIDATION.wf_folds) -> GridResult:
    """Score every cell of a field over pre-fetched bars. Pure and network-free,
    like research.validate_symbol: callers fetch bars, this judges them.

    The gate is applied per cell on the cell's own DSR and own OOS Sharpe
    (with the field's P(OOS loss)), which asks whether that cell would have been
    promotable standing alone. research.py's verdict instead pairs the selected
    variant's DSR with the select-then-trade OOS Sharpe; that number is reported
    separately as select_oos_sharpe.
    """
    matrix = build_return_matrix(candidates, {symbol: bars})
    cscv = cscv_report(matrix, n_blocks=n_blocks)
    wf = walk_forward(candidates, bars, n_folds=n_folds)
    dsr = deflated_sharpes_from_matrix(matrix, len(candidates))

    by_label = {c.label: c.strategy for c in candidates}
    cells = []
    for label in matrix.columns:
        cell_dsr = dsr[label]
        oos = wf.variant_oos_sharpe.get(label, float("nan"))
        cells.append(CellResult(
            label=label,
            params=strategy_params(by_label[label]),
            sharpe=sharpe_ratio(matrix[label]),
            dsr=cell_dsr,
            oos_sharpe=oos,
            passes=VALIDATION.passes_validation(cell_dsr, cscv.p_oos_loss, oos),
            max_dd=drawdown_of(matrix[label]),
            traded=traded_in(matrix[label]),
        ))
    return GridResult(symbol=symbol, cells=cells, cscv=cscv,
                      select_oos_sharpe=wf.oos_sharpe, n_trials=len(candidates),
                      n_blocks=n_blocks, n_folds=n_folds, spec=spec,
                      field_correlation=field_correlation(matrix))


def run_grid(bars: pd.DataFrame, symbol: str = SYMBOL,
             asset_class: AssetClass = AssetClass.US_EQUITY,
             slippage: float = SLIPPAGE_PCT,
             entry_bdays: Sequence[int] = ENTRY_BDAYS,
             exit_days: Sequence[int] = EXIT_DAYS,
             n_blocks: int = VALIDATION.n_cscv_blocks,
             n_folds: int = VALIDATION.wf_folds) -> GridResult:
    """The TurnOfMonth field through score_field."""
    return score_field(
        grid_candidates(symbol, asset_class, slippage, entry_bdays, exit_days),
        bars, symbol, TOM_FIELD, n_blocks=n_blocks, n_folds=n_folds)


def json_number(x):
    """JSON has no NaN literal, and readers that accept one disagree about what
    it means; a not-computable number is written as null."""
    return None if isinstance(x, float) and x != x else x


def _cell_entry(result: GridResult, cell: CellResult) -> dict:
    """One cell's numbers, including its own CSCV standing: oos_rank_mean is the
    share of the field it beats out of sample averaged over every split, and
    is_best_share how often it was the in-sample winner the field was judged
    on. A cell with a high is_best_share and a low oos_rank_mean is the one
    driving the field's overfitting numbers."""
    nan = float("nan")
    return {
        "label": cell.label,
        "params": {key: json_number(value) for key, value in cell.params.items()},
        "sharpe": json_number(cell.sharpe),
        "dsr": json_number(cell.dsr),
        "oos_sharpe": json_number(cell.oos_sharpe),
        "max_dd": json_number(cell.max_dd),
        "oos_rank_mean": json_number(result.cscv.oos_rank_mean.get(cell.label, nan)),
        "is_best_share": json_number(result.cscv.is_best_share.get(cell.label, nan)),
        "passes": cell.passes,
        "traded": cell.traded,
    }


def build_grid_report(result: GridResult,
                      generated_at: datetime | None = None) -> dict:
    """The experiment as a record: every cell's numbers plus the summary the
    findings are read off. Network-free — callers pass a computed result."""
    at = generated_at or datetime.now(timezone.utc)
    is_best, dsr_best = result.is_best, result.dsr_best
    headline = result.headline
    return {
        "generated_at": at.isoformat(),
        "symbol": result.symbol,
        "market_data_feed": MARKET_DATA_FEED,
        "equity_start": EQUITY_START.date().isoformat(),
        "n_trials": result.n_trials,
        "n_blocks": result.n_blocks,
        "n_folds": result.n_folds,
        "field": {"name": result.spec.name,
                  "headline_label": result.spec.headline_label,
                  "axes": list(result.spec.axes)},
        "grid": result.axes,
        "degenerate_cells": list(result.degenerate_labels),
        "cells": [_cell_entry(result, c) for c in result.cells],
        "summary": {
            "p_oos_loss": json_number(result.p_oos_loss),
            "pbo": json_number(result.pbo),
            "median_logit": json_number(result.cscv.median_logit),
            "degradation_slope": json_number(result.cscv.degradation_slope),
            "field_correlation": {
                "mean": json_number(result.field_correlation.mean),
                "min": json_number(result.field_correlation.min),
            },
            "select_then_trade_oos_sharpe": json_number(result.select_oos_sharpe),
            "n_passing": result.n_passing,
            "n_dsr_passing": result.n_dsr_passing,
            "oos_loss_blocks_gate": result.oos_loss_blocks_gate,
            "n_cells": len(result.cells),
            "is_best": _cell_entry(result, is_best) | {
                "oos_rank": result.is_best_oos_rank,
                "haircut": json_number(result.is_best_haircut),
            },
            "dsr_best": _cell_entry(result, dsr_best),
            "disagrees": result.disagrees,
            "headline": (_cell_entry(result, headline)
                         | {"oos_rank": result.oos_rank(headline.label)}
                         if headline else None),
        },
    }


def _fmt(x: float, spec: str = ".3f") -> str:
    return "n/a" if x != x else format(x, spec)  # x != x is True only for NaN


def print_grid(result: GridResult) -> None:
    header = (f"{'Symbol':<9}{'Variant':<17}{'Sharpe':>8}{'DSR':>8}{'OOS':>8}"
              f"{'Gate':>7}")
    print(header)
    print("-" * len(header))
    for c in result.cells:
        marks = []
        if c.label == result.is_best.label:
            marks.append("<-- IS-best")
        if c.label == result.dsr_best.label:
            marks.append("<-- DSR-best")
        if c.label == result.spec.headline_label:
            marks.append("<-- live")
        print(f"{result.symbol:<9}{c.label:<17}{_fmt(c.sharpe, '.2f'):>8}"
              f"{_fmt(c.dsr):>8}{_fmt(c.oos_sharpe, '.2f'):>8}"
              f"{'pass' if c.passes else '-':>7}  {'  '.join(marks)}".rstrip())

    is_best = result.is_best
    n = len(result.cells)
    corr = result.field_correlation
    print(f"\n{'':<9}-> field P(OOS loss) {_fmt(result.p_oos_loss)}  "
          f"select-then-trade OOS-Sharpe "
          f"{_fmt(result.select_oos_sharpe, '.2f')}  gate {result.n_passing}/{n} cells"
          f"  (DSR bar {result.n_dsr_passing}/{n})")
    print(f"{'':<9}-> advisory: PBO {_fmt(result.pbo)}, median logit "
          f"{_fmt(result.cscv.median_logit, '.2f')}, degradation slope "
          f"{_fmt(result.cscv.degradation_slope, '.2f')}, pairwise correlation "
          f"mean {_fmt(corr.mean, '.2f')} / min {_fmt(corr.min, '.2f')}")
    print(f"{'':<9}-> naive IS-best {is_best.label}: Sharpe "
          f"{_fmt(is_best.sharpe, '.2f')} -> OOS {_fmt(is_best.oos_sharpe, '.2f')} "
          f"(haircut {_fmt(result.is_best_haircut, '.2f')}), OOS rank "
          f"{result.is_best_oos_rank}/{n}")
    relation = "differs from" if result.disagrees else "is also"
    print(f"{'':<9}-> DSR-best {result.dsr_best.label} (DSR {_fmt(result.dsr_best.dsr)}) "
          f"{relation} IS-best {is_best.label} (DSR {_fmt(is_best.dsr)})")
    if result.headline:
        h = result.headline
        print(f"{'':<9}-> live {h.label}: Sharpe {_fmt(h.sharpe, '.2f')}  "
              f"DSR {_fmt(h.dsr)}  OOS {_fmt(h.oos_sharpe, '.2f')}  gate "
              f"{'PASS' if h.passes else 'FAIL'}")


def _binding_criterion(result: GridResult) -> str:
    """Which gate criterion actually closed the field, so a zero survivor count
    is not misread as an absence of evidence."""
    if result.n_passing:
        return (f"The gate's binding constraint is per-cell: {result.n_passing} "
                f"cells clear all three criteria.")
    if result.oos_loss_blocks_gate:
        return ("The binding criterion is the field's P(OOS loss) "
                f"({_fmt(result.p_oos_loss)} > {VALIDATION.p_oos_loss_max}), not "
                "any cell's own evidence — one scalar that closes every cell at "
                "once, and it says CSCV's in-sample winner lost money out of "
                "sample on more than half the splits.")
    return ("No cell clears the DSR bar at the honest trial count, which is the "
            "binding criterion here.")


def axis_phrase(result: GridResult) -> str:
    """The field's dimensions as prose: "fast_period 5-50 x slow_period 40-100"."""
    return " x ".join(f"{axis} {values[0]}-{values[-1]}"
                      for axis, values in result.axes.items() if values)


def build_grid_markdown(result: GridResult, generated_at: datetime) -> str:
    """The findings in prose, for a reader who will not open the JSON."""
    is_best, dsr_best, h = result.is_best, result.dsr_best, result.headline
    n = len(result.cells)
    lines = [
        f"# {result.symbol} {result.spec.name} grid — "
        f"{generated_at.date().isoformat()}",
        "",
        f"Exhaustive {result.spec.name} grid on {result.symbol}: "
        f"{axis_phrase(result)} "
        f"= {n} cells, daily bars from {EQUITY_START.date().isoformat()} on the "
        f"{MARKET_DATA_FEED} feed. Every Deflated Sharpe here is computed at "
        f"n_trials={result.n_trials} — the whole grid, which is the deflation a "
        f"hand-picked shortlist never pays.",
        "",
        "## What the search manufactures",
        "",
        f"The naive winner — argmax full-window Sharpe, no deflation — is "
        f"{is_best.label} at Sharpe {_fmt(is_best.sharpe, '.2f')}. Walk-forward "
        f"puts it at OOS Sharpe {_fmt(is_best.oos_sharpe, '.2f')}, rank "
        f"{result.is_best_oos_rank} of {n} in the field: a haircut of "
        f"{_fmt(result.is_best_haircut, '.2f')} Sharpe from picking on in-sample "
        f"numbers. Field P(OOS loss) is {_fmt(result.p_oos_loss)} "
        f"(gate: <= {VALIDATION.p_oos_loss_max}). The honest select-then-trade "
        f"path — each fold's in-sample winner traded through the next fold — "
        f"returns OOS Sharpe {_fmt(result.select_oos_sharpe, '.2f')}.",
        "",
        f"Advisory, not gated: field PBO {_fmt(result.pbo)}, median logit "
        f"{_fmt(result.cscv.median_logit, '.2f')}, degradation slope "
        f"{_fmt(result.cscv.degradation_slope, '.2f')}, and pairwise return "
        f"correlation across the field of mean "
        f"{_fmt(result.field_correlation.mean, '.2f')} / min "
        f"{_fmt(result.field_correlation.min, '.2f')}. "
        + (result.spec.pbo_note or ""),
        "",
        f"Deflation picks {dsr_best.label} (DSR {_fmt(dsr_best.dsr)}), which "
        + (f"is not the naive winner {is_best.label} (DSR {_fmt(is_best.dsr)}): "
           "ranking on undeflated in-sample Sharpe and ranking on evidence "
           "disagree over this field."
           if result.disagrees else
           "is also the naive winner: over this field the two rankings agree at "
           "the top.")
        + f" {result.n_passing} of {n} cells clear the full gate on their own "
        f"numbers (DSR >= {VALIDATION.dsr_min}, field P(OOS loss) <= "
        f"{VALIDATION.p_oos_loss_max}, own OOS Sharpe > 0), and "
        f"{result.n_dsr_passing} of {n} clear the DSR bar.",
        "",
        _binding_criterion(result),
        "",
        "## The live cell",
        "",
    ]
    if h:
        lines.append(
            f"{h.label} is what the live config trades — rank "
            f"{result.oos_rank(h.label)} of {n} in the field on OOS Sharpe. "
            f"Scored exactly "
            f"like every other cell: full-window Sharpe {_fmt(h.sharpe, '.2f')}, "
            f"DSR {_fmt(h.dsr)} at n_trials={result.n_trials}, walk-forward OOS "
            f"Sharpe {_fmt(h.oos_sharpe, '.2f')} — it "
            + ("CLEARS" if h.passes else "FAILS")
            + " the gate as one trial among "
            f"{result.n_trials}."
            + ("" if h.passes else
               " Its recorded validation_ref DSR was deflated against a "
               "hand-picked shortlist; at the honest trial count the citation "
               "does not hold. That is a finding about the evidence, not a "
               "signal to trade differently today.")
        )
    else:
        lines.append(f"{result.spec.headline_label} is not in this grid.")
    lines += ["", f"Per-cell numbers: `{result.spec.report_prefix}-"
                  f"{generated_at.date().isoformat()}.json`. "
                  "This is an experiment record, not a promotion artifact — the "
                  "docs/validation/ verdicts are unaffected.", ""]
    return "\n".join(lines)


def grid_record_paths(generated_at: datetime,
                      directory: str = RESEARCH_REPORT_DIR,
                      prefix: str = TOM_FIELD.report_prefix
                      ) -> tuple[Path, Path]:
    """The JSON and markdown record one run writes. One derivation, so a caller
    can ask whether a run's record already exists without guessing at the name
    the writers will use."""
    stem = f"{prefix}-{generated_at.date().isoformat()}"
    return Path(directory) / f"{stem}.json", Path(directory) / f"{stem}.md"


def write_grid_report(report: dict, generated_at: datetime,
                      directory: str = RESEARCH_REPORT_DIR,
                      prefix: str = TOM_FIELD.report_prefix) -> str:
    path = grid_record_paths(generated_at, directory, prefix)[0]
    os.makedirs(directory, exist_ok=True)
    # "x": a record of a run that happened is never overwritten by a later one.
    with open(path, "x") as fh:
        json.dump(report, fh, indent=2, allow_nan=False)
        fh.write("\n")
    return str(path)


def write_grid_markdown(result: GridResult, generated_at: datetime,
                        directory: str = RESEARCH_REPORT_DIR,
                        prefix: str = TOM_FIELD.report_prefix) -> str:
    path = grid_record_paths(generated_at, directory, prefix)[1]
    os.makedirs(directory, exist_ok=True)
    with open(path, "x") as fh:
        fh.write(build_grid_markdown(result, generated_at))
    return str(path)


def refuse_existing_record(paths: Sequence[Path]) -> bool:
    """True when a record already exists, so the caller can stop before it
    spends a run it cannot record."""
    taken = [str(p) for p in paths if p.exists()]
    if not taken:
        return False
    print(f"An experiment record already exists: {', '.join(taken)}. Move it "
          f"aside before re-running; a run is never overwritten.",
          file=sys.stderr)
    return True


def run_tom_experiment() -> int:
    # One timestamp for the whole run: the record's name is fixed before the
    # work starts, so the run cannot discover at the end that it has nowhere to
    # land.
    generated_at = datetime.now(timezone.utc)
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY") or None
    if not api_key:
        print("The TOM grid experiment needs Alpaca API keys.", file=sys.stderr)
        return 1
    if refuse_existing_record(grid_record_paths(generated_at,
                                                RESEARCH_REPORT_DIR)):
        return 1
    md = MarketData(api_key, os.getenv("ALPACA_SECRET_KEY") or None,
                    feed=MARKET_DATA_FEED)
    try:
        bars = md.get_bars(SYMBOL, AssetClass.US_EQUITY, DAILY, EQUITY_START)
    except Exception as exc:
        print(f"{SYMBOL}: bars could not be fetched ({type(exc).__name__}: {exc})",
              file=sys.stderr)
        return 1
    print(f"{SYMBOL}: {len(bars)} daily bars, "
          f"{bars.index[0].date()} -> {bars.index[-1].date()} ({MARKET_DATA_FEED})")
    print(f"\nExhaustive TOM grid: {len(ENTRY_BDAYS)} x {len(EXIT_DAYS)} = "
          f"{len(ENTRY_BDAYS) * len(EXIT_DAYS)} cells, DSR deflated at the full "
          f"grid size.\n")

    result = run_grid(bars)
    print_grid(result)

    json_path = write_grid_report(build_grid_report(result, generated_at),
                                  generated_at, RESEARCH_REPORT_DIR)
    md_path = write_grid_markdown(result, generated_at, RESEARCH_REPORT_DIR)
    print(f"\nExperiment recorded in {json_path} and {md_path} — an experiment "
          f"record, not a promotion artifact.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """No argument runs the TurnOfMonth grid; `trend` runs the trend-search
    experiment in bot.search."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return run_tom_experiment()
    if args[0] == "trend":
        from bot.search import main as trend_main
        return trend_main(args[1:])
    print("usage: python -m bot.optimize [trend]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
