"""Backtest-overfitting validation: pure, network-free.

Three tools that judge whether a variant's backtest edge is real or mined:

- build_return_matrix: run each candidate over the full window and assemble a
  T x N daily-return matrix (one column per candidate).
- cscv_report: Combinatorially Symmetric Cross-Validation over every symmetric
  in-sample/out-of-sample split — probability of backtest overfitting, the
  probability the in-sample winner loses out-of-sample, the logit distribution,
  the performance-degradation slope, and each candidate's out-of-sample standing.
  cscv_pbo returns the overfitting probability on its own.
- walk_forward: rolling folds that pick the in-sample winner and score it on the
  immediately-following out-of-sample window, then concatenate the OOS returns;
  it also scores every candidate on each window to expose per-variant OOS Sharpe.

Not-computable guards return float("nan") so callers test with math.isnan.
"""

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bot.backtest import Backtester, BacktestReport
from bot.metrics import deflated_sharpe_ratio, sharpe_ratio
from bot.strategies.base import Strategy
from config import InstrumentConfig

INITIAL_EQUITY = 100_000.0


@dataclass(frozen=True)
class Candidate:
    """A variant to validate: a label plus the (config, strategy) pair the
    Backtester runs, mirroring how research.py/diagnose.py express variants."""
    label: str
    cfg: InstrumentConfig
    strategy: Strategy


@dataclass(frozen=True)
class FoldResult:
    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    chosen_label: str
    oos_sharpe: float


@dataclass(frozen=True)
class WalkForwardResult:
    folds: list[FoldResult]
    oos_returns: pd.Series      # concatenated out-of-sample daily returns
    oos_sharpe: float           # Sharpe of the concatenation
    variant_oos_sharpe: dict[str, float]  # label -> Sharpe of its concatenated OOS


@dataclass(frozen=True)
class CscvReport:
    """What CSCV sees across every symmetric split, keyed panel by panel.

    pbo is one panel and not a verdict on its own: it asks only where the
    in-sample winner ranks out-of-sample, so a family of near-duplicates whose
    members have near-identical true skill scores near 0.5 whether the shared
    edge is real or absent. p_oos_loss reads the sign of the winner's
    out-of-sample Sharpe instead and separates those two cases.

    The per-candidate panels are keyed by column label and cover every split,
    not only the splits a column won in-sample, so a caller can ask how stable
    one chosen candidate's out-of-sample standing is across the field.
    """
    pbo: float                      # splits whose IS winner ranks bottom half OOS
    p_oos_loss: float               # splits whose IS winner has a negative OOS Sharpe
    median_logit: float             # median logit of the IS winner's relative OOS rank
    degradation_slope: float        # OLS slope of the winner's OOS Sharpe on its IS Sharpe
    logits: tuple[float, ...]       # per-split logits, in split-enumeration order
    oos_rank_mean: dict[str, float]  # label -> mean relative OOS rank over all splits
    is_best_share: dict[str, float]  # label -> fraction of splits the label won IS


def _daily_returns(report_curve: pd.Series) -> pd.Series:
    """Daily returns of an equity curve. Calendar days the resample invents over
    a non-trading gap are dropped before differencing, so the return across a
    weekend lands on the Monday instead of being stranded against an empty
    Sunday. Matches BacktestReport._daily_curve."""
    return report_curve.resample("1D").last().dropna().pct_change()


def _run_backtest(candidate: Candidate, bars: pd.DataFrame,
                  sim_start: pd.Timestamp, strategy: Strategy | None = None
                  ) -> tuple[pd.Series, BacktestReport]:
    """One candidate over bars, trading from sim_start onward (earlier bars warm
    up indicators), as (daily returns, report). No circuit breaker, matching
    research.

    strategy stands in for the candidate's own, so a caller that has to observe
    the run from inside the strategy can pass a wrapper around it and still take
    the returns and the report off the one backtest."""
    strategy = candidate.strategy if strategy is None else strategy
    prepared = strategy.prepare(bars)
    bt = Backtester(initial_equity=INITIAL_EQUITY, halt_enabled=False)
    report = bt.run([(candidate.cfg, strategy, prepared)], sim_start)
    return _daily_returns(report.equity_curve), report


def _run_returns(candidate: Candidate, bars: pd.DataFrame,
                 sim_start: pd.Timestamp) -> pd.Series:
    """Daily returns of one candidate over bars, trading from sim_start onward
    (earlier bars warm up indicators). No circuit breaker, matching research."""
    return _run_backtest(candidate, bars, sim_start)[0]


def _common_span(series: dict[str, pd.Series]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Latest first date and earliest last date across candidates — the window
    every candidate actually observed."""
    blank = sorted(label for label, s in series.items() if s.empty)
    if blank:
        raise ValueError(f"candidates produced no returns: {blank}")
    start = max(s.index.min() for s in series.values())
    end = min(s.index.max() for s in series.values())
    if start > end:
        raise ValueError("candidate return windows do not overlap: latest start "
                         f"{start} is after earliest end {end}")
    return start, end


def build_return_matrix(candidates: list[Candidate],
                        bars_by_symbol: dict[str, pd.DataFrame],
                        sim_start: pd.Timestamp | None = None) -> pd.DataFrame:
    """T x N daily returns, one column per candidate label. Trades from sim_start
    onward (earlier bars warm up indicators); default sim_start is each
    candidate's first bar, i.e. the full window with no warmup carve-out.

    Outer-joined on date, then trimmed to the span every candidate observed. The
    0.0 fill stands for a genuine flat day interior to a candidate's own window;
    leading and trailing non-overlap is trimmed away instead, because zero-vol
    padding there would flatter a short-history candidate's Sharpe and DSR.
    Candidates whose windows do not overlap at all raise ValueError.
    """
    series = {}
    for c in candidates:
        bars = bars_by_symbol[c.cfg.symbol]
        series[c.label] = _run_returns(c, bars,
                                       bars.index[0] if sim_start is None else sim_start)
    frame = pd.DataFrame(series).sort_index()
    if not series:
        return frame
    start, end = _common_span(series)
    return frame.loc[start:end].fillna(0.0)


def deflated_sharpes_from_matrix(matrix: pd.DataFrame, n_trials: int) -> dict[str, float]:
    """DSR per column of an already-built return matrix (no re-backtesting).

    n_trials is the number of variants tried; var_across_trials is the sample
    variance (ddof=1) of the columns' observed per-period Sharpes — the trials
    whose maximum the deflation benchmark is drawn from. A single variant has no
    dispersion (ddof=1 is undefined at n=1) so var is 0.0, and the DSR reduces to
    an undeflated PSR.

    Two effects push the benchmark up and so UNDERSTATE DSR: correlated parameter
    variants overcount n_trials, and ddof=1 estimates a larger variance than
    ddof=0. Both err conservative — the intended bias for a promotion gate.
    """
    sharpes = _column_sharpe(matrix.to_numpy(dtype=float))
    var_across_trials = float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
    return {
        col: deflated_sharpe_ratio(matrix[col], n_trials, var_across_trials)
        for col in matrix.columns
    }


def deflated_sharpes(candidates: list[Candidate],
                     bars_by_symbol: dict[str, pd.DataFrame],
                     sim_start: pd.Timestamp | None = None) -> dict[str, float]:
    """DSR per candidate label. Builds the return matrix once, then defers to
    deflated_sharpes_from_matrix. sim_start aligns the return window with a
    caller's backtest window (see build_return_matrix)."""
    matrix = build_return_matrix(candidates, bars_by_symbol, sim_start)
    return deflated_sharpes_from_matrix(matrix, len(candidates))


def _column_sharpe(returns: np.ndarray) -> np.ndarray:
    """Per-period Sharpe per column (mean/std, no annualization). Length is
    constant across columns within a split, so this ranks identically to an
    annualized Sharpe."""
    mean = returns.mean(axis=0)
    std = returns.std(axis=0, ddof=1)
    out = np.zeros_like(mean)
    np.divide(mean, std, out=out, where=std > 0)
    return out


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Slope of the least-squares line through (x, y). nan when x has fewer than
    two distinct values and the slope is unidentified."""
    if np.unique(x).size < 2:
        return float("nan")
    centered = x - x.mean()
    return float(centered @ (y - y.mean()) / (centered @ centered))


def cscv_report(matrix: pd.DataFrame, n_blocks: int = 16) -> CscvReport:
    """Every CSCV panel over a T x N return matrix, in one pass over the splits.

    Split rows into n_blocks contiguous equal blocks; for every choice of
    n_blocks/2 blocks as in-sample (the rest out-of-sample), score each column
    by Sharpe on both legs and take the in-sample winner. A column's relative
    out-of-sample rank is the fraction of the field it beats out-of-sample, so a
    winner ranking below 0.5 has fallen to the bottom half: pbo is how often
    that happens, and median_logit is the median of logit(rank) over splits. A
    rank is a multiple of 1/N and has no logit at either endpoint, so it is
    clipped half a step inside the unit interval — the midpoint of its own bin.

    Returns all-nan scalars and empty panels when not computable (< 2 columns,
    or fewer than 2 rows per block so a group would be a single row). n_blocks
    must be even for a symmetric IS/OOS split.

    Serial-dependence caveat: a position spanning a block boundary correlates
    the IS and OOS blocks it straddles, so the splits are not fully independent
    and the direction of the resulting bias is unclear.
    """
    if n_blocks % 2 != 0:
        raise ValueError(f"n_blocks must be even for a symmetric split, got {n_blocks}")
    values = matrix.to_numpy(dtype=float)
    t, n_cols = values.shape
    labels = [str(col) for col in matrix.columns]
    if n_cols < 2 or n_blocks < 2 or t < 2 * n_blocks:
        nan = float("nan")
        return CscvReport(nan, nan, nan, nan, (), {}, {})

    block_size = t // n_blocks
    blocks = values[: block_size * n_blocks].reshape(n_blocks, block_size, n_cols)
    half = n_blocks // 2
    half_step = 1.0 / (2 * n_cols)

    overfit = 0
    oos_losses = 0
    total = 0
    logits: list[float] = []
    is_best_sharpe: list[float] = []
    oos_sharpe_of_best: list[float] = []
    rank_sum = np.zeros(n_cols)
    best_counts = np.zeros(n_cols)
    for combo in itertools.combinations(range(n_blocks), half):
        oos = [b for b in range(n_blocks) if b not in combo]
        is_sharpe = _column_sharpe(blocks[list(combo)].reshape(-1, n_cols))
        oos_sharpe = _column_sharpe(blocks[oos].reshape(-1, n_cols))
        best = int(np.argmax(is_sharpe))
        # ranks[j]: the fraction of the field column j beats out-of-sample
        ranks = (oos_sharpe[np.newaxis, :] < oos_sharpe[:, np.newaxis]).mean(axis=1)

        if ranks[best] < 0.5:  # IS winner beats fewer than half the field OOS
            overfit += 1
        if oos_sharpe[best] < 0.0:
            oos_losses += 1
        clipped = min(max(float(ranks[best]), half_step), 1.0 - half_step)
        logits.append(math.log(clipped / (1.0 - clipped)))
        is_best_sharpe.append(float(is_sharpe[best]))
        oos_sharpe_of_best.append(float(oos_sharpe[best]))
        rank_sum += ranks
        best_counts[best] += 1
        total += 1

    return CscvReport(
        pbo=overfit / total,
        p_oos_loss=oos_losses / total,
        median_logit=float(np.median(logits)),
        degradation_slope=_ols_slope(np.asarray(is_best_sharpe),
                                     np.asarray(oos_sharpe_of_best)),
        logits=tuple(logits),
        oos_rank_mean=dict(zip(labels, (rank_sum / total).tolist(), strict=True)),
        is_best_share=dict(zip(labels, (best_counts / total).tolist(), strict=True)),
    )


def cscv_pbo(matrix: pd.DataFrame, n_blocks: int = 16) -> float:
    """Probability of backtest overfitting via CSCV: the fraction of symmetric
    IS/OOS splits whose in-sample winner lands in the bottom half out-of-sample.

    One panel of the CSCV report, kept as a scalar for callers that want only
    this number. On its own it cannot tell a near-duplicate family with a real
    shared edge from the same family with none — cscv_report carries the panels
    that can, and the caveats that apply to all of them.
    """
    return cscv_report(matrix, n_blocks).pbo


def walk_forward(candidates: list[Candidate], bars: pd.DataFrame,
                 n_folds: int = 4) -> WalkForwardResult:
    """Rolling walk-forward over bars shared by all candidates.

    Splits into n_folds+1 equal blocks; fold i trains on block i (picking the
    IS-best candidate by Sharpe) and scores every candidate on block i+1, warming
    up on all earlier bars via sim_start. oos_returns/oos_sharpe follow only the
    fold winner (the honest select-then-trade path); variant_oos_sharpe reports
    each candidate's own concatenated OOS returns across folds.

    Truncating the frame is what keeps a training run from reaching test bars, so
    this assumes prepare() is causal — a prepare() that reads forward within its
    own window scores clean here. diagnose.probe_lookahead is the enforcement.
    """
    n = len(bars)
    block = n // (n_folds + 1)
    if block == 0 or not candidates:
        return WalkForwardResult([], pd.Series(dtype=float), float("nan"), {})

    folds: list[FoldResult] = []
    oos_parts: list[pd.Series] = []
    variant_parts: dict[str, list[pd.Series]] = {c.label: [] for c in candidates}
    for i in range(n_folds):
        train_start_i = i * block
        test_start_i = (i + 1) * block
        test_end_i = n if i == n_folds - 1 else (i + 2) * block
        train_start = bars.index[train_start_i]
        test_start = bars.index[test_start_i]

        train_df = bars.iloc[:test_start_i]
        best = max(candidates,
                   key=lambda c: sharpe_ratio(_run_returns(c, train_df, train_start)))

        test_df = bars.iloc[:test_end_i]
        test_returns = {c.label: _run_returns(c, test_df, test_start).dropna()
                        for c in candidates}
        for label, r in test_returns.items():
            variant_parts[label].append(r)

        oos = test_returns[best.label]
        folds.append(FoldResult(train_start, test_start, bars.index[test_end_i - 1],
                                best.label, sharpe_ratio(oos)))
        oos_parts.append(oos)

    oos_returns = pd.concat(oos_parts) if oos_parts else pd.Series(dtype=float)
    variant_oos_sharpe = {label: sharpe_ratio(pd.concat(parts))
                          for label, parts in variant_parts.items()}
    return WalkForwardResult(folds, oos_returns, sharpe_ratio(oos_returns),
                             variant_oos_sharpe)
