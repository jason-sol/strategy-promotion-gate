"""Promotion-gate calibration against realised forward performance.

The gate promotes on DSR >= dsr_min, P(OOS loss) <= p_oos_loss_max and a
positive walk-forward OOS Sharpe. The ceiling on P(OOS loss) is a chosen
number, and nothing recorded so far says whether it sits anywhere near where it
earns its keep. This replays the gate at a series of historical cutoffs — the
same harness research.py runs, over bars truncated to each cutoff — and then
measures what the variant the gate selected actually did over the year that
followed.

What it measures:
- the gate's three inputs as of each cutoff, plus the advisory PBO,
- the selected variant's realised forward Sharpe and total return over the next
  year, the round trips and held bars behind them, and the symbol's
  buy-and-hold Sharpe alongside for reference,
- the rank correlation of each of those inputs against forward Sharpe, which
  says which criterion carries the forward signal, read over every row and
  again over only the rows whose variant was in the market — a variant that
  never entered scores Sharpe 0.00 and would otherwise rank as a measured zero,
- for a range of candidate P(OOS loss) ceilings, how passers and failers
  separate on forward Sharpe and how many verdicts each ceiling would change.

Output is an experiment record under docs/research/, deliberately not the
docs/validation/ promotion artifact: nothing here promotes anything, and the
row count is far too small to settle a ceiling on its own.

Schema policy: a record with no `schema_version` is the pre-1 shape, whose
`calibration` is one flat calibration object. Version 1 reports `calibration`
as {all_rows, traded_only}. The version is bumped only when an existing key
changes shape or meaning — a key added alongside the existing ones does not
bump it, so a consumer must tolerate keys it does not know.

A record is never overwritten by a later run: both files are opened "x", so
regenerating a given day's record means deleting it first, and the run it
replaces stays in git history.

Usage: MARKET_DATA_FEED=sip op run --env-file .env -- uv run python -m bot.gate_backtest
"""

import argparse
import logging
import math
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from bot import optimize
from bot.data import MarketData
from bot.metrics import sharpe_ratio
from bot.optimize import RESEARCH_REPORT_DIR, json_number
from bot.research import (
    CRYPTO_START,
    DAILY,
    EQUITY_START,
    EXPERIMENTS,
    make_cfg,
    validate_symbol,
)
from bot.strategies.base import Action, Strategy
# The runner is validation's, deliberately: the forward leg has to be scored on
# the same simulator, starting equity and daily-return convention as the
# walk-forward OOS Sharpe it is compared against.
from bot.validation import Candidate, _run_backtest
from config import MARKET_DATA_FEED, VALIDATION, AssetClass

log = logging.getLogger(__name__)

REPORT_PREFIX = "gate-backtest"

# Bumped when the record's shape changes, so a consumer can refuse a layout it
# does not know. 1: calibration as {all_rows, traded_only}.
SCHEMA_VERSION = 1

FORWARD_DAYS = 365
# A forward window worth scoring. Equities trade ~252 days a year and crypto
# ~365, so the same calendar year yields different bar counts and the bound has
# to follow the asset class rather than the window length.
MIN_FORWARD_BARS = 200
MIN_FORWARD_BARS_CRYPTO = 300

# The ceilings the calibration reads the gate at. The live ceiling is the last
# of them, so every table has the status quo in it as a column to compare to.
CEILINGS: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5)

# Equity history starts 2016, so a 2019 cutoff already carries four years of
# in-sample bars. Crypto starts 2020 and its early cutoffs fail feasibility on
# their own numbers rather than being excluded by hand.
DEFAULT_CUTOFF_YEARS: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023, 2024)


def default_cutoffs(years: Sequence[int] = DEFAULT_CUTOFF_YEARS) -> list[date]:
    """Year-end cutoffs. The truncation is by date, so each one resolves to the
    last trading day of its year without a calendar to consult."""
    return [date(year, 12, 31) for year in years]


def parse_cutoffs(text: str) -> list[date]:
    """Comma-separated ISO dates. Raises ValueError on anything else, so the
    argument parser refuses the run rather than silently dropping a cutoff."""
    pieces = [piece.strip() for piece in text.split(",")]
    dates = []
    for piece in pieces:
        if not piece:
            continue
        try:
            dates.append(date.fromisoformat(piece))
        except ValueError as exc:
            raise ValueError(f"{piece!r} is not an ISO date (YYYY-MM-DD)") from exc
    if not dates:
        raise ValueError("no cutoff dates given")
    return dates


def parse_symbols(text: str) -> list[str]:
    symbols = [piece.strip().upper() for piece in text.split(",") if piece.strip()]
    if not symbols:
        raise ValueError("no symbols given")
    return symbols


def shortlists(experiments: Iterable[tuple] = EXPERIMENTS) -> dict[str, tuple]:
    """research's hand-picked shortlists, symbol -> (asset_class, slippage,
    [(label, factory)]) — the same grouping research.main judges, so the replay
    gates on the field the live verdicts were drawn from."""
    grouped: dict[str, tuple] = {}
    for symbol, asset_class, slippage, label, factory in experiments:
        grouped.setdefault(symbol, (asset_class, slippage, []))[2].append(
            (label, factory))
    return grouped


@dataclass(frozen=True)
class GateRow:
    """One (cutoff, symbol) cell: what the gate saw, and what happened next."""
    cutoff: date
    symbol: str
    variant: str          # the variant selection would have promoted
    dsr: float
    p_oos_loss: float
    pbo: float            # advisory, not a gate input
    oos_sharpe: float
    forward_sharpe: float          # realised, annualised, over the next year
    forward_return: float          # realised total return over the same window
    forward_trades: int            # round trips the variant closed in it
    forward_exposure_bars: int     # forward bars it held a position through
    # Entries it signalled in the window — signals, not fills: the risk manager
    # can refuse to size one, and a refused signal recurs on later bars.
    forward_entry_signals: int
    buy_and_hold_sharpe: float     # the symbol's own, for reference
    history_bars: int
    forward_bars: int
    first_train_bars: int             # walk-forward's first train block
    # Variants that cannot warm up inside that block. They read flat in the
    # earliest training folds, so selection there cannot reach them — research
    # runs its shortlists the same way, and a replay that dropped the cell
    # instead would be stricter than the gate it is calibrating.
    warmup_flat_variants: tuple[str, ...]

    @property
    def promoted(self) -> bool:
        return VALIDATION.passes_validation(self.dsr, self.p_oos_loss,
                                            self.oos_sharpe)

    @property
    def traded(self) -> bool:
        """Whether the forward Sharpe is a measurement at all. A variant that
        never entered scores 0.00 exactly as one that traded to a flat year
        does, and only the position it took separates the two.

        Both counts are needed: a variant still holding at the window's end
        closed no round trip, and one stopped out on the bar after its entry
        carried no bar into a later decision.

        An entry with neither behind it is not one of them. It has no held bar
        to measure, so its forward "Sharpe" is the slippage on that single
        fill — a number the entry count discloses rather than qualifies."""
        return self.forward_trades > 0 or self.forward_exposure_bars > 0

    def passes_at(self, ceiling: float) -> bool:
        """The full gate with the P(OOS loss) ceiling moved. Delegates to the
        live gate with one field replaced rather than restating its three
        criteria, so a change to the gate reaches this calibration too."""
        return replace(VALIDATION, p_oos_loss_max=ceiling).passes_validation(
            self.dsr, self.p_oos_loss, self.oos_sharpe)

    def within_ceiling(self, ceiling: float) -> bool:
        """The P(OOS loss) criterion alone, NaN passing through as the gate's
        does — too few candidates to compute CSCV is not evidence."""
        return math.isnan(self.p_oos_loss) or self.p_oos_loss <= ceiling

    @property
    def clears_dsr(self) -> bool:
        return self.dsr >= VALIDATION.dsr_min


@dataclass(frozen=True)
class SkippedCell:
    """A cell that was not scored, and why. Recorded rather than dropped: a
    silently absent cell reads as one the gate rejected."""
    cutoff: date
    symbol: str
    reason: str


def split_bars(bars: pd.DataFrame, cutoff: date,
               forward_days: int = FORWARD_DAYS
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(bars at or before the cutoff, bars in the forward_days after it).

    The cutoff is a calendar date, so the comparison runs on each bar's own
    date: a daily bar stamped at any hour of the cutoff day is history, and the
    gate never sees a bar it could not have had."""
    dates = bars.index.date
    end = cutoff + timedelta(days=forward_days)
    return bars[dates <= cutoff], bars[(dates > cutoff) & (dates <= end)]


def first_train_bars(history: pd.DataFrame, n_folds: int) -> int:
    """Bars in walk-forward's first train block. Fold 0 trains on
    bars.iloc[:block] with block = n // (n_folds + 1), so the block length is
    the whole of the fold arithmetic that matters here."""
    return len(history) // (n_folds + 1)


def warmup_flat_variants(
        variants: Sequence[tuple[str, Callable[[], Strategy]]],
        train_bars: int) -> tuple[str, ...]:
    """Labels whose warm-up outruns the first train block. Such a variant never
    fires there, and sharpe_ratio scores its flat curve 0.0 — so the earliest
    folds cannot select it. Disclosed rather than acted on: research's own
    shortlists carry variants like this, and the replay's job is to reproduce
    the gate, not to tighten it."""
    return tuple(label for label, factory in variants
                 if factory().min_bars > train_bars)


def history_shortfall(history: pd.DataFrame,
                      variants: Sequence[tuple[str, Callable[[], Strategy]]],
                      n_blocks: int, n_folds: int) -> str | None:
    """Why the truncated history cannot carry the gate, or None if it can.

    Two bounds. CSCV needs two rows per block, or a block is a single row and
    every panel comes back NaN. And something in the field has to be able to
    warm up inside walk-forward's first train block: when even the shortest
    warm-up outruns it, no variant can trade before the first test fold and
    there is no verdict to replay. A field where only some variants fall short
    is scored, with those labels disclosed on the row."""
    n = len(history)
    if n < 2 * n_blocks:
        return f"{n} history bars, CSCV needs {2 * n_blocks}"
    train_bars = first_train_bars(history, n_folds)
    shortest = min(factory().min_bars for _label, factory in variants)
    if shortest > train_bars:
        return (f"shortest warm-up {shortest} bars exceeds the "
                f"{train_bars}-bar first train block")
    return None


def forward_shortfall(forward: pd.DataFrame,
                      asset_class: AssetClass) -> str | None:
    """Why the forward window is too short to measure, or None if it is not."""
    need = (MIN_FORWARD_BARS_CRYPTO if asset_class == AssetClass.CRYPTO
            else MIN_FORWARD_BARS)
    if len(forward) < need:
        return f"{len(forward)} forward bars, needs {need}"
    return None


@dataclass(frozen=True)
class ForwardRun:
    """What one variant did over the forward window: its daily returns, the
    round trips it closed, the bars it spent holding a position, and the
    entries it signalled."""
    returns: pd.Series
    trades: int
    exposure_bars: int
    entry_signals: int


class _ExposureProbe(Strategy):
    """A variant with a counter on it. The backtester hands signal() the live
    position, so the variant's own decision points are where a position can be
    observed bar by bar without the report having to carry one; everything else
    delegates, since what is being measured is the wrapped variant unchanged.

    The bar count is bars carried INTO a decision, which is one short of the
    true holding period: an entry is placed after that bar's decision, and a
    stop clears the position before the next one. Round trips are counted from
    the report, so nothing depends on this measuring the entry bar — but an
    entry on the window's final bar has no later decision and closes nothing,
    so the entry signals are counted here too.

    An entry signal is not quite a fill: the risk manager can refuse to size
    one. Over a single instrument with no correlated book it refuses only a
    position that rounds to zero shares, which is why the signal stands in for
    the fill rather than the report being asked for a position it does not
    carry."""

    def __init__(self, inner: Strategy):
        self.inner = inner
        self.exposure_bars = 0
        self.entry_signals = 0

    @property
    def min_bars(self) -> int:
        return self.inner.min_bars

    @property
    def stop_atr_mult(self) -> float:
        return self.inner.stop_atr_mult

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        return self.inner.prepare(bars)

    def signal(self, df, i, position):
        if position is not None:
            self.exposure_bars += 1
        signal = self.inner.signal(df, i, position)
        if position is None and signal.action in (Action.LONG, Action.SHORT):
            self.entry_signals += 1
        return signal


def forward_run(candidate: Candidate, bars: pd.DataFrame,
                forward: pd.DataFrame) -> ForwardRun:
    """One variant traded across the forward window.

    Prepared on every bar up to that window's end so its indicators warm on the
    history the gate already saw, with sim_start at the first bar after the
    cutoff so trading begins where the gate's evidence stops. No circuit
    breaker, matching research. Returns, round trips and held bars all come off
    the one run: measuring any of them separately would mean backtesting the
    window twice. signal() is called only from sim_start onward, so the held
    bars are the forward window's alone."""
    frame = bars.loc[:forward.index[-1]]
    probe = _ExposureProbe(candidate.strategy)
    returns, report = _run_backtest(candidate, frame, forward.index[0],
                                    strategy=probe)
    # The 0.0 fill is the first bar's absent pct_change, not a flat day.
    return ForwardRun(returns.fillna(0.0), len(report.trades),
                      probe.exposure_bars, probe.entry_signals)


def total_return(returns: pd.Series) -> float:
    return float((1.0 + returns.fillna(0.0)).prod() - 1.0)


def evaluate_cell(symbol: str, asset_class: AssetClass, slippage: float,
                  variants: Sequence[tuple[str, Callable[[], Strategy]]],
                  bars: pd.DataFrame, cutoff: date,
                  forward_days: int = FORWARD_DAYS,
                  n_blocks: int = VALIDATION.n_cscv_blocks,
                  n_folds: int = VALIDATION.wf_folds) -> GateRow | SkippedCell:
    """Run the gate as of one cutoff, then measure the year that followed. Pure
    and network-free, like research.validate_symbol: callers fetch bars."""
    history, forward = split_bars(bars, cutoff, forward_days)
    reason = (history_shortfall(history, variants, n_blocks, n_folds)
              or forward_shortfall(forward, asset_class))
    if reason:
        return SkippedCell(cutoff, symbol, reason)

    verdict = validate_symbol(symbol, asset_class, slippage, variants, history,
                              n_blocks=n_blocks, n_folds=n_folds)
    factory = dict(variants)[verdict.best_label]
    candidate = Candidate(verdict.best_label,
                          make_cfg(symbol, asset_class, slippage), factory())
    realised = forward_run(candidate, bars, forward)
    train_bars = first_train_bars(history, n_folds)
    return GateRow(
        cutoff=cutoff,
        symbol=symbol,
        variant=verdict.best_label,
        dsr=verdict.best_dsr,
        p_oos_loss=verdict.p_oos_loss,
        pbo=verdict.pbo,
        oos_sharpe=verdict.oos_sharpe,
        forward_sharpe=sharpe_ratio(realised.returns),
        forward_return=total_return(realised.returns),
        forward_trades=realised.trades,
        forward_exposure_bars=realised.exposure_bars,
        forward_entry_signals=realised.entry_signals,
        buy_and_hold_sharpe=sharpe_ratio(forward["close"].pct_change()),
        history_bars=len(history),
        forward_bars=len(forward),
        first_train_bars=train_bars,
        warmup_flat_variants=warmup_flat_variants(variants, train_bars),
    )


@dataclass(frozen=True)
class Split:
    """Forward Sharpe on each side of one criterion. An empty bucket has no
    mean, so it reads NaN rather than 0.0 — a bucket nobody landed in is not a
    bucket that scored zero."""
    n_pass: int
    n_fail: int
    mean_pass: float
    mean_fail: float

    @property
    def separation(self) -> float:
        """How much better the passers did. NaN whenever a bucket is empty."""
        return self.mean_pass - self.mean_fail


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def split_rows(rows: Sequence[GateRow],
               predicate: Callable[[GateRow], bool]) -> Split:
    passers = [r.forward_sharpe for r in rows if predicate(r)]
    failers = [r.forward_sharpe for r in rows if not predicate(r)]
    return Split(len(passers), len(failers), _mean(passers), _mean(failers))


def _average_ranks(values: Sequence[float]) -> list[float]:
    """One-based ranks, tied values sharing the average of the ranks they span."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start
        while (stop + 1 < len(order)
               and values[order[stop + 1]] == values[order[start]]):
            stop += 1
        shared = (start + stop) / 2.0 + 1.0
        for k in range(start, stop + 1):
            ranks[order[k]] = shared
        start = stop + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation, ties taking their average rank.

    NaN when fewer than two pairs survive or either ranking is constant: a
    series with no dispersion has no ordering to correlate against. Pairs where
    either side is NaN are dropped rather than ranked, since a not-computable
    number has no place in an ordering."""
    pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if x == x and y == y]
    if len(pairs) < 2:
        return float("nan")
    rx = _average_ranks([p[0] for p in pairs])
    ry = _average_ranks([p[1] for p in pairs])
    mx, my = _mean(rx), _mean(ry)
    dx = [r - mx for r in rx]
    dy = [r - my for r in ry]
    denominator = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    if denominator == 0.0:
        return float("nan")
    return float(sum(a * b for a, b in zip(dx, dy, strict=True)) / denominator)


@dataclass(frozen=True)
class CeilingRow:
    """One candidate P(OOS loss) ceiling, read three ways: the full gate at that
    ceiling, the loss-probability criterion on its own, and how many verdicts
    the move away from the live ceiling would change."""
    ceiling: float
    full: Split
    p_oos_loss_only: Split
    changed: int


@dataclass(frozen=True)
class Calibration:
    n_rows: int
    # Rows whose variant was in the market, counted whether or not the
    # traded-only read was computed: an absent read is not an empty exclusion.
    n_traded: int
    spearman_p_oos_loss: float
    spearman_dsr: float
    spearman_pbo: float
    dsr_only: Split       # the DSR bar alone; no ceiling enters it
    ceilings: list[CeilingRow]
    base_ceiling: float
    # The same calibration over the rows whose variant actually traded, or None
    # when no row was excluded and the two reads are the same one.
    traded_only: "Calibration | None" = None

    @property
    def traded(self) -> "Calibration":
        """The traded-only read, which is this one when every variant traded."""
        return self.traded_only or self

    @property
    def n_no_trade(self) -> int:
        return self.n_rows - self.n_traded

    @property
    def panels(self) -> dict[str, float]:
        return {"P(OOS loss)": self.spearman_p_oos_loss,
                "DSR": self.spearman_dsr,
                "PBO": self.spearman_pbo}

    @property
    def strongest_predictor(self) -> tuple[str, float]:
        """(name, rho) of the gate input that ranks forward Sharpe most
        strongly, by magnitude. NaN never wins; all-NaN has no answer."""
        panels = self.panels
        finite = {name: rho for name, rho in panels.items() if rho == rho}
        if not finite:
            return "none", float("nan")
        name = max(finite, key=lambda k: abs(finite[k]))
        return name, finite[name]

    @property
    def best_ceiling(self) -> CeilingRow | None:
        """The ceiling whose full-gate split separates forward Sharpe furthest.
        None when no ceiling puts rows on both sides."""
        usable = [c for c in self.ceilings if c.full.separation == c.full.separation]
        return max(usable, key=lambda c: c.full.separation) if usable else None


def calibrate(rows: Sequence[GateRow], ceilings: Sequence[float] = CEILINGS,
              base_ceiling: float = VALIDATION.p_oos_loss_max,
              split_no_trade: bool = True) -> Calibration:
    """Rank correlations and ceiling splits over the scored rows.

    A row whose variant never traded scores Sharpe 0.00 and would otherwise
    enter the rankings as a measured zero, so the same calibration is computed
    again over the rows that did trade — once, not recursively, since the
    traded rows all traded."""
    forward = [r.forward_sharpe for r in rows]
    traded = [r for r in rows if r.traded]
    return Calibration(
        n_rows=len(rows),
        n_traded=len(traded),
        spearman_p_oos_loss=spearman([r.p_oos_loss for r in rows], forward),
        spearman_dsr=spearman([r.dsr for r in rows], forward),
        spearman_pbo=spearman([r.pbo for r in rows], forward),
        dsr_only=split_rows(rows, lambda r: r.clears_dsr),
        ceilings=[
            CeilingRow(
                ceiling=c,
                full=split_rows(rows, lambda r, c=c: r.passes_at(c)),
                p_oos_loss_only=split_rows(rows,
                                           lambda r, c=c: r.within_ceiling(c)),
                changed=sum(1 for r in rows
                            if r.passes_at(c) != r.passes_at(base_ceiling)),
            )
            for c in ceilings
        ],
        base_ceiling=base_ceiling,
        traded_only=(calibrate(traded, ceilings, base_ceiling,
                               split_no_trade=False)
                     if split_no_trade and len(traded) != len(rows) else None),
    )


@dataclass(frozen=True)
class GateBacktest:
    rows: list[GateRow]
    skipped: list[SkippedCell]
    cutoffs: list[date]
    forward_days: int
    n_blocks: int
    n_folds: int
    calibration: Calibration
    # The feed the bars were fetched from. On the result rather than read from
    # the environment where the record is written, so a record reproduces from
    # what the run used.
    market_data_feed: str = MARKET_DATA_FEED


def run_gate_backtest(grouped: dict[str, tuple],
                      bars_cache: dict[str, pd.DataFrame],
                      cutoffs: Sequence[date],
                      forward_days: int = FORWARD_DAYS,
                      n_blocks: int = VALIDATION.n_cscv_blocks,
                      n_folds: int = VALIDATION.wf_folds) -> GateBacktest:
    """Every (cutoff, symbol) cell with bars, scored or recorded as skipped.

    One cell's failure costs that cell only: every backtest already paid for
    stays in the record rather than being thrown away over one bad instrument,
    which is how research.judge_symbols treats the same problem."""
    rows: list[GateRow] = []
    skipped: list[SkippedCell] = []
    for cutoff in cutoffs:
        for symbol, (asset_class, slippage, variants) in grouped.items():
            if symbol not in bars_cache:
                continue
            try:
                cell = evaluate_cell(symbol, asset_class, slippage, variants,
                                     bars_cache[symbol], cutoff, forward_days,
                                     n_blocks=n_blocks, n_folds=n_folds)
            except Exception as exc:
                log.error("%s @ %s: cell failed (%s); the run continues",
                          symbol, cutoff, exc)
                skipped.append(SkippedCell(cutoff, symbol,
                                           f"{type(exc).__name__}: {exc}"))
                continue
            if isinstance(cell, GateRow):
                rows.append(cell)
            else:
                skipped.append(cell)
    return GateBacktest(rows=rows, skipped=skipped, cutoffs=list(cutoffs),
                        forward_days=forward_days, n_blocks=n_blocks,
                        n_folds=n_folds, calibration=calibrate(rows))


def _row_entry(r: GateRow) -> dict:
    return {
        "cutoff": r.cutoff.isoformat(),
        "symbol": r.symbol,
        "variant": r.variant,
        "dsr": json_number(r.dsr),
        "p_oos_loss": json_number(r.p_oos_loss),
        "pbo": json_number(r.pbo),
        "oos_sharpe": json_number(r.oos_sharpe),
        "promoted": r.promoted,
        "forward_sharpe": json_number(r.forward_sharpe),
        "forward_return": json_number(r.forward_return),
        "forward_trades": r.forward_trades,
        "forward_exposure_bars": r.forward_exposure_bars,
        "forward_entry_signals": r.forward_entry_signals,
        "traded": r.traded,
        "buy_and_hold_sharpe": json_number(r.buy_and_hold_sharpe),
        "history_bars": r.history_bars,
        "forward_bars": r.forward_bars,
        "first_train_bars": r.first_train_bars,
        "warmup_flat_variants": list(r.warmup_flat_variants),
    }


def _split_entry(split: Split) -> dict:
    return {
        "n_pass": split.n_pass,
        "n_fail": split.n_fail,
        "mean_pass": json_number(split.mean_pass),
        "mean_fail": json_number(split.mean_fail),
        "separation": json_number(split.separation),
    }


def _calibration_entry(cal: Calibration) -> dict:
    return {
        "n_rows": cal.n_rows,
        "base_ceiling": cal.base_ceiling,
        "spearman": {
            "p_oos_loss": json_number(cal.spearman_p_oos_loss),
            "dsr": json_number(cal.spearman_dsr),
            "pbo": json_number(cal.spearman_pbo),
        },
        "dsr_only": _split_entry(cal.dsr_only),
        "ceilings": [{"ceiling": c.ceiling,
                      "full": _split_entry(c.full),
                      "p_oos_loss_only": _split_entry(c.p_oos_loss_only),
                      "changed": c.changed}
                     for c in cal.ceilings],
    }


def build_report(result: GateBacktest,
                 generated_at: datetime | None = None) -> dict:
    """The experiment as a record: every scored row, every skipped cell, and the
    calibration the findings are read off. Network-free — callers pass a
    computed result."""
    at = generated_at or datetime.now(timezone.utc)
    cal = result.calibration
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": at.isoformat(),
        "market_data_feed": result.market_data_feed,
        "equity_start": EQUITY_START.date().isoformat(),
        "crypto_start": CRYPTO_START.date().isoformat(),
        "cutoffs": [c.isoformat() for c in result.cutoffs],
        "forward_days": result.forward_days,
        "n_blocks": result.n_blocks,
        "n_folds": result.n_folds,
        "gate": {"dsr_min": VALIDATION.dsr_min,
                 "p_oos_loss_max": VALIDATION.p_oos_loss_max},
        "rows": [_row_entry(r) for r in result.rows],
        "skipped": [{"cutoff": s.cutoff.isoformat(), "symbol": s.symbol,
                     "reason": s.reason} for s in result.skipped],
        # Both reads, always: traded_only repeats all_rows when no row was
        # excluded, so a consumer never has to reconstruct the comparison.
        "calibration": {"all_rows": _calibration_entry(cal),
                        "traded_only": _calibration_entry(cal.traded)},
    }


def _fmt(x: float, spec: str = ".3f") -> str:
    return "n/a" if x != x else format(x, spec)  # x != x is True only for NaN


def _trade_cell(r: GateRow) -> str:
    """`trades / bars held`, with the entry count appended only when it is the
    only thing separating the row from one that never took a position — a third
    number on every row would cost more than it tells."""
    cell = f"{r.forward_trades} / {r.forward_exposure_bars}"
    if r.forward_trades or r.forward_exposure_bars or not r.forward_entry_signals:
        return cell
    entries = "entry" if r.forward_entry_signals == 1 else "entries"
    return f"{cell} / {r.forward_entry_signals} {entries}"


def print_rows(result: GateBacktest) -> None:
    header = (f"{'Cutoff':<12}{'Symbol':<9}{'Variant':<17}{'DSR':>8}"
              f"{'P(loss)':>9}{'OOS':>7}{'Gate':>7}{'Fwd':>7}"
              f"{'Trd / Held':>18}{'B&H':>7}")
    print(header)
    print("-" * len(header))
    for r in result.rows:
        print(f"{r.cutoff.isoformat():<12}{r.symbol:<9}{r.variant:<17}"
              f"{_fmt(r.dsr):>8}{_fmt(r.p_oos_loss):>9}"
              f"{_fmt(r.oos_sharpe, '.2f'):>7}"
              f"{'pass' if r.promoted else '-':>7}"
              f"{_fmt(r.forward_sharpe, '.2f'):>7}"
              f"{_trade_cell(r):>18}"
              f"{_fmt(r.buy_and_hold_sharpe, '.2f'):>7}")
    for s in result.skipped:
        print(f"{s.cutoff.isoformat():<12}{s.symbol:<9}skipped: {s.reason}")


def _spearman_summary(cal: Calibration) -> str:
    return ", ".join(f"{label} {_fmt(value, '+.2f')}"
                     for label, value in cal.panels.items())


def headline(cal: Calibration) -> str:
    """The sentence the record is read for: which criterion ranks forward
    Sharpe, and whether any ceiling below the live one separates better."""
    if not cal.n_rows:
        return ("No rows were scored, so there is nothing to calibrate against: "
                "every cell was infeasible.")
    name, rho = cal.strongest_predictor
    ranking = (f"Over {cal.n_rows} rows the gate inputs rank forward Sharpe at "
               f"{_spearman_summary(cal)} (Spearman); ")
    if name == "none":
        ranking += "none of them is computable over this run."
    elif cal.n_traded == cal.n_rows:
        ranking += (f"{name} carries the most of that ranking at "
                    f"{_fmt(rho, '+.2f')}, and every row's selected variant was "
                    f"in the market.")
    elif cal.traded_only is None:
        ranking += (f"{name} carries the most of that ranking at "
                    f"{_fmt(rho, '+.2f')} over {cal.n_rows} rows, of which "
                    f"{cal.n_traded} were in the market; this read does not "
                    f"carry the traded-only ranking.")
    else:
        ranking += (f"{name} carries the most of that ranking at "
                    f"{_fmt(rho, '+.2f')} over {cal.n_rows} rows, "
                    f"{_fmt(cal.traded.panels[name], '+.2f')} over the "
                    f"{cal.traded.n_rows} rows that traded.")

    base = next((c for c in cal.ceilings if c.ceiling == cal.base_ceiling), None)
    best = cal.best_ceiling
    if best is None or base is None or base.full.separation != base.full.separation:
        return (ranking + " No candidate ceiling puts rows on both sides of the "
                "gate, so no ceiling can be compared against another here.")
    if best.ceiling == cal.base_ceiling:
        return (ranking + f" No ceiling below the live {cal.base_ceiling} "
                f"separates passers from failers better than it does "
                f"(separation {_fmt(base.full.separation, '+.2f')} Sharpe).")
    return (ranking + f" A {best.ceiling} ceiling separates passers from failers "
            f"by {_fmt(best.full.separation, '+.2f')} Sharpe against "
            f"{_fmt(base.full.separation, '+.2f')} at the live "
            f"{cal.base_ceiling}, changing {best.changed} of {cal.n_rows} "
            f"verdicts.")


def _row_table(rows: Sequence[GateRow]) -> list[str]:
    lines = ["| cutoff | symbol | variant | DSR | P(OOS loss) | PBO | OOS | "
             "gate | forward Sharpe | forward return | fwd trades / bars held | "
             "B&H Sharpe | warm-up flat |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
             "--- | --- | --- |"]
    for r in rows:
        lines.append(
            f"| {r.cutoff.isoformat()} | {r.symbol} | {r.variant} | "
            f"{_fmt(r.dsr)} | {_fmt(r.p_oos_loss)} | {_fmt(r.pbo)} | "
            f"{_fmt(r.oos_sharpe, '.2f')} | "
            f"{'PROMOTE' if r.promoted else 'REJECT'} | "
            f"{_fmt(r.forward_sharpe, '.2f')} | "
            f"{_fmt(r.forward_return * 100, '+.1f')}% | "
            f"{_trade_cell(r)} | "
            f"{_fmt(r.buy_and_hold_sharpe, '.2f')} | "
            f"{len(r.warmup_flat_variants)} |")
    return lines


def _ceiling_table(cal: Calibration) -> list[str]:
    lines = ["| ceiling | pass | fail | mean fwd Sharpe (pass) | "
             "mean fwd Sharpe (fail) | separation | verdicts changed | "
             "P(OOS loss) alone: pass/fail | separation |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in cal.ceilings:
        lines.append(
            f"| {c.ceiling} | {c.full.n_pass} | {c.full.n_fail} | "
            f"{_fmt(c.full.mean_pass, '.2f')} | {_fmt(c.full.mean_fail, '.2f')} | "
            f"{_fmt(c.full.separation, '+.2f')} | {c.changed} | "
            f"{c.p_oos_loss_only.n_pass}/{c.p_oos_loss_only.n_fail} | "
            f"{_fmt(c.p_oos_loss_only.separation, '+.2f')} |")
    return lines


def build_markdown(result: GateBacktest, generated_at: datetime) -> str:
    """The findings in prose, for a reader who will not open the JSON."""
    cal = result.calibration
    dsr = cal.dsr_only
    lines = [
        f"# Promotion-gate calibration — {generated_at.date().isoformat()}",
        "",
        headline(cal),
        "",
        f"The gate ({VALIDATION.dsr_min} DSR floor, {VALIDATION.p_oos_loss_max} "
        f"P(OOS loss) ceiling, positive walk-forward OOS Sharpe) was replayed at "
        f"{len(result.cutoffs)} cutoffs over research's own shortlists, on bars "
        f"truncated to each cutoff and the {result.market_data_feed} feed. The "
        f"variant the gate selected was then traded over the following "
        f"{result.forward_days} days, warmed on the history the gate saw.",
        "",
        "## Rows",
        "",
    ]
    if result.rows:
        lines += _row_table(result.rows)
    else:
        lines.append("No rows were scored: every cell was infeasible.")
    lines += ["", "## Ceilings", "",
              f"Each candidate ceiling read against the live "
              f"{cal.base_ceiling}. The DSR bar alone, which no ceiling moves, "
              f"splits {dsr.n_pass}/{dsr.n_fail} at a separation of "
              f"{_fmt(dsr.separation, '+.2f')} Sharpe — the comparison that "
              f"says whether the loss-probability criterion or the DSR floor is "
              f"carrying the forward signal.",
              "",
              "The rank correlations in the headline are ceiling-independent: "
              "a ceiling moves which rows land on each side of the gate, not "
              "how the gate's inputs rank forward Sharpe.",
              ""]
    lines += _ceiling_table(cal)
    if not cal.n_rows:
        pass  # nothing was scored, so there is no second read to report
    elif cal.n_traded == cal.n_rows:
        lines += ["", "Every row's selected variant held a position in its "
                  "forward window, so the traded-only calibration is this same "
                  "read."]
    elif cal.traded_only is None:
        lines += ["", f"{cal.n_no_trade} of {cal.n_rows} rows' selected "
                  f"variants never held a position in the forward window. This "
                  f"read does not carry the traded-only calibration, so the "
                  f"tables above are the all-rows read alone."]
    elif not cal.traded.n_rows:
        lines += ["", f"No selected variant held a position in its forward "
                  f"window, so there is no traded-only calibration to read: "
                  f"all {cal.n_rows} rows score Sharpe 0.00 as absent evidence "
                  f"rather than as a measured zero."]
    else:
        traded = cal.traded
        lines += ["",
                  f"The same read with the {cal.n_no_trade} "
                  f"{'row' if cal.n_no_trade == 1 else 'rows'} whose variant "
                  f"never held a position dropped. Over the {traded.n_rows} "
                  f"rows whose selected variant was in the market, the gate "
                  f"inputs rank forward Sharpe "
                  f"at {_spearman_summary(traded)} (Spearman), and the DSR bar "
                  f"alone splits {traded.dsr_only.n_pass}/"
                  f"{traded.dsr_only.n_fail} at "
                  f"{_fmt(traded.dsr_only.separation, '+.2f')} Sharpe.",
                  ""]
        lines += _ceiling_table(traded)
    if result.skipped:
        lines += ["", "## Skipped cells", "",
                  "Not scored, and not counted as rejections: a cell with no "
                  "variant able to warm up, or no forward window to measure, "
                  "is absent evidence rather than a gate that said no.", ""]
        lines += [f"- {s.cutoff.isoformat()} {s.symbol}: {s.reason}"
                  for s in result.skipped]
    lines += [
        "",
        "## Caveat",
        "",
        f"n is {cal.n_rows} rows, and they are overlapping rather than "
        f"independent draws: consecutive cutoffs share most of their in-sample "
        f"history, and adjacent forward windows cover neighbouring years of the "
        f"same few instruments. Read this as a calibration "
        f"read, not a statistical proof — it says which direction the evidence "
        f"points, not where a ceiling belongs.",
        "",
        f"A row whose selected variant never held a position in the forward "
        f"window scores Sharpe 0.00 — a variant that never entered, not one "
        f"that traded to a flat year — and enters the rank correlations as "
        f"skill 0. {cal.n_no_trade} of {cal.n_rows} rows read that way here; "
        f"the `fwd trades / bars held` column is the count, and the "
        f"traded-only tables above are the same calibration with those rows "
        f"dropped. The two halves of that column differ: a variant still "
        f"holding when the window ended closed no round trip, and bars held "
        f"counts the bars it carried a position into its own next decision. A "
        f"variant that entered on the final bar has neither: there is no held "
        f"bar to measure and its forward Sharpe is the slippage on that one "
        f"fill, so the row is a no-trade row and the entry count is appended "
        f"to the column as disclosure.",
        "",
        "Dropping those rows conditions on a forward-window outcome — whether "
        "the variant ever took a position, which is only known after the fact "
        "— so the traded-only read is not an unbiased estimate of the same "
        "quantity as the all-rows read. It answers a narrower question: how "
        "the gate's inputs rank forward Sharpe among the cells where a "
        "position was taken.",
        "",
        "The `warm-up flat` column counts the shortlist variants whose warm-up "
        "outruns walk-forward's first train block at that cutoff (the block "
        "length is `first_train_bars` in the JSON, the labels "
        "`warmup_flat_variants`). Rows with warm-up-flat variants replay "
        "research's own behaviour, where those variants cannot be selected in "
        "the earliest training folds.",
        "",
        f"Per-row numbers: `{REPORT_PREFIX}-"
        f"{generated_at.date().isoformat()}.json`. This is an experiment "
        f"record, not a promotion artifact — the docs/validation/ verdicts are "
        f"unaffected.",
        "",
    ]
    return "\n".join(lines)


def record_paths(generated_at: datetime,
                 directory: str = RESEARCH_REPORT_DIR) -> tuple[Path, Path]:
    return optimize.grid_record_paths(generated_at, directory, REPORT_PREFIX)


def write_report(report: dict, generated_at: datetime,
                 directory: str = RESEARCH_REPORT_DIR) -> str:
    return optimize.write_grid_report(report, generated_at, directory=directory,
                                      prefix=REPORT_PREFIX)


def write_markdown(result: GateBacktest, generated_at: datetime,
                   directory: str = RESEARCH_REPORT_DIR) -> str:
    path = record_paths(generated_at, directory)[1]
    os.makedirs(directory, exist_ok=True)
    # "x": a record of a run that happened is never overwritten by a later one.
    with open(path, "x") as fh:
        fh.write(build_markdown(result, generated_at))
    return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bot.gate_backtest",
        description="Calibrate the promotion gate's P(OOS loss) ceiling against "
                    "realised forward performance.")
    parser.add_argument(
        "--cutoffs", type=parse_cutoffs, default=default_cutoffs(),
        metavar="YYYY-MM-DD,...",
        help="gate cutoff dates (default: the last trading day of "
             f"{DEFAULT_CUTOFF_YEARS[0]}-{DEFAULT_CUTOFF_YEARS[-1]})")
    parser.add_argument(
        "--symbols", type=parse_symbols, default=None, metavar="SYM,...",
        help="restrict to these symbols of research's shortlists "
             "(default: all of them)")
    parser.add_argument(
        "--forward-days", type=int, default=FORWARD_DAYS, metavar="N",
        help=f"length of the forward window in calendar days "
             f"(default: {FORWARD_DAYS})")
    return parser


def fetch_bars(md: MarketData, grouped: dict[str, tuple]
               ) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Full history per symbol, from research's own start dates, alongside
    symbol -> error for the ones that could not be fetched. A symbol absent
    from the record would otherwise read as one the gate rejected."""
    bars_cache: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for symbol, (asset_class, _slippage, _variants) in grouped.items():
        start = CRYPTO_START if asset_class == AssetClass.CRYPTO else EQUITY_START
        try:
            df = md.get_bars(symbol, asset_class, DAILY, start)
        except Exception as exc:
            log.error("%s: bars could not be fetched (%s); the run continues "
                      "without it", symbol, exc)
            errors[symbol] = f"{type(exc).__name__}: {exc}"
            continue
        print(f"{symbol}: {len(df)} daily bars, "
              f"{df.index[0].date()} -> {df.index[-1].date()}")
        bars_cache[symbol] = df
    return bars_cache, errors


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(
        list(sys.argv[1:] if argv is None else argv))
    # One timestamp for the whole run: the record's name is fixed before the
    # work starts, so the run cannot discover at the end that it has nowhere to
    # land.
    generated_at = datetime.now(timezone.utc)
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY") or None
    if not api_key:
        print("The gate calibration experiment needs Alpaca API keys.",
              file=sys.stderr)
        return 1
    if optimize.refuse_existing_record(record_paths(generated_at,
                                                    RESEARCH_REPORT_DIR)):
        return 1

    grouped = shortlists()
    if args.symbols:
        missing = [s for s in args.symbols if s not in grouped]
        if missing:
            print(f"Not in research's shortlists: {', '.join(missing)}",
                  file=sys.stderr)
            return 2
        grouped = {s: grouped[s] for s in args.symbols}

    md = MarketData(api_key, os.getenv("ALPACA_SECRET_KEY") or None,
                    feed=MARKET_DATA_FEED)
    bars_cache, fetch_errors = fetch_bars(md, grouped)
    if not bars_cache:
        print("No bars could be fetched; nothing to calibrate.", file=sys.stderr)
        return 1

    print(f"\nReplaying the gate at {len(args.cutoffs)} cutoffs over "
          f"{len(bars_cache)} symbols, then measuring the following "
          f"{args.forward_days} days.\n")
    result = run_gate_backtest(grouped, bars_cache, args.cutoffs,
                               forward_days=args.forward_days)
    unfetched = [SkippedCell(cutoff, symbol, reason)
                 for cutoff in args.cutoffs
                 for symbol, reason in fetch_errors.items()]
    result = replace(result, skipped=unfetched + result.skipped)
    print_rows(result)
    print()
    print(headline(result.calibration))

    json_path = write_report(build_report(result, generated_at), generated_at,
                             RESEARCH_REPORT_DIR)
    md_path = write_markdown(result, generated_at, RESEARCH_REPORT_DIR)
    print(f"\nExperiment recorded in {json_path} and {md_path} — an experiment "
          f"record, not a promotion artifact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
