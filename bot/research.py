"""Long-window strategy research over daily bars.

Fetches the maximum history Alpaca offers and judges every variant with the
backtest-overfitting harness (bot.validation): CSCV probability of overfitting
per symbol, walk-forward out-of-sample Sharpe, and a Deflated Sharpe Ratio per
variant. A variant only deserves promotion if it clears all three.

Every run records its verdicts in docs/validation/validation-<date>.json, which is what a
strategy config cites (InstrumentConfig.validation_ref) to justify enabling an
instrument.

Usage: op run --env-file=.env -- uv run python -m bot.research
"""

import contextlib
import json
import logging
import os
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from bot.data import MarketData
from bot.strategies.base import Strategy
from bot.strategies.rsi_mean_reversion import RsiMeanReversion
from bot.strategies.trend_following import TrendFollowing
from bot.strategies.ts_momentum import TsMomentum
from bot.strategies.turn_of_month import TurnOfMonth
from bot.validation import (
    Candidate,
    build_return_matrix,
    CscvReport,
    cscv_report,
    deflated_sharpes_from_matrix,
    walk_forward,
)
from config import (
    MARKET_DATA_FEED,
    RISK,
    VALIDATION,
    VALIDATION_REPORT_DIR,
    AssetClass,
    InstrumentConfig,
    StrategyName,
)

log = logging.getLogger(__name__)

# Artifact format version. Bump when an entry's meaning changes, so a citation
# written against the old shape is not silently reread as the new one.
SCHEMA_VERSION = 2

DAILY = 1440
EQUITY_START = datetime(2016, 1, 1, tzinfo=timezone.utc)
CRYPTO_START = datetime(2020, 1, 1, tzinfo=timezone.utc)


def rsi2(entry_below: float) -> RsiMeanReversion:
    return RsiMeanReversion(entry_below=entry_below, atr_period=RISK.atr_period)


def trend(fast: int, slow: int, allow_short: bool = False) -> TrendFollowing:
    return TrendFollowing(fast_period=fast, slow_period=slow, trail_atr_mult=3.0,
                          atr_period=RISK.atr_period, allow_short=allow_short)


def tsmom(lookbacks: list[int] | None = None) -> TsMomentum:
    return TsMomentum(lookbacks=lookbacks, atr_period=RISK.atr_period)


def tom(entry_window_bdays: int, exit_day_of_month: int) -> TurnOfMonth:
    return TurnOfMonth(entry_window_bdays=entry_window_bdays,
                       exit_day_of_month=exit_day_of_month, atr_period=RISK.atr_period)


# (symbol, asset_class, slippage_pct, variant label, strategy factory)
EXPERIMENTS = [
    ("SPY", AssetClass.US_EQUITY, 0.0001, "RSI2<10 200sma", lambda: rsi2(10)),
    ("SPY", AssetClass.US_EQUITY, 0.0001, "RSI2<15 200sma", lambda: rsi2(15)),
    ("SPY", AssetClass.US_EQUITY, 0.0001, "RSI2<5 200sma", lambda: rsi2(5)),
    ("SPY", AssetClass.US_EQUITY, 0.0001, "EMA 50/200 long", lambda: trend(50, 200)),
    ("SPY", AssetClass.US_EQUITY, 0.0001, "TSMOM ens 6-12m", lambda: tsmom()),
    ("SPY", AssetClass.US_EQUITY, 0.0001, "TSMOM 12-1", lambda: tsmom([252])),
    ("QQQ", AssetClass.US_EQUITY, 0.0001, "RSI2<10 200sma", lambda: rsi2(10)),
    ("QQQ", AssetClass.US_EQUITY, 0.0001, "RSI2<15 200sma", lambda: rsi2(15)),
    ("QQQ", AssetClass.US_EQUITY, 0.0001, "EMA 50/200 long", lambda: trend(50, 200)),
    ("QQQ", AssetClass.US_EQUITY, 0.0001, "TSMOM ens 6-12m", lambda: tsmom()),
    ("BTC/USD", AssetClass.CRYPTO, 0.0005, "EMA 10/50 long", lambda: trend(10, 50)),
    ("BTC/USD", AssetClass.CRYPTO, 0.0005, "EMA 20/100 long", lambda: trend(20, 100)),
    ("BTC/USD", AssetClass.CRYPTO, 0.0005, "EMA 50/200 long", lambda: trend(50, 200)),
    ("BTC/USD", AssetClass.CRYPTO, 0.0005, "TSMOM ens 6-12m", lambda: tsmom()),
    # GLD moved 4h -> daily 2026-09-11: on the live IEX feed the 4h series is
    # missing roughly two thirds of SIP's bars, so the 4h backtest and the live
    # bot were measuring different instruments. A daily bar aggregates the whole
    # session and survives the feed difference.
    ("GLD", AssetClass.US_EQUITY, 0.0001, "EMA 50/200 long", lambda: trend(50, 200)),
    ("GLD", AssetClass.US_EQUITY, 0.0001, "EMA 20/100 long", lambda: trend(20, 100)),
    ("GLD", AssetClass.US_EQUITY, 0.0001, "EMA 10/50 long", lambda: trend(10, 50)),
    ("USO", AssetClass.US_EQUITY, 0.0005, "EMA 20/100 long", lambda: trend(20, 100)),
    ("USO", AssetClass.US_EQUITY, 0.0005, "EMA 20/100 l+s", lambda: trend(20, 100, True)),
    ("USO", AssetClass.US_EQUITY, 0.0005, "EMA 50/200 l+s", lambda: trend(50, 200, True)),
    ("USO", AssetClass.US_EQUITY, 0.0005, "TSMOM ens 6-12m", lambda: tsmom()),
    # TOM windows overlap heavily, so the variants are highly correlated and
    # CSCV's IS-winner pick is near-random: VOO's PBO reads noisy/pessimistic.
    ("VOO", AssetClass.US_EQUITY, 0.0001, "TOM 2/3", lambda: tom(2, 3)),
    ("VOO", AssetClass.US_EQUITY, 0.0001, "TOM 3/3", lambda: tom(3, 3)),
    ("VOO", AssetClass.US_EQUITY, 0.0001, "TOM 2/5", lambda: tom(2, 5)),
]


def make_cfg(symbol: str, asset_class: AssetClass, slippage: float) -> InstrumentConfig:
    return InstrumentConfig(symbol=symbol, asset_class=asset_class,
                            strategy=StrategyName.TREND_FOLLOWING,
                            timeframe_minutes=DAILY, slippage_pct=slippage)


@dataclass(frozen=True)
class VariantResult:
    label: str
    dsr: float
    oos: float  # walk-forward per-variant OOS Sharpe
    strategy: str = ""
    params: dict = field(default_factory=dict)
    params_omitted: list = field(default_factory=list)


@dataclass(frozen=True)
class SymbolValidation:
    symbol: str
    variants: list[VariantResult]
    cscv: CscvReport     # every CSCV panel over the symbol's variants
    oos_sharpe: float
    best_label: str      # best walk-forward OOS Sharpe: the variant we'd promote
    buy_and_hold: float  # full-window buy-and-hold return, for reference

    @property
    def best(self) -> VariantResult:
        return next(v for v in self.variants if v.label == self.best_label)

    @property
    def best_dsr(self) -> float:
        return self.best.dsr

    @property
    def pbo(self) -> float:
        return self.cscv.pbo

    @property
    def p_oos_loss(self) -> float:
        return self.cscv.p_oos_loss

    @property
    def promoted(self) -> bool:
        return VALIDATION.passes_validation(self.best_dsr, self.p_oos_loss,
                                            self.oos_sharpe)


def _to_jsonable(value):
    """Everything written has to survive json.dump(allow_nan=False): NaN becomes
    null (JSON has no NaN literal, and readers that accept one disagree about
    whether it means zero, missing or invalid), numpy scalars become their Python
    equivalents, and an unrecognized type becomes its repr — a field nobody
    anticipated must not cost the whole artifact."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value != value:  # NaN only
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return repr(value)


def _json_key(key):
    """The object key _to_jsonable would write for this mapping key, or None if
    the key is not a scalar.

    Mirrors _to_jsonable's `str(k)` deliberately: a collision check run against
    a different spelling would clear keys the writer then merges. Non-scalar
    keys are omitted rather than stringified, since str-of-tuple does not round
    trip to anything a reader can use."""
    scalar = key.item() if isinstance(key, np.generic) else key
    return str(key) if isinstance(scalar, (str, bool, int, float)) else None


def _is_param_value(value) -> bool:
    """Whether a strategy attribute is recordable as a parameter: a JSON scalar
    (numpy's included) or a container of them. Anything else is bulk state the
    strategy carries — PeadDrift's events table being the case in point — which
    the artifact names as omitted rather than trying to write.

    Scalar mapping keys count, since JSON object keys are strings and a reader
    loses only the key's Python type. Keys that collide once stringified do not:
    writing them would silently drop entries, and params_omitted is the honest
    place for a value the harvest cannot record in full."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_param_value(v) for v in value)
    if isinstance(value, dict):
        keys = [_json_key(k) for k in value]
        return (all(k is not None for k in keys) and len(set(keys)) == len(keys)
                and all(_is_param_value(v) for v in value.values()))
    return False


def strategy_params(strategy: Strategy) -> dict:
    """The parameters that define a variant: its public constructor attributes
    plus the hard-stop multiple, which several strategies expose only as a
    property. Private attributes are implementation, not a recorded parameter."""
    params = {key: _to_jsonable(value) for key, value in vars(strategy).items()
              if not key.startswith("_") and _is_param_value(value)}
    params["stop_atr_mult"] = strategy.stop_atr_mult
    return params


def omitted_params(strategy: Strategy) -> list[str]:
    """Public attributes strategy_params could not record. Named in the artifact
    because they were part of what was validated: a PEAD verdict computed over a
    particular events table is not a verdict over any events table."""
    return sorted(key for key, value in vars(strategy).items()
                  if not key.startswith("_") and not _is_param_value(value))


def _best_by_oos(columns: list[str], variant_oos_sharpe: dict[str, float]) -> str:
    """Label with the highest walk-forward OOS Sharpe, ties broken by column
    order. Falls back to the first column when no variant has a finite OOS
    Sharpe (e.g. bars too short for any fold)."""
    best, best_oos = columns[0], float("-inf")
    for label in columns:
        oos = variant_oos_sharpe.get(label, float("nan"))
        if oos == oos and oos > best_oos:  # oos == oos is False only for NaN
            best, best_oos = label, oos
    return best


def _buy_and_hold(bars: pd.DataFrame) -> float:
    close = bars["close"]
    return float(close.iloc[-1] / close.iloc[0] - 1) if len(close) > 1 else 0.0


def _reject_duplicate_labels(symbol: str, labels: list[str]) -> None:
    """The return matrix is keyed by label, so shortlist entries sharing one
    collapse into a single column and the artifact would then record the loser
    as never tried."""
    seen: set[str] = set()
    duplicates = sorted({label for label in labels
                         if label in seen or seen.add(label)})
    if duplicates:
        raise ValueError(f"{symbol}: duplicate shortlist labels {duplicates}")


def validate_symbol(symbol: str, asset_class: AssetClass, slippage: float,
                    variants: list[tuple[str, Callable[[], Strategy]]],
                    bars: pd.DataFrame,
                    n_blocks: int = VALIDATION.n_cscv_blocks,
                    n_folds: int = VALIDATION.wf_folds) -> SymbolValidation:
    """Run the full overfitting harness for one symbol's variants over pre-fetched
    bars. Pure and network-free: callers fetch bars, this judges them.

    Selection picks the variant with the best walk-forward OOS Sharpe. Ranking on
    OOS reintroduces mild selection bias over the OOS window itself; the
    DSR/CSCV gate is computed independently of that ranking and remains the
    guard against promoting a mined edge."""
    _reject_duplicate_labels(symbol, [label for label, _ in variants])
    cfg = make_cfg(symbol, asset_class, slippage)
    candidates = [Candidate(label, cfg, factory()) for label, factory in variants]

    matrix = build_return_matrix(candidates, {symbol: bars})
    columns = list(matrix.columns)
    cscv = cscv_report(matrix, n_blocks=n_blocks)
    wf = walk_forward(candidates, bars, n_folds=n_folds)

    dsr = deflated_sharpes_from_matrix(matrix, len(candidates))
    best_label = _best_by_oos(columns, wf.variant_oos_sharpe)
    by_label = {c.label: c.strategy for c in candidates}

    return SymbolValidation(
        symbol=symbol,
        variants=[VariantResult(label, dsr[label],
                                wf.variant_oos_sharpe.get(label, float("nan")),
                                type(by_label[label]).__name__,
                                strategy_params(by_label[label]),
                                omitted_params(by_label[label]))
                  for label in columns],
        cscv=cscv,
        oos_sharpe=wf.oos_sharpe,
        best_label=best_label,
        buy_and_hold=_buy_and_hold(bars),
    )


def _fmt(x: float, spec: str = ".3f") -> str:
    return "n/a" if x != x else format(x, spec)  # x != x is True only for NaN


def print_validation(result: SymbolValidation) -> None:
    for v in result.variants:
        mark = "  <-- best" if v.label == result.best_label else ""
        print(f"{result.symbol:<9}{v.label:<17}{_fmt(v.dsr):>8}"
              f"{_fmt(v.oos, '.2f'):>8}{mark}")
    verdict = "PROMOTE" if result.promoted else "REJECT"
    print(f"{'':<9}-> P(OOS loss) {_fmt(result.p_oos_loss)}  PBO {_fmt(result.pbo)}"
          f"  OOS-Sharpe {_fmt(result.oos_sharpe, '.2f')}"
          f"  B&H {_fmt(result.buy_and_hold * 100, '+.1f')}%"
          f"  VERDICT: {verdict} ({result.best_label})")


def judge_symbols(grouped: dict[str, tuple], bars_cache: dict[str, pd.DataFrame],
                  n_blocks: int = VALIDATION.n_cscv_blocks,
                  n_folds: int = VALIDATION.wf_folds
                  ) -> tuple[list[SymbolValidation], dict[str, str]]:
    """Judge every symbol with fetched bars, printing each verdict as it lands,
    and return the results alongside symbol -> error for the ones that failed.

    One symbol's failure costs that symbol only. Every backtest in the run has
    already been paid for by the time judging starts, so raising out of the loop
    would throw away the whole artifact over one bad instrument."""
    results: list[SymbolValidation] = []
    errors: dict[str, str] = {}
    for symbol, (asset_class, slippage, variants) in grouped.items():
        if symbol not in bars_cache:
            continue
        try:
            result = validate_symbol(symbol, asset_class, slippage, variants,
                                     bars_cache[symbol], n_blocks=n_blocks,
                                     n_folds=n_folds)
            print_validation(result)
            print()
        except Exception as exc:
            log.error("%s: validation failed (%s); the artifact will record the "
                      "failure and the run continues", symbol, exc)
            errors[symbol] = f"{type(exc).__name__}: {exc}"
            continue
        results.append(result)
    return results, errors


def _verdict_entry(r: SymbolValidation) -> dict:
    nan = float("nan")
    entry = {
        "symbol": r.symbol,
        "variant": r.best_label,
        "strategy": r.best.strategy,
        "params": _to_jsonable(r.best.params),
        "dsr": _to_jsonable(r.best_dsr),
        "oos_sharpe": _to_jsonable(r.oos_sharpe),
        "variant_oos": _to_jsonable(r.best.oos),
        "p_oos_loss": _to_jsonable(r.p_oos_loss),
        "pbo": _to_jsonable(r.pbo),
        "median_logit": _to_jsonable(r.cscv.median_logit),
        "degradation_slope": _to_jsonable(r.cscv.degradation_slope),
        "variant_oos_rank_mean": _to_jsonable(
            r.cscv.oos_rank_mean.get(r.best_label, nan)),
        "variant_is_best_share": _to_jsonable(
            r.cscv.is_best_share.get(r.best_label, nan)),
        # Every variant measured, not just the one selection kept: without the
        # field, a reader cannot tell a variant that was never tried from one
        # that was tried and lost.
        "variants": [{"variant": v.label,
                      "dsr": _to_jsonable(v.dsr),
                      "oos_sharpe": _to_jsonable(v.oos)}
                     for v in r.variants],
        "verdict": "PROMOTE" if r.promoted else "REJECT",
    }
    if r.best.params_omitted:
        entry["params_omitted"] = list(r.best.params_omitted)
    return entry


def build_validation_report(results: Iterable[SymbolValidation],
                            errors: dict[str, str] | None = None,
                            generated_at: datetime | None = None) -> dict:
    """The run's verdicts as a citable artifact: one entry per symbol, carrying
    the variant selection would promote and the same PROMOTE/REJECT gate
    print_validation reports. Network-free — callers pass computed results.

    The gate's three inputs are recorded as the gate sees them: dsr is the
    selected variant's, p_oos_loss is the symbol's CSCV probability that the
    in-sample winner loses money out of sample, and oos_sharpe is the
    walk-forward select-then-trade Sharpe — each fold's in-sample winner traded
    through the next fold, concatenated. variant_oos is the selected variant's
    own concatenated OOS Sharpe, the number the printed table's best row shows;
    the two differ whenever selection changed hands between folds.

    variants lists every variant on the symbol's shortlist in shortlist order
    with its own dsr and walk-forward OOS Sharpe — the printed table's rows. The
    selected variant appears among them; the others are what selection passed
    over, which is the difference between a variant that was never measured and
    one that was measured and lost.

    pbo, median_logit and degradation_slope are CSCV's other panels, recorded as
    advisory context rather than gate inputs, and variant_oos_rank_mean /
    variant_is_best_share are the selected variant's own standing across the
    splits: how much of the field it beats out of sample on average, and how
    often it was the in-sample winner being judged.

    Symbols in errors are recorded beside the verdicts as verdict "ERROR", so a
    partial run still says what it could not judge and one field discriminates
    every row. params_omitted names bulk state the harvest could not record but
    which the verdict was nonetheless computed over.

    A strategy config enabling an instrument should cite an entry here
    (InstrumentConfig.validation_ref), which is what makes the verdict a
    promotion gate rather than a number that scrolled past once."""
    at = generated_at or datetime.now(timezone.utc)
    entries = [_verdict_entry(r) for r in results]
    entries += [{"symbol": symbol, "verdict": "ERROR", "error": message}
                for symbol, message in (errors or {}).items()]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": at.isoformat(),
        "market_data_feed": MARKET_DATA_FEED,
        "equity_start": EQUITY_START.date().isoformat(),
        "crypto_start": CRYPTO_START.date().isoformat(),
        "results": entries,
    }


class ValidationArtifactDegraded(RuntimeError):
    """A same-day rerun would downgrade a recorded verdict."""


_VERDICT_RANK = {"PROMOTE": 2, "REJECT": 1, "ERROR": 0}


def _verdict_rank(verdict: object) -> int:
    """A verdict this code does not recognise ranks with REJECT: it is not
    evidence of a promotion, but it still says more than "could not judge"."""
    return _VERDICT_RANK.get(verdict, _VERDICT_RANK["REJECT"])


def _degraded_symbols(path: str, report: dict) -> list[tuple[str, str, str]]:
    """(symbol, recorded verdict, new verdict) for every symbol the new report
    ranks strictly below the existing artifact. Upgrades and equal verdicts are
    not degradations and are left to overwrite.

    An unreadable existing file guards nothing: refusing there would let a
    corrupt artifact block the write that would replace it."""
    try:
        with open(path) as fh:
            existing = json.load(fh)
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as exc:
        log.warning("%s could not be read (%s); overwrite guard skipped",
                    path, exc)
        return []
    recorded = {entry.get("symbol"): entry.get("verdict")
                for entry in existing.get("results", [])}
    degraded = []
    for entry in report.get("results", []):
        symbol = entry.get("symbol")
        if symbol not in recorded:
            continue
        was, now = recorded[symbol], entry.get("verdict")
        if _verdict_rank(now) < _verdict_rank(was):
            degraded.append((symbol, str(was), str(now)))
    return sorted(degraded)


def write_validation_report(report: dict, generated_at: datetime,
                            directory: str = VALIDATION_REPORT_DIR,
                            allow_degrade: bool = False) -> str:
    """Write the report as validation-<run date>.json and return its path. One
    file per run date: a re-run the same day supersedes the earlier verdict
    rather than leaving two records of the same evidence.

    Superseding may not downgrade: PROMOTE outranks REJECT outranks ERROR, and
    a rerun that drops any symbol's recorded verdict down that ranking is
    refused, because a keyless or offline rerun would otherwise silently erase
    the evidence a validation_ref cites. allow_degrade writes it anyway.

    The write goes through a sibling temp file and os.replace, so a report that
    cannot be serialised leaves the existing artifact untouched — a truncated
    one would both erase the evidence and disarm this guard, which stands down
    on a file it cannot parse."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory,
                        f"validation-{generated_at.date().isoformat()}.json")
    degraded = [] if allow_degrade else _degraded_symbols(path, report)
    if degraded:
        changes = ", ".join(f"{symbol} {was} -> {now}"
                            for symbol, was, now in degraded)
        raise ValidationArtifactDegraded(
            f"Refusing to overwrite {path}: the rerun downgrades {changes} "
            f"and the recorded verdicts are cited evidence. Fix the run (API "
            f"keys, market data access) and rerun, or delete {path} to "
            f"overwrite it deliberately.")
    fd, temp = tempfile.mkstemp(dir=directory, prefix=".validation-",
                                suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(report, fh, indent=2, allow_nan=False)
            fh.write("\n")
        os.replace(temp, path)
    finally:
        # Gone already once the replace lands; still here if anything raised.
        with contextlib.suppress(OSError):
            os.unlink(temp)
    return path


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY") or None
    secret = os.getenv("ALPACA_SECRET_KEY") or None
    if not api_key:
        print("Equity research needs Alpaca API keys.")
    md = MarketData(api_key, secret, feed=MARKET_DATA_FEED)

    # symbol -> (asset_class, slippage, [(label, factory)])
    grouped: dict[str, tuple] = {}
    for symbol, asset_class, slippage, label, factory in EXPERIMENTS:
        grouped.setdefault(symbol, (asset_class, slippage, []))[2].append((label, factory))

    bars_cache: dict[str, pd.DataFrame] = {}
    fetch_errors: dict[str, str] = {}
    for symbol, (asset_class, _slip, _variants) in grouped.items():
        if asset_class == AssetClass.US_EQUITY and not api_key:
            # Recorded, not skipped silently: a symbol absent from the artifact
            # would otherwise read as "never validated" rather than "not run".
            fetch_errors[symbol] = "no API key — not attempted"
            continue
        start = CRYPTO_START if asset_class == AssetClass.CRYPTO else EQUITY_START
        try:
            df = md.get_bars(symbol, asset_class, DAILY, start)
            print(f"{symbol}: {len(df)} daily bars, "
                  f"{df.index[0].date()} -> {df.index[-1].date()}")
        except Exception as exc:
            log.error("%s: bars could not be fetched (%s); the run continues "
                      "without it", symbol, exc)
            fetch_errors[symbol] = f"{type(exc).__name__}: {exc}"
            continue
        bars_cache[symbol] = df

    print(f"\nOverfitting harness: DSR and walk-forward OOS Sharpe per variant "
          f"(n_trials =\nvariant count), the CSCV panels per symbol. Selection "
          f"picks the best walk-forward\nOOS Sharpe; promote when DSR >= "
          f"{VALIDATION.dsr_min}, P(OOS loss) <= {VALIDATION.p_oos_loss_max}, "
          f"OOS Sharpe > 0.\nPBO is reported alongside, advisory.\n")
    header = f"{'Symbol':<9}{'Variant':<17}{'DSR':>8}{'OOS':>8}"
    print(header)
    print("-" * len(header))

    results, judge_errors = judge_symbols(grouped, bars_cache)
    generated_at = datetime.now(timezone.utc)
    report = build_validation_report(results, fetch_errors | judge_errors,
                                     generated_at=generated_at)
    try:
        path = write_validation_report(report, generated_at)
    except ValidationArtifactDegraded as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Verdicts recorded in {path} — cite an entry from it in "
          f"InstrumentConfig.validation_ref when enabling an instrument.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
