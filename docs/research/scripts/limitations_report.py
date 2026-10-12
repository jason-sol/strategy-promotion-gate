"""Limitations record for a validation artifact.

Re-runs the walk-forward select-then-trade path for every symbol the artifact
judged, then states how much that evidence can carry: the Sharpe's standard
error and 95% interval, the trade count behind it, and both trial counts (the
variants the gate deflated against, and the wider search that produced the
shortlist where one was recorded). Live config and realized slippage sit beside
it so the record also says how far what runs is from what was validated.

Every number is computed or read from a file; nothing is typed in. A symbol
whose re-run misses the artifact's oos_sharpe by more than REPRODUCE_TOL is
still recorded, flagged reproducible: false with a diagnosis generated from the
re-run's own folds, and its interval and trade count are labelled as re-run
figures. Missing data or a fold replay that disagrees with walk_forward aborts.

    op run --env-file=.env -- uv run python docs/research/scripts/limitations_report.py \
        --as-of 2026-10-12
"""

import argparse
import csv
import json
import math
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from bot.data import MarketData  # noqa: E402
from bot.metrics import sharpe_ratio, sharpe_standard_error  # noqa: E402
from bot.portfolio import SLIPPAGE_MIN_SAMPLE, SlippageStat, slippage_summary  # noqa: E402
from bot.research import (  # noqa: E402
    CRYPTO_START,
    DAILY,
    EQUITY_START,
    EXPERIMENTS,
    _to_jsonable,
    make_cfg,
)
from bot.validation import Candidate, _run_backtest, walk_forward  # noqa: E402
from config import (  # noqa: E402
    EXECUTION_CSV,
    INSTRUMENTS,
    VALIDATION,
    AssetClass,
    InstrumentConfig,
    StrategyName,
)

OUT_DIR = REPO_ROOT / "docs" / "research"
VALIDATION_DIR = REPO_ROOT / "docs" / "validation"
DEFAULT_AS_OF = "2026-10-12"
Z_95 = 1.96
# Bar revisions between the artifact run and this one move the Sharpe a little;
# more than this means the re-run is not the path the artifact judged.
REPRODUCE_TOL = 0.01
# The fold re-run that counts trades must be the walk-forward's own backtest.
FOLD_REPLAY_TOL = 1e-9


@dataclass(frozen=True)
class Row:
    symbol: str
    trades: int
    sharpe: float
    sharpe_se: float
    sharpe_ci_low: float
    sharpe_ci_high: float
    gate_trials: int
    search_trials: int | None


def build_rows(returns: dict[str, pd.Series], trades: dict[str, int],
               gate_trials: dict[str, int],
               search_trials: dict[str, int | None]) -> list[Row]:
    rows = []
    for symbol, series in returns.items():
        sharpe = sharpe_ratio(series)
        se = sharpe_standard_error(series)
        rows.append(Row(symbol=symbol, trades=trades[symbol], sharpe=sharpe,
                        sharpe_se=se, sharpe_ci_low=sharpe - Z_95 * se,
                        sharpe_ci_high=sharpe + Z_95 * se,
                        gate_trials=gate_trials[symbol],
                        search_trials=search_trials.get(symbol)))
    return rows


def live_status(cfg: InstrumentConfig | None, entry: dict | None = None) -> str:
    if cfg is None:
        return "not in lineup"
    if not cfg.enabled:
        return "disabled"
    if cfg.validation_ref:
        if entry is not None and live_variant(cfg) != entry["variant"]:
            return "symbol PROMOTE; live variant not the selected one (owner decision)"
        return "PROMOTE"
    if cfg.override_reason:
        return "enabled by override"
    return "enabled without verdict"


def live_variant(cfg: InstrumentConfig | None) -> str | None:
    """The live config spelled the way bot.research labels its variants, so the
    two columns compare by eye."""
    if cfg is None:
        return None
    p = cfg.params
    try:
        if cfg.strategy == StrategyName.RSI_MEAN_REVERSION:
            return (f"RSI{p.get('rsi_period', 2)}<{p['entry_below']:g} "
                    f"{p['regime_period']}sma")
        if cfg.strategy == StrategyName.TREND_FOLLOWING:
            side = "l+s" if p.get("allow_short") else "long"
            return f"EMA {p['fast_period']}/{p['slow_period']} {side}"
        if cfg.strategy == StrategyName.TURN_OF_MONTH:
            return f"TOM {p['entry_window_bdays']}/{p['exit_day_of_month']}"
    except KeyError:
        pass
    return f"{cfg.strategy.value} {p}"


def search_trials_from(docs: Iterable[dict]) -> dict[str, int]:
    """symbol -> trials in a recorded search: trend-search files carry the count
    under exhaustive, the TOM grid at the top level."""
    out = {}
    for doc in docs:
        exhaustive = doc.get("exhaustive")
        n = exhaustive.get("n_trials") if isinstance(exhaustive, dict) else doc.get("n_trials")
        if n is not None:
            out[doc["symbol"]] = int(n)
    return out


def slippage_by_symbol(stats: Iterable[SlippageStat], symbols: Iterable[str],
                       min_sample: int = SLIPPAGE_MIN_SAMPLE) -> dict[str, str | list[dict]]:
    by_symbol: dict[str, list[SlippageStat]] = {}
    for s in stats:
        by_symbol.setdefault(s.symbol, []).append(s)
    out: dict[str, str | list[dict]] = {}
    for symbol in symbols:
        rows = by_symbol.get(symbol)
        if not rows:
            out[symbol] = "no local sample"
            continue
        n = sum(s.count for s in rows)
        if n < min_sample:
            out[symbol] = f"insufficient sample (n={n})"
            continue
        out[symbol] = [{"kind": s.kind, "count": s.count, "mean_bps": s.mean_bps,
                        "median_bps": s.median_bps, "worst_bps": s.worst_bps}
                       for s in rows]
    return out


def read_fills(path: Path) -> tuple[list[SlippageStat], int]:
    """Slippage stats and the row count behind them. slippage_summary reads a
    missing file as an empty log; here that would pass off a wrong path as "no
    fills", so it fails instead."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"fills log not found: {path}")
    with open(path, newline="") as fh:
        rows = sum(1 for _ in csv.DictReader(fh))
    return slippage_summary(str(path)), rows


def reproduction_mismatch(symbol: str, computed: float, recorded: float,
                          tol: float = REPRODUCE_TOL) -> str | None:
    if math.isfinite(computed) and abs(computed - recorded) <= tol:
        return None
    return (f"{symbol}: re-run oos_sharpe {computed:.3f} vs artifact {recorded:.3f} "
            f"(tolerance {tol})")


UNDETERMINED_CAUSE = (
    "cause not determined — the artifact records no code SHA or bar checksum; the only "
    "bot/validation.py change since (26fcdd3) moves code without changing behaviour; "
    "revised SIP bars is the leading hypothesis")


@dataclass(frozen=True)
class FoldDiag:
    fold: int
    in_sample: dict[str, float]  # label -> in-sample Sharpe the selection ranked
    winner: str
    winner_in_sample_trades: int


def describe_mismatch(folds: list[FoldDiag], variant_oos_rerun: dict[str, float],
                      variant_oos_artifact: dict[str, float],
                      tol: float = REPRODUCE_TOL) -> str:
    """What in the re-run makes the select-then-trade path fragile, stated from
    the re-run's own numbers."""
    parts = []
    for f in folds:
        best = max(f.in_sample.values())
        tied = [label for label, v in f.in_sample.items() if v == best]
        if len(tied) > 1:
            parts.append(f"fold {f.fold}: exact in-sample tie between "
                         f"{' and '.join(tied)} ({best:.4f}); selection kept {f.winner}")
        if f.winner_in_sample_trades == 0:
            parts.append(f"fold {f.fold}: winner {f.winner} made no in-sample trades "
                         f"(selected on in-sample Sharpe {f.in_sample[f.winner]:.4f})")
    moved, stable = [], []
    for label, rerun in variant_oos_rerun.items():
        recorded = variant_oos_artifact.get(label)
        if recorded is None:
            continue
        if abs(rerun - recorded) > tol:
            moved.append(f"{label} {recorded:.3f} -> {rerun:.3f}")
        else:
            stable.append(label)
    if moved:
        parts.append("variant OOS moved between runs: " + ", ".join(moved))
    if stable:
        parts.append(f"variant OOS within {tol}: " + ", ".join(stable))
    parts.append(UNDETERMINED_CAUSE)
    return "; ".join(parts) + "."


def symbol_record(row: Row, entry: dict, cfg: InstrumentConfig | None,
                  slippage, reproducible: bool, note: str | None = None) -> dict:
    """One symbol's record. Interval and trade count always come from the re-run
    series and are filed under `rerun`, so a non-reproducible row can never read
    as the artifact's own evidence."""
    return {
        "symbol": row.symbol,
        "validated_variant": entry["variant"],
        "verdict": entry["verdict"],
        "reproducible": reproducible,
        "oos_sharpe_artifact": entry["oos_sharpe"],
        "oos_sharpe_rerun": row.sharpe,
        "reproduction_note": None if reproducible else note,
        "rerun": {"sharpe": row.sharpe, "sharpe_se": row.sharpe_se,
                  "sharpe_ci_low": row.sharpe_ci_low,
                  "sharpe_ci_high": row.sharpe_ci_high, "trades": row.trades},
        "variant_oos": entry["variant_oos"],
        "dsr": entry["dsr"],
        "gate_trials": row.gate_trials,
        "search_trials": row.search_trials,
        "live_variant": live_variant(cfg),
        "live_status": live_status(cfg, entry),
        "slippage": slippage,
    }


def _fmt(x, spec: str = ".2f") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return format(x, spec)


def _slippage_cell(value) -> str:
    if isinstance(value, str):
        return value
    return "; ".join(f"{s['kind']} n={s['count']} median {_fmt(s['median_bps'], '.1f')} bps"
                     for s in value)


def _fills_line(source: dict | None) -> list[str]:
    if not source:
        return []
    line = f"Fills: {source['file']} ({source['rows']} rows)"
    return [f"{line}: {source['note']}" if source.get("note") else line, ""]


def render_markdown(record: dict) -> str:
    tol = record.get("reproduce_tolerance", REPRODUCE_TOL)
    lines = [
        f"# Limitations record, as of {record['as_of']} "
        f"(artifact `{record['artifact']}`)",
        "",
        "Generated by `docs/research/scripts/limitations_report.py` from the JSON beside "
        "this file; do not edit by hand.",
        "",
        *_fills_line(record.get("fills_source")),
        "| Symbol | Validated variant | Verdict | Live variant | Live status "
        "| OOS Sharpe (artifact) | OOS Sharpe (re-run) | SE (re-run) | 95% CI (re-run) "
        "| Variant OOS | DSR | OOS trades (re-run) | Gate trials | Search trials "
        "| Live slippage |",
        "|---|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    footnotes = []
    for s in record["symbols"]:
        r = s["rerun"]
        symbol = s["symbol"]
        if not s["reproducible"]:
            footnotes.append(
                f"[^{len(footnotes) + 1}]: not reproducible within {tol}: artifact "
                f"oos_sharpe {_fmt(s['oos_sharpe_artifact'], '.3f')}, re-run "
                f"{_fmt(s['oos_sharpe_rerun'], '.3f')}. Re-run columns on this row "
                f"describe the re-run, not the artifact. {s['reproduction_note']}")
            symbol = f"{symbol} [^{len(footnotes)}]"
        ci = f"[{_fmt(r['sharpe_ci_low'])}, {_fmt(r['sharpe_ci_high'])}]"
        search = "not recorded" if s["search_trials"] is None else str(s["search_trials"])
        lines.append(
            f"| {symbol} | {s['validated_variant']} | {s['verdict']} "
            f"| {s['live_variant'] or 'n/a'} | {s['live_status']} "
            f"| {_fmt(s['oos_sharpe_artifact'])} | {_fmt(r['sharpe'])} "
            f"| {_fmt(r['sharpe_se'])} | {ci} "
            f"| {_fmt(s['variant_oos'])} | {_fmt(s['dsr'], '.3f')} | {r['trades']} "
            f"| {s['gate_trials']} | {search} | {_slippage_cell(s['slippage'])} |")
    if footnotes:
        lines += [""] + footnotes
    notes = record.get("notes", {})
    if notes:
        lines += ["", "## Definitions", ""]
        lines += [f"- **{key}**: {text}" for key, text in notes.items()]
    return "\n".join(lines) + "\n"


NOTES = {
    "OOS Sharpe": ("walk-forward select-then-trade Sharpe, the number the gate judges: "
                   "'artifact' as recorded, 're-run' recomputed here from today's bars; a "
                   f"row whose two differ by more than {REPRODUCE_TOL} is footnoted as not "
                   "reproducible. Annualized with sqrt(252) for every symbol, crypto "
                   "included"),
    "SE / 95% CI": ("bot.metrics.sharpe_standard_error (Mertens 2002, non-normal "
                    f"returns) on the re-run daily series; CI = re-run Sharpe +/- {Z_95} SE"),
    "Variant OOS": ("the validated variant's own concatenated OOS Sharpe (artifact); "
                    "differs from the gate number whenever fold selection changed hands"),
    "DSR": "the validated variant's Deflated Sharpe Ratio (artifact)",
    "OOS trades": ("closed round-trips on the select-then-trade path: each fold's "
                   "in-sample winner re-run over its test window; a position still open "
                   "at a fold's end is not counted"),
    "Gate trials": "shortlist size the DSR deflated against (len(candidates))",
    "Search trials": ("configurations tried in a recorded search that produced the "
                      "shortlist; 'not recorded' means no search file exists, not zero"),
    "Live slippage": (f"bot.portfolio.slippage_summary over the fills log; below "
                      f"{SLIPPAGE_MIN_SAMPLE} measured fills per symbol it is reported "
                      "as a sample size only"),
}


def _dated(path: Path) -> str | None:
    """ISO date at the end of a file stem, or None when there is none."""
    stamp = "-".join(path.stem.split("-")[-3:])
    try:
        return date.fromisoformat(stamp).isoformat()
    except ValueError:
        return None


def _dated_files(directory: Path, pattern: str, as_of: str) -> list[tuple[str, Path]]:
    dated = ((_dated(p), p) for p in directory.glob(pattern))
    return sorted((d, p) for d, p in dated if d is not None and d <= as_of)


def _latest(pattern: str, directory: Path, as_of: str) -> Path | None:
    files = _dated_files(directory, pattern, as_of)
    return files[-1][1] if files else None


def load_search_docs(directory: Path, as_of: str) -> tuple[list[dict], list[str]]:
    """Every search record dated on or before as_of, oldest first, so
    search_trials_from keeps the latest count per symbol."""
    files = sorted(_dated_files(directory, "trend-search-*.json", as_of)
                   + _dated_files(directory, "tom-grid-*.json", as_of))
    docs = [json.loads(p.read_text()) for _, p in files]
    return docs, [p.name for _, p in files]


def _shortlists() -> dict[str, tuple[AssetClass, float, list]]:
    grouped: dict[str, tuple] = {}
    for symbol, asset_class, slippage, label, factory in EXPERIMENTS:
        grouped.setdefault(symbol, (asset_class, slippage, []))[2].append((label, factory))
    return grouped


def _oos_trades(candidates: list[Candidate], bars: pd.DataFrame, wf) -> int:
    """Trades on the select-then-trade path, replaying each fold winner exactly as
    walk_forward ran it and confirming the replay reproduces the fold's Sharpe."""
    by_label = {c.label: c for c in candidates}
    total = 0
    for fold in wf.folds:
        test_end_i = bars.index.get_loc(fold.test_end) + 1
        c = by_label[fold.chosen_label]
        returns, report = _run_backtest(c, bars.iloc[:test_end_i], fold.test_start)
        replay = sharpe_ratio(returns.dropna())
        same = (math.isnan(replay) and math.isnan(fold.oos_sharpe)) or \
            abs(replay - fold.oos_sharpe) <= FOLD_REPLAY_TOL
        if not same:
            raise RuntimeError(f"{c.cfg.symbol}: fold replay Sharpe {replay} != "
                               f"walk-forward {fold.oos_sharpe}")
        total += report.instrument_stats(c.cfg.symbol).total_trades
    return total


def _fold_diags(candidates: list[Candidate], bars: pd.DataFrame, wf) -> list[FoldDiag]:
    """Every candidate's in-sample Sharpe per fold, re-run on walk_forward's own
    training slice; the winner must come out the same or the diagnosis is of a
    different path."""
    diags = []
    for i, fold in enumerate(wf.folds):
        train = bars.iloc[:bars.index.get_loc(fold.test_start)]
        runs = {c.label: _run_backtest(c, train, fold.train_start) for c in candidates}
        in_sample = {label: sharpe_ratio(r) for label, (r, _) in runs.items()}
        winner = max(candidates, key=lambda c: in_sample[c.label]).label
        if winner != fold.chosen_label:
            raise RuntimeError(f"{candidates[0].cfg.symbol}: fold {i} replay picked "
                               f"{winner}, walk-forward picked {fold.chosen_label}")
        trades = runs[winner][1].instrument_stats(candidates[0].cfg.symbol).total_trades
        diags.append(FoldDiag(i, in_sample, winner, trades))
    return diags


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--as-of", default=DEFAULT_AS_OF, help="date stamped on the outputs")
    ap.add_argument("--artifact", default=None,
                    help="validation artifact (default: latest dated on or before --as-of)")
    ap.add_argument("--fills", default=str(REPO_ROOT / EXECUTION_CSV),
                    help="execution log to summarize slippage from")
    ap.add_argument("--fills-note", default=None,
                    help="provenance of the fills log, recorded beside it")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifact_path = (Path(args.artifact).resolve() if args.artifact
                     else _latest("validation-*.json", VALIDATION_DIR, args.as_of))
    if artifact_path is None:
        print("no validation artifact found", file=sys.stderr)
        return 1
    artifact = json.loads(artifact_path.read_text())
    generated_at = datetime.fromisoformat(artifact["generated_at"])
    results = [r for r in artifact["results"] if r.get("verdict") != "ERROR"]

    fills_path = Path(args.fills)
    try:
        fill_stats, fill_rows = read_fills(fills_path)
    except FileNotFoundError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 1

    from dotenv import load_dotenv
    load_dotenv()
    md = MarketData(os.getenv("ALPACA_API_KEY") or None,
                    os.getenv("ALPACA_SECRET_KEY") or None,
                    feed=artifact["market_data_feed"])
    shortlists = _shortlists()

    returns, trades, gate_trials, notes, reproducible = {}, {}, {}, {}, {}
    for entry in results:
        symbol = entry["symbol"]
        asset_class, slippage, variants = shortlists[symbol]
        labels = [label for label, _ in variants]
        recorded_labels = [v["variant"] for v in entry["variants"]]
        if labels != recorded_labels:
            print(f"{symbol}: EXPERIMENTS shortlist {labels} no longer matches the "
                  f"artifact's {recorded_labels}", file=sys.stderr)
            return 1
        cfg = make_cfg(symbol, asset_class, slippage)
        candidates = [Candidate(label, cfg, factory()) for label, factory in variants]
        start = CRYPTO_START if asset_class == AssetClass.CRYPTO else EQUITY_START
        bars = md.get_bars(symbol, asset_class, DAILY, start, end=generated_at)
        bars = bars[bars.index < generated_at]
        if bars.empty:
            print(f"{symbol}: no bars before {generated_at}", file=sys.stderr)
            return 1
        print(f"{symbol}: {len(bars)} daily bars {bars.index[0].date()} -> "
              f"{bars.index[-1].date()}")
        wf = walk_forward(candidates, bars, n_folds=VALIDATION.wf_folds)
        returns[symbol] = wf.oos_returns
        trades[symbol] = _oos_trades(candidates, bars, wf)
        gate_trials[symbol] = len(candidates)
        rerun = sharpe_ratio(wf.oos_returns)
        problem = reproduction_mismatch(symbol, rerun, entry["oos_sharpe"])
        print(problem or f"{symbol}: reproduced, re-run {rerun:.4f} vs artifact "
                         f"{entry['oos_sharpe']:.4f}")
        reproducible[symbol] = problem is None
        if problem:
            notes[symbol] = describe_mismatch(
                _fold_diags(candidates, bars, wf), wf.variant_oos_sharpe,
                {v["variant"]: v["oos_sharpe"] for v in entry["variants"]})

    search_docs, search_sources = load_search_docs(OUT_DIR, args.as_of)
    search_trials = search_trials_from(search_docs)
    rows = build_rows(returns, trades, gate_trials, search_trials)
    live = {c.symbol: c for c in INSTRUMENTS}
    slippage = slippage_by_symbol(fill_stats, [e["symbol"] for e in results])
    symbols = [symbol_record(row, entry, live.get(row.symbol), slippage[row.symbol],
                             reproducible[row.symbol], notes.get(row.symbol))
               for row, entry in zip(rows, results, strict=True)]

    record = {
        "schema_version": 1,
        "as_of": args.as_of,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact": str(artifact_path.relative_to(REPO_ROOT)),
        "artifact_generated_at": artifact["generated_at"],
        "market_data_feed": artifact["market_data_feed"],
        "bars_truncated_before": generated_at.isoformat(),
        "reproduce_tolerance": REPRODUCE_TOL,
        "wf_folds": VALIDATION.wf_folds,
        "search_sources": search_sources,
        "fills_source": {"file": fills_path.name, "rows": fill_rows,
                         "note": args.fills_note},
        "slippage_min_sample": SLIPPAGE_MIN_SAMPLE,
        "notes": NOTES,
        "symbols": symbols,
    }
    record = _to_jsonable(record)
    out_json = OUT_DIR / f"limitations-{args.as_of}.json"
    out_md = OUT_DIR / f"limitations-{args.as_of}.md"
    with open(out_json, "w") as fh:
        json.dump(record, fh, indent=2, allow_nan=False)
        fh.write("\n")
    out_md.write_text(render_markdown(record))
    print(f"wrote {out_json.relative_to(REPO_ROOT)} and {out_md.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
