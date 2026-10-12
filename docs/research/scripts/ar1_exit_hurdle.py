"""Kaminski & Lo (2014) exit hurdle for every enabled sleeve.

"When do stop-loss rules stop losses?" (J. Financial Markets 18) gives a
sufficient condition for a path-dependent exit rule to add value on an AR(1)
return process: rho >= SR, where rho is the lag-1 autocorrelation of the
STRATEGY's own returns and SR is that strategy's Sharpe measured at the SAME
sampling frequency as rho. Daily here, so SR_daily = mean/std of daily returns
(an annual Sharpe of 1.0 is ~0.063 daily).

Three series per sleeve:
  (i)   strategy daily returns including flat days -- the literal K&L quantity
  (ii)  in-trade days only -- the conditional autocorrelation given you hold
  (iii) the underlying's raw close-to-close returns -- reference

Everything runs through the repo's own machinery: config.enabled_instruments,
bot.strategies.build_strategy, bot.backtest.Backtester, and
bot.validation._daily_returns (imported rather than re-derived so the daily
convention is bit-identical to what the validation harness scores).

K&L's r_f is the return earned WHILE STOPPED OUT -- the policy parks in the
risk-free asset -- so pi = mu - r_f. This backtester credits nothing to idle
cash (SimAccount.equity sums cash and marks; cash never accrues) and the live
account holds idle cash at zero, so r_f = 0 is the default. --rf re-prices the
hurdle at an annual cash rate; --rf-sensitivity adds the same comparison at
further rates without disturbing the headline.

Outputs are anchored to the repo root and dated by --as-of, never by the CWD:
    op run --env-file=.env -- uv run python docs/research/scripts/ar1_exit_hurdle.py
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from bot.backtest import Backtester  # noqa: E402
from bot.data import MarketData  # noqa: E402
from bot.metrics import TRADING_DAYS_PER_YEAR  # noqa: E402
from bot.research import CRYPTO_START, DAILY, EQUITY_START  # noqa: E402
from bot.strategies import build_strategy  # noqa: E402
from bot.validation import INITIAL_EQUITY, _daily_returns  # noqa: E402
from config import AssetClass, InstrumentConfig, enabled_instruments  # noqa: E402

OUT_DIR = REPO_ROOT / "docs" / "research"
DEFAULT_AS_OF = "2026-09-13"
DEFAULT_RF_SENSITIVITY = (0.04,)
# SIP, not the live IEX feed: data.py's own guidance is that a backtest should
# query the consolidated tape. IEX daily history starts 2018-2020 per symbol.
FEED = os.getenv("MARKET_DATA_FEED", "sip")
# Fill-price match tolerance when recovering an entry bar from its fill price.
FILL_RTOL = 1e-9


def fetch(md: MarketData, cfg: InstrumentConfig) -> pd.DataFrame:
    start = CRYPTO_START if cfg.asset_class == AssetClass.CRYPTO else EQUITY_START
    return md.get_bars(cfg.symbol, cfg.asset_class, DAILY, start)


def run_sleeve(cfg: InstrumentConfig, bars: pd.DataFrame):
    """One single-instrument backtest, full window, no circuit breaker — the
    same call bot.validation._run_returns makes for the harness."""
    strategy = build_strategy(cfg)
    prepared = strategy.prepare(bars)
    bt = Backtester(initial_equity=INITIAL_EQUITY, halt_enabled=False)
    report = bt.run([(cfg, strategy, prepared)], prepared.index[0])
    return bt, report, prepared


def _entry_bar(prepared: pd.DataFrame, cfg: InstrumentConfig, direction: str,
               entry_price: float, lo: int, hi: int) -> int:
    """The bar index an entry filled on, recovered from its fill price.

    Backtester fills entries at close * (1 +/- slippage_pct), so the fill price
    identifies its own bar — but bar closes are cent-rounded and recur, so the
    match is only unique inside a window. Bounding it below by the previous
    exit (a re-entry may land on that same bar, after a hard stop fires) and
    above by this trade's exit leaves exactly one candidate: the first.
    """
    sign = 1.0 if direction == "long" else -1.0
    target = np.asarray(prepared["close"].values[lo:hi], dtype=float) * (
        1.0 + sign * cfg.slippage_pct)
    hits = np.nonzero(np.abs(target - entry_price) <= FILL_RTOL * abs(entry_price))[0]
    if len(hits) == 0:
        raise LookupError(f"{cfg.symbol}: no bar in [{lo}, {hi}) matches entry fill "
                          f"{entry_price!r}")
    return lo + int(hits[0])


def holding_bars(bt: Backtester, report, cfg: InstrumentConfig,
                 prepared: pd.DataFrame) -> list[tuple[int, int]]:
    """(entry_bar, exit_bar) index pairs for every position the sleeve took,
    including one still open at the end of the window."""
    index = prepared.index
    spans = []
    lo = 0
    for trade in report.trades:
        if trade.symbol != cfg.symbol:
            continue
        exit_i = int(index.get_loc(trade.timestamp))
        entry_i = _entry_bar(prepared, cfg, trade.direction, trade.entry_price, lo, exit_i)
        spans.append((entry_i, exit_i))
        lo = exit_i
    open_pos = bt.acct.positions.get(cfg.symbol)
    if open_pos is not None:
        entry_i = _entry_bar(prepared, cfg, open_pos.side, open_pos.entry_price,
                             lo, len(index))
        spans.append((entry_i, len(index) - 1))
    return spans


def acf1(values: np.ndarray) -> float:
    """Standard lag-1 sample autocorrelation: one mean, one denominator — the
    estimator Bartlett's 1/sqrt(n) SE and Ljung-Box are written against."""
    x = np.asarray(values, dtype=float)
    if len(x) < 3:
        return float("nan")
    d = x - x.mean()
    denom = float((d * d).sum())
    return float((d[1:] * d[:-1]).sum() / denom) if denom > 0 else float("nan")


def acf1_pooled(blocks: list[np.ndarray]) -> tuple[float, int, int]:
    """Lag-1 autocorrelation over consecutive pairs WITHIN holding episodes.

    Numerator and denominator run over the same paired observations: this is the
    sample correlation of (x_t, x_t+1) across every within-episode pair. Scaling
    the pair products by the squared deviations of observations that never enter
    a pair -- single-day episodes especially -- shrinks the estimate toward zero
    for no statistical reason. Pairs straddling two episodes are not consecutive
    days and never enter.

    Returns (rho, n_pairs, n_paired_observations).
    """
    lag, lead = [], []
    n_obs = 0
    for block in blocks:
        x = np.asarray(block, dtype=float)
        if len(x) >= 2:
            lag.append(x[:-1])
            lead.append(x[1:])
            n_obs += len(x)
    if not lag:
        return float("nan"), 0, 0
    a = np.concatenate(lag)
    b = np.concatenate(lead)
    if len(a) < 2:
        return float("nan"), len(a), n_obs
    da = a - a.mean()
    db = b - b.mean()
    denom = math.sqrt(float((da * da).sum()) * float((db * db).sum()))
    rho = float((da * db).sum() / denom) if denom > 0 else float("nan")
    return rho, len(a), n_obs


def ljung_box_q(rho: float, n: int) -> tuple[float, float]:
    """Ljung-Box Q at lag 1 and its chi2(1) p-value, in closed form. statsmodels
    is not a dependency of this repo and is deliberately not added: at one lag
    the statistic is a two-line formula and chi2(1) survival is erfc."""
    if n < 3 or math.isnan(rho):
        return float("nan"), float("nan")
    q = n * (n + 2) * rho * rho / (n - 1)
    return q, math.erfc(math.sqrt(q / 2.0))


def hurdle(rho: float, mean: float, sd: float, rf_annual: float) -> dict:
    """The K&L comparison at one cash rate: rho against pi/sigma, pi = mu - r_f.

    r_f is daily-ised by the same TRADING_DAYS_PER_YEAR bot/metrics.py annualises
    Sharpe with (252 for every instrument, crypto included), so the daily hurdle
    and the annualised figure stay on one convention.
    """
    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR
    sr = (mean - rf_daily) / sd if sd > 0 else float("nan")
    return {
        "rf_annual": rf_annual,
        "rf_daily": rf_daily,
        "sr_daily": sr,
        "sr_annual": sr * math.sqrt(TRADING_DAYS_PER_YEAR),
        "clears_hurdle": bool(rho >= sr),
        "margin_rho_minus_sr": rho - sr,
    }


def describe(values, n_pairs_override: int | None = None, rho_override: float | None = None,
             label: str = "", rf_annual: float = 0.0,
             rf_sensitivity: tuple[float, ...] = ()) -> dict:
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 3:
        return {"label": label, "n": n, "note": "too few observations"}
    sd = float(x.std(ddof=1))
    mean = float(x.mean())
    rho = acf1(x) if rho_override is None else rho_override
    n_eff = n if n_pairs_override is None else n_pairs_override + 1
    se = 1.0 / math.sqrt(n_eff)
    z = rho / se if not math.isnan(rho) else float("nan")
    q, p = ljung_box_q(rho, n_eff)
    return {
        "label": label,
        "n": n,
        "n_pairs": n - 1 if n_pairs_override is None else n_pairs_override,
        "mean_daily": mean,
        "std_daily": sd,
        "rho_hat": rho,
        "rho_se": se,
        "rho_z": z,
        "rho_sign": "positive" if rho > 0 else ("negative" if rho < 0 else "zero"),
        "rho_significant_5pct": bool(abs(z) >= 1.959963985),
        "ljung_box_q_lag1": q,
        "ljung_box_p": p,
        "sr_daily_total": mean / sd if sd > 0 else float("nan"),
        **hurdle(rho, mean, sd, rf_annual),
        "rf_sensitivity": [hurdle(rho, mean, sd, alt) for alt in rf_sensitivity],
    }


def span(index) -> dict:
    return {"start": str(pd.Timestamp(index[0]).date()),
            "end": str(pd.Timestamp(index[-1]).date())}


def analyse(cfg: InstrumentConfig, bars: pd.DataFrame, rf_annual: float = 0.0,
            rf_sensitivity: tuple[float, ...] = ()) -> dict:
    bt, report, prepared = run_sleeve(cfg, bars)
    daily = _daily_returns(report.equity_curve).dropna()

    spans = holding_bars(bt, report, cfg, prepared)
    # _daily_returns resamples to UTC day bins, so a bar maps to its floored day.
    # The entry bar is excluded from the episode: its daily return is entry
    # slippage booked before the position has seen a day of market.
    blocks = []
    for entry_i, exit_i in spans:
        days = pd.DatetimeIndex(prepared.index[entry_i + 1:exit_i + 1]).floor("D")
        block = daily.reindex(days).dropna()
        if len(block) > 0:
            blocks.append(block)
    in_trade = pd.concat(blocks).sort_index() if blocks else pd.Series(dtype=float)
    entry_days = pd.DatetimeIndex([prepared.index[i] for i, _ in spans]).floor("D")
    flat = daily.drop(in_trade.index, errors="ignore")
    untraded = flat.drop(flat.index.intersection(entry_days), errors="ignore")

    rho_pooled, n_pairs, n_paired_obs = acf1_pooled([b.to_numpy() for b in blocks])
    kw = {"rf_annual": rf_annual, "rf_sensitivity": rf_sensitivity}
    all_days = describe(daily.to_numpy(), label="strategy_all_days", **kw)
    in_trade_stats = describe(in_trade.to_numpy(), n_pairs_override=n_pairs,
                              rho_override=rho_pooled, label="strategy_in_trade", **kw)
    if in_trade_stats.get("n", 0) >= 3:
        in_trade_stats["n_paired_observations"] = n_paired_obs
        in_trade_stats["rho_naive_concat"] = acf1(in_trade.to_numpy())
    underlying = describe(bars["close"].pct_change().dropna().to_numpy(),
                          label="underlying_close_to_close", **kw)

    for stats, idx in ((all_days, daily.index), (in_trade_stats, in_trade.index),
                       (underlying, bars.index[1:])):
        if len(idx):
            stats["span"] = span(idx)

    share = len(in_trade) / len(daily) if len(daily) else float("nan")
    # Flat days contribute exact zeros to the all-days series, so conditioning on
    # holding rescales both mean and std by a known factor: series (ii) is an
    # arithmetic restatement of series (i), not independent evidence. Checked on
    # total (r_f = 0) Sharpe, because flat days earn zero and not r_f.
    sr_all = all_days.get("sr_daily_total", float("nan"))
    sr_in = in_trade_stats.get("sr_daily_total", float("nan"))
    implied = sr_all / math.sqrt(share) if share > 0 else float("nan")
    identity = {
        "sr_daily_all_days_total": sr_all,
        "share_of_days_in_trade": share,
        "sr_daily_in_trade_implied": implied,
        "sr_daily_in_trade_actual": sr_in,
        "relative_deviation": (sr_in - implied) / implied if implied else float("nan"),
        "note": ("SR_in_trade ~= SR_all_days / sqrt(share_in_trade) because flat days "
                 "are exact zeros; the residual is the entry-bar days, which carry "
                 "slippage but are excluded from the in-trade series."),
    }

    return {
        "symbol": cfg.symbol,
        "strategy": cfg.strategy.value,
        "asset_class": cfg.asset_class.value,
        "params": {k: v for k, v in cfg.params.items()},
        "slippage_pct": cfg.slippage_pct,
        "timeframe_minutes": cfg.timeframe_minutes,
        "bars": {"n": len(bars), **span(bars.index)},
        "trades": sum(1 for t in report.trades if t.symbol == cfg.symbol),
        "episodes": len(blocks),
        "position_open_at_end": cfg.symbol in bt.acct.positions,
        "days_in_trade": int(len(in_trade)),
        "days_flat": int(len(flat)),
        "share_of_days_in_trade": share,
        "mean_episode_days": float(np.mean([len(b) for b in blocks])) if blocks else 0.0,
        # Reconstruction check: a day that is neither held nor an entry bar has
        # no position and no fill, so its return must be exactly zero. Anything
        # else means a position was missed.
        "max_abs_return_on_untraded_days": (
            float(untraded.abs().max()) if len(untraded) else 0.0),
        "in_trade_sr_identity": identity,
        "series": {"strategy_all_days": all_days,
                   "strategy_in_trade": in_trade_stats,
                   "underlying_close_to_close": underlying},
    }


def jsonable(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return repr(value)


def print_table(rows: list[dict], rf_annual: float) -> None:
    print(f"\n### hurdle at r_f = {rf_annual:.2%} annual ###")
    for key, title in (("strategy_all_days", "(i) strategy daily returns, flat days included"),
                       ("strategy_in_trade", "(ii) in-trade days only"),
                       ("underlying_close_to_close", "(iii) underlying close-to-close")):
        header = (f"{'Sleeve':<10}{'n':>6}{'rho':>9}{'SE':>7}{'z':>7}"
                  f"{'SR_d':>9}{'SR_ann':>8}{'margin':>9}  verdict")
        print(f"\n== {title} ==")
        print(header)
        print("-" * len(header))
        for row in rows:
            s = row["series"][key]
            if s.get("n", 0) < 3:
                print(f"{row['symbol']:<10}{s.get('n', 0):>6}   {s.get('note', 'n/a')}")
                continue
            verdict = "CLEARS" if s["clears_hurdle"] else "fails"
            sig = "" if s["rho_significant_5pct"] else "  (rho ~ 0)"
            print(f"{row['symbol']:<10}{s['n']:>6}{s['rho_hat']:>9.4f}{s['rho_se']:>7.4f}"
                  f"{s['rho_z']:>7.2f}{s['sr_daily']:>9.4f}{s['sr_annual']:>8.2f}"
                  f"{s['margin_rho_minus_sr']:>9.4f}  {verdict}{sig}")


def print_sensitivity(rows: list[dict], rf_annual: float) -> None:
    """Series (i) re-priced at a cash rate the bot does not actually earn."""
    print(f"\n### sensitivity: series (i) at r_f = {rf_annual:.2%} annual ###")
    header = f"{'Sleeve':<10}{'rho':>9}{'SR_d':>9}{'margin':>9}  verdict"
    print(header)
    print("-" * len(header))
    for row in rows:
        s = row["series"]["strategy_all_days"]
        alt = next((a for a in s.get("rf_sensitivity", ())
                    if a["rf_annual"] == rf_annual), None)
        if alt is None:
            continue
        verdict = "CLEARS" if alt["clears_hurdle"] else "fails"
        print(f"{row['symbol']:<10}{s['rho_hat']:>9.4f}{alt['sr_daily']:>9.4f}"
              f"{alt['margin_rho_minus_sr']:>9.4f}  {verdict}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--rf", type=float, default=0.0,
        help=("annual risk-free rate earned while flat (default 0.0: the backtester "
              "credits nothing to idle cash and the live account holds it at zero)"))
    parser.add_argument(
        "--rf-sensitivity", type=float, nargs="*", default=list(DEFAULT_RF_SENSITIVITY),
        metavar="RATE",
        help="further annual rates to re-price the same rho against (default 0.04)")
    parser.add_argument(
        "--as-of", default=DEFAULT_AS_OF, metavar="YYYY-MM-DD",
        help=f"date stamp on the output file (default {DEFAULT_AS_OF})")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rf_sensitivity = tuple(args.rf_sensitivity)
    out_json = OUT_DIR / f"ar1-exit-hurdle-{args.as_of}.json"

    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY") or None
    secret = os.getenv("ALPACA_SECRET_KEY") or None
    md = MarketData(api_key, secret, feed=FEED)

    rows = []
    errors = {}
    for cfg in enabled_instruments():
        if cfg.asset_class == AssetClass.US_EQUITY and not api_key:
            errors[cfg.symbol] = "no Alpaca API key — not attempted"
            continue
        try:
            bars = fetch(md, cfg)
            print(f"{cfg.symbol}: {len(bars)} daily bars "
                  f"{bars.index[0].date()} -> {bars.index[-1].date()} ({FEED})")
            rows.append(analyse(cfg, bars, args.rf, rf_sensitivity))
        except Exception as exc:  # noqa: BLE001 — one bad symbol must not kill the run
            errors[cfg.symbol] = f"{type(exc).__name__}: {exc}"
            print(f"{cfg.symbol}: FAILED — {type(exc).__name__}: {exc}", file=sys.stderr)

    print_table(rows, args.rf)
    for alt in rf_sensitivity:
        print_sensitivity(rows, alt)
    if errors:
        print("\nerrors: " + json.dumps(errors))

    generated_at = datetime.now(timezone.utc)
    report = {
        "schema_version": 2,
        "generated_at": generated_at.isoformat(),
        "as_of": args.as_of,
        "question": ("Kaminski & Lo (2014) sufficient condition rho >= SR at a common "
                     "sampling frequency: can a path-dependent exit rule add value?"),
        "feed": FEED,
        "risk_free": {
            "annual": args.rf,
            "daily": args.rf / TRADING_DAYS_PER_YEAR,
            "basis": ("annual / TRADING_DAYS_PER_YEAR (252, the bot/metrics.py "
                      "convention for every instrument, crypto included)"),
            "sensitivity_annual": list(rf_sensitivity),
            "note": ("K&L's r_f is the return earned WHILE STOPPED OUT: their policy "
                     "switches into the risk-free asset, so pi = mu - r_f. "
                     "bot.portfolio.SimAccount.equity is cash plus marks and cash never "
                     "accrues, and the live account holds idle cash at zero, so r_f = 0 "
                     "is the rate this lineup actually earns while flat. Entries under "
                     "rf_sensitivity re-price the same rho against a cash rate the bot "
                     "does not earn; they are a what-if, not the verdict."),
        },
        "annualization": {
            "trading_days_per_year": TRADING_DAYS_PER_YEAR,
            "note": ("bot/metrics.py applies sqrt(252) to every instrument, crypto "
                     "included; followed here so the numbers are comparable with the "
                     "validation artifacts. BTC's annualized figure is therefore "
                     "understated against a sqrt(365) convention."),
        },
        "method": {
            "backtest": ("bot.backtest.Backtester, one instrument per run, full window, "
                         "halt_enabled=False — the bot.validation._run_returns path"),
            "daily_returns": "bot.validation._daily_returns (resample 1D, last, pct_change)",
            "rho": ("standard lag-1 sample autocorrelation, single mean and denominator; "
                    "SE = 1/sqrt(n), Bartlett under the white-noise null"),
            "episode_counts": ("trades counts closed round-trips; episodes counts holding "
                               "episodes, which exceeds trades by one when a position is "
                               "still open at the end of the window"),
            "in_trade_sr_identity": ("flat days are exact zeros, so SR_in_trade is "
                                     "SR_all_days / sqrt(share_in_trade) up to the "
                                     "entry-bar residual: series (ii) restates series (i) "
                                     "rather than confirming it"),
            "in_trade_definition": ("days a position was open entering the day; the entry "
                                    "bar is excluded because its return is entry slippage "
                                    "only. rho_hat is the sample correlation of (x_t, x_t+1) "
                                    "over consecutive pairs WITHIN holding episodes, with "
                                    "numerator and denominator over the same paired "
                                    "observations; rho_naive_concat is the same series "
                                    "treated as one unbroken run, for comparison."),
            "ljung_box": "closed form Q = n(n+2)rho^2/(n-1), chi2(1); statsmodels is not a dependency",
        },
        "sleeves": rows,
        "errors": errors,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n")
    print(f"\nwrote {out_json}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
