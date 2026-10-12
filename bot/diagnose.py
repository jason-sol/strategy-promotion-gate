"""Strategy diagnostics: isolated full-window runs, cost attribution, parameter
sweeps, warm-up window drift, and lookahead leakage. Read-only research tool —
trades nothing.

Usage: op run --env-file=.env -- uv run python -m bot.diagnose
       op run --env-file=.env -- uv run python -m bot.diagnose --startup-window
       ... --startup-window --fail-on-drift 1   (for cron: 1% drift exits nonzero)
       uv run python -m bot.diagnose --slippage
       uv run python -m bot.diagnose --lookahead
"""

import argparse
import math
import os
import sys
from collections import namedtuple
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from bot.backtest import INITIAL_EQUITY, Backtester
from bot.data import MarketData, calendar_days_for_bars
from bot.metrics import summarize_trades
from bot.portfolio import (
    SLIPPAGE_MIN_SAMPLE,
    assumed_slippage_bps,
    slippage_summary,
)
from bot.strategies import build_strategy
from bot.strategies.base import Action, PositionState, Strategy
from bot.strategies.mean_reversion import MeanReversion
from bot.strategies.momentum_breakout import MomentumBreakout
from bot.strategies.trend_following import TrendFollowing
from bot.validation import Candidate, deflated_sharpes
import config
from config import (
    BACKTEST_MONTHS,
    CRYPTO_TAKER_FEE_PCT,
    EXECUTION_CSV,
    MARKET_DATA_FEED,
    RISK,
    AssetClass,
    InstrumentConfig,
    enabled_instruments,
)

# Trend following signals a few times a year: 6 months is too short to judge it.
WINDOW_MONTHS = {"GLD": 30, "USO": 30}

# Width of the leading symbol column every report in this module lines up on.
SYMBOL_COL = 9


def mr(entry_z: float, sma_period: int = 20) -> MeanReversion:
    return MeanReversion(sma_period=sma_period, entry_z=entry_z, allow_short=True,
                         atr_period=RISK.atr_period)


def mb(channel: int, vol_mult: float) -> MomentumBreakout:
    return MomentumBreakout(channel_period=channel, volume_mult=vol_mult,
                            trail_atr_mult=2.0, atr_period=RISK.atr_period)


def tf(fast: int, slow: int) -> TrendFollowing:
    return TrendFollowing(fast_period=fast, slow_period=slow, trail_atr_mult=3.0,
                          atr_period=RISK.atr_period, allow_short=False)


# variants[0] is the current production configuration.
VARIANTS: dict[str, list[tuple[str, Strategy]]] = {
    "SPY": [("z=1.5 (base)", mr(1.5)), ("z=2.0", mr(2.0)), ("z=2.5", mr(2.5)),
            ("z=2.0 sma=50", mr(2.0, sma_period=50))],
    "QQQ": [("z=1.8 (base)", mr(1.8)), ("z=2.3", mr(2.3)), ("z=2.8", mr(2.8)),
            ("z=2.3 sma=50", mr(2.3, sma_period=50))],
    "BTC/USD": [("ch=20 vol=1.5 (base)", mb(20, 1.5)), ("ch=20 vol=1.0", mb(20, 1.0)),
                ("ch=55 vol=1.5", mb(55, 1.5)), ("ch=55 vol=1.0", mb(55, 1.0))],
    "GLD": [("ema 50/200 (base)", tf(50, 200)), ("ema 20/100", tf(20, 100)),
            ("ema 10/50", tf(10, 50))],
    "USO": [("ema 50/200 (base)", tf(50, 200)), ("ema 20/100", tf(20, 100)),
            ("ema 10/50", tf(10, 50))],
}


def estimated_costs(trades, cfg: InstrumentConfig) -> float:
    """Round-trip slippage (+ crypto taker fees) baked into the recorded fills."""
    fee_pct = CRYPTO_TAKER_FEE_PCT if cfg.asset_class == AssetClass.CRYPTO else 0.0
    return sum(t.qty * (t.entry_price + t.exit_price) * (cfg.slippage_pct + fee_pct)
               for t in trades)


IndicatorDrift = namedtuple("IndicatorDrift", ["column", "full", "truncated"])

# Relative: two computations of the same indicator on the same bars agree to
# floating-point noise or they disagree for a reason.
INDICATOR_DRIFT_TOLERANCE = 1e-6


def relative_drift(d: IndicatorDrift) -> float:
    """How far the min_bars value moved, as a fraction of the full-history one.

    Relative to max(1, |full|) so a column that lives near zero is not reported
    for absolute noise. A NaN on either side is unbounded: an indicator with no
    value on one of the two windows does not read differently there, it is not
    there at all, and no threshold should wave that through.
    """
    if math.isnan(d.full) or math.isnan(d.truncated):
        return math.inf
    return abs(d.truncated - d.full) / max(1.0, abs(d.full))


def sweep_history_start(cfg: InstrumentConfig, min_bars: int,
                        now: datetime | None = None) -> datetime:
    """First bar the diagnostic sweep loads for an instrument: its simulated
    window plus enough earlier history to warm every indicator."""
    months = WINDOW_MONTHS.get(cfg.symbol, BACKTEST_MONTHS)
    warmup_days = calendar_days_for_bars(min_bars + RISK.atr_period + 10,
                                         cfg.timeframe_minutes, cfg.asset_class)
    now = now or datetime.now(timezone.utc)
    return now - timedelta(days=months * 30 + warmup_days)


def indicator_drift(strategy: Strategy, bars: pd.DataFrame,
                    tolerance: float = INDICATOR_DRIFT_TOLERANCE) -> list[IndicatorDrift]:
    """Indicator columns whose value at the final bar changes when the strategy
    sees only its declared min_bars of history instead of the full series.

    `bars` is the baseline: the same history the sweep in this module backtests
    on (see sweep_history_start), so a column reported here is one the live
    engine computes differently from the run the validated numbers came from.
    """
    if len(bars) <= strategy.min_bars:
        # Truncation would be a no-op and every column would compare equal —
        # a clean bill of health nobody measured.
        raise ValueError(f"drift needs more than min_bars ({strategy.min_bars}) "
                         f"bars to truncate; got {len(bars)}")
    full = strategy.prepare(bars)
    truncated = strategy.prepare(bars.tail(strategy.min_bars))
    drifted = []
    for column in full.columns:
        if column in bars.columns:
            continue  # raw OHLCV, carried through untouched
        d = IndicatorDrift(column, float(full[column].iloc[-1]),
                           float(truncated[column].iloc[-1]))
        if math.isnan(d.full) and math.isnan(d.truncated):
            continue
        if relative_drift(d) > tolerance:
            drifted.append(d)
    return drifted


LookaheadLeak = namedtuple(
    "LookaheadLeak",
    ["bar_index", "with_future_action", "without_future_action", "leg", "seed"])

# Every sampled bar is decided once per leg. An exit rule only runs holding a
# position and the cover branch of one only holding a short, so a leg left out
# is a rule the probe never reaches.
PROBE_LEGS = ("flat", "long", "short")
LEG_LABELS = {"flat": "flat", "long": "holding long", "short": "holding short"}

# Bars sampled per strategy. Sampling bounds the cost of re-preparing the frame
# once per probed bar; a leak is only ever caught at a bar that gets probed.
LOOKAHEAD_SAMPLES = 18
# Random walks trigger entry gates rarely — an RSI(2)<10 dip or an EMA cross
# lands on a sampled bar of any single walk for barely any of the lineup — so
# the probe unions this many independent walks to reach the entry rules at all.
LOOKAHEAD_SEEDS = 20
LOOKAHEAD_REPORT_LIMIT = 5
# Decisive decisions a leg needs before a clean probe counts as reassurance.
# Measured coverage on the live lineup bottoms out at a single entry, so the
# thinness gets named while it is still above zero rather than after a
# knob-trim has taken it to nothing.
DECISIVE_FLOOR = 5
# The probe holds a position entered this many bars back, riding whatever the
# frame did since, so a trailing stop or a regime exit is live at the probed bar.
PROBE_HOLDING_BARS = 10


def synthetic_bars(n: int, seed: int = 0) -> pd.DataFrame:
    """Deterministic OHLCV on a business-day index.

    Date-indexed because calendar strategies read the bar's date rather than
    its position, and seeded so a reported leak can be reproduced exactly.
    """
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
    wick = np.abs(rng.normal(0.0, 0.004, n)) * close
    return pd.DataFrame(
        {"open": close, "high": close + wick, "low": close - wick, "close": close,
         "volume": rng.uniform(5e5, 2e6, n)},
        index=pd.date_range("2019-01-01", periods=n, freq="B", tz="UTC"))


def lookahead_indices(strategy: Strategy, bars: pd.DataFrame,
                      sample_every: int = 1) -> list[int]:
    """Bars worth probing: past the strategy's warm-up gate, and with at least
    one later bar left to withhold."""
    if sample_every < 1:
        raise ValueError(f"sample_every must be at least 1; got {sample_every}")
    first = max(strategy.min_bars - 1, 1)
    return list(range(first, len(bars) - 1, sample_every))


def _held(bars: pd.DataFrame, i: int, side: str) -> PositionState:
    """A plausible open position at bar i, built only from bars up to i. The
    high-water mark is the best price the side has seen, as the engine keeps
    it: the highest close for a long, the lowest for a short."""
    window = bars["close"].iloc[max(i - PROBE_HOLDING_BARS, 0):i + 1]
    entry = float(window.iloc[0])
    long = side == "long"
    return PositionState(symbol="PROBE", side=side, qty=1.0, entry_price=entry,
                         stop_price=entry * (0.97 if long else 1.03),
                         high_water_mark=float(window.max() if long else window.min()))


def _probe_position(leg: str, bars: pd.DataFrame, i: int) -> PositionState | None:
    return None if leg == "flat" else _held(bars, i, leg)


# One coverage list per leg: held_actions is the held-long leg, short_actions
# the held-short one.
LookaheadProbe = namedtuple(
    "LookaheadProbe", ["leaks", "flat_actions", "held_actions", "short_actions"])


def enters_short(probe: LookaheadProbe) -> bool:
    """Whether the strategy has a short path at all, judged by whether the
    probe ever saw it decide SHORT with a full frame in front of it.

    A cover rule guarding a side the strategy can never enter is unreachable:
    counting its coverage would credit a branch the engine never runs, and
    failing on a leak in it would fail the report over a trade that cannot
    happen.
    """
    return Action.SHORT in probe.flat_actions


def probe_lookahead(strategy: Strategy, bars: pd.DataFrame,
                    sample_every: int = 1, seed: int = 0) -> LookaheadProbe:
    """Bars whose signal changes once the bars after them are withheld.

    Each sampled index i is decided twice: on the full prepared frame, and on a
    frame prepared from bars[:i+1]. Truncation happens before prepare rather
    than by slicing the prepared frame, because a recursive indicator has to be
    recomputed to be comparable at all. Both runs start at bar 0 and share
    every bar up to i, so warm-up cannot explain a disagreement — the truncated
    run is missing nothing but bars the live engine would not have had either.
    A different action at i is therefore the strategy reading the future —
    given the base contract, that prepare() is deterministic and returns rows
    still aligned to the bars it was handed; one that reorders or resamples
    would make index i name two different bars and the comparison meaningless.

    Every index is probed flat, holding a long and holding a short: an exit
    rule leaks as readily as an entry rule, and a cover rule is reached by
    neither of the other two legs. Actions are compared rather than whole
    signals because the action is what the engine executes.

    The full-frame action of each leg is returned alongside the leaks. A probe
    whose every decision is HOLD compares nothing worth comparing, so a clean
    leak report is only evidence when the coverage reaches real branches — and
    reporting both from one pass keeps the two answers describing the same
    probe.

    `seed` names the walk `bars` came from and is carried onto every leak: with
    it, a reported leak replays from synthetic_bars(seed) alone.
    """
    indices = lookahead_indices(strategy, bars, sample_every)
    if not indices:
        raise ValueError(f"lookahead needs more than min_bars ({strategy.min_bars}) "
                         f"bars so a future bar can be withheld; got {len(bars)}")
    full = strategy.prepare(bars)
    leaks = []
    coverage = {leg: [] for leg in PROBE_LEGS}
    for i in indices:
        truncated = strategy.prepare(bars.iloc[:i + 1])
        if len(truncated) != i + 1 or truncated.index[i] != bars.index[i]:
            raise ValueError(f"{type(strategy).__name__}.prepare did not preserve row "
                             f"positions at index {i}: the two frames name different "
                             f"bars, so nothing here is comparable")
        for leg in PROBE_LEGS:
            # A fresh position per call: a strategy that mutates it must not
            # let one evaluation colour the other.
            actions = [strategy.signal(frame, i, _probe_position(leg, bars, i)).action
                       for frame in (full, truncated)]
            coverage[leg].append(actions[0])
            if actions[0] is not actions[1]:
                leaks.append(LookaheadLeak(i, *actions, leg=leg, seed=seed))
    return LookaheadProbe(leaks, coverage["flat"], coverage["long"], coverage["short"])


def probe_frame(strategy: Strategy, seed: int = 0,
                samples: int = LOOKAHEAD_SAMPLES) -> tuple[pd.DataFrame, int]:
    """The synthetic bars and sample stride the probe runs on."""
    bars = synthetic_bars(max(strategy.min_bars * 2, strategy.min_bars + 40), seed=seed)
    stride = math.ceil(len(lookahead_indices(strategy, bars)) / samples)
    return bars, max(stride, 1)


def pooled_lookahead_probe(strategy: Strategy, seeds: int = LOOKAHEAD_SEEDS,
                           samples: int = LOOKAHEAD_SAMPLES) -> LookaheadProbe:
    """Run the detector on synthetic bars sized for the strategy, over seeds
    0..seeds-1, pooling leaks and coverage across the union. Needs no market
    data, so leakage is checkable in CI and with the market shut."""
    pooled = LookaheadProbe([], [], [], [])
    for seed in range(seeds):
        bars, stride = probe_frame(strategy, seed, samples)
        probe = probe_lookahead(strategy, bars, sample_every=stride, seed=seed)
        for pooled_field, field in zip(pooled, probe):
            pooled_field.extend(field)
    return pooled


def decisive(actions: list[Action]) -> int:
    """Probed decisions that were not HOLD. A leg with none of these ran no
    entry or no exit rule, so nothing it reports is evidence of anything."""
    return sum(action is not Action.HOLD for action in actions)


def report_lookahead(seeds: int = LOOKAHEAD_SEEDS) -> int:
    print("Lookahead leakage: every enabled strategy re-decides sampled bars "
          "with the bars\nafter them withheld. A decision that changes is one "
          "the live engine cannot\nreproduce.\n"
          f"Seeds 0-{seeds - 1}, up to {LOOKAHEAD_SAMPLES} sampled bars each, "
          "frames of max(2x min_bars,\nmin_bars+40) bars — the same seeds give "
          "the same report.\nEach bar is decided flat, holding long and holding "
          "short. The short leg counts\nonly for a strategy the probe saw enter "
          "short: a cover rule for a side it\nnever takes is unreachable.\n")
    leaking, unchecked, vacuous, weak = 0, 0, 0, 0
    for cfg in enabled_instruments():
        # Skips are counted, never absorbed: an instrument the probe could not
        # examine is a gap in the check, and reporting one as CLEAN would be
        # the report lying. A strategy reading df.iloc[i+1] raises on the
        # truncated frame, so the loudest leaks arrive as exceptions.
        try:
            strategy = build_strategy(cfg)
        except Exception as exc:
            print(f"{cfg.symbol:<{SYMBOL_COL}}SKIP: cannot build the strategy: {exc}")
            unchecked += 1
            continue
        try:
            probe = pooled_lookahead_probe(strategy, seeds=seeds)
        except Exception as exc:
            print(f"{cfg.symbol:<{SYMBOL_COL}}SKIP: the probe raised: {exc}")
            unchecked += 1
            continue
        legs = [(decisive(probe.flat_actions), "entry"),
                (decisive(probe.held_actions), "exit")]
        leaks = probe.leaks
        if enters_short(probe):
            legs.append((decisive(probe.short_actions), "short-exit"))
        else:
            # Nothing decided SHORT, so the short leg describes a branch the
            # engine has no way into: it neither counts nor fails.
            leaks = [leak for leak in leaks if leak.leg != "short"]
        if leaks:
            leaking += 1
            print(f"{cfg.symbol:<{SYMBOL_COL}}{len(leaks)} leaking decision(s)")
            for leak in leaks[:LOOKAHEAD_REPORT_LIMIT]:
                print(f"{'':<{SYMBOL_COL}}seed {leak.seed} bar {leak.bar_index} "
                      f"({LEG_LABELS[leak.leg]}): "
                      f"{leak.with_future_action.value} with the future, "
                      f"{leak.without_future_action.value} without")
            if len(leaks) > LOOKAHEAD_REPORT_LIMIT:
                print(f"{'':<{SYMBOL_COL}}... {len(leaks) - LOOKAHEAD_REPORT_LIMIT} more")
            continue
        counts = ", ".join(f"{n} {name}" for n, name in legs)
        if any(n == 0 for n, _ in legs):
            vacuous += 1
            print(f"{cfg.symbol:<{SYMBOL_COL}}VACUOUS: {counts} "
                  "decision(s) — an unexercised rule cannot come back clean")
            continue
        if min(n for n, _ in legs) < DECISIVE_FLOOR:
            weak += 1
            print(f"{cfg.symbol:<{SYMBOL_COL}}WEAK: {counts} "
                  "decision(s) — coverage this thin is one knob-trim from vacuous")
            continue
        print(f"{cfg.symbol:<{SYMBOL_COL}}CLEAN ({counts} decision(s) probed)")
    if leaking:
        print(f"\n{leaking} instrument(s) leaking.")
    if unchecked:
        print(f"\n{unchecked} instrument(s) could not be checked.")
    if vacuous:
        print(f"\n{vacuous} instrument(s) probed nothing decisive.")
    if weak:
        # Thin coverage is not a leak and not a gap: nothing here failed, and
        # the exit code stays whatever the three real faults made it.
        print(f"\n{weak} instrument(s) barely decided anything "
              f"(fewer than {DECISIVE_FLOOR} on a leg).")
    return 1 if leaking or unchecked or vacuous else 0


def report_startup_windows(md: MarketData,
                           fail_on_drift_pct: float | None = None) -> int:
    """Warm-up drift per instrument, exiting nonzero on a gap in the coverage.

    `fail_on_drift_pct` additionally fails the report when any listed column's
    relative drift (see relative_drift, as a percentage) exceeds it, so an
    instrument is judged on its worst-drifting column and a closing line names
    every column past the threshold. Left None the exit code answers coverage
    alone: drift is then a finding for a human to read.
    """
    print("Warm-up drift: indicators computed on the sweep's own backtest "
          "history vs on\nthe last min_bars bars. Anything listed reads "
          "differently live than it does in\nthe backtest.\n")
    checked, skipped, breaches = 0, 0, []
    for cfg in enabled_instruments():
        # Staged so the line says which fault it hit: a config typo raises a
        # bare KeyError whose message is only the key name, which on its own
        # reads exactly like a data outage.
        try:
            strategy = build_strategy(cfg)
        except Exception as exc:
            print(f"{cfg.symbol:<{SYMBOL_COL}}SKIP: cannot build the strategy: {exc}")
            skipped += 1
            continue
        try:
            bars = md.get_bars(cfg.symbol, cfg.asset_class, cfg.timeframe_minutes,
                               sweep_history_start(cfg, strategy.min_bars))
            drifted = indicator_drift(strategy, bars)
        except Exception as exc:
            print(f"{cfg.symbol:<{SYMBOL_COL}}SKIP: cannot measure the drift: {exc}")
            skipped += 1
            continue
        checked += 1
        print(f"{cfg.symbol:<{SYMBOL_COL}}{len(bars)} bars fetched, "
              f"min_bars {strategy.min_bars}")
        if not drifted:
            print(f"{'':<{SYMBOL_COL}}no drift")
        for d in drifted:
            print(f"{'':<{SYMBOL_COL}}{d.column}: full {d.full:.6g} vs min_bars-only "
                  f"{d.truncated:.6g}")
            if fail_on_drift_pct is not None:
                pct = relative_drift(d) * 100
                if pct > fail_on_drift_pct:
                    breaches.append(f"{cfg.symbol} {d.column} {pct:.6g}%")
    # Coverage is always a failure: an instrument nobody measured is a hole in
    # the check, and exiting 0 hands cron the same answer as no drift anywhere.
    if not checked:
        print("\nNo instrument could be checked — nothing here was measured.")
    elif skipped:
        print(f"\n{skipped} instrument(s) could not be measured.")
    if breaches:
        print(f"\n{len(breaches)} indicator(s) drift past the "
              f"{fail_on_drift_pct:g}% threshold: {', '.join(breaches)}.")
    return 1 if skipped or not checked or breaches else 0


def report_slippage(path: str = EXECUTION_CSV) -> int:
    stats = slippage_summary(path)
    if not stats:
        print(f"No measured fills in {path} yet.")
        return 0

    header = (f"{'Symbol':<{SYMBOL_COL}}{'Kind':<7}{'N':>4}{'Drop':>6}{'Mean':>9}{'Median':>9}"
              f"{'Worst':>9}{'Spread':>9}{'Assumed':>9}  vs assumed")
    print("Realized slippage since inception, against the decision mid, in bps.\n"
          "Positive = paid the spread, negative = captured it.\n")
    print(header)
    print("-" * len(header))
    for s in stats:
        assumed = assumed_slippage_bps(s.symbol)
        assumed_col = f"{assumed:>9.2f}" if assumed is not None else f"{'n/a':>9}"
        if not s.count:
            measured, verdict = f"{'n/a':>9}" * 4, "no decision quote"
        else:
            measured = (f"{s.mean_bps:>9.2f}{s.median_bps:>9.2f}"
                        f"{s.worst_bps:>9.2f}{s.median_spread_bps:>9.2f}")
            if assumed is None:
                verdict = "not configured"
            else:
                verdict = "OVER" if s.median_bps > assumed else "under"
                if s.count < SLIPPAGE_MIN_SAMPLE:
                    verdict += f" (n<{SLIPPAGE_MIN_SAMPLE})"
        print(f"{s.symbol:<{SYMBOL_COL}}{s.kind:<7}{s.count:>4}{s.dropped:>6}{measured}"
              f"{assumed_col}  {verdict}")
    print("\nDrop counts booked fills carrying no decision quote to measure "
          "against —\nbroker stop fills always land there. Assumed is "
          "InstrumentConfig.slippage_pct\nin bps. Changing it is a decision made by "
          f"hand, not by this tool: wait for\nn>={SLIPPAGE_MIN_SAMPLE} fills on a "
          "symbol, then edit config.py and re-run the backtest\nso the validated "
          "numbers use the new cost.")
    return 0


def _drift_threshold(value: str) -> float:
    pct = float(value)
    if pct < 0:
        raise argparse.ArgumentTypeError(
            f"drift threshold cannot be negative; got {value}")
    return pct


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--startup-window", action="store_true",
                      help="report indicators that drift when the strategy "
                           "sees only min_bars of history")
    mode.add_argument("--slippage", action="store_true",
                      help="report realized fill slippage from fills.csv "
                           "against the configured slippage_pct")
    mode.add_argument("--lookahead", action="store_true",
                      help="report decisions that change when the bars after "
                           "them are withheld (lookahead leakage)")
    parser.add_argument("--fail-on-drift", type=_drift_threshold, metavar="PCT",
                        help="with --startup-window, exit nonzero when an "
                             "indicator's relative drift exceeds PCT percent, "
                             "each instrument judged on its worst-drifting "
                             "column. 0 fails on any drift the report lists at "
                             "all; without the flag drift only prints")
    args = parser.parse_args()

    # Silently ignoring it would give cron a threshold that never runs.
    if args.fail_on_drift is not None and not args.startup_window:
        parser.error("--fail-on-drift gates --startup-window, which was not asked for")

    # Reads a CSV only: no keys, no market data, so it works while execution
    # is exactly what is under suspicion.
    if args.slippage:
        return report_slippage()

    # Synthetic bars: keyless, so leakage stays checkable in CI.
    if args.lookahead:
        return report_lookahead()

    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY") or None
    secret = os.getenv("ALPACA_SECRET_KEY") or None
    md = MarketData(api_key, secret, feed=MARKET_DATA_FEED)

    if args.startup_window:
        return report_startup_windows(md, fail_on_drift_pct=args.fail_on_drift)

    header = (f"{'Symbol':<{SYMBOL_COL}}{'Variant':<22}{'Window':>7}{'Trades':>7}{'Win%':>7}{'PF':>7}"
              f"{'Net P&L':>10}{'Costs':>9}{'Gross':>10}{'MaxDD':>8}{'Sharpe':>8}{'DSR':>7}")
    print(f"Isolated runs, no circuit breaker, ${INITIAL_EQUITY:,.0f} each. "
          "Disabled instruments included for research.\n")
    print(header)
    print("-" * len(header))

    for cfg in config.INSTRUMENTS:
        if cfg.asset_class == AssetClass.US_EQUITY and not api_key:
            print(f"{cfg.symbol:<{SYMBOL_COL}}SKIP: needs API keys")
            continue
        now = datetime.now(timezone.utc)
        months = WINDOW_MONTHS.get(cfg.symbol, BACKTEST_MONTHS)
        sim_start = now - timedelta(days=months * 30)
        variants = VARIANTS.get(cfg.symbol)
        if variants is None:
            # The lineup can gain an instrument before the sweep gains variants
            # for it; that must cost this one row, not the whole diagnostic.
            print(f"{cfg.symbol:<{SYMBOL_COL}}SKIP: no sweep variants")
            continue
        max_min_bars = max(s.min_bars for _, s in variants)
        bars = md.get_bars(cfg.symbol, cfg.asset_class, cfg.timeframe_minutes,
                           sweep_history_start(cfg, max_min_bars, now))
        # DSR deflates each variant's Sharpe by the count tried for this symbol.
        # Trade from sim_start (warming up on earlier bars) so DSR and the Sharpe
        # column below describe the same return window.
        candidates = [Candidate(label, cfg, strategy) for label, strategy in variants]
        dsr = deflated_sharpes(candidates, {cfg.symbol: bars},
                               sim_start=pd.Timestamp(sim_start))
        for label, strategy in variants:
            bt = Backtester(halt_enabled=False)
            report = bt.run([(cfg, strategy, strategy.prepare(bars))],
                            pd.Timestamp(sim_start))
            s = summarize_trades([t.pnl for t in report.trades])
            costs = estimated_costs(report.trades, cfg)
            dd = report.instrument_max_drawdown(cfg.symbol)
            sharpe = report.instrument_sharpe(cfg.symbol)
            pf = f"{s.profit_factor:.2f}" if s.profit_factor != float("inf") else "inf"
            print(f"{cfg.symbol:<{SYMBOL_COL}}{label:<22}{months:>5}mo{s.total_trades:>7}"
                  f"{s.win_rate * 100:>6.1f}%{pf:>7}{s.total_pnl:>10.0f}"
                  f"{costs:>9.0f}{s.total_pnl + costs:>10.0f}{dd * 100:>7.2f}%"
                  f"{sharpe:>8.2f}{dsr[label]:>7.3f}")
        print()
    print("Gross = net P&L with slippage/fee costs added back: positive gross with "
          "negative net\nmeans the signal has edge but costs eat it; negative gross "
          "means the signal itself loses.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
