"""Single source of truth for instruments, strategy parameters, and risk limits."""

import math
import os
from dataclasses import dataclass, field
from enum import Enum


class AssetClass(Enum):
    US_EQUITY = "us_equity"
    CRYPTO = "crypto"


class StrategyName(Enum):
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM_BREAKOUT = "momentum_breakout"
    TREND_FOLLOWING = "trend_following"
    RSI_MEAN_REVERSION = "rsi_mean_reversion"
    TS_MOMENTUM = "ts_momentum"
    TURN_OF_MONTH = "turn_of_month"
    # Signal-logging-only: constructible via build_strategy but deliberately
    # absent from INSTRUMENTS (free earnings data is not point-in-time).
    PEAD_DRIFT = "pead_drift"


@dataclass(frozen=True)
class InstrumentConfig:
    symbol: str
    asset_class: AssetClass
    strategy: StrategyName
    timeframe_minutes: int
    params: dict = field(default_factory=dict)
    # Round-trip cost realism differs by liquidity: ~1bp for SPY/QQQ/GLD,
    # wider for USO and crypto.
    slippage_pct: float = 0.0005
    enabled: bool = True
    # Cites the PROMOTE verdict that justified enabling this instrument, and
    # nothing else: an entry in a docs/validation/validation-<date>.json
    # artifact once one exists, or a recorded research run for a decision that
    # predates the artifact.
    validation_ref: str | None = None
    # Records a deliberate decision to run this instrument WITHOUT a promote
    # verdict — the harness rejected it, or never covered it. It does not
    # silence the startup warning, it changes what the warning says: the fact
    # that no verdict backs this instrument stays visible either way.
    override_reason: str | None = None


INSTRUMENTS: list[InstrumentConfig] = [
    # 2026-07-21 research (bot/research.py, 2016-2026 daily data, 60/40
    # in/out-of-sample): Connors RSI(2) pullbacks above the 200-SMA held up
    # out-of-sample on SPY and QQQ (OOS Sharpe ~1.2, maxDD <2%) and were
    # robust across entry thresholds. The 15-min z-score variant they replace
    # was breakeven after costs.
    InstrumentConfig(
        symbol="SPY",
        asset_class=AssetClass.US_EQUITY,
        strategy=StrategyName.RSI_MEAN_REVERSION,
        timeframe_minutes=1440,
        params={"rsi_period": 2, "entry_below": 10, "regime_period": 200,
                "exit_period": 5, "stop_mult": 3.0, "vol_target": True},
        slippage_pct=0.0001,
        override_reason=("harness verdict REJECT (docs/validation/validation-2026-09-16.json: "
                         "SPY RSI2<10 200sma — DSR 0.941, P(OOS loss) 0.121, variant OOS 0.79, "
                         "select-then-trade OOS -0.10); kept enabled through the paper "
                         "re-measurement period (docs/limitations.md#go-live-no-go)"),
    ),
    # QQQ deliberately runs without the vol-target overlay SPY carries: the
    # promoted variant was measured without it, and a citation has to describe
    # what actually runs.
    InstrumentConfig(
        symbol="QQQ",
        asset_class=AssetClass.US_EQUITY,
        strategy=StrategyName.RSI_MEAN_REVERSION,
        timeframe_minutes=1440,
        params={"rsi_period": 2, "entry_below": 10, "regime_period": 200,
                "exit_period": 5, "stop_mult": 3.0},
        slippage_pct=0.0001,
        validation_ref=("docs/validation/validation-2026-09-16.json: QQQ RSI2<10 "
                        "200sma PROMOTE — symbol DSR 0.981, P(OOS loss) 0.023, "
                        "select-then-trade OOS 0.66, PBO 0.946 advisory; variant DSR "
                        "0.983, variant OOS 0.77 (per-variant table; SIP, 10.7y). The "
                        "walk-forward selected EMA 50/200 long (DSR 0.981, OOS 0.78) "
                        "within noise of RSI2; owner decision 2026-09-16: keep RSI2 "
                        "rather than switch strategies on a within-noise selection"),
    ),
    # Daily 20/100 EMA trend, long/flat (crypto is not shortable on Alpaca).
    # The harness's EMA variants sit within noise of one another (20/100 and
    # 50/200 are ~0.01 apart on DSR) and selection has landed on 50/200 every
    # recorded run. None clears the DSR bar.
    InstrumentConfig(
        symbol="BTC/USD",
        asset_class=AssetClass.CRYPTO,
        strategy=StrategyName.TREND_FOLLOWING,
        timeframe_minutes=1440,
        params={"fast_period": 20, "slow_period": 100, "trail_atr_mult": 3.0,
                "allow_short": False},
        override_reason=("harness verdict REJECT (docs/validation/validation-2026-09-16.json: "
                         "BTC/USD selected EMA 50/200 long — DSR 0.919, P(OOS loss) 0.287, "
                         "variant OOS 0.64, select-then-trade OOS 0.03); the lineup's EMA "
                         "20/100 is in the same shortlist and fails the same DSR bar (0.908, "
                         "own OOS 0.62, per-variant table); kept enabled through the paper "
                         "re-measurement period (docs/limitations.md#go-live-no-go)"),
    ),
    # Daily, not 4h: on the live IEX feed the 4h series carries roughly a third
    # of the bars SIP prints — missing buckets, not merely different ones — so
    # the 4h backtest and the live bot were measuring different instruments. A
    # daily bar aggregates the whole session and survives that feed difference,
    # which puts GLD in the same regime as the rest of the lineup: validated on
    # SIP history, traded on IEX dailies.
    InstrumentConfig(
        symbol="GLD",
        asset_class=AssetClass.US_EQUITY,
        strategy=StrategyName.TREND_FOLLOWING,
        timeframe_minutes=1440,
        params={"fast_period": 10, "slow_period": 50, "trail_atr_mult": 3.0, "allow_short": False},
        slippage_pct=0.0001,
        override_reason=(
            "harness verdict REJECT (docs/validation/validation-2026-09-16.json: "
            "GLD selected EMA 10/50 long — DSR 0.682, P(OOS loss) 0.050, variant "
            "OOS 0.67, select-then-trade OOS 0.52, PBO 0.082, SIP 10.7y); fails on "
            "DSR alone. Owner decision 2026-09-11: run the 10/50 variant the "
            "harness selected rather than the 50/200 it measured at DSR 0.001 / "
            "OOS -0.51, and keep it enabled through the re-measurement period "
            "(docs/limitations.md#go-live-no-go)"),
    ),
    # Daily 20/100 EMA, deliberately long-only: USO structurally decays from
    # futures roll costs, so the edge is being LONG only during sustained oil
    # uptrends and flat otherwise. Positive in both research windows
    # (2016-2022 and 2022-2026) despite buy-and-hold losing money in the first.
    InstrumentConfig(
        symbol="USO",
        asset_class=AssetClass.US_EQUITY,
        strategy=StrategyName.TREND_FOLLOWING,
        timeframe_minutes=1440,
        params={"fast_period": 20, "slow_period": 100, "trail_atr_mult": 3.0,
                "allow_short": False},
        slippage_pct=0.0005,
        override_reason=("harness verdict REJECT (docs/validation/validation-2026-09-16.json: "
                         "USO EMA 20/100 long — DSR 0.877, P(OOS loss) 0.020, variant OOS 0.67, "
                         "select-then-trade OOS 0.52); kept enabled through the paper "
                         "re-measurement period (docs/limitations.md#go-live-no-go)"),
    ),
    # Turn-of-month calendar tilt: long through the last two business days of the
    # month and the first five of the next, flat otherwise. Promoted from research
    # (bot.research VOO variants); validated without the vol-target overlay.
    # VOO tracks the same index as SPY and is deliberately outside the
    # correlation guard: blocking overlap would diverge from the validated
    # behaviour, and each leg's per-trade risk stays bounded at risk_per_trade —
    # times vol_scalar_max on a leg carrying the vol-target overlay.
    InstrumentConfig(
        symbol="VOO",
        asset_class=AssetClass.US_EQUITY,
        strategy=StrategyName.TURN_OF_MONTH,
        timeframe_minutes=1440,
        params={"entry_window_bdays": 2, "exit_day_of_month": 5},
        slippage_pct=0.0001,
        validation_ref=("docs/validation/validation-2026-09-07.json: VOO TOM 2/5 "
                        "PROMOTE — DSR 0.987, P(OOS loss) 0.000, OOS 0.69, PBO 0.407 "
                        "advisory (SIP, 10.7y); "
                        "honest-deflation caveat: DSR 0.916 as 1 of the full 35-cell "
                        "TOM grid, OOS rank 13/35 (docs/research/tom-grid-2026-09-04.md)"),
    ),
]


def enabled_instruments() -> list[InstrumentConfig]:
    """The live lineup, in config order."""
    return [cfg for cfg in INSTRUMENTS if cfg.enabled]


@dataclass(frozen=True)
class RiskConfig:
    # 1 ATR adverse move = this much of equity, times the bounded vol-target
    # scalar where an instrument opts in: the ceiling is x vol_scalar_max.
    risk_per_trade: float = 0.01
    atr_period: int = 14
    stop_atr_mult: float = 1.0          # hard stop distance in ATRs
    max_drawdown: float = 0.10          # circuit breaker: halt this far below peak equity
    max_position_notional_pct: float = 0.95
    target_vol_annual: float = 0.15     # vol-targeting: desired annualized position vol
    vol_scalar_min: float = 0.5         # clip bounds keep it from dominating ATR risk
    vol_scalar_max: float = 1.5
    # Bars of returns for the realized-vol estimate. Live parity requires the
    # fetched bar window to exceed vol_lookback + 1; an opted-in strategy whose
    # min_bars is below this would get NaN->neutral live but a real scalar in
    # backtest. Every live strategy's warm-up now clears that line by a wide
    # margin, so opting one in is a question of validated behaviour, not of
    # window arithmetic: VOO stays off the overlay pending a validation run.
    vol_lookback: int = 20


RISK = RiskConfig()


@dataclass(frozen=True)
class ExecutionConfig:
    # Marketable-limit offset: cross the spread by this many bps so entries fill
    # like a market order but cap slippage. Equity entries only.
    limit_bps: float = 10.0
    # Crypto entries rest below the signal close by this many bps to earn the
    # maker rate (~0.15% vs ~0.25% taker). Unfilled orders are cancelled and the
    # signal re-evaluated at the next bar boundary.
    crypto_maker_offset_bps: float = 5.0
    # Daily-equity signals are stable from the open, but the backtest fills at
    # the bar close — submit only in the last N minutes so live matches backtest.
    near_close_minutes: int = 10
    # No equity submission in the first N minutes after open: spreads are 4-5x
    # wider during the opening auction.
    open_auction_minutes: int = 5
    # Skip an entry when the measured quote spread exceeds this (bps). None
    # disables the gate. IEX spreads read wide, so the gate is conservative.
    max_spread_bps: float | None = None
    # Rest-then-cross ladder (opt-in). When enabled, equity entries first post a
    # passive limit at mid (minus ladder_passive_bps toward the passive side)
    # carrying the OTO stop leg, then escalate to a marketable cross on the next
    # in-window cycle if unfilled. The stop rides both legs.
    ladder_enabled: bool = False
    ladder_passive_bps: float = 0.0
    # Randomize the in-window submission instant by up to this many minutes so the
    # entry time is not perfectly predictable to other participants. 0 disables.
    # Not cryptographic: derived from a hash of (symbol, session date), so it is
    # deterministic per day and survives restarts without re-rolling.
    submit_jitter_max_minutes: float = 0.0
    # A decision bar whose close disagrees with the live quote mid by more than
    # this fraction is a data fault, not an opportunity: trading on it fills at a
    # price the signal never saw. The cycle is skipped and an alert is raised.
    max_decision_divergence_pct: float = 0.02


EXECUTION = ExecutionConfig()


@dataclass(frozen=True)
class ProtectionsConfig:
    """Two stand-downs that stop new risk before the `RISK.max_drawdown` breaker
    has to flatten: a per-symbol cooldown and an account-wide daily loss limit.
    Both default ON."""
    # Consecutive losing exits on one symbol before its entries pause. A losing
    # streak on a single instrument is that strategy out of regime, not noise —
    # stand aside rather than keep paying tuition to find out. 0 disables.
    cooldown_after_losses: int = 3
    # Calendar days a paused symbol stays out. Long enough for a daily-bar
    # strategy to skip several signals, short enough to rejoin a real trend.
    cooldown_days: int = 5
    # Loss from the UTC day's starting equity that trips HALT. A fast intraday
    # bleed should stop NEW risk well before the breaker's forced flatten, which
    # sells everything open at whatever the market is paying. HALT only: open
    # positions keep their stops and the operator decides. 0 disables.
    # Measured on mark-to-market equity, so an unrealized crypto move counts
    # against it — the exposure is real whether or not it has been booked.
    # The window is the UTC day because 00:00 UTC falls between US sessions:
    # one equity session per window, never split across two.
    daily_loss_limit_pct: float = 0.03


PROTECTIONS = ProtectionsConfig()


@dataclass(frozen=True)
class BriefingConfig:
    timezone: str = "Australia/Sydney"
    morning_hour: int = 7
    evening_hour: int = 21


BRIEFING = BriefingConfig()


@dataclass(frozen=True)
class CommandConfig:
    """Inbound email command channel. Inactive until COMMAND_SECRET and the Gmail
    credentials are all set."""
    imap_host: str = "imap.gmail.com"
    mailbox: str = "INBOX"
    # A healthy Gmail IMAP login answers in under two seconds. The rest of a
    # long timeout is not patience, it is stall: the poll is the first thing in
    # the trading cycle, so every second waited here is a second the protective
    # sweep is not running — and BTC's stop is software-supervised, so that
    # delay is its stop-loss latency.
    timeout_seconds: int = 8
    # Consecutive failed polls before the mailbox is left alone, and for how
    # long. Google throttles repeated IMAP logins, and a poll every cycle is
    # what provokes it: ~1,440 logins a day. Backing off is what ends a
    # throttle, so the wait doubles from base to cap, taking a sustained outage
    # from sixty login attempts an hour down to four. The cap stays this side of
    # an hour deliberately: past ~95% of the load removed the politeness gain is
    # marginal, while every extra minute is a minute the channel stays down
    # after Gmail has already recovered.
    breaker_threshold: int = 3
    breaker_base_seconds: int = 300
    breaker_max_seconds: int = 900
    # Downtime before the briefings carry it. Comfortably past the first few
    # backoffs, so a passing throttle stays quiet and a real outage is named.
    # The briefing leaves over Gmail SMTP, which Google meters separately from
    # IMAP — an account too throttled to poll can usually still send.
    down_alert_seconds: int = 1800
    # Replay window. A signed subject outside it is refused in both directions:
    # too old is a replayed capture, too far ahead is a clock the bot cannot
    # reason about. Wide enough to absorb mail-delivery latency.
    max_age_seconds: int = 600
    # Signatures of executed commands remembered for replay refusal. Only has to
    # outlive the window above; anything older is already refused as stale.
    processed_history: int = 50
    # Candidates one cycle will fetch and answer. Every refusal costs an SMTP
    # session, so without a cap a mailbox full of junk addressed to this channel
    # would hold the trading cycle behind it. The overflow waits for a later
    # cycle rather than being dropped.
    max_per_cycle: int = 10
    # Replies to refused mail allowed per UTC day. Only a signed subject proves
    # the sender is the owner, so refusals are the one reply a stranger can
    # drive — and they leave over the Gmail account that also carries the alerts
    # and the briefings, against a ~500/day send quota. Far below anything the
    # owner's own typos would reach.
    refusal_replies_per_day: int = 20


COMMANDS = CommandConfig()


@dataclass(frozen=True)
class ValidationConfig:
    dsr_min: float = 0.95
    p_oos_loss_max: float = 0.5
    n_cscv_blocks: int = 16
    wf_folds: int = 4

    def passes_validation(self, dsr: float, p_oos_loss: float,
                          oos_sharpe: float) -> bool:
        """The three criteria a promotion clears: DSR bounds the selection bias
        of having picked this candidate out of a field of trials, P(OOS loss) is
        how often CSCV's in-sample winner loses money out of sample — whether
        the selected configuration is more likely to lose than not — and OOS
        Sharpe is the honest walk-forward result of selecting then trading.

        CSCV's probability of backtest overfitting is reported alongside these
        but does not gate: it cannot separate a family of near-duplicate
        candidates that share a real edge from one that shares none, because in
        both the in-sample winner pick is near-random.

        A NaN P(OOS loss) is a pass-through: too few candidates to compute CSCV
        is not evidence of overfitting. NaN dsr or NaN/non-positive oos_sharpe
        fail.
        """
        return (
            dsr >= self.dsr_min
            and (math.isnan(p_oos_loss) or p_oos_loss <= self.p_oos_loss_max)
            and oos_sharpe > 0
        )


VALIDATION = ValidationConfig()

# Backtest cost model. Equities are commission-free; crypto pays taker fees.
# Slippage is per-instrument (InstrumentConfig.slippage_pct).
CRYPTO_TAKER_FEE_PCT = 0.0025
# 30 months: the active lineup is trend following, which signals a few times a
# year. Drop back toward 6 if re-enabling the 15-min mean reverters (bar volume).
BACKTEST_MONTHS = 30
BACKTEST_INITIAL_EQUITY = 100_000.0

# Market-data feed for BOTH live decisions and backtests, and they must agree:
# SIP prints the consolidated tape while IEX prints one exchange's share of it,
# so the same strategy sees different highs, lows and closes. That moves ATR,
# which moves every stop level and every position size — a backtest run on the
# other feed is not measuring the system that trades.
MARKET_DATA_FEED = os.getenv("MARKET_DATA_FEED", "iex")

TRADES_CSV = "trades.csv"
DAILY_PNL_CSV = "daily_pnl.csv"
EXECUTION_CSV = "fills.csv"
STATE_FILE = "state.json"
HALT_FILE = "HALT"
# Validation verdicts land here as validation-<date>.json, one file per research
# run. Versioned, unlike the runtime paths above: a validation_ref citing an
# entry has to still resolve in a fresh clone.
VALIDATION_REPORT_DIR = os.path.join("docs", "validation")
