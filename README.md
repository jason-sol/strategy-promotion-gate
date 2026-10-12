# strategy-promotion-gate

A strategy-promotion gate for a paper-trading system: Deflated Sharpe, CSCV/PBO,
walk-forward OOS, and the strategies it rejected.

The bot trades an Alpaca paper account only, and nothing in this repo is a claim
about returns.

- An instrument is promoted only if its selected variant clears DSR ≥ 0.95,
  P(OOS loss) ≤ 0.5 and walk-forward OOS Sharpe > 0. Each research run writes a
  dated artifact to `docs/validation/`, and the artifacts are committed, so every
  verdict below can be checked in a fresh clone.
- PBO is reported but does not gate. On 30 synthetic near-duplicate variants,
  CSCV's PBO reads 0.491 when the family has a true Sharpe of 1.0 and 0.495 when
  it has none ([probe](docs/research/pbo-family-probe-2026-09-07.md)), so it
  says nothing about overlapping parameter grids. On a mixed shortlist, where
  the variants differ in true skill, the same probe found PBO informative, and
  one promotion (QQQ) depends on ignoring it there.
- Three ideas got a NO-GO. Copying Form 4 insider purchases: the published edge
  vanishes at retail-tradeable size
  ([assessment](docs/research/2026-07-26-insider-following-viability.md)).
  Profit-target and scale-out exits: no sleeve clears the Kaminski–Lo AR(1)
  hurdle, so none got a validation trial
  ([note](docs/research/ar1-exit-hurdle-2026-09-13.md)). Real money: exit and
  kill-switch failures, since fixed; a clean 8–12-week paper run is
  outstanding ([go-live](docs/limitations.md#go-live-no-go)).

Two of six instruments pass the gate; the other four run on paper by logged
owner override.

[What the evidence can and cannot show](docs/limitations.md): trade counts,
Sharpe confidence intervals, what DSR deflates for, reproducibility, costs. The
same note covers the data span behind each verdict and how the strategies
compare with buy-and-hold.

## Lineup and verdicts

The bot trades six instruments on daily bars with three strategy families:
Connors RSI(2) mean reversion, EMA-crossover trend following and turn-of-month.
Every trade risks about 1% of equity at its ATR stop, with a correlation filter
and a drawdown circuit breaker (`RISK.max_drawdown`, 10% by default) on top.

| Symbol | Validated variant | DSR | Select-then-trade OOS Sharpe (95% CI) | Variant OOS | Verdict | Live variant | Live status |
|---|---|---:|---|---:|---|---|---|
| SPY | RSI2<10 200sma | 0.941 | -0.10 (artifact); re-run 0.13 [-0.55, 0.81] [^spy] | 0.79 | REJECT | RSI2<10 200sma | enabled by override |
| QQQ [^qqq] | EMA 50/200 long | 0.981 | 0.66 [-0.03, 1.35] | 0.78 | PROMOTE | RSI2<10 200sma | symbol PROMOTE; live variant not the selected one (owner decision) |
| BTC/USD | EMA 50/200 long | 0.919 | 0.03 [-0.73, 0.80] | 0.64 | REJECT | EMA 20/100 long | enabled by override |
| GLD | EMA 10/50 long | 0.682 | 0.52 [-0.16, 1.20] | 0.67 | REJECT | EMA 10/50 long | enabled by override |
| USO | EMA 20/100 long | 0.877 | 0.52 [-0.14, 1.19] | 0.67 | REJECT | EMA 20/100 long | enabled by override |
| VOO | TOM 2/5 | 0.986 | 0.69 [0.00, 1.37] | 0.76 | PROMOTE | TOM 2/5 | PROMOTE |

Source: [`docs/validation/validation-2026-09-16.json`](docs/validation/validation-2026-09-16.json)
(verdicts, DSR, OOS Sharpe) and
[`docs/research/limitations-2026-10-12.json`](docs/research/limitations-2026-10-12.json)
(re-run on current bars, Mertens standard errors, live variant and status).
Select-then-trade OOS picks each walk-forward fold's in-sample winner and trades
it through the next fold; it is the number the gate judges. Variant OOS is the
validated variant's own out-of-sample Sharpe. Re-run and artifact agree to 0.01
on every row except SPY.

[^spy]: SPY does not reproduce: the artifact records -0.10 and a re-run on
    current bars gives 0.13. The cause is undetermined. See
    [Reproducibility](docs/limitations.md#reproducibility).

[^qqq]: QQQ's PROMOTE depends on PBO being advisory. Its PBO is 0.946, and
    under the earlier PBO ≤ 0.5 gate it would REJECT. Its shortlist mixes RSI(2),
    EMA-crossover and TSMOM variants, which is the shape where the
    [probe](docs/research/pbo-family-probe-2026-09-07.md) found PBO informative,
    so 0.946 is a warning, not noise.

Research tools: `uv run python -m bot.research` (max-history in/out-of-sample
strategy comparison vs buy-and-hold) and `uv run python -m bot.diagnose`
(isolated runs, cost attribution, parameter sweeps). Toggle instruments via
`enabled=` in `config.py`.

## Promotion gate

Enabling an instrument in `config.py` is a promotion decision, so it has to cite
the evidence. `bot.research` judges every variant with the backtest-overfitting
harness and promotes only what clears all three thresholds — **DSR ≥ 0.95,
P(OOS loss) ≤ 0.5, walk-forward OOS Sharpe > 0** (`ValidationConfig`) — then
records the run in `docs/validation/validation-<date>.json` (committed, so a
citation still resolves in a fresh clone) at `schema_version` 2: the selected
variant per symbol with its params, DSR, OOS Sharpe, the four CSCV panels
(`p_oos_loss`, `pbo`, `median_logit`, `degradation_slope`), the variant's own
CSCV standing (`variant_oos_rank_mean`, `variant_is_best_share`) and the
PROMOTE/REJECT verdict. A re-run on the same date overwrites that date's file,
so re-cite after re-running.

PBO is in the artifact as an advisory panel for the reason given in the lead:
in a family of near-duplicates CSCV's in-sample winner pick is near-random
whether or not the family has an edge. On a mixed shortlist it does carry information, which
is why the QQQ row is footnoted.

Two fields on the `InstrumentConfig` record the decision, and neither is a
switch — nothing reads their content and no instrument is ever blocked:

- `validation_ref=` cites a **PROMOTE verdict** and nothing else: an artifact
  entry once one exists (e.g. `"validation-2026-09-02.json: VOO TOM 2/5
  PROMOTE"`), or a recorded research run for a decision that predates the
  artifact. Set → startup is silent.
- `override_reason=` records a deliberate decision to run **without** a verdict
  (the harness rejected the instrument, or never covered it). It does not
  silence the startup warning, it changes what the warning says.

What this buys is accountability: every enabled instrument without a verdict is
named in a WARNING at startup, so running something the harness rejected or never
covered stays an explicit owner decision instead of an oversight.

## Research log

Dated records in `docs/research/`. Each one is an experiment record; none of
them changes a verdict in `docs/validation/`.

- [Insider-following, 2026-07-26](docs/research/2026-07-26-insider-following-viability.md):
  NO-GO for live capital. The Form 4 purchase anomaly is well documented, but
  its post-filing return disappears once each signal is capped at a tradeable
  size.
- [AR(1) exit hurdle, 2026-09-13](docs/research/ar1-exit-hurdle-2026-09-13.md):
  closed. No sleeve's lag-1 autocorrelation reaches its daily Sharpe, so a
  profit-target or scale-out exit does not earn a validation trial.
- [PBO family probe, 2026-09-07](docs/research/pbo-family-probe-2026-09-07.md):
  PBO reads about 0.5 for near-duplicate families with or without an edge. The
  gate moved to P(OOS loss) and PBO became advisory.
- [Trend search, 2026-09-16](docs/research/trend-search-2026-09-16.md) (latest
  of five): an NSGA-II-lite search over 180 of 390 USO EMA cells found 7 of the 8
  true Pareto-front cells. The live USO cell fails the gate as one trial among
  390.
- [TOM grid, 2026-09-07](docs/research/tom-grid-2026-09-07.md): the full
  35-cell VOO turn-of-month grid. Deflated against all 35 cells, the live TOM
  2/5 has DSR 0.916 and fails the gate; its PROMOTE rests on a three-variant
  shortlist.
- [Gate backtest, 2026-09-16](docs/research/gate-backtest-2026-09-16.md): the
  gate replayed at six year-end cutoffs (28 rows). On the 23 rows whose
  variant traded, the 6 it passed earned a mean forward Sharpe of 0.49 against
  0.56 for the 17 it failed. Of the gate's inputs, PBO ranked forward Sharpe
  best (Spearman +0.24 over all 28 rows, +0.32 over the 23).
- [Limitations record, 2026-10-12](docs/research/limitations-2026-10-12.md):
  generated re-run of the 2026-09-16 artifact with Sharpe standard errors,
  OOS trade counts, trial counts and live slippage.
- [Limitations](docs/limitations.md): what the evidence can and cannot show,
  including the [go-live NO-GO](docs/limitations.md#go-live-no-go).

## Setup

1. Install [uv](https://docs.astral.sh/uv/), then: `uv sync`
2. Create a free Alpaca account at [alpaca.markets](https://alpaca.markets)
   (no deposit needed for paper trading).
3. In the dashboard, switch to **Paper Trading** and generate an API key pair.
4. `cp .env.example .env` and fill in the keys. Keep `ALPACA_PAPER=true`.

## Commands

```sh
uv run pytest                        # unit tests (no network, no keys needed)
uv run python -m bot.backtest        # 30-month backtest -> summary table + backtest_results.png
uv run python -m bot.main --dry-run  # compute signals, submit nothing
uv run python -m bot.main            # run the paper-trading loop
```

Without API keys the backtest runs BTC/USD only (crypto data is free);
equities need keys.

`uv run python -m bot.diagnose --lookahead` re-decides each enabled strategy's
bars with the bars after them withheld and reports every decision that changes —
a signal that moves once the future is taken away was reading it. It runs on
seeded synthetic bars, so it needs no keys, and `pytest` sweeps the same check
over the live lineup: leakage fails CI, and so does an instrument the probe
could not examine. A single-seed run exercises the exit rules; the sweep unions
20 seeds so every entry gate (an RSI(2)<10 dip, an EMA cross) is reached too —
one random walk triggers almost none of them.

### Keys via 1Password (recommended)

Keep `op://` secret references in `.env` instead of plaintext keys and wrap
commands with `op run`, which resolves them into the process environment once
at launch — nothing secret ever touches disk:

```sh
op run --env-file=.env -- uv run python -m bot.backtest
op run --env-file=.env -- uv run python -m bot.main
```

(`load_dotenv()` never overrides variables `op run` already set, so the
literal `op://` strings in the file are ignored at runtime.)

## Files the bot writes

- `trades.csv` — one row per closed trade
- `daily_pnl.csv` — end-of-day equity and P&L
- `docs/validation/validation-<date>.json` — one research run's promotion
  verdicts (written by `bot.research`, not the trading loop; committed evidence)
- `state.json` — stop prices, trailing high-water marks, peak equity, per-symbol
  losing streaks and cooldowns, and the two equity marks described below
- `HALT` — the bot refuses to trade while it exists. Delete it only after
  reviewing what happened. Two rails write it automatically — the drawdown
  circuit breaker (which also flattens) and the daily loss limit (which does
  not) — plus the `halt` email command. The file carries the reason, so read it
  before clearing.

**The two equity marks.** They look alike and no feature reads both, so edit the
one that belongs to the thing you want to move:

- `day_start_equity` (`{"date", "equity"}`) is today's reference for the
  `daily_loss_limit_pct` daily loss limit and the morning briefing's "down x% on
  the day" flag. It is stamped on the first cycle of each UTC day and reads as
  nothing once that date has passed, so editing it moves today's halt threshold
  and nothing else. No new reference is stamped while `daily_loss_limit_pct` is
  `0`, but a record stamped on a day the limit was on stays in the file — inert,
  on the same date rule.
- `prev_day`/`prev_equity` is the last UTC day a `daily_pnl.csv` row was written
  and the equity it was written at — read *across* the day boundary as the
  baseline for the next row. Editing `prev_equity` changes that row's reported
  P&L; it has no effect on the halt threshold.

**What is safe to delete.** The broker is the only authority on what is open, so
`state.json` is an accelerator: delete it (or lose the volume) and the next cycle
rebuilds a stop for every broker position, cancels every stop with no position
behind it, and re-seeds both equity marks — which costs one `daily_pnl.csv` row,
since the first day after a deletion has no baseline to measure against and
writes none. Deleting it never produces a wrong action on an open position — but
it does weaken protection, and the direction is more trading: an already-earned
drawdown halt is forgiven (the breaker's peak re-seeds at today's equity), an
active cooldown is cleared, and in-flight exits lose the markers that book them
into `trades.csv`/`fills.csv`. The one exception to "more trading" is a resting
crypto entry order: it loses its only marker, and that symbol takes no new entry
until the order fills or you cancel it by hand (the morning briefing carries it
as an entry refusal). `trades.csv`, `daily_pnl.csv` and `fills.csv` are
append-only audit trails and nothing reads them back, so deleting them costs
only history. `HALT` is the one file that is not an accelerator: it records that
trading was stopped, and deleting it resumes trading.

## Risk rules (enforced in code)

- Every trade targets ~1% of equity at its stop: position size = 1% of equity ÷
  stop distance, where the stop distance is the strategy's ATR multiple
  (1×ATR mean reversion, 2×ATR breakout, 3×ATR trend). Volatile instruments
  get smaller positions.
- Realized losses can exceed 1%: crypto pays ~0.5% round-trip taker fees on
  notional, equity stops are market orders that can gap through, and slippage
  is real. The 1% is gross price risk, not a guarantee.
- Stops are attached to equity entries as one atomic OTO order and re-placed
  every cycle if missing; crypto stops are software-checked every 60 seconds.
  Stops are never widened.
- No new BTC longs while SPY **and** QQQ are both long.
- 3 consecutive losing exits on one symbol → that symbol's **entries** pause for
  5 days. Its exits, stops and sweeps are untouched: a cooldown stops new risk,
  it does not abandon the risk already on. Other symbols keep trading, and
  nothing else in the bot changes while the pause stands. To lift one early, see
  "Clearing a symbol's cooldown" below.
- Equity `daily_loss_limit_pct` (3% by default) below the UTC day's starting
  equity → write `HALT`, stop trading. No flatten: open positions keep their
  stops and you decide. Measured on mark-to-market equity, so an unrealized
  crypto move counts. Fires **once per UTC day** — see "Clearing a daily-loss
  `HALT`" below.
- Equity `RISK.max_drawdown` (10% by default) below peak → close everything,
  write `HALT`, stop trading.

Both new rails default ON and are turned off by setting
`ProtectionsConfig.cooldown_after_losses` or `.daily_loss_limit_pct` to `0`.

**Clearing a daily-loss `HALT`.** Clearing it resumes trading for the rest of
the day — the limit spends its one shot per UTC day, so it will not re-halt you
that day no matter how far equity slides. Clearing is your explicit decision to
continue, and the design takes it at face value rather than re-imposing itself
every cycle. The protection re-arms at 00:00 UTC against the new day's own
starting equity.

What still protects you afterwards: every position keeps its stop, and the
`RISK.max_drawdown` circuit breaker sits underneath and fires any time, on any
day, halting **and** flattening. If you would rather stay stood down for the
day, leave the `HALT` file in place — that is what it is for.

**Clearing a symbol's cooldown.** A cooldown expires on its own, so clearing one
is only ever about getting the symbol trading sooner. Three ways: the signed
`clear-cooldown` command — subject
`CMD clear-cooldown BTC/USD <unix-ts> <hmac>`, signed the way `bot/commands.py`
shows — deleting that symbol's entry from `cooldowns` in `state.json`, or
deleting `state.json` itself (which clears every symbol's cooldown, along with
everything else in the file). Only the command also
resets the symbol's losing streak, so the next single loss cannot re-impose the
pause; it is also the remedy for a cooldown record the briefing calls unusable.

## Deploying on Coolify

1. Create a new Coolify application from this GitHub repo — it picks
   up the `Dockerfile` automatically.
2. Set environment variables in Coolify's app settings (1Password `op://`
   references don't resolve there — paste the actual paper keys as Coolify
   secrets): `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_PAPER=true`.
3. Add a persistent volume mounted at `/data` — that's where `trades.csv`,
   `daily_pnl.csv`, `state.json`, and the `HALT` file live. Without it, a
   redeploy wipes the bot's stop-loss memory.
4. Restart policy: always. The bot logs to stdout (hourly heartbeat + one line
   per bar evaluation), visible in Coolify's log view.

## Daily email briefings (Gmail)

The bot emails a morning briefing (7am Sydney: equity, open positions with
unrealized P&L and stops, 7-day win rate, risk flags) and an evening report
(9pm: closed trades, realized P&L, equity change). Times/timezone in
`config.py` (`BriefingConfig`).

Setup:

1. Google account → enable 2-Step Verification → create an **App Password**
   at myaccount.google.com/apppasswords (Gmail SMTP rejects normal passwords).
2. Set env vars (in Coolify or `.env`): `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD`,
   optional `BRIEFING_TO`. Briefings stay silently off until these exist.
3. Set `COMMAND_SECRET` too — a long random string, and **not** the app
   password (the bot refuses to arm the channel if they match: mailbox read
   access must not also confer command authority). Without it the remote
   halt / clear-halt / force-exit / clear-cooldown / status channel ships dark
   and silent. See `bot/commands.py` for the subject format and how to sign one.
4. Test manually: `uv run python -m bot.briefing morning`

## Before you even think about live money

1. Start from the verdict table. In the 2026-09-16 artifact
   only QQQ and VOO PROMOTE; SPY, BTC/USD, GLD and USO REJECT and run on paper
   by owner override. QQQ's live variant is not the one the gate selected.
2. Every 95% CI except VOO's spans zero, and VOO's lower bound is 0.00. Five
   of six instruments have 11–35 out-of-sample round-trips, too few to tell
   their Sharpe from zero ([limitations](docs/limitations.md)).
3. Paper trade for 8–12 weeks with zero failed exits, 20–30 or more trades,
   slippage within the backtest's assumptions and Sharpe consistent with
   validation. That is the outstanding
   [go-live condition](docs/limitations.md#go-live-no-go).
4. Trading loses money for most retail participants. Nothing here is financial
   advice and no result is guaranteed.
