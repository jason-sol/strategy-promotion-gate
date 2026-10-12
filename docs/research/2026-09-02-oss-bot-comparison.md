# Open-source trading bots vs. claude-trading-bot — competitive comparison

Date: 2026-09-02
Question: what do the leading open-source trading bots have that this bot could learn
from, and which of it is worth adopting?
Decision this informs: the improvement roadmap after the exit/stop reliability work
(the "OSS bot comparison follow-ups" task list, kept outside this repository).
Method: deep-mode research — 12 user-nominated sources plus ~20 frameworks surfaced by
survey, synthesized by one agent, then every load-bearing claim independently
re-verified by a second agent using the GitHub API, code search with positive
controls, Wayback Machine (for deleted content), and direct reads of this repo.
Verification verdict: PASS; ten corrections surfaced and applied in this document.

## Headline

**We are not behind on rigor — we are behind on operations.**

The strongest finding is a verified negative: **no open-source trading bot implements
CSCV/PBO, and a deflated Sharpe ratio exists only in vectorbt** (Lean carries a
Probabilistic Sharpe Ratio; the only dedicated library, mlfinlab, 4.9k stars, is dead
since 2021-11). Code search across freqtrade, Lean, nautilus_trader, PyBroker,
Lumibot and Jesse returned zero genuine hits, with working positive controls
(vectorbt's `deflated_sharpe_ratio`, Lean's `ProbabilisticSharpeRatio`) proving the
zero-results are real. `bot/validation.py:127` (`cscv_pbo`) and `bot/metrics.py:117`
(`deflated_sharpe_ratio`) have no equivalent in any 20k–54k-star framework.

Three more of our capabilities are equally unoccupied ground:

- **Realized execution-cost measurement** (fills.csv, achieved-vs-decision-mid) —
  nobody in the field computes intended-vs-achieved fill quality. freqtrade's docs
  state its backtester fills every order at the requested price (no slippage).
- **Stop-reliability engineering** (OTO stop with DAY→GTC promotion, atomic close,
  per-instrument isolation, naked-position restore) — passivbot has no
  exchange-native stops; robobull, asavinov and OpenTrader ship stops that die with
  the process; freqtrade re-places on a poll loop.
- **Backtest/live strategy parity by construction** — Nautilus has code parity;
  freqtrade's docs warn against relying on backtest/live equivalence.

## Source credibility triage

Several nominated repos did not survive scrutiny. Weight the credible tier only.

**Credible tier:** freqtrade, passivbot, OctoBot, asavinov/intelligent-trading-bot,
plus survey-surfaced Lean, Lumibot, PyBroker, nautilus_trader, vectorbt.

**Not credible / dead:**

- **ryanfrigo/kalshi-ai-trading-bot — deleted.** Repo and account 404; the numeric
  user id (18334402) is also gone, which a rename would preserve, proving genuine
  deletion (between 2026-07-13 and 2026-09-02). Wayback shows it had quarter-Kelly
  sizing, a 15% drawdown halt, a 10% daily-loss breaker — and no backtest.
- **MrFadiAi/Polymarket-bot — marketing shell.** 626 stars on 9 commits landed
  within 7 hours of one day; 32 of 41 `src/` files are byte-identical (blob-SHA
  match) to cyl19970726/poly-sdk; unrebutted audit (issue #2) shows `canTrade()` is
  only called from a disabled path and `calculatePositionSize()` is never called.
- **robobull-trading-bot — abandoned since 2022-07** on EOL Node 12/mongoose 5, and
  its backtester is structurally wrong: it re-seeds capital every simulated day
  (src/providers/alpaca/backtest.js:112-117) and sums daily ROI instead of
  compounding. Its declarative `algos.json` indicator schema is the one nice idea.
- **CloddsBot — burst build, dormant, then revived.** 853 stars, 91% of commits in
  Jan–Feb 2026, npm package at ~16 downloads/week; dormant Mar–Jul, then real fixes
  through 2026-09-01. "Walk-forward" is claimed in a header comment and absent from
  the code. Its chronological-holdout ML gate is defensible.
- **Stock-Trading-Bot (codingforentrepreneurs) — honest course companion** that
  never places a trade (no broker integration at all). Worth stealing engineering
  from only: TimescaleDB hypertables, Celery fan-out with an inline escape hatch,
  DB-editable schedules.
- **The Medium listicle** — 403 to direct fetch, recovered via mirror; stale
  (~mid-2025 star counts), half padding. **GitHub topics `strategy-bot` /
  `grid-bot`** — unusable discovery surfaces (4 and 16 repos; mistagged entries and
  a README-only repo with a purchased-star signature at #2).
- **Open-Trader/opentrader — dormant since 2025-06-29.** "100+ exchanges" is seven
  hardcoded in an enum plus a generic CCXT adapter; it deletes accounts outside the
  enum on boot; no hosted docs site (the domain redirects to the repo).

## Per-repo findings (credible tier)

### freqtrade — 53.9k stars, active (2026-09-01), GPL-3.0

The reference implementation of a bot *operations* surface. Crypto-only via CCXT;
Alpaca support closed `not_planned` (issue #7953), so it is not a migration target.

- `IStrategy` v3: three required methods plus ~20 optional callbacks
  (`custom_stoploss`, `custom_exit`, `adjust_trade_position`,
  `confirm_trade_entry`, `order_filled`, `adjust_order_price`).
- **Protections framework** (declarative, config-composed circuit breakers):
  `StoplossGuard`, `MaxDrawdown`, `LowProfitPairs`, `CooldownPeriod`.
- **Telegram control, 43 commands** including `/forceexit`, `/stop`, `/status`,
  `/locks`; plus REST API, FreqUI web dashboard, webhooks. Dry-run is the default.
- **Two bias detectors nobody else has:** `lookahead-analysis` (empirically re-runs
  per signal and diffs the frames to catch future leakage) and `recursive-analysis`
  (catches backtest-vs-live indicator drift from limited startup history).
- Restart reconciliation as first-class code: `startup_update_open_orders()`,
  `handle_onexchange_order()`, `handle_insufficient_funds()`.
- **Weaknesses:** hyperopt (Optuna/NSGA-III) is single-window in-sample — walk-forward
  requested and open since 2022 (issue #7190). Backtest docs state all orders fill
  at the requested price with no slippage (partial fills/spread appear unmodeled,
  by inference from absence). Edge position-sizing module removed in release 2025.6.

### passivbot — 2.1k stars, Unlicense, active (2026-08-31)

Grid/DCA bot with a 42k-line Rust core shared between live and backtest. Largely
AI-generated of late: 1,029 `codex/*` branch merges, all since 2026-05-18.

- **Real multi-objective optimizer:** pymoo NSGA-II/III, 11 default objectives.
- Real per-coin maker/taker fees in the backtest.
- **Best doctrine in the survey — deliberately stateless restart:** state is
  reconstructed from exchange state, fill history, candles, config and clock;
  *"local caches … must not become authoritative trading state"*
  (docs/risk_management.md).
- Equity Hard Stop Loss circuit breaker exists but ships `enabled: false`.
- **Cautionary tale:** default optimizer config is 84 declared bounds (62 free
  parameters) × 200,000 evaluations against one contiguous in-sample window, with
  zero mentions of overfitting/holdout/walk-forward anywhere in 2.4MB of docs.
  Limit fills are fully optimistic (full fill the moment a 1-minute candle wicks
  through). Issue tracker near-dead (2 issues closed in 90 days vs ~3.2k commits).

### OctoBot — 6.5k stars, active, GPL-3.0

448k LOC, 16 trading modes, evaluator→strategy→trading-mode split, tentacle plugin
architecture, Telegram control, first-class paper trading.

- Optimizer is exhaustive grid search randomly subsampled to 10,000 runs; genetic
  mode is declared but raises `NotImplementedError` in OSS.
- Default fitness parameters are percent gain (weight 1) and an R²-of-balance term
  (weight 0) — **drawdown is not in the default fitness at all.**
- Backtest: fees default 0, no slippage, and intra-candle price path is two
  synthetic ticks in a fixed low-then-high order — systematically biasing every
  stop-vs-target race.
- The serious analysis UI (Strategy Designer) is a one-time $99 extension purchase
  (cloud tiers $9.99/$29.99 per month are separate).
- Open issues worth remembering: #3552 (configured testnet, connected live),
  #3190 (SL/TP orders orphaned after restart — the bug our restart-invariant work
  prevents).

### asavinov/intelligent-trading-bot — 1.9k stars, MIT, active (2026-08-30)

The only credible ML entry; publishes live signals to a 2.4k-subscriber Telegram
channel.

- **One genuinely transferable idea:** column-oriented design — features, ML scores
  and boolean signals are all the same kind of object (a column from a registered
  generator, forming a DAG). See docs/column-oriented-design.md.
- Correct walk-forward leakage guard: `train_end = predict_start - label_horizon - 1`
  (scripts/predict_rolling.py:188).
- **No risk management at all** in the trader — no stops, no drawdown limit, no kill
  switch — and no transaction costs in the backtest.

### Survey-surfaced frameworks

- **Lumibot** (2.0k stars, active) — the closest architectural match anywhere:
  Alpaca as first-class broker *and* backtest datasource. No walk-forward, no
  overfitting tooling of any kind.
- **QuantConnect Lean** (21.4k stars, Apache-2.0) — OSS Alpaca adapter
  (Lean.Brokerages.Alpaca), five slippage models incl. market-impact,
  ProbabilisticSharpeRatio, parameter-stability tooling. Live-vs-backtest
  reconciliation is documented in the OSS docs too, but the underlying
  parallel-OOS-backtest mechanism is a cloud service.
- **PyBroker** (3.5k stars) — best OSS walk-forward harness:
  `walkforward(windows, train_size, lookahead, calc_bootstrap, …)` where
  `lookahead` explicitly prevents label leakage across fold boundaries.
- **nautilus_trader** (28.3k stars) — 11 fill models, L3 book, queue position; no
  Alpaca adapter. **vectorbt** (9.0k stars) — the field's only deflated Sharpe;
  purged CV is paywalled in PRO. **zipline-reloaded** — no broker module; its
  Alpaca bridge (pylivetrader) died in 2022.

## Feature-gap matrix

### What the credible tier has that we lack

| Capability | Who has it | Substance or polish | Our state |
|---|---|---|---|
| Remote control / force-exit from a phone | freqtrade (43 Telegram cmds), OctoBot | **Substance** | Email-only, one-way; HALT needs VPS shell |
| Declarative protections (cooldown after N losses, daily-loss limit) | freqtrade | **Substance** | Guards hand-wired in RiskManager/Engine; no cooldown at all |
| Lookahead-bias detector | freqtrade only | **Substance** | Absent |
| Recursive-indicator / startup-window detector | freqtrade only | **Substance** | Absent — and per-strategy `min_bars` is the precondition for the bug class |
| Multi-objective optimizer | passivbot (pymoo) | Substance, dangerous unguarded | No optimizer (deliberately, so far) |
| Rich strategy callback surface | freqtrade (~20 hooks) | Substance | 3-method ABC |
| Strategy registry / plugin discovery | freqtrade, OctoBot | Mild | `build_strategy` 7-branch if-chain |
| Execution-realism fill models | Nautilus, Lean | Substance | Flat per-instrument slippage — but paired with measured fills, arguably more honest |
| Web dashboard | freqtrade FreqUI, OctoBot | **Polish** | None |
| Multi-exchange abstraction | all | **Polish (for us)** | Alpaca-only, correctly |

### What we have that the field lacks

| Our capability | Field equivalent | Verdict |
|---|---|---|
| CSCV/PBO (`bot/validation.py:127`) | None anywhere; mlfinlab dead 2021 | **Unique** |
| Deflated Sharpe (`bot/metrics.py:117`) | vectorbt only; Lean has PSR | **Rare** |
| Walk-forward select-then-score-OOS | PyBroker; asavinov (rolling retrain) | Rare — top ~3 |
| Realized fill-quality measurement (fills.csv) | None | **Unique** |
| Stop-reliability engineering (GTC promotion, atomic close, isolation, naked-restore) | Nobody | **Unique** |
| Backtest/live parity by construction | Nautilus (code parity) | Strong |
| Five-dependency footprint | freqtrade ~100s of deps | Underrated asset |

## Implications

1. **What makes them "better" is operations, not intelligence.** Substance: remote
   control, declarative protections, restart-reconciliation doctrine, bias
   detectors. Polish: dashboards, multi-exchange, marketplaces, LLM layers.
2. **The field's central pathology is optimizing hard against one in-sample
   window** (passivbot 62 free params × 200k evals, zero overfitting guidance;
   OctoBot ranking 10,000 backtests by raw gain; freqtrade hyperopt with no CV).
   Adopting their optimizers without our DSR/PBO harness in front would import
   their central flaw. Conversely, we are uniquely positioned to build the thing
   none of them has: an optimizer whose objective is the deflated Sharpe with
   CSCV/PBO as the promotion gate.
3. **Our worst incidents map onto the field's default failure mode.** The
   month-stuck SPY exit and the broken kill switch are the same class the field
   ships by default (process-bound stops, no remote intervention). The fixes put
   us ahead; the remaining gap is that nothing can force an exit without a shell
   on the VPS.
4. **Star counts are worthless in this niche** (626 stars on 9 commits of
   re-exported code; 442 on a README-only repo).
5. **Do not migrate.** freqtrade closed Alpaca as not-planned; Nautilus has no
   Alpaca adapter; Lumibot lacks all validation tooling; Lean means C# and its
   reconciliation mechanism is a cloud service. Migration trades our strongest
   asset for their ops surface — borrow the ops surface instead.

## Recommendations

Quick wins (hours): declare numpy in pyproject.toml (validation.py imports it, only
transitively present today); a remote force-exit/halt/status command path; a
realized-slippage report from fills.csv feeding measured costs back into backtest
assumptions; a startup-window guard; a dict strategy registry.

Medium (days): CI pipeline (already ranked "highest ratio of safety to effort" in
the 2026-08-31 assessment); declarative protections (cooldown after N stop-outs,
daily-loss limit, on by default); validation as a recorded promotion gate rather
than a manual script; a lookahead-bias detector (tractable because strategies are
pure functions over DataFrames); the restart invariant written down and tested.

Strategic (weeks): small multi-objective optimizer with DSR objective + PBO gate;
optionally publish the validation harness (a maintained, scipy-free CSCV/DSR
library is a real community gap).

Explicitly not worth adopting: multi-exchange abstraction, strategy marketplaces,
grid/DCA/martingale engines, FreqAI-style continuous retraining, LLM decision
layers, and a web dashboard ahead of the items above.

The task-level plan lives in a task list kept outside this repository. Nothing in it displaces the standing gates: the 8–12 week clean
paper run remains the precondition for real money.

## Verification notes

Independent verification passed overall; ten corrections were applied to this
document relative to the raw synthesis:

1. freqtrade issue #2472 dropped (closed in 2020; a discussion, not a CV request —
   #7190 is the live citation).
2. OctoBot default fitness: drawdown is not a default parameter at all (the
   weight-0 term is R² of balance), rather than "drawdown weighted 0".
3. CloddsBot recast from "abandoned" to burst → dormancy → revival (real commits
   through 2026-09-01).
4. Open-Trader: "docs domain parked" corrected to "no hosted docs site" (domain
   live-redirects to GitHub).
5. Lean reconciliation is documented in OSS docs too; only the mechanism is
   cloud-hosted.
6. OctoBot $99 is a one-time extension purchase, not a subscription tier.
7. freqtrade Telegram: 43 commands, not "~40".
8. vectorbt is the smallest-by-stars of the compared frameworks; "only mature
   framework with DSR" softened accordingly.
9. passivbot optimizer scale corrected to 84 bounds / 62 free params; "zero issues
   closed in 90 days" corrected to 2.
10. freqtrade partial-fill/spread absence stated as inference-from-absence, not an
    explicit doc claim.

Out of verification scope (nothing above leans on them): the Medium listicle's
per-repo stats, the topic-page details, Stock-Trading-Bot's internals.

## Sources

- freqtrade: https://github.com/freqtrade/freqtrade · docs: strategy-customization,
  strategy-callbacks, backtesting, hyperopt, lookahead-analysis,
  recursive-analysis, plugins, telegram-usage (freqtrade.io/en/stable/) · issues
  7953, 7190 · https://github.com/freqtrade/freqtrade-strategies
- passivbot: https://github.com/enarjord/passivbot · docs/risk_management.md,
  docs/equity_hard_stop_loss.md, docs/optimizing.md
- OctoBot: https://github.com/Drakkar-Software/OctoBot · issues 3552, 3190 ·
  https://www.octobot.cloud/pricing
- asavinov: https://github.com/asavinov/intelligent-trading-bot ·
  docs/column-oriented-design.md · scripts/predict_rolling.py
- Lumibot: https://github.com/Lumiwealth/lumibot · Lean:
  https://github.com/QuantConnect/Lean +
  https://github.com/QuantConnect/Lean.Brokerages.Alpaca + reconciliation docs
  (writing-algorithms/live-trading/reconciliation) · PyBroker:
  https://github.com/edtechre/pybroker · Nautilus:
  https://github.com/nautechsystems/nautilus_trader · vectorbt:
  https://github.com/polakowo/vectorbt · mlfinlab:
  https://github.com/hudson-and-thames/mlfinlab (dead 2021)
- robobull: https://github.com/michaeljwright/robobull-trading-bot ·
  Stock-Trading-Bot: https://github.com/codingforentrepreneurs/Stock-Trading-Bot ·
  Polymarket-bot: https://github.com/MrFadiAi/Polymarket-bot (upstream
  https://github.com/cyl19970726/poly-sdk, audit issue #2) · CloddsBot:
  https://github.com/alsk1992/CloddsBot (npm `clodds`) · kalshi-ai-trading-bot:
  deleted (Wayback capture 2026-07-13) · OpenTrader:
  https://github.com/Open-Trader/opentrader · Medium listicle:
  https://medium.com/@gwrx2005/top-10-ai-powered-crypto-trading-repositories-on-github-0041862546b6
  (stale) · topics: strategy-bot, grid-bot (unusable)
