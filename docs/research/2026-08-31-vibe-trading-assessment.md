# HKUDS/Vibe-Trading: adoption assessment (2026-08-31)

**Verdict: NO-GO as a dependency, platform, or runtime component. GO as a design reference for one module — `agent/src/live/` — folded into the in-flight exit/stop work rather than opened as a workstream. Two cheap contained borrows noted for later.** Assessment made against Vibe-Trading v0.1.14 (commit `899d3c7`, MIT).

## Question

`github.com/HKUDS/Vibe-Trading` is an LLM-driven trading agent at ~32k stars. What, if anything, should claude-trading-bot take from it — as a dependency, a platform, a source of signals, or a source of designs?

## Findings

### 1. What it actually is, and it is not vapourware

LangGraph/LangChain agent core, FastAPI backend, React 19 frontend, Electron desktop shell. Two entry points in `pyproject.toml`: `vibe-trading = cli:main` and `vibe-trading-mcp = mcp_server:main` — the latter an MCP stdio server exposing `backtest`, `factor_analysis`, `analyze_options`, `pattern_recognition`, `get_market_data`, and the skill loader. 88 bundled skills, multi-provider LLM support including a `langchain-anthropic` extra.

Evidence it is maintained rather than star-farmed: 1,275 issues filed with 15 open at time of writing, many of them precise self-reported bugs (`bug(rebalance): 100% target plus commission aborts instead of scaling basket`). `agent/tests/` runs under pytest with `pytest-socket` to enforce network-free unit tests, ruff and black configured, `[tool.pytest.ini_options]` markers separating unit from integration. The code in `agent/src/live/` uses atomic temp-file + `os.replace` writes and carries docstrings that explain *why* a decision was made, not what the function does.

Counterweight on the hype: their own README carries a security warning that an X account, a Virtuals project, and a token contract are impersonating them. The star count sits in an ecosystem that inflates it. Judge the code, not the badge.

### 2. The one genuinely valuable module: `agent/src/live/`

This is what earns the note. Their live-safety layer is a well-reasoned solution to the failure class that produced our own NO-GO on 2026-08-31.

**`agent/src/live/halt.py` — kill switch as a filesystem sentinel.**

- The switch is the *existence of a file* (`<runtime_root>/live/HALT`), never in-process state. Checked at the top of every order path before any broker call, so it holds when the agent loop is wedged mid-iteration.
- Written atomically: same-directory temp file then `os.replace`, so a partial write can never leave a corrupt sentinel a concurrent reader would misparse.
- **Fail-closed on every axis.** A sentinel that exists but holds empty or malformed JSON is still a halt — existence is authoritative, the JSON payload (`tripped_at` / `by` / `reason`) is attribution only. An unresolvable broker key returns halted rather than raising.
- Global flag beats per-broker: if the global sentinel exists, every broker reads as halted.
- `trip_halt()` is deliberately kept a *pure sentinel writer* — it does not fire the flatten action. The reasoning, stated in the module docstring: a user or an external watchdog may `touch` the file directly, bypassing `trip_halt` entirely, so the runner that *observes* the sentinel must be the single place that drives the side effect. Coupling the broker call to the flag write would make the direct-touch path silently weaker than the API path.
- Register/invoke split for the action: `register_halt_action()` / `on_halt_action()`, with a logged warning when a halt is observed and no action is registered.

**`agent/src/live/runtime/flatten.py` — the preemptive sweep.**

- **Cancel-before-flatten**, with the reason given: "a resting order left live while we flatten could fill against our closing trade and re-open exposure." Cancels quiesce the book first, then the position sweep closes what is actually held.
- **Never retry a failed write.** "Trading is not idempotent. A broker call that errors is recorded and we move on — it is NEVER retried." Same conclusion `bot/retry.py` reached independently, which is mild external validation of our reads-only retry decorator.
- **Detects nested broker rejections inside a successful transport envelope.** `_error_envelope_message()` checks not just `{"status": "error"}` at the top level but also `{"status": "ok", "data": {"status": "error"}}` and `{"status": "ok", "data": {"ok": false}}`. Their own comment on why: checking only the top level "recorded a rejection as accepted and the sweep latched with a live order still resting." That is exactly the bug shape behind our SPY exit — booked as done while the position stayed live.
- Flatten defaults **OFF** (cancel-only) absent explicit authorisation, because auto-submitting market exits is itself a trading action the user may not have authorised.
- Returns a structured per-call report with a `side_effects_attempted` boolean, so the runner can tell whether replay after a restart is still safe. Every cancel, every close, and every error is written to an audit ledger *before* the report returns.

**Siblings worth naming for a later read:** `runtime/reconcile.py` (25KB), `runtime/sweep_latch.py`, `runtime/liveness.py`, `runtime/triggers.py`, `live/order_guard.py` (32KB), `live/enforcement.py` (32KB), `live/audit.py`.

**Mapping onto our blockers.** Our Phase 1 (the go-live remediation; see `docs/limitations.md#go-live-no-go`) landed functional equivalents of items 1–6 independently, and in some places ours is more specific to Alpaca's actual semantics (the `held_for_orders` 403 race and the 10s `has_open_order` poll have no analogue in their broker-agnostic layer). Where they are ahead of us:

| Their design | Our current state |
|---|---|
| Halt checked before *every* order, fail-closed on malformed payload | HALT written before the flatten loop (`bot/main.py`); no fail-closed read contract, no atomic write |
| Cancel-before-flatten ordering, stated invariant | flatten loop is per-symbol try/except; cancel/close ordering not an explicit invariant |
| Nested `ok`-envelope rejection detection | we check the exception path; a broker response that *says* ok but nests a rejection is the residual risk |
| `side_effects_attempted` → replay-after-restart safety | no equivalent signal |
| Every broker call audited before the report returns | trades.csv + fills.csv at reconciliation; no per-call ledger of attempts |

### 3. Validation: we are ahead, with two specific gaps

`agent/backtest/validation.py` offers three tools: a Monte Carlo permutation test that shuffles trade PnL order to test path significance, a bootstrap Sharpe confidence interval, and a sequential-window walk-forward.

It has **no probability of backtest overfitting and no deflated Sharpe**. Nothing in it corrects for multiple testing across candidates. Our `bot/validation.py` — CSCV/PBO, `bot/metrics.deflated_sharpe_ratio`, walk-forward, behind the `VALIDATION.passes_validation` gate (DSR ≥ 0.95, PBO ≤ 0.5, OOS > 0) — is materially stricter on the axis that matters most for a 21-variant sweep in `bot/research.py`. Selection bias across variants is our dominant overfitting risk, and their harness does not address it at all.

Two of their three tools we genuinely lack, and both are cheap:

- **Monte Carlo trade-order permutation** → is the equity *path* better than a random ordering of the same trades? Guards against a Sharpe that rests on one lucky sequence. Yields `p_value_sharpe` and `p_value_max_dd`.
- **Bootstrap Sharpe CI** → resample daily returns for `ci_lower` / `ci_upper` / `prob_positive`. Turns a point Sharpe into an interval, which is what a promotion decision actually wants.

Roughly 120 LOC of numpy against data we already produce (`bot/portfolio.py` writes the trade log; `build_return_matrix` already assembles returns). Recorded as a future borrow, not a task.

### 4. What is a dead end for us

- **The factor zoo is cross-sectional.** `agent/src/factors/zoo/` ships all 101 WorldQuant Alpha101 factors, GTJA191, Qlib158, plus academic and fundamental sets. These rank across a universe — `rank(...)` over hundreds of names is intrinsic to their definitions. We trade 6 hand-picked instruments, so most are undefined for us. Not a source of signals at our universe size.
- **Their US equity backtest is coarser than ours.** `agent/backtest/engines/global_equity.py` is 7KB over yfinance-backed loaders. Our `bot/backtest.py` shares the actual `Strategy` and `RiskManager` objects with `bot/main.py`, models intrabar stop hits, per-instrument slippage and crypto taker fees. Backtest/live parity by construction is our strongest asset and they do not have it.
- **Irrelevant surface:** options analytics, 9-market engines, China A-share/Korea/Vietnam loaders, 13 broker connectors, Electron desktop, and chat channel integrations for Slack/Telegram/Feishu/WhatsApp/QQ/WeChat.

### 5. Why not adopt it as a platform

- **Dependency surface.** ~45 runtime dependencies — langchain, langgraph, ccxt, tushare, akshare, duckdb, scikit-learn, scipy, weasyprint, matplotlib, fastapi, uvicorn — against our 5 (`alpaca-py`, `pandas`, `python-dotenv`, `matplotlib`, `tzdata`). Every one is a supply-chain and upgrade-cost edge on a process that trades real money unattended.
- **`agent/backtest/engines/base.py` is an 87KB monolith.** Not something to take on as a maintained dependency.
- **Their live layer targets an LLM-in-the-loop consent model.** Mandate gates, consent-first commits, and pending-action approval flows exist because a model proposes the orders. Our bot is deterministic pandas and rules; the AI stays in the dev loop. Adopting their agent means importing a class of failure — model proposes a bad order, gate is the only defence — that we currently do not have and should not acquire.

The transferable asset is the *reasoning* in their live module, and reasoning ports without a dependency edge.

### 6. Licensing

MIT at the repository root, so borrowing designs is clean and borrowing code is permissible with attribution. If anything is ever lifted near-verbatim, attribute it at the call site. `agent/src/factors/zoo/alpha101/LICENSE.md` carries its own separate terms and would need independent treatment — moot given §4.

## Alternatives ranked (why this is not the next thing we build)

1. **Finish the in-flight TDD cycle on `fix/exit-stop-reliability`.** The red at the time of writing sits in `tests/test_main_helpers.py` — partial-close booking and vanish-sweep guards. This is the work that clears the NO-GO; nothing below it matters until it is green.
2. **Go-live blockers** — verify the VPS `/data` persistent volume (a redeploy without it wipes stop-loss memory) and get SPY exiting cleanly through the fixed path.
3. **Re-run `bot.backtest` and `bot.research` on the new feed setting.** Item 8c moved every offline tool to `MARKET_DATA_FEED` (default IEX), which changes historical numbers. Old results are not comparable until this is done.
4. **Items 10–11** — clean 8–12 week paper re-measurement, then small live.
5. **A CI pipeline.** 550-plus tests and no `.github/` means every quality gate is a manual `uv run pytest`. Highest ratio of safety to effort on this list.
6. **Harden against §2's residual gaps** — a fail-closed atomic HALT read contract, an explicit cancel-before-close invariant, and nested-envelope rejection detection on Alpaca responses. Their designs are the reference; the code should be ours and TDD'd.
7. **The two validation borrows from §3.**
8. **An isolated `vibe-trading-mcp` research spike** — separate venv, research-only, attached to Claude Code as a tool for idea generation. Never a runtime dependency of `bot/`. Lowest priority: our bottleneck is execution reliability, not idea supply.

## Gates if ever revisited

1. Branch green, items 7 and 9 closed, and a clean 8–12 week paper re-measurement passing the item-10 gates. Nothing new until then.
2. CI in place, so a borrow cannot regress the suite silently.
3. Any borrow arrives as our own code with its own failing-test-first cycle. No `pip install vibe-trading-ai` inside the bot's dependency tree, ever — the MCP spike lives in its own venv or not at all.
4. If the factor zoo is ever revisited, it requires a universe wide enough for cross-sectional ranking to be defined — a different bot, not this one.

## Sources

- Repository — https://github.com/HKUDS/Vibe-Trading (v0.1.14, MIT, commit `899d3c7`)
- Kill-switch sentinel — `agent/src/live/halt.py`
- Preemptive cancel/flatten sweep — `agent/src/live/runtime/flatten.py`
- Backtest validation tools — `agent/backtest/validation.py`
- Factor zoo — `agent/src/factors/zoo/{alpha101,gtja191,qlib158,academic,fundamental}/`
- Backtest engines — `agent/backtest/engines/` (`base.py` 87KB, `global_equity.py`)
- Dependencies and entry points — `pyproject.toml`
- Docs — https://vibetrading.wiki/
- Our validation harness — `bot/validation.py`, `bot/metrics.py`, `config.py` (`ValidationConfig`)
- Our retry policy — `bot/retry.py`
- Our backtest parity — `bot/backtest.py`, `bot/main.py`
- Our open blockers — `docs/limitations.md#go-live-no-go`
