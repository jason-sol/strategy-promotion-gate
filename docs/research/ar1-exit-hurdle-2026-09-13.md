# AR(1) exit hurdle across the live lineup — 2026-09-13

**No sleeve clears rho >= SR on either the full daily series or on in-trade days only. A profit-target or scale-out exit rule does not deserve a validation trial.** The closest miss is VOO at a margin of -0.022, and its rho is not distinguishable from zero (z = 1.49). The five other sleeves miss by 0.035 to 0.144 — 0.8 to 3.8 times the entire daily Sharpe they earn. Three of them (SPY, QQQ, BTC/USD) carry a rho that is significantly *negative*, which is the case K&L Proposition 2 says loss-triggered exits actively damage.

## Question

Kaminski & Lo (2014), *When do stop-loss rules stop losses?* (J. Financial Markets 18, 234–254), give a sufficient condition for a path-dependent exit rule to add value on an AR(1) return process. Their eq. (18): `rho >= pi / sigma`, with rho the lag-1 autocorrelation of the **strategy's own** returns, `pi = mu - r_f` the expected return in excess of the rate earned while stopped out, and sigma its standard deviation — so the right-hand side is that strategy's Sharpe at the **same sampling frequency** as rho. Daily here; an annual Sharpe of 1.0 is roughly 0.063 daily.

**r_f is the rate earned while flat, and this lineup earns zero.** In K&L the stop-loss policy switches into the risk-free asset, so r_f is the return the position earns *while stopped out*, not a generic discount rate. Under the random walk (their Proposition 1) the stopping premium is `Delta_mu = p_o (r_f - mu)`, negative whenever mu > r_f; for AR(1) (Proposition 2) it is `Delta_mu / p_o = -pi + rho*sigma + eta`, and dropping eta gives eq. (18). Here `bot.portfolio.SimAccount.equity` is cash plus marks and cash never accrues, so a flat day in `bot/backtest.py` returns exactly 0.0 — the reconstruction check in the JSON confirms `max_abs_return_on_untraded_days = 0.0` on every sleeve — and the live Alpaca account holds idle cash at zero. `r_f = 0` is therefore the mechanically correct choice for the quantity the condition measures, and the headline tables use it. Table (iv) prices the same rho against a cash rate the bot does not earn.

The decision this informs: whether a profit-target / scale-out exit is worth one of the validation harness's trials. Every trial spent is deflation paid by every other candidate in the same family.

## Findings

**The hurdle scales with how good the sleeve already is, and our sleeves are good.** SPY's in-trade Sharpe is 2.64 annualised (0.166 daily) because conditioning on holding days strips out the 90% of the calendar it sits flat. Clearing the hurdle there would need rho = +0.17, an order of magnitude above anything measured on daily equity data anywhere. This is the mechanism, not an accident: K&L's condition is hardest to clear precisely where the drift is strong, because drift is what a path-dependent exit gives up.

**Sign matters more than the margin.** SPY, QQQ and BTC/USD have significantly negative rho on the full series (z = -4.7, -5.1, -4.5). These are mean-reverting at the daily horizon, which is the regime K&L Proposition 2 says a loss-triggered exit makes *worse*: it sells the dip the process is about to retrace. GLD, USO and VOO sit at rho indistinguishable from zero — no case either way, which is still no case for spending a trial.

**Conditioning on holding does not rescue any of them, and is not a second test.** In-trade rho stays negative or indistinguishable from zero on every sleeve; SPY's is *further* from zero than its all-days rho (-0.169 against -0.091) and significant at z = -2.26. The in-trade Sharpe rises by a factor of 1.7 to 3.3, so the margin gets worse on every single sleeve. But series (ii) cannot fail independently of series (i): flat days contribute exact zeros to the all-days series, so `SR_in_trade ≈ SR_all_days / sqrt(share_in_trade)` — verified in the JSON (`in_trade_sr_identity`) on all six sleeves, largest deviation 3.2% (SPY, from entry-bar days carrying slippage). Conditioning on holding raises the hurdle by a known factor while rho itself moves by at most about one standard error. Read (ii) as the same failure restated at a harder threshold, not as a second one.

**The only cell that clears is a reference series nobody trades.** Raw USO close-to-close clears at rho +0.042 vs SR_daily 0.022 (margin +0.020, rho significant at z = 2.15). It clears because buy-and-hold USO barely earns anything — annual Sharpe 0.35 against a structurally decaying instrument. A hurdle cleared by having almost no edge is not an invitation to add an exit rule; it is the arithmetic working as designed. Note also that the USO *strategy* does not inherit it: the 20/100 trend filter selects the trending stretches and leaves rho at +0.004.

**VOO is the interesting one, and still not interesting enough.** Underlying VOO has rho -0.122 (z = -6.3), strongly mean-reverting. The turn-of-month calendar filter flips the sign of what the sleeve actually experiences to +0.029. That is a real selection effect worth remembering, but at z = 1.49 it is not distinguishable from zero and it still sits 0.022 below SR_daily.

## Table

SIP daily bars, regenerated 2026-09-16 from a clean checkout of this branch. Equities 2016-01-04 to 2026-09-15 (2690 bars, the full Alpaca history); BTC/USD 2021-01-01 to 2026-09-16 (2085 bars, the full crypto history). rho SE is Bartlett 1/sqrt(n); `sig` marks |z| >= 1.96. `r_f = 0` in tables (i)–(iii).

### (i) Strategy daily returns, flat days included — the literal K&L quantity

| Sleeve | n | rho ± SE | z | sig | SR_daily | SR_annual | margin | verdict |
|---|---:|---:|---:|:--:|---:|---:|---:|:--|
| SPY RSI2 (vol-target) | 2689 | -0.0906 ± 0.0193 | -4.70 | yes | 0.0509 | 0.81 | -0.1415 | fails |
| QQQ RSI2 | 2689 | -0.0991 ± 0.0193 | -5.14 | yes | 0.0447 | 0.71 | -0.1438 | fails |
| BTC/USD trend 20/100 | 2084 | -0.0981 ± 0.0219 | -4.48 | yes | 0.0351 | 0.56 | -0.1332 | fails |
| GLD trend 10/50 | 2689 | -0.0078 ± 0.0193 | -0.40 | no | 0.0388 | 0.62 | -0.0466 | fails |
| USO trend 20/100 | 2689 | +0.0091 ± 0.0193 | +0.47 | no | 0.0438 | 0.70 | -0.0348 | fails |
| VOO TOM 2/5 | 2689 | +0.0287 ± 0.0193 | +1.49 | no | 0.0509 | 0.81 | -0.0222 | fails |

### (ii) In-trade days only — conditional on holding

`episodes` counts holding episodes; `pairs` counts the consecutive within-episode day pairs rho is estimated from.

| Sleeve | n | episodes | pairs | rho ± SE | z | sig | SR_daily | SR_annual | margin | verdict |
|---|---:|---:|---:|---:|---:|:--:|---:|---:|---:|:--|
| SPY RSI2 (vol-target) | 269 | 92 | 177 | -0.1690 ± 0.0750 | -2.26 | yes | 0.1661 | 2.64 | -0.3351 | fails |
| QQQ RSI2 | 347 | 93 | 254 | -0.0903 ± 0.0626 | -1.44 | no | 0.1273 | 2.02 | -0.2176 | fails |
| BTC/USD trend 20/100 | 449 | 9 | 440 | -0.1093 ± 0.0476 | -2.29 | yes | 0.0771 | 1.22 | -0.1864 | fails |
| GLD trend 10/50 | 827 | 28 | 799 | -0.0183 ± 0.0354 | -0.52 | no | 0.0704 | 1.12 | -0.0886 | fails |
| USO trend 20/100 | 977 | 14 | 963 | +0.0041 ± 0.0322 | +0.13 | no | 0.0731 | 1.16 | -0.0690 | fails |
| VOO TOM 2/5 | 870 | 137 | 733 | +0.0273 ± 0.0369 | +0.74 | no | 0.0914 | 1.45 | -0.0641 | fails |

### (iii) Underlying close-to-close — reference

| Instrument | n | rho ± SE | z | sig | SR_daily | SR_annual | margin | verdict |
|---|---:|---:|---:|:--:|---:|---:|---:|:--|
| SPY | 2689 | -0.1185 ± 0.0193 | -6.14 | yes | 0.0558 | 0.89 | -0.1743 | fails |
| QQQ | 2689 | -0.1198 ± 0.0193 | -6.21 | yes | 0.0585 | 0.93 | -0.1783 | fails |
| BTC/USD | 2084 | -0.0422 ± 0.0219 | -1.93 | no | 0.0301 | 0.48 | -0.0723 | fails |
| GLD | 2689 | -0.0063 ± 0.0193 | -0.33 | no | 0.0532 | 0.85 | -0.0596 | fails |
| USO | 2689 | +0.0415 ± 0.0193 | +2.15 | yes | 0.0218 | 0.35 | +0.0197 | **clears** |
| VOO | 2689 | -0.1215 ± 0.0193 | -6.30 | yes | 0.0560 | 0.89 | -0.1775 | fails |

### (iv) Sensitivity: series (i) re-priced at r_f = 4% annual

Same rho, same returns; only `pi = mu - r_f` changes, with r_f daily-ised as `0.04 / 252`.

| Sleeve | rho | SR_daily (excess) | margin | verdict |
|---|---:|---:|---:|:--|
| SPY RSI2 (vol-target) | -0.0906 | -0.0694 | -0.0212 | fails |
| QQQ RSI2 | -0.0991 | -0.1072 | +0.0081 | **clears** |
| BTC/USD trend 20/100 | -0.0981 | -0.0474 | -0.0507 | fails |
| GLD trend 10/50 | -0.0078 | -0.0491 | +0.0413 | **clears** |
| USO trend 20/100 | +0.0091 | -0.0392 | +0.0483 | **clears** |
| VOO TOM 2/5 | +0.0287 | -0.0523 | +0.0810 | **clears** |

**What table (iv) means, and what it does not.** At a 4% cash rate four of six sleeves satisfy the sufficient condition, because every sleeve's daily drift is smaller than 4%/252 per day, which makes pi negative and the hurdle trivially easy. That is not a licence to add an exit rule, for two reasons. First, the condition is about a **loss-triggered** exit into an asset that actually pays r_f: the bot earns nothing on idle cash, so the sleeve would be switching into 0%, not 4%, and the quantity that governs its behaviour is the one in table (i). Second, the condition says nothing about profit targets or scale-outs in either direction — clearing it would not be evidence for the rule we were asking about. The verdict stands at r_f = 0. The one thing that would genuinely change it is paying interest on idle cash: sweep the flat balance into a money-market or T-bill position and the hurdle in table (iv) becomes the live one.

## Caveats

**K&L model loss-triggered exits only.** Proposition 2 is about a stop-loss: mean reversion (rho < 0) is hurt by it, momentum (rho > 0) can gain. The inversion people reach for — that a *gain*-triggered exit (a profit target) should therefore help mean-reversion sleeves and hurt momentum ones — is a plausible inference, not a result in the paper. It is not proved there and it is not tested here. Treating it as established is how a 0.022-margin coin flip becomes a shipped exit rule.

**rho >= SR is sufficient, not necessary.** A sleeve failing it is not proved immune to exit-rule improvement; it is only stripped of this particular argument for trying. Given the trial cost, absence of the argument is the decision.

**Every Sharpe here is in-sample.** Each sleeve's parameters were selected on this same window, so the Sharpes in the SR columns are inflated by that selection. The bias runs against the sleeves: an inflated SR raises the hurdle, so the true out-of-sample margins are less negative than the ones tabled. The direction matters for VOO (-0.022) and USO (-0.035); at SPY and QQQ's margins of -0.14 no plausible in-sample premium closes the gap.

**Annualisation is sqrt(252) everywhere, crypto included** — `bot/metrics.py` uses `TRADING_DAYS_PER_YEAR = 252` for every instrument, and that convention is followed here, including for daily-ising r_f, so the figures line up with the validation artifacts. BTC/USD trades 365 days a year, so its annualised Sharpe reads roughly 17% low against a sqrt(365) convention. The hurdle itself is a daily-frequency comparison and is unaffected.

**In-trade series are short, and some have few independent episodes.** BTC/USD's 449 in-trade days come from 9 holding episodes and USO's 977 from 14; the effective sample for rho is closer to the episode count than to the day count, so those SEs understate the real uncertainty. SPY and QQQ have the opposite problem: ~3-day holds mean 92 and 93 episodes yield only 177 and 254 within-episode pairs.

**Trades and episodes are different counts.** `trades` in the JSON counts closed round-trips; `episodes` counts holding episodes and exceeds `trades` by one when a position is still open at the end of the window. QQQ (92 trades, 93 episodes) and BTC/USD (8, 9) each ended the window holding; the other four match. Table (ii) and the caveats above use the episode count, which is what rho is estimated over.

**Definitions used.** A day is in-trade if a position was open *entering* it; the entry bar is excluded because its daily return is entry slippage booked before the position has seen a day of market. In-trade rho is the sample correlation of `(x_t, x_t+1)` over consecutive pairs within holding episodes — numerator and denominator over the same paired observations, so single-day episodes and unpaired block ends do not shrink it toward zero. Pairs straddling two separate trades are not consecutive days and are excluded. The JSON also carries `rho_naive_concat`, the same series treated as one unbroken run, which differs materially for SPY (-0.040 vs -0.169) and QQQ (-0.022 vs -0.090).

**Ljung-Box Q at lag 1 is in the JSON, computed in closed form** (`Q = n(n+2)rho^2/(n-1)`, chi2(1) via erfc). statsmodels is not a dependency of this repo and was not added. At one lag it is a monotone restatement of |z|, so it changes no verdict.

Raw results, including per-sleeve trade counts, holding-period stats, the r_f sensitivity block, the in-trade Sharpe identity check and the reconstruction check: `ar1-exit-hurdle-2026-09-13.json`. Reproduce with `op run --env-file=.env -- uv run python docs/research/scripts/ar1_exit_hurdle.py` — outputs are anchored to the repo root, `--as-of` sets the filename stamp, `--rf` and `--rf-sensitivity` the cash rates.
