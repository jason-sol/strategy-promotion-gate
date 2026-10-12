# Limitations

Numbers come from the generated record [`research/limitations-2026-10-12.md`](research/limitations-2026-10-12.md) ([JSON](research/limitations-2026-10-12.json)), which re-runs the promotion artifact [`validation/validation-2026-09-16.json`](validation/validation-2026-09-16.json) on current bars.

## Trades per instrument

| Symbol | OOS trades | OOS Sharpe (re-run) | 95% CI |
|---|---:|---:|---|
| SPY[^spy] | 25 | 0.13 | [-0.55, 0.81] |
| QQQ | 35 | 0.66 | [-0.03, 1.35] |
| BTC/USD | 11 | 0.03 | [-0.73, 0.80] |
| GLD | 19 | 0.52 | [-0.16, 1.20] |
| USO | 13 | 0.52 | [-0.14, 1.19] |
| VOO | 112 | 0.69 | [0.00, 1.37] |

Five of six instruments have 11–35 out-of-sample round-trips; VOO has 112. The SE is 0.34–0.39 for all six because it scales with the number of OOS days, not trades. Trade count matters through serial correlation: one position held for several days is one outcome spread over correlated daily returns, so the effective sample is much smaller than the day count.

[^spy]: Re-run values; the artifact's −0.102 does not reproduce (see Reproducibility).

## Sharpe uncertainty

Every CI except VOO's spans 0, and VOO's lower bound is 0.00, so none of these Sharpes is clearly distinguishable from 0. The SE is Mertens (2002), which allows non-normal returns. Two caveats:

- It assumes no serial correlation. The multi-day holds above make daily returns autocorrelated (Lo 2002), so these CIs are probably too narrow.
- BTC trades every calendar day but is annualised with √252 like the equities.

## Data and baseline

Daily SIP bars from 2016-01-01 for equities and 2020-01-01 for BTC/USD, truncated at the artifact's 2026-09-16 timestamp, split into 4 walk-forward folds (`bot/research.py`, `config.py`). No buy-and-hold comparison is recorded for the same OOS windows; this is an open gap.

## What DSR deflates for

The gate deflates each Sharpe by the shortlist size, `len(candidates)` in `bot/research.py`: 6 for SPY, 4 for QQQ, BTC/USD and USO, 3 for GLD and VOO. The searches that produced those shortlists are not counted. USO's shortlist came out of a 390-configuration search and VOO's out of a 35-cell grid. Scored as one cell of that grid, VOO's TOM 2/5 has DSR 0.916, below the 0.95 bar. Other symbols have no search file, so their trial count is unknown, not zero. The DSR is optimistic by construction.

The other gate conditions add little. P(OOS loss) ≤ 0.5 never binds in this artifact: the highest value is 0.287 (BTC/USD), and the [probe](research/pbo-family-probe-2026-09-07.md) calls 0.5 lenient. In practice the gate is DSR ≥ 0.95 plus OOS Sharpe > 0.

## PBO is advisory

On a synthetic family of near-duplicate variants, CSCV's PBO reads about 0.5 whether the family has a real edge or none ([probe](research/pbo-family-probe-2026-09-07.md)), so it cannot gate grids of overlapping parameters. The gate uses P(OOS loss) ≤ 0.5 (`p_oos_loss_max`) and reports PBO beside it. VOO's 0.41 comes from three overlapping TOM variants, the case where PBO carries no information. QQQ is the exception: its mixed shortlist (RSI2, EMA, TSMOM) is the case where the probe found PBO informative, it reads 0.946, and under a PBO ≤ 0.5 rule it would REJECT.

## Reproducibility

SPY's select-then-trade OOS does not reproduce: −0.102 in the artifact, 0.128 on re-run. In fold 0 two TSMOM variants tied exactly in-sample, and fold 1 selected EMA 50/200, which made no in-sample trades. Three variants' own OOS Sharpe also moved between runs while three held within 0.01. The cause is undetermined; revised SIP bars are the leading hypothesis. It can't be settled because artifacts record neither a code SHA nor a bar checksum.

## Costs

The backtest charges `slippage_pct` on every fill: 0.0001 for SPY, QQQ, GLD and VOO, and 0.0005 for USO and BTC/USD, plus a 0.0025 taker fee on crypto. The live fills log has 20 rows, 19 measurable: SPY 4, QQQ 7, VOO 6, GLD 1, USO 1, and none for BTC in the local copy. That is below the 20-fill minimum per symbol, so realised slippage is not yet evidence either way.

## Live vs validated

The bot runs on a paper account. Only VOO trades the variant the gate promoted (TOM 2/5). QQQ passed, but it trades RSI2<10 200sma rather than the selected EMA 50/200 long, by owner decision. SPY, BTC/USD, GLD and USO were rejected and run by owner override. BTC also runs EMA 20/100 rather than the validated EMA 50/200.

## Go-live: NO-GO

The 2026-08-31 assessment found exit and kill-switch failures and gave a NO-GO for real money. Those failures have been fixed and deployed, though the order-refusal and halt paths have so far run only against test fakes. Outstanding is a clean 8–12-week paper run: zero failed exits, 20–30 or more trades, slippage within assumptions, and Sharpe consistent with validation.

## Not modelled

- Intraday microstructure: fills are daily close ± slippage.
- Borrow and short constraints: the live lineup is long-only.
- Capacity and market impact.
- Regime change beyond what four walk-forward folds sample.
- Post-publication decay. McLean & Pontiff (2016, *J. Finance*) find that published anomalies weaken after publication, and RSI(2) is a published rule.
