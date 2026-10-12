# USO EMA-crossover trend search vs exhaustive truth — 2026-09-14

*This record predates the Sharpe floor on the drawdown objective — its JSON carries no `sharpe_floor` key. Under the floor its 14-cell true front is 8 cells and its 17-cell search front is 11; the coverage and precision quoted below were measured against the pre-floor fronts.*

An NSGA-II-lite search (seed 0, population 36, budget 180 spent in 9 generations (ceiling 40)) covered 180 unique backtests over the same genome space the exhaustive grid enumerates in 390 cells (fast_period 5-50 x slow_period 40-100 x trail_atr_mult 2.0-4.0 x volume_gate_mult None-1.5), on daily USO bars from 2016-01-01 on the sip feed. Objectives: maximize Deflated Sharpe, minimize max drawdown. Every search DSR is deflated at the search's OWN cumulative trial count, so the deflation tightens as it spends the budget; the exhaustive field deflates at 390.

## Did the search find the truth

The exhaustive Pareto front has 14 cells; the search's front has 17, of which 8 are exactly on the true front — coverage 57% of the true front, precision 47% of the search's. The search's DSR-best cell is T 40/60/3.0 at DSR 0.378 on its own 180 trials; the same cell deflated against the whole 390-cell field is 0.218, against the field's best T 30/60/2.0 at 0.292 — a DSR regret of 0.074. The cheap search reports a bigger number for the same backtest; the trial count is the only thing that makes the two comparable.

| Cell | Sharpe | DSR @ 180 trials | DSR @ 390 trials | maxDD | OOS Sharpe | On true front |
| --- | --- | --- | --- | --- | --- | --- |
| T 20/80/4.0/1.5 | -0.26 | 0.000 | 0.000 | 1.2% | -0.29 | yes |
| T 15/80/4.0/1.2 | 0.03 | 0.006 | 0.002 | 1.4% | 0.03 | yes |
| T 50/100/3.5 | 0.52 | 0.190 | 0.088 | 3.3% | 0.47 | no |
| T 40/80/3.5 | 0.57 | 0.236 | 0.116 | 3.6% | 0.53 | yes |
| T 20/60/3.5/1.5 | 0.17 | 0.024 | 0.007 | 2.0% | 0.19 | no |
| T 50/80/3.5 | 0.52 | 0.193 | 0.090 | 3.4% | 0.48 | no |
| T 30/40/4.0/1.5 | -0.20 | 0.000 | 0.000 | 1.4% | -0.22 | yes |
| T 30/80/2.0 | 0.65 | 0.338 | 0.194 | 3.7% | 0.80 | yes |
| T 30/80/2.0/1.5 | 0.08 | 0.011 | 0.003 | 1.8% | 0.09 | no |
| T 30/60/4.0/1.5 | -0.26 | 0.000 | 0.000 | 1.2% | -0.29 | yes |
| T 15/80/4.0/1.5 | -0.19 | 0.001 | 0.000 | 1.4% | -0.21 | no |
| T 20/60/4.0/1.5 | 0.13 | 0.018 | 0.005 | 1.9% | 0.15 | no |
| T 30/80/3.0/1.2 | 0.34 | 0.077 | 0.030 | 2.1% | 0.38 | no |
| T 40/80/2.0/1.2 | 0.48 | 0.157 | 0.070 | 2.2% | 0.54 | no |
| T 15/100/4.0/1.2 | -0.26 | 0.000 | 0.000 | 1.2% | -0.29 | yes |
| T 40/60/3.0 | 0.69 | 0.378 | 0.218 | 4.7% | 0.71 | no |
| T 15/100/4.0/1.5 | -0.26 | 0.000 | 0.000 | 1.2% | -0.29 | yes |

Walk-forward over the search front picks up a mean 0.00 Sharpe from full-window to out-of-sample, and the select-then-trade path over the front returns OOS Sharpe 0.78. The gate's binding criterion, P(OOS loss), is 0.345 over the 180 genomes the search evaluated and 0.268 over the whole field (gate: <= 0.5) — the search's population is a selected sample, so its number is not the field's and neither licenses the other. PBO is advisory here: 0.387 evaluated, 0.366 over the field, with pairwise return correlation across the field of mean 0.33 / min -0.00. Neighbouring EMA pairs trade the same crosses a few bars apart, so the cells are near-duplicates rather than independent bets and CSCV's in-sample winner pick is closer to random than the cell count suggests.

## The live cell

T 20/100/3.0 is what the live config trades. In the exhaustive field it ranks 4 of 390 on DSR (0.220) and 23 of 390 on walk-forward OOS Sharpe (0.71), with full-window Sharpe 0.70 and max drawdown 4.1%. It is NOT on the exhaustive Pareto front and it FAILS the gate as one trial among 390. That is a finding about the evidence behind the cell, not a signal to trade differently today.

Both runs, the seed and the per-generation trace: `trend-search-2026-09-14.json`. The search replays from that file alone. This is an experiment record, not a promotion artifact — the docs/validation/ verdicts are unaffected.
