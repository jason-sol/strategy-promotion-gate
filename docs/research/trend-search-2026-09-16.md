# USO EMA-crossover trend search vs exhaustive truth — 2026-09-16

An NSGA-II-lite search (seed 0, population 36, budget 180 spent in 8 generations (ceiling 40)) covered 180 unique backtests over the same genome space the exhaustive grid enumerates in 390 cells (fast_period 5-50 x slow_period 40-100 x trail_atr_mult 2.0-4.0 x volume_gate_mult None-1.5), on daily USO bars from 2016-01-01 on the sip feed. Objectives: maximize Deflated Sharpe, minimize max drawdown. Every search DSR is deflated at the search's OWN cumulative trial count, so the deflation tightens as it spends the budget; the exhaustive field deflates at 390.

## Did the search find the truth

The exhaustive Pareto front has 8 cells; the search's front has 7, of which 7 are exactly on the true front — coverage 88% of the true front, precision 100% of the search's. The search's DSR-best cell is T 30/60/2.0 at DSR 0.444 on its own 180 trials; the same cell deflated against the whole 390-cell field is 0.314, against the field's best T 30/60/2.0 at 0.314 — a DSR regret of 0.000. The cheap search reports a bigger number for the same backtest; the trial count is the only thing that makes the two comparable. Both fronts are built over cells with Sharpe > 0; 121 field cells and 53 evaluated cells fall under the floor.

| Cell | Sharpe | DSR @ 180 trials | DSR @ 390 trials | maxDD | OOS Sharpe | On true front |
| --- | --- | --- | --- | --- | --- | --- |
| T 15/80/4.0/1.2 | 0.03 | 0.005 | 0.002 | 1.4% | 0.03 | yes |
| T 40/80/3.5 | 0.57 | 0.204 | 0.114 | 3.6% | 0.53 | yes |
| T 30/80/2.0/1.2 | 0.53 | 0.169 | 0.091 | 1.8% | 0.59 | yes |
| T 30/60/2.0 | 0.78 | 0.444 | 0.314 | 4.0% | 0.93 | yes |
| T 20/100/2.0 | 0.66 | 0.304 | 0.193 | 3.8% | 0.75 | yes |
| T 30/80/2.0 | 0.65 | 0.301 | 0.191 | 3.7% | 0.75 | yes |
| T 40/60/2.0 | 0.66 | 0.312 | 0.200 | 3.8% | 0.77 | yes |

Walk-forward over the search front picks up a mean 0.07 Sharpe from full-window to out-of-sample, and the select-then-trade path over the front returns OOS Sharpe 0.81. The gate's binding criterion, P(OOS loss), is 0.252 over the 180 genomes the search evaluated and 0.268 over the whole field (gate: <= 0.5) — the search's population is a selected sample, so its number is not the field's and neither licenses the other. PBO is advisory here: 0.368 evaluated, 0.366 over the field, with pairwise return correlation across the field of mean 0.33 / min -0.00. Neighbouring EMA pairs trade the same crosses a few bars apart, so the cells are near-duplicates rather than independent bets and CSCV's in-sample winner pick is closer to random than the cell count suggests.

## The live cell

T 20/100/3.0 is what the live config trades. In the exhaustive field it ranks 4 of 390 on DSR (0.217) and 25 of 390 on walk-forward OOS Sharpe (0.67), with full-window Sharpe 0.70 and max drawdown 4.1%. It is NOT on the exhaustive Pareto front and it FAILS the gate as one trial among 390. That is a finding about the evidence behind the cell, not a signal to trade differently today.

Both runs, the seed and the per-generation trace: `trend-search-2026-09-16.json`. The search replays from that file alone. This is an experiment record, not a promotion artifact — the docs/validation/ verdicts are unaffected.
