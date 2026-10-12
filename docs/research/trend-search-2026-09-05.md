# USO EMA-crossover trend search vs exhaustive truth — 2026-09-05

An NSGA-II-lite search (seed 0, population 12, 5 generations, budget 60) spent 38 unique backtests over the same genome space the exhaustive grid enumerates in 130 cells (fast_period 5-50 x slow_period 40-100 x trail_atr_mult 2.0-4.0), on daily USO bars from 2016-01-01 on the sip feed. Objectives: maximize Deflated Sharpe, minimize max drawdown. Every search DSR is deflated at the search's OWN cumulative trial count, so the deflation tightens as it spends the budget; the exhaustive field deflates at 130.

## Did the search find the truth

The exhaustive Pareto front has 13 cells; the search's front has 7, of which 6 are exactly on the true front — coverage 46% of the true front, precision 86% of the search's. The search's DSR-best cell is T 30/60/3.0 at DSR 0.895 on its own 38 trials; the same cell deflated against the whole 130-cell field is 0.881, against the field's best T 30/60/3.0 at 0.881 — a DSR regret of 0.000. The cheap search reports a bigger number for the same backtest; the trial count is the only thing that makes the two comparable.

| Cell | Sharpe | DSR @ 38 trials | DSR @ 130 trials | maxDD | OOS Sharpe | On true front |
| --- | --- | --- | --- | --- | --- | --- |
| T 40/80/4.0 | 0.50 | 0.706 | 0.679 | 3.7% | 0.47 | no |
| T 40/80/2.0 | 0.48 | 0.680 | 0.652 | 2.7% | 0.58 | yes |
| T 30/80/2.0 | 0.65 | 0.837 | 0.818 | 3.7% | 0.81 | yes |
| T 30/60/2.0 | 0.71 | 0.873 | 0.858 | 4.0% | 0.91 | yes |
| T 20/100/3.0 | 0.70 | 0.880 | 0.863 | 4.1% | 0.71 | yes |
| T 40/60/2.0 | 0.66 | 0.842 | 0.824 | 3.8% | 0.83 | yes |
| T 30/60/3.0 | 0.72 | 0.895 | 0.881 | 4.7% | 0.77 | yes |

Walk-forward over the search front picks up a mean 0.09 Sharpe from full-window to out-of-sample, and the select-then-trade path over the front returns OOS Sharpe 0.61. PBO is 0.627 over the 38 genomes the search evaluated and 0.533 over the whole field (gate: <= 0.5) — the search's population is a selected sample, so its PBO is not the field's and neither number licenses the other.

## The live cell

T 20/100/3.0 is what the live config trades. In the exhaustive field it ranks 2 of 130 on DSR (0.863) and 19 of 130 on walk-forward OOS Sharpe (0.71), with full-window Sharpe 0.70 and max drawdown 4.1%. It is ON the exhaustive Pareto front and it FAILS the gate as one trial among 130. That is a finding about the evidence behind the cell, not a signal to trade differently today.

Both runs, the seed and the per-generation trace: `trend-search-2026-09-05.json`. The search replays from that file alone. This is an experiment record, not a promotion artifact — the docs/validation/ verdicts are unaffected.
