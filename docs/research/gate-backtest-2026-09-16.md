# Promotion-gate calibration — 2026-09-16

Over 28 rows the gate inputs rank forward Sharpe at P(OOS loss) +0.16, DSR -0.06, PBO +0.24 (Spearman); PBO carries the most of that ranking at +0.24 over 28 rows, +0.32 over the 23 rows that traded. A 0.05 ceiling separates passers from failers by +0.05 Sharpe against +0.05 at the live 0.5, changing 0 of 28 verdicts.

The gate (0.95 DSR floor, 0.5 P(OOS loss) ceiling, positive walk-forward OOS Sharpe) was replayed at 6 cutoffs over research's own shortlists, on bars truncated to each cutoff and the sip feed. The variant the gate selected was then traded over the following 365 days, warmed on the history the gate saw.

## Rows

| cutoff | symbol | variant | DSR | P(OOS loss) | PBO | OOS | gate | forward Sharpe | forward return | fwd trades / bars held | B&H Sharpe | warm-up flat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2019-12-31 | GLD | EMA 20/100 long | 0.765 | 0.199 | 0.909 | 0.75 | REJECT | 0.00 | +0.0% | 0 / 0 | 1.14 | 2 |
| 2019-12-31 | VOO | TOM 2/5 | 0.914 | 0.065 | 0.966 | 0.66 | REJECT | 1.29 | +4.0% | 13 / 80 | 0.63 | 0 |
| 2020-12-31 | SPY | TSMOM 12-1 | 0.389 | 0.525 | 0.651 | -0.03 | REJECT | 2.08 | +8.2% | 0 / 251 | 2.13 | 3 |
| 2020-12-31 | QQQ | TSMOM ens 6-12m | 0.688 | 0.208 | 0.820 | 0.63 | REJECT | 1.46 | +7.2% | 0 / 251 | 1.51 | 2 |
| 2020-12-31 | GLD | EMA 10/50 long | 0.798 | 0.155 | 0.741 | 0.75 | REJECT | 0.17 | +0.4% | 3 / 70 | -0.41 | 2 |
| 2020-12-31 | VOO | TOM 3/3 | 0.952 | 0.002 | 0.785 | 1.01 | PROMOTE | 1.57 | +3.7% | 12 / 69 | 2.15 | 0 |
| 2021-12-31 | SPY | TSMOM 12-1 | 0.422 | 0.278 | 0.570 | 0.44 | REJECT | -1.72 | -4.4% | 9 / 105 | -0.74 | 1 |
| 2021-12-31 | QQQ | TSMOM ens 6-12m | 0.721 | 0.156 | 0.907 | 0.49 | REJECT | -1.95 | -4.6% | 8 / 71 | -1.10 | 1 |
| 2021-12-31 | GLD | EMA 10/50 long | 0.585 | 0.149 | 0.449 | 0.47 | REJECT | 0.74 | +0.9% | 0 / 34 | 0.13 | 2 |
| 2021-12-31 | USO | EMA 20/100 long | 0.493 | 0.174 | 0.334 | 0.35 | REJECT | 0.00 | +0.0% | 0 / 0 | 0.80 | 3 |
| 2021-12-31 | VOO | TOM 2/5 | 0.988 | 0.000 | 0.685 | 0.96 | PROMOTE | -1.17 | -2.5% | 12 / 80 | -0.74 | 0 |
| 2022-12-31 | SPY | RSI2<15 200sma | 0.554 | 0.318 | 0.609 | -0.20 | REJECT | 0.23 | +0.4% | 14 / 44 | 1.90 | 1 |
| 2022-12-31 | QQQ | TSMOM ens 6-12m | 0.679 | 0.191 | 0.933 | -0.00 | REJECT | 1.91 | +5.8% | 0 / 179 | 2.60 | 1 |
| 2022-12-31 | GLD | EMA 10/50 long | 0.545 | 0.104 | 0.630 | 0.75 | REJECT | -0.36 | -0.9% | 3 / 74 | 0.91 | 2 |
| 2022-12-31 | USO | EMA 20/100 long | 0.586 | 0.052 | 0.142 | 0.87 | REJECT | 0.34 | +0.6% | 2 / 63 | 0.11 | 3 |
| 2022-12-31 | VOO | TOM 3/3 | 0.973 | 0.011 | 0.945 | 0.67 | PROMOTE | 1.38 | +2.6% | 12 / 70 | 1.91 | 0 |
| 2023-12-31 | SPY | TSMOM 12-1 | 0.734 | 0.340 | 0.735 | 0.21 | REJECT | 1.85 | +10.6% | 0 / 250 | 1.92 | 1 |
| 2023-12-31 | QQQ | EMA 50/200 long | 0.944 | 0.092 | 0.933 | 0.35 | REJECT | 0.00 | +0.0% | 0 / 0 | 1.51 | 1 |
| 2023-12-31 | BTC/USD | EMA 50/200 long | 0.919 | 0.042 | 0.507 | 0.52 | REJECT | 0.00 | +0.0% | 0 / 0 | 1.38 | 3 |
| 2023-12-31 | GLD | EMA 10/50 long | 0.575 | 0.141 | 0.608 | 0.51 | REJECT | 1.28 | +4.1% | 2 / 64 | 1.63 | 1 |
| 2023-12-31 | USO | EMA 20/100 long | 0.701 | 0.115 | 0.280 | 0.36 | REJECT | 0.01 | +0.0% | 2 / 74 | 0.62 | 1 |
| 2023-12-31 | VOO | TOM 3/3 | 0.983 | 0.001 | 0.976 | 0.76 | PROMOTE | 0.04 | +0.1% | 13 / 67 | 1.93 | 0 |
| 2024-12-31 | SPY | TSMOM ens 6-12m | 0.813 | 0.114 | 0.725 | 0.20 | REJECT | 1.04 | +3.4% | 5 / 236 | 0.95 | 1 |
| 2024-12-31 | QQQ | EMA 50/200 long | 0.950 | 0.009 | 0.952 | 0.43 | PROMOTE | 1.51 | +2.1% | 1 / 101 | 0.93 | 1 |
| 2024-12-31 | BTC/USD | EMA 50/200 long | 0.938 | 0.235 | 0.883 | 0.99 | REJECT | 0.00 | +0.0% | 0 / 0 | 0.02 | 2 |
| 2024-12-31 | GLD | EMA 10/50 long | 0.666 | 0.112 | 0.369 | 0.37 | REJECT | 1.47 | +3.8% | 1 / 60 | 2.54 | 1 |
| 2024-12-31 | USO | EMA 20/100 long | 0.664 | 0.082 | 0.152 | 0.62 | REJECT | -0.28 | -0.5% | 2 / 18 | -0.20 | 1 |
| 2024-12-31 | VOO | TOM 2/5 | 0.986 | 0.000 | 0.544 | 0.44 | PROMOTE | -0.42 | -1.0% | 14 / 79 | 1.00 | 0 |

## Ceilings

Each candidate ceiling read against the live 0.5. The DSR bar alone, which no ceiling moves, splits 6/22 at a separation of +0.05 Sharpe — the comparison that says whether the loss-probability criterion or the DSR floor is carrying the forward signal.

The rank correlations in the headline are ceiling-independent: a ceiling moves which rows land on each side of the gate, not how the gate's inputs rank forward Sharpe.

| ceiling | pass | fail | mean fwd Sharpe (pass) | mean fwd Sharpe (fail) | separation | verdicts changed | P(OOS loss) alone: pass/fail | separation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 6 | 22 | 0.49 | 0.43 | +0.05 | 0 | 7/21 | -0.04 |
| 0.1 | 6 | 22 | 0.49 | 0.43 | +0.05 | 0 | 11/17 | -0.09 |
| 0.2 | 6 | 22 | 0.49 | 0.43 | +0.05 | 0 | 22/6 | -0.26 |
| 0.3 | 6 | 22 | 0.49 | 0.43 | +0.05 | 0 | 25/3 | -1.05 |
| 0.5 | 6 | 22 | 0.49 | 0.43 | +0.05 | 0 | 27/1 | -1.70 |

The same read with the 5 rows whose variant never held a position dropped. Over the 23 rows whose selected variant was in the market, the gate inputs rank forward Sharpe at P(OOS loss) +0.23, DSR -0.04, PBO +0.32 (Spearman), and the DSR bar alone splits 6/17 at -0.08 Sharpe.

| ceiling | pass | fail | mean fwd Sharpe (pass) | mean fwd Sharpe (fail) | separation | verdicts changed | P(OOS loss) alone: pass/fail | separation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 6 | 17 | 0.49 | 0.56 | -0.08 | 0 | 6/17 | -0.08 |
| 0.1 | 6 | 17 | 0.49 | 0.56 | -0.08 | 0 | 9/14 | -0.11 |
| 0.2 | 6 | 17 | 0.49 | 0.56 | -0.08 | 0 | 18/5 | -0.30 |
| 0.3 | 6 | 17 | 0.49 | 0.56 | -0.08 | 0 | 20/3 | -0.97 |
| 0.5 | 6 | 17 | 0.49 | 0.56 | -0.08 | 0 | 22/1 | -1.61 |

## Skipped cells

Not scored, and not counted as rejections: a cell with no variant able to warm up, or no forward window to measure, is absent evidence rather than a gate that said no.

- 2019-12-31 SPY: shortest warm-up 210 bars exceeds the 201-bar first train block
- 2019-12-31 QQQ: shortest warm-up 210 bars exceeds the 201-bar first train block
- 2019-12-31 BTC/USD: 0 history bars, CSCV needs 32
- 2019-12-31 USO: shortest warm-up 292 bars exceeds the 201-bar first train block
- 2020-12-31 BTC/USD: 0 history bars, CSCV needs 32
- 2020-12-31 USO: shortest warm-up 292 bars exceeds the 251-bar first train block
- 2021-12-31 BTC/USD: shortest warm-up 200 bars exceeds the 73-bar first train block
- 2022-12-31 BTC/USD: shortest warm-up 200 bars exceeds the 146-bar first train block

## Caveat

n is 28 rows, and they are overlapping rather than independent draws: consecutive cutoffs share most of their in-sample history, and adjacent forward windows cover neighbouring years of the same few instruments. Read this as a calibration read, not a statistical proof — it says which direction the evidence points, not where a ceiling belongs.

A row whose selected variant never held a position in the forward window scores Sharpe 0.00 — a variant that never entered, not one that traded to a flat year — and enters the rank correlations as skill 0. 5 of 28 rows read that way here; the `fwd trades / bars held` column is the count, and the traded-only tables above are the same calibration with those rows dropped. The two halves of that column differ: a variant still holding when the window ended closed no round trip, and bars held counts the bars it carried a position into its own next decision. A variant that entered on the final bar has neither: there is no held bar to measure and its forward Sharpe is the slippage on that one fill, so the row is a no-trade row and the entry count is appended to the column as disclosure.

Dropping those rows conditions on a forward-window outcome — whether the variant ever took a position, which is only known after the fact — so the traded-only read is not an unbiased estimate of the same quantity as the all-rows read. It answers a narrower question: how the gate's inputs rank forward Sharpe among the cells where a position was taken.

The `warm-up flat` column counts the shortlist variants whose warm-up outruns walk-forward's first train block at that cutoff (the block length is `first_train_bars` in the JSON, the labels `warmup_flat_variants`). Rows with warm-up-flat variants replay research's own behaviour, where those variants cannot be selected in the earliest training folds.

Per-row numbers: `gate-backtest-2026-09-16.json`. This is an experiment record, not a promotion artifact — the docs/validation/ verdicts are unaffected.
