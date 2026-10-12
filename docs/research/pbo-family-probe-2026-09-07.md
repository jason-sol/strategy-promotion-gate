# Does PBO measure overfitting, or family homogeneity? — 2026-09-07

**DECIDED 2026-09-07 (Jason): B + C implemented.** The promotion gate is now
`p_oos_loss <= 0.5`, PBO is reported advisory beside the other CSCV panels, the
artifacts carry the per-candidate view, and grid/search records persist the
field's correlation summary.

The 0.5 threshold is the paper's reference point for "no better than random",
not a measured operating point — and the synthetic no-edge fields put P(OOS
loss) anywhere in 0.01–0.89, which suggests 0.5 is lenient. Revisit after the
first real-data re-run.

On a synthetic field of 30 near-duplicate parameter variants, CSCV's PBO is a coin flip whether the family has a true Sharpe of 1.0 or of exactly zero: 0.491 vs 0.495 over 20 seeds, std 0.21. PBO ranks the real-edge family lower in 40% of seeds — worse than guessing. The paper's other three outputs, which the repo computes and throws away, separate the same two fields in 95% of seeds. The failure mode is not correlation: scenario (e) is a correlated family whose members differ in *true* skill, and PBO reads 0.006. PBO degenerates when the members have near-identical **true** skill, which is exactly what an overlapping parameter window produces.

Reproduce: `uv run python docs/research/scripts/pbo_family_probe.py` (fixed seeds 0-19, T=1500, N=30, n_blocks=16, 12870 splits; the vectorised replica asserts parity with `bot.validation.cscv_pbo` on all 100 runs).

## What the gate did before this decision

`cscv_pbo` (bot/validation.py:191) returns one float and discards every per-split statistic. `passes_validation` (config.py:301-308) ANDs `pbo <= pbo_max` (0.5) with the per-candidate DSR and OOS Sharpe. optimize.py:286 and search.py:447 pass that one field-level scalar into every cell's `passes`, so a single number closes all 35 TOM cells and all 130 trend cells. research.py:143 uses the same scalar for the live promotion gate.

## The measurement

| Scenario | PBO | P(OOS loss) | mean OOS SR | median logit | degradation slope |
| --- | --- | --- | --- | --- | --- |
| (a) corr family ρ≈0.97, true SR 1.0 | 0.491 | 0.100 | +0.87 | 0.15 | −0.99 |
| (b) corr family ρ≈0.97, true SR 0 | 0.495 | 0.467 | +0.06 | 0.11 | −1.00 |
| (c) 30 independent noise | 0.468 | 0.439 | +0.08 | 0.21 | −0.48 |
| (d) 30 independent + 1 genuinely good | 0.167 | 0.155 | +1.11 | 2.42 | −0.37 |
| (e) corr family with a true optimum | 0.006 | 0.102 | +0.80 | 1.68 | −1.00 |

PBO's sampling distribution over 20 seeds: (a) mean 0.491, std 0.212, range 0.133–0.830; (b) 0.495 / 0.163; (c) 0.468 / 0.171; (d) 0.167 / 0.156; (e) 0.006 / 0.009. A std near 0.2 puts the observed TOM field PBO of 0.608 and trend field PBO of 0.533 inside the noise band around 0.5 — neither is evidence of anything.

## Against the source

Bailey, Borwein, López de Prado & Zhu (SSRN 2326253) present CSCV as a **four-panel diagnostic**: PBO from the logit distribution, performance degradation, probability of OOS loss, and stochastic dominance. `PBO <= 0.5` as a hard promote/reject line is a house rule, not the paper's; 0.5 is the reference point for "no better than random". The field's remedy for near-duplicate candidates is effective-number-of-trials clustering — the same logic already applied in DSR.

## Options

**A — pre-cluster by correlation, run CSCV over medoids.** Measured on the synthetic families: a threshold of ρ<0.90 or ρ<0.95 collapses a 30-member family to a single representative, PBO becomes NaN, and NaN passes through the gate. ρ<0.99 dedupes nothing. On single-strategy grids A is "gate off, honestly labelled". It keeps its bite on multi-strategy shortlists, and the threshold becomes a promotion knob.

**B — demote PBO to advisory.** `cscv_pbo` → `cscv_report` returning PBO, median logit, degradation slope, P(OOS loss) and the logit histogram; drop `pbo <= pbo_max` from `passes_validation`; add `p_oos_loss <= 0.5`. This is *stricter* on family fields, not looser — but it removes a guard that currently bites correctly on heterogeneous fields (live QQQ, PBO 0.962).

**C — additive per-candidate OOS rank distribution for the chosen cell.** Nothing loosens; grids still REJECT for a reason the data says is noise.

Recommendation: **B with C folded in.** Prerequisite either way: grid artifacts store no return matrix, so the correlation structure of the *real* fields cannot be checked offline. Persist a correlation summary in the artifacts first.

## Live impact

None on item 10 — the grids are experiment records and the docs/validation verdicts are unaffected. Live shortlists are heterogeneous by construction, the (c)/(d) shape where PBO is informative. QQQ is the only live symbol where PBO is the sole binding constraint (DSR 0.984, OOS 0.618, PBO 0.962) and 0.962 is not a coin flip.

VOO — the only live PROMOTE — is the one homogeneous shortlist: three overlapping TOM variants, already flagged in research.py:100-101. Its PBO of 0.407 is the least trustworthy number in the artifact and carries no information about whether the edge is real. Fixing this makes VOO's promotion harder to defend, not easier.
