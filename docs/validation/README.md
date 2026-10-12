# Validation artifacts

One `validation-<date>.json` per `python -m bot.research` run: the selected
variant per symbol with its params, `dsr`, `oos_sharpe`, `variant_oos`, the four
CSCV panels (`p_oos_loss`, `pbo`, `median_logit`, `degradation_slope`), the
selected variant's own CSCV standing (`variant_oos_rank_mean`,
`variant_is_best_share`), `variants` and the PROMOTE/REJECT verdict. These files
are the citable evidence behind `InstrumentConfig.validation_ref` — they are
committed so a citation still resolves in a fresh clone. A re-run on the same
date overwrites that date's file.

`variants` is every variant on the symbol's shortlist, in shortlist order, each
row keyed by `variant` and carrying that variant's own `dsr` and walk-forward
`oos_sharpe` — the rows the run prints. The selected variant is among them, with
the same numbers the entry's `dsr` and `variant_oos` carry.

Mind the two `oos_sharpe` keys. Inside a `variants` row it is that variant's own
walk-forward OOS Sharpe. At the top level of an entry it is the symbol's
select-then-trade OOS Sharpe — select per fold, then trade the selection — which
is a different number and the one the verdict gates on.

The key reads in one direction only. Present, it settles whether a variant was
measured: a shortlist entry that was tried and lost is there beside the one that
won. Absent, it settles nothing — version-2 files written before the table
existed record only the selection, and an `ERROR` entry records no measurement
at all — so a missing `variants` means "not recorded", never "not measured".

`verdict` is **DSR ≥ 0.95, P(OOS loss) ≤ 0.5, walk-forward OOS Sharpe > 0**
(`ValidationConfig.passes_validation`). `pbo` is reported advisory and does not
gate: it cannot separate a family of near-duplicate candidates that share a real
edge from one that shares none, because in both CSCV's in-sample winner pick is
near-random (`docs/research/pbo-family-probe-2026-09-07.md`).

`variants` was added to version 2 without a bump: it is additive and every other
key keeps its meaning, so a reader written against an earlier version-2 file
still reads a newer one.

`schema_version` 2 is the P(OOS loss) gate. Version 1 gated on `pbo <= 0.5`, so a
`verdict` read out of a version-1 file answers a different question and the two
are not comparable.
