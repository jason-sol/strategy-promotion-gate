# validation-harness

Backtest-overfitting validation with no scipy dependency. pandas and numpy are
the only runtime requirements — the normal CDF comes from `math.erf` and its
inverse from Acklam's rational approximation.

> The distribution name `validation-harness` is provisional and unclaimed on any
> index. It may change before publication.

## What it does

Three gates over a set of candidate strategy variants, all pure and network-free:

- **Deflated Sharpe with honest trial counts.** The PSR benchmark is raised to
  the expected maximum Sharpe under the null across every variant tried, so a
  headline Sharpe has to clear what selection alone would have produced.
- **CSCV as a set-level report.** Every symmetric in-sample/out-of-sample split
  of the return matrix is scored, and the resulting panels are properties of the
  search that produced the variants rather than of any one column, so they judge
  the family rather than the winner.
- **Select-then-trade walk-forward.** Each fold picks the in-sample winner and
  scores only that choice on the immediately-following out-of-sample window, so
  the reported out-of-sample record includes the cost of choosing wrong.

## Install

Not published yet. `validation-harness` is the intended distribution name and is
unclaimed on PyPI, but the name is only reserved by a first upload — see
[PUBLISHING.md](PUBLISHING.md).

```sh
pip install validation-harness   # once published
uv add validation-harness        # once published
```

From a checkout:

```sh
pip install -e packages/validation-harness
```

## Use

A `Candidate` is a label, the symbol whose bars it reads, and a function that
simulates it and returns an equity curve. The harness never simulates anything
itself — bring your own backtester behind that one function.

```python
import pandas as pd
from validation_harness import (
    Candidate,
    build_return_matrix,
    cscv_report,
    deflated_sharpes_from_matrix,
)

def simulate(bars: pd.DataFrame, sim_start: pd.Timestamp) -> pd.Series:
    """Return an equity curve indexed by timestamp. Bars before sim_start are
    warmup, not trading."""
    ...

candidates = [Candidate(f"z={z}", "AAA", make_runner(z)) for z in (1.0, 1.5, 2.0)]
matrix = build_return_matrix(candidates, {"AAA": bars})

print(deflated_sharpes_from_matrix(matrix, n_trials=len(candidates)))

report = cscv_report(matrix, n_blocks=16)
report.pbo                 # splits whose in-sample winner ranks bottom half OOS
report.p_oos_loss          # splits whose in-sample winner has a negative OOS Sharpe
report.median_logit        # median logit of the winner's relative OOS rank
report.degradation_slope   # OLS slope of the winner's OOS Sharpe on its IS Sharpe
report.logits              # one logit per split, for a histogram
report.oos_rank_mean["z=1.5"]   # that candidate's mean relative OOS rank
report.is_best_share["z=1.5"]   # how often it won in-sample
```

`cscv_pbo(matrix, n_blocks=16)` returns `report.pbo` on its own.

Walk-forward takes one frame shared by every candidate:

```python
from validation_harness import walk_forward

result = walk_forward(candidates, bars, n_folds=4)
result.oos_sharpe           # Sharpe of the select-then-trade path
result.variant_oos_sharpe   # per-candidate concatenated OOS Sharpe
result.folds                # train/test boundaries and the pick per fold
```

Not-computable guards return `float("nan")` — and, for the report's
per-candidate panels, empty maps — so test results with `math.isnan` rather than
comparing against a sentinel.

### Reading the CSCV report

Bailey, Borwein, López de Prado & Zhu present CSCV as a diagnostic to be read
across panels, not as a single number. `pbo` says how often the in-sample winner
falls to the bottom half out-of-sample and `median_logit` says by how much;
`p_oos_loss` says how often that winner simply loses money out-of-sample; and
`degradation_slope` says how much of an in-sample Sharpe survives the split — the
steeper it is, the more of the in-sample edge was selection. `pbo` alone cannot
tell a family of near-duplicate variants sharing a real edge from the same
family sharing none, because near-identical true skill leaves the rank statistic
to the noise either way, while `p_oos_loss` reads the shared signal and
separates the two. The per-candidate panels cover every split, so
`oos_rank_mean` and `is_best_share` answer whether one chosen candidate's
out-of-sample standing is stable rather than an artefact of one lucky split.

## API

| Name | Purpose |
| --- | --- |
| `Candidate` | label + symbol + equity-curve function |
| `build_return_matrix` | T x N daily-return matrix over the shared span |
| `deflated_sharpes` / `deflated_sharpes_from_matrix` | DSR per candidate |
| `cscv_report` | every CSCV panel over the splits, plus the per-candidate view |
| `cscv_pbo` | the probability-of-backtest-overfitting panel on its own |
| `CscvReport` | the CSCV result record |
| `walk_forward` | rolling select-then-trade folds |
| `FoldResult`, `WalkForwardResult` | walk-forward result records |
| `sharpe_ratio`, `probabilistic_sharpe_ratio`, `expected_max_sharpe`, `deflated_sharpe_ratio` | the underlying statistics |

## Caveats

Read these before treating any number here as a verdict.

- **Serial dependence.** A position spanning a CSCV block boundary correlates
  the in-sample and out-of-sample blocks it straddles, so the splits are not
  fully independent and the direction of the resulting PBO bias is unclear.
- **Correlated families.** Correlated parameter variants overcount `n_trials`,
  and the `ddof=1` variance across trials runs larger than `ddof=0`; both push
  the deflation benchmark up and so understate DSR. The bias is conservative,
  which is the intended direction for a promotion gate but not a neutral
  estimate. Near-duplicate variants also flatten `pbo`
  towards 0.5 whether or not the family's shared edge is real, so read
  `p_oos_loss` alongside it.
- **Causality is assumed, not checked.** Walk-forward truncates the frame so a
  training run cannot reach test bars, but a candidate whose simulation reads
  forward within its own window scores clean here and has to be caught
  elsewhere.

## Development

```sh
pytest packages/validation-harness
ruff check packages/validation-harness
```

Release steps live in [PUBLISHING.md](PUBLISHING.md).

## Licence

MIT — see [LICENSE](LICENSE).
