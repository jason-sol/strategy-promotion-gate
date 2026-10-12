"""Scipy-free backtest-overfitting validation.

Three gates over a set of candidate strategy variants, all pure and
network-free, with pandas and numpy as the only runtime dependencies:

- Deflated Sharpe with honest trial counts. The PSR benchmark is raised to the
  expected maximum Sharpe under the null across every variant tried, so a
  headline Sharpe has to clear what selection alone would have produced.
- CSCV as a set-level report. Every symmetric in-sample/out-of-sample split is
  scored, and the panels — probability of backtest overfitting, probability the
  in-sample winner loses out-of-sample, the logit distribution, the
  performance-degradation slope, each candidate's out-of-sample standing — are
  properties of the search that produced the variants rather than of any one
  column.
- Select-then-trade walk-forward. Each fold picks the in-sample winner and
  scores only that choice on the immediately-following out-of-sample window, so
  the reported out-of-sample record includes the cost of choosing wrong.

Caveats worth carrying into any read of the output:

- Serial dependence. A position spanning a CSCV block boundary correlates the
  in-sample and out-of-sample blocks it straddles, so the splits are not fully
  independent and the direction of the resulting PBO bias is unclear.
- Correlated families. Correlated parameter variants overcount n_trials, and
  the ddof=1 variance across trials runs larger than ddof=0; both push the
  deflation benchmark up and so understate DSR. The bias is conservative, which
  is the intended direction for a promotion gate but not a neutral estimate.
  Near-duplicate variants also flatten PBO towards 0.5 whether or not the
  family's shared edge is real, so read p_oos_loss alongside it.
- Causality is assumed, not checked. Walk-forward truncates the frame so a
  training run cannot reach test bars, but a candidate that reads forward
  within its own window scores clean here.
"""

from .metrics import (
    TRADING_DAYS_PER_YEAR,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from .validation import (
    Candidate,
    CscvReport,
    EquityCurve,
    FoldResult,
    WalkForwardResult,
    build_return_matrix,
    cscv_pbo,
    cscv_report,
    deflated_sharpes,
    deflated_sharpes_from_matrix,
    walk_forward,
)

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "Candidate",
    "CscvReport",
    "EquityCurve",
    "FoldResult",
    "WalkForwardResult",
    "build_return_matrix",
    "cscv_pbo",
    "cscv_report",
    "deflated_sharpe_ratio",
    "deflated_sharpes",
    "deflated_sharpes_from_matrix",
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
    "sharpe_ratio",
    "walk_forward",
]

__version__ = "0.1.0"
