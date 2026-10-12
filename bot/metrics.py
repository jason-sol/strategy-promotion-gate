"""Performance metrics for backtest reporting."""

import math
from dataclasses import dataclass

import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# Rounding noise in a constant series runs a couple of ulps, ~1e-16 relative.
_FLAT_RELATIVE_TOLERANCE = 1e-12


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline as a fraction of the peak."""
    if equity.empty:
        return 0.0
    dd = 1 - equity / equity.cummax()
    return float(dd.max())


def _is_flat(std: float, mean: float) -> bool:
    """A constant series carries rounding noise rather than an exactly zero std,
    so dispersion counts only once it clears the rounding scale of the values.
    A zero mean makes that scale zero, which is the exact-equality test."""
    return std <= abs(mean) * _FLAT_RELATIVE_TOLERANCE


def sharpe_ratio(daily_returns: pd.Series) -> float:
    returns = daily_returns.dropna()
    if len(returns) < 2:
        return 0.0
    std = returns.std()
    mean = returns.mean()
    if _is_flat(std, mean):
        return 0.0
    return float(mean / std * math.sqrt(TRADING_DAYS_PER_YEAR))


_EULER_MASCHERONI = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF via Acklam's rational approximation."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1)")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """Bailey & Lopez de Prado PSR on per-period (non-annualized) returns."""
    returns = returns.dropna()
    t = len(returns)
    std = returns.std()
    mean = returns.mean()
    if t < 3 or math.isnan(std) or _is_flat(std, mean):
        return 0.0
    sr = mean / std
    skew = returns.skew()
    kurt = returns.kurt() + 3.0  # non-excess kurtosis
    if math.isnan(sr) or math.isnan(skew) or math.isnan(kurt):
        return 0.0
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return 0.0
    z = (sr - sr_benchmark) * math.sqrt(t - 1) / math.sqrt(denom)
    return float(_norm_cdf(z))


def sharpe_standard_error(returns: pd.Series) -> float:
    """Annualized standard error of the Sharpe ratio under non-normal returns
    (Mertens 2002): the same variance term PSR divides by."""
    returns = returns.dropna()
    t = len(returns)
    std = returns.std()
    mean = returns.mean()
    if t < 3 or math.isnan(std) or _is_flat(std, mean):
        return math.nan
    sr = mean / std
    skew = returns.skew()
    kurt = returns.kurt() + 3.0
    variance = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) / (t - 1)
    if not variance > 0:
        return math.nan
    return float(math.sqrt(variance) * math.sqrt(TRADING_DAYS_PER_YEAR))


def expected_max_sharpe(n_trials: int, var_across_trials: float) -> float:
    """Expected maximum per-period Sharpe under the null across n_trials."""
    if n_trials <= 1 or var_across_trials <= 0:
        return 0.0
    p1 = 1.0 - 1.0 / n_trials
    p2 = 1.0 - 1.0 / (n_trials * math.e)
    if p1 >= 1.0 or p2 >= 1.0:  # trial count so large the quantiles round to 1.0
        return 0.0
    g = _EULER_MASCHERONI
    z = (1.0 - g) * _norm_ppf(p1) + g * _norm_ppf(p2)
    return float(math.sqrt(var_across_trials) * z)


def deflated_sharpe_ratio(
    returns: pd.Series, n_trials: int, var_across_trials: float
) -> float:
    benchmark = expected_max_sharpe(n_trials, var_across_trials)
    return float(probabilistic_sharpe_ratio(returns, benchmark))


@dataclass(frozen=True)
class TradeStats:
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_pnl: float


def summarize_trades(pnls: list[float]) -> TradeStats:
    if not pnls:
        return TradeStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    return TradeStats(
        total_trades=len(pnls),
        win_rate=len(wins) / len(pnls),
        avg_win=gross_win / len(wins) if wins else 0.0,
        avg_loss=-gross_loss / len(losses) if losses else 0.0,
        profit_factor=gross_win / gross_loss if gross_loss > 0 else float("inf"),
        total_pnl=sum(pnls),
    )
