import math

import numpy as np
import pandas as pd
import pytest

from bot.metrics import (
    TRADING_DAYS_PER_YEAR,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sharpe_standard_error,
)


def _normalize(raw, mean, std):
    """Affine-rescale to an exact mean/std, preserving skew and kurtosis."""
    s = pd.Series(raw)
    return (s - s.mean()) / s.std() * std + mean


def test_psr_positive_sharpe_series_is_confident():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.01, 0.01, 300))
    assert probabilistic_sharpe_ratio(returns) > 0.9


def test_psr_is_half_when_benchmark_equals_observed_sharpe():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.005, 0.02, 250))
    observed = returns.mean() / returns.std()
    assert probabilistic_sharpe_ratio(returns, observed) == pytest.approx(0.5, abs=1e-9)


def test_psr_lower_for_negatively_skewed_fat_tailed_series():
    rng = np.random.default_rng(2)
    gaussian = _normalize(rng.standard_normal(1000), mean=0.2, std=1.0)
    skewed = _normalize(-rng.lognormal(0.0, 1.0, 1000), mean=0.2, std=1.0)

    # Same observed (per-period) Sharpe by construction.
    assert (gaussian.mean() / gaussian.std()) == pytest.approx(
        skewed.mean() / skewed.std()
    )
    assert skewed.skew() < 0
    assert abs(gaussian.skew()) < 0.3
    assert probabilistic_sharpe_ratio(skewed) < probabilistic_sharpe_ratio(gaussian)


def test_expected_max_sharpe_zero_for_one_or_fewer_trials():
    assert expected_max_sharpe(1, 0.5) == 0.0
    assert expected_max_sharpe(0, 0.5) == 0.0
    assert expected_max_sharpe(-5, 0.5) == 0.0


def test_expected_max_sharpe_increases_with_trials():
    v = 0.25
    assert expected_max_sharpe(10, v) < expected_max_sharpe(100, v) < expected_max_sharpe(1000, v)


def test_expected_max_sharpe_zero_when_no_variance():
    assert expected_max_sharpe(100, 0.0) == 0.0


def test_expected_max_sharpe_handles_astronomical_trials_without_throwing():
    # 1 - 1/n rounds to exactly 1.0 here; must return 0.0 rather than raise.
    assert expected_max_sharpe(10**16, 0.5) == 0.0


def test_dsr_equals_psr_with_zero_benchmark_for_single_trial():
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0.004, 0.02, 400))
    assert deflated_sharpe_ratio(returns, n_trials=1, var_across_trials=0.0) == pytest.approx(
        probabilistic_sharpe_ratio(returns, 0.0)
    )


def test_dsr_strictly_decreases_as_trials_increase():
    # Moderate Sharpe and sample size keep PSR off the 0/1 rails so the
    # benchmark shift from more trials is visible rather than saturated.
    rng = np.random.default_rng(4)
    returns = pd.Series(rng.normal(0.003, 0.03, 120))
    v = 0.001
    d1 = deflated_sharpe_ratio(returns, 1, v)
    d10 = deflated_sharpe_ratio(returns, 10, v)
    d100 = deflated_sharpe_ratio(returns, 100, v)
    assert d1 > d10 > d100


def test_degenerate_inputs_return_zero():
    constant = pd.Series([1.0] * 50)
    short = pd.Series([0.01, 0.02])
    assert probabilistic_sharpe_ratio(constant) == 0.0
    assert probabilistic_sharpe_ratio(short) == 0.0
    assert deflated_sharpe_ratio(constant, 10, 0.5) == 0.0
    assert deflated_sharpe_ratio(short, 10, 0.5) == 0.0


@pytest.mark.parametrize("value", [0.001, -0.002])
def test_constant_nonzero_series_count_as_flat(value):
    constant = pd.Series([value] * 30)

    # A constant series carries rounding noise rather than an exactly zero std.
    assert 0.0 < constant.std() < 1e-15

    assert sharpe_ratio(constant) == 0.0
    assert probabilistic_sharpe_ratio(constant) == 0.0
    assert deflated_sharpe_ratio(constant, 10, 0.5) == 0.0


def test_small_but_real_dispersion_is_not_treated_as_flat():
    """Dispersion four orders of magnitude below the mean is still dispersion;
    only the rounding scale of the values themselves counts as flat."""
    rng = np.random.default_rng(5)
    returns = pd.Series(rng.normal(0.001, 1e-7, 200))

    assert returns.std() / returns.mean() == pytest.approx(1e-4, rel=0.2)
    assert sharpe_ratio(returns) > 0.0
    assert probabilistic_sharpe_ratio(returns) > 0.9


def test_sharpe_se_normal_returns_matches_closed_form():
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.0005, 0.01, 200_000))
    sr = returns.mean() / returns.std()
    expected = math.sqrt((1 + sr**2 / 2) / (len(returns) - 1)) * math.sqrt(TRADING_DAYS_PER_YEAR)
    assert math.isclose(sharpe_standard_error(returns), expected, rel_tol=0.02)


def test_sharpe_se_negative_skew_widens_error_for_positive_sharpe():
    rng = np.random.default_rng(11)
    base = rng.normal(0.001, 0.01, 5_000)
    crashes = np.where(rng.random(5_000) < 0.01, -0.08, 0.0)
    skewed = pd.Series(base + crashes + 0.0008)
    symmetric = pd.Series(rng.normal(skewed.mean(), skewed.std(), 5_000))
    assert sharpe_standard_error(skewed) > sharpe_standard_error(symmetric)


def test_sharpe_se_degenerate_inputs_return_nan():
    assert math.isnan(sharpe_standard_error(pd.Series([0.01, 0.02])))
    assert math.isnan(sharpe_standard_error(pd.Series([0.01] * 50)))


def test_sharpe_se_matches_exact_formula_on_small_sample():
    returns = pd.Series([0.01, 0.012, 0.009, 0.011, 0.013, 0.008, 0.010, 0.012, 0.011, -0.02])
    t = len(returns)
    sr = returns.mean() / returns.std()
    g3 = returns.skew()
    g4 = returns.kurt() + 3.0
    expected = math.sqrt((1 - g3 * sr + (g4 - 1) / 4 * sr**2) / (t - 1)) * math.sqrt(
        TRADING_DAYS_PER_YEAR
    )
    normal = math.sqrt((1 + sr**2 / 2) / (t - 1)) * math.sqrt(TRADING_DAYS_PER_YEAR)
    assert math.isclose(sharpe_standard_error(returns), expected, rel_tol=1e-12)
    assert abs(expected - normal) / normal > 0.01
