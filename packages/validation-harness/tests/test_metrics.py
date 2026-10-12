import math

import numpy as np
import pandas as pd
import pytest

from validation_harness import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from validation_harness.metrics import _norm_cdf, _norm_ppf


def series(values) -> pd.Series:
    return pd.Series(list(values), dtype=float)


class TestNormal:
    def test_cdf_matches_known_quantiles(self):
        assert _norm_cdf(0.0) == pytest.approx(0.5)
        assert _norm_cdf(1.959963985) == pytest.approx(0.975, abs=1e-9)
        assert _norm_cdf(-1.959963985) == pytest.approx(0.025, abs=1e-9)

    @pytest.mark.parametrize("p", [1e-8, 0.001, 0.02, 0.024, 0.5, 0.9, 0.976, 1 - 1e-8])
    def test_ppf_inverts_cdf_across_all_three_branches(self, p):
        assert _norm_cdf(_norm_ppf(p)) == pytest.approx(p, rel=1e-6, abs=1e-9)

    def test_ppf_is_symmetric_about_the_median(self):
        assert _norm_ppf(0.25) == pytest.approx(-_norm_ppf(0.75), abs=1e-9)

    @pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.5])
    def test_ppf_rejects_probabilities_outside_the_open_unit_interval(self, p):
        with pytest.raises(ValueError, match="must lie in"):
            _norm_ppf(p)


class TestSharpe:
    def test_annualizes_by_trading_days(self):
        returns = series([0.01, 0.02, -0.01, 0.03, 0.0])
        expected = returns.mean() / returns.std() * math.sqrt(252)
        assert sharpe_ratio(returns) == pytest.approx(expected)

    def test_zero_when_too_short_or_flat(self):
        assert sharpe_ratio(series([0.01])) == 0.0
        assert sharpe_ratio(series([])) == 0.0
        assert sharpe_ratio(series([0.0] * 10)) == 0.0
        assert sharpe_ratio(series([0.001] * 30)) == 0.0
        assert sharpe_ratio(series([-0.002] * 30)) == 0.0

    def test_small_but_real_dispersion_is_not_treated_as_flat(self):
        rng = np.random.default_rng(5)
        returns = pd.Series(rng.normal(0.001, 1e-7, 200))

        assert sharpe_ratio(returns) > 0.0

    def test_ignores_leading_nan_from_pct_change(self):
        with_nan = pd.Series([np.nan, 0.01, 0.02, -0.01])
        assert sharpe_ratio(with_nan) == pytest.approx(sharpe_ratio(series([0.01, 0.02, -0.01])))


class TestProbabilisticSharpe:
    def test_rises_with_a_stronger_track_record(self):
        weak = probabilistic_sharpe_ratio(series([0.01, -0.01, 0.02, -0.02, 0.01] * 4))
        strong = probabilistic_sharpe_ratio(series([0.02, 0.01, 0.03, 0.005, 0.02] * 4))
        assert 0.0 < weak < strong < 1.0

    def test_zero_when_not_computable(self):
        assert probabilistic_sharpe_ratio(series([0.01, 0.02])) == 0.0
        assert probabilistic_sharpe_ratio(series([0.0] * 10)) == 0.0

    @pytest.mark.parametrize("value", [0.001, -0.002])
    def test_a_constant_nonzero_series_is_flat(self, value):
        """A constant series carries rounding noise rather than an exactly zero
        std, and noise is not a track record to be confident about."""
        flat = series([value] * 30)

        assert 0.0 < flat.std() < 1e-15
        assert probabilistic_sharpe_ratio(flat) == 0.0
        assert deflated_sharpe_ratio(flat, 10, 0.5) == 0.0

    def test_small_but_real_dispersion_is_not_treated_as_flat(self):
        """Dispersion four orders of magnitude below the mean is still
        dispersion; only the rounding scale of the values counts as flat."""
        rng = np.random.default_rng(5)
        returns = pd.Series(rng.normal(0.001, 1e-7, 200))

        assert probabilistic_sharpe_ratio(returns) > 0.9

    def test_benchmark_shifts_the_probability_down(self):
        returns = series([0.02, 0.01, 0.03, 0.005, 0.02] * 4)
        assert probabilistic_sharpe_ratio(returns, 0.5) < probabilistic_sharpe_ratio(returns)


class TestExpectedMaxSharpe:
    def test_zero_without_dispersion_or_multiple_trials(self):
        assert expected_max_sharpe(1, 0.25) == 0.0
        assert expected_max_sharpe(20, 0.0) == 0.0
        assert expected_max_sharpe(20, -1.0) == 0.0

    def test_grows_with_the_trial_count(self):
        benchmarks = [expected_max_sharpe(n, 0.25) for n in (2, 5, 20, 100)]
        assert benchmarks == sorted(benchmarks)
        assert benchmarks[0] > 0.0

    def test_scales_with_the_square_root_of_variance(self):
        assert expected_max_sharpe(10, 0.04) == pytest.approx(2 * expected_max_sharpe(10, 0.01))

    def test_collapses_to_zero_once_the_quantiles_round_to_one(self):
        """Past ~1e16 trials the 1 - 1/n quantile is 1.0 in float64 and the
        approximation has no headroom, so the benchmark degrades to nothing
        rather than to an infinity."""
        assert expected_max_sharpe(10**17, 0.25) == 0.0


class TestDeflatedSharpe:
    def test_equals_undeflated_psr_for_a_single_trial(self):
        returns = series([0.02, 0.01, 0.03, 0.005, 0.02] * 4)
        assert deflated_sharpe_ratio(returns, 1, 0.0) == pytest.approx(
            probabilistic_sharpe_ratio(returns)
        )

    def test_more_trials_deflate_the_same_track_record(self):
        returns = series([0.02, 0.01, 0.03, 0.005, 0.02] * 4)
        assert deflated_sharpe_ratio(returns, 50, 0.25) < deflated_sharpe_ratio(returns, 2, 0.25)
