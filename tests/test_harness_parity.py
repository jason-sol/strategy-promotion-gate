"""Drift guard between bot/validation.py + bot/metrics.py and the standalone
validation_harness package.

The bot keeps its own copy so the live loop has no package dependency; these
tests fail the moment the two implementations disagree on a shared input.
"""

import dataclasses
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bot import metrics as bot_metrics
from bot import validation as bot_validation
from bot.backtest import Backtester
from bot.strategies import build_strategy
from config import AssetClass, InstrumentConfig, StrategyName
from helpers import daily_bars

HARNESS_SRC = Path(__file__).resolve().parents[1] / "packages" / "validation-harness" / "src"
if str(HARNESS_SRC) not in sys.path:
    sys.path.insert(0, str(HARNESS_SRC))

from validation_harness import metrics as pkg_metrics  # noqa: E402
from validation_harness import validation as pkg_validation  # noqa: E402


def mr_cfg(symbol: str, entry_z: float) -> InstrumentConfig:
    return InstrumentConfig(
        symbol=symbol, asset_class=AssetClass.US_EQUITY,
        strategy=StrategyName.MEAN_REVERSION, timeframe_minutes=1440,
        params={"sma_period": 10, "entry_z": entry_z, "allow_short": False},
        slippage_pct=0.0,
    )


def bot_candidate(label: str, symbol: str, entry_z: float) -> bot_validation.Candidate:
    cfg = mr_cfg(symbol, entry_z)
    return bot_validation.Candidate(label, cfg, build_strategy(cfg))


def pkg_candidate(label: str, symbol: str, entry_z: float) -> pkg_validation.Candidate:
    """Same simulator, reached through the package's curve-function seam — the
    one place the two implementations legitimately differ."""
    cfg = mr_cfg(symbol, entry_z)
    strategy = build_strategy(cfg)

    def run(bars: pd.DataFrame, sim_start: pd.Timestamp) -> pd.Series:
        prepared = strategy.prepare(bars)
        bt = Backtester(initial_equity=bot_validation.INITIAL_EQUITY, halt_enabled=False)
        return bt.run([(cfg, strategy, prepared)], sim_start).equity_curve

    return pkg_validation.Candidate(label, symbol, run)


def wobbly(n: int, amplitude: float = 4.0, slope: float = 0.5) -> list[float]:
    t = np.arange(n)
    return list(100.0 + slope * t + amplitude * np.sin(t / 2.0))


@pytest.fixture
def matrix() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    dates = pd.date_range("2020-01-01", periods=180, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "a": rng.standard_normal(180) * 0.01 + 0.001,
            "b": rng.standard_normal(180) * 0.02,
            "c": rng.standard_normal(180) * 0.005 - 0.0005,
            "d": rng.standard_normal(180) * 0.015 + 0.002,
        },
        index=dates,
    )


class TestMetricsParity:
    @pytest.mark.parametrize("p", [1e-9, 0.01, 0.024, 0.3, 0.5, 0.8, 0.99, 1 - 1e-9])
    def test_normal_cdf_and_ppf_agree_bit_for_bit(self, p):
        assert pkg_metrics._norm_ppf(p) == bot_metrics._norm_ppf(p)
        assert pkg_metrics._norm_cdf(p) == bot_metrics._norm_cdf(p)

    @pytest.mark.parametrize("n_trials", [1, 2, 7, 64, 1000, 10**17])
    @pytest.mark.parametrize("var", [0.0, 0.01, 0.25, 4.0])
    def test_expected_max_sharpe_agrees(self, n_trials, var):
        assert pkg_metrics.expected_max_sharpe(n_trials, var) == (
            bot_metrics.expected_max_sharpe(n_trials, var)
        )

    @pytest.mark.parametrize("n_trials", [1, 3, 25, 250])
    @pytest.mark.parametrize("var", [0.0, 0.02, 0.5])
    def test_deflated_sharpe_agrees_on_a_returns_series(self, matrix, n_trials, var):
        returns = matrix["a"]
        assert pkg_metrics.deflated_sharpe_ratio(returns, n_trials, var) == (
            bot_metrics.deflated_sharpe_ratio(returns, n_trials, var)
        )

    def test_psr_and_sharpe_agree(self, matrix):
        for col in matrix.columns:
            assert pkg_metrics.probabilistic_sharpe_ratio(matrix[col]) == (
                bot_metrics.probabilistic_sharpe_ratio(matrix[col])
            )
            assert pkg_metrics.sharpe_ratio(matrix[col]) == bot_metrics.sharpe_ratio(matrix[col])

    def test_degenerate_series_agree(self):
        for values in ([], [0.01], [0.01, 0.02], [0.01] * 20):
            returns = pd.Series(values, dtype=float)
            assert pkg_metrics.sharpe_ratio(returns) == bot_metrics.sharpe_ratio(returns)
            assert pkg_metrics.probabilistic_sharpe_ratio(returns) == (
                bot_metrics.probabilistic_sharpe_ratio(returns)
            )


def assert_cscv_reports_agree(pkg, bot) -> None:
    """Panel by panel, including the not-computable nans and the dataclass shape
    itself — a field added on one side only is drift too."""
    assert [f.name for f in dataclasses.fields(pkg)] == [f.name for f in dataclasses.fields(bot)]

    for name in ("pbo", "p_oos_loss", "median_logit", "degradation_slope"):
        pkg_value, bot_value = getattr(pkg, name), getattr(bot, name)
        if math.isnan(bot_value):
            assert math.isnan(pkg_value), name
        else:
            assert pkg_value == pytest.approx(bot_value, rel=1e-12, abs=1e-12), name

    assert len(pkg.logits) == len(bot.logits)
    for split, (pkg_logit, bot_logit) in enumerate(zip(pkg.logits, bot.logits, strict=True)):
        assert pkg_logit == pytest.approx(bot_logit, rel=1e-12, abs=1e-12), split

    for name in ("oos_rank_mean", "is_best_share"):
        pkg_panel, bot_panel = getattr(pkg, name), getattr(bot, name)
        assert list(pkg_panel) == list(bot_panel), name
        for label, bot_value in bot_panel.items():
            assert pkg_panel[label] == pytest.approx(bot_value, rel=1e-12, abs=1e-12), (name, label)


class TestCscvParity:
    @pytest.mark.parametrize("n_blocks", [4, 6, 8, 16])
    def test_pbo_agrees_on_a_shared_matrix(self, matrix, n_blocks):
        assert pkg_validation.cscv_pbo(matrix, n_blocks) == (
            bot_validation.cscv_pbo(matrix, n_blocks)
        )

    @pytest.mark.parametrize("n_blocks", [4, 6, 8, 16])
    def test_every_report_panel_agrees_on_a_shared_matrix(self, matrix, n_blocks):
        # 180 rows over 16 blocks leaves a ragged tail both sides have to drop.
        assert_cscv_reports_agree(pkg_validation.cscv_report(matrix, n_blocks),
                                  bot_validation.cscv_report(matrix, n_blocks))

    def test_report_panels_agree_on_the_not_computable_guards(self, matrix):
        for degenerate in (matrix[["a"]], matrix.iloc[:10]):
            assert_cscv_reports_agree(pkg_validation.cscv_report(degenerate),
                                      bot_validation.cscv_report(degenerate))

    def test_odd_block_counts_are_rejected_by_both_reports(self, matrix):
        with pytest.raises(ValueError, match="must be even"):
            pkg_validation.cscv_report(matrix, 5)
        with pytest.raises(ValueError, match="must be even"):
            bot_validation.cscv_report(matrix, 5)

    def test_not_computable_guards_agree(self, matrix):
        single = matrix[["a"]]
        assert math.isnan(pkg_validation.cscv_pbo(single))
        assert math.isnan(bot_validation.cscv_pbo(single))

        short = matrix.iloc[:10]
        assert math.isnan(pkg_validation.cscv_pbo(short))
        assert math.isnan(bot_validation.cscv_pbo(short))

    def test_odd_block_counts_are_rejected_by_both(self, matrix):
        with pytest.raises(ValueError, match="must be even"):
            pkg_validation.cscv_pbo(matrix, 5)
        with pytest.raises(ValueError, match="must be even"):
            bot_validation.cscv_pbo(matrix, 5)

    def test_deflated_sharpes_from_matrix_agrees(self, matrix):
        for n_trials in (1, 4, 100):
            assert pkg_validation.deflated_sharpes_from_matrix(matrix, n_trials) == (
                bot_validation.deflated_sharpes_from_matrix(matrix, n_trials)
            )


class TestReturnMatrixParity:
    def test_identical_over_a_shared_window(self):
        bars_by_symbol = {"AAA": daily_bars(wobbly(120))}
        args = ([("lo", "AAA", 1.0), ("hi", "AAA", 2.0)], bars_by_symbol)

        pkg = pkg_validation.build_return_matrix(
            [pkg_candidate(*a) for a in args[0]], bars_by_symbol
        )
        bot = bot_validation.build_return_matrix(
            [bot_candidate(*a) for a in args[0]], bars_by_symbol
        )
        pd.testing.assert_frame_equal(pkg, bot)
        assert not pkg.empty

    def test_identical_trim_on_offset_histories(self):
        """The head-trim/interior-fill split is the subtlest behaviour here, so
        it gets its own offset-history case."""
        bars_by_symbol = {
            "AAA": daily_bars(wobbly(140), start="2020-01-01"),
            "BBB": daily_bars(wobbly(90, amplitude=3.0), start="2020-03-02"),
        }
        spec = [("early", "AAA", 1.0), ("late", "BBB", 1.5)]

        pkg = pkg_validation.build_return_matrix(
            [pkg_candidate(*s) for s in spec], bars_by_symbol
        )
        bot = bot_validation.build_return_matrix(
            [bot_candidate(*s) for s in spec], bars_by_symbol
        )
        pd.testing.assert_frame_equal(pkg, bot)
        assert pkg.index.min() == bars_by_symbol["BBB"].index.min()

    def test_identical_under_an_explicit_sim_start(self):
        bars_by_symbol = {"AAA": daily_bars(wobbly(120))}
        sim_start = bars_by_symbol["AAA"].index[30]
        spec = [("lo", "AAA", 1.0), ("hi", "AAA", 2.0)]

        pd.testing.assert_frame_equal(
            pkg_validation.build_return_matrix(
                [pkg_candidate(*s) for s in spec], bars_by_symbol, sim_start
            ),
            bot_validation.build_return_matrix(
                [bot_candidate(*s) for s in spec], bars_by_symbol, sim_start
            ),
        )

    def test_disjoint_histories_raise_in_both(self):
        bars_by_symbol = {
            "AAA": daily_bars(wobbly(40), start="2020-01-01"),
            "BBB": daily_bars(wobbly(40), start="2021-01-04"),
        }
        spec = [("old", "AAA", 1.0), ("new", "BBB", 1.0)]

        with pytest.raises(ValueError, match="do not overlap"):
            pkg_validation.build_return_matrix([pkg_candidate(*s) for s in spec], bars_by_symbol)
        with pytest.raises(ValueError, match="do not overlap"):
            bot_validation.build_return_matrix([bot_candidate(*s) for s in spec], bars_by_symbol)


class TestWalkForwardParity:
    def test_folds_and_oos_series_agree(self):
        # 203 does not divide into n_folds + 1 blocks, so the last fold has to
        # absorb the ragged tail rather than stopping a block short.
        bars = daily_bars(wobbly(203))
        spec = [("lo", "AAA", 1.0), ("mid", "AAA", 1.5), ("hi", "AAA", 2.5)]

        pkg = pkg_validation.walk_forward([pkg_candidate(*s) for s in spec], bars, n_folds=4)
        bot = bot_validation.walk_forward([bot_candidate(*s) for s in spec], bars, n_folds=4)

        assert len(pkg.folds) == len(bot.folds) == 4
        for p, b in zip(pkg.folds, bot.folds):
            assert (p.train_start, p.test_start, p.test_end) == (
                b.train_start, b.test_start, b.test_end
            )
            assert p.chosen_label == b.chosen_label
            assert p.oos_sharpe == b.oos_sharpe

        assert pkg.folds[-1].test_end == bot.folds[-1].test_end == bars.index[-1]
        pd.testing.assert_series_equal(pkg.oos_returns, bot.oos_returns)
        assert pkg.oos_sharpe == bot.oos_sharpe
        assert pkg.variant_oos_sharpe == bot.variant_oos_sharpe

    def test_degenerate_windows_agree(self):
        bars = daily_bars(wobbly(3))
        spec = [("lo", "AAA", 1.0)]

        pkg = pkg_validation.walk_forward([pkg_candidate(*s) for s in spec], bars, n_folds=4)
        bot = bot_validation.walk_forward([bot_candidate(*s) for s in spec], bars, n_folds=4)

        assert pkg.folds == bot.folds == []
        assert math.isnan(pkg.oos_sharpe) and math.isnan(bot.oos_sharpe)
        assert pkg.variant_oos_sharpe == bot.variant_oos_sharpe == {}


class TestPackageIsStandalone:
    def test_imports_without_scipy_alpaca_or_the_bot(self):
        """Run in a fresh interpreter: this test process has already imported
        the bot, so only a subprocess can prove the package does not."""
        probe = (
            "import sys;"
            f"sys.path.insert(0, {str(HARNESS_SRC)!r});"
            "import validation_harness;"
            "leaked = sorted(m for m in sys.modules"
            " if m.split('.')[0] in {'scipy', 'alpaca', 'bot', 'config', 'matplotlib'});"
            "print(','.join(leaked));"
            "assert not leaked, leaked"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr

    def test_public_surface_matches_the_bot_names(self):
        import validation_harness

        shared = {
            "Candidate", "CscvReport", "FoldResult", "WalkForwardResult",
            "build_return_matrix", "cscv_pbo", "cscv_report", "deflated_sharpes",
            "deflated_sharpes_from_matrix", "walk_forward",
        }
        assert shared <= set(validation_harness.__all__)
        for name in shared:
            assert hasattr(bot_validation, name), name
