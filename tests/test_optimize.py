import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from bot import optimize
from bot.research import EQUITY_START, strategy_params
from bot.validation import CscvReport
from bot.strategies.trend_following import TrendFollowing
from bot.strategies.turn_of_month import TurnOfMonth
from config import RISK, MARKET_DATA_FEED, VALIDATION
from helpers import daily_bars

# Reduced grid + a short business-day frame: the experiment's shape is what is
# under test, and the full 5x7 grid over a decade of bars is a minutes-long run.
ENTRY = (1, 2)
EXIT = (4, 5)
N_BLOCKS = 4
N_FOLDS = 2

# Trend field, sized so every cell's 4x-slow warm-up fits the feasibility bound
# on a 500-bar frame at N_FOLDS folds (block 166 -> slow <= 31).
T_FAST = (5, 10)
T_SLOW = (20, 30)
T_TRAIL = (2.0, 3.0)


def choppy_uptrend(n: int) -> list[float]:
    t = np.arange(n)
    return list(100.0 + 0.05 * t + 4.0 * np.sin(t / 5.0) + 1.5 * np.sin(t / 1.7))


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return daily_bars(choppy_uptrend(760), start="2019-01-01")  # ~3 years of business days


@pytest.fixture(scope="module")
def grid(bars) -> "optimize.GridResult":
    return optimize.run_grid(bars, entry_bdays=ENTRY, exit_days=EXIT,
                             n_blocks=N_BLOCKS, n_folds=N_FOLDS)


@pytest.fixture(scope="module")
def trend_bars() -> pd.DataFrame:
    return daily_bars(choppy_uptrend(500), start="2019-01-01")


@pytest.fixture(scope="module")
def trend_grid(trend_bars) -> "optimize.GridResult":
    return optimize.score_field(
        optimize.trend_candidates(fasts=T_FAST, slows=T_SLOW, trails=T_TRAIL,
                                  volume_mults=(None,)),
        trend_bars, symbol="USO", spec=optimize.TREND_FIELD,
        n_blocks=N_BLOCKS, n_folds=N_FOLDS)


def test_default_grid_is_the_35_cell_tom_field():
    candidates = optimize.grid_candidates()
    assert optimize.ENTRY_BDAYS == (1, 2, 3, 4, 5)
    assert optimize.EXIT_DAYS == (1, 2, 3, 4, 5, 6, 7)
    assert len(candidates) == 35
    params = {(c.strategy.entry_window_bdays, c.strategy.exit_day_of_month)
              for c in candidates}
    assert params == {(e, x) for e in range(1, 6) for x in range(1, 8)}
    assert all(isinstance(c.strategy, TurnOfMonth) for c in candidates)
    assert all(c.cfg.symbol == "VOO" for c in candidates)


def test_labels_fit_the_printer_contract():
    labels = [c.label for c in optimize.grid_candidates()]
    assert len(set(labels)) == len(labels)
    assert max(len(label) for label in labels) <= 16
    assert optimize.cell_label(2, 5) == "TOM 2/5"
    assert optimize.HEADLINE_LABEL == "TOM 2/5"


def test_run_grid_scores_every_cell(grid):
    assert [c.label for c in grid.cells] == ["TOM 1/4", "TOM 1/5", "TOM 2/4", "TOM 2/5"]
    assert [(c.params["entry_window_bdays"], c.params["exit_day_of_month"])
            for c in grid.cells] == [(1, 4), (1, 5), (2, 4), (2, 5)]
    assert all(math.isfinite(c.sharpe) for c in grid.cells)
    assert all(0.0 <= c.dsr <= 1.0 for c in grid.cells)
    assert all(math.isfinite(c.oos_sharpe) for c in grid.cells)
    assert grid.n_trials == 4
    assert grid.symbol == "VOO"


def test_cells_record_the_full_strategy_params_and_drawdown(grid):
    cell = grid.cell("TOM 2/5")
    assert cell.params == strategy_params(optimize.tom(2, 5))
    assert cell.params["atr_period"] == RISK.atr_period
    assert all(0.0 <= c.max_dd <= 1.0 for c in grid.cells)


def test_dsr_deflates_against_the_full_grid(bars, monkeypatch):
    seen = {}
    real = optimize.deflated_sharpes_from_matrix

    def spy(matrix, n_trials):
        seen["n_trials"] = n_trials
        seen["columns"] = list(matrix.columns)
        return real(matrix, n_trials)

    monkeypatch.setattr(optimize, "deflated_sharpes_from_matrix", spy)
    optimize.run_grid(bars, entry_bdays=ENTRY, exit_days=EXIT,
                      n_blocks=N_BLOCKS, n_folds=N_FOLDS)
    assert seen["n_trials"] == 4  # every cell tried, not the survivors
    assert seen["columns"] == ["TOM 1/4", "TOM 1/5", "TOM 2/4", "TOM 2/5"]


def test_cscv_defaults_to_the_validation_block_count(bars, monkeypatch):
    seen = {}

    def spy(matrix, n_blocks):
        seen["n_blocks"] = n_blocks
        seen["calls"] = seen.get("calls", 0) + 1
        return fake_report()

    monkeypatch.setattr(optimize, "cscv_report", spy)
    optimize.run_grid(bars, entry_bdays=ENTRY, exit_days=EXIT, n_folds=N_FOLDS)
    assert seen["n_blocks"] == VALIDATION.n_cscv_blocks
    # the report pays for every split; one call per field, not one per cell
    assert seen["calls"] == 1


def test_is_best_is_the_undeflated_full_window_winner(grid):
    assert grid.is_best.label == max(grid.cells, key=lambda c: c.sharpe).label
    assert grid.is_best.sharpe == max(c.sharpe for c in grid.cells)


def test_is_best_oos_rank_and_haircut(grid):
    ranked = sorted(grid.cells, key=lambda c: c.oos_sharpe, reverse=True)
    assert grid.is_best_oos_rank == [c.label for c in ranked].index(grid.is_best.label) + 1
    assert 1 <= grid.is_best_oos_rank <= len(grid.cells)
    assert grid.is_best_haircut == pytest.approx(grid.is_best.sharpe - grid.is_best.oos_sharpe)


def test_dsr_best_may_disagree_with_is_best(grid):
    assert grid.dsr_best.dsr == max(c.dsr for c in grid.cells)
    assert grid.disagrees == (grid.dsr_best.label != grid.is_best.label)


def test_gate_count_uses_each_cell_own_numbers(grid):
    expected = [c for c in grid.cells
                if VALIDATION.passes_validation(c.dsr, grid.p_oos_loss,
                                                c.oos_sharpe)]
    assert grid.n_passing == len(expected)
    assert [c.label for c in grid.cells if c.passes] == [c.label for c in expected]


def test_headline_cell_is_scored_like_any_other(grid):
    headline = grid.headline
    assert headline is not None
    assert headline.label == optimize.HEADLINE_LABEL
    assert headline is grid.cell(optimize.HEADLINE_LABEL)
    assert headline.passes == VALIDATION.passes_validation(
        headline.dsr, grid.p_oos_loss, headline.oos_sharpe)


def test_headline_absent_from_a_grid_that_excludes_it(bars):
    result = optimize.run_grid(bars, entry_bdays=(1,), exit_days=(4,),
                               n_blocks=N_BLOCKS, n_folds=N_FOLDS)
    assert result.headline is None
    assert result.cell("TOM 9/9") is None


def test_select_then_trade_oos_is_reported_alongside_the_field(grid):
    assert math.isfinite(grid.select_oos_sharpe)
    # The honest path follows one winner per fold, so it need not equal any
    # single cell's own concatenated OOS Sharpe.
    assert isinstance(grid.select_oos_sharpe, float)


def test_oos_rank_ranks_any_cell_not_just_the_is_best(grid):
    ranked = [c.label for c in sorted(grid.cells, key=lambda c: c.oos_sharpe,
                                      reverse=True)]
    for label in ranked:
        assert grid.oos_rank(label) == ranked.index(label) + 1
    assert grid.oos_rank(grid.is_best.label) == grid.is_best_oos_rank


def test_dsr_survivors_are_counted_apart_from_the_gate(grid):
    assert grid.n_dsr_passing == sum(1 for c in grid.cells
                                     if c.dsr >= VALIDATION.dsr_min)
    assert grid.oos_loss_blocks_gate == (
        grid.p_oos_loss == grid.p_oos_loss
        and grid.p_oos_loss > VALIDATION.p_oos_loss_max)
    assert not hasattr(grid, "pbo_blocks_gate")


def test_the_field_correlation_summarises_how_duplicated_the_field_is(grid):
    corr = grid.field_correlation
    assert -1.0 <= corr.min <= corr.mean <= 1.0


def test_field_correlation_is_not_computable_for_a_single_column(bars):
    result = optimize.run_grid(bars, entry_bdays=(1,), exit_days=(4,),
                               n_blocks=N_BLOCKS, n_folds=N_FOLDS)
    assert math.isnan(result.field_correlation.mean)
    assert math.isnan(result.field_correlation.min)


def tom_params(entry: int, exit_day: int) -> dict:
    return {"entry_window_bdays": entry, "exit_day_of_month": exit_day}


LABELS = ("TOM 1/4", "TOM 2/5", "TOM 3/7", "TOM 5/1")


def fake_report(pbo=0.24, p_oos_loss=0.18, median_logit=0.9,
                degradation_slope=-0.7, labels=LABELS) -> CscvReport:
    """A CSCV report with no splits behind it: the reporting layer only reads
    its numbers."""
    n = len(labels)
    return CscvReport(
        pbo=pbo, p_oos_loss=p_oos_loss, median_logit=median_logit,
        degradation_slope=degradation_slope, logits=(0.1, 0.2),
        oos_rank_mean={label: (i + 1) / n for i, label in enumerate(labels)},
        is_best_share={label: 1.0 / n for label in labels},
    )


def fake_grid(pbo=0.24, select=0.71, nan_cell=False, p_oos_loss=0.18,
              report=None, corr=None) -> "optimize.GridResult":
    """A GridResult with no backtests behind it: the reporting layer only reads
    the numbers."""
    cells = [
        optimize.CellResult("TOM 1/4", tom_params(1, 4), 0.40, 0.60, 0.20, False, 0.11),
        optimize.CellResult("TOM 2/5", tom_params(2, 5), 0.80, 0.96, 0.55, True, 0.07),
        optimize.CellResult("TOM 3/7", tom_params(3, 7), 0.95, 0.90, 0.10, False, 0.19),
    ]
    if nan_cell:
        cells.append(optimize.CellResult("TOM 5/1", tom_params(5, 1), float("nan"),
                                         float("nan"), float("nan"), False,
                                         float("nan")))
    return optimize.GridResult(
        "VOO", cells,
        report if report is not None else fake_report(pbo=pbo,
                                                      p_oos_loss=p_oos_loss),
        select, len(cells), 16, 4, optimize.TOM_FIELD,
        corr if corr is not None else optimize.FieldCorrelation(0.93, 0.71))


def test_report_records_every_cell_and_the_run_context():
    at = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)
    report = optimize.build_grid_report(fake_grid(), generated_at=at)
    assert report["generated_at"] == at.isoformat()
    assert report["symbol"] == "VOO"
    assert report["market_data_feed"] == MARKET_DATA_FEED
    assert report["equity_start"] == EQUITY_START.date().isoformat()
    assert report["n_trials"] == 3
    assert report["n_blocks"] == 16 and report["n_folds"] == 4
    assert [c["label"] for c in report["cells"]] == ["TOM 1/4", "TOM 2/5", "TOM 3/7"]
    first = report["cells"][0]
    assert first["params"] == {"entry_window_bdays": 1, "exit_day_of_month": 4}
    assert (first["sharpe"], first["dsr"], first["oos_sharpe"]) == (0.40, 0.60, 0.20)
    assert first["passes"] is False


def test_report_summary_carries_every_cscv_panel_and_the_field_correlation():
    summary = optimize.build_grid_report(fake_grid())["summary"]
    assert summary["pbo"] == 0.24
    assert summary["p_oos_loss"] == 0.18
    assert summary["median_logit"] == 0.9
    assert summary["degradation_slope"] == -0.7
    assert summary["field_correlation"] == {"mean": 0.93, "min": 0.71}
    assert -1.0 <= summary["field_correlation"]["min"] <= 1.0
    assert -1.0 <= summary["field_correlation"]["mean"] <= 1.0


def test_field_correlation_serialises_as_null_when_not_computable():
    nan = float("nan")
    summary = optimize.build_grid_report(
        fake_grid(corr=optimize.FieldCorrelation(nan, nan)))["summary"]
    assert summary["field_correlation"] == {"mean": None, "min": None}


def test_every_cell_entry_carries_its_own_cscv_standing():
    report = optimize.build_grid_report(fake_grid())
    entries = report["cells"]
    assert [e["label"] for e in entries] == ["TOM 1/4", "TOM 2/5", "TOM 3/7"]
    assert entries[0]["oos_rank_mean"] == pytest.approx(0.25)
    assert entries[0]["is_best_share"] == pytest.approx(0.25)
    for key in ("is_best", "dsr_best", "headline"):
        assert "oos_rank_mean" in report["summary"][key]
        assert "is_best_share" in report["summary"][key]


def test_a_cell_absent_from_the_cscv_panels_records_null():
    report = optimize.build_grid_report(fake_grid(report=fake_report(labels=())))
    assert report["cells"][0]["oos_rank_mean"] is None
    assert report["cells"][0]["is_best_share"] is None


def test_report_summary_carries_the_measurement():
    summary = optimize.build_grid_report(fake_grid())["summary"]
    assert summary["pbo"] == 0.24
    assert summary["select_then_trade_oos_sharpe"] == 0.71
    assert summary["n_passing"] == 1
    assert summary["n_cells"] == 3
    assert summary["is_best"]["label"] == "TOM 3/7"
    assert summary["is_best"]["oos_rank"] == 3
    assert summary["is_best"]["haircut"] == pytest.approx(0.85)
    assert summary["dsr_best"]["label"] == "TOM 2/5"
    assert summary["disagrees"] is True
    assert summary["headline"]["label"] == "TOM 2/5"
    assert summary["headline"]["passes"] is True


def blocked_grid() -> "optimize.GridResult":
    """A field whose P(OOS loss) alone closes the gate: one cell clears the DSR
    bar, and the advisory PBO sits above 0.5 too."""
    cells = [optimize.CellResult("TOM 1/6", tom_params(1, 6), 1.02, 0.977, 0.99,
                                 False, 0.09),
             optimize.CellResult("TOM 2/5", tom_params(2, 5), 0.83, 0.916, 0.77,
                                 False, 0.12)]
    return optimize.GridResult(
        "VOO", cells,
        fake_report(pbo=0.61, p_oos_loss=0.72,
                    labels=("TOM 1/6", "TOM 2/5")),
        0.51, 35, 16, 4, optimize.TOM_FIELD,
        optimize.FieldCorrelation(0.96, 0.88))


def test_summary_separates_the_dsr_bar_from_the_field_oos_loss():
    summary = optimize.build_grid_report(blocked_grid())["summary"]
    assert summary["n_passing"] == 0
    assert summary["n_dsr_passing"] == 1
    assert summary["oos_loss_blocks_gate"] is True
    assert "pbo_blocks_gate" not in summary
    assert summary["headline"]["oos_rank"] == 2


def test_markdown_names_p_oos_loss_as_the_binding_criterion():
    text = optimize.build_grid_markdown(blocked_grid(),
                                        datetime(2026, 9, 5, tzinfo=timezone.utc))
    assert "0 of 2" in text
    assert "1 of 2" in text  # cells clearing the DSR bar on their own evidence
    assert "P(OOS loss)" in text and "0.720" in text
    assert "rank 2 of 2" in text  # the live cell's standing in the field
    # PBO stays reported, and stays out of the binding-criterion sentence
    assert "0.610" in text
    binding = next(ln for ln in text.splitlines()
                   if "binding criterion" in ln)
    assert "0.610" not in binding


def test_report_is_strict_json_with_nan_as_null():
    report = optimize.build_grid_report(fake_grid(nan_cell=True))
    text = json.dumps(report, allow_nan=False)
    assert json.loads(text)["cells"][3]["sharpe"] is None


def test_write_report_lands_in_the_research_directory(tmp_path):
    at = datetime(2026, 9, 5, tzinfo=timezone.utc)
    report = optimize.build_grid_report(fake_grid(), generated_at=at)
    path = optimize.write_grid_report(report, at, directory=str(tmp_path / "research"))
    assert os.path.basename(path) == "tom-grid-2026-09-05.json"
    with open(path) as fh:
        assert json.load(fh)["summary"]["p_oos_loss"] == 0.18
    assert optimize.RESEARCH_REPORT_DIR == os.path.join("docs", "research")


def test_write_markdown_states_the_findings(tmp_path):
    at = datetime(2026, 9, 5, tzinfo=timezone.utc)
    path = optimize.write_grid_markdown(fake_grid(), at,
                                        directory=str(tmp_path / "research"))
    assert os.path.basename(path) == "tom-grid-2026-09-05.md"
    text = open(path).read()
    assert "TOM 2/5" in text and "TOM 3/7" in text
    assert "0.24" in text  # field PBO
    assert "1 of 3" in text  # gate survivors
    assert MARKET_DATA_FEED in text


def test_the_record_paths_are_what_the_writers_write(tmp_path):
    at = datetime(2026, 9, 5, tzinfo=timezone.utc)
    json_path, md_path = optimize.grid_record_paths(at, str(tmp_path))
    report = optimize.build_grid_report(fake_grid(), generated_at=at)
    assert str(json_path) == optimize.write_grid_report(report, at,
                                                        directory=str(tmp_path))
    assert str(md_path) == optimize.write_grid_markdown(fake_grid(), at,
                                                        directory=str(tmp_path))


def test_a_writer_refuses_to_overwrite_an_existing_record(tmp_path):
    # Two runs on one UTC day share a filename, and the second one silently
    # taking the first one's place loses an experiment.
    at = datetime(2026, 9, 5, tzinfo=timezone.utc)
    report = optimize.build_grid_report(fake_grid(), generated_at=at)
    optimize.write_grid_report(report, at, directory=str(tmp_path))
    optimize.write_grid_markdown(fake_grid(), at, directory=str(tmp_path))
    with pytest.raises(FileExistsError):
        optimize.write_grid_report(report, at, directory=str(tmp_path))
    with pytest.raises(FileExistsError):
        optimize.write_grid_markdown(fake_grid(), at, directory=str(tmp_path))


def test_the_grid_cli_refuses_before_it_fetches_anything(monkeypatch, tmp_path,
                                                         capsys):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setattr(optimize, "RESEARCH_REPORT_DIR", str(tmp_path))

    class MustNotFetch:
        def __init__(self, *a, **k):
            raise AssertionError("must not fetch")

    monkeypatch.setattr(optimize, "MarketData", MustNotFetch)
    existing = optimize.grid_record_paths(datetime.now(timezone.utc),
                                          str(tmp_path))[0]
    existing.write_text("{}")
    assert optimize.run_tom_experiment() == 1
    assert str(existing) in capsys.readouterr().err


def test_experiment_never_touches_the_promotion_artifact():
    # No degrade guard, no validation directory: this record promotes nothing.
    assert not hasattr(optimize, "write_validation_report")
    assert not hasattr(optimize, "ValidationArtifactDegraded")
    assert "validation" not in optimize.RESEARCH_REPORT_DIR


def test_print_grid_prints_a_row_per_cell_and_the_findings(capsys):
    optimize.print_grid(fake_grid())
    out = capsys.readouterr().out
    rows = [line for line in out.splitlines() if line.startswith("VOO")]
    assert len(rows) == 3
    assert "<-- IS-best" in out and "<-- DSR-best" in out and "<-- live" in out
    assert "PBO 0.240" in out
    assert "rank 3/3" in out
    assert "1/3" in out  # cells clearing the gate


# --- B1: the grid machinery generalized to a second strategy field -----------


def test_feasible_slow_max_keeps_the_warmup_inside_the_train_block():
    # block = n // (n_folds + 1); a TrendFollowing slow period costs 4x slow
    # bars of warm-up, and the bound spends at most 75% of the first block on it.
    assert optimize.feasible_slow_max(2684, 4) == 100  # USO daily, the real run
    assert optimize.feasible_slow_max(500, 2) == 31
    assert optimize.feasible_slow_max(2684, 4, fraction=0.5) == 67
    assert optimize.feasible_slow_max(8, 4) == 0  # no room at all


def test_an_infeasible_cell_scores_a_silent_zero_not_a_penalty(trend_bars):
    # Why feasibility is a construction-time filter: min_bars is only a signal()
    # self-guard, so a strategy that never fires produces a flat curve whose
    # Sharpe is 0.0 — and in walk_forward's max-by-train-Sharpe pick a 0.0 beats
    # every negative candidate. Infeasibility must never reach the scorer.
    cfg = optimize.make_cfg("USO", optimize.AssetClass.US_EQUITY, 0.0005)
    starved = optimize.Candidate("T 5/200/3.0", cfg,
                                 TrendFollowing(fast_period=5, slow_period=200,
                                                trail_atr_mult=3.0,
                                                atr_period=RISK.atr_period,
                                                allow_short=False))
    returns = optimize.build_return_matrix([starved], {"USO": trend_bars})
    assert starved.strategy.min_bars > len(trend_bars)
    assert (returns["T 5/200/3.0"] == 0.0).all()
    assert optimize.sharpe_ratio(returns["T 5/200/3.0"]) == 0.0


def test_trend_candidates_vary_the_trail_research_trend_hardcodes():
    candidates = optimize.trend_candidates(volume_mults=(None,), fasts=(10,), slows=(40,),
                                           trails=(2.0, 3.5))
    trails = [c.strategy.trail_atr_mult for c in candidates]
    assert trails == [2.0, 3.5]  # research.trend() would give 3.0 for both
    strategy = candidates[0].strategy
    assert isinstance(strategy, TrendFollowing)
    assert (strategy.fast_period, strategy.slow_period) == (10, 40)
    assert strategy.allow_short is False
    assert strategy.atr_period == RISK.atr_period
    assert strategy.vol_gate_percentile is None
    assert all(c.cfg.symbol == "USO" for c in candidates)
    assert all(c.cfg.asset_class == optimize.AssetClass.US_EQUITY
               for c in candidates)
    assert all(c.cfg.slippage_pct == 0.0005 for c in candidates)


def test_trend_candidates_drop_inverted_and_infeasible_cells():
    candidates = optimize.trend_candidates(volume_mults=(None,), fasts=(5, 40, 60), slows=(40, 200),
                                           trails=(3.0,), max_slow=100)
    cells = {(c.strategy.fast_period, c.strategy.slow_period) for c in candidates}
    assert cells == {(5, 40)}  # fast >= slow dropped, slow 200 over the bound


def test_default_trend_field_is_the_declared_grid():
    assert optimize.TREND_FAST == (5, 10, 15, 20, 30, 40, 50)
    assert optimize.TREND_SLOW == (40, 60, 80, 100)
    assert optimize.TREND_TRAIL == (2.0, 2.5, 3.0, 3.5, 4.0)
    assert optimize.TREND_VOLUME_MULT == (None, 1.2, 1.5)
    candidates = optimize.trend_candidates(max_slow=100)
    assert len(candidates) == 390  # 130 EMA cells x 3 volume-gate settings
    assert len({c.label for c in candidates}) == 390
    assert max(len(c.label) for c in candidates) <= 16
    assert optimize.trend_cell_label(20, 100, 3.0, None) == "T 20/100/3.0"
    assert optimize.TREND_FIELD.headline_label == "T 20/100/3.0"
    assert any(c.label == "T 20/100/3.0" for c in candidates)  # the live cell


def test_ungated_cells_keep_the_labels_the_130_cell_run_recorded():
    # The volume axis must not rename the cells it did not change: the ungated
    # 130 stay directly comparable to trend-search-2026-09-07, and the headline
    # cell keeps the name its recorded DSR is filed under.
    ungated = optimize.trend_candidates(max_slow=100, volume_mults=(None,))
    assert len(ungated) == 130
    assert all(c.label.count("/") == 2 for c in ungated)
    assert optimize.trend_cell_label(20, 100, 3.0, 1.5) == "T 20/100/3.0/1.5"


def test_axes_order_the_ungated_cell_first(trend_grid_gated):
    # None is a real value on this axis, so it needs a defined place in the
    # ordering rather than raising when sorted against the floats.
    assert trend_grid_gated.axes["volume_gate_mult"] == [None, 1.5]


def test_a_flat_return_series_never_opened_a_position():
    assert optimize.traded_in(pd.Series([0.0, 0.0, 0.0])) is False
    assert optimize.traded_in(pd.Series([0.0, -0.01, 0.0])) is True
    assert optimize.traded_in(pd.Series([float("nan"), float("nan")])) is False
    assert optimize.traded_in(pd.Series([], dtype=float)) is False


def test_a_gate_that_suppresses_every_entry_is_marked_not_traded(trend_bars):
    # sharpe_ratio returns 0.0 for a flat curve and walk_forward selects by
    # max(Sharpe), so a cell that never trades BEATS every losing cell. The
    # warm-up feasibility filter cannot catch this one: whether a volume gate
    # starves the strategy depends on the bars, not on the parameters.
    field = optimize.score_field(
        optimize.trend_candidates(fasts=(5,), slows=(20,), trails=(2.0,),
                                  volume_mults=(None, 1e9)),
        trend_bars, symbol="USO", spec=optimize.TREND_FIELD,
        n_blocks=4, n_folds=2)
    by_label = {c.label: c for c in field.cells}
    assert by_label["T 5/20/2.0"].traded is True
    assert by_label["T 5/20/2.0/1000000000.0"].traded is False
    assert field.degenerate_labels == ("T 5/20/2.0/1000000000.0",)


def test_report_counts_degenerate_cells_so_a_flat_front_is_visible(trend_bars):
    field = optimize.score_field(
        optimize.trend_candidates(fasts=(5,), slows=(20,), trails=(2.0,),
                                  volume_mults=(None, 1e9)),
        trend_bars, symbol="USO", spec=optimize.TREND_FIELD,
        n_blocks=4, n_folds=2)
    report = optimize.build_grid_report(field)
    assert report["degenerate_cells"] == ["T 5/20/2.0/1000000000.0"]
    assert report["cells"][0]["traded"] in (True, False)


def test_volume_axis_is_declared_on_the_field_spec():
    assert optimize.TREND_FIELD.axes == ("fast_period", "slow_period",
                                         "trail_atr_mult", "volume_gate_mult")


def test_gated_cells_construct_a_gated_strategy():
    gated = optimize.trend_candidates(fasts=(20,), slows=(100,), trails=(3.0,),
                                      volume_mults=(1.5,), max_slow=100)
    assert len(gated) == 1
    assert gated[0].strategy.volume_gate_mult == 1.5


@pytest.fixture
def trend_grid_gated(trend_bars):
    return optimize.score_field(
        optimize.trend_candidates(fasts=(5,), slows=(20,), trails=(2.0,),
                                  volume_mults=(None, 1.5)),
        trend_bars, symbol="USO", spec=optimize.TREND_FIELD,
        n_blocks=4, n_folds=2)


def test_score_field_scores_a_trend_field_like_the_tom_grid(trend_grid):
    assert len(trend_grid.cells) == 8  # 2 fast x 2 slow x 2 trail, all feasible
    assert trend_grid.symbol == "USO"
    assert trend_grid.spec is optimize.TREND_FIELD
    assert trend_grid.n_trials == 8
    assert all(math.isfinite(c.sharpe) for c in trend_grid.cells)
    assert all(0.0 <= c.dsr <= 1.0 for c in trend_grid.cells)
    assert all(math.isfinite(c.oos_sharpe) for c in trend_grid.cells)
    assert all(0.0 <= c.max_dd <= 1.0 for c in trend_grid.cells)
    # A live cell outside this reduced field must read as absent, not as zero.
    assert trend_grid.headline is None


def test_scored_trend_cells_carry_their_genome_in_params(trend_grid):
    cell = trend_grid.cell("T 5/20/2.0")
    assert cell is not None
    assert cell.params["fast_period"] == 5
    assert cell.params["slow_period"] == 20
    assert cell.params["trail_atr_mult"] == 2.0
    assert cell.params["stop_atr_mult"] == 2.0


def test_run_grid_is_the_tom_field_through_the_same_pipeline(grid):
    assert grid.spec is optimize.TOM_FIELD
    assert grid.spec.headline_label == optimize.HEADLINE_LABEL
    assert grid.spec.report_prefix == "tom-grid"
    assert optimize.TREND_FIELD.report_prefix == "trend-search"


def test_report_grid_block_follows_the_field_axes(trend_grid):
    report = optimize.build_grid_report(trend_grid)
    assert report["grid"] == {"fast_period": [5, 10], "slow_period": [20, 30],
                              "trail_atr_mult": [2.0, 3.0],
                              "volume_gate_mult": [None]}
    assert report["symbol"] == "USO"
    assert report["cells"][0]["max_dd"] == pytest.approx(
        trend_grid.cells[0].max_dd)


def test_artifact_prefix_comes_from_the_field(tmp_path, trend_grid):
    at = datetime(2026, 9, 5, tzinfo=timezone.utc)
    directory = str(tmp_path / "research")
    json_path = optimize.write_grid_report(
        optimize.build_grid_report(trend_grid, at), at, directory=directory,
        prefix=trend_grid.spec.report_prefix)
    md_path = optimize.write_grid_markdown(trend_grid, at, directory=directory,
                                           prefix=trend_grid.spec.report_prefix)
    assert os.path.basename(json_path) == "trend-search-2026-09-05.json"
    assert os.path.basename(md_path) == "trend-search-2026-09-05.md"


def test_markdown_names_the_field_and_its_axes(trend_grid):
    at = datetime(2026, 9, 5, tzinfo=timezone.utc)
    text = optimize.build_grid_markdown(trend_grid, at)
    assert optimize.TREND_FIELD.name in text
    assert "fast_period 5-10" in text
    assert "slow_period 20-30" in text
    assert "trail_atr_mult 2.0-4.0" not in text  # the field's own range, not the default
    assert "USO" in text
    assert "8 cells" in text
