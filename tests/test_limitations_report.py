import json
import math

import pandas as pd
import pytest

from bot.portfolio import SlippageStat
from config import AssetClass, InstrumentConfig, StrategyName
from docs.research.scripts.limitations_report import (
    FoldDiag,
    Row,
    build_rows,
    describe_mismatch,
    live_status,
    live_variant,
    load_search_docs,
    read_fills,
    render_markdown,
    reproduction_mismatch,
    search_trials_from,
    slippage_by_symbol,
    symbol_record,
)


def test_build_rows_reports_ci_and_both_trial_counts():
    returns = {"QQQ": pd.Series([0.01, -0.005, 0.007, 0.002, -0.001] * 60)}
    trades = {"QQQ": 12}
    gate_trials = {"QQQ": 4}
    search_trials = {"QQQ": None}
    rows = build_rows(returns, trades, gate_trials, search_trials)
    (row,) = rows
    assert isinstance(row, Row)
    assert row.symbol == "QQQ" and row.trades == 12
    assert row.gate_trials == 4 and row.search_trials is None
    assert row.sharpe_ci_low < row.sharpe < row.sharpe_ci_high
    assert math.isclose(row.sharpe_ci_high - row.sharpe, 1.96 * row.sharpe_se, rel_tol=1e-9)
    assert math.isclose(row.sharpe - row.sharpe_ci_low, 1.96 * row.sharpe_se, rel_tol=1e-9)


def test_build_rows_degenerate_returns_give_nan_interval():
    (row,) = build_rows({"GLD": pd.Series([0.0] * 50)}, {"GLD": 0}, {"GLD": 3}, {})
    assert math.isnan(row.sharpe_se)
    assert math.isnan(row.sharpe_ci_low) and math.isnan(row.sharpe_ci_high)
    assert row.search_trials is None


def _cfg(strategy, params, **kw):
    return InstrumentConfig(symbol="X", asset_class=AssetClass.US_EQUITY,
                            strategy=strategy, timeframe_minutes=1440,
                            params=params, **kw)


def test_live_status_classifies_promote_override_and_disabled():
    trend = StrategyName.TREND_FOLLOWING
    assert live_status(_cfg(trend, {}, validation_ref="v.json: PROMOTE")) == "PROMOTE"
    assert live_status(_cfg(trend, {}, override_reason="kept")) == "enabled by override"
    assert live_status(_cfg(trend, {}, enabled=False, override_reason="x")) == "disabled"
    assert live_status(_cfg(trend, {})) == "enabled without verdict"
    assert live_status(None) == "not in lineup"


def test_live_status_flags_promoted_symbol_running_another_variant():
    qqq = _cfg(StrategyName.RSI_MEAN_REVERSION, {"entry_below": 10, "regime_period": 200},
               validation_ref="validation-2026-09-16.json: QQQ PROMOTE")
    assert (live_status(qqq, {"variant": "EMA 50/200 long"})
            == "symbol PROMOTE; live variant not the selected one (owner decision)")
    assert live_status(qqq, {"variant": "RSI2<10 200sma"}) == "PROMOTE"


def test_live_variant_labels_match_research_spelling():
    assert live_variant(_cfg(StrategyName.RSI_MEAN_REVERSION,
                             {"entry_below": 10, "regime_period": 200})) == "RSI2<10 200sma"
    assert live_variant(_cfg(StrategyName.TREND_FOLLOWING,
                             {"fast_period": 20, "slow_period": 100})) == "EMA 20/100 long"
    assert live_variant(_cfg(StrategyName.TREND_FOLLOWING,
                             {"fast_period": 20, "slow_period": 100,
                              "allow_short": True})) == "EMA 20/100 l+s"
    assert live_variant(_cfg(StrategyName.TURN_OF_MONTH,
                             {"entry_window_bdays": 2, "exit_day_of_month": 5})) == "TOM 2/5"
    assert live_variant(None) is None


def test_search_trials_reads_exhaustive_and_flat_shapes():
    docs = [{"symbol": "USO", "exhaustive": {"n_trials": 390}},
            {"symbol": "VOO", "n_trials": 35},
            {"symbol": "GLD"}]
    assert search_trials_from(docs) == {"USO": 390, "VOO": 35}


def _stat(symbol, kind, count):
    return SlippageStat(symbol, kind, count, 0, 1.0, 0.5, 3.0, 2.0)


def test_slippage_by_symbol_reports_sample_size_honestly():
    stats = [_stat("QQQ", "entry", 5), _stat("QQQ", "exit", 2), _stat("SPY", "entry", 20)]
    out = slippage_by_symbol(stats, ["QQQ", "SPY", "BTC/USD"], min_sample=20)
    assert out["QQQ"] == "insufficient sample (n=7)"
    assert out["BTC/USD"] == "no local sample"
    assert out["SPY"] == [{"kind": "entry", "count": 20, "mean_bps": 1.0,
                           "median_bps": 0.5, "worst_bps": 3.0}]


def test_read_fills_counts_rows_and_feeds_slippage(tmp_path):
    path = tmp_path / "fills.csv"
    path.write_text(
        "timestamp,symbol,side,event,decision_mid,decision_spread_bps,limit_price,"
        "fill_price,fill_qty,slippage_bps\n"
        "2026-10-01T14:00:00+00:00,QQQ,buy,entry,100,1,100.1,100.02,1,2.0\n"
        "2026-10-02T14:00:00+00:00,QQQ,sell,exit,101,1,100.9,100.99,1,1.0\n")
    stats, rows = read_fills(path)
    assert rows == 2
    out = slippage_by_symbol(stats, ["QQQ", "SPY"])
    assert out == {"QQQ": "insufficient sample (n=2)", "SPY": "no local sample"}


def test_read_fills_missing_file_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.csv"):
        read_fills(tmp_path / "nope.csv")


def _write(path, doc):
    path.write_text(json.dumps(doc))


def test_load_search_docs_takes_latest_per_symbol_on_or_before_as_of(tmp_path):
    _write(tmp_path / "trend-search-2026-09-05.json", {"symbol": "GLD", "exhaustive": {"n_trials": 60}})
    _write(tmp_path / "trend-search-2026-09-12.json", {"symbol": "USO", "exhaustive": {"n_trials": 130}})
    _write(tmp_path / "trend-search-2026-09-14.json", {"symbol": "USO", "exhaustive": {"n_trials": 390}})
    _write(tmp_path / "trend-search-2026-11-01.json", {"symbol": "USO", "exhaustive": {"n_trials": 999}})
    _write(tmp_path / "trend-search-latest.json", {"symbol": "GLD", "exhaustive": {"n_trials": 1}})
    _write(tmp_path / "tom-grid-2026-09-07.json", {"symbol": "VOO", "n_trials": 35})
    docs, sources = load_search_docs(tmp_path, "2026-10-12")
    assert search_trials_from(docs) == {"GLD": 60, "USO": 390, "VOO": 35}
    assert "trend-search-latest.json" not in sources
    assert "trend-search-2026-11-01.json" not in sources


def test_reproduction_mismatch_names_symbol_and_both_values():
    assert reproduction_mismatch("QQQ", 0.655, 0.658, tol=0.01) is None
    message = reproduction_mismatch("QQQ", 0.60, 0.658, tol=0.01)
    assert "QQQ" in message and "0.600" in message and "0.658" in message
    assert reproduction_mismatch("QQQ", float("nan"), 0.658, tol=0.01) is not None


def _entry(oos=-0.102):
    return {"symbol": "SPY", "variant": "RSI2<10 200sma", "verdict": "REJECT",
            "oos_sharpe": oos, "variant_oos": 0.785, "dsr": 0.941}


def _row(sharpe=0.128):
    return Row("SPY", 9, sharpe, 0.3, sharpe - 0.588, sharpe + 0.588, 6, None)


def test_symbol_record_flags_mismatch_and_labels_rerun_stats():
    rec = symbol_record(_row(), _entry(), None, "insufficient sample (n=4)",
                        reproducible=False, note="drifted")
    assert rec["reproducible"] is False and rec["reproduction_note"] == "drifted"
    assert rec["oos_sharpe_artifact"] == -0.102 and rec["oos_sharpe_rerun"] == 0.128
    assert rec["rerun"] == {"sharpe": 0.128, "sharpe_se": 0.3,
                            "sharpe_ci_low": 0.128 - 0.588,
                            "sharpe_ci_high": 0.128 + 0.588, "trades": 9}
    assert "sharpe" not in rec and "trades" not in rec
    assert rec["live_status"] == "not in lineup"


def test_symbol_record_marks_match_reproducible():
    rec = symbol_record(_row(sharpe=-0.1), _entry(), None, "no local sample",
                        reproducible=True, note="ignored")
    assert rec["reproducible"] is True and rec["reproduction_note"] is None


def test_describe_mismatch_states_ties_flat_winners_and_moved_variants():
    folds = [FoldDiag(0, {"TSMOM ens": 0.99214, "TSMOM 12-1": 0.99214, "RSI2<10": 0.4},
                      "TSMOM ens", 3),
             FoldDiag(1, {"EMA": 0.0, "RSI2<10": -0.4}, "EMA", 0),
             FoldDiag(2, {"EMA": 0.1, "RSI2<10": 1.2}, "RSI2<10", 7)]
    note = describe_mismatch(folds, {"TSMOM ens": 0.514, "RSI2<10": 0.787},
                             {"TSMOM ens": 0.296, "RSI2<10": 0.785})
    assert "fold 0" in note and "TSMOM ens" in note and "TSMOM 12-1" in note
    assert "0.9921" in note and "tie" in note
    assert "fold 1" in note and "EMA" in note and "no in-sample trades" in note
    assert "fold 2" not in note
    assert "0.296 -> 0.514" in note
    assert "RSI2<10" in note.split("within")[-1]
    assert "cause not determined" in note and "26fcdd3" in note
    assert "leading hypothesis" in note


def _md_record(rec):
    return {"as_of": "2026-10-12",
            "artifact": "docs/validation/validation-2026-09-16.json",
            "fills_source": {"file": "fills-live.csv", "rows": 20, "note": "container copy"},
            "symbols": [rec]}


def test_markdown_header_names_artifact_and_as_of():
    rec = symbol_record(_row(sharpe=-0.1), _entry(), None, "insufficient sample (n=7)",
                        reproducible=True)
    md = render_markdown(_md_record(rec))
    header = md.splitlines()[0]
    assert "validation-2026-09-16.json" in header and "2026-10-12" in header
    assert "fills-live.csv (20 rows): container copy" in md
    assert "| SPY |" in md and "insufficient sample (n=7)" in md


def test_markdown_footnotes_non_reproducible_row_and_keeps_artifact_value():
    rec = symbol_record(_row(), _entry(), None, "no local sample",
                        reproducible=False, note="bars drifted")
    md = render_markdown(_md_record(rec))
    (line,) = [ln for ln in md.splitlines() if ln.startswith("| SPY")]
    assert line.startswith("| SPY [^1] |")
    cells = [c.strip() for c in line.split("|")]
    assert cells[6] == "-0.10" and cells[7] == "0.13"
    assert "[^1]: not reproducible" in md and "bars drifted" in md
