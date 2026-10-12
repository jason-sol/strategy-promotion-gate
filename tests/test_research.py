import inspect
import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from functools import partial

import dotenv
import numpy as np
import pandas as pd
import pytest

from bot import research
from bot.metrics import probabilistic_sharpe_ratio
from bot.research import (
    CRYPTO_START,
    DAILY,
    EQUITY_START,
    EXPERIMENTS,
    SymbolValidation,
    ValidationArtifactDegraded,
    VariantResult,
    build_validation_report,
    judge_symbols,
    omitted_params,
    print_validation,
    strategy_params,
    tom,
    tsmom,
    validate_symbol,
    write_validation_report,
)
from bot.strategies import build_strategy
from bot.strategies.pead_drift import PeadDrift
from bot.strategies.ts_momentum import TsMomentum
from bot.strategies.turn_of_month import TurnOfMonth
from bot.validation import (
    Candidate,
    CscvReport,
    build_return_matrix,
    deflated_sharpes,
    deflated_sharpes_from_matrix,
    walk_forward,
)
from config import (
    INSTRUMENTS,
    MARKET_DATA_FEED,
    VALIDATION_REPORT_DIR,
    RISK,
    VALIDATION,
    AssetClass,
    InstrumentConfig,
    StrategyName,
)
from helpers import daily_bars

# Contiguous calendar days: no weekend gap, so resampling to 1D is the identity.
calendar_bars = partial(daily_bars, freq="1D")


def mr_cfg(symbol: str, entry_z: float) -> InstrumentConfig:
    return InstrumentConfig(
        symbol=symbol, asset_class=AssetClass.US_EQUITY,
        strategy=StrategyName.MEAN_REVERSION, timeframe_minutes=1440,
        params={"sma_period": 10, "entry_z": entry_z, "allow_short": False},
        slippage_pct=0.0,
    )


def candidate(label: str, symbol: str, entry_z: float) -> Candidate:
    cfg = mr_cfg(symbol, entry_z)
    return Candidate(label, cfg, build_strategy(cfg))




def choppy_uptrend(n: int) -> list[float]:
    t = np.arange(n)
    return list(100.0 + 0.5 * t + 4.0 * np.sin(t / 2.0))


def divergent_closes(seed: int) -> list[float]:
    """A regime whose volatile opening block (in full-sample, out of every OOS
    window) rewards the deep-dip variant, while the calmer remainder rewards the
    shallow-dip variant. Full-sample ranking and walk-forward OOS ranking then
    disagree, isolating which one drives selection."""
    rng = np.random.default_rng(seed)
    n = 260
    t = np.arange(n)
    early = t < n * 0.2
    amp = np.where(early, rng.uniform(5, 12), rng.uniform(1, 3))
    freq = np.where(early, rng.uniform(1.5, 3), rng.uniform(0.4, 0.8))
    phase = rng.uniform(0, 6.28)
    noise = rng.standard_normal(n) * rng.uniform(0.5, 2.0)
    drift = rng.uniform(0.1, 0.5)
    return list(100.0 + drift * t + amp * np.sin(t / freq + phase) + noise)


class TestPassesValidation:
    @pytest.mark.parametrize(
        "dsr,p_oos_loss,oos_sharpe,expected",
        [
            (0.96, 0.4, 1.0, True),          # clear pass
            (0.95, 0.5, 0.01, True),         # boundaries inclusive, oos just positive
            (0.94, 0.4, 1.0, False),         # dsr below floor
            (0.96, 0.6, 1.0, False),         # p_oos_loss above ceiling
            (0.96, 0.51, 1.0, False),        # just above the ceiling still fails
            (0.96, float("nan"), 1.0, True), # nan p_oos_loss is a pass-through
            (0.96, 0.4, 0.0, False),         # oos must be strictly positive
            (0.96, 0.4, -0.5, False),        # negative oos fails
            (float("nan"), 0.4, 1.0, False), # nan dsr fails
            (0.96, 0.4, float("nan"), False),# nan oos fails
        ],
    )
    def test_truth_table(self, dsr, p_oos_loss, oos_sharpe, expected):
        assert VALIDATION.passes_validation(dsr, p_oos_loss, oos_sharpe) is expected

    def test_pbo_is_not_an_input_to_the_gate(self):
        assert not hasattr(VALIDATION, "pbo_max")
        assert "pbo" not in inspect.signature(VALIDATION.passes_validation).parameters


class TestValidateSymbol:
    def test_summary_ranks_dominant_variant_and_reports_probabilities(self):
        bars = calendar_bars(choppy_uptrend(260))
        variants = [
            ("dip-buyer", lambda: build_strategy(mr_cfg("AAA", 1.0))),
            ("inert", lambda: build_strategy(mr_cfg("AAA", 6.0))),
        ]
        result = validate_symbol(
            "AAA", AssetClass.US_EQUITY, 0.0, variants, bars,
            n_blocks=4, n_folds=4,
        )
        assert {v.label for v in result.variants} == {"dip-buyer", "inert"}
        assert result.best_label == "dip-buyer"
        assert 0.0 <= result.best_dsr <= 1.0
        assert 0.0 <= result.pbo <= 1.0
        assert 0.0 <= result.p_oos_loss <= 1.0
        assert isinstance(result.promoted, bool)

    def test_selection_follows_walk_forward_oos_not_full_sample(self):
        bars = calendar_bars(divergent_closes(259))
        variants = [
            ("z1.0", lambda: build_strategy(mr_cfg("AAA", 1.0))),
            ("z2.0", lambda: build_strategy(mr_cfg("AAA", 2.0))),
        ]
        # Full-sample observed Sharpe ranks z2.0 top; walk-forward OOS ranks z1.0.
        matrix = build_return_matrix(
            [candidate("z1.0", "AAA", 1.0), candidate("z2.0", "AAA", 2.0)],
            {"AAA": bars},
        )
        fs = {c: matrix[c].mean() / matrix[c].std() for c in matrix.columns}
        assert max(fs, key=fs.get) == "z2.0"

        result = validate_symbol(
            "AAA", AssetClass.US_EQUITY, 0.0, variants, bars, n_blocks=4, n_folds=4,
        )
        assert result.best_label == "z1.0"

    def test_variant_carries_walk_forward_oos_sharpe(self):
        bars = calendar_bars(divergent_closes(259))
        variants = [
            ("z1.0", lambda: build_strategy(mr_cfg("AAA", 1.0))),
            ("z2.0", lambda: build_strategy(mr_cfg("AAA", 2.0))),
        ]
        result = validate_symbol(
            "AAA", AssetClass.US_EQUITY, 0.0, variants, bars, n_blocks=4, n_folds=4,
        )
        wf = walk_forward(
            [candidate("z1.0", "AAA", 1.0), candidate("z2.0", "AAA", 2.0)],
            bars, n_folds=4,
        )
        by_label = {v.label: v.oos for v in result.variants}
        assert by_label == pytest.approx(wf.variant_oos_sharpe)

    def test_all_nan_oos_falls_back_to_first_column(self):
        # Bars too short for any fold: walk-forward yields no per-variant OOS, so
        # selection defers to matrix column order rather than crashing.
        bars = calendar_bars(choppy_uptrend(6))
        variants = [
            ("first", lambda: build_strategy(mr_cfg("AAA", 1.0))),
            ("second", lambda: build_strategy(mr_cfg("AAA", 2.0))),
        ]
        result = validate_symbol(
            "AAA", AssetClass.US_EQUITY, 0.0, variants, bars, n_blocks=4, n_folds=4,
        )
        assert result.best_label == "first"

    def test_print_marks_oos_winner_and_shows_oos_column(self, capsys):
        bars = calendar_bars(divergent_closes(259))
        variants = [
            ("z1.0", lambda: build_strategy(mr_cfg("AAA", 1.0))),
            ("z2.0", lambda: build_strategy(mr_cfg("AAA", 2.0))),
        ]
        result = validate_symbol(
            "AAA", AssetClass.US_EQUITY, 0.0, variants, bars, n_blocks=4, n_folds=4,
        )
        print_validation(result)
        out = capsys.readouterr().out
        winner_line = next(ln for ln in out.splitlines() if "<-- best" in ln)
        assert "z1.0" in winner_line
        oos = next(v.oos for v in result.variants if v.label == "z1.0")
        assert format(oos, ".2f") in winner_line

    def test_duplicate_shortlist_labels_are_refused(self):
        # The return matrix is keyed by label, so two variants sharing one would
        # collapse into a single column and the artifact would then assert the
        # loser was never tried.
        bars = calendar_bars(choppy_uptrend(260))
        variants = [
            ("EMA 20/100 long", lambda: build_strategy(mr_cfg("AAA", 1.0))),
            ("EMA 20/100 long", lambda: build_strategy(mr_cfg("AAA", 2.0))),
            ("EMA 50/200 long", lambda: build_strategy(mr_cfg("AAA", 3.0))),
        ]
        with pytest.raises(ValueError, match="EMA 20/100 long"):
            validate_symbol("AAA", AssetClass.US_EQUITY, 0.0, variants, bars,
                            n_blocks=4, n_folds=4)


class TestSleeveFactories:
    def test_tsmom_default_is_ensemble_with_risk_atr(self):
        s = tsmom()
        assert isinstance(s, TsMomentum)
        assert s.lookbacks == [126, 147, 168, 189, 210, 231, 252]
        assert s.skip_days == 21
        assert s.atr_period == RISK.atr_period

    def test_tsmom_single_lookback_is_classic_12_1(self):
        s = tsmom([252])
        assert isinstance(s, TsMomentum)
        assert s.lookbacks == [252]
        assert s.atr_period == RISK.atr_period

    def test_tom_sets_window_params_and_risk_atr(self):
        s = tom(2, 3)
        assert isinstance(s, TurnOfMonth)
        assert s.entry_window_bdays == 2
        assert s.exit_day_of_month == 3
        assert s.atr_period == RISK.atr_period


class TestExperiments:
    def _grouped(self):
        groups: dict[str, list[str]] = {}
        for symbol, _ac, _slip, label, _factory in EXPERIMENTS:
            groups.setdefault(symbol, []).append(label)
        return groups

    def test_new_sleeves_appear_under_expected_symbols(self):
        groups = self._grouped()
        assert "TSMOM ens 6-12m" in groups["SPY"]
        assert "TSMOM 12-1" in groups["SPY"]
        assert "TSMOM ens 6-12m" in groups["QQQ"]
        assert "TSMOM ens 6-12m" in groups["BTC/USD"]
        assert "TSMOM ens 6-12m" in groups["USO"]
        assert groups["VOO"] == ["TOM 2/3", "TOM 3/3", "TOM 2/5"]

    def test_voo_is_us_equity_with_expected_slippage(self):
        voo = [row for row in EXPERIMENTS if row[0] == "VOO"]
        assert voo, "VOO must be present"
        assert all(ac == AssetClass.US_EQUITY and slip == 0.0001
                   for _sym, ac, slip, _label, _f in voo)

    def test_btc_and_uso_tsmom_use_expected_slippage(self):
        for symbol in ("BTC/USD", "USO"):
            row = next(r for r in EXPERIMENTS
                       if r[0] == symbol and r[3] == "TSMOM ens 6-12m")
            assert row[2] == 0.0005

    def test_every_label_fits_print_column(self):
        # print_validation lays the variant column out 16 wide + padding.
        assert all(len(label) <= 16 for _sym, _ac, _slip, label, _f in EXPERIMENTS)

    def test_variant_counts_per_symbol(self):
        counts = {sym: len(labels) for sym, labels in self._grouped().items()}
        assert counts == {"SPY": 6, "QQQ": 4, "BTC/USD": 4, "GLD": 3, "USO": 4,
                          "VOO": 3}

    def test_each_symbol_group_is_contiguous(self):
        symbols = [row[0] for row in EXPERIMENTS]
        # A symbol must not reappear after another symbol has intervened.
        seen: set[str] = set()
        for prev, cur in zip(symbols, symbols[1:]):
            if cur != prev:
                assert cur not in seen, f"{cur} group is not contiguous"
                seen.add(prev)

    def test_factories_construct_without_error(self):
        for _sym, _ac, _slip, _label, factory in EXPERIMENTS:
            assert factory() is not None

    def test_tsmom_min_bars_below_shortest_history_train_block(self):
        # BTC has the least history (~2030 daily bars). Walk-forward's first train
        # block is n/(wf_folds+1); the ensemble must warm up well inside it.
        btc_bars = 2030
        train_block = btc_bars // (VALIDATION.wf_folds + 1)
        assert tsmom().min_bars < train_block


class TestTurnOfMonthEndToEnd:
    def test_validate_symbol_runs_over_tz_aware_daily_bars(self):
        # Research bars carry a tz-aware UTC DatetimeIndex; TOM tz-strips internally.
        bars = calendar_bars(choppy_uptrend(260))
        assert bars.index.tz is not None
        variants = [
            ("TOM 2/3", lambda: tom(2, 3)),
            ("TOM 3/3", lambda: tom(3, 3)),
            ("TOM 2/5", lambda: tom(2, 5)),
        ]
        result = validate_symbol(
            "VOO", AssetClass.US_EQUITY, 0.0001, variants, bars,
            n_blocks=4, n_folds=4,
        )
        labels = {v.label for v in result.variants}
        assert labels == {"TOM 2/3", "TOM 3/3", "TOM 2/5"}
        assert result.best_label in labels
        for v in result.variants:
            assert math.isnan(v.dsr) or 0.0 <= v.dsr <= 1.0
            assert isinstance(v.oos, float)
        assert math.isnan(result.pbo) or 0.0 <= result.pbo <= 1.0
        assert math.isnan(result.p_oos_loss) or 0.0 <= result.p_oos_loss <= 1.0
        assert isinstance(result.oos_sharpe, float)
        assert isinstance(result.promoted, bool)


class TestDeflatedSharpes:
    def test_best_dsr_is_below_its_undeflated_psr(self):
        bars_by_symbol = {"AAA": calendar_bars(choppy_uptrend(260))}
        candidates = [
            candidate("z1.0", "AAA", 1.0),
            candidate("z1.5", "AAA", 1.5),
            candidate("z2.0", "AAA", 2.0),
        ]
        matrix = build_return_matrix(candidates, bars_by_symbol)
        per_period = {c: matrix[c].mean() / matrix[c].std() for c in matrix
                      if matrix[c].std() > 0}
        best = max(per_period, key=per_period.get)

        dsr = deflated_sharpes(candidates, bars_by_symbol)
        psr_best = probabilistic_sharpe_ratio(matrix[best], 0.0)

        assert dsr[best] < psr_best  # deflation against >1 trial lowers it
        assert all(0.0 <= v <= 1.0 for v in dsr.values())

    def test_from_matrix_matches_building_from_candidates(self):
        bars_by_symbol = {"AAA": calendar_bars(choppy_uptrend(260))}
        candidates = [candidate("z1.0", "AAA", 1.0), candidate("z2.0", "AAA", 2.0)]
        matrix = build_return_matrix(candidates, bars_by_symbol)
        assert (deflated_sharpes_from_matrix(matrix, len(candidates))
                == deflated_sharpes(candidates, bars_by_symbol))

    def test_single_variant_uses_zero_variance_not_nan(self):
        # ddof=1 sample variance is undefined for one column; DSR must stay a
        # finite probability (the undeflated PSR), not NaN.
        bars_by_symbol = {"AAA": calendar_bars(choppy_uptrend(260))}
        candidates = [candidate("solo", "AAA", 1.0)]
        matrix = build_return_matrix(candidates, bars_by_symbol)
        dsr = deflated_sharpes_from_matrix(matrix, 1)
        assert not math.isnan(dsr["solo"])
        assert dsr["solo"] == probabilistic_sharpe_ratio(matrix["solo"], 0.0)


def cscv(pbo=0.1, p_oos_loss=0.1, median_logit=0.42, degradation_slope=-0.8,
         oos_rank_mean=None, is_best_share=None) -> CscvReport:
    """A hand-built CSCV report: the reporting layer only reads its numbers."""
    return CscvReport(
        pbo=pbo, p_oos_loss=p_oos_loss, median_logit=median_logit,
        degradation_slope=degradation_slope, logits=(0.3, 0.5),
        oos_rank_mean=oos_rank_mean if oos_rank_mean is not None
        else {"winner": 0.75, "also-ran": 0.25},
        is_best_share=is_best_share if is_best_share is not None
        else {"winner": 0.9, "also-ran": 0.1},
    )


def symbol_validation(symbol="AAA", dsr=0.99, oos=1.2, pbo=0.1, p_oos_loss=0.1,
                      strategy="TurnOfMonth", params=None,
                      params_omitted=(), report=None) -> SymbolValidation:
    """A hand-built harness result, so report tests never fetch bars."""
    return SymbolValidation(
        symbol=symbol,
        variants=[
            VariantResult("winner", dsr, oos, strategy,
                          params if params is not None else {"entry_window_bdays": 2},
                          list(params_omitted)),
            VariantResult("also-ran", 0.10, -0.5, strategy, {"entry_window_bdays": 9}),
        ],
        cscv=report if report is not None else cscv(pbo=pbo, p_oos_loss=p_oos_loss),
        oos_sharpe=oos,
        best_label="winner",
        buy_and_hold=0.25,
    )


def test_print_validation_reports_p_oos_loss_beside_pbo(capsys):
    print_validation(symbol_validation(pbo=0.962, p_oos_loss=0.184))
    out = capsys.readouterr().out
    verdict_line = next(ln for ln in out.splitlines() if "VERDICT" in ln)
    assert "PBO 0.962" in verdict_line
    assert "0.184" in verdict_line
    assert "P(OOS loss)" in verdict_line


class TestBuildValidationReport:
    def test_header_records_the_run_provenance(self):
        report = build_validation_report(
            [symbol_validation()],
            generated_at=datetime(2026, 9, 2, 4, 30, tzinfo=timezone.utc),
        )
        assert report["generated_at"] == "2026-09-02T04:30:00+00:00"
        assert report["market_data_feed"] == MARKET_DATA_FEED
        assert report["equity_start"] == EQUITY_START.date().isoformat()
        assert report["crypto_start"] == CRYPTO_START.date().isoformat()

    def test_one_entry_per_symbol_carries_the_promoted_variant(self):
        report = build_validation_report([
            symbol_validation("AAA"), symbol_validation("BBB"),
        ])
        assert [r["symbol"] for r in report["results"]] == ["AAA", "BBB"]
        entry = report["results"][0]
        assert entry["variant"] == "winner"
        assert entry["strategy"] == "TurnOfMonth"
        assert entry["params"] == {"entry_window_bdays": 2}
        assert entry["dsr"] == pytest.approx(0.99)
        assert entry["oos_sharpe"] == pytest.approx(1.2)
        assert entry["pbo"] == pytest.approx(0.1)

    def test_entry_carries_every_cscv_panel_not_just_pbo(self):
        report = build_validation_report([symbol_validation(
            report=cscv(pbo=0.9, p_oos_loss=0.2, median_logit=1.5,
                        degradation_slope=-0.6))])
        entry = report["results"][0]
        assert entry["pbo"] == pytest.approx(0.9)
        assert entry["p_oos_loss"] == pytest.approx(0.2)
        assert entry["median_logit"] == pytest.approx(1.5)
        assert entry["degradation_slope"] == pytest.approx(-0.6)
        # the per-candidate view for the variant the verdict is about
        assert entry["variant_oos_rank_mean"] == pytest.approx(0.75)
        assert entry["variant_is_best_share"] == pytest.approx(0.9)

    def test_a_variant_missing_from_the_cscv_panels_records_null(self):
        report = build_validation_report([symbol_validation(
            report=cscv(oos_rank_mean={}, is_best_share={}))])
        entry = report["results"][0]
        assert entry["variant_oos_rank_mean"] is None
        assert entry["variant_is_best_share"] is None

    def test_entry_lists_every_shortlist_variant_in_order(self):
        result = SymbolValidation(
            symbol="AAA",
            variants=[
                VariantResult("EMA 50/200 long", 0.42, 0.31, "TrendFollowing"),
                VariantResult("EMA 20/100 long", 0.97, 1.40, "TrendFollowing"),
                VariantResult("EMA 10/50 long", 0.11, -0.22, "TrendFollowing"),
            ],
            cscv=cscv(oos_rank_mean={}, is_best_share={}),
            oos_sharpe=1.40,
            best_label="EMA 20/100 long",
            buy_and_hold=0.25,
        )
        entry = build_validation_report([result])["results"][0]
        assert [v["variant"] for v in entry["variants"]] == [
            "EMA 50/200 long", "EMA 20/100 long", "EMA 10/50 long"]
        assert [v["dsr"] for v in entry["variants"]] == pytest.approx(
            [0.42, 0.97, 0.11])
        assert [v["oos_sharpe"] for v in entry["variants"]] == pytest.approx(
            [0.31, 1.40, -0.22])

    def test_variants_carry_the_numbers_print_validation_prints(self, capsys):
        result = symbol_validation()
        print_validation(result)
        printed = [ln for ln in capsys.readouterr().out.splitlines() if "->" not in ln]
        entry = build_validation_report([result])["results"][0]
        for row in entry["variants"]:
            line = next(ln for ln in printed if row["variant"] in ln)
            assert format(row["dsr"], ".3f") in line
            assert format(row["oos_sharpe"], ".2f") in line

    def test_selected_variant_is_among_the_variants_with_matching_numbers(self):
        entry = build_validation_report(
            [symbol_validation(dsr=0.97, oos=1.35)])["results"][0]
        selected = [v for v in entry["variants"] if v["variant"] == entry["variant"]]
        assert len(selected) == 1
        assert selected[0]["dsr"] == pytest.approx(entry["dsr"])
        assert selected[0]["oos_sharpe"] == pytest.approx(entry["variant_oos"])

    def test_variant_nan_metrics_serialize_as_null(self):
        nan = float("nan")
        result = SymbolValidation(
            symbol="AAA",
            variants=[VariantResult("winner", 0.99, 1.2, "TurnOfMonth"),
                      VariantResult("unmeasurable", nan, nan, "TurnOfMonth")],
            cscv=cscv(oos_rank_mean={}, is_best_share={}),
            oos_sharpe=1.2,
            best_label="winner",
            buy_and_hold=0.0,
        )
        entry = build_validation_report([result])["results"][0]
        assert entry["variants"][1] == {"variant": "unmeasurable",
                                        "dsr": None, "oos_sharpe": None}
        assert json.loads(json.dumps(entry, allow_nan=False)) == entry

    @pytest.mark.parametrize(
        "dsr,p_oos_loss,oos,expected",
        [
            (0.99, 0.1, 1.2, "PROMOTE"),
            (0.99, float("nan"), 1.2, "PROMOTE"),   # nan is a pass-through
            (0.90, 0.1, 1.2, "REJECT"),
            (0.99, 0.7, 1.2, "REJECT"),
            (0.99, 0.1, -0.3, "REJECT"),
            (float("nan"), 0.1, 1.2, "REJECT"),
        ],
    )
    def test_verdict_follows_passes_validation(self, dsr, p_oos_loss, oos,
                                               expected):
        result = symbol_validation(dsr=dsr, p_oos_loss=p_oos_loss, oos=oos)
        assert result.promoted is (expected == "PROMOTE")
        report = build_validation_report([result])
        assert report["results"][0]["verdict"] == expected

    @pytest.mark.parametrize(
        "pbo,p_oos_loss,expected",
        [
            (0.9, 0.1, "PROMOTE"),  # PBO is advisory: it cannot close the gate
            (0.1, 0.9, "REJECT"),   # P(OOS loss) is what binds
        ],
    )
    def test_verdict_follows_p_oos_loss_not_pbo(self, pbo, p_oos_loss, expected):
        result = symbol_validation(pbo=pbo, p_oos_loss=p_oos_loss)
        report = build_validation_report([result])
        assert report["results"][0]["pbo"] == pytest.approx(pbo)
        assert report["results"][0]["verdict"] == expected

    def test_nan_metrics_serialize_as_null(self):
        nan = float("nan")
        report = build_validation_report([
            symbol_validation(dsr=nan, oos=nan,
                              report=CscvReport(nan, nan, nan, nan, (), {}, {})),
        ])
        entry = report["results"][0]
        assert entry["dsr"] is None and entry["pbo"] is None
        assert entry["p_oos_loss"] is None and entry["median_logit"] is None
        assert entry["degradation_slope"] is None
        assert entry["oos_sharpe"] is None
        # allow_nan=False proves no NaN survived anywhere: bare NaN is invalid JSON
        # and readers that accept it disagree about what it means.
        assert json.loads(json.dumps(report, allow_nan=False)) == report

    def test_empty_run_still_produces_a_readable_report(self):
        report = build_validation_report([])
        assert report["results"] == []
        assert report["generated_at"]


class TestWriteValidationReport:
    def test_file_is_named_for_the_run_date(self, tmp_path):
        at = datetime(2026, 9, 2, 4, 30, tzinfo=timezone.utc)
        report = build_validation_report([symbol_validation()], generated_at=at)
        path = write_validation_report(report, at, str(tmp_path))
        assert os.path.basename(path) == "validation-2026-09-02.json"
        with open(path) as fh:
            assert json.load(fh) == report

    def test_variant_rows_survive_the_writer(self, tmp_path):
        at = datetime(2026, 9, 2, tzinfo=timezone.utc)
        report = build_validation_report([symbol_validation()], generated_at=at)
        path = write_validation_report(report, at, str(tmp_path))
        with open(path) as fh:
            written = json.load(fh)
        assert [v["variant"] for v in written["results"][0]["variants"]] == [
            "winner", "also-ran"]

    def test_missing_directory_is_created(self, tmp_path):
        at = datetime(2026, 9, 2, tzinfo=timezone.utc)
        report = build_validation_report([symbol_validation()], generated_at=at)
        path = write_validation_report(report, at, str(tmp_path / "docs" / "validation"))
        assert os.path.exists(path)

    def test_rerunning_the_same_day_overwrites_rather_than_accumulates(self, tmp_path):
        at = datetime(2026, 9, 2, tzinfo=timezone.utc)
        write_validation_report(
            build_validation_report([symbol_validation("AAA")], generated_at=at),
            at, str(tmp_path))
        path = write_validation_report(
            build_validation_report([symbol_validation("BBB")], generated_at=at),
            at, str(tmp_path))
        assert len(list(tmp_path.iterdir())) == 1
        with open(path) as fh:
            assert json.load(fh)["results"][0]["symbol"] == "BBB"

    def test_a_rerun_downgrading_a_verdict_to_error_is_refused(self, tmp_path):
        at = datetime(2026, 9, 2, tzinfo=timezone.utc)
        good = build_validation_report([symbol_validation("AAA")], generated_at=at)
        path = write_validation_report(good, at, str(tmp_path))
        keyless = build_validation_report([], {"AAA": "no API key — not attempted"},
                                          generated_at=at)
        with pytest.raises(ValidationArtifactDegraded) as excinfo:
            write_validation_report(keyless, at, str(tmp_path))
        assert "AAA" in str(excinfo.value)
        with open(path) as fh:
            assert json.load(fh) == good

    def test_a_degrading_rerun_can_be_forced(self, tmp_path):
        at = datetime(2026, 9, 2, tzinfo=timezone.utc)
        write_validation_report(
            build_validation_report([symbol_validation("AAA")], generated_at=at),
            at, str(tmp_path))
        keyless = build_validation_report([], {"AAA": "no API key"}, generated_at=at)
        path = write_validation_report(keyless, at, str(tmp_path),
                                       allow_degrade=True)
        with open(path) as fh:
            assert json.load(fh)["results"][0]["verdict"] == "ERROR"

    def test_a_rerun_replacing_an_error_with_a_verdict_overwrites(self, tmp_path):
        at = datetime(2026, 9, 2, tzinfo=timezone.utc)
        write_validation_report(
            build_validation_report([], {"AAA": "no API key"}, generated_at=at),
            at, str(tmp_path))
        path = write_validation_report(
            build_validation_report([symbol_validation("AAA")], generated_at=at),
            at, str(tmp_path))
        with open(path) as fh:
            assert json.load(fh)["results"][0]["verdict"] == "PROMOTE"

    def test_an_error_row_for_an_unrecorded_symbol_writes_freely(self, tmp_path):
        at = datetime(2026, 9, 2, tzinfo=timezone.utc)
        write_validation_report(
            build_validation_report([symbol_validation("AAA")], generated_at=at),
            at, str(tmp_path))
        path = write_validation_report(
            build_validation_report([symbol_validation("AAA")],
                                    {"BBB": "no API key"}, generated_at=at),
            at, str(tmp_path))
        with open(path) as fh:
            verdicts = {e["symbol"]: e["verdict"] for e in json.load(fh)["results"]}
        assert verdicts == {"AAA": "PROMOTE", "BBB": "ERROR"}

    def test_a_corrupt_existing_artifact_does_not_block_the_write(
            self, tmp_path, caplog):
        at = datetime(2026, 9, 2, tzinfo=timezone.utc)
        (tmp_path / "validation-2026-09-02.json").write_text("{not json")
        report = build_validation_report([], {"AAA": "no API key"}, generated_at=at)
        with caplog.at_level(logging.WARNING):
            path = write_validation_report(report, at, str(tmp_path))
        with open(path) as fh:
            assert json.load(fh) == report
        assert "validation-2026-09-02.json" in caplog.text

    def test_the_default_directory_is_versioned_not_runtime_state(self):
        # A citation has to survive a fresh clone, so the artifact is committed
        # alongside the docs rather than left in the untracked runtime dir.
        assert VALIDATION_REPORT_DIR == os.path.join("docs", "validation")


class TestWriteValidationReportIsAtomic:
    """A half-written artifact erases cited evidence and disarms the guard that
    protects it, because an unparseable file makes the guard stand down."""

    def _unserialisable(self, at):
        report = build_validation_report([symbol_validation("AAA")],
                                         generated_at=at)
        report["results"][0]["dsr"] = float("nan")
        return report

    def test_a_failed_write_leaves_the_existing_artifact_byte_identical(self, tmp_path):
        at = datetime(2026, 9, 2, tzinfo=timezone.utc)
        good = build_validation_report([symbol_validation("AAA")], generated_at=at)
        path = write_validation_report(good, at, str(tmp_path))
        with open(path, "rb") as fh:
            before = fh.read()

        with pytest.raises(ValueError):
            write_validation_report(self._unserialisable(at), at, str(tmp_path))

        with open(path, "rb") as fh:
            assert fh.read() == before

    def test_a_failed_write_leaves_no_scratch_file_behind(self, tmp_path):
        at = datetime(2026, 9, 2, tzinfo=timezone.utc)
        path = write_validation_report(
            build_validation_report([symbol_validation("AAA")], generated_at=at),
            at, str(tmp_path))

        with pytest.raises(ValueError):
            write_validation_report(self._unserialisable(at), at, str(tmp_path))

        assert [p.name for p in tmp_path.iterdir()] == [os.path.basename(path)]

    def test_the_degrade_guard_still_arms_after_a_failed_write(self, tmp_path):
        at = datetime(2026, 9, 2, tzinfo=timezone.utc)
        write_validation_report(
            build_validation_report([symbol_validation("AAA")], generated_at=at),
            at, str(tmp_path))
        with pytest.raises(ValueError):
            write_validation_report(self._unserialisable(at), at, str(tmp_path))

        keyless = build_validation_report([], {"AAA": "no API key"},
                                          generated_at=at)
        with pytest.raises(ValidationArtifactDegraded):
            write_validation_report(keyless, at, str(tmp_path))

    def test_a_first_write_that_fails_creates_no_artifact(self, tmp_path):
        at = datetime(2026, 9, 2, tzinfo=timezone.utc)
        with pytest.raises(ValueError):
            write_validation_report(self._unserialisable(at), at, str(tmp_path))
        assert list(tmp_path.iterdir()) == []


class TestWriteValidationReportRefusesEveryDowngrade:
    """PROMOTE outranks REJECT outranks ERROR. Any strict fall is refused: a
    same-day rerun that turns a cited PROMOTE into a REJECT erases the same
    evidence an ERROR row would."""

    def _reject(self, at, symbol="AAA"):
        return build_validation_report(
            [symbol_validation(symbol, dsr=0.10, oos=-0.5, pbo=0.9)],
            generated_at=at)

    def _promote(self, at, symbol="AAA"):
        return build_validation_report([symbol_validation(symbol)],
                                       generated_at=at)

    def test_a_rerun_downgrading_a_promote_to_reject_is_refused(self, tmp_path):
        at = datetime(2026, 9, 2, tzinfo=timezone.utc)
        promoted = self._promote(at)
        path = write_validation_report(promoted, at, str(tmp_path))

        with pytest.raises(ValidationArtifactDegraded) as excinfo:
            write_validation_report(self._reject(at), at, str(tmp_path))

        assert "AAA" in str(excinfo.value)
        with open(path) as fh:
            assert json.load(fh) == promoted

    def test_the_refusal_names_the_old_and_the_new_verdict_per_symbol(self, tmp_path):
        at = datetime(2026, 9, 2, tzinfo=timezone.utc)
        write_validation_report(self._promote(at), at, str(tmp_path))

        with pytest.raises(ValidationArtifactDegraded) as excinfo:
            write_validation_report(self._reject(at), at, str(tmp_path))

        assert "AAA PROMOTE -> REJECT" in str(excinfo.value)

    def test_a_rerun_upgrading_a_reject_to_promote_overwrites(self, tmp_path):
        at = datetime(2026, 9, 2, tzinfo=timezone.utc)
        write_validation_report(self._reject(at), at, str(tmp_path))
        path = write_validation_report(self._promote(at), at, str(tmp_path))
        with open(path) as fh:
            assert json.load(fh)["results"][0]["verdict"] == "PROMOTE"

    def test_a_rerun_repeating_the_same_reject_overwrites(self, tmp_path):
        at = datetime(2026, 9, 2, tzinfo=timezone.utc)
        write_validation_report(self._reject(at), at, str(tmp_path))
        path = write_validation_report(self._reject(at), at, str(tmp_path))
        with open(path) as fh:
            assert json.load(fh)["results"][0]["verdict"] == "REJECT"

    def test_a_rerun_downgrading_a_reject_to_error_is_refused(self, tmp_path):
        at = datetime(2026, 9, 2, tzinfo=timezone.utc)
        write_validation_report(self._reject(at), at, str(tmp_path))
        keyless = build_validation_report([], {"AAA": "no API key"},
                                          generated_at=at)
        with pytest.raises(ValidationArtifactDegraded) as excinfo:
            write_validation_report(keyless, at, str(tmp_path))
        assert "AAA REJECT -> ERROR" in str(excinfo.value)

    def test_a_promote_downgraded_to_an_unknown_verdict_is_refused(self, tmp_path):
        """An unrecognised verdict ranks with REJECT: it is not evidence of a
        promotion, so writing it over one loses the promotion."""
        at = datetime(2026, 9, 2, tzinfo=timezone.utc)
        write_validation_report(self._promote(at), at, str(tmp_path))
        odd = self._promote(at)
        odd["results"][0]["verdict"] = "MAYBE"
        with pytest.raises(ValidationArtifactDegraded) as excinfo:
            write_validation_report(odd, at, str(tmp_path))
        assert "AAA PROMOTE -> MAYBE" in str(excinfo.value)

    def test_an_unknown_recorded_verdict_still_outranks_error(self, tmp_path):
        at = datetime(2026, 9, 2, tzinfo=timezone.utc)
        odd = self._promote(at)
        odd["results"][0]["verdict"] = "MAYBE"
        write_validation_report(odd, at, str(tmp_path))
        keyless = build_validation_report([], {"AAA": "no API key"},
                                          generated_at=at)
        with pytest.raises(ValidationArtifactDegraded):
            write_validation_report(keyless, at, str(tmp_path))

    def test_every_degrading_symbol_is_named(self, tmp_path):
        at = datetime(2026, 9, 2, tzinfo=timezone.utc)
        write_validation_report(
            build_validation_report([symbol_validation("AAA"),
                                     symbol_validation("BBB")], generated_at=at),
            at, str(tmp_path))
        rerun = build_validation_report(
            [symbol_validation("AAA", dsr=0.10, oos=-0.5, pbo=0.9)],
            {"BBB": "no API key"}, generated_at=at)
        with pytest.raises(ValidationArtifactDegraded) as excinfo:
            write_validation_report(rerun, at, str(tmp_path))
        message = str(excinfo.value)
        assert "AAA PROMOTE -> REJECT" in message
        assert "BBB PROMOTE -> ERROR" in message

    def test_a_promote_to_reject_rerun_can_be_forced(self, tmp_path):
        at = datetime(2026, 9, 2, tzinfo=timezone.utc)
        write_validation_report(self._promote(at), at, str(tmp_path))
        path = write_validation_report(self._reject(at), at, str(tmp_path),
                                       allow_degrade=True)
        with open(path) as fh:
            assert json.load(fh)["results"][0]["verdict"] == "REJECT"


class TestResearchMainExitCode:
    """A refused write must not read as a clean run: whatever schedules this
    only sees the exit code."""

    def _keyless_run(self, monkeypatch):
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        monkeypatch.setattr(research, "MarketData", lambda *a, **k: _NoBars())
        monkeypatch.setattr(research, "judge_symbols", lambda *a, **k: ([], {}))

    def test_a_refused_write_exits_nonzero_and_says_why(self, monkeypatch, capsys):
        self._keyless_run(monkeypatch)

        def refuse(*args, **kwargs):
            raise ValidationArtifactDegraded("AAA PROMOTE -> ERROR")

        monkeypatch.setattr(research, "write_validation_report", refuse)

        assert research.main() == 1
        captured = capsys.readouterr()
        assert "AAA PROMOTE -> ERROR" in captured.err

    def test_a_written_report_exits_zero(self, monkeypatch, capsys, tmp_path):
        self._keyless_run(monkeypatch)
        monkeypatch.setattr(research, "write_validation_report",
                            lambda *a, **k: str(tmp_path / "validation.json"))
        assert research.main() == 0
        assert "validation.json" in capsys.readouterr().out


class _NoBars:
    """A MarketData stand-in that never yields bars, so main() runs keyless."""

    def get_bars(self, *args, **kwargs):
        raise RuntimeError("no market data")


class TestVariantCarriesStrategyIdentity:
    def test_validate_symbol_records_the_strategy_and_its_params(self):
        bars = calendar_bars(choppy_uptrend(260))
        variants = [("dip-buyer", lambda: build_strategy(mr_cfg("AAA", 1.0)))]
        result = validate_symbol(
            "AAA", AssetClass.US_EQUITY, 0.0, variants, bars, n_blocks=4, n_folds=4,
        )
        variant = result.variants[0]
        assert variant.strategy == "MeanReversion"
        assert variant.params["entry_z"] == pytest.approx(1.0)
        assert "stop_atr_mult" in variant.params
        # Private attributes are an implementation detail, not a recorded parameter.
        assert not any(k.startswith("_") for k in variant.params)


class TestHarnessCoversTheLineup:
    """research.make_cfg hardcodes DAILY (research.py:110), so a non-daily
    instrument listed in EXPERIMENTS is measured as an instrument that does not
    exist. Coverage and timeframe therefore have to move together."""

    def test_gld_is_covered_by_the_research_harness(self):
        labels = [row[3] for row in EXPERIMENTS if row[0] == "GLD"]
        assert labels == ["EMA 50/200 long", "EMA 20/100 long", "EMA 10/50 long"]

    def test_gld_is_us_equity_with_expected_slippage(self):
        gld = [row for row in EXPERIMENTS if row[0] == "GLD"]
        assert gld, "GLD must be present"
        assert all(ac == AssetClass.US_EQUITY and slip == 0.0001
                   for _sym, ac, slip, _label, _f in gld)

    def test_every_researched_symbol_trades_the_timeframe_it_is_measured_on(self):
        researched = {row[0] for row in EXPERIMENTS}
        offenders = {c.symbol: c.timeframe_minutes for c in INSTRUMENTS
                     if c.symbol in researched and c.timeframe_minutes != DAILY}
        assert offenders == {}, (
            f"measured on {DAILY}min but traded on something else: {offenders}")


_CITED_ARTIFACT = re.compile(r"docs/validation/validation-\d{4}-\d{2}-\d{2}\.json")
_VERDICT_WORDS = ("PROMOTE", "REJECT")


def quoted_variant(symbol: str, text: str) -> str | None:
    """The variant label a citation quotes: what sits between the symbol and the
    em dash, with a trailing verdict word dropped."""
    match = re.search(rf"{re.escape(symbol)} (?:selected )?(.+?) —", text)
    if match is None:
        return None
    label = match.group(1).strip()
    for verdict in _VERDICT_WORDS:
        label = label.removesuffix(f" {verdict}")
    return label


def citation_problems(symbol: str, text: str | None) -> list[str]:
    """Every way a `<artifact>: <SYMBOL> <variant> — ...` citation fails to
    resolve: a path that is not in the repo, an artifact with no row for the
    symbol, or a label that artifact never recorded measuring."""
    problems = []
    for path in _CITED_ARTIFACT.findall(text or ""):
        if not os.path.exists(path):
            problems.append(f"{symbol}: {path} is not in the repo")
            continue
        with open(path) as fh:
            entries = json.load(fh)["results"]
        entry = next((e for e in entries if e["symbol"] == symbol), None)
        if entry is None:
            problems.append(f"{symbol}: {path} carries no row for {symbol}")
            continue
        label = quoted_variant(symbol, text)
        # No variants table means the file vouches for its selection and
        # nothing else: absent is "not recorded", not "not measured".
        measured = [v["variant"] for v in entry.get("variants", [])] or [entry["variant"]]
        if label not in measured:
            problems.append(f"{symbol}: {path} records {measured}, not {label!r}")
    return problems


class TestValidationRefs:
    def test_voo_cites_the_research_that_promoted_it(self):
        voo = next(c for c in INSTRUMENTS if c.symbol == "VOO")
        assert voo.validation_ref and "VOO" in voo.validation_ref
        assert "PROMOTE" in voo.validation_ref
        assert voo.override_reason is None  # a verdict needs no override

    def test_qqq_cites_the_research_that_promoted_it(self):
        qqq = next(c for c in INSTRUMENTS if c.symbol == "QQQ")
        assert qqq.validation_ref and "QQQ" in qqq.validation_ref
        assert "PROMOTE" in qqq.validation_ref
        assert qqq.override_reason is None  # a verdict needs no override

    def test_gld_runs_the_variant_the_harness_selected(self):
        # A citation has to describe what actually runs. The 2026-09-10 run
        # measured 50/200 at DSR 0.001 / OOS -0.51 and selected 10/50.
        gld = next(c for c in INSTRUMENTS if c.symbol == "GLD")
        assert gld.params["fast_period"] == 10
        assert gld.params["slow_period"] == 50

    def test_gld_override_cites_its_reject_rather_than_a_coverage_gap(self):
        gld = next(c for c in INSTRUMENTS if c.symbol == "GLD")
        assert gld.validation_ref is None  # a REJECT is not a promotion
        assert gld.override_reason and "REJECT" in gld.override_reason

    def test_every_instrument_cites_an_artifact_that_names_its_variant(self):
        problems, cited = [], set()
        for cfg in INSTRUMENTS:
            for text in (cfg.validation_ref, cfg.override_reason):
                if text and _CITED_ARTIFACT.search(text):
                    cited.add(cfg.symbol)
                problems += citation_problems(cfg.symbol, text)
        assert problems == []
        assert cited == {c.symbol for c in INSTRUMENTS}

    def test_a_path_missing_from_the_repo_is_caught(self):
        assert citation_problems("SPY", "docs/validation/validation-1999-01-01.json: "
                                        "SPY RSI2<10 200sma — DSR 0.9")

    def test_a_label_the_artifact_never_measured_is_caught(self):
        assert citation_problems("SPY", "docs/validation/validation-2026-09-16.json: "
                                        "SPY RSI2<42 200sma — DSR 0.9")

    def test_a_measured_but_unselected_label_resolves_through_the_variants_table(self):
        assert citation_problems("SPY", "docs/validation/validation-2026-09-16.json: "
                                        "SPY EMA 50/200 long — DSR 0.677") == []

    def test_an_artifact_without_a_variants_table_vouches_only_for_its_selection(self):
        assert citation_problems("QQQ", "docs/validation/validation-2026-09-07.json: "
                                        "QQQ RSI2<10 200sma PROMOTE — DSR 0.984") == []
        assert citation_problems("QQQ", "docs/validation/validation-2026-09-07.json: "
                                        "QQQ EMA 50/200 long PROMOTE — DSR 0.98")

    def test_both_fields_are_optional_and_default_to_none(self):
        cfg = InstrumentConfig(
            symbol="ZZZ", asset_class=AssetClass.US_EQUITY,
            strategy=StrategyName.TREND_FOLLOWING, timeframe_minutes=1440,
        )
        assert cfg.validation_ref is None and cfg.override_reason is None


class _StubStrategy:
    """Duck-typed stand-in: strategy_params only reads public attributes and the
    stop multiple, so a param harvest test needs no working strategy."""
    stop_atr_mult = 3.0

    def __init__(self, **attrs):
        self.__dict__.update(attrs)


class TestParamHarvest:
    def test_a_carried_dataframe_is_not_a_recorded_parameter(self):
        # PeadDrift holds its events table as a public attribute. State the
        # strategy carries is not a parameter, and it would sink the JSON write.
        strategy = PeadDrift(events=pd.DataFrame({"symbol": ["AAA"]}), sue_min=1.5)
        params = strategy_params(strategy)
        assert "events" not in params
        assert params["sue_min"] == pytest.approx(1.5)
        assert params["stop_atr_mult"] == pytest.approx(3.0)

    def test_numpy_scalars_are_recorded_as_plain_numbers(self):
        params = strategy_params(_StubStrategy(width=np.float32(0.25), n=np.int64(7)))
        assert params["width"] == 0.25 and type(params["width"]) is float
        assert params["n"] == 7 and type(params["n"]) is int

    def test_a_full_report_over_a_dataframe_carrying_strategy_serializes(self):
        strategy = PeadDrift(events=pd.DataFrame({"symbol": ["AAA"]}))
        result = SymbolValidation(
            symbol="AAA",
            variants=[VariantResult("pead", 0.99, 1.2, "PeadDrift",
                                    strategy_params(strategy))],
            cscv=cscv(), oos_sharpe=1.2, best_label="pead", buy_and_hold=0.1,
        )
        report = build_validation_report([result])
        assert json.loads(json.dumps(report, allow_nan=False)) == report

    def test_a_non_string_keyed_dict_is_recorded_not_omitted(self):
        # The artifact presents params_omitted as what it could not record, so a
        # dict the harvest does write must not be named there.
        strategy = _StubStrategy(by_month={1: 0.5, 12: -0.25})
        params = strategy_params(strategy)
        assert params["by_month"] == {"1": 0.5, "12": -0.25}
        assert "by_month" not in omitted_params(strategy)

    def test_a_dict_whose_keys_collide_as_strings_is_omitted(self):
        # Stringifying would silently drop one of the two keys; naming the
        # attribute is honest where writing half of it is not.
        strategy = _StubStrategy(mixed={1: "a", "1": "b"})
        assert "mixed" not in strategy_params(strategy)
        assert omitted_params(strategy) == ["mixed"]

    def test_an_unrecognised_value_degrades_to_its_repr(self):
        # Second line of defence: the harvest filters, and anything that slips
        # past is written as its repr rather than costing the whole artifact.
        result = symbol_validation(params={"opaque": _StubStrategy()})
        entry = build_validation_report([result])["results"][0]
        assert "_StubStrategy" in entry["params"]["opaque"]
        assert json.dumps(entry, allow_nan=False)


class TestSelectedVariantOos:
    def test_entry_carries_the_selected_variants_own_oos_sharpe(self):
        result = symbol_validation(dsr=0.99, oos=1.2, pbo=0.1)
        entry = build_validation_report([result])["results"][0]
        assert entry["variant_oos"] == pytest.approx(result.best.oos)

    def test_a_nan_variant_oos_serializes_as_null(self):
        result = SymbolValidation(
            symbol="AAA",
            variants=[VariantResult("only", 0.99, float("nan"), "X", {})],
            cscv=cscv(), oos_sharpe=1.2, best_label="only", buy_and_hold=0.0,
        )
        assert build_validation_report([result])["results"][0]["variant_oos"] is None


def _boom():
    raise RuntimeError("judge exploded")


class TestJudgeSymbols:
    def _grouped(self, *symbols) -> dict[str, tuple]:
        return {s: (AssetClass.US_EQUITY, 0.0,
                    [("dip-buyer", lambda: build_strategy(mr_cfg(s, 1.0)))])
                for s in symbols}

    def test_one_symbols_failure_costs_only_that_symbol(self, caplog):
        grouped = self._grouped("AAA")
        grouped["BAD"] = (AssetClass.US_EQUITY, 0.0, [("boom", _boom)])
        bars = {"AAA": calendar_bars(choppy_uptrend(260)),
                "BAD": calendar_bars(choppy_uptrend(260))}

        with caplog.at_level(logging.ERROR):
            results, errors = judge_symbols(grouped, bars, n_blocks=4, n_folds=4)

        assert [r.symbol for r in results] == ["AAA"]
        assert "judge exploded" in errors["BAD"]
        assert "BAD" in caplog.text

    def test_a_failed_symbol_prints_no_partial_table_rows(self, capsys):
        grouped: dict[str, tuple] = {"BAD": (AssetClass.US_EQUITY, 0.0,
                                             [("boom", _boom)])}
        judge_symbols(grouped, {"BAD": calendar_bars(choppy_uptrend(260))},
                      n_blocks=4, n_folds=4)
        assert capsys.readouterr().out == ""

    def test_a_printing_failure_is_recorded_like_any_other(self, monkeypatch, caplog):
        # The print is inside the guard too: a formatting fault on one symbol's
        # numbers must not cost the artifact everything already computed.
        monkeypatch.setattr("bot.research.print_validation",
                            lambda result: (_ for _ in ()).throw(
                                ValueError("bad format")))
        with caplog.at_level(logging.ERROR):
            results, errors = judge_symbols(
                self._grouped("AAA"), {"AAA": calendar_bars(choppy_uptrend(260))},
                n_blocks=4, n_folds=4)
        assert results == []
        assert "bad format" in errors["AAA"]

    def test_a_symbol_without_bars_is_skipped_silently(self, caplog):
        with caplog.at_level(logging.ERROR):
            results, errors = judge_symbols(self._grouped("AAA"), {}, n_blocks=4,
                                            n_folds=4)
        assert results == [] and errors == {}
        assert caplog.text == ""

    def test_the_artifact_records_the_failure_beside_the_verdicts(self):
        result = symbol_validation("AAA")
        report = build_validation_report([result],
                                         errors={"BAD": "RuntimeError: boom"})
        assert [r["symbol"] for r in report["results"]] == ["AAA", "BAD"]
        assert report["results"][1] == {"symbol": "BAD", "verdict": "ERROR",
                                        "error": "RuntimeError: boom"}
        assert json.dumps(report, allow_nan=False)

    def test_every_row_carries_a_verdict_to_discriminate_on(self):
        report = build_validation_report([symbol_validation("AAA")],
                                         errors={"BAD": "RuntimeError: boom"})
        assert [r["verdict"] for r in report["results"]] == ["PROMOTE", "ERROR"]


class TestArtifactShape:
    def test_the_schema_is_versioned(self):
        assert build_validation_report([])["schema_version"] == 2

    def test_omitted_params_are_named_on_the_entry(self):
        # The events table was part of what was validated even though it is not
        # a recordable parameter, so the artifact says it was dropped.
        result = symbol_validation(params_omitted=["events"])
        assert build_validation_report([result])["results"][0][
            "params_omitted"] == ["events"]

    def test_a_clean_harvest_adds_no_omission_noise(self):
        assert "params_omitted" not in build_validation_report(
            [symbol_validation()])["results"][0]

    def test_validate_symbol_names_what_the_harvest_dropped(self):
        bars = calendar_bars(choppy_uptrend(60))
        events = pd.DataFrame([(pd.Timestamp("2020-01-15").date(), "bmo", 2.0)],
                              columns=["announce_date", "hour", "sue"])
        variants = [("pead", lambda: PeadDrift(events=events))]
        result = validate_symbol("AAA", AssetClass.US_EQUITY, 0.0, variants, bars,
                                 n_blocks=4, n_folds=4)
        assert result.variants[0].params_omitted == ["events"]
