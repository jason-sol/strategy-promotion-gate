import csv
import math
from datetime import datetime, timezone

import pytest

from bot.portfolio import (
    DailyPnlLog,
    ExecutionLog,
    SimAccount,
    TradeLog,
    assumed_slippage_bps,
    slippage_bps,
    slippage_summary,
)


class TestSimAccount:
    def test_open_long_preserves_equity(self):
        acct = SimAccount(cash=100_000)
        acct.open_position("SPY", "long", qty=100, fill_price=100.0, stop_price=98.0)
        assert acct.equity({"SPY": 100.0}) == pytest.approx(100_000)

    def test_long_gains_when_price_rises(self):
        acct = SimAccount(cash=100_000)
        acct.open_position("SPY", "long", qty=100, fill_price=100.0, stop_price=98.0)
        assert acct.equity({"SPY": 110.0}) == pytest.approx(101_000)

    def test_close_long_realizes_pnl(self):
        acct = SimAccount(cash=100_000)
        acct.open_position("SPY", "long", qty=100, fill_price=100.0, stop_price=98.0)
        pnl = acct.close_position("SPY", fill_price=105.0)
        assert pnl == pytest.approx(500)
        assert acct.cash == pytest.approx(100_500)
        assert "SPY" not in acct.positions

    def test_short_gains_when_price_falls(self):
        acct = SimAccount(cash=100_000)
        acct.open_position("SPY", "short", qty=100, fill_price=100.0, stop_price=102.0)
        assert acct.equity({"SPY": 90.0}) == pytest.approx(101_000)
        pnl = acct.close_position("SPY", fill_price=90.0)
        assert pnl == pytest.approx(1_000)
        assert acct.cash == pytest.approx(101_000)

    def test_fees_reduce_cash(self):
        acct = SimAccount(cash=100_000)
        acct.open_position("BTC/USD", "long", qty=1.0, fill_price=50_000.0,
                           stop_price=48_000.0, fee=125.0)
        pnl = acct.close_position("BTC/USD", fill_price=50_000.0, fee=125.0)
        assert acct.equity({}) == pytest.approx(99_750)
        assert pnl == pytest.approx(-250)


class TestCsvLogs:
    def test_trade_log_appends_rows_with_single_header(self, tmp_path):
        path = tmp_path / "trades.csv"
        log = TradeLog(str(path))
        log.record("2026-07-01T10:00:00", "SPY", "long", 100.0, 105.0, 500.0, 100)
        TradeLog(str(path)).record("2026-07-01T11:00:00", "QQQ", "short", 400.0, 395.0, 250.0, 50)
        rows = list(csv.reader(path.open()))
        assert rows[0] == ["timestamp", "instrument", "direction", "entry_price",
                           "exit_price", "pnl", "position_size"]
        assert len(rows) == 3
        assert rows[2][1] == "QQQ"

    def test_daily_pnl_log(self, tmp_path):
        path = tmp_path / "daily.csv"
        log = DailyPnlLog(str(path))
        log.record("2026-07-01", 100_500.0, 500.0)
        rows = list(csv.reader(path.open()))
        assert rows[0] == ["date", "equity", "pnl"]
        assert rows[1] == ["2026-07-01", "100500.0", "500.0"]

    def test_execution_log_appends_rows_with_single_header(self, tmp_path):
        path = tmp_path / "fills.csv"
        log = ExecutionLog(str(path))
        log.record("2026-07-01T10:00:00", "SPY", "buy", "entry", 500.0, 4.0,
                   500.5, 500.4, 100, 8.0)
        ExecutionLog(str(path)).record("2026-07-01T11:00:00", "SPY", "sell", "exit",
                                       505.0, 4.0, None, 504.8, 100, 3.96)
        rows = list(csv.reader(path.open()))
        assert rows[0] == ["timestamp", "symbol", "side", "event", "decision_mid",
                           "decision_spread_bps", "limit_price", "fill_price",
                           "fill_qty", "slippage_bps"]
        assert len(rows) == 3
        assert rows[1][3] == "entry"
        assert rows[2][3] == "exit"


class TestSlippageBps:
    def test_buy_above_mid_is_positive(self):
        # Bought at 100.10 vs a 100.00 decision mid: 10 bps of give-up.
        assert slippage_bps("buy", 100.0, 100.10) == pytest.approx(10.0)

    def test_sell_below_mid_is_positive(self):
        # Sold at 99.90 vs a 100.00 decision mid: 10 bps of give-up.
        assert slippage_bps("sell", 100.0, 99.90) == pytest.approx(10.0)

    def test_buy_below_mid_is_negative_savings(self):
        # A maker buy filled below mid captured 5 bps rather than paid it.
        assert slippage_bps("buy", 100.0, 99.95) == pytest.approx(-5.0)

    def test_long_and_buy_treated_identically(self):
        assert slippage_bps("long", 100.0, 100.10) == pytest.approx(10.0)
        assert slippage_bps("short", 100.0, 99.90) == pytest.approx(10.0)

    def test_nan_mid_propagates(self):
        assert math.isnan(slippage_bps("buy", float("nan"), 100.0))

    def test_nonpositive_mid_is_nan(self):
        assert math.isnan(slippage_bps("buy", 0.0, 100.0))


def _fills(tmp_path, rows, name="fills.csv"):
    """Write a fills.csv the way ExecutionLog would, one dict per row."""
    path = tmp_path / name
    log = ExecutionLog(str(path))
    for row in rows:
        log.record(*[row.get(col, "") for col in ExecutionLog.header])
    return str(path)


def _row(symbol="SPY", event="entry", side="buy", mid=100.0, spread=2.0,
         fill=100.01, slip=1.0, qty=10):
    return {"timestamp": "2026-07-01T10:00:00+00:00", "symbol": symbol, "side": side,
            "event": event, "decision_mid": mid, "decision_spread_bps": spread,
            "limit_price": fill, "fill_price": fill, "fill_qty": qty,
            "slippage_bps": slip}


class TestSlippageSummary:
    def test_missing_file_yields_no_records(self, tmp_path):
        assert slippage_summary(str(tmp_path / "absent.csv")) == []

    def test_header_only_file_yields_no_records(self, tmp_path):
        assert slippage_summary(_fills(tmp_path, [])) == []

    def test_entries_and_exits_are_separate_records(self, tmp_path):
        path = _fills(tmp_path, [_row(event="entry", slip=2.0),
                                 _row(event="exit", side="sell", slip=4.0)])
        stats = slippage_summary(path)
        assert [(s.symbol, s.kind, s.count) for s in stats] == [
            ("SPY", "entry", 1), ("SPY", "exit", 1)]

    def test_stop_fills_are_their_own_kind_not_folded_into_exits(self, tmp_path):
        """A broker stop fires without a decision quote, so main.py books it with
        a NaN mid. Folding it into `exit` would hide the costliest exits behind
        the measurable ones."""
        path = _fills(tmp_path, [_row(event="exit", side="sell", slip=2.0),
                                 _row(event="stop_fill", side="sell",
                                      mid="nan", slip="nan")])
        stats = slippage_summary(path)
        assert [(s.kind, s.count, s.dropped) for s in stats] == [
            ("exit", 1, 0), ("stop", 0, 1)]

    def test_a_kind_with_nothing_measurable_is_still_reported(self, tmp_path):
        """Silence and 'seven stops we could not measure' must not look alike."""
        path = _fills(tmp_path, [_row(symbol="VOO", event="stop_fill", side="sell",
                                      mid="nan", slip="nan")] * 7)
        stat = slippage_summary(path)[0]
        assert (stat.symbol, stat.kind, stat.count, stat.dropped) == ("VOO", "stop", 0, 7)
        assert math.isnan(stat.median_bps)
        assert math.isnan(stat.worst_bps)

    def test_dropped_rows_are_counted_per_symbol_and_kind(self, tmp_path):
        path = _fills(tmp_path, [_row(symbol="SPY", event="entry", slip=2.0),
                                 _row(symbol="SPY", event="entry", mid="nan", slip="nan"),
                                 _row(symbol="QQQ", event="entry", mid="nan", slip="nan")])
        assert [(s.symbol, s.count, s.dropped) for s in slippage_summary(path)] == [
            ("QQQ", 0, 1), ("SPY", 1, 1)]

    def test_crypto_maker_fill_counts_as_an_entry(self, tmp_path):
        path = _fills(tmp_path, [_row(symbol="BTC/USD", event="crypto_maker_filled",
                                      slip=-3.0)])
        assert [(s.symbol, s.kind) for s in slippage_summary(path)] == [
            ("BTC/USD", "entry")]

    def test_placement_and_cancellation_rows_are_not_fills(self, tmp_path):
        path = _fills(tmp_path, [_row(symbol="BTC/USD", event="crypto_maker_placed",
                                      fill="", slip=""),
                                 _row(symbol="BTC/USD", event="crypto_maker_cancelled",
                                      fill="", slip="")])
        assert slippage_summary(path) == []

    def test_mean_and_median_over_an_odd_sample(self, tmp_path):
        path = _fills(tmp_path, [_row(slip=1.0), _row(slip=2.0), _row(slip=9.0)])
        stat = slippage_summary(path)[0]
        assert stat.count == 3
        assert stat.median_bps == pytest.approx(2.0)
        assert stat.mean_bps == pytest.approx(4.0)

    def test_worst_fill_is_the_most_paid_not_the_largest_magnitude(self, tmp_path):
        # -20 is a 20bp saving, not the worst outcome; +3 is the worst give-up.
        path = _fills(tmp_path, [_row(slip=-20.0), _row(slip=3.0)])
        assert slippage_summary(path)[0].worst_bps == pytest.approx(3.0)

    def test_median_spread_at_decision_is_reported(self, tmp_path):
        path = _fills(tmp_path, [_row(spread=2.0), _row(spread=4.0), _row(spread=12.0)])
        assert slippage_summary(path)[0].median_spread_bps == pytest.approx(4.0)

    def test_rows_without_a_decision_mid_are_skipped(self, tmp_path):
        # Stop fills book with a NaN mid: there is no decision quote to measure against.
        path = _fills(tmp_path, [_row(slip=2.0),
                                 _row(mid="nan", slip="nan"),
                                 _row(mid="", slip="")])
        stat = slippage_summary(path)[0]
        assert (stat.count, stat.dropped) == (1, 2)

    def test_rows_without_a_fill_price_are_skipped(self, tmp_path):
        path = _fills(tmp_path, [_row(slip=2.0), _row(fill="", slip="")])
        stat = slippage_summary(path)[0]
        assert (stat.count, stat.dropped) == (1, 1)

    def test_a_nonpositive_mid_is_skipped(self, tmp_path):
        path = _fills(tmp_path, [_row(slip=2.0), _row(mid=0.0, slip=0.0)])
        stat = slippage_summary(path)[0]
        assert (stat.count, stat.dropped) == (1, 1)

    def test_a_malformed_row_does_not_lose_its_siblings(self, tmp_path):
        path = _fills(tmp_path, [_row(slip=2.0), _row(mid="not-a-number", slip="oops"),
                                 _row(slip=6.0)])
        stat = slippage_summary(path)[0]
        assert (stat.count, stat.dropped) == (2, 1)
        assert stat.median_bps == pytest.approx(4.0)

    def test_a_row_with_an_unparseable_spread_still_counts_its_slippage(self, tmp_path):
        path = _fills(tmp_path, [_row(spread="", slip=2.0), _row(spread=6.0, slip=4.0)])
        stat = slippage_summary(path)[0]
        assert stat.count == 2
        assert stat.median_spread_bps == pytest.approx(6.0)

    def test_records_are_ordered_by_symbol_then_entries_before_exits(self, tmp_path):
        path = _fills(tmp_path, [_row(symbol="QQQ", event="exit", side="sell"),
                                 _row(symbol="QQQ", event="entry"),
                                 _row(symbol="GLD", event="entry")])
        assert [(s.symbol, s.kind) for s in slippage_summary(path)] == [
            ("GLD", "entry"), ("QQQ", "entry"), ("QQQ", "exit")]

    def test_a_row_without_a_symbol_is_not_grouped_under_a_blank_name(self, tmp_path):
        path = _fills(tmp_path, [_row(symbol="", slip=2.0), _row(symbol="SPY", slip=4.0)])
        assert [s.symbol for s in slippage_summary(path)] == ["SPY"]

    def test_a_corrupt_log_yields_no_records(self, tmp_path):
        """A half-written or binary log must not take the briefing down with it."""
        path = tmp_path / "fills.csv"
        path.write_bytes(b"timestamp,symbol\n\xff\xfe\x00garbage\n")
        assert slippage_summary(str(path)) == []

    def test_an_unreadable_file_yields_no_records(self, tmp_path):
        path = tmp_path / "dir-not-a-file.csv"
        path.mkdir()
        assert slippage_summary(str(path)) == []


class TestSlippageSummaryWindow:
    CUTOFF = datetime(2026, 7, 15, tzinfo=timezone.utc)

    def _at(self, when, **kwargs):
        row = _row(**kwargs)
        row["timestamp"] = when
        return row

    def test_fills_before_the_cutoff_are_excluded(self, tmp_path):
        path = _fills(tmp_path, [self._at("2026-07-01T10:00:00+00:00", slip=2.0),
                                 self._at("2026-07-20T10:00:00+00:00", slip=8.0)])
        stat = slippage_summary(path, since=self.CUTOFF)[0]
        assert (stat.count, stat.median_bps) == (1, 8.0)

    def test_a_fill_exactly_on_the_cutoff_is_included(self, tmp_path):
        path = _fills(tmp_path, [self._at("2026-07-15T00:00:00+00:00", slip=8.0)])
        assert slippage_summary(path, since=self.CUTOFF)[0].count == 1

    def test_no_cutoff_reports_since_inception(self, tmp_path):
        path = _fills(tmp_path, [self._at("2020-01-01T10:00:00+00:00", slip=2.0),
                                 self._at("2026-07-20T10:00:00+00:00", slip=8.0)])
        assert slippage_summary(path)[0].count == 2

    def test_an_out_of_window_row_is_not_counted_as_dropped(self, tmp_path):
        """Out of window is not the same failure as unmeasurable."""
        path = _fills(tmp_path, [self._at("2026-07-01T10:00:00+00:00",
                                          mid="nan", slip="nan"),
                                 self._at("2026-07-20T10:00:00+00:00", slip=8.0)])
        stat = slippage_summary(path, since=self.CUTOFF)[0]
        assert (stat.count, stat.dropped) == (1, 0)

    def test_a_naive_timestamp_is_read_as_utc(self, tmp_path):
        path = _fills(tmp_path, [self._at("2026-07-20T10:00:00", slip=8.0)])
        assert slippage_summary(path, since=self.CUTOFF)[0].count == 1

    def test_an_unparseable_timestamp_cannot_be_placed_in_the_window(self, tmp_path):
        path = _fills(tmp_path, [self._at("not-a-timestamp", slip=8.0)])
        assert slippage_summary(path, since=self.CUTOFF) == []

    def test_an_unparseable_timestamp_still_counts_without_a_window(self, tmp_path):
        path = _fills(tmp_path, [self._at("not-a-timestamp", slip=8.0)])
        assert slippage_summary(path)[0].count == 1


class TestAssumedSlippageBps:
    def test_a_configured_fraction_becomes_basis_points(self):
        # SPY is configured at slippage_pct=0.0001 — 0.01% of price, i.e. 1 bp.
        assert assumed_slippage_bps("SPY") == pytest.approx(1.0)

    def test_a_wider_instrument_reads_higher(self):
        # USO at 0.0005 is 0.05%, i.e. 5 bps.
        assert assumed_slippage_bps("USO") == pytest.approx(5.0)

    def test_a_symbol_outside_the_lineup_has_no_assumption(self):
        assert assumed_slippage_bps("TLT") is None
