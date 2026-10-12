"""Kill-switch semantics: the sentinel's existence is authoritative, the write is
atomic, and an unreadable sentinel halts rather than trades."""
import json
import os
import stat

import pytest

from bot.halt import clear_halt, halt_tripped, read_halt, trip_halt


class TestTripAndRead:
    def test_untripped_when_no_sentinel(self, tmp_path):
        assert halt_tripped(str(tmp_path / "HALT")) is False

    def test_trip_then_tripped(self, tmp_path):
        path = str(tmp_path / "HALT")
        trip_halt("drawdown 10.4% from peak", path)
        assert halt_tripped(path) is True

    def test_reason_and_timestamp_round_trip(self, tmp_path):
        path = str(tmp_path / "HALT")
        trip_halt("drawdown 10.4% from peak", path)
        payload = read_halt(path)
        assert payload["reason"] == "drawdown 10.4% from peak"
        assert payload["tripped_at"].endswith("+00:00")

    def test_read_returns_none_when_untripped(self, tmp_path):
        assert read_halt(str(tmp_path / "HALT")) is None

    def test_retrip_overwrites_with_the_latest_reason(self, tmp_path):
        path = str(tmp_path / "HALT")
        trip_halt("first", path)
        trip_halt("second", path)
        assert read_halt(path)["reason"] == "second"

    def test_parent_directory_is_created(self, tmp_path):
        path = str(tmp_path / "nested" / "dir" / "HALT")
        trip_halt("deep", path)
        assert halt_tripped(path) is True


class TestExistenceIsAuthoritative:
    def test_empty_sentinel_still_halts(self, tmp_path):
        path = tmp_path / "HALT"
        path.write_text("")
        assert halt_tripped(str(path)) is True

    def test_malformed_sentinel_still_halts(self, tmp_path):
        path = tmp_path / "HALT"
        path.write_text("not json {{{")
        assert halt_tripped(str(path)) is True

    def test_a_legacy_plaintext_sentinel_keeps_its_reason(self, tmp_path):
        # The pre-JSON format was a bare line. A sentinel written by the old
        # build is still on the VPS; reading it as "reason unavailable" would
        # strip the one thing the halt alert exists to tell a human.
        path = tmp_path / "HALT"
        path.write_text("drawdown halt at 2026-08-31T00:00:00+00:00 equity=85000\n")
        assert read_halt(str(path))["reason"].startswith("drawdown halt at")

    def test_an_empty_sentinel_reads_as_empty_attribution(self, tmp_path):
        path = tmp_path / "HALT"
        path.write_text("")
        assert read_halt(str(path)) == {}

    def test_legacy_plaintext_sentinel_still_halts(self, tmp_path):
        # The pre-existing format was a bare line, not JSON. An upgrade must not
        # read an old sentinel as "not halted".
        path = tmp_path / "HALT"
        path.write_text("drawdown halt at 2026-08-31T00:00:00+00:00 equity=85000\n")
        assert halt_tripped(str(path)) is True

    def test_non_dict_json_falls_back_to_its_raw_text(self, tmp_path):
        # One rule for anything that is not our object shape: show the human
        # whatever the file actually says.
        path = tmp_path / "HALT"
        path.write_text(json.dumps(["halted"]))
        assert read_halt(str(path))["reason"] == '["halted"]' 


class TestFailClosed:
    @pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses permission bits")
    def test_unreadable_sentinel_directory_halts(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "HALT").write_text("{}")
        vault.chmod(0o000)
        try:
            assert halt_tripped(str(vault / "HALT")) is True
        finally:
            vault.chmod(stat.S_IRWXU)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses permission bits")
    def test_unreadable_directory_halts_even_with_no_sentinel(self, tmp_path):
        # Cannot prove absence through a directory we cannot stat, so refuse to
        # trade rather than assume the switch is clear.
        vault = tmp_path / "vault"
        vault.mkdir()
        vault.chmod(0o000)
        try:
            assert halt_tripped(str(vault / "HALT")) is True
        finally:
            vault.chmod(stat.S_IRWXU)


class TestAtomicWrite:
    def test_no_temp_file_left_behind(self, tmp_path):
        trip_halt("done", str(tmp_path / "HALT"))
        assert [p.name for p in tmp_path.iterdir()] == ["HALT"]

    def test_a_failed_write_leaves_the_prior_sentinel_intact(self, tmp_path, monkeypatch):
        path = str(tmp_path / "HALT")
        trip_halt("first", path)
        monkeypatch.setattr("bot.halt.os.replace",
                            lambda *a: (_ for _ in ()).throw(OSError("disk full")))
        with pytest.raises(OSError):
            trip_halt("second", path)
        assert halt_tripped(path) is True
        assert read_halt(path)["reason"] == "first"


class TestClear:
    def test_clear_removes_the_sentinel(self, tmp_path):
        path = str(tmp_path / "HALT")
        trip_halt("temporary", path)
        assert clear_halt(path) is True
        assert halt_tripped(path) is False

    def test_clear_is_false_when_nothing_to_clear(self, tmp_path):
        assert clear_halt(str(tmp_path / "HALT")) is False
