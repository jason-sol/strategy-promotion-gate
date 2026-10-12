"""The seam between Engine and its ops surface.

Every ops behaviour is already covered through the Engine delegates in
tests/test_main_helpers.py. What is only observable here is the wiring: the
delegates carry no logic, and the module constants they thread — the halt
sentinel and the fetch cushion — are read from bot.main at call time, so
rebinding one there reaches the ops surface.
"""

from unittest.mock import MagicMock

import pytest
from helpers import patch_main_paths

from bot import main as bot_main
from bot import ops
from bot.main import Engine


@pytest.fixture
def engine(tmp_path, monkeypatch):
    patch_main_paths(monkeypatch, tmp_path)
    return Engine(MagicMock(), MagicMock(), dry_run=False)


class TestDelegation:
    """Each delegate forwards and returns, and adds nothing of its own."""

    def test_engine_owns_an_ops_collaborator(self, engine):
        assert isinstance(engine.ops, ops.Ops)
        assert engine.ops.engine is engine

    def test_maybe_send_briefings_forwards(self, engine):
        engine.ops = MagicMock()
        assert engine.maybe_send_briefings() is engine.ops.maybe_send_briefings()

    def test_process_commands_forwards_the_halt_path(self, engine, tmp_path):
        engine.ops = MagicMock()
        engine.process_commands()
        engine.ops.process_commands.assert_called_once_with(str(tmp_path / "HALT"))

    def test_execute_command_forwards_the_command_and_the_halt_path(
            self, engine, tmp_path):
        engine.ops = MagicMock()
        command = object()
        engine.execute_command(command)
        engine.ops.execute_command.assert_called_once_with(
            command, str(tmp_path / "HALT"))

    def test_check_startup_windows_forwards_the_fetch_cushion(self, engine):
        engine.ops = MagicMock()
        engine.check_startup_windows()
        engine.ops.check_startup_windows.assert_called_once_with(
            bot_main.BAR_WINDOW_EXTRA)

    def test_warn_unvalidated_instruments_stays_importable_from_main(self):
        assert bot_main.warn_unvalidated_instruments is ops.warn_unvalidated_instruments


class TestPatchPointsSurvive:
    """Names bot.main no longer uses itself but callers still reach through it.
    Deleting either as an unused import breaks command handling silently."""

    def test_main_still_exposes_the_commands_module(self):
        assert bot_main.commands is ops.commands

    def test_rebinding_the_halt_path_in_main_reaches_the_ops_surface(
            self, engine, tmp_path, monkeypatch):
        monkeypatch.setattr("bot.main.HALT_FILE", str(tmp_path / "MOVED_HALT"))
        engine.execute_command(MagicMock(verb="halt", args="wiring check"))
        assert (tmp_path / "MOVED_HALT").exists()
