"""Smoke tests for Phase0 CLI."""

from __future__ import annotations

import runpy
import subprocess
import sys

import pytest


def test_cli_help_exits_zero() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "src.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tutorial" in proc.stdout


def test_main_returns_zero() -> None:
    from src.cli import main

    assert main([]) == 0


def test_run_module_as_main_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["src.cli"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("src.cli", run_name="__main__")
    assert exc.value.code == 0
