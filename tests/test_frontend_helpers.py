"""Lightweight tests for Streamlit helper paths (no browser)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repo_pythonpath_is_repository_root() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from frontend.app import _repo_pythonpath

    assert Path(_repo_pythonpath()).resolve() == REPO_ROOT.resolve()


def test_subprocess_timeout_expired_has_cmd_and_timeout() -> None:
    import subprocess

    exc = subprocess.TimeoutExpired(cmd=["python", "-c", "pass"], timeout=0.5)
    assert exc.timeout == 0.5
    assert exc.cmd == ["python", "-c", "pass"]


def test_build_full_batch_cli_cmd_fixed_rank_with_frame_hop() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from frontend.app import _build_full_batch_cli_cmd

    cmd = _build_full_batch_cli_cmd(
        inp=REPO_ROOT / "in.flac",
        outp=REPO_ROOT / "out.flac",
        window_length=256,
        max_mem_mb=512,
        mode_fixed_rank=True,
        rank_or_energy=64,
        frame_size=512,
        hop=128,
    )
    assert cmd[1:9] == [
        "-m",
        "src.cli",
        str(REPO_ROOT / "in.flac"),
        str(REPO_ROOT / "out.flac"),
        "-L",
        "256",
        "--max-memory-mb",
        "512",
    ]
    assert cmd[9:] == ["-k", "64", "--frame-size", "512", "--hop", "128"]


def test_build_full_batch_cli_cmd_energy_only() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from frontend.app import _build_full_batch_cli_cmd

    cmd = _build_full_batch_cli_cmd(
        inp=Path("/tmp/a.flac"),
        outp=Path("/tmp/b.flac"),
        window_length=128,
        max_mem_mb=256,
        mode_fixed_rank=False,
        rank_or_energy=0.95,
        frame_size=None,
        hop=None,
    )
    assert "--energy-fraction" in cmd
    i = cmd.index("--energy-fraction")
    assert cmd[i + 1] == "0.95"
    assert "--frame-size" not in cmd
    assert "--hop" not in cmd
