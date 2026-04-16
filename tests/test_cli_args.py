"""CLI argument validation (audit: positive ints and energy range)."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def _run_cli(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "src.cli", *args],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": "src"},
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_argparse_rejects_non_positive_window_length(tmp_path: Path) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    sf.write(inp, np.zeros((50, 2)), 48_000, format="FLAC", subtype="PCM_24")
    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_cli(
        repo_root,
        [str(inp), str(out), "-L", "0", "-k", "8"],
    )
    assert proc.returncode == 2


def test_cli_argparse_rejects_non_positive_rank(tmp_path: Path) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    sf.write(inp, np.zeros((50, 2)), 48_000, format="FLAC", subtype="PCM_24")
    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_cli(repo_root, [str(inp), str(out), "-k", "-1"])
    assert proc.returncode == 2


def test_cli_rejects_energy_fraction_out_of_range(tmp_path: Path) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    sf.write(inp, np.zeros((50, 2)), 48_000, format="FLAC", subtype="PCM_24")
    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_cli(
        repo_root,
        [str(inp), str(out), "--energy-fraction", "0"],
    )
    assert proc.returncode == 2
    assert "energy-fraction" in proc.stderr.lower()

    proc2 = _run_cli(
        repo_root,
        [str(inp), str(out), "--energy-fraction", "1.5"],
    )
    assert proc2.returncode == 2


def test_cli_rejects_w_corr_threshold_out_of_range(tmp_path: Path) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    sf.write(inp, np.zeros((50, 2)), 48_000, format="FLAC", subtype="PCM_24")
    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_cli(
        repo_root,
        [str(inp), str(out), "-k", "8", "--w-corr-threshold", "-0.1"],
    )
    assert proc.returncode == 2
    proc2 = _run_cli(
        repo_root,
        [str(inp), str(out), "-k", "8", "--w-corr-threshold", "2"],
    )
    assert proc2.returncode == 2


def test_cli_argparse_rejects_non_positive_frame_size(tmp_path: Path) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    sf.write(inp, np.zeros((50, 2)), 48_000, format="FLAC", subtype="PCM_24")
    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_cli(
        repo_root,
        [str(inp), str(out), "-k", "8", "--frame-size", "0"],
    )
    assert proc.returncode == 2


def test_cli_rejects_long_input_via_hsp_max_samples_env_only(tmp_path: Path) -> None:
    """Env-only ``HSP_MAX_SAMPLES`` (no ``--max-samples``); expect exit code 2."""
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    sf.write(
        inp,
        np.zeros((200, 2), dtype=np.float64),
        48_000,
        format="FLAC",
        subtype="PCM_24",
    )
    repo_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": "src", "HSP_MAX_SAMPLES": "50"}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            str(inp),
            str(out),
            "-L",
            "16",
            "-k",
            "8",
            "--frame-size",
            "64",
            "--hop",
            "32",
        ],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    combined = (proc.stderr + proc.stdout).lower()
    assert "limit" in combined or "50" in combined
