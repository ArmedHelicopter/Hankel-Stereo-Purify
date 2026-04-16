"""Smoke test for the CLI entrypoint (week 4 delivery)."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def test_cli_processes_short_flac(tmp_path: Path) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    rng = np.random.default_rng(2026)
    stereo = (0.01 * rng.standard_normal((800, 2))).astype(np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            str(inp),
            str(out),
            "-L",
            "32",
            "-k",
            "16",
            "--frame-size",
            "128",
            "--hop",
            "64",
            "--max-memory-mb",
            "500",
        ],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": "src"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out.is_file()
    y, sr = sf.read(out, dtype="float64", always_2d=True)
    assert sr == 48_000
    assert y.shape[0] == stereo.shape[0]


def test_cli_energy_fraction_runs(tmp_path: Path) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    rng = np.random.default_rng(2027)
    stereo = (0.01 * rng.standard_normal((800, 2))).astype(np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            str(inp),
            str(out),
            "-L",
            "32",
            "--energy-fraction",
            "0.95",
            "--frame-size",
            "128",
            "--hop",
            "64",
        ],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": "src"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out.is_file()


def test_cli_rejects_rank_with_energy_fraction(tmp_path: Path) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    stereo = np.zeros((100, 2), dtype=np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            str(inp),
            str(out),
            "-k",
            "8",
            "--energy-fraction",
            "0.9",
        ],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": "src"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0


def test_cli_exits_nonzero_on_missing_input(tmp_path: Path) -> None:
    out = tmp_path / "out.flac"
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            str(tmp_path / "missing.flac"),
            str(out),
        ],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": "src"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
