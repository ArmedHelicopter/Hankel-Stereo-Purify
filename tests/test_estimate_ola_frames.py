"""Regression tests for scripts/estimate_ola_frames.py (subprocess)."""

import subprocess
import sys
from pathlib import Path

from src.facade.ola import list_frame_starts

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "estimate_ola_frames.py"


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def test_estimate_ola_frames_count_matches_list_frame_starts() -> None:
    n, f, h = 100, 64, 32
    expected = len(list_frame_starts(n, f, h))
    cp = _run_script(str(n), str(f), str(h))
    assert cp.returncode == 0, cp.stderr
    first = cp.stdout.strip().splitlines()[0]
    assert int(first) == expected


def test_estimate_ola_frames_window_length_prints_min_l_2k() -> None:
    n, f, h = 200, 64, 32
    cp = _run_script(str(n), str(f), str(h), "-L", "16")
    assert cp.returncode == 0, cp.stderr
    lines = [ln.strip() for ln in cp.stdout.strip().splitlines()]
    assert lines[0] == str(len(list_frame_starts(n, f, h)))
    assert any("min(L,2K)=16" in ln for ln in lines)


def test_estimate_ola_frames_window_length_too_large_warns_stderr() -> None:
    cp = _run_script("100", "64", "32", "-L", "80")
    assert cp.returncode == 0, cp.stderr
    assert "too large" in cp.stderr
