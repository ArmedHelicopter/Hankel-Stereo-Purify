"""Synthetic failure-mode experiment tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

import scripts.run_failure_modes as fm

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "run_failure_modes.py"


def _rms(x: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def _tiny_config() -> fm.ExperimentConfig:
    return fm.ExperimentConfig(
        sample_rate=8_000,
        duration_seconds=0.12,
        seed=1234,
        window_length=16,
        frame_size=64,
        energy_fraction=0.9,
        bypass_freq=2_000.0,
    )


def test_synthetic_generators_are_deterministic_stereo_and_non_silent() -> None:
    config = _tiny_config()
    for name in fm.DEFAULT_CASES:
        first = fm.generate_case(name, config)
        second = fm.generate_case(name, config)

        assert first.clean.shape == (config.num_samples, 2)
        assert first.noise.shape == (config.num_samples, 2)
        assert first.noisy.shape == (config.num_samples, 2)
        assert first.sample_rate == config.sample_rate
        assert np.allclose(first.clean, second.clean)
        assert np.allclose(first.noise, second.noise)
        assert np.all(np.isfinite(first.clean))
        assert np.all(np.isfinite(first.noise))
        assert np.all(np.isfinite(first.noisy))
        assert _rms(first.clean) > 0.0
        assert _rms(first.noise) > 0.0


def test_variant_metrics_identity_is_finite_and_nearly_zero_error() -> None:
    config = _tiny_config()
    case = fm.generate_case("low_snr", config)

    metrics = fm.variant_metrics(
        clean=case.clean,
        noisy=case.clean,
        output=case.clean,
        sample_rate=case.sample_rate,
        bypass_freq=config.bypass_freq,
        rank_trace=[2, 3, 3, 4],
    )

    for value in metrics.values():
        if isinstance(value, float):
            assert np.isfinite(value)
    assert metrics["clean_error_rms"] < 1e-12
    assert metrics["residual_rms"] < 1e-12
    assert metrics["residual_clean_projection_ratio"] == 0.0
    assert metrics["rank_count"] == 4
    assert metrics["rank_max_delta"] == 1.0


def test_failure_modes_script_smoke_outputs_artifacts(tmp_path: Path) -> None:
    output_root = tmp_path / "failure_modes"
    cp = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--cases",
            "transient_attack",
            "--duration",
            "0.12",
            "--window-length",
            "16",
            "--frame-size",
            "64",
            "--output-root",
            str(output_root),
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert cp.returncode == 0, cp.stderr
    root_summary = output_root / "summary.json"
    case_dir = output_root / "transient_attack"
    case_summary = case_dir / "summary.json"
    assert root_summary.is_file()
    assert (output_root / "summary.md").is_file()
    assert case_summary.is_file()
    assert (case_dir / "summary.md").is_file()

    summary = json.loads(case_summary.read_text(encoding="utf-8"))
    assert set(summary["variants"]) == set(fm.VARIANT_NAMES)
    for source_name in ("clean", "noise", "input"):
        path = case_dir / f"{source_name}.wav"
        assert path.is_file()
        data, sr = sf.read(path, dtype="float64", always_2d=True)
        assert sr == 8_000
        assert data.shape[1] == 2
        assert np.all(np.isfinite(data))

    for variant_name in fm.VARIANT_NAMES:
        paths = summary["variant_paths"][variant_name]
        for key in ("output", "residual", "clean_error"):
            path = Path(paths[key])
            assert path.is_file()
            data, sr = sf.read(path, dtype="float64", always_2d=True)
            assert sr == 8_000
            assert data.shape[1] == 2
            assert np.all(np.isfinite(data))

        metrics = summary["variants"][variant_name]
        assert metrics["rank_count"] > 0
        for value in metrics.values():
            if isinstance(value, float):
                assert np.isfinite(value)
