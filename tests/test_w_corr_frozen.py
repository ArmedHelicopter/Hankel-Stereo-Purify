"""MSSA denoising comparison test.

Compares three modes on real audio:
  1. Original input (ground truth)
  2. Energy-only truncation (no W-correlation) — baseline
  3. Energy + frozen W-correlation — current implementation
  4. Energy + naive W-correlation — per-frame recomputation (ideal)

Outputs 10s MP3 files for listening comparison.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from src.core.process_frame import process_frame
from src.core.stages.c_svd import (
    _energy_truncated_factors,
    _reconstruct_usvh,
    _w_corr_keep_indices,
    _zero_s_except_indices,
    _SvdStepState,
    _EnergySvdStep,
)
from src.core.strategies.truncation import EnergyThresholdStrategy


# ---------------------------------------------------------------------------
# Naive energy SVD step: recompute W-correlation on every frame
# ---------------------------------------------------------------------------

class _NaiveEnergySvdStep:
    """Energy-threshold SVD step; W-correlation recomputed on every frame."""

    __slots__ = ("_strat", "_w_corr_threshold", "_window_length", "state")

    def __init__(
        self,
        strat: EnergyThresholdStrategy,
        *,
        w_corr_threshold: float | None,
        window_length: int | None,
    ) -> None:
        self._strat = strat
        self._w_corr_threshold = w_corr_threshold
        self._window_length = window_length
        self.state = _SvdStepState()

    def _filter_w_corr_naive(
        self,
        u: NDArray[np.float64],
        s: NDArray[np.float64],
        vh: NDArray[np.float64],
        corr_wl: int,
        thr: float,
    ) -> NDArray[np.float64]:
        k = int(s.shape[0])
        keep = _w_corr_keep_indices(u, s, vh, corr_wl, thr)
        if k >= 1 and (keep.size == 0 or not np.any(keep == 0)):
            keep = np.sort(
                np.unique(np.concatenate((np.array([0], dtype=np.intp), keep)))
            )
            keep = keep[keep < k]
        return _zero_s_except_indices(s, keep)

    def __call__(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        from src.core.stages.c_svd import _prepare_svd_frame

        a, corr_wl = _prepare_svd_frame(
            data, self._w_corr_threshold, self._window_length
        )
        u, s, vh = _energy_truncated_factors(a, self._strat, self.state)
        if self._w_corr_threshold is not None:
            if corr_wl is None:
                raise ValueError("window_length required with w_corr_threshold")
            s = self._filter_w_corr_naive(
                u, s, vh, corr_wl, float(self._w_corr_threshold)
            )
        return _reconstruct_usvh(u, s, vh).astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class GridResult:
    F: int
    L: int
    hop: int
    snr_energy_vs_original: float
    snr_wcorr_vs_original: float
    snr_naive_vs_original: float
    snr_wcorr_vs_energy: float
    snr_naive_vs_energy: float
    snr_naive_vs_wcorr: float
    frozen_time_s: float
    naive_time_s: float
    energy_time_s: float


def compute_snr(reference: NDArray, test: NDArray) -> float:
    noise = reference - test
    noise_power = float(np.sum(noise ** 2))
    if noise_power < 1e-30:
        return 999.0
    signal_power = float(np.sum(reference ** 2))
    return 10.0 * np.log10(signal_power / noise_power)


# ---------------------------------------------------------------------------
# OLA helpers
# ---------------------------------------------------------------------------

def sqrt_hanning_weights(F: int) -> NDArray[np.float64]:
    n = np.arange(F, dtype=np.float64)
    return np.sqrt(0.5 * (1.0 - np.cos(2.0 * np.pi * n / F)))


def ola_reconstruct(
    frames: list[NDArray[np.float64]],
    frame_starts: list[int],
    F: int,
    total_samples: int,
) -> NDArray[np.float64]:
    window = sqrt_hanning_weights(F)
    output = np.zeros((total_samples, 2), dtype=np.float64)
    weight_sum = np.zeros(total_samples, dtype=np.float64)
    for frame, start in zip(frames, frame_starts):
        end = start + F
        output[start:end] += frame * window[:, np.newaxis]
        weight_sum[start:end] += window
    mask = weight_sum > 1e-10
    output[mask] /= weight_sum[mask, np.newaxis]
    return output


def compute_frame_starts(total_samples: int, F: int, hop: int) -> list[int]:
    starts = []
    pos = 0
    while pos + F <= total_samples:
        starts.append(pos)
        pos += hop
    return starts


# ---------------------------------------------------------------------------
# Run single config
# ---------------------------------------------------------------------------

def run_single_config(
    audio: NDArray[np.float64],
    sr: int,
    F: int,
    L: int,
    hop: int,
    energy_fraction: float,
    w_corr_threshold: float,
    output_dir: Path,
    suffix: str,
) -> GridResult:
    """Run energy-only, frozen, naive on the full audio slice."""

    frame_starts = compute_frame_starts(audio.shape[0], F, hop)
    slices = [audio[s:s + F].copy() for s in frame_starts if s + F <= audio.shape[0]]
    actual_starts = [s for s in frame_starts if s + F <= audio.shape[0]]

    if not slices:
        raise ValueError("No valid frames")

    strat = EnergyThresholdStrategy(energy_fraction)
    tag = f"F{F}_L{L}_hop{hop}"

    # --- Energy only (no W-correlation) ---
    energy_step = _EnergySvdStep(strat, w_corr_threshold=None, window_length=L)
    energy_frames = []
    t0 = time.perf_counter()
    for s in slices:
        energy_frames.append(process_frame(s, window_length=L, svd_step=energy_step))
    energy_time = time.perf_counter() - t0

    # --- Frozen W-correlation ---
    frozen_step = _EnergySvdStep(strat, w_corr_threshold=w_corr_threshold, window_length=L)
    frozen_frames = []
    t0 = time.perf_counter()
    for s in slices:
        frozen_frames.append(process_frame(s, window_length=L, svd_step=frozen_step))
    frozen_time = time.perf_counter() - t0

    # --- Naive W-correlation ---
    naive_step = _NaiveEnergySvdStep(strat, w_corr_threshold=w_corr_threshold, window_length=L)
    naive_frames = []
    t0 = time.perf_counter()
    for s in slices:
        naive_frames.append(process_frame(s, window_length=L, svd_step=naive_step))
    naive_time = time.perf_counter() - t0

    # --- OLA reconstruct ---
    total_samples = actual_starts[-1] + F
    energy_out = ola_reconstruct(energy_frames, actual_starts, F, total_samples)
    frozen_out = ola_reconstruct(frozen_frames, actual_starts, F, total_samples)
    naive_out = ola_reconstruct(naive_frames, actual_starts, F, total_samples)

    # --- Original input (slice to match output length) ---
    original = audio[:total_samples]

    # --- Save files ---
    sf.write(str(output_dir / f"original_{tag}{suffix}"), original, sr)
    sf.write(str(output_dir / f"energy_only_{tag}{suffix}"), energy_out, sr)
    sf.write(str(output_dir / f"energy_wcorr_{tag}{suffix}"), frozen_out, sr)
    sf.write(str(output_dir / f"energy_naive_{tag}{suffix}"), naive_out, sr)
    sf.write(str(output_dir / f"diff_energy_vs_wcorr_{tag}{suffix}"), energy_out - frozen_out, sr)
    sf.write(str(output_dir / f"diff_wcorr_vs_naive_{tag}{suffix}"), frozen_out - naive_out, sr)

    return GridResult(
        F=F, L=L, hop=hop,
        snr_energy_vs_original=compute_snr(original, energy_out),
        snr_wcorr_vs_original=compute_snr(original, frozen_out),
        snr_naive_vs_original=compute_snr(original, naive_out),
        snr_wcorr_vs_energy=compute_snr(energy_out, frozen_out),
        snr_naive_vs_energy=compute_snr(energy_out, naive_out),
        snr_naive_vs_wcorr=compute_snr(frozen_out, naive_out),
        frozen_time_s=frozen_time,
        naive_time_s=naive_time,
        energy_time_s=energy_time,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        PROJ = Path(__file__).resolve().parent.parent
    except NameError:
        PROJ = Path(os.getcwd())

    audio_dir = PROJ / "data" / "raw" / "Brendel_Beethoven_Piano_Music_Vol9"
    output_dir = PROJ / "data" / "processed" / "w_corr_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    src_file = audio_dir / "09_Op27_No2_I_Adagio_sostenuto.mp3"
    if not src_file.exists():
        mp3s = sorted(audio_dir.glob("*.mp3"))
        if not mp3s:
            raise FileNotFoundError(f"No MP3 files in {audio_dir}")
        src_file = mp3s[0]

    print(f"Source: {src_file.name}")
    audio, sr = sf.read(str(src_file), dtype="float64")
    print(f"  sr={sr}, shape={audio.shape}, duration={audio.shape[0]/sr:.1f}s")

    # Skip initial silence, take ~4s for ~3s output (keep test fast)
    skip_samples = int(sr * 0.5)
    audio = audio[skip_samples:skip_samples + int(sr * 4.5)]
    print(f"  Using {audio.shape[0]} samples ({audio.shape[0]/sr:.1f}s)")

    energy_fraction = 0.9
    w_corr_threshold = 0.3

    # Configs: small L (should work) vs large L (may diverge)
    configs = [
        (1024, 256, 1024),
        (1024, 512, 1024),
    ]

    results: list[GridResult] = []
    suffix = src_file.suffix

    for i, (F, L, hop) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] F={F} L={L} hop={hop} (frames: {len(compute_frame_starts(audio.shape[0], F, hop))})")
        try:
            result = run_single_config(
                audio, sr, F, L, hop,
                energy_fraction, w_corr_threshold,
                output_dir, suffix,
            )
            results.append(result)
            print(f"  SNR vs original:")
            print(f"    energy_only:  {result.snr_energy_vs_original:.1f}dB")
            print(f"    energy+wcorr: {result.snr_wcorr_vs_original:.1f}dB")
            print(f"    energy+naive: {result.snr_naive_vs_original:.1f}dB")
            print(f"  SNR between outputs:")
            print(f"    wcorr vs energy: {result.snr_wcorr_vs_energy:.1f}dB")
            print(f"    naive vs energy: {result.snr_naive_vs_energy:.1f}dB")
            print(f"    naive vs wcorr:  {result.snr_naive_vs_wcorr:.1f}dB")
            print(f"  Time: energy={result.energy_time_s:.1f}s  frozen={result.frozen_time_s:.1f}s  naive={result.naive_time_s:.1f}s")
        except Exception as e:
            print(f"  FAILED: {e}")

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    print(f"\nSummary: {summary_path}")
    print(f"Output files in: {output_dir}")


if __name__ == "__main__":
    main()
