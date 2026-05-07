"""W-correlation frozen vs naive comparison test.

Grid search over (F, L, hop) parameter combinations.
For each combination:
  1. Read 10 frames from a source audio file
  2. Run frozen W-correlation (current implementation)
  3. Run naive W-correlation (recompute every frame)
  4. Save frozen/naive/diff output files for listening comparison
  5. Report energy ratio and SNR metrics
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
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
        """Recompute W-correlation indices on every call (no freezing)."""
        k = int(s.shape[0])
        keep = _w_corr_keep_indices(u, s, vh, corr_wl, thr)
        # Ensure component 0 is always kept
        if k >= 1 and (keep.size == 0 or not np.any(keep == 0)):
            keep = np.sort(
                np.unique(
                    np.concatenate((np.array([0], dtype=np.intp), keep))
                )
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
                raise ValueError(
                    "window_length must be a positive int when "
                    "w_corr_threshold is set.",
                )
            s = self._filter_w_corr_naive(
                u, s, vh, corr_wl, float(self._w_corr_threshold)
            )
        reconstructed = _reconstruct_usvh(u, s, vh)
        return reconstructed.astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class FrameMetrics:
    frame_idx: int
    frozen_energy: float
    naive_energy: float
    diff_energy: float
    snr_db: float


@dataclass
class GridResult:
    F: int
    L: int
    hop: int
    energy_ratio_db: float
    avg_snr_db: float
    frozen_time_s: float
    naive_time_s: float
    output_frozen: str
    output_naive: str
    output_diff: str


def compute_snr(reference: NDArray, test: NDArray) -> float:
    """Signal-to-noise ratio in dB."""
    noise = reference - test
    noise_power = float(np.sum(noise ** 2))
    if noise_power < 1e-30:
        return 999.0
    signal_power = float(np.sum(reference ** 2))
    return 10.0 * np.log10(signal_power / noise_power)


def energy_ratio_db(signal_a: NDArray, signal_b: NDArray) -> float:
    """Energy of (a - b) / energy of a in dB."""
    diff = signal_a - signal_b
    e_diff = float(np.sum(diff ** 2))
    e_ref = float(np.sum(signal_a ** 2))
    if e_ref < 1e-30:
        return -999.0
    return 10.0 * np.log10(e_diff / e_ref)


# ---------------------------------------------------------------------------
# Audio slicing
# ---------------------------------------------------------------------------

def slice_audio_frames(
    audio: NDArray[np.float64],
    frame_starts: list[int],
    F: int,
    num_frames: int,
) -> list[NDArray[np.float64]]:
    """Extract num_frames stereo slices from audio at given frame_starts."""
    slices = []
    for start in frame_starts[:num_frames]:
        end = start + F
        if end <= audio.shape[0]:
            slices.append(audio[start:end].copy())
    return slices


def compute_frame_starts(
    total_samples: int,
    F: int,
    hop: int,
) -> list[int]:
    """Compute valid frame start positions (OLA-style)."""
    starts = []
    pos = 0
    while pos + F <= total_samples:
        starts.append(pos)
        pos += hop
    return starts


# ---------------------------------------------------------------------------
# OLA reconstruction helpers
# ---------------------------------------------------------------------------

def sqrt_hanning_weights(F: int) -> NDArray[np.float64]:
    """Square-root Hanning window weights (matches project convention)."""
    n = np.arange(F, dtype=np.float64)
    return np.sqrt(0.5 * (1.0 - np.cos(2.0 * np.pi * n / F)))


def ola_reconstruct(
    frames: list[NDArray[np.float64]],
    frame_starts: list[int],
    F: int,
    total_samples: int,
) -> NDArray[np.float64]:
    """Overlap-add reconstruction with sqrt-Hanning window."""
    window = sqrt_hanning_weights(F)
    output = np.zeros((total_samples, 2), dtype=np.float64)
    weight_sum = np.zeros(total_samples, dtype=np.float64)
    for frame, start in zip(frames, frame_starts):
        end = start + F
        output[start:end] += frame * window[:, np.newaxis]
        weight_sum[start:end] += window
    # Normalize
    mask = weight_sum > 1e-10
    output[mask] /= weight_sum[mask, np.newaxis]
    return output


# ---------------------------------------------------------------------------
# Run single (F, L, hop) combination
# ---------------------------------------------------------------------------

def run_single_config(
    audio: NDArray[np.float64],
    sr: int,
    F: int,
    L: int,
    hop: int,
    num_frames: int,
    energy_fraction: float,
    w_corr_threshold: float,
    output_dir: Path,
    suffix: str,
) -> GridResult:
    """Run frozen vs naive comparison for one parameter combination."""

    frame_starts = compute_frame_starts(audio.shape[0], F, hop)
    if len(frame_starts) < num_frames:
        raise ValueError(
            f"Not enough frames: need {num_frames}, got {len(frame_starts)} "
            f"(F={F}, hop={hop}, total_samples={audio.shape[0]})"
        )

    slices = slice_audio_frames(audio, frame_starts, F, num_frames)
    actual_starts = frame_starts[:num_frames]

    strat = EnergyThresholdStrategy(energy_fraction)

    # --- Frozen ---
    frozen_step = _EnergySvdStep(
        strat,
        w_corr_threshold=w_corr_threshold,
        window_length=L,
    )
    frozen_frames = []
    t0 = time.perf_counter()
    for s in slices:
        frozen_frames.append(process_frame(s, window_length=L, svd_step=frozen_step))
    frozen_time = time.perf_counter() - t0

    # --- Naive ---
    naive_step = _NaiveEnergySvdStep(
        strat,
        w_corr_threshold=w_corr_threshold,
        window_length=L,
    )
    naive_frames = []
    t0 = time.perf_counter()
    for s in slices:
        naive_frames.append(process_frame(s, window_length=L, svd_step=naive_step))
    naive_time = time.perf_counter() - t0

    # --- OLA reconstruct ---
    total_samples = actual_starts[-1] + F
    frozen_out = ola_reconstruct(frozen_frames, actual_starts, F, total_samples)
    naive_out = ola_reconstruct(naive_frames, actual_starts, F, total_samples)
    diff_out = frozen_out - naive_out

    # --- Metrics ---
    ratio = energy_ratio_db(frozen_out, naive_out)
    snr = compute_snr(frozen_out, naive_out)

    # --- Save files ---
    tag = f"F{F}_L{L}_hop{hop}"
    path_frozen = output_dir / f"frozen_{tag}{suffix}"
    path_naive = output_dir / f"naive_{tag}{suffix}"
    path_diff = output_dir / f"diff_{tag}{suffix}"

    sf.write(str(path_frozen), frozen_out, sr)
    sf.write(str(path_naive), naive_out, sr)
    sf.write(str(path_diff), diff_out, sr)

    return GridResult(
        F=F,
        L=L,
        hop=hop,
        energy_ratio_db=ratio,
        avg_snr_db=snr,
        frozen_time_s=frozen_time,
        naive_time_s=naive_time,
        output_frozen=str(path_frozen),
        output_naive=str(path_naive),
        output_diff=str(path_diff),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run grid search over parameter combinations."""

    try:
        PROJ = Path(__file__).resolve().parent.parent
    except NameError:
        PROJ = Path(os.getcwd())
    audio_dir = PROJ / "data" / "raw" / "Brendel_Beethoven_Piano_Music_Vol9"
    output_dir = PROJ / "data" / "processed" / "w_corr_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pick a source file (Adagio sostenuto - slow, good for testing)
    src_file = audio_dir / "09_Op27_No2_I_Adagio_sostenuto.mp3"
    if not src_file.exists():
        # Fallback: first mp3 found
        mp3s = sorted(audio_dir.glob("*.mp3"))
        if not mp3s:
            raise FileNotFoundError(f"No MP3 files in {audio_dir}")
        src_file = mp3s[0]

    print(f"Source: {src_file.name}")

    # Read audio (soundfile handles mp3 via libsndfile)
    audio, sr = sf.read(str(src_file), dtype="float64")
    print(f"  sr={sr}, shape={audio.shape}, duration={audio.shape[0]/sr:.1f}s")

    # Use a short slice (skip initial silence, take next 10 seconds)
    skip_samples = int(sr * 0.5)  # Skip first 0.5s of silence
    start_sample = min(skip_samples, audio.shape[0])
    end_sample = min(start_sample + sr * 10, audio.shape[0])
    audio = audio[start_sample:end_sample]
    print(f"  Using samples {start_sample} to {end_sample} ({audio.shape[0]/sr:.1f}s)")

    num_frames = 10
    energy_fraction = 0.9
    w_corr_threshold = 0.3

    # Parameter grid
    grid = []
    for F in [512, 1024, 2048]:
        for L_ratio in [4, 3, 2]:  # L = F // ratio
            L = F // L_ratio
            if L < 16:
                continue
            for hop_ratio in [4, 2]:  # hop = F // ratio
                hop = F // hop_ratio
                if hop < 1:
                    continue
                grid.append((F, L, hop))

    print(f"\nGrid: {len(grid)} combinations, {num_frames} frames each\n")

    results: list[GridResult] = []
    suffix = src_file.suffix  # .mp3

    for i, (F, L, hop) in enumerate(grid):
        tag = f"F{F}_L{L}_hop{hop}"
        try:
            result = run_single_config(
                audio, sr, F, L, hop, num_frames,
                energy_fraction, w_corr_threshold,
                output_dir, suffix,
            )
            results.append(result)
            print(
                f"[{i+1}/{len(grid)}] {tag}: "
                f"ratio={result.energy_ratio_db:.1f}dB  "
                f"snr={result.avg_snr_db:.1f}dB  "
                f"frozen={result.frozen_time_s:.3f}s  "
                f"naive={result.naive_time_s:.3f}s"
            )
        except Exception as e:
            print(f"[{i+1}/{len(grid)}] {tag}: FAILED - {e}")

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    print(f"\nSummary: {summary_path}")

    # Print top results by SNR
    if results:
        results.sort(key=lambda r: r.avg_snr_db, reverse=True)
        print("\nTop 3 by SNR:")
        for r in results[:3]:
            print(
                f"  F={r.F} L={r.L} hop={r.hop}: "
                f"SNR={r.avg_snr_db:.1f}dB  ratio={r.energy_ratio_db:.1f}dB"
            )

        # Print top results by lowest error (highest ratio_db, meaning most negative)
        results.sort(key=lambda r: r.energy_ratio_db)
        print("\nTop 3 by lowest error (most negative ratio):")
        for r in results[:3]:
            print(
                f"  F={r.F} L={r.L} hop={r.hop}: "
                f"ratio={r.energy_ratio_db:.1f}dB  SNR={r.avg_snr_db:.1f}dB"
            )


if __name__ == "__main__":
    main()
