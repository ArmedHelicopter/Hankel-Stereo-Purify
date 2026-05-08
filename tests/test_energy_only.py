"""Pure energy-only denoising validation.

Outputs:
  - original.mp3   — source clip
  - energy.mp3     — energy-only denoised (energy_fraction=0.9, hop=F/2)
  - diff.mp3       — original − energy (removed component)

Usage:
  cd Hankel-Stereo-Purify && python -m tests.test_energy_only
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from src.core.process_frame import process_frame
from src.core.stages.c_svd import _EnergySvdStep
from src.core.strategies.truncation import EnergyThresholdStrategy


def sqrt_hanning_weights(F: int) -> NDArray[np.float64]:
    n = np.arange(F, dtype=np.float64)
    return np.sqrt(0.5 * (1.0 - np.cos(2.0 * np.pi * n / F)))


def ola_reconstruct(frames, frame_starts, F, total_samples):
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


def compute_snr(reference: NDArray, test: NDArray) -> float:
    noise = reference - test
    noise_power = float(np.sum(noise ** 2))
    if noise_power < 1e-30:
        return 999.0
    signal_power = float(np.sum(reference ** 2))
    return 10.0 * np.log10(signal_power / noise_power)


def compute_rms(x: NDArray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def main():
    try:
        PROJ = Path(__file__).resolve().parent.parent
    except NameError:
        PROJ = Path(os.getcwd())

    audio_dir = PROJ / "data" / "raw" / "Brendel_Beethoven_Piano_Music_Vol9"
    output_dir = PROJ / "data" / "processed" / "energy_only_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    src_file = audio_dir / "09_Op27_No2_I_Adagio_sostenuto.mp3"
    if not src_file.exists():
        mp3s = sorted(audio_dir.glob("*.mp3"))
        src_file = mp3s[0]

    print(f"Source: {src_file.name}")
    audio, sr = sf.read(str(src_file), dtype="float64")

    # Skip initial silence, take ~5s
    skip = int(sr * 0.5)
    duration_s = 5.0
    audio = audio[skip:skip + int(sr * duration_s)]
    print(f"  Clip: {audio.shape[0]} samples ({audio.shape[0]/sr:.1f}s), sr={sr}")

    # Parameters
    F, L, hop = 1024, 256, 512  # hop=F/2, 50% overlap with sqrt-Hann
    energy_fraction = 0.9

    print(f"\nConfig: F={F} L={L} hop={hop} energy={energy_fraction}")

    # Frame slicing
    frame_starts = []
    pos = 0
    while pos + F <= audio.shape[0]:
        frame_starts.append(pos)
        pos += hop

    slices = [audio[s:s+F].copy() for s in frame_starts]
    total_samples = frame_starts[-1] + F
    original = audio[:total_samples]

    print(f"  Frames: {len(slices)}")

    # Energy-only SVD step (no W-correlation)
    strat = EnergyThresholdStrategy(energy_fraction)
    svd_step = _EnergySvdStep(strat, w_corr_threshold=None, window_length=L)

    print("\nRunning energy-only denoising...")
    t0 = time.perf_counter()
    denoised_frames = []
    for s in slices:
        denoised_frames.append(process_frame(s, window_length=L, svd_step=svd_step))
    elapsed = time.perf_counter() - t0

    energy_out = ola_reconstruct(denoised_frames, frame_starts, F, total_samples)
    diff = original - energy_out

    # Metrics
    snr = compute_snr(original, energy_out)
    rms_orig = compute_rms(original)
    rms_energy = compute_rms(energy_out)
    rms_diff = compute_rms(diff)

    print(f"\n--- Results ---")
    print(f"  Time:        {elapsed:.2f}s ({len(slices)/elapsed:.1f} frames/s)")
    print(f"  SNR:         {snr:.1f} dB")
    print(f"  RMS original:{rms_orig:.6f}")
    print(f"  RMS energy:  {rms_energy:.6f}")
    print(f"  RMS diff:    {rms_diff:.6f}")
    print(f"  Diff/Orig:   {rms_diff/rms_orig*100:.1f}%")

    # Save outputs
    suffix = src_file.suffix
    sf.write(str(output_dir / f"original{suffix}"), original, sr)
    sf.write(str(output_dir / f"energy{suffix}"), energy_out, sr)
    sf.write(str(output_dir / f"diff{suffix}"), diff, sr)

    print(f"\nSaved to {output_dir}/")
    print(f"  original{suffix}  — source clip")
    print(f"  energy{suffix}    — energy-only denoised")
    print(f"  diff{suffix}      — removed component")


if __name__ == "__main__":
    main()
