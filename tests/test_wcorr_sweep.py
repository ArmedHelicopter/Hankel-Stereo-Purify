"""W-correlation threshold sweep test.

On the validated baseline (energy-only, hop=F/2), sweep W-correlation thresholds
to find if any setting improves or doesn't degrade quality.
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
from src.core.stages.c_svd import _EnergySvdStep
from src.core.strategies.truncation import EnergyThresholdStrategy


@dataclass
class SweepResult:
    threshold: float
    snr_vs_original: float
    snr_vs_baseline: float
    time_s: float


def compute_snr(reference: NDArray, test: NDArray) -> float:
    noise = reference - test
    noise_power = float(np.sum(noise ** 2))
    if noise_power < 1e-30:
        return 999.0
    signal_power = float(np.sum(reference ** 2))
    return 10.0 * np.log10(signal_power / noise_power)


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


def run_config(audio, sr, F, L, hop, energy_fraction, w_corr_threshold, output_dir, suffix, tag):
    frame_starts = []
    pos = 0
    while pos + F <= audio.shape[0]:
        frame_starts.append(pos)
        pos += hop

    slices = [audio[s:s+F].copy() for s in frame_starts if s + F <= audio.shape[0]]
    actual_starts = [s for s in frame_starts if s + F <= audio.shape[0]]
    total_samples = actual_starts[-1] + F
    original = audio[:total_samples]

    strat = EnergyThresholdStrategy(energy_fraction)

    # Baseline: energy only
    baseline_step = _EnergySvdStep(strat, w_corr_threshold=None, window_length=L)
    baseline_frames = []
    for s in slices:
        baseline_frames.append(process_frame(s, window_length=L, svd_step=baseline_step))
    baseline_out = ola_reconstruct(baseline_frames, actual_starts, F, total_samples)

    # W-correlation
    wcorr_step = _EnergySvdStep(strat, w_corr_threshold=w_corr_threshold, window_length=L)
    wcorr_frames = []
    t0 = time.perf_counter()
    for s in slices:
        wcorr_frames.append(process_frame(s, window_length=L, svd_step=wcorr_step))
    elapsed = time.perf_counter() - t0
    wcorr_out = ola_reconstruct(wcorr_frames, actual_starts, F, total_samples)

    # Save
    sf.write(str(output_dir / f"baseline_{tag}{suffix}"), baseline_out, sr)
    sf.write(str(output_dir / f"wcorr_{tag}{suffix}"), wcorr_out, sr)
    sf.write(str(output_dir / f"diff_{tag}{suffix}"), baseline_out - wcorr_out, sr)

    return SweepResult(
        threshold=w_corr_threshold,
        snr_vs_original=compute_snr(original, wcorr_out),
        snr_vs_baseline=compute_snr(baseline_out, wcorr_out),
        time_s=elapsed,
    )


def main():
    try:
        PROJ = Path(__file__).resolve().parent.parent
    except NameError:
        PROJ = Path(os.getcwd())

    audio_dir = PROJ / "data" / "raw" / "Brendel_Beethoven_Piano_Music_Vol9"
    output_dir = PROJ / "data" / "processed" / "wcorr_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    src_file = audio_dir / "09_Op27_No2_I_Adagio_sostenuto.mp3"
    if not src_file.exists():
        mp3s = sorted(audio_dir.glob("*.mp3"))
        src_file = mp3s[0]

    print(f"Source: {src_file.name}")
    audio, sr = sf.read(str(src_file), dtype="float64")

    # Skip silence, ~3s audio (keep test fast)
    skip = int(sr * 0.5)
    audio = audio[skip:skip + int(sr * 3.5)]
    print(f"  {audio.shape[0]} samples ({audio.shape[0]/sr:.1f}s)")

    F, L, hop = 1024, 256, 512
    energy_fraction = 0.9
    suffix = src_file.suffix

    # Baseline first
    print(f"\nBaseline: F={F} L={L} hop={hop} energy={energy_fraction}")
    baseline_step = _EnergySvdStep(
        EnergyThresholdStrategy(energy_fraction),
        w_corr_threshold=None, window_length=L
    )
    frame_starts = []
    pos = 0
    while pos + F <= audio.shape[0]:
        frame_starts.append(pos)
        pos += hop
    slices = [audio[s:s+F].copy() for s in frame_starts if s + F <= audio.shape[0]]
    actual_starts = [s for s in frame_starts if s + F <= audio.shape[0]]
    total_samples = actual_starts[-1] + F
    original = audio[:total_samples]

    baseline_frames = []
    for s in slices:
        baseline_frames.append(process_frame(s, window_length=L, svd_step=baseline_step))
    baseline_out = ola_reconstruct(baseline_frames, actual_starts, F, total_samples)
    sf.write(str(output_dir / f"original{suffix}"), original, sr)
    sf.write(str(output_dir / f"baseline{suffix}"), baseline_out, sr)
    baseline_snr = compute_snr(original, baseline_out)
    print(f"  SNR vs original: {baseline_snr:.1f}dB")

    # Sweep thresholds
    thresholds = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    print(f"\nSweeping W-correlation thresholds: {thresholds}\n")

    results = []
    for thr in thresholds:
        tag = f"thr{thr:.1f}".replace(".", "p")
        try:
            r = run_config(audio, sr, F, L, hop, energy_fraction, thr, output_dir, suffix, tag)
            results.append(r)
            print(f"  threshold={thr:.1f}: SNR_vs_orig={r.snr_vs_original:.1f}dB  SNR_vs_baseline={r.snr_vs_baseline:.1f}dB  time={r.time_s:.1f}s")
        except Exception as e:
            print(f"  threshold={thr:.1f}: FAILED - {e}")

    # Summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "baseline_snr": baseline_snr,
            "sweep": [asdict(r) for r in results]
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSummary: {summary_path}")

    if results:
        best = max(results, key=lambda r: r.snr_vs_original)
        print(f"\nBest threshold: {best.threshold:.1f} (SNR={best.snr_vs_original:.1f}dB)")


if __name__ == "__main__":
    main()
