"""Wiener soft weighting vs energy hard truncation comparison.

Outputs:
  - energy.mp3      — energy truncation (energy_fraction=0.9)
  - wiener_*.mp3    — Wiener with different noise_fraction
  - diff_*.mp3      — energy − wiener

Usage: cd Hankel-Stereo-Purify && python -m tests.test_wiener
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
from src.core.stages.c_svd import _EnergySvdStep, _WienerSvdStep
from src.core.strategies.truncation import EnergyThresholdStrategy, WienerStrategy


@dataclass
class Result:
    label: str
    snr_vs_original: float
    snr_vs_energy: float
    rms: float
    rms_diff_vs_energy: float
    diff_pct: float
    time_s: float


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


def run_ola(audio, F, L, hop, svd_step):
    frame_starts = []
    pos = 0
    while pos + F <= audio.shape[0]:
        frame_starts.append(pos)
        pos += hop
    slices = [audio[s:s+F].copy() for s in frame_starts]
    total_samples = frame_starts[-1] + F

    t0 = time.perf_counter()
    out_frames = [process_frame(s, window_length=L, svd_step=svd_step) for s in slices]
    elapsed = time.perf_counter() - t0

    return ola_reconstruct(out_frames, frame_starts, F, total_samples), elapsed


def main():
    try:
        PROJ = Path(__file__).resolve().parent.parent
    except NameError:
        PROJ = Path(os.getcwd())

    audio_dir = PROJ / "data" / "raw" / "Brendel_Beethoven_Piano_Music_Vol9"
    output_dir = PROJ / "data" / "processed" / "wiener_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    src_file = audio_dir / "09_Op27_No2_I_Adagio_sostenuto.mp3"
    if not src_file.exists():
        mp3s = sorted(audio_dir.glob("*.mp3"))
        src_file = mp3s[0]

    print(f"Source: {src_file.name}")
    audio, sr = sf.read(str(src_file), dtype="float64")
    skip = int(sr * 0.5)
    audio = audio[skip:skip + int(sr * 1.0)]
    print(f"  Clip: {audio.shape[0]} samples ({audio.shape[0]/sr:.1f}s)")

    F, L, hop = 1024, 256, 512
    suffix = src_file.suffix
    original = audio

    # Energy baseline
    print(f"\n=== Energy baseline (energy_fraction=0.9) ===")
    energy_step = _EnergySvdStep(EnergyThresholdStrategy(0.9), w_corr_threshold=None, window_length=L)
    energy_out, t_energy = run_ola(audio, F, L, hop, energy_step)
    sf.write(str(output_dir / f"energy{suffix}"), energy_out, sr)
    rms_energy = compute_rms(energy_out)
    print(f"  SNR={compute_snr(original[:energy_out.shape[0]], energy_out):.1f}dB  RMS={rms_energy:.6f}  time={t_energy:.1f}s")

    # Wiener sweep
    noise_fracs = [0.05, 0.1, 0.15, 0.2, 0.3]
    print(f"\n=== Wiener sweep ===\n")
    print(f"{'nf':>5} {'SNR_orig':>9} {'SNR_vs_e':>9} {'RMS':>9} {'diff%':>7} {'time':>6}")

    results = []
    for nf in noise_fracs:
        wiener_step = _WienerSvdStep(WienerStrategy(nf))
        wiener_out, elapsed = run_ola(audio, F, L, hop, wiener_step)
        tag = f"nf{nf:.2f}".replace(".", "p")
        sf.write(str(output_dir / f"wiener_{tag}{suffix}"), wiener_out, sr)

        diff = energy_out - wiener_out
        sf.write(str(output_dir / f"diff_{tag}{suffix}"), diff, sr)

        snr_orig = compute_snr(original[:wiener_out.shape[0]], wiener_out)
        snr_vs_e = compute_snr(energy_out, wiener_out)
        rms_w = compute_rms(wiener_out)
        rms_d = compute_rms(diff)
        diff_pct = rms_d / rms_energy * 100 if rms_energy > 1e-30 else 0

        r = Result(
            label=f"wiener_nf{nf}",
            snr_vs_original=round(snr_orig, 1),
            snr_vs_energy=round(snr_vs_e, 1),
            rms=round(rms_w, 6),
            rms_diff_vs_energy=round(rms_d, 6),
            diff_pct=round(diff_pct, 1),
            time_s=round(elapsed, 1),
        )
        results.append(r)
        print(f"{nf:>5.2f} {snr_orig:>8.1f}dB {snr_vs_e:>8.1f}dB {rms_w:>9.6f} {diff_pct:>6.1f}% {elapsed:>5.1f}s")

    # Summary
    summary = {
        "energy_baseline": {"snr_vs_original": round(compute_snr(original[:energy_out.shape[0]], energy_out), 1), "rms": round(rms_energy, 6)},
        "wiener_sweep": [asdict(r) for r in results],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutput: {output_dir}/")
    print(f"  energy{suffix} — hard truncation baseline")
    for nf in noise_fracs:
        tag = f"nf{nf:.2f}".replace(".", "p")
        print(f"  wiener_{tag}{suffix} / diff_{tag}{suffix}")


if __name__ == "__main__":
    main()
