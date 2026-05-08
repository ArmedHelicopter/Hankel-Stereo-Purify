"""W-correlation performance analysis vs pure energy baseline.

Outputs per threshold:
  - energy.mp3           — pure energy baseline
  - wcorr_{thr}.mp3      — W-correlation result
  - diff_{thr}.mp3       — energy − wcorr (what W-corr additionally removes)
  - summary.json         — SNR metrics

diff = energy_out − wcorr_out
  - If W-corr removes MORE than energy: diff is the extra removed signal+noise
  - If W-corr keeps more: diff is negative (wcorr adds back components)
  - diff ≈ 0 means W-corr ≈ energy (no filtering effect)

Usage: cd Hankel-Stereo-Purify && python -m tests.test_wcorr_analysis
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
class AnalysisResult:
    threshold: float
    snr_vs_original: float
    snr_diff_energy_vs_wcorr: float
    rms_energy: float
    rms_wcorr: float
    rms_diff: float
    diff_ratio_pct: float
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


def run_frames(audio, F, L, hop, energy_fraction, w_corr_threshold):
    strat = EnergyThresholdStrategy(energy_fraction)
    svd_step = _EnergySvdStep(strat, w_corr_threshold=w_corr_threshold, window_length=L)

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
    output_dir = PROJ / "data" / "processed" / "wcorr_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    src_file = audio_dir / "09_Op27_No2_I_Adagio_sostenuto.mp3"
    if not src_file.exists():
        mp3s = sorted(audio_dir.glob("*.mp3"))
        src_file = mp3s[0]

    print(f"Source: {src_file.name}")
    audio, sr = sf.read(str(src_file), dtype="float64")
    skip = int(sr * 0.5)
    audio = audio[skip:skip + int(sr * 5.0)]
    print(f"  Clip: {audio.shape[0]} samples ({audio.shape[0]/sr:.1f}s)")

    F, L, hop = 1024, 256, 512
    energy_fraction = 0.9
    suffix = src_file.suffix

    # Pure energy baseline
    print(f"\n=== Baseline: energy={energy_fraction}, hop=F/2 ===")
    energy_out, t_energy = run_frames(audio, F, L, hop, energy_fraction, None)
    sf.write(str(output_dir / f"energy{suffix}"), energy_out, sr)
    print(f"  RMS={compute_rms(energy_out):.6f}  time={t_energy:.1f}s")

    # W-correlation sweep
    thresholds = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    print(f"\n=== W-correlation sweep ===\n")
    print(f"{'thr':>5} {'SNR_orig':>9} {'SNR_diff':>9} {'RMS_diff':>9} {'diff%':>7} {'time':>6}")

    results = []
    for thr in thresholds:
        tag = f"thr{thr:.1f}".replace(".", "p")
        wcorr_out, elapsed = run_frames(audio, F, L, hop, energy_fraction, thr)
        sf.write(str(output_dir / f"wcorr_{tag}{suffix}"), wcorr_out, sr)

        diff = energy_out - wcorr_out
        sf.write(str(output_dir / f"diff_{tag}{suffix}"), diff, sr)

        snr_orig = compute_snr(audio[:energy_out.shape[0]], wcorr_out)
        snr_diff = compute_snr(energy_out, wcorr_out)
        rms_e = compute_rms(energy_out)
        rms_w = compute_rms(wcorr_out)
        rms_d = compute_rms(diff)
        diff_pct = rms_d / rms_e * 100 if rms_e > 1e-30 else 0

        r = AnalysisResult(
            threshold=thr,
            snr_vs_original=round(snr_orig, 1),
            snr_diff_energy_vs_wcorr=round(snr_diff, 1),
            rms_energy=round(rms_e, 6),
            rms_wcorr=round(rms_w, 6),
            rms_diff=round(rms_d, 6),
            diff_ratio_pct=round(diff_pct, 1),
            time_s=round(elapsed, 1),
        )
        results.append(r)
        print(f"{thr:>5.1f} {snr_orig:>8.1f}dB {snr_diff:>8.1f}dB {rms_d:>9.6f} {diff_pct:>6.1f}% {elapsed:>5.1f}s")

    # Summary
    summary = {
        "baseline": {
            "rms": round(compute_rms(energy_out), 6),
            "time_s": round(t_energy, 1),
        },
        "sweep": [asdict(r) for r in results],
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== Interpretation ===")
    print(f"diff = energy_out − wcorr_out")
    print(f"  SNR_diff (dB): energy vs wcorr similarity (higher = more similar)")
    print(f"  diff%: RMS of extra removal by wcorr vs energy baseline")
    print(f"")
    print(f"  thr=0.1: wcorr barely filters → diff≈0 → same as energy")
    print(f"  thr=0.9: wcorr aggressively filters → large diff → removes more")
    print(f"")
    print(f"Output: {output_dir}/")
    print(f"  energy{suffix}         — pure energy baseline")
    for r in results:
        tag = f"thr{r.threshold:.1f}".replace(".", "p")
        print(f"  wcorr_{tag}{suffix}    — W-corr thr={r.threshold}")
        print(f"  diff_{tag}{suffix}     — energy − wcorr (additional removal)")


if __name__ == "__main__":
    main()
