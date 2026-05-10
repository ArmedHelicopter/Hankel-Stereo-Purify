"""svds (partial) vs svd (full) comparison.

Compares: SNR between outputs, timing.
Usage: cd Hankel-Stereo-Purify && python -m tests.test_svds_vs_svd
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import scipy.linalg
import soundfile as sf
from numpy.typing import NDArray
from scipy.sparse.linalg import svds

from src.core.stages.diagonal import fast_diagonal_average


def svd_full(a: NDArray) -> NDArray:
    """Full SVD reconstruction (ground truth)."""
    u, s, vh = scipy.linalg.svd(a, full_matrices=False)
    return (u * s) @ vh


def svds_partial(a: NDArray, k_probe: int) -> NDArray:
    """Partial svds reconstruction."""
    m, n = a.shape
    mn = min(m, n)
    k = min(k_probe, mn - 1)  # svds needs k < min(m,n)

    u_k, s_k, vh_k = svds(a, k=k, which="LM")

    # Sort descending
    order = np.argsort(s_k)[::-1]
    u_k, s_k, vh_k = u_k[:, order], s_k[order], vh_k[order, :]

    return (u_k * s_k) @ vh_k


def main():
    try:
        PROJ = Path(__file__).resolve().parent.parent
    except NameError:
        PROJ = Path(".")

    src = PROJ / "data/raw/Brendel_Beethoven_Piano_Music_Vol9/09_Op27_No2_I_Adagio_sostenuto.mp3"
    audio, sr = sf.read(str(src), dtype="float64")
    audio = audio[int(sr * 0.5) : int(sr * 1.0)]

    from src.core.stages.hankel import hankel_embed
    from src.core.stages.multichannel import combine_hankel_blocks

    F, L = 1024, 256

    # Build one frame
    frame = audio[:F]
    h_l, h_r = hankel_embed(L, frame)
    joint = combine_hankel_blocks(h_l, h_r)
    m, n = joint.shape
    mn = min(m, n)
    print(f"Matrix: {m}x{n}, min={mn}\n")

    # Ground truth: full SVD
    t0 = time.perf_counter()
    full_out = svd_full(joint)
    t_full = time.perf_counter() - t0
    print(f"Full SVD: {t_full:.3f}s")

    # Partial svds at different k values
    k_values = [8, 16, 32, 64, 128]
    print(f"\n{'k_probe':>8} {'SNR_vs_full':>12} {'time':>8} {'speedup':>8}")
    print("-" * 40)

    for k_probe in k_values:
        if k_probe >= mn:
            continue
        t0 = time.perf_counter()
        partial_out = svds_partial(joint, k_probe)
        t_partial = time.perf_counter() - t0

        diff = full_out - partial_out
        snr = 10 * np.log10(np.sum(full_out ** 2) / max(np.sum(diff ** 2), 1e-30))
        speedup = t_full / t_partial if t_partial > 0 else float("inf")

        print(f"{k_probe:>8} {snr:>11.1f}dB {t_partial:>7.3f}s {speedup:>7.1f}x")

    # Also test: full SVD vs full SVD hard truncation (energy 0.9)
    print(f"\n--- Full SVD vs Energy truncation (full frame pipeline) ---")
    from src.core.process_frame import process_frame
    from src.core.stages.svd import _EnergySvdStep
    from src.core.strategies.truncation import EnergyThresholdStrategy

    hop = F // 2
    starts = list(range(0, len(audio) - F + 1, hop))
    slices = [audio[s : s + F].copy() for s in starts]
    total = starts[-1] + F
    orig = audio[:total]
    w = np.sqrt(0.5 * (1 - np.cos(2 * np.pi * np.arange(F) / F)))

    def ola(fr):
        out = np.zeros((total, 2))
        ws = np.zeros(total)
        for f, st in zip(fr, starts):
            out[st : st + F] += f * w[:, np.newaxis]
            ws[st : st + F] += w
        m = ws > 1e-10
        out[m] /= ws[m, np.newaxis]
        return out

    # Energy
    step_e = _EnergySvdStep(EnergyThresholdStrategy(0.9))
    t0 = time.perf_counter()
    frames_e = [process_frame(s, window_length=L, svd_step=step_e) for s in slices]
    t_e = time.perf_counter() - t0
    energy_out = ola(frames_e)
    snr_e = 10 * np.log10(np.sum(orig ** 2) / np.sum((orig - energy_out) ** 2))
    print(f"Energy (svds partial): SNR={snr_e:.1f}dB time={t_e:.2f}s ({len(slices) / t_e:.0f} f/s)")


if __name__ == "__main__":
    main()
