#!/usr/bin/env python3
"""Compare bandpass filter vs no-filter MSSA denoising.

Tests Brendel piano and RCA violin recordings.
Outputs: timing, SNR, diff audio files.
"""

import os
import sys
import time
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed", "bandpass_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test clips (10s segments from previous analysis)
CLIPS = [
    {
        "name": "brendel_op27",
        "input": "/tmp/test_brendel_stereo.wav",
        "desc": "Brendel Beethoven Piano Op.27 No.1 (10s stereo)",
    },
    {
        "name": "rca_beethoven",
        "input": "/tmp/test_rca_stereo.wav",
        "desc": "RCA Heifetz Beethoven Violin Concerto (10s stereo)",
    },
]


def compute_snr(original: np.ndarray, processed: np.ndarray) -> float:
    """SNR = 10 * log10(Σx² / Σ(x-y)²)"""
    diff = original - processed
    signal_power = np.sum(original ** 2)
    noise_power = np.sum(diff ** 2)
    if noise_power < 1e-20:
        return float("inf")
    return 10.0 * np.log10(signal_power / noise_power)


def process_file(input_path: str, output_path: str, bypass_freq: float | None = None) -> float:
    """Process one file, return wall time."""
    from src.facade.purifier import AudioPurifier
    
    kwargs = {"energy_fraction": 0.9}
    if bypass_freq is not None:
        kwargs["bypass_freq"] = bypass_freq
    
    purifier = AudioPurifier(256, **kwargs)
    
    t0 = time.perf_counter()
    purifier.process_file(input_path, output_path)
    return time.perf_counter() - t0


def main():
    results = []
    
    for clip in CLIPS:
        print(f"\n{'='*60}")
        print(f"Testing: {clip['desc']}")
        print(f"Input: {clip['input']}")
        
        if not os.path.exists(clip['input']):
            print(f"  SKIP: input file not found")
            continue
        
        # Read original for SNR reference
        sr_orig, orig = wavfile.read(clip['input'])
        if orig.ndim > 1:
            orig = orig[:, 0]
        orig_f = orig.astype(np.float64) / 32768.0
        
        # --- Baseline: no filter ---
        baseline_out = os.path.join(OUTPUT_DIR, f"{clip['name']}_baseline.wav")
        print(f"\n  [1/2] Baseline (no filter)...")
        t_baseline = process_file(clip['input'], baseline_out, bypass_freq=None)
        
        sr_b, baseline = wavfile.read(baseline_out)
        if baseline.ndim > 1:
            baseline = baseline[:, 0]
        baseline_f = baseline.astype(np.float64)
        if baseline_f.max() > 1.0:
            baseline_f = baseline_f / 32768.0
        
        # Align lengths
        min_len = min(len(orig_f), len(baseline_f))
        snr_baseline = compute_snr(orig_f[:min_len], baseline_f[:min_len])
        
        # Diff: original - baseline
        diff_baseline = orig_f[:min_len] - baseline_f[:min_len]
        diff_baseline_int16 = (diff_baseline * 32767).clip(-32768, 32767).astype(np.int16)
        diff_baseline_path = os.path.join(OUTPUT_DIR, f"{clip['name']}_diff_baseline.wav")
        wavfile.write(diff_baseline_path, sr_orig, diff_baseline_int16)
        
        print(f"    Time: {t_baseline:.2f}s | SNR: {snr_baseline:.1f}dB")
        
        # --- Bandpass filter: bypass < 2kHz ---
        bp_out = os.path.join(OUTPUT_DIR, f"{clip['name']}_bypass2k.wav")
        print(f"  [2/2] Bandpass (bypass < 2000Hz)...")
        t_bp = process_file(clip['input'], bp_out, bypass_freq=2000.0)
        
        sr_bp, bp_result = wavfile.read(bp_out)
        if bp_result.ndim > 1:
            bp_result = bp_result[:, 0]
        bp_f = bp_result.astype(np.float64)
        if bp_f.max() > 1.0:
            bp_f = bp_f / 32768.0
        
        min_len2 = min(len(orig_f), len(bp_f))
        snr_bp = compute_snr(orig_f[:min_len2], bp_f[:min_len2])
        
        # Diff: original - bandpass
        diff_bp = orig_f[:min_len2] - bp_f[:min_len2]
        diff_bp_int16 = (diff_bp * 32767).clip(-32768, 32767).astype(np.int16)
        diff_bp_path = os.path.join(OUTPUT_DIR, f"{clip['name']}_diff_bypass2k.wav")
        wavfile.write(diff_bp_path, sr_orig, diff_bp_int16)
        
        # Diff: baseline vs bandpass
        min_len3 = min(len(baseline_f), len(bp_f))
        diff_bl_bp = baseline_f[:min_len3] - bp_f[:min_len3]
        diff_bl_bp_int16 = (diff_bl_bp * 32767).clip(-32768, 32767).astype(np.int16)
        diff_bl_bp_path = os.path.join(OUTPUT_DIR, f"{clip['name']}_diff_baseline_vs_bypass.wav")
        wavfile.write(diff_bl_bp_path, sr_orig, diff_bl_bp_int16)
        
        speedup = t_baseline / t_bp if t_bp > 0 else float("inf")
        
        print(f"    Time: {t_bp:.2f}s | SNR: {snr_bp:.1f}dB | Speedup: {speedup:.2f}x")
        
        results.append({
            "clip": clip['desc'],
            "baseline_time": t_baseline,
            "baseline_snr": snr_baseline,
            "bp_time": t_bp,
            "bp_snr": snr_bp,
            "speedup": speedup,
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Clip':<45} {'Baseline':>12} {'Bypass2k':>12} {'Speedup':>8}")
    print(f"{'':45} {'time SNR':>12} {'time SNR':>12} {'':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['clip']:<45} {r['baseline_time']:>5.1f}s {r['baseline_snr']:>5.1f}dB "
              f"{r['bp_time']:>5.1f}s {r['bp_snr']:>5.1f}dB {r['speedup']:>7.2f}x")
    
    print(f"\nOutput files in: {OUTPUT_DIR}/")
    print("  *_baseline.wav        — no-filter output")
    print("  *_bypass2k.wav        — bandpass output")
    print("  *_diff_baseline.wav   — original - baseline")
    print("  *_diff_bypass2k.wav   — original - bandpass")
    print("  *_diff_baseline_vs_bypass.wav — baseline - bandpass")


if __name__ == "__main__":
    main()
