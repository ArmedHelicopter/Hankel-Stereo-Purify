"""Test heuristic weighting with efficient SNR computation."""

import numpy as np
import soundfile as sf
import tempfile
import os
from scipy.linalg import hankel

# Add src to path
import sys
sys.path.insert(0, '/home/exusiai/_dev/Hankel-Stereo-Purify')

from src.core.strategies.heuristic import HeuristicMultiFeatureStrategy


def compute_snr(reference: np.ndarray, test: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio in dB."""
    noise = reference - test
    signal_power = np.mean(reference ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-10:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


def diagonal_average_vectorized(matrix: np.ndarray, signal_length: int) -> np.ndarray:
    """Vectorized diagonal averaging for Hankel matrix reconstruction."""
    rows, cols = matrix.shape
    
    # Create index arrays for diagonal averaging
    i_indices, j_indices = np.indices((rows, cols))
    diag_indices = i_indices + j_indices
    
    # Flatten arrays for bincount
    flat_diag_indices = diag_indices.ravel()
    flat_matrix = matrix.ravel()
    
    # Use bincount for efficient diagonal averaging
    result = np.bincount(flat_diag_indices, weights=flat_matrix, minlength=signal_length)
    counts = np.bincount(flat_diag_indices, minlength=signal_length)
    
    # Avoid division by zero
    counts[counts == 0] = 1
    return result / counts


def test_strategies():
    """Test different weighting strategies."""
    # Convert MP3 to WAV first
    mp3_path = "/home/exusiai/_dev/Hankel-Stereo-Purify/data/raw/Brendel_Beethoven_Piano_Music_Vol9/01_Op27_No1_I_Andante.mp3"
    
    # Create temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        temp_wav_path = temp_wav.name
    
    # Convert using ffmpeg
    import subprocess
    cmd = ['ffmpeg', '-i', mp3_path, '-t', '3', '-ar', '44100', '-ac', '1', temp_wav_path, '-y']
    subprocess.run(cmd, capture_output=True)
    
    # Read audio
    data, sr = sf.read(temp_wav_path)
    os.unlink(temp_wav_path)
    
    print(f"采样率: {sr}, 采样点: {len(data)}, 时长: {len(data)/sr:.1f}s")
    
    # Hankel matrix parameters
    L = 256  # Window length
    K = len(data) - L + 1  # Number of windows
    
    # Build Hankel matrix
    X = hankel(data[:K], data[K-1:])
    print(f"Hankel矩阵形状: {X.shape}")
    
    # SVD decomposition
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    print(f"SVD分解完成: {len(s)}个奇异值")
    
    # Test strategies
    print("\n策略对比:")
    print(f"{'策略':<25} {'SNR (dB)':<15} {'保留组件':<15} {'权重均值':<15}")
    print("-" * 70)
    
    # 1. Energy truncation (0.9)
    energy_weights = np.zeros_like(s)
    energy = s * s
    total_energy = np.sum(energy)
    cum_energy = np.cumsum(energy) / total_energy
    k_energy = np.searchsorted(cum_energy, 0.9) + 1
    energy_weights[:k_energy] = 1.0
    reconstructed_energy = (U * (s * energy_weights)) @ Vt
    signal_energy = diagonal_average_vectorized(reconstructed_energy, len(data))
    snr_energy = compute_snr(data, signal_energy)
    print(f"{'Energy (0.9)':<25} {snr_energy:<15.2f} {k_energy:<15} {np.mean(energy_weights):<15.3f}")
    
    # 2. Wiener weighting
    k = len(s)
    n_noise = max(1, int(k * 0.1))
    noise_var = np.mean(s[-n_noise:] ** 2)
    s_sq = s * s
    with np.errstate(divide="ignore", invalid="ignore"):
        wiener_weights = np.where(s_sq > noise_var, 1.0 - noise_var / s_sq, 0.0)
    reconstructed_wiener = (U * (s * wiener_weights)) @ Vt
    signal_wiener = diagonal_average_vectorized(reconstructed_wiener, len(data))
    snr_wiener = compute_snr(data, signal_wiener)
    print(f"{'Wiener (0.1)':<25} {snr_wiener:<15.2f} {np.sum(wiener_weights > 0.1):<15} {np.mean(wiener_weights):<15.3f}")
    
    # 3. Heuristic weighting
    heuristic = HeuristicMultiFeatureStrategy()
    heuristic_weights = heuristic.get_weights(U, s, Vt, len(data))
    reconstructed_heuristic = (U * (s * heuristic_weights)) @ Vt
    signal_heuristic = diagonal_average_vectorized(reconstructed_heuristic, len(data))
    snr_heuristic = compute_snr(data, signal_heuristic)
    print(f"{'Heuristic':<25} {snr_heuristic:<15.2f} {np.sum(heuristic_weights > 0.1):<15} {np.mean(heuristic_weights):<15.3f}")
    
    # 4. Heuristic with different thresholds
    print("\n启发式方法参数调优:")
    for sfm_low, sfm_high in [(0.1, 0.5), (0.2, 0.6), (0.3, 0.7)]:
        heuristic_tuned = HeuristicMultiFeatureStrategy(
            sfm_threshold_low=sfm_low,
            sfm_threshold_high=sfm_high,
        )
        weights_tuned = heuristic_tuned.get_weights(U, s, Vt, len(data))
        reconstructed_tuned = (U * (s * weights_tuned)) @ Vt
        signal_tuned = diagonal_average_vectorized(reconstructed_tuned, len(data))
        snr_tuned = compute_snr(data, signal_tuned)
        print(f"  SFM阈值({sfm_low}-{sfm_high}): SNR={snr_tuned:.2f}dB, 权重均值={np.mean(weights_tuned):.3f}")


if __name__ == "__main__":
    print("测试启发式多特征加权策略")
    print("=" * 50)
    
    # Test strategies
    test_strategies()
