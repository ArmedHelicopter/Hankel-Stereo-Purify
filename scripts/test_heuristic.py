"""Test script for heuristic multi-feature weighting strategy."""

import numpy as np
import soundfile as sf
import tempfile
import os
from scipy.linalg import hankel

# Add src to path
import sys
sys.path.insert(0, '/home/exusiai/_dev/Hankel-Stereo-Purify')

from src.core.strategies.heuristic import (
    HeuristicMultiFeatureStrategy,
    apply_heuristic_weighting,
)


def compute_snr(reference: np.ndarray, test: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio in dB."""
    noise = reference - test
    signal_power = np.mean(reference ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-10:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


def test_heuristic_on_audio():
    """Test heuristic strategy on piano audio."""
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
    
    # Test different strategies
    strategies = {
        "Energy (0.9)": lambda u, s, vh: apply_energy_truncation(u, s, vh, 0.9),
        "Wiener (0.1)": lambda u, s, vh: apply_wiener_weighting(u, s, vh, 0.1),
        "Heuristic": lambda u, s, vh: apply_heuristic_weighting(u, s, vh, len(data)),
    }
    
    print("\n策略对比:")
    print(f"{'策略':<20} {'SNR (dB)':<15} {'权重统计':<30}")
    print("-" * 65)
    
    for name, strategy_func in strategies.items():
        try:
            # Apply strategy
            reconstructed_matrix = strategy_func(U, s, Vt)
            
            # Reconstruct signal from Hankel matrix (diagonal averaging)
            reconstructed_signal = diagonal_average(reconstructed_matrix, len(data))
            
            # Compute SNR (compare with original)
            snr = compute_snr(data, reconstructed_signal)
            
            # For heuristic strategy, also show weight statistics
            if name == "Heuristic":
                heuristic = HeuristicMultiFeatureStrategy()
                weights = heuristic.get_weights(U, s, Vt, len(data))
                weight_stats = f"均值={np.mean(weights):.3f}, 标准差={np.std(weights):.3f}"
            else:
                weight_stats = ""
            
            print(f"{name:<20} {snr:<15.2f} {weight_stats:<30}")
        except Exception as e:
            print(f"{name:<20} Error: {str(e)}")


def apply_energy_truncation(u, s, vh, threshold):
    """Apply energy threshold truncation."""
    energy = s * s
    total_energy = np.sum(energy)
    cum_energy = np.cumsum(energy) / total_energy
    k = np.searchsorted(cum_energy, threshold) + 1
    k = min(k, len(s))
    
    # Truncate
    u_trunc = u[:, :k]
    s_trunc = s[:k]
    vh_trunc = vh[:k, :]
    
    # Reconstruct
    return (u_trunc * s_trunc) @ vh_trunc


def apply_wiener_weighting(u, s, vh, noise_fraction):
    """Apply Wiener soft weighting."""
    k = len(s)
    n_noise = max(1, int(k * noise_fraction))
    noise_var = np.mean(s[-n_noise:] ** 2)
    
    # Wiener gain
    s_sq = s * s
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(s_sq > noise_var, 1.0 - noise_var / s_sq, 0.0)
    
    weighted_s = s * weights
    return (u * weighted_s) @ vh


def diagonal_average(matrix, signal_length):
    """Average along diagonals to reconstruct signal from Hankel matrix."""
    rows, cols = matrix.shape
    result = np.zeros(signal_length)
    counts = np.zeros(signal_length)
    
    for i in range(rows):
        for j in range(cols):
            idx = i + j
            if idx < signal_length:
                result[idx] += matrix[i, j]
                counts[idx] += 1
    
    # Avoid division by zero
    counts[counts == 0] = 1
    return result / counts


def analyze_component_features():
    """Analyze features of individual SVD components."""
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
    
    # Hankel matrix parameters
    L = 256  # Window length
    K = len(data) - L + 1  # Number of windows
    
    # Build Hankel matrix
    X = hankel(data[:K], data[K-1:])
    
    # SVD decomposition
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Analyze first 10 components
    print("\n组件特征分析:")
    print(f"{'组件':<5} {'奇异值':<12} {'能量%':<10} {'SFM':<10} {'时域结构':<12} {'信号概率':<12}")
    print("-" * 70)
    
    heuristic = HeuristicMultiFeatureStrategy()
    total_energy = np.sum(s * s)
    
    for i in range(min(10, len(s))):
        # Reconstruct component i
        component_matrix = U[:, i:i+1] @ (s[i] * Vt[i:i+1, :])
        component_signal = component_matrix[0, :]
        
        # Compute features
        signal_prob, sfm, energy_ratio, temporal = heuristic.compute_feature_weights(
            component_signal, s[i] * s[i], total_energy
        )
        
        print(f"{i:<5} {s[i]:<12.4f} {energy_ratio*100:<10.2f} {sfm:<10.4f} {temporal:<12.4f} {signal_prob:<12.4f}")


if __name__ == "__main__":
    print("测试启发式多特征加权策略")
    print("=" * 50)
    
    # Analyze component features
    analyze_component_features()
    
    print("\n" + "=" * 50)
    print("策略对比测试")
    print("=" * 50)
    
    # Test strategies
    test_heuristic_on_audio()
