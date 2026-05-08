"""Test heuristic weighting weights only."""

import numpy as np
import soundfile as sf
import tempfile
import os
from scipy.linalg import hankel

# Add src to path
import sys
sys.path.insert(0, '/home/exusiai/_dev/Hankel-Stereo-Purify')

from src.core.strategies.heuristic import HeuristicMultiFeatureStrategy


def test_heuristic_weights():
    """Test heuristic weighting weights distribution."""
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
    
    # Test heuristic weighting
    heuristic = HeuristicMultiFeatureStrategy()
    weights = heuristic.get_weights(U, s, Vt, len(data))
    
    print("\n权重分布:")
    print(f"{'组件':<5} {'奇异值':<12} {'能量%':<10} {'权重':<10} {'权重*奇异值':<12}")
    print("-" * 55)
    
    for i in range(min(20, len(s))):
        energy_pct = (s[i] * s[i]) / np.sum(s * s) * 100
        print(f"{i:<5} {s[i]:<12.4f} {energy_pct:<10.2f} {weights[i]:<10.4f} {weights[i] * s[i]:<12.4f}")
    
    print(f"\n权重统计:")
    print(f"权重均值: {np.mean(weights):.4f}")
    print(f"权重标准差: {np.std(weights):.4f}")
    print(f"权重最小值: {np.min(weights):.4f}")
    print(f"权重最大值: {np.max(weights):.4f}")
    print(f"权重>0.5的组件数: {np.sum(weights > 0.5)}")
    print(f"权重>0.9的组件数: {np.sum(weights > 0.9)}")
    
    # Compare with Wiener weights
    print("\n与Wiener权重对比:")
    k = len(s)
    n_noise = max(1, int(k * 0.1))
    noise_var = np.mean(s[-n_noise:] ** 2)
    s_sq = s * s
    with np.errstate(divide="ignore", invalid="ignore"):
        wiener_weights = np.where(s_sq > noise_var, 1.0 - noise_var / s_sq, 0.0)
    
    print(f"Wiener权重均值: {np.mean(wiener_weights):.4f}")
    print(f"Wiener权重>0.5的组件数: {np.sum(wiener_weights > 0.5)}")
    print(f"Wiener权重>0.9的组件数: {np.sum(wiener_weights > 0.9)}")
    
    # Show top 10 components by weight
    print("\n权重最高的10个组件:")
    top_indices = np.argsort(weights)[::-1][:10]
    for idx in top_indices:
        energy_pct = (s[idx] * s[idx]) / np.sum(s * s) * 100
        print(f"  组件{idx}: 奇异值={s[idx]:.4f}, 能量={energy_pct:.2f}%, 权重={weights[idx]:.4f}")


if __name__ == "__main__":
    print("测试启发式多特征加权策略权重分布")
    print("=" * 50)
    
    # Test weights
    test_heuristic_weights()
