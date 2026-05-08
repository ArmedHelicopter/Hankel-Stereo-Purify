"""Quick test for heuristic multi-feature weighting strategy."""

import numpy as np
import soundfile as sf
import tempfile
import os
from scipy.linalg import hankel

# Add src to path
import sys
sys.path.insert(0, '/home/exusiai/_dev/Hankel-Stereo-Purify')

from src.core.strategies.heuristic import HeuristicMultiFeatureStrategy


def test_component_features():
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
    
    # Analyze first 10 components
    print("\n组件特征分析:")
    print(f"{'组件':<5} {'奇异值':<12} {'能量%':<10} {'SFM':<10} {'时域结构':<12} {'信号概率':<12}")
    print("-" * 70)
    
    heuristic = HeuristicMultiFeatureStrategy()
    total_energy = np.sum(s * s)
    
    for i in range(min(10, len(s))):
        # Reconstruct component i
        component_matrix = U[:, i:i+1] @ (s[i] * Vt[i:i+1, :])
        
        # Use diagonal averaging
        component_signal = heuristic._diagonal_average(component_matrix, len(data))
        
        # Compute features
        signal_prob, sfm, energy_ratio, temporal = heuristic.compute_feature_weights(
            component_signal, s[i] * s[i], total_energy
        )
        
        print(f"{i:<5} {s[i]:<12.4f} {energy_ratio*100:<10.2f} {sfm:<10.4f} {temporal:<12.4f} {signal_prob:<12.4f}")


if __name__ == "__main__":
    print("测试启发式多特征加权策略")
    print("=" * 50)
    
    # Analyze component features
    test_component_features()
