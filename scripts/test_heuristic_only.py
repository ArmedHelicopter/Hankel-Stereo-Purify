"""Quick test for heuristic method only."""

import os
import sys
import tempfile
import subprocess

# Add src to path
sys.path.insert(0, '/home/exusiai/_dev/Hankel-Stereo-Purify')

from src.facade.purifier import AudioPurifier
from src.core.strategies.truncation import HeuristicStrategy


def convert_mp3_to_wav(mp3_path: str, duration: int = 3) -> str:
    """Convert MP3 to WAV using ffmpeg (keep stereo)."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        temp_wav_path = temp_wav.name
    
    cmd = ['ffmpeg', '-i', mp3_path, '-t', str(duration), '-ar', '44100', temp_wav_path, '-y']
    subprocess.run(cmd, capture_output=True)
    return temp_wav_path


def test_heuristic_only():
    """Test heuristic method only."""
    # Paths
    mp3_path = "/home/exusiai/_dev/Hankel-Stereo-Purify/data/raw/Brendel_Beethoven_Piano_Music_Vol9/01_Op27_No1_I_Andante.mp3"
    output_dir = "/home/exusiai/_dev/Hankel-Stereo-Purify/data/processed/heuristic_comparison"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert MP3 to WAV (3 seconds)
    print("转换MP3到WAV...")
    wav_path = convert_mp3_to_wav(mp3_path, duration=3)
    print(f"WAV文件: {wav_path}")
    
    # Test heuristic method
    print("\n测试启发式多特征加权...")
    heuristic_strategy = HeuristicStrategy(
        sfm_weight=0.4,
        energy_weight=0.4,
        temporal_weight=0.2,
        sfm_threshold_low=0.2,
        sfm_threshold_high=0.6,
        energy_threshold=0.01,
        temporal_threshold=0.3,
    )
    purifier_heuristic = AudioPurifier(
        window_length=256,
        heuristic_strategy=heuristic_strategy,
    )
    
    output_path = os.path.join(output_dir, "heuristic_test.wav")
    
    print("开始处理...")
    try:
        purifier_heuristic.process_file(wav_path, output_path)
        print(f"输出: {output_path}")
        
        # Generate diff file
        import soundfile as sf
        import numpy as np
        
        original, sr = sf.read(wav_path)
        output, _ = sf.read(output_path)
        
        min_len = min(len(original), len(output))
        original = original[:min_len]
        output = output[:min_len]
        
        diff = original - output
        diff_path = os.path.join(output_dir, "heuristic_test_diff.wav")
        sf.write(diff_path, diff, sr)
        
        # Compute statistics
        original_power = np.mean(original ** 2)
        diff_power = np.mean(diff ** 2)
        output_power = np.mean(output ** 2)
        
        if diff_power > 1e-10:
            snr = 10 * np.log10(original_power / diff_power)
        else:
            snr = float('inf')
        
        print(f"SNR: {snr:.2f} dB")
        print(f"原始功率: {original_power:.6f}")
        print(f"输出功率: {output_power:.6f}")
        print(f"Diff功率: {diff_power:.6f}")
        
    except Exception as e:
        print(f"错误: {e}")
    
    # Clean up
    os.unlink(wav_path)
    
    print("\n测试完成!")


if __name__ == "__main__":
    test_heuristic_only()
