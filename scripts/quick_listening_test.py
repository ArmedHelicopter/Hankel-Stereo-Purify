"""Quick listening test - generate comparison files."""

import os
import sys
import tempfile
import subprocess

# Add src to path
sys.path.insert(0, '/home/exusiai/_dev/Hankel-Stereo-Purify')

from src.facade.purifier import AudioPurifier
from src.core.strategies.truncation import HeuristicStrategy


def main():
    """Generate comparison files for listening test."""
    # Paths
    mp3_path = "/home/exusiai/_dev/Hankel-Stereo-Purify/data/raw/Brendel_Beethoven_Piano_Music_Vol9/01_Op27_No1_I_Andante.mp3"
    output_dir = "/home/exusiai/_dev/Hankel-Stereo-Purify/data/processed/listening_test"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert MP3 to WAV (3 seconds for faster testing)
    print("转换MP3到WAV (3秒)...")
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        wav_path = temp_wav.name
    
    cmd = ['ffmpeg', '-i', mp3_path, '-t', '3', '-ar', '44100', wav_path, '-y']
    subprocess.run(cmd, capture_output=True)
    print(f"WAV文件: {wav_path}")
    
    # Test 1: Energy truncation (0.9) - baseline
    print("\n1. Energy truncation (0.9)")
    try:
        purifier_energy = AudioPurifier(
            window_length=256,
            energy_fraction=0.9,
        )
        energy_output = os.path.join(output_dir, "energy_0.9.wav")
        purifier_energy.process_file(wav_path, energy_output)
        print(f"  输出: {energy_output}")
    except Exception as e:
        print(f"  错误: {e}")
    
    # Test 2: Heuristic multi-feature weighting
    print("\n2. Heuristic multi-feature weighting")
    try:
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
        heuristic_output = os.path.join(output_dir, "heuristic.wav")
        purifier_heuristic.process_file(wav_path, heuristic_output)
        print(f"  输出: {heuristic_output}")
    except Exception as e:
        print(f"  错误: {e}")
    
    # Generate diff files
    print("\n生成diff文件...")
    try:
        import soundfile as sf
        import numpy as np
        
        # Read original
        original, sr = sf.read(wav_path)
        
        # Read energy output
        if os.path.exists(os.path.join(output_dir, "energy_0.9.wav")):
            energy_output_data, _ = sf.read(os.path.join(output_dir, "energy_0.9.wav"))
            min_len = min(len(original), len(energy_output_data))
            diff_energy = original[:min_len] - energy_output_data[:min_len]
            sf.write(os.path.join(output_dir, "energy_0.9_diff.wav"), diff_energy, sr)
            print("  energy_0.9_diff.wav")
        
        # Read heuristic output
        if os.path.exists(os.path.join(output_dir, "heuristic.wav")):
            heuristic_output_data, _ = sf.read(os.path.join(output_dir, "heuristic.wav"))
            min_len = min(len(original), len(heuristic_output_data))
            diff_heuristic = original[:min_len] - heuristic_output_data[:min_len]
            sf.write(os.path.join(output_dir, "heuristic_diff.wav"), diff_heuristic, sr)
            print("  heuristic_diff.wav")
        
    except Exception as e:
        print(f"  生成diff文件时出错: {e}")
    
    # Clean up
    os.unlink(wav_path)
    
    print("\n" + "=" * 50)
    print("测试完成!")
    print(f"输出文件在: {output_dir}")
    print("\n请听感对比以下文件:")
    print("1. energy_0.9.wav - 能量截断 (基线)")
    print("2. heuristic.wav - 启发式多特征加权")
    print("\n以及对应的diff文件（移除的内容）:")
    print("1. energy_0.9_diff.wav")
    print("2. heuristic_diff.wav")
    print("\n请关注:")
    print("- 哪个方法降噪效果最好？")
    print("- 哪个方法保留的信号最完整？")
    print("- 哪个方法引入的伪影最少？")


if __name__ == "__main__":
    main()
