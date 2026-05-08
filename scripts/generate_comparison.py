"""Test script to generate comparison files for different denoising methods."""

import os
import sys
import tempfile
import subprocess

# Add src to path
sys.path.insert(0, '/home/exusiai/_dev/Hankel-Stereo-Purify')

from src.facade.purifier import AudioPurifier
from src.core.strategies.truncation import HeuristicStrategy


def convert_mp3_to_wav(mp3_path: str, duration: int = 10) -> str:
    """Convert MP3 to WAV using ffmpeg (keep stereo)."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        temp_wav_path = temp_wav.name
    
    cmd = ['ffmpeg', '-i', mp3_path, '-t', str(duration), '-ar', '44100', temp_wav_path, '-y']
    subprocess.run(cmd, capture_output=True)
    return temp_wav_path


def test_method(method_name: str, purifier: AudioPurifier, input_path: str, output_dir: str):
    """Test a denoising method and generate output files."""
    output_path = os.path.join(output_dir, f"{method_name}.wav")
    diff_path = os.path.join(output_dir, f"{method_name}_diff.wav")
    
    print(f"测试 {method_name}...")
    
    # Process audio
    try:
        purifier.process_file(input_path, output_path)
        print(f"  输出: {output_path}")
    except Exception as e:
        print(f"  错误: {e}")
        return
    
    # Generate diff file (original - output)
    try:
        import soundfile as sf
        import numpy as np
        
        # Read original and output
        original, sr = sf.read(input_path)
        output, _ = sf.read(output_path)
        
        # Make sure they have the same length
        min_len = min(len(original), len(output))
        original = original[:min_len]
        output = output[:min_len]
        
        # Compute diff
        diff = original - output
        
        # Save diff
        sf.write(diff_path, diff, sr)
        print(f"  Diff: {diff_path}")
        
        # Compute some statistics
        original_power = np.mean(original ** 2)
        diff_power = np.mean(diff ** 2)
        output_power = np.mean(output ** 2)
        
        if diff_power > 1e-10:
            snr = 10 * np.log10(original_power / diff_power)
        else:
            snr = float('inf')
        
        print(f"  SNR: {snr:.2f} dB")
        print(f"  原始功率: {original_power:.6f}")
        print(f"  输出功率: {output_power:.6f}")
        print(f"  Diff功率: {diff_power:.6f}")
        
    except Exception as e:
        print(f"  生成diff文件时出错: {e}")


def main():
    """Main test function."""
    # Paths
    mp3_path = "/home/exusiai/_dev/Hankel-Stereo-Purify/data/raw/Brendel_Beethoven_Piano_Music_Vol9/01_Op27_No1_I_Andante.mp3"
    output_dir = "/home/exusiai/_dev/Hankel-Stereo-Purify/data/processed/heuristic_comparison"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert MP3 to WAV (10 seconds)
    print("转换MP3到WAV...")
    wav_path = convert_mp3_to_wav(mp3_path, duration=10)
    print(f"WAV文件: {wav_path}")
    
    # Test different methods
    print("\n测试不同降噪方法:")
    print("=" * 50)
    
    # 1. Energy truncation (0.9)
    print("\n1. Energy truncation (0.9)")
    purifier_energy = AudioPurifier(
        window_length=256,
        energy_fraction=0.9,
    )
    test_method("energy_0.9", purifier_energy, wav_path, output_dir)
    
    # 2. Wiener weighting (0.1)
    print("\n2. Wiener weighting (0.1)")
    # Disable CUDA for now due to bugs
    import os
    os.environ['HSP_DISABLE_CUDA'] = '1'
    purifier_wiener = AudioPurifier(
        window_length=256,
        wiener_noise_fraction=0.1,
    )
    test_method("wiener_0.1", purifier_wiener, wav_path, output_dir)
    
    # 3. Heuristic multi-feature weighting
    print("\n3. Heuristic multi-feature weighting")
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
    test_method("heuristic", purifier_heuristic, wav_path, output_dir)
    
    # 4. Heuristic with different parameters
    print("\n4. Heuristic (SFM阈值 0.3-0.7)")
    heuristic_strategy2 = HeuristicStrategy(
        sfm_weight=0.4,
        energy_weight=0.4,
        temporal_weight=0.2,
        sfm_threshold_low=0.3,
        sfm_threshold_high=0.7,
        energy_threshold=0.01,
        temporal_threshold=0.3,
    )
    purifier_heuristic2 = AudioPurifier(
        window_length=256,
        heuristic_strategy=heuristic_strategy2,
    )
    test_method("heuristic_sfm0.3-0.7", purifier_heuristic2, wav_path, output_dir)
    
    # Clean up
    os.unlink(wav_path)
    
    print("\n" + "=" * 50)
    print("测试完成!")
    print(f"输出文件在: {output_dir}")
    print("\n请听感对比以下文件:")
    print("1. energy_0.9.wav - 能量截断")
    print("2. wiener_0.1.wav - Wiener软加权")
    print("3. heuristic.wav - 启发式多特征加权")
    print("4. heuristic_sfm0.3-0.7.wav - 启发式(SFM阈值0.3-0.7)")
    print("\n以及对应的diff文件（移除的内容）:")
    print("1. energy_0.9_diff.wav")
    print("2. wiener_0.1_diff.wav")
    print("3. heuristic_diff.wav")
    print("4. heuristic_sfm0.3-0.7_diff.wav")


if __name__ == "__main__":
    main()
