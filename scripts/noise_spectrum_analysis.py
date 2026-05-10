#!/usr/bin/env python3
"""
分析钢琴录音的噪声频段分布。
对比噪声段和乐音段的频谱特征，确定噪声集中在哪些频段。
"""
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch
import subprocess
import os
import sys
import json

def mp3_to_wav(mp3_path, wav_path="/tmp/temp_analysis.wav"):
    """ffmpeg convert mp3 to wav"""
    subprocess.run([
        "ffmpeg", "-y", "-i", mp3_path, "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", wav_path
    ], capture_output=True)
    return wav_path

def load_audio(mp3_path, sr=44100):
    wav_path = mp3_to_wav(mp3_path)
    rate, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]  # mono
    data = data.astype(np.float64) / 32768.0
    return rate, data

def find_silence_segments(data, sr, threshold_db=-35, min_duration=0.5):
    """找静音段（用于估计噪声）"""
    frame_len = int(sr * 0.02)  # 20ms frames
    hop = frame_len // 2
    rms = []
    for i in range(0, len(data) - frame_len, hop):
        frame = data[i:i+frame_len]
        rms.append(np.sqrt(np.mean(frame**2) + 1e-10))
    rms = np.array(rms)
    rms_db = 20 * np.log10(rms + 1e-10)
    
    # 找连续低于阈值的段
    silent = rms_db < threshold_db
    segments = []
    start = None
    for i, s in enumerate(silent):
        if s and start is None:
            start = i
        elif not s and start is not None:
            dur = (i - start) * hop / sr
            if dur >= min_duration:
                segments.append((start * hop, i * hop))
            start = None
    return segments

def compute_spectrum_stats(data, sr, nperseg=4096):
    """计算功率谱密度"""
    freqs, psd = welch(data, sr, nperseg=nperseg, noverlap=nperseg//2)
    return freqs, psd

def compute_spectrogram(data, sr, nperseg=2048, hop=512):
    """计算时频谱"""
    from scipy.signal import stft
    freqs, times, Zxx = stft(data, sr, nperseg=nperseg, noverlap=nperseg-hop)
    return freqs, times, np.abs(Zxx)

def analyze_file(mp3_path, segment_start=0, segment_duration=10):
    """分析单个文件"""
    print(f"\n{'='*60}")
    print(f"分析: {os.path.basename(mp3_path)}")
    print(f"{'='*60}")
    
    sr, data = load_audio(mp3_path)
    print(f"采样率: {sr}, 时长: {len(data)/sr:.1f}s")
    
    # 取一段分析
    start_sample = int(segment_start * sr)
    end_sample = int((segment_start + segment_duration) * sr)
    segment = data[start_sample:min(end_sample, len(data))]
    
    # 找静音段
    silence_segs = find_silence_segments(segment, sr)
    
    if silence_segs:
        # 取最长的静音段做噪声估计
        longest = max(silence_segs, key=lambda x: x[1] - x[0])
        noise_segment = segment[longest[0]:longest[1]]
        print(f"噪声估计段: {longest[0]/sr:.2f}s - {longest[1]/sr:.2f}s (长度: {(longest[1]-longest[0])/sr:.2f}s)")
    else:
        # 没有明显静音段，取能量最低的10%
        print("未找到明显静音段，取能量最低段估计噪声")
        frame_len = int(sr * 0.1)
        hop = frame_len // 4
        min_energy = float('inf')
        min_idx = 0
        for i in range(0, len(segment) - frame_len, hop):
            energy = np.sum(segment[i:i+frame_len]**2)
            if energy < min_energy:
                min_energy = energy
                min_idx = i
        noise_segment = segment[min_idx:min_idx+frame_len]
    
    # 乐音段：取能量最高的部分
    frame_len = int(sr * 1.0)
    hop = frame_len // 2
    max_energy = 0
    max_idx = 0
    for i in range(0, len(segment) - frame_len, hop):
        energy = np.sum(segment[i:i+frame_len]**2)
        if energy > max_energy:
            max_energy = energy
            max_idx = i
    music_segment = segment[max_idx:max_idx+frame_len]
    
    # 计算频谱
    freqs_noise, psd_noise = compute_spectrum_stats(noise_segment, sr)
    freqs_music, psd_music = compute_spectrum_stats(music_segment, sr)
    
    # 转dB
    psd_noise_db = 10 * np.log10(psd_noise + 1e-20)
    psd_music_db = 10 * np.log10(psd_music + 1e-20)
    
    # 计算信噪比随频率的变化
    snr_per_freq = psd_music_db - psd_noise_db
    
    # 按频段统计
    bands = [
        ("低频", 20, 200),
        ("中低频", 200, 1000),
        ("中频", 1000, 4000),
        ("中高频", 4000, 8000),
        ("高频", 8000, 16000),
        ("超高频", 16000, 22050),
    ]
    
    print(f"\n频段能量分布:")
    print(f"{'频段':<10} {'噪声功率(dB)':<15} {'乐音功率(dB)':<15} {'SNR(dB)':<10} {'噪声占比%':<10}")
    
    noise_total_power = np.sum(psd_noise)
    music_total_power = np.sum(psd_music)
    
    noise_band_powers = []
    
    for name, low, high in bands:
        mask = (freqs_noise >= low) & (freqs_noise < high)
        if mask.sum() == 0:
            continue
        noise_p = np.sum(psd_noise[mask])
        music_p = np.sum(psd_music[mask])
        noise_db = 10 * np.log10(noise_p + 1e-20)
        music_db = 10 * np.log10(music_p + 1e-20)
        snr = music_db - noise_db
        noise_pct = noise_p / noise_total_power * 100
        noise_band_powers.append((name, noise_pct, snr))
        print(f"{name:<10} {noise_db:<15.1f} {music_db:<15.1f} {snr:<10.1f} {noise_pct:<10.1f}")
    
    # 找噪声集中的频段（SNR最低的频段）
    print(f"\n噪声集中频段（SNR最低）:")
    sorted_bands = sorted(noise_band_powers, key=lambda x: x[2])
    for name, pct, snr in sorted_bands:
        print(f"  {name}: 噪声占总噪声{pct:.1f}%, SNR={snr:.1f}dB")
    
    return {
        "freqs": freqs_noise.tolist(),
        "psd_noise_db": psd_noise_db.tolist(),
        "psd_music_db": psd_music_db.tolist(),
        "snr_per_freq": snr_per_freq.tolist(),
    }

def main():
    raw_dir = "/home/exusiai/_dev/Hankel-Stereo-Purify/data/raw/Brendel_Beethoven_Piano_Music_Vol9"
    files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.mp3')])
    
    # 分析前3个文件作为样本
    results = {}
    for f in files[:3]:
        path = os.path.join(raw_dir, f)
        try:
            results[f] = analyze_file(path, segment_start=5, segment_duration=15)
        except Exception as e:
            print(f"Error analyzing {f}: {e}")
    
    # 保存结果
    out_path = "/home/exusiai/_dev/Hankel-Stereo-Purify/data/noise_spectrum_analysis.json"
    with open(out_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f"\n结果保存到: {out_path}")

if __name__ == "__main__":
    main()
