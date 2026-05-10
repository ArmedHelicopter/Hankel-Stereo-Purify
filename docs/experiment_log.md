# 实验日志 (Experiment Log)

本文件记录 Hankel-Stereo-Purify 的降噪实验过程、结果和决策。按时间倒序排列。

---

## 2026-05-10: BPW 默认策略切换

### 决策

将默认 pipeline 从全频 MSSA 切到带通白化候选默认：

```text
bypass_freq = 2000Hz
highband_whiten = True
whiten_alpha = 0.75
```

原因：

- 四路实验中，全频 MSSA 的 residual 可听到乐音，说明音乐损伤更明显。
- 裸带通基本接近原始，但高频残留比例仍高，不能有效压 hiss。
- 带通白化的 diff 主要是噪声，roundtrip 对照继续排除 STFT 自身偷偷降噪。
- Alpha sweep 与 Gemini 初筛都提示 `a75` 比 `a100` 更可能保留钢琴尾音和高频空气感。

### 接口

- CLI 默认启用 `--bypass-freq 2000 --highband-whiten --whiten-alpha 0.75`。
- `--fullband` 恢复旧的全频 MSSA。
- `--no-highband-whiten` 保留带通，但关闭白化，得到裸 `bp`。
- API 默认与 CLI 对齐；显式 `bypass_freq=None` 可恢复全频路径。

### 风险

`alpha=0.75` 仍是单片段候选默认，不是最终定型。下一步细扫输出放在 `data/processed/a09b/`，测试 `a75/a80/a85/a90/a95/a100`，再结合 Gemini listwise 与人工复听确认是否需要调整默认。

### 快速细扫结果

10s 全量细扫运行时间过长，改为 3s 快扫以便当晚收工：

- 输出：`data/processed/a09b/`
- 时长：3s
- 候选：`a75/a80/a85/a90/a95/a100`

| Variant | alpha | high/orig | diff RMS | SNR |
|---------|-------|-----------|----------|-----|
| a75 | 0.75 | 0.2039 | 1.9257e-04 | 30.70dB |
| a80 | 0.80 | 0.1221 | 2.0851e-04 | 30.00dB |
| a85 | 0.85 | 0.0833 | 2.1711e-04 | 29.65dB |
| a90 | 0.90 | 0.0679 | 2.2165e-04 | 29.47dB |
| a95 | 0.95 | 0.0626 | 2.2401e-04 | 29.38dB |
| a100 | 1.00 | 0.0608 | 2.2523e-04 | 29.33dB |

解释：

- `a75 -> a80` 的高频残留下降最明显。
- `a85` 以后继续压低 high/orig 的收益快速变小，但 diff RMS 持续增加。
- 数值上 `a80/a85` 是新的重点听感候选；代码默认仍暂定 `a75`，因为它的改动最小，且此前 Gemini 偏好 `a75`。下一步应听 `a75/a80/a85`，而不是继续只比较 `a75/a100`。

---

## 2026-05-10: Alpha Sweep 09 - 白化强度扫描

### 目的

继续优化 `bpw`，测试白化强度是否需要从当前 `alpha=1.0` 降低。新公式：

```text
whiten:   Z_white = Z / profile^alpha
unwhiten: Z_back  = Z_processed * profile^alpha
```

其中：

- `alpha=0`：不做频谱白化，接近裸带通。
- `alpha=1`：当前 `bpw`。
- 中间值：弱白化，目标是在“去噪强度”和“听感自然”之间找折中。

### 输出

- 目录：`data/processed/a09/`
- 主文件：`orig.wav`, `a0.wav`, `a25.wav`, `a50.wav`, `a75.wav`, `a100.wav`
- Diff：`data/processed/a09/d/orig_a*.wav`
- 指标：`data/processed/a09/sum.json`, `data/processed/a09/sum.md`
- A/B：`data/processed/a09/ab/a50_a75/`, `a75_a100/`, `a50_a100/`

### 数值结果

| Variant | alpha | high/orig | diff RMS | SNR |
|---------|-------|-----------|----------|-----|
| orig | - | 1.000 | 0 | inf |
| a0 | 0.00 | 0.9276 | 6.2627e-05 | 40.59dB |
| a25 | 0.25 | 0.9320 | 6.2231e-05 | 40.64dB |
| a50 | 0.50 | 0.9282 | 5.2718e-05 | 42.08dB |
| a75 | 0.75 | 0.2479 | 2.1247e-04 | 29.98dB |
| a100 | 1.00 | 0.0559 | 2.6027e-04 | 28.21dB |

所有主输出校验为 10.000s、44100Hz、stereo、finite、WAV FLOAT。

### 初步解释

- `a0/a25/a50` 的 high/orig 都约为 0.93，说明弱白化几乎没有真正压掉高频残留。
- `a75` 是明显拐点：high/orig 从约 0.93 降到 0.248，开始有效处理高频噪声。
- `a100` 最强：high/orig 进一步降到 0.0559，但 diff RMS 也最大。
- 下一步应重点盲听 `a50_a75` 和 `a75_a100`：如果 `a75` 已经足够安静且更自然，它可能比 `a100` 更稳；如果 `a75` 仍留 hiss，继续保留 `a100` 作为当前最佳。

### A/B 复听记录

`data/processed/a09/ab/a50_a75/`：

- A 明显噪声。
- B 噪声不明显，但听感奇怪。
- 解码：A = `a50`，B = `a75`。

`data/processed/a09/ab/a50_a100/`：

- A 无噪声。
- B 有噪声。
- 解码：A = `a100`，B = `a50`。

`data/processed/a09/ab/a75_a100/`：

- A/B 均无噪声。
- 听起来没有特别大差异。
- 解码：A = `a100`，B = `a75`。

解释：

- `a50` 基本可以排除：数值上 high/orig 仍为 0.928，听感上也稳定留噪。
- `a75` 已进入有效降噪区间，但在 `a50_a75` 中出现“听感奇怪”；这可能是刚刚跨过压噪拐点后，高频残留比例变化带来的质感不自然。
- `a100` 和 `a75` 在直接 A/B 中差异不大，且两者都无明显噪声；当前不能仅凭这一轮判断 `a100` 过强。
- 由于此前 `bpw` = `a100` 在四路实验中“噪声不明显且听感正常”，当前默认推荐仍保持 `alpha=1.0`。但 `a75` 值得作为候选继续做更细 sweep。

### Gemini 批量听感初筛

调用方式：

- Transport：OpenAI-compatible chat/audio，经 `GOOGLE_GEMINI_BASE_URL`
- Model：`gemini-2.5-pro-1m`
- 输出：`data/processed/a09/gemini_eval_openai.json`
- 可复用脚本复跑输出：`data/processed/a09/gemini_eval_script.json`

注意：配置中的 `gemini-3.1-pro` 在当前分组下无可用渠道；`/v1/models` 显示可用 Gemini 模型为 `gemini-2.5-flash`、`gemini-2.5-pro-1m`。官方 `google-genai` 文件上传路径在该中转上返回 404，因此本次改用 OpenAI-compatible `input_audio` base64 消息。

匿名映射：

- A = `a100.wav`
- B = `a75.wav`
- C = `a50.wav`

Gemini 返回排名：

```text
B > C > A
```

分数：

| Variant | hiss_noise | musical_damage | unnatural_artifacts | overall_quality |
|---------|------------|----------------|---------------------|-----------------|
| a50 | 7 | 1 | 0 | 5 |
| a75 | 3 | 2 | 3 | 8 |
| a100 | 1 | 7 | 8 | 2 |

解释：

- Gemini 判断 `a50` 留噪明显，但音乐损伤低。
- Gemini 判断 `a100` 最安静，但音乐损伤和伪影最重。
- Gemini 选择 `a75` 为整体最佳，这与人工听感中“a75/a100 均无噪声但差异不大、a75 有一点奇怪”的记录并不完全冲突：Gemini 更强烈惩罚了 `a100` 的潜在音乐损伤/伪影。
- 这提示下一步应围绕 `0.75-1.0` 做细扫，而不是直接把 `1.0` 固化为最终值。

脚本化复跑结果：

- 匿名映射：A = `a50.wav`, B = `a75.wav`, C = `a100.wav`
- 排名：`B > A > C`
- 结论仍一致：`a75` 最优；`a50` 更透明但留噪，`a100` 最安静但过度损伤高频空气感/尾音，并引入更明显的不自然伪影。

### 决策

当时保留 `--whiten-alpha` 实验参数，默认仍为 `1.0`。后续快速细扫与 BPW 默认策略切换已将当前候选默认更新为 `0.75`；下一步重点听 `a75/a80/a85`。

---

## 2026-05-10: 四路对照实验 - 全频 MSSA vs 裸带通 vs 带通白化

### 目的

验证两个问题：

1. 裸 `--bypass-freq 2000` 是否会比白化带通保留更多高频噪声。
2. 当前 whitening 结果是否需要放回公平对照中，与全频 MSSA、裸带通、原始音频一起比较。

同时修正一个实验精度问题：高频分支临时 WAV 现在统一写 `WAV FLOAT`，包括普通裸带通路径和 whitening artifact baseline。此前普通路径使用默认 WAV 写出，可能经过 PCM16 量化；对极低能量 high-band 来说这会引入不该有的量化误差。

### 测试材料与参数

- 原始文件：`data/raw/Brendel_Beethoven_Piano_Music_Vol9/09_Op27_No2_I_Adagio_sostenuto.mp3`
- 切片：0s 起始，10s，44100Hz，stereo
- 输出目录：`data/processed/four_way_bandpass_whiten_09_adagio_10s/`
- 短入口：`data/processed/fw09/`
- MSSA 参数：`-L 256 --energy-fraction 0.9 --frame-size 1024`
- 带通 cutoff：`--bypass-freq 2000`

### 命令

```bash
python scripts/run_four_way_bandpass_whiten_experiment.py
```

### 四路输出

- `data/processed/four_way_bandpass_whiten_09_adagio_10s/original.wav`
- `data/processed/four_way_bandpass_whiten_09_adagio_10s/fullband_energy_mssa.wav`
- `data/processed/four_way_bandpass_whiten_09_adagio_10s/bandpass_no_whiten.wav`
- `data/processed/four_way_bandpass_whiten_09_adagio_10s/bandpass_whiten.wav`

短名等价入口：

- `data/processed/fw09/orig.wav`
- `data/processed/fw09/full.wav`
- `data/processed/fw09/bp.wav`
- `data/processed/fw09/bpw.wav`
- `data/processed/fw09/sum.json`

所有主输出与 residual 均校验为 10.000s、44100Hz、stereo、finite、WAV FLOAT。

### 数值结果

| Variant | high RMS | high/original | diff RMS | SNR vs original |
|---------|----------|---------------|----------|-----------------|
| original | 2.6236e-04 | 1.000 | 0 | inf |
| fullband_energy_mssa | 2.9551e-04 | 1.126 | 1.4233e-03 | 13.46dB |
| bandpass_no_whiten | 2.4337e-04 | 0.928 | 6.2627e-05 | 40.59dB |
| bandpass_whiten | 1.4659e-05 | 0.0559 | 2.6027e-04 | 28.21dB |

Residual：

- `diff_original_vs_fullband_energy_mssa.wav`：RMS=1.4233e-03
- `diff_original_vs_bandpass_no_whiten.wav`：RMS=6.2627e-05
- `diff_original_vs_bandpass_whiten.wav`：RMS=2.6027e-04
- `diff_bandpass_no_whiten_vs_bandpass_whiten.wav`：RMS=2.4082e-04

Whitening roundtrip 继续成立：

- `roundtrip_diff_rms` = 3.5401e-20
- `roundtrip_snr_db` = inf

### 盲听包

已生成两组 RMS 常数增益匹配 A/B，不做 EQ、压缩、限制器或滤波：

- `data/processed/four_way_bandpass_whiten_09_adagio_10s/listening_eval/ab_fullband_vs_bandpass_whiten/`
- `data/processed/four_way_bandpass_whiten_09_adagio_10s/listening_eval/ab_bandpass_no_whiten_vs_bandpass_whiten/`

听前不要打开各目录下的 `answer_key.json`，也不要先看 `summary.json` 中的 blind mapping。

### 解释

这轮结果说明：裸带通在这个 10s Adagio 切片上确实比白化带通保留了更多 2kHz 以上残留。裸带通的 high RMS 仍有原始高频的 92.8%，而白化带通只剩 5.59%。

但这不等于“带通滤波器制造或放大噪声”。更准确的解释是：

```text
裸带通 = low_band 直接旁路 + high_band 送入 MSSA
```

低频本来就绕过处理；高频分支如果不白化，MSSA/SVD 看到的能量结构仍不利于截断，噪声没有被充分推到低能量方向，所以输出更接近原始 high_band，也就保留了更多 hiss。

白化带通则先把高频噪声谱做固定尺度重映射，使 MSSA 更容易把宽带噪声当成可截断的低能量成分。由于 roundtrip diff 仍在数值误差级别，当前证据继续支持：白化本身近似可逆，主要贡献来自白化后 MSSA 的截断行为，而不是 STFT 预处理偷偷降噪。

全频 MSSA 的 high RMS 高于原始值，且 `original - fullband` diff RMS 最大。它可能在全频上引入了更大的整体改动，不能只靠 high RMS 判断为更好；需要结合 residual 与盲听继续判断音乐损伤程度。

### Diff 复听记录

人工复听 residual：

- `data/processed/fw09/d/orig_full.wav`：能听到明显乐音和噪音。
- `data/processed/fw09/d/orig_bp.wav`：几乎听不到任何声音。
- `data/processed/fw09/d/orig_bpw.wav`：主要只能听到噪音，没有明显乐音。
- `data/processed/fw09/d/bp_bpw.wav`：主要只能听到噪音，没有明显乐音。

解释：

- `orig_full` 有明显乐音，说明全频 MSSA 在这个参数下不只是去噪，也改掉了可听音乐结构；这与它较大的 diff RMS 一致。
- `orig_bp` 几乎无声，说明裸带通输出几乎等于原始音频。它的音乐保真很好，但也意味着它没有明显移除高频 hiss；这与 high RMS 仍有原始高频 92.8% 一致。
- `orig_bpw` 和 `bp_bpw` 主要是噪音，说明白化带通相对原始/裸带通主要移除的是噪声残留，而不是稳定乐音结构。

### A/B 复听记录

`data/processed/fw09/ab/full_bpw/`：

- A 噪声明显。
- B 噪声不明显。
- 解码：A = `fullband_energy_mssa`，B = `bandpass_whiten`。

解释：

- 这不说明白化方向错了，反而支持 `bandpass_whiten` 优于全频 MSSA。
- 这组 A/B 只做常数 RMS 增益匹配，不做 EQ、压缩、限制器或滤波；因此听到的噪声差异不应解释成简单音量错觉。
- 结合 diff 复听：`orig_full` 有乐音残留，而 `orig_bpw` 主要是噪音，说明全频 MSSA 既有噪声/伪影问题，也有音乐损伤；白化带通目前更符合“主要移除噪声”的目标。

`data/processed/fw09/ab/bp_bpw/`：

- A 无明显噪声，但是听感诡异。
- B 无明显噪声，且听感正常。
- 解码：A = `bandpass_no_whiten`，B = `bandpass_whiten`。

解释：

- 这说明裸带通虽然在这组 RMS 匹配盲听中不显噪，但听感自然度不如白化带通。
- 由于 `orig_bp` 几乎无声，裸带通本质上更接近原始；如果 A 仍显得诡异，可能来自高频分支 OLA/MSSA 对极小 high-band 的轻微相位/瞬态扰动，或残留 hiss 与乐音高频的比例不自然。
- B 同时满足“噪声不明显”和“听感正常”，因此当前主观结果继续支持 `bandpass_whiten`，但后续需要用更多片段确认它不是单片段偶然。

### 决策

1. 修正高频临时 WAV 精度问题后，保留 `WAV FLOAT` 作为高频临时文件默认实验路径。
2. 裸带通不再作为“已经充分降噪”的 baseline 使用；它更像是“只保护低频、但高频截断仍不足”的对照。
3. 下一步优先听两组四路盲听包，再决定是否推进 cutoff sweep / whitening alpha sweep / 多片段批量复现。

---

## 2026-05-10: 可逆高频噪声谱白化 + Roundtrip 对照

### 目的

验证一个新的高频分支预条件化假设：

> 在 `--bypass-freq` 带通分支中，先对 `high_band` 做可逆噪声谱白化，再送入 MSSA/SVD，最后反白化。白化本身不应直接降噪；真正的改善应来自它改变了 MSSA 看到的能量分布。

这轮实验专门加入 `roundtrip` 对照，用来排除“STFT 白化本身偷偷降噪”的悖论。

### 测试材料

- 原始文件：`data/raw/Brendel_Beethoven_Piano_Music_Vol9/09_Op27_No2_I_Adagio_sostenuto.mp3`
- 正式测试切片：10s, 44100Hz, stereo, float WAV
- 输出目录：`data/processed/whiten_exp_09_adagio_10s/`

### 命令

```bash
ffmpeg -y \
  -i data/raw/Brendel_Beethoven_Piano_Music_Vol9/09_Op27_No2_I_Adagio_sostenuto.mp3 \
  -t 10 -ar 44100 -c:a pcm_f32le \
  data/processed/whiten_exp_09_adagio_10s/input_10s.wav

python -m src.cli \
  data/processed/whiten_exp_09_adagio_10s/input_10s.wav \
  data/processed/whiten_exp_09_adagio_10s/output_whiten.wav \
  -L 256 \
  --energy-fraction 0.9 \
  --frame-size 1024 \
  --bypass-freq 2000 \
  --highband-whiten \
  --whiten-artifact-dir data/processed/whiten_exp_09_adagio_10s/artifacts
```

### Pipeline

当前白化实验只作用在带通分支：

```text
original
  -> split_signal(cutoff=2000Hz)
       low_band:  <2kHz bypass, 不进 SVD
       high_band: >2kHz residual, 进入实验分支

high_band
  -> STFT
  -> 按频率估计 profile(f)
  -> whiten:   Z(f,t) / profile(f)
  -> ISTFT 得到 whitened_high
  -> OLA + MSSA/SVD
  -> STFT
  -> unwhiten: Z_processed(f,t) * profile(f)
  -> ISTFT 得到 processed_high

final = low_band + processed_high
```

### 白化与反白化原理

白化不是降噪器，而是一个可逆的频率尺度变换。

先对高频信号做 STFT：

```text
high_band -> Z(f,t)
```

从全文件高频 STFT 幅度中，按频率估计一个固定 profile：

```text
profile(f) = percentile_20(|Z(f,t)|), 左右声道合并估计
```

白化：

```text
Z_white(f,t) = Z(f,t) / max(profile(f), eps)
```

反白化：

```text
Z_back(f,t) = Z_white(f,t) * profile(f)
```

如果中间不经过 MSSA，二者相乘相除抵消：

```text
Z_back(f,t) = Z(f,t) / profile(f) * profile(f)
            = Z(f,t)
```

所以理论上：

```text
unwhiten(whiten(high_band)) ~= high_band
```

再加回低频：

```text
roundtrip = low_band + unwhiten(whiten(high_band))
original  = low_band + high_band
```

因此：

```text
original - roundtrip ~= 0
```

这就是 `diff_original_vs_roundtrip.wav` 的意义。如果它没有声音，就说明 STFT 白化/反白化本身几乎没有改音频，也没有自己做降噪。

### Roundtrip 对照为什么重要

如果没有 roundtrip，对 whitening 输出变好会有两种解释：

1. STFT 白化本身就是一个隐藏降噪器。
2. STFT 白化只是改变 MSSA/SVD 的输入坐标，让 SVD 截断更符合“乐声高能量、噪声低能量”的先验。

roundtrip 用来区分这两种情况。

本轮结果中：

```text
roundtrip = low_band + unwhiten(whiten(high_band))
diff_original_vs_roundtrip = original - roundtrip
```

`diff_original_vs_roundtrip.wav` 完全无可闻声音，且数值误差极低。这说明白化 + 反白化本身近似恒等变换。也就是说，改善来源不应归因于 STFT 预处理直接降噪，而应归因于 MSSA/SVD 在白化坐标下做了不同截断。

### 实验产物

主输出：

- `data/processed/whiten_exp_09_adagio_10s/input_10s.wav`
- `data/processed/whiten_exp_09_adagio_10s/output_whiten.wav`

Artifact：

- `data/processed/whiten_exp_09_adagio_10s/artifacts/roundtrip.wav`
- `data/processed/whiten_exp_09_adagio_10s/artifacts/baseline_no_whiten.wav`
- `data/processed/whiten_exp_09_adagio_10s/artifacts/whitened_output.wav`
- `data/processed/whiten_exp_09_adagio_10s/artifacts/diff_baseline_vs_whiten.wav`
- `data/processed/whiten_exp_09_adagio_10s/artifacts/diff_original_vs_whiten.wav`
- `data/processed/whiten_exp_09_adagio_10s/artifacts/diff_original_vs_roundtrip.wav`
- `data/processed/whiten_exp_09_adagio_10s/artifacts/metrics.json`

### 数值结果

| 指标 | 值 |
|------|-----|
| samplerate | 44100 |
| bypass_freq | 2000Hz |
| STFT nperseg | 2048 |
| STFT noverlap | 1024 |
| profile percentile | 20 |
| profile min | 1e-08 |
| profile median | 2.8286e-06 |
| profile max | 4.4255e-06 |
| roundtrip_diff_rms | 3.5401e-20 |
| roundtrip_snr_db | Infinity |
| baseline_vs_whiten_diff_rms | 2.4140e-04 |
| baseline_vs_whiten_snr_db | 28.87dB |
| original_vs_whiten_diff_rms | 2.6027e-04 |
| original_vs_whiten_snr_db | 28.21dB |

### 听感观察

- `diff_original_vs_roundtrip.wav`：完全无声。
- 三个 diff 文件均未听到明确乐声残留。
- `baseline_no_whiten.wav` 中仍有明显噪声和乐声。
- `whitened_output.wav` 相比 baseline 的差异主要不像是乐声成分。

### 归一化 Diff 复听与 A/B 盲听包

已生成复听与盲听产物：

- `data/processed/whiten_exp_09_adagio_10s/listening_eval/summary.json`
- `data/processed/whiten_exp_09_adagio_10s/listening_eval/diff_norm/diff_baseline_vs_whiten_peak_norm.wav`
- `data/processed/whiten_exp_09_adagio_10s/listening_eval/diff_norm/diff_baseline_vs_whiten_rms_norm.wav`
- `data/processed/whiten_exp_09_adagio_10s/listening_eval/diff_norm/diff_original_vs_whiten_peak_norm.wav`
- `data/processed/whiten_exp_09_adagio_10s/listening_eval/diff_norm/diff_original_vs_whiten_rms_norm.wav`
- `data/processed/whiten_exp_09_adagio_10s/listening_eval/diff_norm/diff_original_vs_roundtrip_unmodified.wav`
- `data/processed/whiten_exp_09_adagio_10s/listening_eval/ab_blind/A.wav`
- `data/processed/whiten_exp_09_adagio_10s/listening_eval/ab_blind/B.wav`
- `data/processed/whiten_exp_09_adagio_10s/listening_eval/ab_blind/score_sheet.md`
- `data/processed/whiten_exp_09_adagio_10s/listening_eval/ab_blind/answer_key.json`

归一化策略：

- Peak norm：峰值归一化到 -1dBFS。
- RMS norm：全文件 RMS 归一化到 -38dBFS，并保留 -1dBFS 峰值天花板；本轮未触发峰值天花板。
- Roundtrip diff：RMS=3.5401e-20、peak=8.6736e-19，低于数值阈值；不做强行归一化，只复制为 unmodified 证据文件。
- A/B：`baseline_no_whiten.wav` 与 `whitened_output.wav` 只做常数增益 RMS 匹配，不做 EQ、压缩、限制器或滤波。

人工复听结果：

- 归一化 diff 复听：除 roundtrip 外均无明显乐声，噪声明显；peak norm 版本听感非常刺耳，符合高频噪声 residual 被大幅放大的预期。
- A/B 盲听：A 噪声明显，B 噪声不明显。
- 解码：A = `baseline_no_whiten.wav`，B = `whitened_output.wav`。
- 该结果进一步支持“白化改善 MSSA 截断，而不是 STFT 自身降噪”：roundtrip 不改变音频，diff 主要是噪声，盲听中 whitening 输出噪声更低。

### 结论

这轮实验强支持以下判断：

1. **Roundtrip 成立**：白化 + 反白化本身近似可逆，不直接降噪，不直接改变音乐。
2. **STFT 悖论被排除**：`diff_original_vs_roundtrip.wav` 无声，说明 STFT 预条件化本身不是主要降噪来源。
3. **白化方向有效**：白化改变了 MSSA/SVD 的输入能量结构，使 SVD 截断更接近预期先验：噪声进入低能量部分，乐声结构保留在高能量部分。
4. **归一化 diff + A/B 均支持 whitening**：放大 residual 后未听到明显乐声，盲听中 `whitened_output.wav` 噪声不明显，而 `baseline_no_whiten.wav` 噪声明显。
5. **仍需复现**：下一步应在更多曲目、不同 cutoff 和不同片段上复现实验。

### 当时决策

本实验完成时，`--highband-whiten` 仍作为默认关闭的实验开关。后续四路对照、alpha sweep 和快速细扫已将该判断推进为当前默认 BPW 策略（见本文顶部“BPW 默认策略切换”）。

保留的实验规范：

1. 白化实验必须有 `roundtrip` 对照。
2. 输出放在 `data/processed/...`。
3. 同时保存 `roundtrip`、`baseline_no_whiten`、`whitened_output` 和 diff artifacts。
4. 先检查 `diff_original_vs_roundtrip.wav`，确认预处理可逆，再评价 MSSA 降噪贡献。

---

## 2026-05-10: 噪声频段分布分析（Gemini音频模态验证）

### 目的
确定钢琴/小提琴录音中噪声的频段分布，验证预处理带通滤波策略的可行性。

### 方法
用 Gemini 2.5 Flash（原生音频模态）直接"听"录音片段，分析噪声类型和频段分布。对比传统 SNR 能量指标与 AI 感知判断的差异。

### 测试材料
- Brendel Beethoven 钢琴奏鸣曲（1980s数字录音，44100Hz MP3）
- RCA Living Stereo Heifetz 小提琴协奏曲（1960s模拟录音，DSD64转PCM）

### 结果

**Brendel 钢琴（3个片段一致）：**
- 噪声类型：宽带磁带嘶声
- 噪声频段：2kHz-15kHz+，4-5kHz以上最明显
- 干净频段：<1-2kHz

**RCA Living Stereo 小提琴（3个录音一致）：**
- 贝多芬：嘶声 2.5kHz-18kHz，干净 80Hz-2.5kHz
- 西贝柳斯：嘶声 3kHz-10kHz，干净 150Hz-2kHz
- 勃拉姆斯：嘶声 3kHz-15kHz+，干净 <2kHz
- 低频有轻微隆隆声（20-150Hz），能量极低

**关键发现：SNR能量指标与AI感知判断矛盾。**
SNR分析显示噪声集中在中低频（200-1kHz），但Gemini听后确认该频段全是干净乐音。高频SNR≈0dB不是因为噪声大，而是钢琴/小提琴在该频段本身能量低。SNR是能量比值，不等于感知噪声。

### 结论
不同年代、不同乐器的录音，噪声分布模式一致：**高频嘶声（2-3kHz以上），低中频干净。**

### 决策
采用带通滤波预处理策略：
1. **低通bypass**：<2kHz 直接跳过SVD，零处理损失
2. **高通截取**：>2kHz 送入 Hankel+SVD 处理
3. **高频段内能量重排**：让噪声→低能量、乐音泛音→高能量（后续实现）

### 输出文件
`data/noise_spectrum_analysis.json`（SNR分析数据）

---


## 2026-05-08: svds (partial) vs svd (full) 对比

### 目的

Wiener 软加权当前用全量 `scipy.linalg.svd`（O(mn·min(m,n))），比能量截断的 `svds`（O(mn·k)）慢 ~10x。测试能否用 svds 近似 full SVD 的 Wiener 效果。

### 方法

单帧 Wiener 对比：
1. Full SVD → 全部奇异值取尾部估噪 → Wiener 权重（ground truth）
2. svds(top-k) → 用残差能量均摊估噪 → Wiener 权重（近似）

### 测试条件

256×1538 矩阵（L=256, F=1024, stereo），noise_fraction=0.1

### 结果

**单帧 svds 近似质量：**

| k_probe | vs full SNR | 耗时 | 加速比 |
|---------|------------|------|--------|
| 8 | 20.8dB | 0.054s | 25.7x |
| 16 | 29.6dB | 0.086s | 16.2x |
| 32 | 30.4dB | 0.165s | 8.5x |
| 64 | 31.7dB | 0.344s | 4.1x |
| 128 | 34.2dB | 0.582s | 2.4x |
| full | — | 1.397s | 1x |

**全管道对比（1s 音频, 33 frames, Beethoven Adagio）：**

| 方法 | SNR vs 原始 | 速度 |
|------|------------|------|
| 能量截断（svds partial） | 15.9dB | 12 f/s |
| Wiener（full SVD） | **44.9dB** | 1 f/s |

### 分析

svds 近似质量不足：k=64 时 vs full 只有 31.7dB。原因是 svds 丢弃的尾部奇异值被均摊到噪声估计，实际噪声在各分量上分布不均匀。

Full SVD 是 Wiener 的正确路径（44.9dB），瓶颈在每帧 1.4s。33 帧→37s。加速需改算法而非换 SVD 后端。

### 决策

两个优化方向：
1. **CPU 向量化**：利用 Hankel 矩阵的 Toeplitz 结构，用 FFT 替代部分 SVD 计算
2. **CUDA**：GPU 并行化 SVD + Wiener 权重计算

详见下一节讨论。

---

## 2026-05-08: Wiener 软加权

### 假设

能量截断按奇异值做 0/1 决策（保留/丢弃），把弱谐波和噪声一起扔掉。用连续权重替代硬截断可以保留弱信号。

### 方法

Wiener 软加权：对每个 SVD 分量赋予连续权重

```
w_i = max(0, 1 - σ_noise² / σ_i²)
```

噪声方差从底部 `noise_fraction` 的奇异值估计。

### 测试条件

Beethoven Op.27 No.2 Adagio, 44100Hz stereo, F=1024 L=256 hop=512

### 结果（1s clip, ~33 frames）

| 方法 | SNR vs 原始 | 相对能量baseline | 每帧耗时 |
|------|------------|-----------------|---------|
| 能量截断 (0.9) | 16.0dB | — | 0.085s |
| Wiener nf=0.1 | **43.9dB** | +27.9dB | 0.180s |
| Wiener nf=0.2 | **40.5dB** | +24.5dB | 0.216s |

- Wiener 保留 105.8% 的能量 baseline RMS（比硬截断保留更多信号）
- 弱谐波被连续衰减而非完全丢弃
- 代价：全量 SVD ~2x 慢于部分 SVD

### 决策

Wiener 软加权显著优于硬截断。后续需优化 SVD 计算速度（`svds` 部分分解或 CUDA）。

### 输出文件

`data/processed/wiener_test/`

---

## 2026-05-08: W-correlation 修复与实验

### 问题

W-correlation 修复前所有阈值 SNR 均为 ~5.5dB（vs baseline 13.5dB）。

### 根因分析

三个独立 bug 叠加：

1. **权重公式错误**：`grouping.py` 内积计算用 `weighted @ arr.T`（加权×非加权），正确应为 `weighted @ weighted.T`。
2. **分组逻辑缺陷**：旧逻辑仅保留与第一个分量相关性高的分量（`w_mat[1:, 0] >= thr`）。SVD 分量正交，钢琴各谐波在不同分量中，W-correlation 低→只保留基频分量，谐波全部被误删。
3. **冻结状态导致帧间不一致**：`energy_w_corr_frozen=True` 锁死第一帧的分组关系。当后续帧的截断秩 k 变化时，冻结索引无法适配新分量（如 frame 0 时 k=5，冻结 [0-4]；后续帧 k=6 时第 6 个分量被无条件丢弃）。

### 修复方案

- 权重公式：改为 `weighted @ weighted.T`
- 分组逻辑：层次聚类（average linkage）替代锚点分组；保留所有能量 ≥5% 总能量的聚类
- 冻结状态：移除，每帧重算聚类

### 结果（5s clip）

| 阈值 | SNR vs 原始 | SNR vs baseline | diff% |
|------|------------|----------------|-------|
| baseline (纯能量) | 13.5dB | — | — |
| 0.1 | 13.4dB | 32.5dB | 2.4% |
| 0.3 | 13.1dB | 27.7dB | 4.1% |
| 0.5 | 13.0dB | 26.4dB | 4.8% |
| 0.7 | 12.9dB | 26.0dB | 5.0% |
| 0.9 | 12.8dB | 25.4dB | 5.4% |

### 结论

W-correlation 修复后从 5.5dB 恢复到 13.4dB，但仍**不如不加**（baseline 13.5dB）。根本原因：W-correlation 在能量截断之后做额外过滤，只会减少信号不会增加信号。

### 输出文件

`data/processed/wcorr_analysis/`

---

## 2026-05-08: OLA 重叠比实验

### 问题

sqrt-Hanning 窗在非 50% 重叠时产生幅度调制伪影（滋滋声）。

### 原因

sqrt-Hanning 窗完美重构条件：

$$\sum_k w^2(n - k \cdot hop) = C$$

只有 `hop=F`（无重叠）和 `hop=F/2`（50% 重叠）满足。其他重叠比导致窗函数叠加不恒定。

### 决策

程序锁死 `hop = frame_size // 2`，移除 `hop_size` 参数。

### 输出文件

`data/processed/energy_only_test/`

---

## 2026-05-08: 纯能量截断基线验证

### 测试条件

Beethoven Op.27 No.2 Adagio, 5s, F=1024 L=256 hop=512 energy=0.9

### 结果

| 指标 | 值 |
|------|-----|
| SNR | 13.6 dB |
| RMS diff / original | 20.9% |
| 处理速度 | 16.2 frames/s |

- diff 中可听到音乐成分（弱谐波被误删）
- 确认硬截断的信号丢失问题

### 输出文件

`data/processed/energy_only_test/`（original.mp3, energy.mp3, diff.mp3）
