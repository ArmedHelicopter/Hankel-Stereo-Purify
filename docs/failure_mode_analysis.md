# HSP 合成失败模式分析

本实验对应 [`20260511081500==z--hsp-next-steps.md`](20260511081500==z--hsp-next-steps.md) 中的 **P1-B：失败模式分析**。目标不是证明 BPW 一定更好，而是用可控合成信号找出默认 `BPW + MSSA` 的边界：什么时候保真，什么时候误删信号，什么时候声像或瞬态开始不稳定。

## 方法

入口脚本：

```bash
PYTHONPATH=. python scripts/run_failure_modes.py \
  --output-root data/processed/fm
```

常用参数：

```bash
--cases transient_attack,low_snr
--duration 0.75
--sample-rate 8000
--window-length 32
--frame-size 128
--energy-fraction 0.9
--bypass-freq 2000
--seed 20260511
```

第一版只使用合成数据，不依赖真实录音、`ffmpeg`、外部数据集或深度学习模型。每个 case 都生成：

- `clean.wav`：无噪参考信号。
- `noise.wav`：合成噪声。
- `input.wav`：`clean + noise`。
- `fullband_energy_mssa.wav`：旧全频 MSSA。
- `bandpass_no_whiten.wav`：2kHz 分频，低频 bypass，高频 MSSA，无白化。
- `bpw_default.wav`：当前默认 BPW，2kHz 分频 + 高频白化 + MSSA。
- `residual_<variant>.wav`：`input - output`，用于听被移除的内容。
- `clean_error_<variant>.wav`：`clean - output`，用于定位相对干净参考的误差。
- `summary.json` / `summary.md`：指标汇总。

根目录还会写 `summary.json` / `summary.md`，方便横向比较所有 case。

输出路径遵循项目实验规范：可复用产物必须写入
`data/processed/<short-name>/`，本实验的规范短名是 `data/processed/fm/`。
不要把 `/tmp/...` 作为文档中的规范输出路径；长参数说明写进
`summary.json` / `summary.md`。

## 四类合成失败模式

| Case | 目的 | 主要风险 |
|------|------|----------|
| `transient_attack` | 衰减谐波音符 + 高频短 attack | 瞬态被平滑、diff 中出现乐音攻击头 |
| `stereo_decorrelation` | 左右声道刻意使用不同谐波结构 | 联合块假设过强，声像被拉回或漂移 |
| `nonstationary_noise` | 噪声随时间扫过 2kHz 分频点 | 固定 BPW 分频无法跟踪非平稳噪声 |
| `low_snr` | 弱谐波信号埋在强宽带噪声下 | 低秩假设失效，SVD 截断误删主体信号 |

## 指标解读

核心指标：

- `input_snr_db`：输入相对 clean 的信噪比。
- `output_snr_db`：输出相对 clean 的信噪比。
- `snr_improvement_db`：输出 SNR 减输入 SNR。
- `clean_error_rms`：`clean - output` 的 RMS；越低越保真。
- `residual_rms`：`input - output` 的 RMS；表示移除了多少内容。
- `residual_clean_projection_ratio`：residual 与 clean 的归一化投影；越高说明被移除内容越像原始信号，误删风险越高。
- `highband_retention_ratio`：输出高频 RMS 相对 clean 高频 RMS；过低可能误删高频谐波，过高可能噪声残留。
- `stereo_corr_delta`：输出左右声道相关性相对 clean 的变化。
- `mid_side_ratio_delta`：输出 mid/side 能量结构相对 clean 的变化。
- `rank_mean` / `rank_std` / `rank_max_delta`：能量阈值路径每帧选择的 rank 摘要；`rank_max_delta` 高时优先听是否有调制伪影。

建议先看 `residual_clean_projection_ratio` 和 `clean_error_rms`。如果 SNR 变好但这两个指标也变坏，说明算法可能是在“去噪”的同时删掉了信号。

## 首轮默认配置结果

2026-05-13 的默认合成运行使用：

```bash
PYTHONPATH=. python scripts/run_failure_modes.py \
  --output-root data/processed/fm
```

客观指标中，`bpw_default` 在四个 case 的 SNR improvement 都最高：

| Case | fullband | bandpass no whiten | bpw default | 客观结论 |
|------|----------|--------------------|-------------|----------|
| `transient_attack` | +2.02 dB | +0.49 dB | **+3.20 dB** | BPW 最强，同时高频保留更受控 |
| `stereo_decorrelation` | +2.16 dB | +0.43 dB | **+3.13 dB** | BPW 最强，residual-clean 投影最低 |
| `nonstationary_noise` | -0.57 dB | +0.04 dB | **+3.68 dB** | 全频 MSSA 明显失败，BPW 最强 |
| `low_snr` | +0.67 dB | +0.45 dB | **+3.14 dB** | BPW 最强，但 rank 波动较大 |

Gemini 听感复核应优先使用 reference-based packet：把 `clean.wav` 和
`input.wav` 明确作为参考音频，三个候选输出继续匿名评分。只给候选输出的
candidate-only prompt 在合成信号上不够稳定，容易把残余噪声和合成音色误判成
“自然/不自然”。

单个 case 可用下面的形式复跑：

```bash
PYTHONPATH=. python scripts/prepare_llm_audio_eval.py \
  --call \
  --model gemini-2.5-flash \
  --reference-audio clean_reference=data/processed/fm/transient_attack/clean.wav \
  --reference-audio noisy_input=data/processed/fm/transient_attack/input.wav \
  --audio fullband_energy_mssa=data/processed/fm/transient_attack/fullband_energy_mssa.wav \
  --audio bandpass_no_whiten=data/processed/fm/transient_attack/bandpass_no_whiten.wav \
  --audio bpw_default=data/processed/fm/transient_attack/bpw_default.wav \
  --output-dir data/processed/fm/transient_attack/gemini_reference_flash
```

Reference-based `gemini-2.5-flash` 首轮结果：

| Case | Gemini winner | Ranking | 解释 |
|------|---------------|---------|------|
| `transient_attack` | `bpw_default` | BPW > 全频 > 裸带通 | 认为 BPW 去噪最多且 attack 保留接近 clean |
| `stereo_decorrelation` | `bpw_default` | BPW > 全频 > 裸带通 | 认为 BPW 最接近 clean，声像无明显损伤 |
| `nonstationary_noise` | `bandpass_no_whiten` | 裸带通 > 全频 > BPW | 与客观指标冲突；BPW 的 SNR improvement 最高，需人工复听 |
| `low_snr` | `bpw_default` | BPW > 全频 > 裸带通 | 认为 BPW 在强噪下保留主体最好 |

结论：默认 BPW 不是“找到了失败 case”，反而在这四个合成压力测试上整体最稳。
真正需要继续查的是 `nonstationary_noise` 的主观/客观冲突，以及 `low_snr` 的
`rank_max_delta=13` 是否会在更长片段中表现为泵动或闪烁伪影。

## 下一步用法

1. 先跑默认配置，确认四类失败模式的相对排序。
2. 对最差 case 做参数扫描，例如 `--energy-fraction 0.85/0.9/0.95`、`--bypass-freq 1500/2000/2500`。
3. 把最有代表性的 residual 进行听感检查：residual 如果能听出清晰乐音或攻击头，说明该 case 是真实失败边界。
4. 将稳定复现的失败模式反哺 P1-A：分析 Hankel 窗长、联合块、能量 rank 波动和 BPW 白化各自承担了什么责任。

这份实验是诊断基座，不是最终 benchmark。它先回答“哪里会坏”，再决定是否进入 STFT/DL 对比、Prony/ESPRIT 理论推导或多尺度 MSSA。
