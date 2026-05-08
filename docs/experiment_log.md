# 实验日志 (Experiment Log)

本文件记录 Hankel-Stereo-Purify 的降噪实验过程、结果和决策。按时间倒序排列。

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
