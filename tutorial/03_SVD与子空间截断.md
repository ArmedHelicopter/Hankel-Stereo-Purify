# 第 3 章：SVD 与子空间截断

## 直觉先行

把联合矩阵想成一堆**基方向**的叠加：奇异值 \(\sigma_i\) 越大，对应方向在数据里越「实」。截断 = 只保留前 \(k\) 个方向，其余当噪声抹掉。能量阈值则是：从大到小累加能量，加到够某一比例就停——**每帧 k 可能变**。

类比：棱镜分光后，你只保留最亮的几条谱线重构颜色，其余当杂散光扔掉。

## 决策复盘（Why vs Why not）

**固定秩 `-k`：为什么有时用 `svds` 有时用 dense `svd`？**  
当 \(k < \min(m,n)\)，`scipy.sparse.linalg.svds` 只求前 \(k\) 个三元组，避免完整谱。若 \(k \ge \min(m,n)\)，截断 SVD 退化为 thin SVD——见 [`_fixed_rank_truncated_factors`](../src/core/stages/c_svd.py)。

**能量 `--energy-fraction`：实现上如何避免「每帧都 full SVD」？**  
能量规则仍依赖**完整谱**上的 `get_k`；实现上 [`_energy_truncated_factors`](../src/core/stages/c_svd.py) 先用若干次 `svds` 试探（带退避上限），仅在必要时退化为一次 [`scipy.linalg.svd`](../src/core/stages/c_svd.py)。数学定义未改；最坏情况仍可能走到 full SVD。

**W-correlation（可选）：为什么贵？**  
冷路径要对每个分量做 rank-1 重构 + 对角平均，再算 \(k\times k\) 加权相关。能量 + W 模式下，当前实现**首帧完整标定、后续帧只掩码 \(\sigma\)**，避免每帧 k 变化就重算整张 W——见 `make_svd_step` 模块文档字符串与内联注释。

**为什么不默认开 W？**  
算力与内存陡增；默认关闭，把决策交给使用者。

### 固定秩 vs 能量：对照

| 模式 | 代码落点 | 数学/行为要点 |
|------|----------|----------------|
| 固定秩 `-k` | `_fixed_rank_truncated_factors`、`FixedRankStrategy` | \(k\) 由配置给定；可 `svds` 或退化为 thin `svd` |
| 能量 `--energy-fraction` | `EnergyThresholdStrategy.get_k`、`make_svd_step` 内能量分支 | 需全谱或等价信息以累计能量；每帧 **\(k\) 可变** |
| W-correlation（能量子路径，可选） | `_w_corr_keep_indices`、`_filter_w_corr`、`energy_w_corr_frozen` | 冷路径贵；首帧完整、后续掩码 \(\sigma\) 的冻结语义 |

### 最小公式（抄笔记）

截断后低秩重构可写为 **\((U \odot \sigma) V^\mathsf{H}\)**（\(\odot\) 为按列缩放）；与实现 **`(u * s) @ vh`** 逐项对应（广播避免显式 `diag`）。

### 代码锚点

| 定位 | 路径 |
|------|------|
| `make_svd_step`：固定秩 / 能量两分支 | [`src/core/stages/c_svd.py`](../src/core/stages/c_svd.py) |
| \(k\) 与谱 | 固定秩用 `FixedRankStrategy.k`；能量用 `EnergyThresholdStrategy.get_k(s)`（[`truncation.py`](../src/core/strategies/truncation.py)） |
| 策略与数值对齐测试 | [`tests/test_c_svd.py`](../tests/test_c_svd.py) |

## 思维挂钩

| 代码 | 专业课 | 软件工程 |
|------|--------|----------|
| \(U\Sigma V^\mathsf{T}\) 截断 | 秩、k、主子空间 | 配置类 + `make_svd_step` 工厂分支（`TruncationStrategy` 为类型别名） |
| `EnergyThresholdStrategy.get_k` | 累计能量、分位数思想 | 配置与算法分离 |
| `(u * s) @ vh` | 避免显式 `diag` | 广播与 GEMM 融合 |

## 晦涩点与建议

- **晦涩**：`make_fixed_rank_svd_step` 是 `make_svd_step` 在固定秩下的薄封装，初学者易重复造轮子。  
- **建议**：先看 [`src/core/stages/__init__.py`](../src/core/stages/__init__.py) 的 `__all__`，再读 [`test_c_svd.py`](../tests/test_c_svd.py) 里二者数值对齐测试。
- **晦涩**：能量分支 `_energy_truncated_factors` 中 `k_probe` 与 `_SVDS_ENERGY_PROBE_CAP`——**不是**改数学定义，而是限制最坏情况下反复 `svds` 的次数。  
- **建议**：对照 [`docs/software_design.md`](../docs/software_design.md) §2.2 与 [`scripts/benchmark_pipeline.py`](../scripts/benchmark_pipeline.py) 实测阶段 C 占比。

**下一章**：低秩矩阵如何变回左右声道波形。
