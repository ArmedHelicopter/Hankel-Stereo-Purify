# 软件设计与架构规约 (Software Design Specification)

## 1. 架构原则 (Architectural Principles)

本项目 (`Hankel-Stereo-Purify`) 以**可验证的数值路径**与**清晰的 I/O 边界**为先：核心 MSSA 步骤以普通函数与可调用对象表达，避免仅为“模式”而增加调度层。

## 2. 当前实现要点（与代码一致）

### 2.1 MSSA 数值链（`process_frame` 单函数）

单帧处理由 [`process_frame`](../src/core/process_frame.py) 顺序调用 [`hankel_embed`](../src/core/stages/a_hankel.py) → [`combine_hankel_blocks`](../src/core/stages/b_multichannel.py) → [`make_svd_step`](../src/core/stages/c_svd.py) 返回的可调用对象 → [`diagonal_reconstruct`](../src/core/stages/d_diagonal.py)。[`AudioPurifier`](../src/facade/purifier.py) 在 `_make_denoise_frame_fn` 中用 `functools.partial(process_frame, window_length=..., svd_step=...)` 绑定参数，无 Stage 类或 Pipeline 对象。

**澄清（避免误审旧架构）：** 仓库**不存在**独立文件 `src/core/pipeline.py`（历史命名已让位于 [`array_types.py`](../src/core/array_types.py) 等）；[`src/core/pipeline/`](../src/core/pipeline/__init__.py) 仅为兼容 re-export `process_frame`，**无** `MSSAStage(ABC, Generic[...])`、无动态 `Pipeline.execute` 循环。

库与 CLI 均通过 **`AudioPurifier(...)`** 构造门面（构造函数内完成参数校验）。

### 2.2 截断配置（非经典 Strategy 多态）

* **类型：** [`FixedRankStrategy`](../src/core/strategies/truncation.py) 与 [`EnergyThresholdStrategy`](../src/core/strategies/truncation.py) 为两个独立配置类；`TruncationStrategy` 在源码中为 **`FixedRankStrategy | EnergyThresholdStrategy` 的类型别名**，**不是**抽象基类。`make_svd_step` 在**构造可调用对象时**按具体类型分支，而非运行期虚表派发。
* **W-correlation（可选）：** 在 `make_svd_step` 内对奇异值向量做过滤；能量模式下首帧可标定保留索引并缓存（见 `c_svd.py`）。算力上：各秩分量经 rank-1 矩阵反对角聚合（实现为逐秩 **`O(m \cdot n)`** 临时块，避免一次性物化完整 `(k,m,n)` 张量），随后 `compute_w_correlation_matrix` 为 **\(O(k^2 \cdot L_{\text{seq}})\)**（\(L_{\text{seq}}=m+n-1\)）；未开启时无此两项。单帧对角阶段占比可用 `python scripts/benchmark_pipeline.py --diag-split` 观察。
* **可观测性（门面）：** `process_file` 将 MSSA 链上的 `ValueError` 映射为 `ProcessingError` 时，日志含 **`format_exception_origin`** 给出的 `文件名:行号:函数`；CLI 对 `ProcessingError` 额外打印 **`__cause__` 来源**，便于区分 `c_svd` / `d_diagonal` / 配置校验等，而无需解析异常字符串。

### 2.3 外观 (`AudioPurifier` + `SoundfileOlaEngine`)

* [`AudioPurifier`](../src/facade/purifier.py)：`process_file`、路径与配置校验、构造单帧去噪函数；组合 [`SoundfileOlaEngine`](../src/facade/soundfile_ola.py) 执行整文件流式路径。
* [`SoundfileOlaEngine`](../src/facade/soundfile_ola.py)：OLA 主循环、PCM 队列、可选 memmap（不再以 Mixin 混入）。
* CLI：[`src/cli.py`](../src/cli.py)；可选前端：仓库根目录 [`frontend/app.py`](../frontend/app.py)。

### 2.4 超参数构造

* 超参数在 **`AudioPurifier(...)`** 构造时校验（`truncation_rank` 与 `energy_fraction` 互斥等）。
* 环境变量 `HSP_MAX_SAMPLES` 与 CLI `--max-samples` 在 **构造阶段** 解析；非法值抛出 `ConfigurationError`。

### 2.5 与架构批判的对照（实现现状，非新增抽象）

* **调度层：** 已移除独立的 Stage 类、`MssaFramePipeline` 与 `AudioPurifierBuilder`；单帧数学链由 [`process_frame`](../src/core/process_frame.py) 直接表达，截断与 W-correlation 逻辑在 [`c_svd.py`](../src/core/stages/c_svd.py) 的 `make_svd_step` 所返回的**可调用对象**内（[`_FixedRankSvdStep`](../src/core/stages/c_svd.py) / [`_EnergySvdStep`](../src/core/stages/c_svd.py)），状态字段见 `_SvdStepState`。
* **反对角平均：** [`d_diagonal.py`](../src/core/stages/d_diagonal.py) 对固定 `(m,n)` 预计算 `t_flat = i+j`，用 `numpy.bincount` 聚合；仅对 batch 维（如立体声 `B=2`）做 Python 循环，**不再**为全量 scatter 分配 `O(B·mn)` 整型索引表。
* **W-correlation / 能量路径：** 可选 W-correlation 与能量截断的算力与内存阶见 §2.2；能量模式含对同一矩阵的多次 `svds` 探测与可能的全 `svd` 回退，行为由常量帽与浮点容差约束，**非闭式一步解**。
* **门面异常：** `process_file` 对线性代数与 ARPACK 异常显式映射；`ValueError` 映射为 `ProcessingError` 并记录 `format_exception_origin`；其余未捕获类型落入 `except Exception` 并记完整栈（`logger.exception`），对外仍为泛化文案——**属刻意粗分桶**，排障依赖日志与 `__cause__` 链。包装时使用 **`raise ProcessingError(...) from exc`**：Python 将 `exc` 置于 **`__cause__`**，标准 traceback 以「链式」展示，**并非**丢弃 SciPy/ARPACK 来源；若库调用方坚持「不包一层、直接收到 `LinAlgError`」，属不同 API 契约，当前默认不裸透传。所有经门面包装的 [`ProcessingError`](../src/core/exceptions.py) 均带可机读字段 **`origin_exception_type`**（``module.QualName``，由 [`exception_fully_qualified_name`](../src/core/exceptions.py) 生成），CLI 在失败时额外打印该行，便于区分例如 `builtins.MemoryError` 与未单独映射的数值栈类型。

### 2.6 参数选择经验（实测结论）

以下基于 Beethoven 钢琴录音（44100Hz stereo MP3）的实测，非理论推导。

#### OLA hop 约束

sqrt-Hanning 窗仅在 **hop = F**（无重叠）和 **hop = F/2**（50% 重叠）时归一化正确。其他重叠比会导致窗函数叠加不恒定，产生幅度调制伪影（可闻的「滋滋」声）。

| hop | 重叠比 | 状态 | 原因 |
|-----|--------|------|------|
| F | 0% | ✓ | 每点仅一帧覆盖，窗函数只乘一次 |
| F/2 | 50% | ✓ | cos² + sin² = 1，sqrt-Hanning 完美叠加 |
| F/4 | 75% | ✗ | 四帧叠加不恒定 → 幅度调制 → 滋滋声 |

**程序已锁死 hop = frame_size // 2**，移除了 `hop_size` 参数。不再允许用户配置其他重叠比。

#### W-correlation 实验记录

**问题根因分析（2026-05-08）：**

W-correlation 修复前所有阈值 SNR 均为 ~5.5dB（vs baseline 13.5dB），三个独立 bug 叠加：

1. **权重公式错误**：`grouping.py` 内积计算用 `weighted @ arr.T`（加权×非加权），正确应为 `weighted @ weighted.T`。
2. **分组逻辑缺陷**：旧逻辑仅保留与第一个分量相关性高的分量（`w_mat[1:, 0] >= thr`）。SVD 分量正交，钢琴各谐波在不同分量中，W-correlation 低→只保留基频分量，谐波全部被误删。
3. **冻结状态导致帧间不一致**：`energy_w_corr_frozen=True` 锁死第一帧的分组关系。当后续帧的截断秩 k 变化时，冻结索引无法适配新分量（如 frame 0 时 k=5，冻结 [0-4]；后续帧 k=6 时第 6 个分量被无条件丢弃）。

**修复方案：**
- 权重公式：改为 `weighted @ weighted.T`
- 分组逻辑：层次聚类（average linkage）替代锚点分组；保留所有能量 ≥5% 总能量的聚类
- 冻结状态：移除，每帧重算聚类

**修复后性能（Beethoven Adagio, 5s, F=1024 L=256 hop=512 energy=0.9）：**

| 阈值 | SNR vs 原始 | SNR vs baseline | diff% | 听感 |
|------|------------|----------------|-------|------|
| baseline (纯能量) | 13.5dB | — | — | 降噪有效 |
| 0.1 | 13.4dB | 32.5dB | 2.4% | ≈ 能量 baseline |
| 0.3 | 13.1dB | 27.7dB | 4.1% | 轻微差异 |
| 0.5 | 13.0dB | 26.4dB | 4.8% | 略有衰减 |
| 0.7 | 12.9dB | 26.0dB | 5.0% | 可感知衰减 |
| 0.9 | 12.8dB | 25.4dB | 5.4% | 明显衰减 |

diff = energy_out − wcorr_out，代表 W-correlation 在能量截断基础上额外移除的内容。

**结论**：W-correlation 修复后从 5.5dB 恢复到 13.4dB，但仍**不如不加**（baseline 13.5dB）。根本原因：W-correlation 在能量截断之后做额外过滤，只会减少信号不会增加信号。聚类再保守也改变不了「硬截断丢信号」的本质。

**决策**：W-correlation 不应默认启用。下一步探索方向：Wiener 软加权（连续权重替代硬截断）。

#### 硬截断的根本局限

能量截断和 W-correlation 都基于同一个前提：**按奇异值排序可以分离信号与噪声**。这个前提在钢琴音频上不成立：

- SVD 按能量排序分量，但不区分「有用信号」和「噪声」
- 钢琴的弱谐波（高次泛音、共振峰）奇异值小，落在噪声区间
- 硬截断（保留前 k 个、丢弃其余）把弱谐波和噪声一起扔掉
- diff = original − denoised 中可以听到音乐成分，证实信号被误删

**这不是参数问题，是框架问题**。任何 0/1 截断策略都无法避免：要么砍不够（噪声残留），要么砍太多（信号丢失），没有中间态。

**方向**：用连续权重替代 0/1 决策。对每个分量赋予 [0, 1] 的权重（基于 SNR 估计），而非 keep/discard 二值选择。Wiener 软加权是这个方向的数学最优解（MMSE）。

#### 逐帧自适应阈值（未实现）

当前 `energy_fraction` 为全局固定值，所有帧共用同一个截断阈值。这隐含一个假设：所有帧的最优砍法相同。

实际上不同帧的情况不同：
- 静音段：噪声主导，应砍更狠（更低 threshold）
- 音符段：信号主导，应更保守（更高 threshold）
- 瞬态段：能量分散，砍错即失真

全局固定阈值是对所有帧取同一个妥协点，不可能是每帧的最优。但逐帧自适应也引入新风险：
- 相邻帧 k 差异大 → 调制伪影（类似 W-correlation 的问题）
- 需要一个「判断该帧该用什么阈值」的机制，本身又是信号分析问题

当前架构选择全局固定阈值，换取帧间稳定性。逐帧自适应可作为后续优化方向，但需先解决帧间一致性的约束。

#### 截断秩的稳定性假设

能量截断模式隐含一个假设：**截断秩 k 在帧间稳定**。实测发现：
- `EnergyThresholdStrategy(0.9)` 选出的 k 在帧间有波动
- 噪声源稳定的假设**不能推导出截断秩恒定**——信号的频谱结构随时间变化（如钢琴音符切换），导致每帧的奇异值分布不同
- `energy_k_prev` 用作 warm-start（`c_svd.py`），但这只是加速探测，不保证 k 恒定

这本身不一定是问题——逐帧自适应 k 是能量截断的设计意图。但如果 k 波动剧烈（如在瞬态处跳变），可能引入类似 W-correlation 的调制伪影。可通过 `scripts/benchmark_pipeline.py` 观察阶段 C 的 k 值分布。

### 2.7 测试与验证方法

#### SNR 对比法

通过信噪比（SNR）量化降噪效果和失真程度。

**定义**：
```
SNR = 10 · log10(Σx² / Σ(x - y)²)
```
其中 x 为参考信号，y 为测试信号。

**用法**：
- SNR(原始, 输出)：衡量整体变化（含降噪+失真）
- SNR(基线, 变体)：衡量变体相对于基线的差异
- SNR 越高 = 差异越小

**局限**：SNR 不区分「删噪声」和「删信号」。差值 = 噪声 + 误删信号，分不开。

#### 差值信号分析法

输出 `diff = A - B`，即两个版本的差值信号。

**判据**：
- diff 可听到音乐/人声 → A 或 B 误删了信号
- diff 是白噪声状弥散 → A 和 B 的差异是均匀的噪声级变化
- diff 有瞬态/突起 → A 和 B 在特定帧有剧烈差异

**文件命名约定**：
```
original_*.mp3    — 原始输入
baseline_*.mp3    — 基线（能量截断 only）
variant_*.mp3     — 变体（如加 W-correlation）
diff_*.mp3        — baseline - variant
```

#### 参数扫描法

对关键参数（如 `energy_fraction`、W-correlation 阈值）取多个值，生成对比文件和 SNR 指标。

**流程**：
1. 固定其他参数，扫目标参数
2. 对每个值：生成输出文件 + 计算 SNR
3. 画 SNR-参数曲线，找拐点（SNR 不再上升或开始下降的位置）
4. 听拐点附近的输出，主观验证

**代码位置**：`tests/test_wcorr_sweep.py`（W-correlation 阈值扫描示例）

## 3. 标准化工程目录映射 (Directory Structure)

```text
src/
├── core/
│   ├── array_types.py      # FloatArray 等 ndarray 类型别名（原 pipeline.py 仅类型，无调度器）
│   ├── process_frame.py    # 单帧 MSSA A→D（`process_frame`）
│   ├── pipeline/            # 兼容 re-export：`from .pipeline import process_frame`
│   ├── stages/               # hankel / multichannel / svd / diagonal 函数与工厂
│   │   ├── a_hankel.py
│   │   ├── b_multichannel.py
│   │   ├── c_svd.py
│   │   └── d_diagonal.py
│   └── strategies/
│       ├── truncation.py
│       ├── windowing.py
│       └── grouping.py
├── io/
├── facade/
│   ├── purifier.py           # AudioPurifier、process_file
│   ├── soundfile_ola.py    # SoundfileOlaEngine：OLA + 队列 + memmap
│   ├── pcm_producer.py
│   └── ola.py
├── utils/
│   └── logger.py
└── cli.py
```

与 `src/` **并列**：可选 Streamlit [`frontend/app.py`](../frontend/app.py)。数据平面主路径为 `src/cli.py`。
