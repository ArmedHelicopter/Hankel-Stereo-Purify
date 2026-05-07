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

**建议**：默认使用 `hop = F/2`（50% 重叠），除非有明确的性能需求才用 `hop = F`。

#### W-correlation 现状

实测 W-correlation（`--w-corr-threshold 0.3`）在能量截断基础上**劣化音质**：

| 模式 | SNR vs 原始 | 听感 |
|------|-------------|------|
| 能量截断 only | ~12-13dB | 降噪有效，无电流声 |
| 能量 + frozen W | ~5dB | 电流声，明显失真 |
| 能量 + naive W | ~6dB | 电流声，略好于 frozen 但仍差 |

W-correlation 引入的问题：
1. 可闻的「滋滋」电流声（时变滤波器调制伪影）
2. SNR 劣化 ~7-8dB（有用信号被误删）
3. frozen vs naive 差异 ~12dB，但两者都比能量截断差

**结论**：当前实现的 W-correlation 策略**不应默认启用**。如需启用，阈值和冻结策略需要重新评估。

#### 截断秩的稳定性假设

能量截断模式隐含一个假设：**截断秩 k 在帧间稳定**。实测发现：
- `EnergyThresholdStrategy(0.9)` 选出的 k 在帧间有波动
- 噪声源稳定的假设**不能推导出截断秩恒定**——信号的频谱结构随时间变化（如钢琴音符切换），导致每帧的奇异值分布不同
- `energy_k_prev` 用作 warm-start（`c_svd.py`），但这只是加速探测，不保证 k 恒定

这本身不一定是问题——逐帧自适应 k 是能量截断的设计意图。但如果 k 波动剧烈（如在瞬态处跳变），可能引入类似 W-correlation 的调制伪影。可通过 `scripts/benchmark_pipeline.py` 观察阶段 C 的 k 值分布。

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
