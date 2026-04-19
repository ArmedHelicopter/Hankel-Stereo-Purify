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

以下与「薄封装 / 热路径 / 异常分桶」类审查结论对齐，避免文档与执行路径脱节：

* **调度层：** 已移除独立的 Stage 类、`MssaFramePipeline` 与 `AudioPurifierBuilder`；单帧数学链由 [`process_frame`](../src/core/process_frame.py) 直接表达，截断与 W-correlation 逻辑在 [`c_svd.py`](../src/core/stages/c_svd.py) 的 `make_svd_step` 所返回的**可调用对象**内（[`_FixedRankSvdStep`](../src/core/stages/c_svd.py) / [`_EnergySvdStep`](../src/core/stages/c_svd.py)），状态字段见 `_SvdStepState`。
* **反对角平均：** [`d_diagonal.py`](../src/core/stages/d_diagonal.py) 对固定 `(m,n)` 预计算 `t_flat = i+j`，用 `numpy.bincount` 聚合；仅对 batch 维（如立体声 `B=2`）做 Python 循环，**不再**为全量 scatter 分配 `O(B·mn)` 整型索引表。
* **W-correlation / 能量路径：** 可选 W-correlation 与能量截断的算力与内存阶见 §2.2；能量模式含对同一矩阵的多次 `svds` 探测与可能的全 `svd` 回退，行为由常量帽与浮点容差约束，**非闭式一步解**。
* **门面异常：** `process_file` 对线性代数与 ARPACK 异常显式映射；`ValueError` 映射为 `ProcessingError` 并记录 `format_exception_origin`；其余未捕获类型落入 `except Exception` 并记完整栈（`logger.exception`），对外仍为泛化文案——**属刻意粗分桶**，排障依赖日志与 `__cause__` 链。包装时使用 **`raise ProcessingError(...) from exc`**：Python 将 `exc` 置于 **`__cause__`**，标准 traceback 以「链式」展示，**并非**丢弃 SciPy/ARPACK 来源；若库调用方坚持「不包一层、直接收到 `LinAlgError`」，属不同 API 契约，当前默认不裸透传。所有经门面包装的 [`ProcessingError`](../src/core/exceptions.py) 均带可机读字段 **`origin_exception_type`**（``module.QualName``，由 [`exception_fully_qualified_name`](../src/core/exceptions.py) 生成），CLI 在失败时额外打印该行，便于区分例如 `builtins.MemoryError` 与未单独映射的数值栈类型。

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
