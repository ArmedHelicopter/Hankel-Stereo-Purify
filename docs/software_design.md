# 软件设计与架构规约 (Software Design Specification)

## 1. 架构原则 (Architectural Principles)

本项目 (`Hankel-Stereo-Purify`) 以**可验证的数值路径**与**清晰的 I/O 边界**为先：核心 MSSA 步骤以普通函数与可调用对象表达，避免仅为“模式”而增加调度层。

## 2. 当前实现要点（与代码一致）

### 2.1 MSSA 数值链（`process_frame` 单函数）

单帧处理由 [`process_frame`](../src/core/process_frame.py) 顺序调用 [`hankel_embed`](../src/core/stages/hankel.py) → [`combine_hankel_blocks`](../src/core/stages/multichannel.py) → [`make_svd_step`](../src/core/stages/svd.py) 返回的可调用对象 → [`diagonal_reconstruct`](../src/core/stages/diagonal.py)。[`AudioPurifier`](../src/facade/purifier.py) 在 `_make_denoise_frame_fn` 中用 `functools.partial(process_frame, window_length=..., svd_step=...)` 绑定参数，无 Stage 类或 Pipeline 对象。

**澄清（避免误审旧架构）：** 仓库**不存在**独立 pipeline 调度层；历史命名已让位于 [`array_types.py`](../src/core/array_types.py) 与 [`process_frame.py`](../src/core/process_frame.py)。当前实现**无** `MSSAStage(ABC, Generic[...])`、无动态 `Pipeline.execute` 循环。

库与 CLI 均通过 **`AudioPurifier(...)`** 构造门面（构造函数内完成参数校验）。

### 2.2 截断配置（非经典 Strategy 多态）

* **类型：** [`FixedRankStrategy`](../src/core/strategies/truncation.py) 与 [`EnergyThresholdStrategy`](../src/core/strategies/truncation.py) 为两个独立配置类；`TruncationStrategy` 在源码中为 **`FixedRankStrategy | EnergyThresholdStrategy` 的类型别名**，**不是**抽象基类。`make_svd_step` 在**构造可调用对象时**按具体类型分支，而非运行期虚表派发。
* **默认 BPW 预处理：** `AudioPurifier(...)` 默认做 2kHz bandpass split，高频分支做可逆谱白化后进入 MSSA；旧全频 MSSA 需显式选择（CLI `--fullband` 或 API `bypass_freq=None, highband_whiten=False`）。
* **可观测性（门面）：** `process_file` 将 MSSA 链上的 `ValueError` 映射为 `ProcessingError` 时，日志含 **`format_exception_origin`** 给出的 `文件名:行号:函数`；CLI 对 `ProcessingError` 额外打印 **`__cause__` 来源**，便于区分 `svd` / `diagonal` / 配置校验等，而无需解析异常字符串。

### 2.3 外观 (`AudioPurifier` + `SoundfileOlaEngine`)

* [`AudioPurifier`](../src/facade/purifier.py)：`process_file`、路径与配置校验、构造单帧去噪函数；组合 [`SoundfileOlaEngine`](../src/facade/soundfile_ola.py) 执行整文件流式路径。
* [`SoundfileOlaEngine`](../src/facade/soundfile_ola.py)：OLA 主循环、PCM 队列、可选 memmap（不再以 Mixin 混入）。
* CLI：[`src/cli.py`](../src/cli.py)；可选前端：仓库根目录 [`frontend/app.py`](../frontend/app.py)。

### 2.4 超参数构造

* 超参数在 **`AudioPurifier(...)`** 构造时校验（`truncation_rank` 与 `energy_fraction` 互斥等）。
* 环境变量 `HSP_MAX_SAMPLES` 与 CLI `--max-samples` 在 **构造阶段** 解析；非法值抛出 `ConfigurationError`。

### 2.5 与架构批判的对照（实现现状，非新增抽象）

* **调度层：** 已移除独立的 Stage 类、`MssaFramePipeline` 与 `AudioPurifierBuilder`；单帧数学链由 [`process_frame`](../src/core/process_frame.py) 直接表达，截断逻辑在 [`svd.py`](../src/core/stages/svd.py) 的 `make_svd_step` 所返回的**可调用对象**内（[`_FixedRankSvdStep`](../src/core/stages/svd.py) / [`_EnergySvdStep`](../src/core/stages/svd.py)），状态字段见 `_SvdStepState`。
* **反对角平均：** [`diagonal.py`](../src/core/stages/diagonal.py) 对固定 `(m,n)` 预计算 `t_flat = i+j`，用 `numpy.bincount` 聚合；仅对 batch 维（如立体声 `B=2`）做 Python 循环，**不再**为全量 scatter 分配 `O(B·mn)` 整型索引表。
* **能量路径：** 能量模式含对同一矩阵的多次 `svds` 探测与可能的全 `svd` 回退，行为由常量帽与浮点容差约束，**非闭式一步解**。
* **门面异常：** `process_file` 对线性代数与 ARPACK 异常显式映射；`ValueError` 映射为 `ProcessingError` 并记录 `format_exception_origin`；其余未捕获类型落入 `except Exception` 并记完整栈（`logger.exception`），对外仍为泛化文案——**属刻意粗分桶**，排障依赖日志与 `__cause__` 链。包装时使用 **`raise ProcessingError(...) from exc`**：Python 将 `exc` 置于 **`__cause__`**，标准 traceback 以「链式」展示，**并非**丢弃 SciPy/ARPACK 来源；若库调用方坚持「不包一层、直接收到 `LinAlgError`」，属不同 API 契约，当前默认不裸透传。所有经门面包装的 [`ProcessingError`](../src/core/exceptions.py) 均带可机读字段 **`origin_exception_type`**（``module.QualName``，由 [`exception_fully_qualified_name`](../src/core/exceptions.py) 生成），CLI 在失败时额外打印该行，便于区分例如 `builtins.MemoryError` 与未单独映射的数值栈类型。

### 2.6 BPW 默认预处理架构

基于 Gemini 音频模态分析与后续四路/alpha 实验（见[实验日志](experiment_log.md)），当前默认 pipeline 采用 **BPW**：bandpass split + high-band whitening + MSSA。目标是让低中频绕过 SVD，避免硬截断误删主体乐音；高频分支先做可逆噪声谱白化，使 MSSA/SVD 更符合“噪声低能量、乐声结构高能量”的截断先验。

**处理流程：**

```text
输入信号
  -> split_signal(cutoff=2000Hz)
  -> low_band bypass
  -> high_band
       -> estimate frequency-only noise profile
       -> whiten: STFT bin / profile^alpha
       -> OLA + MSSA
       -> unwhiten: STFT bin * profile^alpha
  -> low_band + processed_high
```

**默认参数：**

- `bypass_freq = 2000Hz`
- `highband_whiten = True`
- `whiten_alpha = 0.75`

**回退路径：**

- CLI `--fullband` / API `bypass_freq=None, highband_whiten=False`：旧全频 MSSA。
- CLI `--no-highband-whiten` / API `highband_whiten=False`：裸带通，仅 high band 进入 MSSA。

**实现：**
- `src/core/stages/filter.py`：zero-phase Butterworth low split，`high_band = signal - low_band` 保持可重构。
- `src/core/stages/whitening.py`：固定频率尺度白化/反白化；不做 mask、阈值、谱减、Wiener 或 bin 删除。
- `src/facade/purifier.py`：全文件 split 后仅将 high-band 临时 WAV FLOAT 送入现有 OLA+MSSA；白化 artifact 可保存 roundtrip、baseline、diff 和 metrics。

白化本身通过 `roundtrip = unwhiten(whiten(high_band))` 对照验证近似可逆；降噪贡献应归因于白化改变 MSSA 输入能量分布后触发的 SVD 截断，而不是 STFT 预处理直接降噪。

### 2.7 参数选择经验（实测结论）

以下基于 Beethoven 钢琴录音（44100Hz stereo MP3）的实测，非理论推导。

#### OLA hop 约束

sqrt-Hanning 窗仅在 **hop = F**（无重叠）和 **hop = F/2**（50% 重叠）时归一化正确。其他重叠比会导致幅度调制伪影。

**程序已锁死 hop = frame_size // 2**，移除了 `hop_size` 参数。详见 [实验日志](experiment_log.md#2026-05-08-ola-重叠比实验)。

#### BPW 默认策略

当前默认不再直接全频硬截断，而是使用 BPW：

```text
bypass_freq = 2000Hz
highband_whiten = True
whiten_alpha = 0.75
```

四路对照与 alpha sweep 的当前结论：全频 MSSA 的 diff 中更容易出现乐音残留；裸带通更接近原始但高频噪声保留多；带通白化的 residual 主要是噪声，且 roundtrip 对照显示白化/反白化本身近似恒等。

#### 硬截断的根本局限

能量截断基于同一个前提：**按奇异值排序可以分离信号与噪声**。这个前提在钢琴音频上不稳定：

- SVD 按能量排序分量，但不区分「有用信号」和「噪声」
- 钢琴的弱谐波（高次泛音、共振峰）奇异值小，落在噪声区间
- 硬截断把弱谐波和噪声一起扔掉
- diff = original − denoised 中可以听到音乐成分

**方向**：通过带通与可逆高频白化重排 MSSA 看到的能量结构，而不是在全频输出后继续追加后置补偿。

#### 逐帧自适应阈值（未实现）

当前 `energy_fraction` 为全局固定值，所有帧共用同一个截断阈值。这隐含一个假设：所有帧的最优砍法相同。

实际上不同帧的情况不同：
- 静音段：噪声主导，应砍更狠（更低 threshold）
- 音符段：信号主导，应更保守（更高 threshold）
- 瞬态段：能量分散，砍错即失真

全局固定阈值是对所有帧取同一个妥协点，不可能是每帧的最优。但逐帧自适应也引入新风险：
- 相邻帧 k 差异大 → 调制伪影
- 需要一个「判断该帧该用什么阈值」的机制，本身又是信号分析问题

当前架构选择全局固定阈值，换取帧间稳定性。逐帧自适应可作为后续优化方向，但需先解决帧间一致性的约束。

#### 截断秩的稳定性假设

能量截断模式隐含一个假设：**截断秩 k 在帧间稳定**。实测发现：
- `EnergyThresholdStrategy(0.9)` 选出的 k 在帧间有波动
- 噪声源稳定的假设**不能推导出截断秩恒定**——信号的频谱结构随时间变化（如钢琴音符切换），导致每帧的奇异值分布不同
- `energy_k_prev` 用作 warm-start（`svd.py`），但这只是加速探测，不保证 k 恒定

这本身不一定是问题——逐帧自适应 k 是能量截断的设计意图。但如果 k 波动剧烈（如在瞬态处跳变），可能引入调制伪影。可通过 `scripts/benchmark_pipeline.py` 观察阶段 C 的耗时与 k 值分布。

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
variant_*.mp3     — 变体（如不同 `whiten_alpha`）
diff_*.mp3        — baseline - variant
```

#### 参数扫描法

对关键参数（如 `energy_fraction`、`whiten_alpha`）取多个值，生成对比文件和 SNR/高频残留指标。

**流程**：
1. 固定其他参数，扫目标参数
2. 对每个值：生成输出文件 + 计算 SNR
3. 画 SNR-参数曲线，找拐点（SNR 不再上升或开始下降的位置）
4. 听拐点附近的输出，主观验证

**代码位置**：`scripts/run_alpha_sweep_09.py` / `scripts/run_alpha_sweep_09b.py`（白化强度扫描示例）

## 3. 标准化工程目录映射 (Directory Structure)

```text
src/
├── core/
│   ├── array_types.py      # FloatArray 等 ndarray 类型别名（原 pipeline.py 仅类型，无调度器）
│   ├── process_frame.py    # 单帧 MSSA A→D（`process_frame`）
│   ├── pipeline/            # 兼容 re-export：`from .pipeline import process_frame`
│   ├── stages/               # hankel / multichannel / svd / diagonal 函数与工厂
│   │   ├── hankel.py
│   │   ├── multichannel.py
│   │   ├── svd.py
│   │   ├── diagonal.py
│   │   ├── filter.py
│   │   └── whitening.py
│   └── strategies/
│       ├── truncation.py
│       └── windowing.py
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
