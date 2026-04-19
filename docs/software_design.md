# 软件设计与架构规约 (Software Design Specification)

> **适用性**：数学目标与分层适用于完整产品。**`main` 上 `src/`** 提供与设计一致的包结构与占位 API；**可执行 MSSA 实现以 `tutorial` 分支为准**。环境与构建见 [根目录 README §3 运行环境与依赖](../README.md#环境与构建)；分支见 [`BRANCHES.md`](BRANCHES.md)；文档索引见 [docs/README.md](README.md)。

## 1. 架构原则 (Architectural Principles)

本项目在实现上强调：**可验证的数值路径**、**清晰的 I/O 边界**、**薄门面**。核心 MSSA 步骤以普通函数与可调用对象表达，避免仅为「模式」而增加与当前代码不一致的调度层。

- **高内聚、低耦合**：算法留在 `src/core/stages` 与 `strategies`；流式与 OLA 在门面与 I/O 层组合。
- **异常不吞没**：门面在映射数值或配置错误时保留 `__cause__` 链，并提供可机读异常类型字段供 CLI 分桶。
- **敏感配置外置**：不在仓库中硬编码密钥或令牌。

## 2. 数学建模与实现映射

本节从**数学对象—软件边界**说明 MSSA 降噪在代码中的落点。**符号、模块 A–D 的严格定义与验收口径**以 [`prd.md`](prd.md) §2 为唯一事实来源；此处不重述完整推导，只建立**建模层次 ↔ 包与调用拓扑**。**算法假设**（截断 SVD、W-correlation、标定窗与平稳性等）集中在 **§2.6**，与 PRD 行为对齐。

**术语**速查见 [`glossary.md`](glossary.md)（与 PRD §2 符号对照阅读时可减少跳跃）。

为便于在 GitHub 与 VS Code / Cursor 内置 Markdown 预览中渲染：**行内**用 `$...$`，**独立成行**的公式用 `$$ ... $$`（勿用 ` ```math ` 围栏——多数本地预览会把它当成普通代码块，公式不渲染）。与 [`prd.md`](prd.md) 约定一致。

### 2.1 建模目标（抽象）

- **输入**：离散时间上的立体声观测（左、右声道可视为两条同步序列）；工程上经解码为浮点 PCM，再按 **F-02** 做分帧与加窗（短时处理）。
- **核心假设**：在单帧或短窗内，**确定性成分**（谐波结构、空间相干）张成较低维子空间；**宽带底噪**主要落在补空间或高阶奇异方向。联合 Hankel + 截断 SVD 将该假设落实为**可计算的秩截断与重构**（与 PRD 模块 C、D 一致）。
- **输出**：与输入同采样率、同通道数的**去噪后序列**；立体声相位关系由 **联合块矩阵**（模块 B）约束，避免左右独立 SVD（**F-03**）。

### 2.2 建模层次与数据流

自顶向下可概括为：

```text
文件级 PCM 流 → 分帧 / OLA（facade、io）→ 单帧或片段上的向量样本
    → 每声道 Hankel 嵌入 HL, HR（stages A）
    → 联合块矩阵 X_total（stage B）
    → 截断 SVD / 子空间投影（stage C + strategies/truncation）
    → 对角平均得到每声道重构样本（stage D）
    → 帧合成与写出（facade OLA）
```

`process_frame`（见 §3）实现**单帧**上的 A→B→C→D；**全局**连续性由窗长、跳步与 OLA 在门面层保证（PRD **F-02**）。

### 2.3 数学对象与源码落点

下列条目与 §4 中 `core/` 子模块一一对应；公式与 PRD 对齐。

- **窗长 $L$、轨迹列数 $K$**：Hankel 轨迹矩阵形状与滞后结构。典型落点：`strategies/windowing`、`process_frame` 绑定参数；嵌入在 `stages/a_hankel`。
- **$H_{L}$、$H_{R}$**：左右声道 Hankel 矩阵。`stages/a_hankel.py`、`stages/b_multichannel.py`。
- **$\mathbf{X}_{\mathrm{total}} = [H_{L}, H_{R}]$**：联合块矩阵（立体声相干）。`stages/b_multichannel.py`。
- **SVD 与截断**：$\mathbf{X}_{\mathrm{total}} = U \Sigma V^{T}$，截断得 $\Sigma_k$，$\hat{\mathbf{X}} = U \Sigma_k V^{T}$；子空间分解与秩选择。`stages/c_svd.py`、`strategies/truncation.py`。
- **对角平均 $\hat{x}_n$**：低秩矩阵回到一维序列。`stages/d_diagonal.py`。
- **W-correlation / 分组**：奇异分量聚类后再重构。`strategies/grouping.py`。
- **能量或秩策略**：$k$、累计能量阈值等。`FixedRankStrategy` / `EnergyThresholdStrategy`（与 **F-04** 对齐）。

### 2.4 决策变量、约束与 PRD 的对应关系


| 类别              | 建模/工程含义          | PRD 锚点    |
| --------------- | ---------------- | --------- |
| 帧长、跳步、窗函数       | 短时平稳与边界平滑        | **F-02**  |
| 联合矩阵与单通道独立分解的取舍 | 相位/声像一致性         | **F-03**  |
| 截断秩或能量阈值        | 信号子空间 vs 噪声子空间分界 | **F-04**  |
| 峰值内存、流式读        | 实现侧硬约束           | **NF-01** |
| 轨迹矩阵零拷贝构造       | 与数值基准一致          | **NF-02** |


### 2.5 与单帧数值链（§3）的关系

[`prd.md`](prd.md) 的**模块 A–D** 与 §3.1 中 `process_frame` 的四步**一一顺序对应**：嵌入 → 联合 → SVD 步（含截断策略）→ 对角重构。门面层将「文件」折叠为「帧序列」，对每帧调用该链，再在采样域做 OLA 叠加；数学上相当于在**短时窗内**重复上述算子，而非改写 A–D 的代数定义。

### 2.6 算法假设（建模约束）

与 PRD 行为一致前提下的**可操作假设**（不重复公式推导）：

- **截断 SVD**：对 $\mathbf{X}_{\mathrm{total}}$ 使用**确定性**截断（固定秩或能量阈值）；奇异值/子空间便于与调试、回归对齐。**不默认**随机化 SVD；若将来引入须单独约定可解释性与复现性（参见 PRD 对数值路径的要求）。
- **W-correlation / 分组**：W 权重由窗长 $L$、序列长度与对角平均重叠度决定；`strategies/grouping` 中按 W-correlation 聚类分量后再重构，阈值由配置与数据决定。
- **标定窗与平稳性**：能量阈值、噪声底等若由**前几帧或短窗**估计，隐含假设为**标定窗内环境噪声近似平稳**。遇剧烈电平变化、剪辑拼接或强非平稳干扰时，应**缩短/滑动标定窗**、避免依赖单点估计，或**改用手动秩**，以免将信号结构误判为噪声。

## 3. 实现要点与模块边界（`main` 占位，`tutorial` 可交付）

### 3.1 单帧 MSSA 数值链（`process_frame`）

单帧处理由 `process_frame`（`src/core/process_frame.py`）顺序调用：

1. `hankel_embed`（`a_hankel.py`）
2. `combine_hankel_blocks`（`b_multichannel.py`）
3. `make_svd_step(...)` 返回的 **可调用对象**（`c_svd.py`）
4. `diagonal_reconstruct`（`d_diagonal.py`）

门面（如 `AudioPurifier`）通过 `functools.partial(process_frame, window_length=..., svd_step=...)` 绑定参数。单帧编排**仅经** `src/core/process_frame.py` 暴露，请使用 `from src.core.process_frame import process_frame`（或从包 `__init__` 再导出，若有），**不**再设单独的 `pipeline` 别名包。

### 3.2 截断配置（非经典 Strategy 多态）

- `FixedRankStrategy` 与 `EnergyThresholdStrategy`（`truncation.py`）为**具体配置类型**；`TruncationStrategy` 在源码中为二者的**类型别名**，**不是**必须被继承的抽象接口。
- `make_svd_step` 在**构造可调用对象时**按具体类型分支一次；运行期每帧不在 `isinstance` 上反复派发。

### 3.3 外观（Facade）与流式路径

- `AudioPurifier`（`facade/purifier.py`）：`process_file`、路径与配置校验、构造单帧去噪函数；组合 `SoundfileOlaEngine`（`soundfile_ola.py`）执行整文件流式路径；必要时配合 `pcm_producer` 等有界队列。
- CLI：`src/cli.py`。

## 4. 标准化工程目录映射（当前仓库典型布局）

`src/` 结构随演进可能微调。下表为**树形结构 + 同行简要职责**（`#` 后为说明；变更影响分析时按路径定位）。

```text
src/
├── cli.py                          # 命令行入口：参数/帮助，调用门面或占位逻辑
├── core/
│   ├── array_types.py              # NumPy 数组类型别名（如 Float64Array）
│   ├── process_frame.py            # 单帧 MSSA 编排：依次调用 stages A→B→C→D
│   ├── stages/
│   │   ├── a_hankel.py             # 模块 A：嵌入，一维序列 → Hankel 矩阵
│   │   ├── b_multichannel.py       # 模块 B：联合左右块 → X_total
│   │   ├── c_svd.py                # 模块 C：SVD+截断步（可调用对象）
│   │   └── d_diagonal.py           # 模块 D：对角平均 → 一维重构
│   ├── strategies/
│   │   ├── truncation.py           # 截断配置：固定秩 / 能量阈值等
│   │   ├── windowing.py            # 加窗与帧边界（配合 OLA、F-02）
│   │   └── grouping.py             # W-correlation 分量分组
│   ├── exceptions.py               # 领域异常基类
│   └── linalg_errors.py            # 线代/SVD 相关异常
├── io/                             # 解码白名单、流式读、立体声供给（配合 F-01 内存）
├── facade/
│   ├── purifier.py                 # AudioPurifier：路径校验、整文件流程
│   ├── soundfile_ola.py            # soundfile 驱动 OLA：分帧、单帧去噪、叠接写出
│   ├── pcm_producer.py             # PCM 生产者：块入有界队列（NF-01 / 预取）
│   └── ola.py                      # 重叠相加通用辅助
└── utils/                          # 日志等共享工具，与 core 算法解耦
```

## 5. 计算选型：CPU 与向量化

### 5.1 为何以 CPU + LAPACK/BLAS 为主路径

- **块尺寸与访存**：流式分帧下单帧 Hankel 矩阵维度通常属中小型；GPU 侧需 Host–Device 往返与 kernel 启动，对小矩阵往往得不偿失。
- **数值路径**：稠密 SVD 在 CPU 上由成熟 LAPACK 实现；强分支与迭代终止的算子与 CPU 控制流、向量指令更匹配。
- **可复现基准**：与 PRD 中 **NF-02**（零拷贝轨迹构造、`as_strided` 与 NumPy 基准对齐）一致时，CPU 上双精度线性代数输出便于作为黄金参考，与硬件/驱动无关的舍入差异更小。

### 5.2 为何在 CPU 实现上强调向量化思想

此处「向量化」指：**在 CPU 上让数据以连续、分块、可被 SIMD 与缓存有效利用的方式参与计算**，并把数值热点留在 **BLAS/LAPACK/编译后的 NumPy 内核**中，而不是在 Python 层用标量循环逐元素拼矩阵。原因包括：

- **与数学对象一致**：Hankel 块矩阵、乘法与 SVD 属于**稠密线性代数**；LAPACK/BLAS 的实现本身就是面向寄存器与向量指令的。在 CPU 上坚持向量化路径，是**算法语义与执行模型对齐**，而非额外技巧。
- **规避 Python 解释开销**：对 $O(LK)$ 量级矩阵若在 Python 中用多重 `for` 逐元素写入，会叠加解释器、对象分配与动态调度成本，且难以被 JIT 稳定优化；通过 **NumPy 视图、步长与底层 C/Fortran 内核** 一次完成大块运算，才能把时间花在**浮点**上。
- **访存与带宽**：CPU 对**顺序/分块访存**远敏感于「随机标量读写」；向量化布局（含合理 `stride`）更利于缓存行利用与 SIMD，与单帧矩阵尺度下的**内存带宽**约束相符。
- **落实 NF-02**：PRD 要求**零拷贝**构造轨迹矩阵；这直接排斥「手写循环填元素」式构造，等价于在工程上采用**数据平面向量化**（视图 + 底层算子）。
- **可测试与可对比**：向量化路径落在少数可命名算子（矩阵构建、SVD）上，便于单测与回归；Python 细循环混写易导致平台相关的时间与舍入差异，不利于「黄金基准」。

### 5.3 与仓库模块的对应：何处优先向量化

下表按 §4 源码树，说明**数值热点**上宜采用的向量化思路（与 §5.2 一致）；**编排与配置**层本身不写重循环。

| 区域 | 模块 / 路径 | 向量化思路（要点） |
|------|----------------|---------------------|
| 核心链 | `stages/a_hankel` | Hankel 轨迹矩阵用 **视图 + 步长**（如 `as_strided`）构造，禁止 Python 逐元写入 |
| 核心链 | `stages/b_multichannel` | 左右块 **拼接/分块** 在 `ndarray` 上一次性完成，避免按元素拼列表再 `array()` |
| 核心链 | `stages/c_svd` | **SVD** 走 `numpy.linalg` / LAPACK；截断为切片或掩膜作用于奇异值向量，避免手写全矩阵三重循环 |
| 核心链 | `stages/d_diagonal` | **对角平均**：用按对角索引的向量化聚合（或等价 `np` 算子），避免对 $(i,j)$ 纯 Python 二重循环 |
| 策略 | `strategies/windowing` | 帧与窗函数 **逐点乘**（broadcast / `multiply`），整块更新 |
| 策略 | `strategies/grouping` | **W-correlation**、相关矩阵与张量运算，优先 **NumPy/BLAS** 表达，避免手写分量间 Python 循环 |
| 策略 | `strategies/truncation` | 多为**配置数据**（秩、能量阈值）；无热点，但传入 `c_svd` 的参数应支持**向量化后的张量契约** |
| 编排 | `core/process_frame` | **仅调度** A→B→C→D，不把数值循环写在编排函数内 |
| I/O | `io/` | 解码后的 PCM 以 **连续 `float64` 缓冲** 进入核心；分块读时保持切片视图，避免逐样本 Python 处理 |
| 门面 | `facade/ola` 等 | **重叠相加**：按 hop 对齐的向量累加、窗加权，用整块数组运算完成帧合成 |

**说明**：`facade/purifier`、`pcm_producer` 以**流程与队列**为主，热点仍在 `stages`；`utils/` 若含数值辅助，亦应遵循「小块 NumPy、大块不调 Python 内层循环」。

### 5.4 工程落点：NumPy 与 BLAS

- Hankel 嵌入等应通过 **NumPy 视图与步长** 避免 Python 层显式循环构造大矩阵（与 NF-02、§5.2–§5.3 一致）。
- 块矩阵乘法与 SVD 依赖底层 **BLAS/LAPACK**；环境差异主要体现在链接的库与线程数，不改变算法语义。

### 5.5 与 GPU 的关系

若未来引入 GPU，应作为**可选加速路径**，且不得破坏 CPU 黄金路径的可复现性与 CLI 契约；详细流式与 I/O 解耦见 [`architecture_design.md`](architecture_design.md)。

## 6. 测试与 CI（约定）

静态检查与测试配置见仓库根目录 [`pyproject.toml`](../pyproject.toml) 与 [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)。各分支上测试覆盖面可能不同，以该分支的 `Makefile` / CI 为准。