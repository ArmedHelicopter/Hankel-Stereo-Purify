# 软件设计与架构规约 (Software Design Specification)

> **适用性**：数学目标与分层适用于完整产品。**工程实现、路径与 API 以当前仓库 `src/` 为准**（完整管线在 `tutorial` 分支上维护）。环境与分支见 [`SETUP_AND_BUILD.md`](SETUP_AND_BUILD.md)、[`BRANCHES.md`](BRANCHES.md)；文档索引见 [`README.md`](README.md)。

## 1. 架构原则 (Architectural Principles)

本项目在实现上强调：**可验证的数值路径**、**清晰的 I/O 边界**、**薄门面**。核心 MSSA 步骤以普通函数与可调用对象表达，避免仅为「模式」而增加与当前代码不一致的调度层。

- **高内聚、低耦合**：算法留在 `src/core/stages` 与 `strategies`；流式与 OLA 在门面与 I/O 层组合。
- **异常不吞没**：门面在映射数值或配置错误时保留 `__cause__` 链，并提供可机读异常类型字段供 CLI 分桶。
- **敏感配置外置**：不在仓库中硬编码密钥或令牌。

## 2. 当前实现要点（与仓库代码一致）

### 2.1 单帧 MSSA 数值链（`process_frame`）

单帧处理由 `process_frame`（`src/core/process_frame.py`）顺序调用：

1. `hankel_embed`（`a_hankel.py`）
2. `combine_hankel_blocks`（`b_multichannel.py`）
3. `make_svd_step(...)` 返回的 **可调用对象**（`c_svd.py`）
4. `diagonal_reconstruct`（`d_diagonal.py`）

门面（如 `AudioPurifier`）通过 `functools.partial(process_frame, window_length=..., svd_step=...)` 绑定参数。若存在 `src/core/pipeline/__init__.py`，其职责仅为对 `process_frame` 做 re-export，便于兼容 `from ...pipeline import process_frame` 类导入。

### 2.2 截断配置（非经典 Strategy 多态）

- `FixedRankStrategy` 与 `EnergyThresholdStrategy`（`truncation.py`）为**具体配置类型**；`TruncationStrategy` 在源码中为二者的**类型别名**，**不是**必须被继承的抽象接口。
- `make_svd_step` 在**构造可调用对象时**按具体类型分支一次；运行期每帧不在 `isinstance` 上反复派发。

### 2.3 外观（Facade）与流式路径

- `AudioPurifier`（`facade/purifier.py`）：`process_file`、路径与配置校验、构造单帧去噪函数；组合 `SoundfileOlaEngine`（`soundfile_ola.py`）执行整文件流式路径；必要时配合 `pcm_producer` 等有界队列。
- CLI：`src/cli.py`。

## 3. 标准化工程目录映射（当前仓库典型布局）

`src/` 结构随演进可能微调；典型布局如下：

```text
src/
├── cli.py
├── core/
│   ├── array_types.py
│   ├── process_frame.py
│   ├── pipeline/          # 可选：仅 re-export process_frame
│   ├── stages/
│   │   ├── a_hankel.py
│   │   ├── b_multichannel.py
│   │   ├── c_svd.py
│   │   └── d_diagonal.py
│   ├── strategies/
│   │   ├── truncation.py
│   │   ├── windowing.py
│   │   └── grouping.py
│   ├── exceptions.py
│   └── linalg_errors.py
├── io/
├── facade/
│   ├── purifier.py
│   ├── soundfile_ola.py
│   ├── pcm_producer.py
│   └── ola.py
└── utils/
```

## 4. 计算选型：CPU 与向量化

### 4.1 为何以 CPU + LAPACK/BLAS 为主路径

- **块尺寸与访存**：流式分帧下单帧 Hankel 矩阵维度通常属中小型；GPU 侧需 Host–Device 往返与 kernel 启动，对小矩阵往往得不偿失。
- **数值路径**：稠密 SVD 在 CPU 上由成熟 LAPACK 实现；强分支与迭代终止的算子与 CPU 控制流、向量指令更匹配。
- **可复现基准**：与 PRD 中 **NF-02**（零拷贝轨迹构造、`as_strided` 与 NumPy 基准对齐）一致时，CPU 上双精度线性代数输出便于作为黄金参考，与硬件/驱动无关的舍入差异更小。

### 4.2 向量化与 BLAS

- Hankel 嵌入等应通过 **NumPy 视图与步长** 避免 Python 层显式循环构造大矩阵（与 NF-02 对齐）。
- 块矩阵乘法与 SVD 依赖底层 **BLAS/LAPACK**；环境差异主要体现在链接的库与线程数，不改变算法语义。

### 4.3 与 GPU 的关系

若未来引入 GPU，应作为**可选加速路径**，且不得破坏 CPU 黄金路径的可复现性与 CLI 契约；详细流式与 I/O 解耦见 [`architecture_design.md`](architecture_design.md)。

## 5. 信号与算法假设

### 5.1 截断 SVD

- 对联合块 Hankel 矩阵做 **完整（或等价于确定性的）截断 SVD**，按秩或能量阈值保留主方向，其余置零后重构。
- 奇异值与左/右奇异向量直接对应可解释的子空间分解，便于与截断策略、调试和回归测试对齐。

### 5.2 相对随机 SVD

- **随机化 SVD** 在极大矩阵上可降低复杂度，但近似秩与随机种子、过采样参数相关，可解释性与逐版本比特级复现成本更高。
- 本产品设计优先采用 **确定性截断 SVD**；若将来引入随机算法，应单独标注为性能选项并明确与截断策略的契约。

### 5.3 W-correlation 与分组

- **W-correlation**（及 `strategies/grouping.py` 中的相关逻辑）用于衡量分量之间的加权相关性，支持将奇异分量分组后再重构，减轻错误分组带来的音乐性伪影。具体阈值与分组策略由配置与数据决定。

### 5.4 前几帧与平稳性

- 能量阈值、噪声底估计等可能依赖 **前几帧或短窗** 的统计；隐含假设包括：**环境噪声在标定窗内近似平稳**。若输入含剧烈电平变化、剪辑拼接或强非平稳干扰，应 **缩短/滑动标定窗、禁用单点估计或改用手动秩**，避免将非噪声结构误判为底噪。

## 6. 测试与 CI（约定）

静态检查与测试配置见仓库根目录 [`pyproject.toml`](../pyproject.toml) 与 [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)。各分支上测试覆盖面可能不同，以该分支的 `Makefile` / CI 为准。
