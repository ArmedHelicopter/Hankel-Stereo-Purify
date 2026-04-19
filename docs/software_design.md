# 软件设计与架构规约 (Software Design Specification)

> **适用性**：下文中的数学目标与分层思路适用于完整产品。**工程实现、路径与 API 以 `tutorial` 分支为准**；当前 **`main`** 为 Phase0 骨架（无完整 MSSA 管线）。分支说明与路径映射见 [`PHASE0_BRANCH_GUIDE.md`](PHASE0_BRANCH_GUIDE.md)。

## 1. 架构原则 (Architectural Principles)

本项目在实现上强调：**可验证的数值路径**、**清晰的 I/O 边界**、**薄门面**。核心 MSSA 步骤以普通函数与可调用对象表达，避免仅为「模式」而增加与当前代码不一致的调度层。

- **高内聚、低耦合**：算法留在 `src/core/stages` 与 `strategies`；流式与 OLA 在门面与 I/O 层组合。
- **异常不吞没**：门面在映射数值或配置错误时保留 `__cause__` 链，并提供可机读异常类型字段供 CLI 分桶。
- **敏感配置外置**：不在仓库中硬编码密钥或令牌。

## 2. 当前实现要点（与 `tutorial` 代码一致）

### 2.1 单帧 MSSA 数值链（`process_frame`）

单帧处理由 `process_frame`（`src/core/process_frame.py`）顺序调用：

1. `hankel_embed`（`a_hankel.py`）
2. `combine_hankel_blocks`（`b_multichannel.py`）
3. `make_svd_step(...)` 返回的 **可调用对象**（`c_svd.py`）
4. `diagonal_reconstruct`（`d_diagonal.py`）

门面（如 `AudioPurifier`）通过 `functools.partial(process_frame, window_length=..., svd_step=...)` 绑定参数。**不存在**历史上草案中的抽象基类 `MSSAStage`、也不存在对多阶段动态分发的 `Pipeline.execute` 循环。兼容场景下仅通过 `src/core/pipeline/__init__.py` 对 `process_frame` 做 re-export（若该文件存在于当前分支）。

### 2.2 截断配置（非经典 Strategy 多态）

- `FixedRankStrategy` 与 `EnergyThresholdStrategy`（`truncation.py`）为**具体配置类型**；`TruncationStrategy` 在源码中为二者的**类型别名**，**不是**必须被继承的抽象接口。
- `make_svd_step` 在**构造可调用对象时**按具体类型分支一次；运行期每帧不在 `isinstance` 上反复派发。

### 2.3 外观（Facade）与流式路径

- `AudioPurifier`（`facade/purifier.py`）：`process_file`、路径与配置校验、构造单帧去噪函数；组合 `SoundfileOlaEngine`（`soundfile_ola.py`）执行整文件流式路径；必要时配合 `pcm_producer` 等有界队列。
- CLI：`src/cli.py`。可选前端见 PRD F-05；实现位于 `frontend/`（若分支包含）。

### 2.4 与早期草案的区别（已不采用）

以下出现在早期设计草案中，**当前实现未采用**，仅作历史记录以免评审误读：

- 统一抽象基类 `MSSAStage` + `execute(data)` + 外部 Pipeline 调度器串行传张量。
- `MSSAPurifierBuilder` 链式建造者作为**唯一**实例化路径。

超参数校验在 **`AudioPurifier(...)`** 构造时完成，与「Builder 专属校验」目标等价，但形态为构造函数而非 Fluent Builder。

## 3. 标准化工程目录映射（`tutorial` 参照）

完整树以 **`tutorial`** 上 `src/` 为准；典型结构如下（随演进可能微调）：

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

## 4. 测试与 CI（约定）

静态检查与测试配置见仓库根目录 [`pyproject.toml`](../pyproject.toml) 与 [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)。**`main`** 上若仅为 Phase0 骨架，测试面可能小于 `tutorial`，以各分支实际配置为准。
