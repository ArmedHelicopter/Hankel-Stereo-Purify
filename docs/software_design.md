# 软件设计与架构规约 (Software Design Specification)

## 1. 架构原则 (Architectural Principles)
本项目 (`Hankel-Stereo-Purify`) 的纯软件实现阶段严格遵循高内聚、低耦合的面向对象设计原则。为支持团队并行开发并规避代码冲突，核心数据流必须与具体数学算子的实现解耦。

## 2. 核心设计模式矩阵 (Core Design Patterns)

系统底层架构基于以下四种经典 GoF 设计模式构建：

### 2.1 流水线模式 (Pipeline / Chain of Responsibility)
* **应用场景：** 解耦 MSSA 算法的四个核心数学模块（A-Hankel化, B-通道拼接, C-SVD截断, D-对角重构）。
* **规约：** * 定义统一的抽象基类 `MSSAStage`。
  * 任何数学模块必须继承该基类并实现单一的 `execute(data)` 方法。
  * 严禁模块之间直接互相调用。所有模块由外部的 Pipeline 调度器按顺序传入和传出张量数据。

### 2.2 策略模式 (Strategy Pattern)
* **应用场景：** 动态切换算法细节（如降维截断策略、加窗平滑函数）。
* **规约：**
  * 定义 `TruncationStrategy` 接口（含 `get_k()` 方法）。
  * 派生具体策略类，如 `FixedRankStrategy`（固定秩）和 `EnergyThresholdStrategy`（能量阈值）。
  * 核心 SVD 模块只调用策略接口，不包含具体的阈值判断 `if-else` 逻辑。
  * **可选 W-correlation（`CSVDStage`）**：`w_corr_threshold` 与 `compute_w_correlation_matrix` 输出的 `W[i,0]` 比较（阈值须在 `[0,1]`，与矩阵元素同域）。能量自适应秩模式下仅在**首帧**完整计算保留分量下标，后续帧与当前秩求交后复用，避免每帧重算整张 `W`（详见 `c_svd.py` 类文档）。

### 2.3 外观模式 (Facade Pattern)
* **应用场景：** 向命令行（[`src/cli.py`](../src/cli.py)）与可选控制平面（仓库根目录 [`frontend/app.py`](../frontend/app.py)，Streamlit EDA + 全量任务子进程）隐藏重叠相加（Overlap-Add）、PCM 生产者线程与流式读取的状态机。
* **规约：**
  * 封装顶层类 `AudioPurifier`（整文件 OLA 与队列逻辑在 [`src/facade/soundfile_ola.py`](../src/facade/soundfile_ola.py) 的 `SoundfileOlaMixin` 中，由 `AudioPurifier` 混入）。
  * 对外仅暴露极其简单的 API，例如 `process_file(input_path, output_path)`。CLI 与前端不得直接操作 `numpy` 矩阵或实例化底层的 `MSSAStage` 模块；全量批处理应通过子进程调用 CLI（见 PRD NF-05）。

### 2.4 建造者模式 (Builder Pattern)
* **应用场景：** 管理系统初始化时庞杂的超参数（$L, k$, 帧长，跳步）。
* **规约：**
  * 使用 `MSSAPurifierBuilder` 提供链式调用接口（Fluent Interface）。
  * 确保系统实例化时的参数校验（如帧长必须大于跳步）在 Builder 内部完成，防止产生非法状态的处理器实例。
  * 环境变量 `HSP_MAX_SAMPLES`（与 CLI `--max-samples` 对应）在 **`build()` 构造 `AudioPurifier` 时**即参与解析；若值为非法非空整数，应在该阶段抛出 `ConfigurationError`，而非推迟到 `process_file`。

## 3. 标准化工程目录映射 (Directory Structure)

基于上述模式，代码库的 `src/` 目录结构被严格限定如下。团队成员需在各自负责的子目录内独立开发：

```text
src/
├── core/                   # 核心计算逻辑 (纯粹的数学与张量操作)
│   ├── pipeline.py         # 流水线调度器与 MSSAStage 抽象类
│   ├── stages/             # 流水线节点
│   │   ├── a_hankel.py
│   │   ├── b_multichannel.py
│   │   ├── c_svd.py
│   │   └── d_diagonal.py
│   └── strategies/         # 策略与辅助
│       ├── truncation.py   # 截断策略
│       ├── windowing.py    # 加窗策略
│       └── grouping.py     # W-correlation 矩阵（管线内可选过滤）
├── io/                     # 数据流边界（libsndfile）
│   ├── audio_stream.py     # 块读取与元数据
│   ├── audio_formats.py    # 后缀白名单与写出参数
│   ├── stereo_soundfile.py # 立体声约束等
│   ├── sndfile_capabilities.py
│   └── io_messages.py      # 统一 I/O 错误文案
├── facade/                 # 顶层外观
│   ├── purifier.py         # AudioPurifier、MSSAPurifierBuilder、process_file 入口
│   ├── soundfile_ola.py    # SoundfileOlaMixin：OLA 主循环、memmap、PCM 队列侧消费
│   ├── pcm_producer.py     # 生产者线程：有界队列 + 毒丸
│   └── ola.py              # 帧起点、Hanning、OLA 辅助
├── utils/
│   └── logger.py
└── cli.py                  # 数据平面：命令行入口
```

与 `src/` **并列**于仓库根目录：**可选** Streamlit 控制平面 [`frontend/app.py`](../frontend/app.py)（PRD F-05；依赖 [`requirements-frontend.txt`](../requirements-frontend.txt)）。数据平面主路径仍为 `src/cli.py`。