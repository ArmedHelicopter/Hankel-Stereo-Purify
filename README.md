# Hankel-Stereo-Purify

> 🎵 基于多通道奇异谱分析(MSSA)的高保真音频降噪系统

## 项目概述

本项目通过 **多通道奇异谱分析 (MSSA)** 算法对立体声音频（**FLAC / WAV / AIFF / OGG** 等 libsndfile 可读容器）进行降噪。它面向高保真音频处理，强调：

- ✅ 保持左右声道相位关系，防止声像漂移
- ✅ 最大程度保留高频谐波结构
- ✅ 支持超大文件流式处理，优先零拷贝实现
- ✅ 采用现代 Python 类型契约与静态校验

**运行环境**：**Python 3.10+**（与 [`.github/workflows/ci.yml`](.github/workflows/ci.yml) 矩阵、`pyproject.toml` 中 `mypy` 的 `python_version` 一致）。

### MVP 范围与后续规划

| 内容 | 状态 |
|------|------|
| 数据平面 CLI、立体声 PCM（多格式 I/O）、OLA、固定秩或能量阈值截断 | **本仓库 MVP** |
| Streamlit/EDA 前端（PRD F-05） | **可选**（见下文「可选前端」；异步生产者-消费者流水线仍属二期其他项） |

**W-correlation**：（1）**离线**：[`src/core/strategies/grouping.py`](src/core/strategies/grouping.py) 的 `compute_w_correlation_matrix` 可对分量矩阵做分组评估。（2）**管线内**：[`src/core/stages/c_svd.py`](src/core/stages/c_svd.py) 的 `CSVDStage` 支持 `w_corr_threshold` 与 `window_length`（`L`）。**CLI** 可选 `--w-corr-threshold`（与 `-L` 配合）；**未传该参数时**行为与旧版一致（不做 W 过滤）。`MSSAPurifierBuilder.set_w_corr_threshold(...)` 供库用户使用。

运行时依赖见 [`requirements.txt`](requirements.txt)（核心栈仅 `numpy` / `scipy` / `soundfile` / `tqdm` / `colorlog`；**未**将 `matplotlib`、`librosa` 纳入核心安装，以减小体积。可选 Streamlit 前端见 [`requirements-frontend.txt`](requirements-frontend.txt)）。

### 可选前端（PRD F-05 / NF-05）

仓库提供 **Streamlit** 本地控制平面，目录为 [`frontend/`](frontend/)，依赖单独列出（**不**并入核心 [`requirements.txt`](requirements.txt)）：

```bash
python -m pip install -r requirements-frontend.txt
```

在项目根目录执行：

```bash
streamlit run frontend/app.py
```

终端会打印本地 URL（多为 `http://localhost:8501`）。[`frontend/app.py`](frontend/app.py) 已将仓库根目录加入 `sys.path`，一般**无需**再设置 `PYTHONPATH=src`。仓库内 [`.streamlit/config.toml`](.streamlit/config.toml) 提供橙白琥珀暗色基底；页面级「录音窗口」复古未来主义样式（含 CRT 扫描线叠层）由 [`frontend/app.py`](frontend/app.py) 注入 CSS，与 PRD F-05 逻辑无关。若在 **WSL2** 中 Windows 浏览器无法打开 `localhost:8501`，可尝试 `streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501` 后用 WSL 的局域网地址访问，或检查防火墙与端口转发。`soundfile` 仍依赖系统已安装的 **libsndfile**（与 CLI 相同；例如 Ubuntu/Debian 常见为 `libsndfile1`，依发行版而定）。

**能力概要**：

- **EDA 频谱**：仅按「预览时长」从输入文件读取**有限帧数**的 PCM（`soundfile` 限制 `frames`），在内存中做短片段降噪与 `matplotlib` 频谱对比；**不会**为展示而整文件读入内存。
- **全量批处理**：通过 **`subprocess`** 调用 `python -m src.cli …`，传入绝对路径与 `-L`、`-k` / `--energy-fraction` 等参数；子进程独立运行，与 Streamlit 进程内存隔离（**NF-05**）。
- 前端**不得**替代 CLI 处理 GB 级文件的主路径；大文件请始终使用 CLI 或「全量批处理」页启动子进程。
- **全量批处理**侧栏可设置「CLI 子进程超时（秒）」：`0` 表示不限制；避免在慢速 NFS/SMB 或异常卡死时 Streamlit 界面永久无响应。

### 阻塞读与网络文件系统（NFS / SMB）

`libsndfile` / `soundfile` 的阻塞读取**没有应用层超时**（见 [`src/io/audio_stream.py`](src/io/audio_stream.py) 模块说明）。输入位于慢速或挂起的网络挂载上时，**进程可能长时间无响应**；排障可设 **`HSP_LOG_IO_TRACE=1`** 观察打开/读块日志。建议大文件使用**本地磁盘**路径；需要可中断的全量任务时优先在终端直接运行 `python -m src.cli`，或为 Streamlit 子进程设置**有限超时**。

---

## 目标读者

本说明面向刚接触项目的大一同学，帮助你快速完成：

- Python 环境初始化
- 本地依赖安装
- 代码结构理解
- 开发与调试流程

---

## 项目结构图

```text
Hankel-Stereo-Purify/                  项目根目录
├── .github/                           GitHub Actions（Ruff、mypy、pytest）
│   └── workflows/
├── .pre-commit-config.yaml            本地提交前检查
├── README.md
├── requirements.txt                   核心运行时依赖（CLI / 库）
├── requirements-dev.txt               开发 + CI（含 pytest-cov；覆盖率阈值见 pyproject）
├── requirements-frontend.txt          可选 Streamlit 前端（含 -r requirements.txt）
├── .streamlit/config.toml             本地 Streamlit 主题（橙白琥珀，与前端「录音窗口」UI 一致）
├── pyproject.toml                     Ruff、mypy、pytest 等工具配置
├── Makefile                           make check（与 CI 对齐：lint + typecheck + test）
├── src/
│   ├── cli.py                         命令行入口
│   ├── core/
│   │   ├── exceptions.py
│   │   ├── linalg_errors.py           数值/线性代数异常类型聚合（供门面等）
│   │   ├── pipeline.py                Pipeline 与各 MSSAStage
│   │   ├── stages/
│   │   │   ├── a_hankel.py
│   │   │   ├── b_multichannel.py
│   │   │   ├── c_svd.py
│   │   │   └── d_diagonal.py
│   │   └── strategies/
│   │       ├── grouping.py            W-correlation；管线内可选过滤
│   │       ├── truncation.py
│   │       └── windowing.py
│   ├── facade/
│   │   ├── purifier.py                AudioPurifier / MSSAPurifierBuilder
│   │   ├── soundfile_ola.py           soundfile 路径下 OLA + PCM 队列（由 purifier 混入）
│   │   ├── pcm_producer.py            有界队列 + 生产者线程
│   │   └── ola.py                     帧起点、Hanning、OLA 辅助
│   ├── io/
│   │   ├── audio_stream.py
│   │   ├── audio_formats.py           后缀白名单、写出参数
│   │   ├── stereo_soundfile.py
│   │   ├── sndfile_capabilities.py
│   │   └── io_messages.py             I/O 层统一错误文案（供 audio_stream 等）
│   └── utils/
│       └── logger.py
├── frontend/                          可选 Streamlit（PRD F-05）
├── tutorial/                          实战教材入口：TUTORIAL_INDEX.md（第 0～6 章）
├── scripts/
│   ├── benchmark_pipeline.py
│   ├── estimate_ola_frames.py         OLA 帧数 / min(L,2K) 粗算
│   └── run_with_peak_rss.sh
├── tests/                             pytest；覆盖率 `--cov-fail-under` 见 pyproject
├── data/
│   ├── raw/
│   └── processed/
└── docs/                              PRD、架构与算法参考
```

### 依赖文件一览

| 文件 | 用途 |
|------|------|
| [`requirements.txt`](requirements.txt) | 安装核心库后即可运行 `python -m src.cli` |
| [`requirements-dev.txt`](requirements-dev.txt) | 本地开发与 CI（`pytest`、`ruff`、`mypy`、`pre-commit` 等） |
| [`requirements-frontend.txt`](requirements-frontend.txt) | 可选：Streamlit + `matplotlib`（在核心依赖之上追加） |

### 支持的文件格式（CLI / `AudioPurifier`）

输入与输出后缀须为白名单之一（小写不敏感）：**`.flac`、`.wav`、`.aiff`、`.aif`、`.ogg`**。写出时：无损路径为 **PCM_24**（FLAC/WAV/AIFF）；**`.ogg`** 使用 **Vorbis**（有损，与旧版仅 FLAC 时的叙事不同，课程验收可仍只用 `.flac`）。实际能否打开某种容器取决于本机 **libsndfile** 编译选项。代码中可用 `from src.io import libsndfile_build_summary`（或 `python -c "from src.io.sndfile_capabilities import libsndfile_build_summary; print(libsndfile_build_summary())"`，`PYTHONPATH=src`）查看本机暴露的版本/格式列表（可能因 soundfile 版本而为 `None`）。

**与 PRD 的关系**：需求文档 **F-01** 以 FLAC 为主叙事；实现上在相同流式约束下扩展了多种容器，便于试听与对比。

---

## 数据准备

本项目使用的标准测试音频文件可从以下链接下载：

🔗 **[标准测试音频文件下载](https://drive.google.com/drive/folders/14k0_B0eWyXBGHwFon9WLjC5BhwiD3KhR?usp=drive_link)**

下载后，请将 FLAC 文件放入项目的 `data/raw/` 目录：

```bash
mkdir -p data/raw
mv ~/Downloads/*.flac data/raw/
```

如果您希望保留处理后版本，可在 `data/processed/` 目录下保存输出文件。

---

## 命令行降噪（交付使用）

在项目根目录、已安装依赖的前提下，使用 **立体声** 文件（**`.flac` / `.wav` / `.aiff` / `.ogg` 等**，见上表）作为输入。示例仍用 FLAC：

```bash
PYTHONPATH=src python -m src.cli data/raw/sample.flac data/processed/sample_out.flac \
  -L 256 -k 64
```

能量阈值截断（与 `-k` 互斥）示例：

```bash
PYTHONPATH=src python -m src.cli data/raw/sample.flac data/processed/sample_out.flac \
  -L 256 --energy-fraction 0.95
```

常用参数：

| 参数 | 含义 |
|------|------|
| `-L` / `--window-length` | Hankel 窗口长度 \(L\)（**正整数**），须与算法阶段 A 一致 |
| `-k` / `--rank` | 固定 SVD 截断秩（**正整数**）；不得超过当前帧下的矩阵秩（facade 会校验）。与 `--energy-fraction` **二选一**；若两者都不写则默认 `k=64` |
| `--energy-fraction` | 累积奇异值能量阈值，须在 **(0,1]**（CLI 入口校验）；每帧自适应秩（**不能与 `-k` 同用**） |
| `--frame-size` | OLA 每帧样本数（**正整数**）；默认由 \(L\) 推导，且须 **≥ \(L\)** |
| `--hop` | 帧移（**正整数**）；默认 `frame_size // 2`，须 **小于 frame_size** |
| `--max-memory-mb` | OLA 累加器允许占用的内存预算（Mebibytes，**须为正整数**）；超出时对大缓冲使用临时内存映射文件 |
| `--max-samples` | 若输入**每声道样本数**超过 `N` 则拒绝（可选正整数）；与下方 `HSP_MAX_SAMPLES` 二选一优先级为：**命令行优先** |
| `--w-corr-threshold` | 可选：管线内 **W-correlation** 阈值（浮点）；启用后每帧额外计算相关矩阵，**耗时显著上升**。与 `-L` 共用同一 Hankel 窗长 |

**退出码（CLI）**：`0` 成功；`1` 为 `HankelPurifyError` 中除配置类以外的失败（如 I/O、数值处理）；`2` 为 **`ConfigurationError`**（含无效参数组合、路径与输入同文件、以及超过 `--max-samples` / `HSP_MAX_SAMPLES` 等配置性拒绝）。`argparse` 解析失败通常也为非零（多为 `2`）。`KeyboardInterrupt` 不捕获，由 shell 显示为 `130` 等。极少数**未**包装为 `HankelPurifyError` 的异常（例如解释器或依赖在 `build`/`process_file` 之外的故障）仍会以退出码 `1` 结束；默认仅记录简短错误信息，设置 **`HSP_DEBUG=1`** 时 CLI 会打印完整回溯。与 [`process_file`](src/facade/purifier.py) 内部将意外错误映射为 `ProcessingError` 的路径不同。

查看帮助与版本：

```bash
PYTHONPATH=src python -m src.cli --help
PYTHONPATH=src python -m src.cli --version
```

与 PRD **NF-01**（峰值常驻内存约 2GB 水位）相关的测量，可在 Linux 上使用 GNU `time` 或仓库脚本（需可执行权限：`chmod +x scripts/run_with_peak_rss.sh`）：

```bash
./scripts/run_with_peak_rss.sh data/raw/sample.flac data/processed/sample_out.flac -- \
  -L 256 -k 64
```

在输出末尾查找 `Maximum resident set size`（单位一般为 KB）。大文件可在本地复现；CI 仅跑小样本自动化测试。

### 性能与复杂度（调参直觉）

- **主成本**：每个 OLA 帧在 Hankel 嵌入后需做一次 **SVD**（[`src/core/stages/c_svd.py`](src/core/stages/c_svd.py)）：**固定秩**时优先截断 `svds` 或 dense 截断 SVD；**能量阈值**时需每帧 **dense 完整谱** 以定秩。总时间大致随 **帧数**（由音频长度、`frame_size`、`hop` 决定）线性增长；减小 `hop` 会显著增加帧数与耗时。
- **内存**：立体声累加缓冲与可选 memmap 见 `--max-memory-mb` 与 PRD NF-01。
- **可选计时**：设置环境变量 **`HSP_PROFILE_OLA=1`**（或 `true`/`yes`）可在日志中输出整段 `_run_processing` 的 wall time（默认关闭，无额外分支开销可忽略）。
- **可选 `CSVDStage` 管线内 W-correlation**（`--w-corr-threshold` 或库 API）：冷路径含 rank-1 重构、对角平均与 \(k \times k\) 加权相关矩阵。**能量自适应秩**下仅在**第一帧**做一次完整 W 标定并冻结保留下标，后续帧只做交集掩码（**固定秩**下仍按秩缓存，与旧版一致）。**未传该参数时无此开销。** 若需对比开关前后的单帧耗时，可在项目根运行 `python scripts/benchmark_pipeline.py --w-corr-threshold <0..1>`（与 `--energy-fraction` / `--rank` 等组合；见脚本 `--help`）。

**帧数与 SVD 次数**：每帧调用一次流水线（含一次 SVD）。令 \(N\) 为样本数、\(F=\) `frame_size`、\(H=\) `hop`，帧起点个数为 \(|\texttt{list\_frame\_starts}(N,F,H)|\)，实现见 [`src/facade/ola.py`](src/facade/ola.py)（最后一帧可能延长以覆盖尾部）。快速查询：

```bash
PYTHONPATH=src python scripts/estimate_ola_frames.py <num_samples> <frame_size> <hop>
```

**单帧 SVD 规模（上界直觉）**：设 Hankel 窗长为 \(L\)（`-L`），OLA 帧长为 \(F\)（`frame_size`），则每声道 Hankel 列数 \(K=F-L+1\)，联合块矩阵形状约为 \(L \times 2K\)。**固定秩**（`-k`）：每帧对联合矩阵做截断 SVD；在 `k < min(行,列)` 时实现优先使用 `scipy.sparse.linalg.svds`，否则一次 dense `scipy.linalg.svd` 后截断（见 `src/core/stages/c_svd.py`）。**能量阈值**（`--energy-fraction`）：每帧需要完整奇异值谱以确定秩，因此对每帧做一次 dense `scipy.linalg.svd`。dense SVD 的渐近阶常记为 \(O\bigl(\min(L,\,2K)^3\bigr)\) 量级（实现依赖 LAPACK）；总耗时还乘以 **帧数**（见上式与 `estimate_ola_frames`）。可用 `scripts/estimate_ola_frames.py --window-length <L>` 在打印帧数的同时打印 \(\min(L,2K)\) 供粗算。

### 环境变量（日志与诊断）

| 变量 | 含义 |
|------|------|
| `HSP_LOG_FILE` | 若设为 `none` / `0` / `false` / `off` 或空字符串，则**不写**默认的 `logs/purify.log`，仅控制台；否则为日志文件路径（进程将在该路径创建/追加，请自行选择可信目录）。未设置时行为与旧版一致（写入 `logs/purify.log`）。若创建目录或文件失败，会自动降级为仅控制台，避免在只读工作目录下崩溃。 |
| `HSP_PROFILE_OLA` | 设为 `1`/`true`/`yes` 时输出 OLA+MSSA 段 wall time（见上）。 |
| `HSP_LOG_IO_TRACE` | 设为 `1`/`true`/`yes` 时，在打开元数据与块流路径上额外打一条 **INFO** 日志（默认关闭）。**不**提供 I/O 超时；若输入在慢速/卡死的网络盘上，进程仍可能长时间阻塞。 |
| `HSP_DEBUG` | 设为 `1`/`true`/`yes` 时，CLI 对未预期异常使用 **`logger.exception`** 打印完整回溯；未设置时仅记录异常类型与简短 `repr`（避免日志刷屏）。 |
| `HSP_MAX_SAMPLES` | 若设为**正整数**，拒绝每声道样本数大于该值的输入（与 `--max-samples` 语义相同；**未**在命令行指定 `--max-samples` 时生效）。非法非空值会在 **`MSSAPurifierBuilder().build()` / `AudioPurifier(...)` 构造阶段**抛出 `ConfigurationError`，而非等到 `process_file`。 |

### 验收与性能记录（模板）

在交付或回归时，建议记录以下字段（可粘贴到 Issue 或内部文档）：

| 项目 | 记录值 |
|------|--------|
| 输入 FLAC 大小 / 解码时长 | |
| 命令行参数（`-L`、`-k`、`--frame-size`、`--hop`、`--max-memory-mb`） | |
| Wall time（秒） | |
| Max RSS（`time -v`） | |
| 备注（机器内存、磁盘类型） | |

---

## 快捷环境配置（推荐）

下面的步骤适合初学者，建议在项目根目录执行。

### 1. 创建虚拟环境

```bash
python3 -m venv .venv
```

### 2. 激活虚拟环境

- macOS / Linux：
  ```bash
  source .venv/bin/activate
  ```
- Windows PowerShell：
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```

### 3. 安装项目依赖

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**系统库**：`soundfile` 依赖本机已安装的 **libsndfile**（例如 Ubuntu/Debian：`sudo apt install libsndfile1`）。

如需与 CI 或他人环境 **逐位对齐依赖版本**，可在虚拟环境内使用 `pip freeze > requirements.lock.txt` 存档，或使用 `pip-tools`（`pip-compile`）/`uv pip compile` 从 `requirements*.txt` 生成锁定文件后再安装；本仓库默认仍以 `requirements.txt` 中的 **下限版本** 为主。

### 4. 安装开发依赖（推荐）

```bash
python -m pip install -r requirements-dev.txt
```

> `requirements-dev.txt` 包含开发工具和预提交钩子依赖：`pre-commit`、`ruff`、`mypy`、`pytest` 等。

### 5. 验证安装是否成功

```bash
python -m pip show mypy
python -m pip show pytest
```

如果上述命令都能正常执行，就表示项目环境配置成功。

---

## 开发入门步骤

### 本地运行测试

```bash
PYTHONPATH=src python -m pytest tests/
```

### 运行类型检查

```bash
PYTHONPATH=src python -m mypy src/ tests/
```

### 运行代码风格检查

```bash
python -m ruff check .
```

（`pyproject.toml` 中 Ruff 已 `extend-exclude` [`frontend/`](frontend/)，可选 Streamlit 代码不参与核心风格门禁。）

### 运行完整检查流程

与 CI 一致：Ruff 检查、`mypy`（含 `tests/`）、全量 `pytest`：

```bash
make check
```

默认 GitHub Actions 步骤见 [`.github/workflows/ci.yml`](.github/workflows/ci.yml)（`ruff check`、`mypy src/ tests/`、`pytest tests/`）。

（`make format` 仅做 `ruff format`，不包含在 `check` 中，以免改动工作区未保存内容。）

**供应链（可选）**：维护者可在本地或自有流水线中定期运行 `pip-audit -r requirements.txt`（需另行安装 `pip-audit`）对照已知漏洞；本仓库 **不** 将该项强制纳入默认 CI。

---

## 代码规范与预提交检查（Pre-commit Hooks）

本项目使用 `pre-commit` + `Ruff` 组合，目的在于把低级的格式错误拦截在本地，避免 CI 失败。

- 先激活虚拟环境，并确保已经执行过：
  ```bash
  python -m pip install -r requirements-dev.txt
  ```
- 在首次克隆项目后，执行一次：
  ```bash
  python -m pre_commit install
  ```
- 之后的正常提交流程就是：
  ```bash
  git commit
  ```
- 如果钩子拦截了提交，说明 `Ruff` 已自动修复了格式问题（例如删除多余空格或优化 import）。
  这时请重新执行：
  ```bash
  git add .
  git commit
  ```

这个流程能把简单错误和风格问题留在本地解决，保持 CI 清洁且更高效。

---

## 为什么要加 `PYTHONPATH=src`

项目源码放在 `src/` 目录下，测试代码使用 `from src...` 方式导入。

如果不设置 `PYTHONPATH=src`，Python 可能找不到 `src` 包，从而出现导入失败。

---

## 代码开发建议

### 1. 先理解模块职责

- `src/core/pipeline.py`：定义 `MSSAStage` 抽象类和流水线调度器
- `src/core/stages/a_hankel.py`：实现 Hankel 嵌入
- `src/core/stages/b_multichannel.py`：多通道组合
- `src/core/stages/c_svd.py`：SVD 分解与截断
- `src/core/stages/d_diagonal.py`：对角平均化重构
- `src/facade/purifier.py`：对外接口与边界校验
- `src/utils/logger.py`：日志输出工具

### 2. 遵循“边界校验在 facade，核心算子裸奔”的原则

- 变量合法性检查应放在 `src/facade/purifier.py` 或 I/O 层
- 核心阶段模块不应在内部做数据 shape 验证
- 这有助于保持高性能、避免重复检查

### 3. 先写测试，再实现功能

本项目推荐使用 TDD：

- 先编写 `tests/test_a_hankel.py` 中的数学与内存共享测试
- 然后补齐 `src/core/stages/a_hankel.py` 等模块

---

## 重要提示

- 本项目使用 `numpy.typing.NDArray[np.float64]`，要求数据类型严格为 **double 精度浮点数**
- `src/core/stages/a_hankel.py` 中使用 `numpy.lib.stride_tricks.as_strided` 实现零拷贝
- 日志与进度条使用 `src/utils/logger.py` 中的 `TqdmLoggingHandler`
- CI 已配置 `PYTHONPATH=src`，避免 `from src...` 导入失败
- 输入与输出不得为同一路径（或硬链接指向同一 inode），否则 `process_file` 会拒绝并抛出 `ConfigurationError`
- 路径校验与打开文件之间存在通常的 TOCTOU 窗口；本工具按本地显式路径使用场景设计，不防御恶意同机路径竞态

---

## 推荐阅读

- [`tutorial/TUTORIAL_INDEX.md`](tutorial/TUTORIAL_INDEX.md)：代码锚点与三十分钟跟读路径（面向学习与验收）
- [`docs/prd.md`](docs/prd.md)：产品需求与算法说明
- `docs/MSSA 声学信号去噪.pdf`：MSSA 理论参考
