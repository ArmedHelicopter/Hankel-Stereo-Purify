# Hankel-Stereo-Purify 实战教材索引

面向 AI 专业：把本仓库当作**可运行的线性代数与软件工程标本**。阅读顺序建议自上而下；每章独立成篇，可单跳。

## 课程学习目标（可验收）

每条可在对应章节的「最小公式」或「代码锚点」中找到落点；命题时可直接选用。

| 数学原理 | 代码实现 | 软件工程 | 主要章节 |
|----------|----------|----------|----------|
| 解释 Hankel 与一维序列的反对角线对应；写出 \(K=F-L+1\) 及单声道块 \(L\times K\) | 定位 `AHankelStage.execute` 与 `as_strided`；跟读 `test_a_hankel` | 说明视图与 `ascontiguousarray` 的取舍 | 01 |
| 解释联合块 \([H_L,H_R]\) 与「一次联合 SVD」的动机（相位锁定） | 定位 `BMultichannelStage`；描述 `Pipeline` 链上 `tuple`→`ndarray` 的类型变化 | 指出 PRD F-03 与 `_build_pipeline` 中四阶段组装 | 02 |
| 对比固定秩截断与能量阈值的数学假设；说出何时 full SVD | 在 `CSVDStage.execute` 标出 `FixedRankStrategy` 与能量分支分叉；指认 `(u*s)@vh` | 指认 `TruncationStrategy`、能量+W 首帧冻结语义 | 03 |
| 解释对角平均输出长度 \(L+K-1\) 与联合输出形状 | 定位 `DDiagonalStage`、`fast_diagonal_average` | 理解 D 与 OLA 除法不在同一模块的原因 | 04 |
| 复述 OLA 与短时平稳假设 | 跟读 `list_frame_starts`、`AudioPurifier` 内层循环 | 指认 `pcm_producer` 背压、memmap、`Queue` | 05 |
| 区分 `ConfigurationError` / `AudioIOError` / `ProcessingError` 适用场景 | 从 `src/cli.py` 追到 `process_file` 的异常映射 | 指认 `validate_io_paths`；说明 `test_purifier_producer_errors` 测了什么契约 | 06 |

## 三十分钟路径（验收用）

计时从打开本索引开始，完成下列步骤即算通过「能自己找到主链路与三处工程决策」：

1. **2 min**：读上表任一行，记下三个文件名待查。
2. **8 min**：打开 [`src/core/pipeline.py`](../src/core/pipeline.py)，确认 `Pipeline.execute` 如何串行调用各 `execute`。
3. **10 min**：依次打开 [`a_hankel.py`](../src/core/stages/a_hankel.py) → [`b_multichannel.py`](../src/core/stages/b_multichannel.py) → [`c_svd.py`](../src/core/stages/c_svd.py)（仅看 `CSVDStage.execute` 分支）→ [`d_diagonal.py`](../src/core/stages/d_diagonal.py)，在纸上画出单帧形状流。
4. **10 min — 三处工程决策**（指认即可，不要求背代码）：
   - **背压**：[`src/facade/pcm_producer.py`](../src/facade/pcm_producer.py) 有界队列与 `put` 超时。
   - **白名单 I/O**：[`src/io/audio_formats.py`](../src/io/audio_formats.py) `validate_io_paths` / 后缀集合。
   - **异常与退出码**：[`src/core/exceptions.py`](../src/core/exceptions.py) 分层 + [`src/cli.py`](../src/cli.py) 对 `ConfigurationError` vs 其它 `HankelPurifyError` 的 `sys.exit`。

**运行代码前**：在项目根目录设置 `PYTHONPATH=src`，详见项目根目录说明文档中的环境与命令示例。

## 本地环境与可选前端（Streamlit）

与 PRD **F-05** / **NF-05** 一致：控制平面是**可选**的；先保证核心管线可跑，再装前端专用依赖。

| 步骤 | 命令 / 说明 |
|------|-------------|
| 核心依赖 | `python -m pip install -r requirements.txt`（含 `numpy`、`scipy`、`soundfile` 等） |
| 系统库 | `soundfile` 依赖本机已安装的 **libsndfile**（各发行版包名不同；排障见项目根目录说明文档） |
| 前端额外包 | `python -m pip install -r requirements-frontend.txt`（在上一行基础上增加 `streamlit`、`matplotlib`） |
| 启动界面 | 在**仓库根目录**执行 `streamlit run frontend/app.py`；终端会打印 URL，一般为 `http://localhost:8501` |
| `PYTHONPATH` | [`frontend/app.py`](../frontend/app.py) 会把仓库根目录加入 `sys.path`，启动 Streamlit 时通常**无需**再手动设 `PYTHONPATH=src` |
| WSL2 | 若 Windows 浏览器打不开 `localhost:8501`，可试 `streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501` 后用 WSL IP 访问，或检查防火墙与端口转发 |
| 大文件 | 全量任务以 **CLI 子进程** 为主路径；界面侧「全量批处理」亦为 subprocess 隔离内存（NF-05），勿在 Streamlit 进程内整文件载入超大 PCM |

## 章节一览

| 章 | 文件 | 一句话 |
|----|------|--------|
| 0 | [00_架构总览.md](00_架构总览.md) | 一页纸：数据流、工程亮点、符号表、与后续章节映射 |
| 1 | [01_从波形到矩阵.md](01_从波形到矩阵.md) | 滑动窗口如何把一维采样变成 Hankel 矩阵（零拷贝视图） |
| 2 | [02_立体声联合块.md](02_立体声联合块.md) | 左右矩阵为何要拼成一块再算（相位锁定） |
| 3 | [03_SVD与子空间截断.md](03_SVD与子空间截断.md) | 秩、k、能量阈值在代码里对应哪几行 |
| 4 | [04_对角平均与重构.md](04_对角平均与重构.md) | 矩阵如何回到波形；为何截断会破坏 Hankel 结构 |
| 5 | [05_OLA与流式门面.md](05_OLA与流式门面.md) | 帧、hop、窗；生产者队列与 memmap 决策 |
| 6 | [06_防御性设计与测试.md](06_防御性设计与测试.md) | 异常类型、白名单 I/O、pytest 如何当契约 |

**代码锚点总览**：核心链 `src/core/pipeline.py` → 阶段 `src/core/stages/{a_hankel,b_multichannel,c_svd,d_diagonal}.py` → 门面 `src/facade/purifier.py`。

## 可选习题（各一条）

- **数学**：给定 \(F=512,\,L=256\)，手算 \(K\) 与联合矩阵列数 \(2K\)。
- **实现**：在 IDE 中从 `Pipeline.execute` 跳到 `CSVDStage.execute`，用注释标出固定秩分支与能量分支的分叉条件。
- **工程**：对照 PRD **NF-05** 与 [`frontend/app.py`](../frontend/app.py) 页首说明，用一段话解释为何全量任务应通过子进程 CLI 而非在控制平面进程内直接载入整文件。
