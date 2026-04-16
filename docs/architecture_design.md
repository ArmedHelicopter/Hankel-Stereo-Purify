# 架构设计与底层瓶颈分析 (Architecture Design & Bottleneck Analysis)

**与当前仓库的关系（请先读）**：下文 **§1–§3** 为计算机体系结构、存储层次与异构计算的**教学与方案对比**，用于解释「为何可能采用生产者-消费者、为何 SVD 常绑 CPU」等背景；**本仓库 MVP** 中解码由**生产者线程**经**有界队列**向主线程供 PCM 块，主线程顺序执行 OLA + MSSA（数值计算未并行化）。§1 中的异步预取对比仍属规划/教学叙述。真实威胁模型、环境变量与性能主因见 **§4** 与 [`README.md`](../README.md)「性能与复杂度」小节。

## 1. 生产者-消费者模型 (Producer-Consumer Model) 原理
生产者-消费者模型是并发编程中用于解决数据生成速率与数据处理速率不匹配的标准架构模式。其核心思想是通过引入共享的、线程安全的缓冲区，将数据的获取（I/O 绑定）与数据的计算（CPU 绑定）在空间与时间上解耦。

在音频降噪流式架构中，该模型的具体映射如下：



1. **生产者 (Producer - 磁盘 I/O 线程)：**
   负责调用底层 API（如 `soundfile.blocks`），从硬盘读取 FLAC 数据并解码为 PCM 浮点矩阵。读取完成后，将该矩阵推入缓冲区。
2. **缓冲区 (Thread-safe Bounded Queue)：**
   内存中预分配的一段有界队列。它具备互斥锁（Mutex）等同步机制，确保多线程并发访问时的指针安全。此队列的最大长度（`maxsize`）是控制峰值内存、杜绝 OOM 异常的关键硬件边界约束。
3. **消费者 (Consumer - 计算线程)：**
   主计算线程，调用底层 LAPACK 库。持续从缓冲区弹出数据块，执行高密度的数学算子（如时间复杂度为 $O(N^3)$ 的 SVD 分解），并将重构结果写入输出流。

**本质机理：** 将串行阻塞的 `[读硬盘 -> 算矩阵 -> 写硬盘]` 转换为异步并行管线，确保只要缓冲区非空，CPU 的浮点运算单元（FPU）即保持高负荷运转。

## 2. 存储器层次结构与访存瓶颈 (Memory Hierarchy & I/O Bottleneck)
硬盘吞吐量显著限制程序性能的根本原因，在于计算机体系结构中的存储器层次结构差异与冯·诺依曼架构的访存机制。

### 2.1 物理延迟的数量级差异
各级存储介质的物理响应时间存在巨大的数量级跳跃：



* **CPU 寄存器 / L1 Cache：** ~1 ns
* **物理内存 (RAM)：** ~100 ns
* **固态硬盘 (NVMe SSD)：** ~10,000 ns
* **机械硬盘 (HDD)：** ~10,000,000 ns

CPU 执行 SVD 矩阵运算时，在寄存器和 L1/L2 缓存间搬运浮点数的时间处于纳秒级。而向操作系统发起读取硬盘数据块的系统调用（System Call）时，面临的是微秒乃至毫秒级的延迟。

### 2.2 同步阻塞与流水线停顿 (Synchronous Blocking & Pipeline Stall)
在单线程同步架构中，程序严格串行执行。当系统指令读取下一帧音频流时，CPU 向磁盘控制器发出指令，当前线程随即被操作系统挂起（Blocked）。在硬盘寻道或闪存电平读取期间，CPU 完全处于闲置状态，等待 I/O 中断信号返回。此 I/O Wait 状态导致计算流水线严重停顿。

### 2.3 吞吐量失配与异步预取 (Throughput Mismatch & Prefetching)
* **I/O 密集型状态：** 若 SVD 计算耗时远小于硬盘 I/O 耗时，系统受限于硬盘的低吞吐量，整体处理帧率下降。
* **流水线解耦：** 引入生产者-消费者模型后，生产者线程可在消费者线程执行张量分解的同时，提前向硬盘发起异步预读取（Prefetching）。当当前帧计算完成时，次帧数据已驻留于缓存或主存队列中，从而从宏观层面掩盖了底层的 I/O 等待时间。

## 3. 异构计算选型：CPU 与 GPU 的架构边界 (Heterogeneous Computing: CPU vs. GPU)
在流式分帧 MSSA 架构中，系统强制选用 CPU（绑定 LAPACK/BLAS）执行核心矩阵运算，而非调用显卡（GPU），其核心依据如下：

### 3.1 物理总线约束与通信开销 (PCIe Bus Latency)
流式音频处理的输入张量特征为“高频次、小尺寸”。选用 GPU 需跨越 PCIe 总线执行 `Host -> Device` 与 `Device -> Host` 的数据显存搬移。对于微小维度的块 Hankel 矩阵，PCIe 的物理传输延迟与 CUDA Kernel 的初始化启动开销（空间复杂度 $O(N^2)$ 级别），将远超 GPU 算术逻辑单元（ALU）所节约的计算时间（时间复杂度 $O(N^3)$ 级别），导致严重的吞吐量反转。

### 3.2 算子底层架构同构性 (Micro-architectural Isomorphism)
SVD 求解（如 QR 迭代或分治法）属于强迭代数值算法，包含大量条件跳转与动态循环终止判定。
* **GPU (SIMT 架构)：** 单指令多线程架构极度契合无数据依赖的乘加运算（MAC），但处理 SVD 时会因逻辑分支导致严重的线程执行发散（Warp Divergence），算力利用率极低。
* **CPU 架构：** 具备极深的控制流水线与强悍的动态分支预测（Dynamic Branch Prediction）单元。配合 AVX 指令集，处理此类高逻辑复杂度的中小型代数算子具有显著的架构优势。



### 3.3 黄金基准模型验证定位 (Golden Model Verification)
本纯软件实现阶段的核心定位是建立无损的黄金基准模型。为确保后续在设计底层专用硬件（如基于 MXFP8 等低精度规范的专用乘加阵列）时具备绝对精确的测试对比基准，必须依赖 CPU 在 IEEE-754 双精度浮点（FP64）标准下串行产出确定的测试向量（Test Vectors），规避 GPU 并发调度引入的浮点舍入不确定性。

## 4. 与本仓库当前实现的对照（安全模型、运行时与瓶颈）

本节与 [`README.md`](../README.md) 保持一致，便于架构读者不翻应用文档也能对齐**真实代码路径**。

### 4.1 威胁模型（简化）

- 进程仅读写**用户显式传入的本地路径**，无网络服务面。
- 输入/输出扩展名由 [`src/io/audio_formats.py`](../src/io/audio_formats.py) **白名单**约束；不根据用户字符串执行外部解码器命令。
- **同一路径/硬链接**：[`AudioPurifier._validate_paths`](../src/facade/purifier.py) 使用解析路径比较与 `samefile`；若 `samefile` 因权限/跨设备失败则**保守拒绝**（`ConfigurationError`），避免无法判定是否覆盖同一 inode。校验与后续打开之间存在典型 **TOCTOU**（路径被替换）窗口；本地批处理工具按「用户显式路径」信任模型处理，不防恶意同机竞态。
- **并发**：解码路径为**生产者线程 + 有界队列 + 毒丸**（[`src/facade/purifier.py`](../src/facade/purifier.py)）；MSSA/OLA 仍在**主线程**上顺序执行，无多线程数值流水线。

### 4.2 环境变量（日志与粗测）

| 变量 | 作用（摘要） |
|------|----------------|
| `HSP_LOG_FILE` | 设为 `none`/`0`/`off` 等可关闭默认文件日志；若设为路径，进程将在该路径**创建/追加**日志文件（权限允许时），勿指向敏感目录。 |
| `HSP_PROFILE_OLA` | 设为 `1`/`true`/`yes` 时记录整段 OLA+MSSA 的 wall time。 |
| `HSP_LOG_IO_TRACE` | 设为 `1`/`true`/`yes` 时在 [`audio_stream.py`](../src/io/audio_stream.py) 打开路径时多打一条 INFO（默认关）。**不**实现读超时；NFS 等慢路径仍可能阻塞。 |
| `HSP_MAX_SAMPLES` | 正整数时拒绝超过该**每声道样本数**的输入；非法非空值在构建 [`AudioPurifier`](../src/facade/purifier.py) 时 `ConfigurationError`。命令行 `--max-samples` 优先。 |

### 4.3 性能主因与帧数估计

- **主 CPU 成本**：每 OLA 帧一次 **SVD 数值分解**（[`c_svd.py`](../src/core/stages/c_svd.py)）。**固定秩**：`k < min(m,n)` 时优先 `svds`，否则一次 dense `scipy.linalg.svd` 后截断。**能量阈值**：每帧需完整奇异值谱，故每帧一次 dense `scipy.linalg.svd`。帧数随 `list_frame_starts(N, frame_size, hop)` 增长；单帧矩阵约为 \(L \times 2K\)（\(K=\) `frame_size` \(-L+1\)），渐近阶常记 \(O(\min(L,2K)^3)\) 量级（与 README 一致）。
- **脚本**：[`scripts/estimate_ola_frames.py`](../scripts/estimate_ola_frames.py) 打印帧起点个数；可加 `--window-length L` 打印 \(\min(L,2K)\) 供粗算。

### 4.4 异常分层（面向调用方）

- **I/O / 格式**：`AudioIOError`（含 libsndfile 映射）。
- **配置**：`ConfigurationError`。
- **数值 / 处理**：`ProcessingError`（例如 `numpy.linalg.LinAlgError` 与未预期内部错误在 [`process_file`](../src/facade/purifier.py) 中映射），与上述类型均继承 `HankelPurifyError`。CLI：[`src/cli.py`](../src/cli.py) 对 `ConfigurationError` 使用退出码 **2**，其余 `HankelPurifyError` 使用 **1**（见 README「退出码」）。