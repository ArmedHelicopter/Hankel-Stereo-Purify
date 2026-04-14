# 需求规格说明书 (PRD): Hankel-Stereo-Purify

## 1. 项目概述 (Project Overview)
本项目旨在开发一套基于多通道奇异谱分析 (MSSA) 的高保真音频降噪系统。针对 1950 年代早期模拟双声道录音（特定输入源：Jascha Heifetz, RCA "Living Stereo" FLAC 文件），系统需在滤除宽带高斯类底噪（Tape Hiss）的同时，严格保全原始音频的高频谐波结构与声道间绝对相位差。

## 2. 数学模块与算法流水线 (Algorithm Pipeline)
本系统核心降噪逻辑由四个串行的数学模块构成。输入数据为离散时间序列 $X = (x_1, x_2, \dots, x_N)$，核心目标是通过空间投影实现非平稳信号与高斯底噪的正交分离。

### 模块 A：Hankel 矩阵化模块 (Embedding & Hankelization)
* **功能定义：** 将一维音频时间序列映射为高维滞后相空间矩阵。
* **数学实现：** 设定窗口长度 $L$，列数 $K = N - L + 1$。构造矩阵 $H$，使得矩阵元素满足 $h_{i,j} = x_{i+j-1}$。该矩阵必须具有反角线元素相等的 Hankel 结构特性。
* **输入/输出：**
    * 输入：一维序列 $X$ (长度 $N$)，标量 $L$。
    * 输出：二维矩阵 $H \in \mathbb{R}^{L \times K}$。



### 模块 B：MSSA 块矩阵组合模块 (Multichannel Block Construction)
* **功能定义：** 建立立体声（双声道）信号的空间相干性约束。
* **数学实现：** 分别对左声道序列 $X_{left}$ 和右声道序列 $X_{right}$ 执行模块 A，生成两个 Hankel 矩阵 $H_L$ 和 $H_R$。将其水平拼接，构造多通道块 Hankel 矩阵 $\mathbf{X}_{total}$：
    $$\mathbf{X}_{total} = [H_L, H_R] \in \mathbb{R}^{L \times 2K}$$

### 模块 C：SVD 正交分解与截断模块 (SVD & Subspace Truncation)
* **功能定义：** 求解块 Hankel 矩阵的主成分方向，剥离噪声子空间。
* **数学实现：**
    1.  对 $\mathbf{X}_{total}$ 执行奇异值分解：$\mathbf{X}_{total} = U \Sigma V^T$。
    2.  提取对角矩阵 $\Sigma$ 中的奇异值 $\sigma_i$，计算各分量的能量贡献率。
    3.  设定截断秩 $k$。保留前 $k$ 个对应于确定性信号（小提琴基频与谐波）的奇异值，将剩余对应于宽带随机底噪的奇异值置零，生成截断奇异值矩阵 $\Sigma_k$。
    4.  计算低秩近似矩阵：$\mathbf{\hat{X}}_{total} = U \Sigma_k V^T$。
* **输入/输出：**
    * 输入：二维块矩阵 $\mathbf{X}_{total}$，标量 $k$。
    * 输出：降噪后的低秩二维块矩阵 $\mathbf{\hat{X}}_{total}$。



### 模块 D：对角平均化重构模块 (Diagonal Averaging)
* **功能定义：** 将降噪后的低秩近似矩阵逆向映射回一维时间序列。
* **数学实现：** 由于截断重构后的矩阵 $\mathbf{\hat{X}}_{total}$ 破坏了严格的 Hankel 结构，必须沿其次对角线（反斜对角线）对元素求平均值。对于子矩阵中的每一项，通过计算 $\hat{x}_n = \frac{1}{|S_n|} \sum_{(i,j) \in S_n} \hat{h}_{i,j}$ （其中 $S_n$ 为第 $n$ 条反角线上的元素集合），重构出滤波后的离散序列。
* **输入/输出：**
    * 输入：低秩二维块矩阵 $\mathbf{\hat{X}}_{total}$。
    * 输出：降噪后的左声道一维序列 $\hat{X}_{left}$ 与右声道一维序列 $\hat{X}_{right}$。

## 3. 功能需求 (Functional Requirements)
* **F-01 [I/O 管线]:** 原生支持超大体积（$\geq 300\text{MB}$）FLAC 无损格式的流式读取与多通道 PCM 数据解析。
* **F-02 [数据分块]:** 实现基于短时平稳性假设的重叠分帧处理（Overlap-Add 结构），支持自定义帧长（Frame Size）与跳步（Hop Size），并应用加窗函数（如 Hanning 窗）以抑制截断泄漏。
* **F-03 [联合降噪]:** 构建左右声道特征联合分解矩阵（即模块 B），避免对单通道独立处理所引发的声像漂移现象。
* **F-04 [特征界定]:** 提供奇异值累计能量贡献率计算接口，支持通过预设阈值或硬截断秩 $k$ 自动切分信号子空间与噪声子空间。

## 4. 非功能性需求与系统边界 (Non-Functional Requirements)
* **NF-01 [内存管理限制]:** 针对高分辨率音频导致的高维状态空间灾难，系统必须采用流式批处理架构。在处理完整 $300\text{MB}$ FLAC 文件时，运行时峰值内存占用 (Peak Memory Footprint) 必须严格控制在 $2\text{GB}$ 以内，杜绝 OOM (Out-Of-Memory) 异常。
* **NF-02 [计算资源优化]:** 在构建状态轨迹矩阵时，严禁使用高层语言的显式循环赋值。必须调用底层内存视图步长偏移机制（如 `numpy.lib.stride_tricks.as_strided`），实现零拷贝（Zero-copy）的数据重塑，优化总线访存带宽消耗。
* **NF-03 [声学物理指标]:** 降噪处理后的重构信号，其左右声道在全频带内的相位响应差与原信号的残差应趋近于零（即实现严格的 Phase-locking）；梅尔频谱图（Mel-spectrogram）对比需证明 $10\text{kHz}$ 以上的高频谐波能量未发生显著衰减。

## 5. 验收标准 (Acceptance Criteria)
1. 成功读取 $300\text{MB}$ 测试用 FLAC 文件，并无报错输出同等采样率的重构 FLAC 文件。
2. 性能探针（Profiler）数据证明单线程处理全流程峰值内存消耗 $< 2\text{GB}$。
3. 算法输出的截断误差与对角平均重构误差在理论容差范围内。