# 架构设计与 I/O 流式要点 (Architecture Design)

> **适用性**：本文说明**生产者-消费者与访存瓶颈**；**具体类名与线程模型以 `tutorial` 分支为准**（如 `SoundfileOlaEngine`、`pcm_producer`）。当前 **`main`** 可能无完整实现。CPU/GPU 与向量化选型见 [`software_design.md`](software_design.md) §5。

## 1. 生产者-消费者模型 (Producer-Consumer Model)

生产者-消费者用于解耦 **数据获取（I/O）** 与 **数值计算（CPU）**。

在音频降噪流式架构中的典型映射：

1. **生产者**：通过 `soundfile.blocks` 等从磁盘顺序读解码块，推入有界队列。
2. **缓冲区**：线程安全有界队列；`maxsize` 等参数约束峰值内存。
3. **消费者**：从队列取块，执行单帧 MSSA（A–D），写出或交给下一环节。

只要队列非空，计算线程可保持忙碌，从而在宏观上掩盖部分磁盘延迟。

## 2. 存储层次与 I/O 瓶颈

磁盘与内存延迟数量级差异显著；同步单线程「读—算—写」会在 I/O 等待时闲置 CPU。异步预取与有界队列可在**不突破内存预算**的前提下提高流水线利用率；细节仍受 NF-01（峰值内存）约束。

## 3. 异构计算（CPU / GPU）

本阶段核心矩阵运算以 **CPU + LAPACK/BLAS** 为基准路径；理由包括小中块矩阵上 PCIe 往返与 kernel 开销、SVD 类算子的控制流特征，以及可复现黄金向量需求。完整论述见 [`software_design.md`](software_design.md) §5，本文不重复展开。
