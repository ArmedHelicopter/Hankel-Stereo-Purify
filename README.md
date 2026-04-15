# Hankel-Stereo-Purify

> 🎵 基于多通道奇异谱分析(MSSA)的高保真音频降噪系统

## 项目概述

本项目通过 **多通道奇异谱分析 (MSSA)** 算法为1950年代模拟双声道录音（如 Jascha Heifetz RCA "Living Stereo" 系列）进行高保真降噪。系统能够：

- ✅ 滤除宽带高斯底噪（Tape Hiss），同时保留原始音频的高频谐波结构
- ✅ 保持左右声道的绝对相位差（Phase-locking），防止声像漂移
- ✅ 支持超大文件流式处理（300MB+ FLAC 无损格式），峰值内存控制 < 2GB
- ✅ 零拷贝数据结构优化（`numpy.stride_tricks`），最大化计算效率

---

## 核心算法架构

系统从一维音频时间序列 $X = (x_1, x_2, \dots, x_N)$ 到降噪后的序列 $\hat{X}$，经过四个串联数学模块：

### 模块 A：Hankel 矩阵化 (Embedding)

将一维时间序列映射到高维滞后相空间：

- 窗口长度 $L$，列数 $K = N - L + 1$
- 构造 Hankel 矩阵 $H \in \mathbb{R}^{L \times K}$，满足 $h_{i,j} = x_{i+j-1}$
- 必须满足反角线元素相等的 Hankel 结构特性

**实现模块：** [src/core/stages/a_hankel.py](src/core/stages/a_hankel.py)

### 模块 B：MSSA 块矩阵组合 (Multichannel Block Construction)

建立立体声信号的空间相干性约束：

- 分别对左声道 $X_{left}$ 和右声道 $X_{right}$ 执行模块A，生成 $H_L$ 和 $H_R$
- 水平拼接构造多通道块 Hankel 矩阵：
  $$\mathbf{X}_{total} = [H_L, H_R] \in \mathbb{R}^{L \times 2K}$$
- 联合降噪避免单通道处理所致的声像漂移

**实现模块：** [src/core/stages/b_multichannel.py](src/core/stages/b_multichannel.py)

### 模块 C：SVD 正交分解与截断 (Subspace Truncation)

分离信号和噪声子空间：

1. 对 $\mathbf{X}_{total}$ 执行奇异值分解：$\mathbf{X}_{total} = U \Sigma V^T$
2. 计算各分量的能量贡献率，设定截断秩 $k$
3. 保留对应于确定性信号的奇异值，零化噪声分量：$\mathbf{\hat{X}}_{total} = U \Sigma_k V^T$

**参数控制：**
- 预设阈值方式：自动切分信号/噪声子空间
- 硬截断秩方式：显式指定 $k$ 值

### 模块 D：对角平均化重构 (Diagonal Averaging)

将降噪后的二维矩阵逆向映射为一维序列：

- 沿反斜对角线计算元素平均值：
  $$\hat{x}_n = \frac{1}{|S_n|} \sum_{(i,j) \in S_n} \hat{h}_{i,j}$$
- 恢复左右声道的降噪序列 $\hat{X}_{left}$ 和 $\hat{X}_{right}$

---

## 仓库结构

该项目当前已按 `docs/software_design.md` 的标准化目录映射创建占位模块，后续可以在这些子目录中完成各模块实现：

```
Hankel-Stereo-Purify/
├── README.md                          # 项目说明文档
├── requirements.txt                   # Python 依赖清单
├── src/                               # 核心算法实现
│   ├── __init__.py                    # 包初始化
│   ├── app.py                         # 可选的 EDA / 控制平面入口
│   ├── cli.py                         # 命令行入口
│   ├── core/                          # 核心计算逻辑
│   │   ├── __init__.py
│   │   ├── pipeline.py                # 流水线调度器与 MSSAStage 抽象类
│   │   ├── stages/                    # 流水线节点占位文件
│   │   │   ├── a_hankel.py
│   │   │   ├── b_multichannel.py
│   │   │   ├── c_svd.py
│   │   │   └── d_diagonal.py
│   │   └── strategies/                # 策略模式占位文件
│   │       ├── __init__.py
│   │       ├── truncation.py
│   │       └── windowing.py
│   ├── facade/                        # 外观层占位文件
│   │   ├── __init__.py
│   │   └── purifier.py
│   └── io/                            # I/O 边界层占位文件
│       ├── __init__.py
│       └── audio_stream.py
├── docs/                              # 文档与参考资料
│   ├── prd.md                         # 产品需求规格说明书 (PRD)
│   └── MSSA 声学信号去噪.pdf          # 相关学术文献
├── notebooks/                         # Jupyter 实验笔记本
└── data/                              # 数据目录
    ├── raw/                           # 原始音频文件(输入)
    └── processed/                     # 处理后的降噪音频(输出)
```

> 说明：以上 `src/` 子目录及文件为占位实现，后续可根据设计规范补充具体业务逻辑。
> 旧顶层文件路径（如 `src/audio_utils.py`、`src/hankel_matrix.py`、`src/mssa_core.py`）已被移除，项目现已统一为新 `src/` 结构路线。

---

## 安装与依赖

### 环境要求

- Python >= 3.8
- NumPy >= 1.24.0
- SciPy >= 1.10.0

### 依赖安装

```bash
pip install -r requirements.txt
```

**核心依赖说明：**

| 包名 | 版本 | 用途 |
|------|------|------|
| numpy | >=1.24.0 | 数值计算与矩阵运算 |
| scipy | >=1.10.0 | SVD 分解与科学计算 |
| librosa | >=0.10.0 | 音频特征提取与分析 |
| matplotlib | >=3.7.0 | 可视化（谱图、波形等） |
| soundfile | >=0.12.1 | FLAC 文件读写 |

---

## 数据准备

本项目需要音频文件进行处理。请按照以下步骤准备数据：

### 下载测试数据

1. **访问数据存储链接**

   点击下方链接进入 Google Drive 文件夹：
   
   🔗 **[下载音频数据集](https://drive.google.com/drive/folders/14k0_B0eWyXBGHwFon9WLjC5BhwiD3KhR?usp=drive_link)**

2. **下载所需文件**

   - 选择要处理的 FLAC 音频文件
   - 点击下载按钮存储到本地

3. **放置到项目目录**

   将下载的音频文件放入项目的 `data/raw/` 文件夹中：

   ```bash
   # 创建数据目录（如果不存在）
   mkdir -p data/raw
   
   # 将 FLAC 文件移动到此目录
   mv ~/Downloads/*.flac data/raw/
   ```

### 目录结构示例

```
data/raw/
├── input.flac              # 原始音频文件（示例名称）
├── heifetz_rca_living.flac # RCA "Living Stereo" 系列（示例）
└── ... # 其他 FLAC 文件
```

### 支持的文件格式

- **FLAC** (.flac) - 无损压缩格式，推荐格式
- 采样率：44.1 kHz / 48 kHz / 96 kHz 等
- 声道：立体声（双声道）

---

## 快速开始

### 基本使用示例

```python
from src.facade.purifier import AudioPurifier

purifier = AudioPurifier()
purifier.process_file('data/raw/input.flac', 'data/processed/output.flac')
```

---

## 核心特性

| 特性 | 说明 |
|-----|------|
| **联合多通道处理** | 避免单通道独立处理导致的声像漂移 |
| **超大文件支持** | 流式分块处理，支持 300MB+ FLAC 文件 |
| **内存高效** | 峰值内存占用 < 2GB，采用零拷贝数据结构 |
| **相位保护** | 严格保持左右声道的绝对相位差 |
| **高频保留** | 10kHz 以上的谐波能量未显著衰减 |
| **参数灵活** | 支持能量阈值自适应与硬截断秩控制 |

---

## 性能目标

- ✅ **读取性能**：成功读取 300MB+ 测试 FLAC 文件，无报错
- ✅ **内存管理**：单线程处理峰值内存 < 2GB
- ✅ **声学指标**：
  - 左右声道相位响应残差趋近于零
  - 梅尔频谱图对比证明高频谐波能量保留完整

---

## 验收标准

1. ✅ 成功处理 300MB+ FLAC 文件，输出等采样率重构文件
2. ✅ 性能探针证明峰值内存消耗 < 2GB
3. ✅ 算法截断误差与对角平均重构误差在理论容差范围内
4. ✅ 梅尔频谱与波形对比验证降噪效果

---

## 文件映射关系

```
输入 FLAC
   ↓
[src/io/audio_stream.py] 流式读取与分帧预处理
   ↓
[src/core/stages/a_hankel.py] 模块A - Hankel 矩阵化 (1D→2D)
   ↓
[src/core/stages/b_multichannel.py] 模块B - 多通道块矩阵组合
   ↓
[src/core/stages/c_svd.py] 模块C - SVD 分解与截断
   ↓
[src/core/stages/d_diagonal.py] 模块D - 对角平均化重构 (2D→1D)
   ↓
[src/facade/purifier.py] 外观层调用
```

---

## 文档与参考

- 📄 [产品需求规格说明 (PRD)](docs/prd.md) - 完整的需求定义与数学推导
- 📖 [学术文献](docs/MSSA%20声学信号去噪.pdf) - MSSA 理论基础

