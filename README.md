# Hankel-Stereo-Purify

> 🎵 基于多通道奇异谱分析(MSSA)的高保真音频降噪系统

## 项目概述

本项目通过 **多通道奇异谱分析 (MSSA)** 算法对立体声 FLAC 音频进行降噪。它面向高保真音频处理，强调：

- ✅ 保持左右声道相位关系，防止声像漂移
- ✅ 最大程度保留高频谐波结构
- ✅ 支持超大文件流式处理，优先零拷贝实现
- ✅ 采用现代 Python 类型契约与静态校验

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
├── .github/                           GitHub Actions 配置，自动运行 CI 检查
│   └── workflows/                     CI 工作流文件夹
├── .pre-commit-config.yaml            pre-commit 钩子配置，提交前检查代码
├── .gitignore                         忽略不需要提交的临时文件和缓存
├── README.md                          项目说明文档
├── requirements.txt                   运行时依赖包列表
├── requirements-dev.txt               开发依赖包列表
├── pyproject.toml                     Ruff 和 Mypy 等工具配置文件
├── src/                               源代码目录
│   ├── app.py                         程序入口应用定义
│   ├── cli.py                         命令行入口
│   ├── core/                          核心计算逻辑目录
│   │   ├── __init__.py                 core 包初始化文件
│   │   ├── exceptions.py               自定义异常类型定义
│   │   ├── pipeline.py                 流水线调度和阶段接口定义
│   │   ├── stages/                     分阶段处理代码
│   │   │   ├── __init__.py
│   │   │   ├── a_hankel.py             Hankel 嵌入阶段
│   │   │   ├── b_multichannel.py       多通道矩阵组合阶段
│   │   │   ├── c_svd.py                SVD 分解阶段
│   │   │   └── d_diagonal.py           对角平均化重构阶段
│   │   └── strategies/                 阶段策略实现目录
│   │       ├── __init__.py
│   │       ├── truncation.py          截断策略实现
│   │       └── windowing.py           窗口策略实现
│   ├── facade/                        对外调用接口目录
│   │   ├── __init__.py
│   │   └── purifier.py                外部调用入口和参数校验
│   ├── io/                            输入输出相关代码
│   │   ├── __init__.py
│   │   └── audio_stream.py            音频读写处理
│   └── utils/                         工具辅助代码
│       ├── __init__.py
│       └── logger.py                  日志处理
├── tests/                             测试代码目录
│   ├── __init__.py
│   └── test_a_hankel.py               Hankel 阶段单元测试
├── data/                              数据存放目录
│   ├── processed/                     处理结果文件夹
│   └── raw/                           原始音频文件夹
└── docs/                              项目文档目录
```

---

## 数据准备

本项目使用的标准测试音频文件可从以下链接下载：

🔗 **[标准测试音频文件下载](https://drive.google.com/drive/folders/14k0_B0eWyXBGHwFon9WLjC5BhwiD3KhR?usp=drive_link)**

下载后，请将 FLAC 文件放入项目的 `data/raws/` 目录：

```bash
mkdir -p data/raws
mv ~/Downloads/*.flac data/raws/
```

如果您希望保留处理后版本，可在 `data/processed/` 目录下保存输出文件。

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
PYTHONPATH=src python -m pytest tests/test_a_hankel.py
```

### 运行类型检查

```bash
PYTHONPATH=src python -m mypy src/ tests/
```

### 运行代码风格检查

```bash
python -m ruff check .
```

### 运行完整检查流程

```bash
make check
```

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

---

## 推荐阅读

- `docs/prd.md`：产品需求与算法说明
- `docs/MSSA 声学信号去噪.pdf`：MSSA 理论参考
