# 环境与构建

本文档描述本仓库的**运行环境、依赖安装、构建与校验**，作为 Phase0 工程基线的一部分。详细需求与设计见 [`prd.md`](prd.md)、[`software_design.md`](software_design.md)；文档索引见 [`README.md`](README.md)。

## 1. 运行环境

- **Python**：3.10+（与 CI、[`pyproject.toml`](../pyproject.toml) 中 `mypy` 配置一致）。
- **系统库**：`soundfile` 依赖本机已安装的 **libsndfile**（各发行版包名不同）。

## 2. 依赖安装

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

## 3. 开发分支约定

完整 MSSA 管线、CLI 与测试在 **`tutorial`** 分支上维护。检出：

```bash
git checkout tutorial
```

分支用途简述见 [`BRANCHES.md`](BRANCHES.md)。

## 4. 构建与运行

将源码根加入 Python 路径：

```bash
export PYTHONPATH=src
```

- **当前 `main`**：占位 CLI，用于 CI 与包布局验证：

```bash
python -m src.cli --help
```

- **`tutorial`**：完整降噪 CLI，参数与示例见该分支根目录 [`README.md`](../README.md) 或 `python -m src.cli --help`。

## 5. 静态检查与测试

与 [`Makefile`](../Makefile)、[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) 对齐：

```bash
make check
```

或等价：

```bash
ruff check src tests
PYTHONPATH=src mypy src tests
PYTHONPATH=src pytest tests/
```

## 6. 源码树职责（变更影响分析）

| 路径 | 职责 |
|------|------|
| `src/core/process_frame.py` | 单帧 MSSA 链编排 |
| `src/core/stages/` | Hankel、联合块、SVD 步、对角重构 |
| `src/core/strategies/` | 截断、加窗、W-correlation 分组 |
| `src/facade/` | `AudioPurifier`、OLA 引擎、PCM 生产者 |
| `src/io/` | 格式白名单、流式读、立体声封装 |
| `src/cli.py` | 命令行入口 |

## 7. 关联文档

- [`prd.md`](prd.md)：功能与非功能需求、数学模块 A–D。
- [`software_design.md`](software_design.md)：分层、计算选型、信号与算法假设、目录约定。
