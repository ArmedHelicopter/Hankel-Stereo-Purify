# Hankel-Stereo-Purify

基于多通道奇异谱分析（MSSA）的立体声音频降噪项目。

## 分支说明（必读）

| 分支 | 含义 |
|------|------|
| **`main`** | **Phase0**：规划骨架 + 文档；占位 CLI；**不包含**完整 MSSA 管线。 |
| **`tutorial`** | **可交付实现**：完整 `src/`、测试与教程文档。 |

```bash
git checkout tutorial
```

分支策略、`main` 与 `tutorial` 对照及 Phase0 路径映射见 **[`docs/PHASE0_BRANCH_GUIDE.md`](docs/PHASE0_BRANCH_GUIDE.md)**。

## Phase0（`main`）快速开始

**环境**：Python **3.10+**。

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
export PYTHONPATH=src
python -m src.cli --help
```

**本地检查**（与 CI 对齐）：

```bash
make check   # lint + typecheck + test；需 Makefile 中命令可用
# 或手动：
ruff check src tests
PYTHONPATH=src mypy src tests
PYTHONPATH=src pytest tests/
```

## 可交付版本（`tutorial`）

安装、CLI 参数与大文件流式说明，请在切换到 **`tutorial`** 后阅读该分支上的 README 与 `docs/`。

## 仓库内其他文档

根目录 [`docs/`](docs/) 保留 PRD、软件设计等产品文档（面向**完整系统**）；分支与 Phase0 说明以 **[`docs/PHASE0_BRANCH_GUIDE.md`](docs/PHASE0_BRANCH_GUIDE.md)** 为准。

## 许可与引用

沿用本仓库既有许可与课程引用约定；若以 Phase0 仓库提交作业，请按课程要求标明分支与 commit。
