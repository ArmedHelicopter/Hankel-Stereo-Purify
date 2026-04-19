# main 与 tutorial 分支对比（快照）

## 撰写原因（为何需要本文档）

- **可追溯**：评审与协作者不必自行执行 `git diff`；差异有单一入口说明，并可对照固定 commit。
- **防误解**：`main` 与 `tutorial` **不是**两套无关仓库，而是**同一历史线上的前后状态**（`tutorial` 在 `main` 之上追加提交）。避免把 Phase0 规划写成与实现脱节的空话。
- **支撑倒推**：Phase0 文档是「从可交付实现归纳出的合理前置设计」；本表证明归纳对象来自**可枚举、可核对**的分支差异。
- **协作与 onboarding**：先读本文再读仓库根目录 [`README.md`](../../README.md) 与 [`docs/phase0/README.md`](README.md)，理解「骨架分支 vs 可交付分支」成本最低。

## 元信息（可复现）

| 项目 | 值 |
|------|-----|
| 快照日期 | 2026-04-19 |
| `git merge-base main tutorial` | `db2cc151eb2d995c1a5bf057bae1ff2799f1c5f3`（与替换骨架**前**的 `main` 顶端一致） |
| 本文对比的 `main`（旧顶端） | `db2cc151eb2d995c1a5bf057bae1ff2799f1c5f3` |
| `tutorial` 顶端 | `ff9f489a2634adf743f27db95c06cda2d2d5411d` |

**复现命令**（只读）：

```bash
git merge-base main tutorial
git diff --shortstat main...tutorial
git diff --stat main...tutorial
git log --oneline main..tutorial
```

## 量级摘要（`tutorial` 相对上述 `main`）

- **82 files changed**, **6004 insertions(+), 373 deletions(-)**（`git diff --shortstat main...tutorial` 输出）。

## 分类对照（目录与职责）

下列依据 `git diff --stat main...tutorial` 与 `git ls-tree` 归纳；**不**评价设计优劣，仅陈述增量。

| 区域 | `main`（旧） | `tutorial` 增量要点 |
|------|--------------|---------------------|
| 顶层 | 无 `frontend/`、`.streamlit/`、`scripts/`、`tutorial/` | 新增 **Streamlit 前端**、**脚本**（基准与辅助）、**教程 Markdown 章节** |
| `src/core/` | 含 `pipeline.py` 等早期形态 | **`process_frame` 单帧链**、`core/pipeline/` 兼容 re-export、`array_types`、`grouping`（W-correlation）等 |
| `src/facade/` | 较简 `purifier` | **OLA 引擎**、PCM 生产者、扩展门面与错误路径 |
| `src/io/` | 较简 | **格式白名单**、立体声读写封装、`io_messages`、能力探测 |
| `tests/` | 少量 | **大量回归与冒烟**（CLI、purifier 流、对角、SVD、前端辅助等） |
| `docs/` | PRD / 软件设计等 | 正文修订与 **tutorial 成稿无冲突**；产品文档仍以根目录 `docs/*.md` 为准参阅 |
| CI / 工具 | 较旧 `actions` | 较新 checkout/setup、`pyproject.toml` 中 pytest 覆盖率门槛等 |

> 若当前检出的是 **已替换为 Phase0 骨架后的 `main`**，则与 `tutorial` 的差异会**大于**上表统计；上表锚定的是 **旧 `main`（`db2cc15`）↔ `tutorial`（`ff9f489`）**，用于说明可交付增量从何而来。

## 与 Phase0 文档集的关系

- **本文档**：分支与树级**事实**（文件增删、命令可复现）。
- [`04_reverse_from_tutorial.md`](04_reverse_from_tutorial.md)：Phase0 **条目 → tutorial 模块/路径** 的映射（需求溯源）。
- [`01_scope_and_non_goals.md`](01_scope_and_non_goals.md) 等：规划视角的**目标与边界**，应与上两类交叉引用，而非重复粘贴实现代码。
