# Phase0 规划文档索引

本目录收录 **Phase0（规划与骨架）** 材料。当前 **`main` 分支** 刻意保持为**轻量骨架**；**可运行的 MSSA 降噪管线、测试全集与可选前端**在 **`tutorial` 分支**。

## 为何 split `main` 与 `tutorial`

- **`main`**：合同/课程意义上的「Phase0 交付」——目录小、依赖少、文档可读，便于评审规划与分支策略。
- **`tutorial`**：同一 Git 历史上后续提交叠加的**可交付实现**（见 [`COMPARISON_main_vs_tutorial.md`](COMPARISON_main_vs_tutorial.md) 中的统计与对照）。

## 阅读顺序

1. [`COMPARISON_main_vs_tutorial.md`](COMPARISON_main_vs_tutorial.md) — `main`（旧顶端）与 `tutorial` 的差异事实与撰写原因。
2. [`01_scope_and_non_goals.md`](01_scope_and_non_goals.md) — 范围与非目标。
3. [`02_architecture_direction.md`](02_architecture_direction.md) — 分层与原则（规划版）。
4. [`03_interfaces_and_quality_bar.md`](03_interfaces_and_quality_bar.md) — 接口与质量门禁。
5. [`04_reverse_from_tutorial.md`](04_reverse_from_tutorial.md) — Phase0 条目到 `tutorial` 实现路径的映射。

仓库根目录另有 [`prd.md`](../prd.md)、[`software_design.md`](../software_design.md) 等：它们描述**完整产品**叙事；在 Phase0 骨架上请以 **本目录** 与 **对比文档** 为准理解当前分支职责，需要实现细节时 **checkout `tutorial`**。
