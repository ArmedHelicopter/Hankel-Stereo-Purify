# 文档索引（Phase0 工程基线）

## Phase0 基线包含项（核对）

用于自检「工程基线是否齐全」，**不是**功能验收清单；与 [`TUTORIAL_CODE_AUDIT_CHECKLIST.md`](TUTORIAL_CODE_AUDIT_CHECKLIST.md)（面向 **`tutorial`** 发版审计）区分使用。

- [ ] 根 [README](../README.md)：§3 环境与依赖、§4 分支、§5 构建、§6 测试数据、`make check` 等可复现
- [ ] [BRANCHES.md](BRANCHES.md)：`main` / `tutorial` 分工与切换方式明确
- [ ] [prd.md](prd.md) 与 [software_design.md](software_design.md)：需求编号（F-xx / NF-xx）与设计落点一致
- [ ] CI（[`pyproject.toml`](../pyproject.toml)、[`.github/workflows/ci.yml`](../.github/workflows/ci.yml)）与本地 `make check` 可对齐

## 推荐阅读顺序（约 5 分钟起）

1. [根目录 README](../README.md) **§4 开发分支约定** → [BRANCHES.md](BRANCHES.md)：先弄清在哪个分支开发与验证。
2. 同一 README **§3（环境与构建）**、**§7**：依赖安装与 `make check`。
3. [prd.md](prd.md) **§1 项目概述**、**§3 功能需求**：行为与编号（不必先啃 §2 全部公式）。
4. [software_design.md](software_design.md) **§2.1–§2.3**、**§4**：工程映射与 `src/` 目录树。
5. 需要流式 I/O 与队列时再读 [architecture_design.md](architecture_design.md)。
6. **非数学背景**：PRD §2 可后读；优先 **software_design §2.3–§2.4** 与 [术语表](glossary.md)。

**请勿**将下列文档当作入门首读：**[TUTORIAL_CODE_AUDIT_CHECKLIST.md](TUTORIAL_CODE_AUDIT_CHECKLIST.md)** 面向 **`tutorial`** 分支的**发版前 / Code review** 全量核对，条目多、默认读者已熟悉代码与模块名。

---

| 类别 | 文档 | 说明 |
|------|------|------|
| 环境与构建 | [根目录 README §3（环境与构建）](../README.md#环境与构建) | 依赖、分支、构建与测试命令、工具链 |
| 分支 | [`BRANCHES.md`](BRANCHES.md) | `tutorial` / `main` 用途与切换 |
| 需求 | [`prd.md`](prd.md) | 功能 F-xx、非功能 NF-xx、数学模块 A–D |
| 设计 | [`software_design.md`](software_design.md) | 数学建模（含算法假设）、实现映射、`process_frame` 链、计算选型、目录结构 |
| 架构补充 | [`architecture_design.md`](architecture_design.md) | 流式 I/O 与生产者-消费者（压缩版） |
| 术语 | [`glossary.md`](glossary.md) | MSSA、Hankel、OLA 等简释 |
| 协作与计划 | [`internal/development_plan.md`](internal/development_plan.md) | 历史协作与里程碑（非当前基线必读） |
| 质量与审计 | [`TUTORIAL_CODE_AUDIT_CHECKLIST.md`](TUTORIAL_CODE_AUDIT_CHECKLIST.md) | **tutorial** 发版前 / Code review 检查项（非入门路径） |
| 环境与构建（重定向） | [`SETUP_AND_BUILD.md`](SETUP_AND_BUILD.md) | 完整说明见根 README，避免重复维护 |
