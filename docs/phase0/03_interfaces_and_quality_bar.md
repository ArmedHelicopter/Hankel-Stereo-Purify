# Phase0 接口与质量门禁

## 对外接口（目标）

- **CLI**：`python -m src.cli …`（完整参数集在 `tutorial` 的 `src/cli.py`）。
- **库**：门面类构造 + `process_file` 等（见 `tutorial` 的 `src/facade/purifier.py`）。
- **环境**：`PYTHONPATH=src` 与 `requirements*.txt` 与 README 说明一致。

当前 **Phase0 的 `main`**：`src.cli` 仅输出说明并指向 **`tutorial`**，用于证明包布局与 CI 可运行。

## 质量门禁（本仓库约定）

与 [`pyproject.toml`](../../pyproject.toml) 及 [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) 对齐：

| 工具 | 作用 |
|------|------|
| Ruff | 风格与常见错误 |
| Mypy (`strict`) | 静态类型 |
| Pytest + coverage | 测试与覆盖率（Phase0 对 `src` 设最低覆盖率门槛，见 `pyproject`） |

**说明**：Phase0 骨架代码量极小，覆盖率门槛针对 **`src`** 占位包；可交付版本在 `tutorial` 上维持更高测试面。

## 与对比文档的关系

分支级差异见 [`COMPARISON_main_vs_tutorial.md`](COMPARISON_main_vs_tutorial.md)；其中包含 CI/pyproject 在 `tutorial` 相对旧 `main` 的演进说明。
