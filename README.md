# Hankel-Stereo-Purify

基于多通道奇异谱分析（MSSA）的立体声音频降噪项目：在滤除宽带底噪的同时保全声道相干性与高频结构。

## 1. 项目简介

本节仓库为 **Phase0 工程基线**：`main` 侧重与设计一致的目录占位与 CI；**可执行 MSSA 降噪逻辑**以 **`tutorial`** 分支为准。需求与设计的权威说明见下文「相关文档」。**Phase0 基线自检清单**（文档/CI/分支）见 [docs/README.md](docs/README.md) 文首。

## 2. 文档导航

- **[docs/README.md](docs/README.md)**：Phase0 工程基线文档索引（需求、设计、架构、审计清单等）。

<a id="环境与构建"></a>

## 3. 运行环境与依赖

### 3.1 运行环境

- **Python**：3.10+（与 CI、[pyproject.toml](pyproject.toml) 中 `mypy` 配置一致）。
- **系统库**：完整音频解码路径依赖 **`soundfile`** 与本机 **libsndfile**（各发行版包名不同）。**说明**：根目录 [requirements.txt](requirements.txt) 在 Phase0 仅声明 **numpy**（占位与 CI）；**不**包含 `soundfile`。真实 I/O 栈的依赖锁请以 **`tutorial`** 分支上的 `requirements.txt` / `requirements-dev.txt` 为准；当前 `main` 上最小 CLI 与测试**不强制**安装 soundfile。

### 3.2 依赖安装

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

## 4. 开发分支约定

完整 MSSA 实现、CLI 与测试在 **`tutorial`** 分支上维护：

```bash
git checkout tutorial
```

分支用途见 [docs/BRANCHES.md](docs/BRANCHES.md)。

## 5. 构建与运行

```bash
export PYTHONPATH=src
```

- **当前 `main`**：与设计一致的**目录占位**（`core` / `stages` / `facade` / `io` 等与 [docs/software_design.md](docs/software_design.md) §4 对齐；对外 API 多为 `NotImplementedError`）+ 最小 CLI，用于 CI 与布局校验：

```bash
python -m src.cli --help
```

- **`tutorial`**：可执行降噪实现与 CLI；参数与示例见该分支根目录本文件或 `python -m src.cli --help`。

## 6. 测试数据

项目**不**在 Git 中附带大体积音频。开发或本地试听时，请从共享目录下载测试素材；**本地原始音频统一放在** **`data/raws/`**（勿提交音频文件，见 [.gitignore](.gitignore)）。

- **下载地址（须保留于 README，勿改为占位或删除）**：[Google Drive 测试数据文件夹](https://drive.google.com/drive/folders/14k0_B0eWyXBGHwFon9WLjC5BhwiD3KhR?usp=sharing)
- **同上（可复制 URL）**：`https://drive.google.com/drive/folders/14k0_B0eWyXBGHwFon9WLjC5BhwiD3KhR?usp=sharing`（需使用有访问权限的 Google 账号；若仅见登录页，请向仓库维护者申请权限。）
- **本地放置**：在仓库根目录执行 `mkdir -p data/raws`，将下载得到的原始文件放入该目录；CLI 或脚本若以路径传参，请指向 `data/raws/...` 下具体文件。

## 7. 静态检查、测试与格式

### 7.1 与 CI 一致（`make check`）

[Makefile](Makefile) 中 `check` = **lint + typecheck + test**（**不包含** `ruff format`）：

```bash
make check
```

等价于：

```bash
ruff check src tests
PYTHONPATH=src mypy src tests
PYTHONPATH=src pytest tests/
```

与 [.github/workflows/ci.yml](.github/workflows/ci.yml) 一致；`pytest` 覆盖率等见 [pyproject.toml](pyproject.toml) 中 `[tool.pytest.ini_options]`。

### 7.2 代码格式（Ruff Format）

```bash
make format
# 或
ruff format src tests
```

可先 `make format` 再 `make check`。

### 7.3 Git 提交钩子（pre-commit，可选）

[requirements-dev.txt](requirements-dev.txt) 含 `pre-commit`；配置见 [.pre-commit-config.yaml](.pre-commit-config.yaml)（**ruff**、**ruff-format**）。

```bash
pre-commit install
pre-commit run --all-files
```

CI **未**单独执行 `pre-commit` 可执行文件；未装钩子时至少执行 `make check`（及按需 `make format`）。

## 8. 源码树职责（简述）

**可运行降噪逻辑**以 **`tutorial`** 分支为准。`main` 上目录与模块职责的**完整说明**（含树形注释）见 [docs/software_design.md](docs/software_design.md) **§4**，此处不重复维护表格。

## 9. 相关文档

- [docs/prd.md](docs/prd.md)：功能与非功能需求、数学模块 A–D。
- [docs/software_design.md](docs/software_design.md)：数学建模（含 §2.6 算法假设）、`process_frame` 链、计算选型、目录约定。
- [docs/glossary.md](docs/glossary.md)：术语简释（MSSA、Hankel、OLA 等）。
- [LICENSE](LICENSE)：开源许可（MIT）。

若需按文档导航与阅读顺序入门，优先跟随 [docs/README.md](docs/README.md) 中的「推荐阅读顺序」。
