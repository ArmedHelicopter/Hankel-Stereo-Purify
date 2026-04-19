# `tutorial` 分支全量代码审计清单

> **用途**：**质量审计、发版前或 Code review** 时的核对清单；与需求/设计文档并列，**非** Phase0 文档主阅读路径。文档索引起见 [docs/README.md](README.md)。

本文档面向 **`tutorial` 分支**（可交付实现）做**代码与运行审计**时的检查项，按模块与风险域组织。审计时请在本地 **`git checkout tutorial`** 后对照源码逐项勾选或记录结论。

**范围说明**：包含 `src/`（约 30 个模块）、`tests/`（约 30 个测试文件）、`scripts/`、`frontend/`（可选控制平面）。不包含第三方库实现。

---

## 0. 审计前准备

| 序号 | 检查项 | 备注 |
|------|--------|------|
| 0.1 | 确认分支与提交：`git rev-parse HEAD`，必要时与发布 tag 对齐 | |
| 0.2 | 环境与 CI 对齐：Python 3.10/3.11、`requirements.txt` + `requirements-dev.txt` | 见 `.github/workflows/ci.yml` |
| 0.3 | 工具链：`ruff check`、`mypy src tests`（`PYTHONPATH=src`）、`pytest`（覆盖率阈值见 `pyproject.toml` 中 `--cov-fail-under`） | |
| 0.4 | 系统依赖：本机 **libsndfile** 与 `soundfile` 可打开白名单格式 | 见 `src/io/sndfile_capabilities.py` |

---

## 1. 入口与 CLI（`src/cli.py`）

| 序号 | 检查项 | 关注点 |
|------|--------|--------|
| 1.1 | 所有参数与互斥组（`-k` / `--energy-fraction` 等）与 `AudioPurifier` 构造一致 | 避免 CLI 与门面双重校验不一致 |
| 1.2 | 正整数、路径等 `argparse` 类型转换失败时文案清晰 | `_positive_int` 等 |
| 1.3 | 退出码约定与实现一致：文档注释（ConfigurationError → 2 等） | 对照 `main()` 中 `sys.exit` |
| 1.4 | `--w-corr-threshold` 与 `-L` 等组合在帮助文案中说明依赖关系 | 与 `validate_w_corr_threshold` 一致 |
| 1.5 | 日志与异常：`ProcessingError` 打印 `origin_exception_type` / `__cause__` 链是否便于排障 | |

---

## 2. 门面与配置（`src/facade/purifier.py`）

| 序号 | 检查项 | 关注点 |
|------|--------|--------|
| 2.1 | `AudioPurifier.__init__`：`truncation_rank` 与 `energy_fraction` 互斥、`window_length` 正整数 | |
| 2.2 | `HSP_MAX_SAMPLES` 解析：非法值抛 `ConfigurationError` 且带 `from exc` | `_resolve_max_input_samples` |
| 2.3 | `w_corr_threshold` 经 `validate_w_corr_threshold` | |
| 2.4 | `process_file`：`validate_io_paths` 在打开文件前调用 | |
| 2.5 | 数值异常映射：`linalg_errors` 集合与 `except` 分支是否覆盖 `svds`/full `svd` 路径 | 文档注释要求显式扩展而非字符串 `__module__` 嗅探 |
| 2.6 | `ProcessingError` 包装是否 `raise ... from exc`，`origin_exception_type` 是否设置 | |
| 2.7 | `last resort` 的 `except Exception` 是否仅作兜底且打日志 | |

---

## 3. 单帧 MSSA 链（`src/core/process_frame.py` + `stages/`）

| 序号 | 检查项 | 关注点 |
|------|--------|--------|
| 3.1 | 四步顺序固定：Hankel → `combine_hankel_blocks` → `svd_step` → `diagonal_reconstruct` | 无隐藏 Stage 调度 |
| 3.2 | `a_hankel`：`as_strided` / 视图与文档、测试中的零拷贝断言一致 | `tests/test_a_hankel.py` |
| 3.3 | `b_multichannel`：联合块形状 `(L, 2K)` 与 stereo 约定 | |
| 3.4 | `c_svd`：固定秩与能量分支、`make_svd_step` 闭包/可调用类状态（W-correlation 缓存、能量 warm-start） | 峰值内存路径（W-correlation） |
| 3.5 | `d_diagonal`：`batched_diagonal_average` 与 `t=i+j` 聚合正确性 | `tests/test_d_diagonal.py` |
| 3.6 | `grouping`：`compute_w_correlation_matrix` 输入形状与窗口长度 | |

---

## 4. 截断策略（`src/core/strategies/truncation.py`）

| 序号 | 检查项 | 关注点 |
|------|--------|--------|
| 4.1 | `FixedRankStrategy` / `EnergyThresholdStrategy` 为数据类式配置，无运行期多态虚表 | 与 `software_design.md` 一致 |
| 4.2 | 能量阈值与 `c_svd` 中 `_energy_truncated_factors` 探测上限、回退 full `svd` 行为可理解 | `_SVDS_ENERGY_PROBE_CAP` 等 monkeypatch 点 |

---

## 5. I/O 与格式（`src/io/`）

| 序号 | 检查项 | 关注点 |
|------|--------|--------|
| 5.1 | `audio_formats.py`：输入/输出后缀白名单与 `soundfile_write_kwargs` 一致 | OGG 为 Vorbis 等文档约定 |
| 5.2 | `audio_stream.py`：仅顺序读、立体声通道数在 `stereo_soundfile` 中强制 | `require_stereo_channels` |
| 5.3 | **阻塞读无超时**：慢 NFS/挂起时行为与 README 说明一致 | 模块注释中已声明 |
| 5.4 | I/O 错误映射为 `AudioIOError` 与统一消息 | `io_messages.py` |

---

## 6. 流式 OLA 与并发（`src/facade/soundfile_ola.py`、`pcm_producer.py`、`ola.py`）

| 序号 | 检查项 | 关注点 |
|------|--------|--------|
| 6.1 | 有界队列 `PCM_QUEUE_MAXSIZE`、生产者 `put` 超时与 `abort_event` | 防死锁、毒丸 `None` 仅入队一次 |
| 6.2 | `producer_error` 列表的并发契约：仅生产者线程写入，主线程在毒丸后读 | 见 `pcm_producer` 文档字符串 |
| 6.3 | 异常或正常结束路径均调用 `_shutdown_pcm_producer`（或等价）：置位 abort、join、排空 | 避免资源泄漏或毒丸无法入队 |
| 6.4 | `daemon` 线程与 `PRODUCER_JOIN_TIMEOUT_S`：超时日志是否足够醒目 | |
| 6.5 | memmap 分支：`use_memmap` 与 `max_working_memory_bytes` 决策与 OOM 失败路径 | `ola_memmap_allocation_failed` |
| 6.6 | `list_frame_starts`、Hanning 权重与 OLA 归一化与测试一致 | `tests/test_ola.py` 等 |

---

## 7. 异常体系（`src/core/exceptions.py`、`linalg_errors.py`）

| 序号 | 检查项 | 关注点 |
|------|--------|--------|
| 7.1 | 层次：`HankelPurifyError` → `AudioIOError` / `ConfigurationError` / `ProcessingError` | |
| 7.2 | `ProcessingError` 的 `code`、`origin_exception_type` 在 CLI/日志中是否被消费 | |
| 7.3 | `linalg_errors` 集合与 `purifier.process_file` 中 `except` 元组同步更新 | 新增 SciPy 异常类型时 |

---

## 8. 日志（`src/utils/logger.py`）

| 序号 | 检查项 | 关注点 |
|------|--------|--------|
| 8.1 | 环境变量 `HSP_LOG_IO_TRACE` 等与 I/O 追踪文档一致 | `tests/test_logger_env.py` |
| 8.2 | 无敏感信息写入日志 | |

---

## 9. 脚本与基准（`scripts/`）

| 序号 | 检查项 | 关注点 |
|------|--------|--------|
| 9.1 | `benchmark_pipeline.py`：`PYTHONPATH`、导入路径与 `src` 布局一致 | 常 `sys.path` 注入 |
| 9.2 | `run_with_peak_rss.sh` 等 shell 与文档中的用法 | 可执行权限与 bash 兼容性 |

---

## 10. 可选前端（`frontend/app.py`）

| 序号 | 检查项 | 关注点 |
|------|--------|--------|
| 10.1 | 全量任务经 `subprocess` 调 CLI，不直接读整文件 | PRD/NF-05 类约束 |
| 10.2 | 子进程超时与用户输入校验 | `subprocess.TimeoutExpired` |
| 10.3 | 依赖仅在 `requirements-frontend.txt`，不污染核心 `requirements.txt` | |

---

## 11. 测试与回归（`tests/`）

| 序号 | 检查项 | 关注点 |
|------|--------|--------|
| 11.1 | 覆盖率：`pytest` 默认 `--cov-fail-under` 与 CI 一致 | |
| 11.2 | 核心链：`test_process_frame.py`、`test_pipeline_mssa.py`、`test_c_svd.py` | |
| 11.3 | 门面流：`test_purifier_stream.py`、`test_purifier_producer_errors.py` | 背压、中止、错误 |
| 11.4 | CLI：`test_cli_smoke.py`、`test_cli_unit.py`、`test_cli_args.py` | |
| 11.5 | 可选格式：`test_ogg_optional.py` 与 skip 条件 | |

---

## 12. 安全与滥用面（横切）

| 序号 | 检查项 | 关注点 |
|------|--------|--------|
| 12.1 | 路径：仅后缀白名单；是否需解析 `Path.resolve()` 防止奇怪相对路径（按产品威胁模型） | 当前以 soundfile 打开为准 |
| 12.2 | 无硬编码密钥；环境变量仅用于非敏感配置（如 `HSP_MAX_SAMPLES`） | |
| 12.3 | 子进程/线程：用户可控参数（超时、路径长度）是否有上限或拒绝策略 | 前端与 CLI |

---

## 13. 文档与代码一致性

| 序号 | 检查项 | 关注点 |
|------|--------|--------|
| 13.1 | `README.md`（tutorial 上）与 CLI 参数、格式表一致 | |
| 13.2 | `docs/software_design.md` 与 `process_frame` / 无 `MSSAStage` 叙述一致 | |
| 13.3 | [根目录 README §3 运行环境与依赖](../README.md#环境与构建) 与 [`docs/software_design.md`](software_design.md) §4、[`docs/BRANCHES.md`](BRANCHES.md) 与当前分支一致 | 文档变更后复查 |

---

## 附录：模块与文件速查（`tutorial`）

| 区域 | 路径 |
|------|------|
| CLI | `src/cli.py` |
| 门面 | `src/facade/purifier.py`, `soundfile_ola.py`, `pcm_producer.py`, `ola.py` |
| 核心 | `src/core/process_frame.py`, `stages/*.py`, `strategies/*.py` |
| I/O | `src/io/audio_formats.py`, `audio_stream.py`, `stereo_soundfile.py`, … |
| 异常 | `src/core/exceptions.py`, `linalg_errors.py` |
| 日志 | `src/utils/logger.py` |
| 脚本 | `scripts/benchmark_pipeline.py`, `estimate_ola_frames.py` |
| 前端 | `frontend/app.py` |

审计完成后，建议在 issue 或评审记录中归档：**已审计 commit**、**未覆盖项**、**发现的问题与严重级别**。
