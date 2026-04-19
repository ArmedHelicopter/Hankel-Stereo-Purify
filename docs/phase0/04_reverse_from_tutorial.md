# 倒推附录：Phase0 条目 → `tutorial` 实现路径

本表用于评审溯源：**左侧**为 Phase0 规划条目，**右侧**为在 **`tutorial` 分支**上可打开的路径（不拷贝代码正文）。

| Phase0 条目 | `tutorial` 上建议打开的位置 |
|-------------|---------------------------|
| 单帧 MSSA 顺序 | `src/core/process_frame.py`；`src/core/stages/a_hankel.py` → `b_multichannel.py` → `c_svd.py` → `d_diagonal.py` |
| 截断策略（固定秩 / 能量） | `src/core/strategies/truncation.py`；`make_svd_step` 在 `c_svd.py` |
| W-correlation（可选） | `src/core/strategies/grouping.py`；`c_svd.py` 中阈值路径 |
| 门面与 OLA | `src/facade/purifier.py`；`soundfile_ola.py`；`ola.py`；`pcm_producer.py` |
| CLI 与退出码 | `src/cli.py` |
| I/O 与格式白名单 | `src/io/audio_formats.py`；`audio_stream.py`；`stereo_soundfile.py` |
| 异常模型 | `src/core/exceptions.py`；`linalg_errors.py` |
| 回归与冒烟测试 | `tests/`（如 `test_process_frame.py`、`test_purifier_stream.py`、`test_c_svd.py`） |
| 可选前端 | `frontend/app.py`；`requirements-frontend.txt` |
| 脚本与基准 | `scripts/benchmark_pipeline.py` 等 |
| 教程与验收表 | `tutorial/TUTORIAL_INDEX.md` 及各章 `.md` |

与 [`COMPARISON_main_vs_tutorial.md`](COMPARISON_main_vs_tutorial.md) 的分工：**本文**是需求到模块的映射；**对比文档**是 Git 事实与行数级增量。
