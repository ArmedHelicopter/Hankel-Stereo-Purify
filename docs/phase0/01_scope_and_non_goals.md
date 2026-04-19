# Phase0 范围与非目标

本文从 **`tutorial` 分支已具备的能力**倒推：若回到 Phase0，规划上应锁定哪些**边界**，以免范围膨胀。

## 目标（与可交付版本一致的方向）

- **立体声 PCM 降噪**：多通道奇异谱分析（MSSA）数值链，保持左右声道相位关系（与 PRD 叙事一致）。
- **工程约束**：流式/分帧处理大文件；类型注解与静态检查；测试驱动关键数值与 I/O 契约。
- **用户入口**：命令行与（规划上可选）本地控制平面；敏感信息不入库。

## 非目标（明确写清，避免 Phase0 被误判为半成品）

- **不在 Phase0 骨架分支追求功能 parity**：当前 `main` 上的占位 CLI **不**实现 Hankel/SVD/OLA；完整实现见 `tutorial`。
- **前端**：Streamlit 等为 **可选**；核心验收以 CLI/库与测试为准（与 PRD F-05 一致）。
- **实时低延迟**：以离线/准离线批处理为主，除非单独立项。

## 与 `tutorial` 的对照

| Phase0 主题 | `tutorial` 中可核对的落点（示例） |
|-------------|-------------------------------------|
| 单帧 MSSA 链 | `src/core/process_frame.py` 与 stages `a_*`…`d_*` |
| 门面与 OLA | `src/facade/purifier.py`、`soundfile_ola.py` |
| I/O 与白名单 | `src/io/audio_formats.py`、`stereo_soundfile.py` |
| 教程与验收表 | `tutorial/TUTORIAL_INDEX.md` |

详见 [`04_reverse_from_tutorial.md`](04_reverse_from_tutorial.md)。
