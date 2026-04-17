# 第 5 章：OLA 与流式门面

## 直觉先行

整首歌太长，不能一次塞进一个巨型 Hankel。做法是：**滑窗分帧**，每帧跑一次 A→B→C→D，再把各帧结果用窗函数加权叠回去（Overlap-Add）。这样每帧矩阵小，峰值内存可控；帧与帧重叠消除边界爆音。

流式读盘：解码线程往**有界队列**里塞块，主线程消费拼缓冲——避免一次性读入全文件。

## 决策复盘（Why vs Why not）

**为什么 memmap 累加器？**  
当样本数 × 每样本字节超过 `--max-memory-mb` 预算，累加缓冲落到临时文件映射，换 **I/O 带宽** 换 **RAM**——见 [`src/facade/soundfile_ola.py`](../src/facade/soundfile_ola.py) 中 `_allocate_ola_accumulators`（由 `AudioPurifier` 继承 `SoundfileOlaMixin` 使用）。

**为什么生产者 `put` 带超时 + 毒丸？**  
阻塞读 + 有界队列：若消费者挂了，生产者不能无限 `put`；毒丸 `None` 让消费者退出，避免死锁。见 [`pcm_producer.py`](../src/facade/pcm_producer.py) 与 PRD **NF-04** 预留的生产者-消费者形态。

**为什么不把 OLA 写进 `core/stages`？**  
核心阶段只做「单帧数学」；帧索引、窗函数在 [`ola.py`](../src/facade/ola.py)，**关注点分离**：核心可测、门面可换 I/O 策略。

### 代码锚点

| 主题 | 定位 | 路径 |
|------|------|------|
| 帧起点、hop、窗 | `list_frame_starts` 等 | [`src/facade/ola.py`](../src/facade/ola.py) |
| 整文件 OLA 主循环、memmap 累加器 | `_run_processing_soundfile`、`_ola_mssa_loop_write`、`_allocate_ola_accumulators` | [`src/facade/soundfile_ola.py`](../src/facade/soundfile_ola.py)（`SoundfileOlaMixin`） |
| 门面入口与管线构建 | `process_file`、`_run_processing`、`_build_pipeline` | [`src/facade/purifier.py`](../src/facade/purifier.py) |
| 有界队列、超时 `put`、毒丸 `None` | 队列容量常量与消费者协议 | [`src/facade/pcm_producer.py`](../src/facade/pcm_producer.py) |
| 路径白名单 | `validate_io_paths`、后缀集合 | [`src/io/audio_formats.py`](../src/io/audio_formats.py) |

**工程三条线（各一条路径）**：**背压**——生产者线程 + 有界 `Queue`；**毒丸**——结束哨兵避免消费者悬挂；**memmap**——超 `--max-memory-mb` 时累加落盘映射，换 I/O 换 RAM。

## 思维挂钩

| 代码 | 专业课 | 软件工程 |
|------|--------|----------|
| Hanning 窗 × 帧 | 窗函数频谱泄漏 | 乘窗再 OLA 是标准 DSP 套路 |
| `Queue(maxsize=3)` | 背压（backpressure） | 并发：事件、超时、daemon 线程 |
| `validate_io_paths` | — | 白名单扩展名、防误用容器 |

## 晦涩点与建议

- **晦涩**：整文件 OLA 与 PCM 队列曾集中在门面单文件，现已拆入 [`soundfile_ola.py`](../src/facade/soundfile_ola.py)；线程与累加器仍要对照两处。  
- **建议**：阅读顺序：`process_file` → `_run_processing` → `_run_processing_soundfile`（均在 `purifier` / mixin 链上）。先画「主线程：OLA 循环；子线程：读块」的时序图，再对照 `_run_processing_soundfile`。

**下一章**：异常、测试与「工业级」边界。
