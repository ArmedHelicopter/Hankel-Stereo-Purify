"""PCM producer thread: bounded queue + poison pill for ``AudioPurifier`` streaming."""

from __future__ import annotations

import queue
import threading
from queue import Queue

import numpy as np
from numpy.typing import NDArray

from src.core.exceptions import AudioIOError
from src.io.audio_stream import AudioStream

# 有界队列：限制「已解码、待消费」的块数量，防止生产者无界堆积导致 OOM。
PCM_QUEUE_MAXSIZE = 3

# 生产者 put 超时（秒）：队列满时轮询 abort_event，避免消费者退出后永久死锁。
PUT_TIMEOUT_S = 0.25

# 主线程等待生产者结束的最长时间（秒）；超时则记录告警（线程为 daemon）。
PRODUCER_JOIN_TIMEOUT_S = 120.0


def producer_fill_queue(
    input_path: str,
    block_size: int,
    pcm_queue: Queue[NDArray[np.float64] | None],
    producer_error: list[BaseException],
    abort_event: threading.Event,
) -> None:
    """生产者线程：顺序读取 PCM 块并入队；结束时投入毒丸 ``None``。

    **并发契约**：仅本线程向 ``producer_error`` 追加异常；主线程只在从队列取到
    毒丸之后读取该列表（见 ``AudioPurifier._append_pcm_until`` /
    ``_raise_if_producer_failed``）。

    毒丸机制：无论正常结束还是 ``AudioIOError``，均在 ``finally`` 中投入且仅投入一次
    ``None``，使主线程消费者能从 ``get()`` 唤醒并结束，避免无限阻塞。

    防死锁：若主线程因 MSSA 失败而中止，``abort_event`` 置位；本循环在每次
    ``put`` 前检查该事件，且 ``put`` 使用超时以便在队列满时仍能轮询中止。

    **与消费者的契约：** 主线程在异常或正常结束路径上必须调用
    ``AudioPurifier._shutdown_pcm_producer``（置位 ``abort_event``、排空队列、
    ``join`` 生产者）。否则队列可能长期满载，毒丸 ``None`` 无法入队，生产者
    会在 ``finally`` 中阻塞重试；daemon 线程不会永久卡住进程退出，但会造成
    资源释放延迟。
    """
    sentinel_sent = False
    try:
        stream = AudioStream(input_path, block_size=block_size)
        for block in stream.read_blocks():
            if abort_event.is_set():
                break
            block_arr = np.asarray(block, dtype=np.float64, order="C")
            while True:
                if abort_event.is_set():
                    return
                try:
                    pcm_queue.put(block_arr, timeout=PUT_TIMEOUT_S)
                    break
                except queue.Full:
                    continue
    except AudioIOError as exc:
        producer_error.append(exc)
    finally:
        while not sentinel_sent:
            try:
                pcm_queue.put(None, timeout=PUT_TIMEOUT_S)
                sentinel_sent = True
            except queue.Full:
                continue
