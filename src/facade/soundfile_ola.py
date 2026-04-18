"""Soundfile + OLA + PCM producer loop used by ``AudioPurifier`` via composition.

The producer thread target is injected (default: ``pcm_producer.producer_fill_queue``)
so tests can substitute ``src.facade.purifier._producer_fill_queue`` without a lazy
import cycle between this module and ``purifier``.
"""

from __future__ import annotations

import contextlib
import os
import queue
import tempfile
import threading
from collections.abc import Callable
from queue import Queue
from typing import Any, NamedTuple

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from tqdm import tqdm

from src.core.exceptions import AudioIOError, ConfigurationError
from src.facade.ola import list_frame_starts
from src.facade.pcm_producer import (
    PCM_QUEUE_MAXSIZE,
    PRODUCER_JOIN_TIMEOUT_S,
)
from src.facade.pcm_producer import (
    producer_fill_queue as default_producer_fill_queue,
)
from src.io.audio_formats import soundfile_write_kwargs
from src.io.audio_stream import read_audio_metadata
from src.io.io_messages import (
    audio_io_failed_pair,
    empty_or_invalid_audio_length,
    ola_memmap_allocation_failed,
    pcm_stream_ended_before_ola_complete,
)
from src.io.stereo_soundfile import require_stereo_channels


class SfOlaPrep(NamedTuple):
    """Metadata and sizing for one soundfile OLA+MSSA pass (internal)."""

    num_samples: int
    samplerate: int
    starts: list[int]
    use_memmap: bool
    block_size: int


class SoundfileOlaEngine:
    """PCM queue + OLA/MSSA accumulation + soundfile write."""

    def __init__(
        self,
        *,
        logger: Any,
        max_input_samples: int | None,
        max_working_memory_bytes: int,
        producer_fill_queue: Callable[..., None] = default_producer_fill_queue,
    ) -> None:
        self.logger = logger
        self.max_input_samples = max_input_samples
        self.max_working_memory_bytes = max_working_memory_bytes
        self._producer_fill_queue = producer_fill_queue

    @staticmethod
    def _raise_if_producer_failed(
        producer_error: list[BaseException],
    ) -> None:
        if producer_error:
            raise producer_error[0]

    def _append_pcm_until(
        self,
        *,
        pcm_queue: Queue[NDArray[np.float64] | None],
        producer_error: list[BaseException],
        buf: NDArray[np.float64],
        buffer_base: int,
        need_global_end: int,
    ) -> NDArray[np.float64]:
        """从队列拉取块并拼接到 ``buf``，直至覆盖全局样本下标 ``need_global_end``。"""
        out = buf
        while buffer_base + int(out.shape[0]) < need_global_end:
            item = pcm_queue.get()
            if item is None:
                self._raise_if_producer_failed(producer_error)
                raise AudioIOError(pcm_stream_ended_before_ola_complete())
            chunk = np.asarray(item, dtype=np.float64, order="C")
            if chunk.ndim == 1:
                chunk = np.ascontiguousarray(chunk[:, np.newaxis])
            if out.size == 0:
                out = chunk
            else:
                out = np.concatenate((out, chunk), axis=0)
        return out

    def _drain_pcm_queue(
        self,
        pcm_queue: Queue[NDArray[np.float64] | None],
    ) -> None:
        """非阻塞排空队列，便于生产者线程 join。

        主线程在 ``_append_pcm_until`` 中取走毒丸 ``None`` 并抛出后，队列可能已空；
        若此处再阻塞 ``get()``，会与已退出的生产者形成死锁。
        """
        while True:
            try:
                pcm_queue.get_nowait()
            except queue.Empty:
                break

    def _prepare_sf_ola_prep(
        self,
        input_path: str,
        output_path: str,
        f_size: int,
        hop: int,
    ) -> SfOlaPrep:
        try:
            meta = read_audio_metadata(input_path)
        except AudioIOError as exc:
            raise AudioIOError(audio_io_failed_pair(input_path, output_path)) from exc
        num_samples = int(meta["frames"])
        if self.max_input_samples is not None and num_samples > self.max_input_samples:
            raise ConfigurationError(
                f"Input has {num_samples} samples per channel; limit is "
                f"{self.max_input_samples} (see --max-samples or HSP_MAX_SAMPLES)."
            )
        samplerate = int(meta["samplerate"])
        require_stereo_channels(
            int(meta["channels"]),
            context=f"metadata input={input_path!r}",
        )
        starts = list_frame_starts(num_samples, f_size, hop)
        if not starts:
            raise AudioIOError(empty_or_invalid_audio_length())
        bytes_needed = num_samples * 24
        use_memmap = bytes_needed > self.max_working_memory_bytes
        block_size = max(4096, f_size)
        return SfOlaPrep(
            num_samples,
            samplerate,
            starts,
            use_memmap,
            block_size,
        )

    def _allocate_ola_accumulators(
        self,
        num_samples: int,
        use_memmap: bool,
        tmp_dir: str | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if use_memmap:
            assert tmp_dir is not None
            mmap_out = os.path.join(tmp_dir, "acc.dat")
            mmap_w = os.path.join(tmp_dir, "wsum.dat")
            try:
                out_acc = np.memmap(
                    mmap_out,
                    dtype=np.float64,
                    mode="w+",
                    shape=(num_samples, 2),
                )
                wsum_1d = np.memmap(
                    mmap_w,
                    dtype=np.float64,
                    mode="w+",
                    shape=(num_samples,),
                )
            except (OSError, MemoryError) as exc:
                raise AudioIOError(ola_memmap_allocation_failed()) from exc
            return out_acc, wsum_1d
        return (
            np.zeros((num_samples, 2), dtype=np.float64),
            np.zeros(num_samples, dtype=np.float64),
        )

    def _start_pcm_producer_thread(
        self,
        input_path: str,
        block_size: int,
        pcm_queue: Queue[NDArray[np.float64] | None],
        producer_error: list[BaseException],
        abort_event: threading.Event,
    ) -> threading.Thread:
        th = threading.Thread(
            target=self._producer_fill_queue,
            name="HSP-AudioProducer",
            args=(
                input_path,
                block_size,
                pcm_queue,
                producer_error,
                abort_event,
            ),
            daemon=True,
        )
        th.start()
        return th

    def _ola_mssa_loop_write(
        self,
        *,
        prep: SfOlaPrep,
        output_path: str,
        denoise_frame: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        f_size: int,
        w_sqrt: NDArray[np.float64],
        w_sq: NDArray[np.float64],
        out_acc: NDArray[np.float64],
        wsum_1d: NDArray[np.float64],
        pcm_queue: Queue[NDArray[np.float64] | None],
        producer_error: list[BaseException],
    ) -> None:
        num_samples = prep.num_samples
        samplerate = prep.samplerate
        starts = prep.starts
        frame = np.zeros((f_size, 2), dtype=np.float64)
        x_win_buf = np.zeros((f_size, 2), dtype=np.float64)
        weighted_buf = np.zeros((f_size, 2), dtype=np.float64)
        buffer_base = 0
        buf = np.empty((0, 2), dtype=np.float64)

        with tqdm(
            total=len(starts),
            desc="OLA/MSSA",
            unit="frame",
            leave=True,
        ) as pbar:
            for idx, start in enumerate(starts):
                need_global_end = min(start + f_size, num_samples)
                buf = self._append_pcm_until(
                    pcm_queue=pcm_queue,
                    producer_error=producer_error,
                    buf=buf,
                    buffer_base=buffer_base,
                    need_global_end=need_global_end,
                )
                rel = start - buffer_base
                need = need_global_end - start
                frame.fill(0.0)
                frame[:need] = buf[rel : rel + need]

                np.multiply(frame, w_sqrt, out=x_win_buf)
                denoised: NDArray[np.float64] = denoise_frame(x_win_buf)
                np.multiply(denoised, w_sqrt, out=weighted_buf)
                end = min(start + f_size, num_samples)
                sl = end - start
                out_acc[start:end] += weighted_buf[:sl]
                wsum_1d[start:end] += w_sq[:sl]

                if idx + 1 < len(starts):
                    next_s = starts[idx + 1]
                    drop = next_s - buffer_base
                    if drop > 0:
                        buf = buf[drop:]
                        buffer_base += drop
                else:
                    buf = np.empty((0, 2), dtype=np.float64)
                    buffer_base = need_global_end

                pbar.update(1)

        denom = np.maximum(wsum_1d, 1e-12)
        output = out_acc / denom[:, np.newaxis]

        if prep.use_memmap:
            output = np.asarray(output, dtype=np.float64)

        np.clip(output, -1.0, 1.0, out=output)
        wkwargs = soundfile_write_kwargs(output_path)
        sf.write(
            output_path,
            output,
            samplerate,
            **wkwargs,
        )

    def _shutdown_pcm_producer(
        self,
        abort_event: threading.Event,
        pcm_queue: Queue[NDArray[np.float64] | None],
        producer_thread: threading.Thread | None,
    ) -> None:
        abort_event.set()
        if producer_thread is not None:
            self._drain_pcm_queue(pcm_queue)
            producer_thread.join(timeout=PRODUCER_JOIN_TIMEOUT_S)
            if producer_thread.is_alive():
                self.logger.warning(
                    "Producer thread %r did not finish within %ss; it is daemon "
                    "and will not block process exit.",
                    producer_thread.name,
                    PRODUCER_JOIN_TIMEOUT_S,
                )

    def run_soundfile_ola(
        self,
        input_path: str,
        output_path: str,
        denoise_frame: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        f_size: int,
        hop: int,
        w_sqrt: NDArray[np.float64],
        w_sq: NDArray[np.float64],
    ) -> None:
        """Stream input, run per-frame MSSA via ``denoise_frame``, OLA, write output."""
        prep = self._prepare_sf_ola_prep(input_path, output_path, f_size, hop)
        pcm_queue: Queue[NDArray[np.float64] | None] = Queue(
            maxsize=PCM_QUEUE_MAXSIZE,
        )
        producer_error: list[BaseException] = []
        abort_event = threading.Event()
        producer_thread: threading.Thread | None = None

        accum_cm = (
            tempfile.TemporaryDirectory(prefix="hsp_ola_")
            if prep.use_memmap
            else contextlib.nullcontext()
        )
        try:
            with accum_cm as tmp_dir:
                out_acc, wsum_1d = self._allocate_ola_accumulators(
                    prep.num_samples,
                    prep.use_memmap,
                    tmp_dir,
                )
                producer_thread = self._start_pcm_producer_thread(
                    input_path,
                    prep.block_size,
                    pcm_queue,
                    producer_error,
                    abort_event,
                )
                try:
                    self._ola_mssa_loop_write(
                        prep=prep,
                        output_path=output_path,
                        denoise_frame=denoise_frame,
                        f_size=f_size,
                        w_sqrt=w_sqrt,
                        w_sq=w_sq,
                        out_acc=out_acc,
                        wsum_1d=wsum_1d,
                        pcm_queue=pcm_queue,
                        producer_error=producer_error,
                    )
                finally:
                    if prep.use_memmap:
                        del out_acc
                        del wsum_1d
        finally:
            self._shutdown_pcm_producer(abort_event, pcm_queue, producer_thread)
