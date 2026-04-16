import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from src.core.exceptions import AudioIOError, ConfigurationError, HankelPurifyError
from src.core.pipeline import Pipeline
from src.core.stages.a_hankel import AHankelStage
from src.core.stages.b_multichannel import BMultichannelStage
from src.core.stages.c_svd import CSVDStage
from src.core.stages.d_diagonal import DDiagonalStage
from src.core.strategies.truncation import (
    EnergyThresholdStrategy,
    FixedRankStrategy,
    TruncationStrategy,
)
from src.facade.ola import list_frame_starts, sqrt_hanning_weights
from src.io.audio_stream import read_flac_metadata
from src.utils.logger import get_logger


class AudioPurifier:
    """Stereo FLAC I/O: OLA (F-02), MSSA per frame, optional memmap (NF-01).

    Validate paths/config here; stages trust inputs for speed.
    """

    def __init__(
        self,
        window_length: int,
        truncation_rank: int,
        *,
        energy_fraction: float | None = None,
        frame_size: int | None = None,
        hop_size: int | None = None,
        max_working_memory_bytes: int = 1_500_000_000,
    ) -> None:
        self.window_length = window_length
        self.truncation_rank = truncation_rank
        self.energy_fraction = energy_fraction
        self.frame_size = (
            frame_size
            if frame_size is not None
            else max(self.window_length + 8, (self.window_length * 3 + 1) // 2)
        )
        self.hop_size = (
            hop_size if hop_size is not None else max(1, self.frame_size // 2)
        )
        self.max_working_memory_bytes = max_working_memory_bytes
        self.logger = get_logger(self.__class__.__name__)

    def process_file(self, input_path: str, output_path: str) -> None:
        """Process one audio file and write the denoised result."""
        try:
            self._validate_paths(input_path, output_path)
            self._validate_configuration()
            self._run_processing(input_path, output_path)
        except HankelPurifyError as exc:
            self.logger.error("处理失败：%s", exc)
            raise
        except Exception:
            self.logger.exception("Unexpected error during audio purification")
            raise

    def _validate_paths(self, input_path: str, output_path: str) -> None:
        input_file = Path(input_path)
        if input_file.suffix.lower() != ".flac":
            raise ConfigurationError("Unsupported audio format: only FLAC is allowed.")
        if not input_file.exists():
            raise AudioIOError(f"Input file does not exist: {input_path}")
        if not input_file.is_file():
            raise AudioIOError(f"Input path is not a file: {input_path}")

        output_file = Path(output_path)
        if output_file.suffix.lower() != ".flac":
            raise ConfigurationError("Output file must use .flac extension.")
        if not output_file.parent.exists():
            try:
                output_file.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise AudioIOError(
                    f"Unable to create output directory: {output_file.parent}"
                ) from exc

    def _validate_configuration(self) -> None:
        if self.window_length <= 0:
            raise ConfigurationError("Window length must be a positive integer.")
        if self.frame_size < self.window_length:
            raise ConfigurationError("frame_size must be >= window_length.")
        if self.hop_size <= 0 or self.hop_size >= self.frame_size:
            raise ConfigurationError(
                "hop_size must be positive and smaller than frame_size."
            )
        if self.energy_fraction is not None:
            if not 0.0 < self.energy_fraction <= 1.0:
                raise ConfigurationError("energy_fraction must be in (0, 1].")
            return
        if self.truncation_rank <= 0:
            raise ConfigurationError("Truncation rank must be a positive integer.")
        if self.truncation_rank > self.window_length:
            raise ConfigurationError("Truncation rank cannot exceed window length.")
        k_hankel = self.frame_size - self.window_length + 1
        max_svd_rank = min(self.window_length, 2 * k_hankel)
        if self.truncation_rank > max_svd_rank:
            raise ConfigurationError(
                "truncation_rank exceeds SVD matrix rank for this frame_size "
                f"(max {max_svd_rank})."
            )

    def _build_pipeline(self) -> Pipeline:
        strat: TruncationStrategy
        if self.energy_fraction is not None:
            strat = EnergyThresholdStrategy(self.energy_fraction)
        else:
            strat = FixedRankStrategy(self.truncation_rank)
        return Pipeline(
            [
                AHankelStage(self.window_length),
                BMultichannelStage(),
                CSVDStage(strat),
                DDiagonalStage(),
            ],
        )

    def _run_processing(self, input_path: str, output_path: str) -> None:
        self.logger.info("开始处理流式数据...")
        meta = read_flac_metadata(input_path)
        if meta["channels"] != 2:
            raise AudioIOError("Only stereo FLAC (2 channels) is supported.")
        num_samples = meta["frames"]
        samplerate = meta["samplerate"]
        f_size = self.frame_size
        hop = self.hop_size
        w_sqrt = sqrt_hanning_weights(f_size)[:, np.newaxis]
        w_sq = (w_sqrt * w_sqrt).ravel()

        pipeline = self._build_pipeline()
        starts = list_frame_starts(num_samples, f_size, hop)
        if not starts:
            raise AudioIOError("Empty or invalid audio length.")

        # Heuristic RAM for full-length float64 OLA buffers (stereo × samples × 8 bytes
        # plus margin); exceeds budget → memmap spill under prefix `hsp_ola_`.
        bytes_needed = num_samples * 24
        use_memmap = bytes_needed > self.max_working_memory_bytes

        tmp_dir: str | None = None
        mmap_out: str = ""
        mmap_w: str = ""

        out_acc: NDArray[np.float64]
        wsum_1d: NDArray[np.float64]
        if use_memmap:
            tmp_dir = tempfile.mkdtemp(prefix="hsp_ola_")
            mmap_out = os.path.join(tmp_dir, "acc.dat")
            mmap_w = os.path.join(tmp_dir, "wsum.dat")
            out_acc = np.memmap(
                mmap_out, dtype=np.float64, mode="w+", shape=(num_samples, 2)
            )
            wsum_1d = np.memmap(
                mmap_w, dtype=np.float64, mode="w+", shape=(num_samples,)
            )
        else:
            out_acc = np.zeros((num_samples, 2), dtype=np.float64)
            wsum_1d = np.zeros(num_samples, dtype=np.float64)

        try:
            with sf.SoundFile(input_path) as snd:
                for start in starts:
                    frame = np.zeros((f_size, 2), dtype=np.float64)
                    snd.seek(start)
                    need = min(f_size, num_samples - start)
                    chunk = snd.read(need, dtype="float64", always_2d=True)
                    frame[: chunk.shape[0]] = chunk
                    x_win = frame * w_sqrt
                    denoised: NDArray[np.float64] = pipeline.execute(x_win)
                    weighted = denoised * w_sqrt
                    end = min(start + f_size, num_samples)
                    sl = end - start
                    out_acc[start:end] += weighted[:sl]
                    wsum_1d[start:end] += w_sq[:sl]

            denom = np.maximum(wsum_1d, 1e-12)
            output = out_acc / denom[:, np.newaxis]

            if use_memmap:
                output = np.asarray(output, dtype=np.float64)

            np.clip(output, -1.0, 1.0, out=output)
            sf.write(
                output_path,
                output,
                samplerate,
                format="FLAC",
                subtype="PCM_24",
            )
        finally:
            if use_memmap:
                del out_acc
                del wsum_1d
                if tmp_dir is not None:
                    try:
                        if mmap_out:
                            os.unlink(mmap_out)
                        if mmap_w:
                            os.unlink(mmap_w)
                        os.rmdir(tmp_dir)
                    except OSError:
                        pass

        self.logger.info("流式数据处理完成")


class MSSAPurifierBuilder:
    """Fluent builder: set `window_length`, then fixed rank or energy fraction."""

    def __init__(self) -> None:
        self.params: dict[str, Any] = {}

    def set_window_length(self, value: int) -> "MSSAPurifierBuilder":
        self.params["window_length"] = value
        return self

    def set_truncation_rank(self, value: int) -> "MSSAPurifierBuilder":
        self.params["truncation_rank"] = value
        return self

    def set_energy_fraction(self, value: float) -> "MSSAPurifierBuilder":
        """Use cumulative singular-value energy (mutually exclusive with fixed rank)."""
        self.params["energy_fraction"] = value
        return self

    def set_frame_size(self, value: int) -> "MSSAPurifierBuilder":
        self.params["frame_size"] = value
        return self

    def set_hop_size(self, value: int) -> "MSSAPurifierBuilder":
        self.params["hop_size"] = value
        return self

    def set_max_working_memory_bytes(self, value: int) -> "MSSAPurifierBuilder":
        self.params["max_working_memory_bytes"] = value
        return self

    def build(self) -> AudioPurifier:
        if "window_length" not in self.params:
            raise ConfigurationError(
                "Missing window_length: call set_window_length(...) before build()."
            )
        wl = self.params["window_length"]
        if not isinstance(wl, int) or wl <= 0:
            raise ConfigurationError("window_length must be a positive integer.")

        energy_fraction = self.params.get("energy_fraction")
        truncation_rank = self.params.get("truncation_rank")

        if energy_fraction is not None and truncation_rank is not None:
            raise ConfigurationError(
                "Use set_energy_fraction(...) or set_truncation_rank(...), not both."
            )

        mem = self.params.get("max_working_memory_bytes", 1_500_000_000)
        fs = self.params.get("frame_size")
        hs = self.params.get("hop_size")

        if energy_fraction is not None:
            return AudioPurifier(
                wl,
                0,
                energy_fraction=energy_fraction,
                frame_size=fs,
                hop_size=hs,
                max_working_memory_bytes=mem,
            )

        if truncation_rank is None:
            raise ConfigurationError(
                "Missing truncation mode: call set_truncation_rank(...) for fixed k, "
                "or set_energy_fraction(...) for energy-based rank."
            )

        return AudioPurifier(
            wl,
            truncation_rank,
            frame_size=fs,
            hop_size=hs,
            max_working_memory_bytes=mem,
        )


# Progress bar pseudocode for large-scale stream processing:
#
# total_blocks = compute_total_blocks_from_stream(input_path, block_size)
# logger.info("开始处理流式数据...")
# with tqdm(total=total_blocks, desc="Purifying audio", unit="block") as progress:
#     for block_index, block_data in enumerate(stream_blocks(input_path, block_size)):
#         purified_block = process_block(block_data)
#         write_block(purified_block, output_path)
#         progress.update(1)
#
# logger.info("流式数据处理完成：%s 块已处理", total_blocks)
