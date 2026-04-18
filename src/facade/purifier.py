"""Facade: OLA + MSSA over stereo PCM via libsndfile.

``process_file`` maps failures to stable ``HankelPurifyError`` subclasses.
NumPy/SciPy ``LinAlgError`` and sparse ``ArpackError`` / ``ArpackNoConvergence``
(from ``svds``) are caught explicitly; extend ``src.core.linalg_errors`` when new
numeric stack types need first-class mapping. The final ``except Exception`` is a
last resort—prefer adding explicit branches over ``__module__`` string checks.
"""

import os
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from src.core.exceptions import (
    AudioIOError,
    ConfigurationError,
    HankelPurifyError,
    ProcessingError,
    ProcessingFailureCode,
    exception_fully_qualified_name,
    format_exception_origin,
    validate_w_corr_threshold,
)
from src.core.linalg_errors import MSSA_ARPACK_ERRORS, MSSA_LINALG_ERRORS
from src.core.process_frame import process_frame
from src.core.stages.c_svd import make_svd_step
from src.core.strategies.truncation import (
    EnergyThresholdStrategy,
    FixedRankStrategy,
    TruncationStrategy,
)
from src.facade.ola import sqrt_hanning_weights
from src.facade.pcm_producer import producer_fill_queue
from src.facade.soundfile_ola import SoundfileOlaEngine
from src.io.audio_formats import validate_io_paths
from src.io.io_messages import (
    audio_io_failed_pair,
    input_file_does_not_exist,
    input_path_not_a_file,
    unable_to_create_output_directory,
)
from src.utils.logger import get_logger


def _resolve_max_input_samples(explicit: int | None) -> int | None:
    """Caller explicit limit wins over ``HSP_MAX_SAMPLES``."""
    if explicit is not None:
        if explicit <= 0:
            raise ConfigurationError("max_input_samples must be a positive integer.")
        return explicit
    raw = os.environ.get("HSP_MAX_SAMPLES")
    if raw is None or not str(raw).strip():
        return None
    try:
        n = int(str(raw).strip(), 10)
    except ValueError as exc:
        raise ConfigurationError(
            "HSP_MAX_SAMPLES must be a positive integer if set."
        ) from exc
    if n <= 0:
        raise ConfigurationError("HSP_MAX_SAMPLES must be positive if set.")
    return n


# Backwards-compatible name for tests that monkeypatch the producer entrypoint.
_producer_fill_queue = producer_fill_queue


class AudioPurifier:
    """Stereo PCM I/O via soundfile/OLA (F-02), MSSA per frame, optional memmap (NF-01).

    Validate paths/config here; numerical steps trust inputs for speed.
    Streaming OLA is delegated to ``SoundfileOlaEngine`` (composition).
    """

    window_length: int
    truncation_rank: int
    energy_fraction: float | None

    def __init__(
        self,
        window_length: int,
        *,
        truncation_rank: int | None = None,
        energy_fraction: float | None = None,
        frame_size: int | None = None,
        hop_size: int | None = None,
        max_working_memory_bytes: int = 1_500_000_000,
        max_input_samples: int | None = None,
        w_corr_threshold: float | None = None,
    ) -> None:
        if not isinstance(window_length, int) or window_length <= 0:
            raise ConfigurationError("window_length must be a positive integer.")
        if truncation_rank is not None and energy_fraction is not None:
            raise ConfigurationError(
                "Use truncation_rank=... or energy_fraction=..., not both."
            )
        if truncation_rank is None and energy_fraction is None:
            raise ConfigurationError(
                "Missing truncation mode: pass truncation_rank=... for fixed k, "
                "or energy_fraction=... for energy-based rank."
            )
        validate_w_corr_threshold(w_corr_threshold)

        self.window_length = window_length
        self.w_corr_threshold = w_corr_threshold
        if energy_fraction is not None:
            self.energy_fraction = energy_fraction
            self.truncation_rank = 0
        else:
            assert truncation_rank is not None
            self.truncation_rank = truncation_rank
            self.energy_fraction = None

        self.frame_size = (
            frame_size
            if frame_size is not None
            else max(self.window_length + 8, (self.window_length * 3 + 1) // 2)
        )
        self.hop_size = (
            hop_size if hop_size is not None else max(1, self.frame_size // 2)
        )
        self.max_working_memory_bytes = max_working_memory_bytes
        self.max_input_samples = _resolve_max_input_samples(max_input_samples)
        self.logger = get_logger(self.__class__.__name__)
        self._ola_engine = SoundfileOlaEngine(
            logger=self.logger,
            max_input_samples=self.max_input_samples,
            max_working_memory_bytes=self.max_working_memory_bytes,
            producer_fill_queue=_producer_fill_queue,
        )

    def process_file(self, input_path: str, output_path: str) -> None:
        """Process one audio file and write the denoised result."""
        try:
            self._validate_paths(input_path, output_path)
            self._validate_configuration()
            self._run_processing(input_path, output_path)
        except HankelPurifyError as exc:
            self.logger.error("处理失败：%s", exc)
            raise
        except MSSA_LINALG_ERRORS as exc:
            self.logger.error("数值计算失败（线性代数 / dense SVD）：%s", exc)
            raise ProcessingError(
                "MSSA numerical step failed (linear algebra).",
                code=ProcessingFailureCode.MSSA_NUMERIC,
                origin_exception_type=exception_fully_qualified_name(exc),
            ) from exc
        except MSSA_ARPACK_ERRORS as exc:
            self.logger.error("数值计算失败（稀疏 ARPACK / svds）：%s", exc)
            raise ProcessingError(
                "MSSA numerical step failed (linear algebra).",
                code=ProcessingFailureCode.MSSA_NUMERIC,
                origin_exception_type=exception_fully_qualified_name(exc),
            ) from exc
        except ValueError as exc:
            self.logger.error(
                "处理失败（ValueError）[stage=%s]：%s",
                format_exception_origin(exc),
                exc,
            )
            raise ProcessingError(
                "MSSA step failed (constraint or internal error).",
                code=ProcessingFailureCode.CONSTRAINT_VALUE,
                origin_exception_type=exception_fully_qualified_name(exc),
            ) from exc
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            # Last resort after HankelPurify / LinAlg / ARPACK / ValueError. Prefer
            # adding explicit except above for stable numeric stacks (see
            # linalg_errors) or new HankelPurifyError mappings—not string/__module__
            # routing. Typical hits: MemoryError, bugs in deps, or types not yet
            # listed alongside MSSA_LINALG_ERRORS / MSSA_ARPACK_ERRORS.
            self.logger.exception(
                "Unexpected error during audio purification (%s)",
                type(exc).__name__,
            )
            raise ProcessingError(
                "Unexpected error during audio processing (see logs).",
                code=ProcessingFailureCode.UNEXPECTED,
                origin_exception_type=exception_fully_qualified_name(exc),
            ) from exc

    def _validate_paths(self, input_path: str, output_path: str) -> None:
        input_file = Path(input_path)
        output_file = Path(output_path)
        validate_io_paths(input_file, output_file)
        if not input_file.exists():
            raise AudioIOError(input_file_does_not_exist(input_path))
        if not input_file.is_file():
            raise AudioIOError(input_path_not_a_file(input_path))

        if not output_file.parent.exists():
            try:
                output_file.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise AudioIOError(
                    unable_to_create_output_directory(str(output_file.parent))
                ) from exc

        in_res = input_file.resolve()
        out_res = output_file.resolve()
        if in_res == out_res:
            raise ConfigurationError(
                "Input and output must differ; in-place overwrite would corrupt audio."
            )
        if output_file.exists() and output_file.is_file():
            try:
                if os.path.samefile(input_path, output_path):
                    raise ConfigurationError(
                        "Input and output refer to the same file (e.g. hard link)."
                    )
            except ConfigurationError:
                raise
            except OSError as exc:
                raise ConfigurationError(
                    "Cannot verify that input and output refer to distinct files "
                    "(filesystem identity check failed). Use different paths or fix "
                    "permissions."
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

    def _make_denoise_frame_fn(
        self,
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        strat: TruncationStrategy
        if self.energy_fraction is not None:
            strat = EnergyThresholdStrategy(self.energy_fraction)
        else:
            strat = FixedRankStrategy(self.truncation_rank)
        wc = self.w_corr_threshold
        c_wl = self.window_length if wc is not None else None
        svd_step = make_svd_step(strat, w_corr_threshold=wc, window_length=c_wl)
        return partial(
            process_frame,
            window_length=self.window_length,
            svd_step=svd_step,
        )

    def _run_processing(self, input_path: str, output_path: str) -> None:
        self.logger.info("开始处理流式数据...")
        f_size = self.frame_size
        hop = self.hop_size
        w_sqrt = sqrt_hanning_weights(f_size)[:, np.newaxis]
        w_sq = (w_sqrt * w_sqrt).ravel()

        denoise_frame = self._make_denoise_frame_fn()

        profile = os.environ.get("HSP_PROFILE_OLA", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        wall0 = time.perf_counter() if profile else 0.0

        try:
            self._ola_engine.run_soundfile_ola(
                input_path,
                output_path,
                denoise_frame,
                f_size,
                hop,
                w_sqrt,
                w_sq,
            )
        except (OSError, sf.LibsndfileError) as exc:
            raise AudioIOError(audio_io_failed_pair(input_path, output_path)) from exc

        if profile:
            self.logger.info(
                "HSP_PROFILE_OLA wall time: %.3fs",
                time.perf_counter() - wall0,
            )

        self.logger.info("流式数据处理完成")
