"""Facade: OLA + MSSA over stereo PCM via libsndfile.

``process_file`` maps exceptions to stable ``HankelPurifyError`` subclasses. For
``RuntimeError``, classification uses ``type(exc).__module__`` with prefix
``numpy`` or ``scipy`` (including submodules e.g. ``numpy.linalg``) as the
numerical stack; other ``RuntimeError`` types are **not** treated as LAPACK/SVD
failures and get a distinct ``ProcessingError`` message so they are not confused
with ``ValueError``/constraint paths. Third-party code that wraps NumPy/SciPy may
use different exception classes; those still surface via the generic
``Exception`` → ``ProcessingError`` path when not ``RuntimeError``.
"""

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from src.core.exceptions import (
    AudioIOError,
    ConfigurationError,
    HankelPurifyError,
    ProcessingError,
    validate_w_corr_threshold,
)
from src.core.linalg_errors import MSSA_LINALG_ERRORS
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
from src.facade.ola import sqrt_hanning_weights
from src.facade.pcm_producer import producer_fill_queue
from src.facade.soundfile_ola import SoundfileOlaMixin
from src.io.audio_formats import validate_io_paths
from src.io.io_messages import (
    audio_io_failed_pair,
    input_file_does_not_exist,
    input_path_not_a_file,
    unable_to_create_output_directory,
)
from src.utils.logger import get_logger


def _resolve_max_input_samples(explicit: int | None) -> int | None:
    """CLI/builder explicit limit wins over ``HSP_MAX_SAMPLES``."""
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


class AudioPurifier(SoundfileOlaMixin):
    """Stereo PCM I/O via soundfile/OLA (F-02), MSSA per frame, optional memmap (NF-01).

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
        max_input_samples: int | None = None,
        w_corr_threshold: float | None = None,
    ) -> None:
        self.window_length = window_length
        self.truncation_rank = truncation_rank
        self.energy_fraction = energy_fraction
        self.w_corr_threshold = w_corr_threshold
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
        validate_w_corr_threshold(w_corr_threshold)
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
        except MSSA_LINALG_ERRORS as exc:
            self.logger.error("数值计算失败（线性代数 / SVD）：%s", exc)
            raise ProcessingError(
                "MSSA numerical step failed (linear algebra)."
            ) from exc
        except ValueError as exc:
            self.logger.error("处理失败（数值约束或内部错误）：%s", exc)
            raise ProcessingError(
                "MSSA step failed (constraint or internal error)."
            ) from exc
        except RuntimeError as exc:
            mod = getattr(type(exc), "__module__", "") or ""
            if mod.startswith(("numpy", "scipy")):
                self.logger.error(
                    "数值计算失败（%s）：%s",
                    type(exc).__name__,
                    exc,
                )
                raise ProcessingError(
                    "MSSA numerical step failed (linear algebra)."
                ) from exc
            self.logger.error(
                "处理失败（未归类为 numpy/scipy 数值栈的 RuntimeError）：%s",
                exc,
            )
            raise ProcessingError(
                "MSSA step failed (runtime error outside numpy/scipy numerical stack)."
            ) from exc
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            self.logger.exception(
                "Unexpected error during audio purification (%s)",
                type(exc).__name__,
            )
            raise ProcessingError(
                "Unexpected error during audio processing (see logs)."
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

    def _build_pipeline(self) -> Pipeline:
        strat: TruncationStrategy
        if self.energy_fraction is not None:
            strat = EnergyThresholdStrategy(self.energy_fraction)
        else:
            strat = FixedRankStrategy(self.truncation_rank)
        wc = self.w_corr_threshold
        c_wl = self.window_length if wc is not None else None
        return Pipeline(
            [
                AHankelStage(self.window_length),
                BMultichannelStage(),
                CSVDStage(strat, w_corr_threshold=wc, window_length=c_wl),
                DDiagonalStage(),
            ],
        )

    def _run_processing(self, input_path: str, output_path: str) -> None:
        self.logger.info("开始处理流式数据...")
        f_size = self.frame_size
        hop = self.hop_size
        w_sqrt = sqrt_hanning_weights(f_size)[:, np.newaxis]
        w_sq = (w_sqrt * w_sqrt).ravel()

        pipeline = self._build_pipeline()

        profile = os.environ.get("HSP_PROFILE_OLA", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        wall0 = time.perf_counter() if profile else 0.0

        try:
            self._run_processing_soundfile(
                input_path,
                output_path,
                pipeline,
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

    def set_max_input_samples(self, value: int | None) -> "MSSAPurifierBuilder":
        """Reject inputs longer than this many samples per channel (optional)."""
        self.params["max_input_samples"] = value
        return self

    def set_w_corr_threshold(self, value: float | None) -> "MSSAPurifierBuilder":
        """Optional W-correlation filter in ``CSVDStage`` (uses window_length as L)."""
        validate_w_corr_threshold(value)
        self.params["w_corr_threshold"] = value
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
        mis = self.params.get("max_input_samples")
        wct = self.params.get("w_corr_threshold")

        if energy_fraction is not None:
            return AudioPurifier(
                wl,
                0,
                energy_fraction=energy_fraction,
                frame_size=fs,
                hop_size=hs,
                max_working_memory_bytes=mem,
                max_input_samples=mis,
                w_corr_threshold=wct,
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
            max_input_samples=mis,
            w_corr_threshold=wct,
        )
