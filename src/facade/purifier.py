"""Facade: OLA + MSSA over stereo PCM via libsndfile.

``process_file`` maps failures to stable ``HankelPurifyError`` subclasses.
NumPy/SciPy ``LinAlgError`` and sparse ``ArpackError`` / ``ArpackNoConvergence``
(from ``svds``) are caught explicitly; extend ``src.core.linalg_errors`` when new
numeric stack types need first-class mapping. The final ``except Exception`` is a
last resort—prefer adding explicit branches over ``__module__`` string checks.
"""

import json
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
)
from src.core.linalg_errors import MSSA_ARPACK_ERRORS, MSSA_LINALG_ERRORS
from src.core.process_frame import process_frame
from src.core.stages.filter import split_signal
from src.core.stages.svd import make_svd_step
from src.core.stages.whitening import (
    WhiteningProfile,
    estimate_noise_profile,
    rms,
    roundtrip_whiten_signal,
    snr_db,
    unwhiten_signal,
    whiten_signal,
)
from src.core.strategies.truncation import (
    EnergyThresholdStrategy,
    FixedRankStrategy,
    TruncationStrategy,
)
from src.facade.ola import sqrt_hanning_weights
from src.facade.pcm_producer import producer_fill_queue
from src.facade.soundfile_ola import SoundfileOlaEngine
from src.io.audio_formats import soundfile_write_kwargs, validate_io_paths
from src.io.audio_stream import read_audio_metadata
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
        max_working_memory_bytes: int = 1_500_000_000,
        max_input_samples: int | None = None,
        bypass_freq: float | None = 2_000.0,
        highband_whiten: bool = True,
        whiten_alpha: float = 0.75,
        whitening_artifact_dir: str | Path | None = None,
        use_cuda: bool = False,
    ) -> None:
        if not isinstance(window_length, int) or window_length <= 0:
            raise ConfigurationError("window_length must be a positive integer.")
        modes = [v for v in (truncation_rank, energy_fraction) if v is not None]
        if len(modes) == 0:
            raise ConfigurationError(
                "Missing truncation mode: pass truncation_rank=... for fixed k "
                "or energy_fraction=... for energy-based rank."
            )
        if len(modes) > 1:
            raise ConfigurationError(
                "Use exactly one of truncation_rank, energy_fraction; not both."
            )

        self.window_length = window_length
        self.truncation_rank = 0
        self.energy_fraction = None
        if energy_fraction is not None:
            self.energy_fraction = energy_fraction
        elif truncation_rank is not None:
            self.truncation_rank = truncation_rank

        self.frame_size = (
            frame_size
            if frame_size is not None
            else max(self.window_length + 8, (self.window_length * 3 + 1) // 2)
        )
        # 50% overlap is the only mathematically valid setting for sqrt-Hanning
        # OLA — other ratios cause amplitude modulation artifacts (buzzing).
        self.hop_size = max(1, self.frame_size // 2)
        self.bypass_freq = bypass_freq
        self.highband_whiten = bool(highband_whiten)
        self.whiten_alpha = float(whiten_alpha)
        self.whitening_artifact_dir = (
            Path(whitening_artifact_dir) if whitening_artifact_dir is not None else None
        )
        self.use_cuda = use_cuda
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
        if self.highband_whiten and self.bypass_freq is None:
            raise ConfigurationError(
                "highband_whiten requires bypass_freq / --bypass-freq."
            )
        if self.whitening_artifact_dir is not None and not self.highband_whiten:
            raise ConfigurationError("whitening_artifact_dir requires highband_whiten.")
        if self.highband_whiten and not 0.0 <= self.whiten_alpha <= 1.0:
            raise ConfigurationError("whiten_alpha must be in [0, 1].")
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
        svd_step = make_svd_step(strat, use_cuda=self.use_cuda)
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

        if self.bypass_freq is not None:
            try:
                self._run_with_bandpass(
                    input_path,
                    output_path,
                    denoise_frame,
                    f_size,
                    hop,
                    w_sqrt,
                    w_sq,
                )
            except (OSError, sf.LibsndfileError) as exc:
                raise AudioIOError(
                    audio_io_failed_pair(input_path, output_path)
                ) from exc
        else:
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
                raise AudioIOError(
                    audio_io_failed_pair(input_path, output_path)
                ) from exc

        if profile:
            self.logger.info(
                "HSP_PROFILE_OLA wall time: %.3fs",
                time.perf_counter() - wall0,
            )
        self.logger.info("流式数据处理完成")

    def _run_with_bandpass(
        self,
        input_path: str,
        output_path: str,
        denoise_frame: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        f_size: int,
        hop: int,
        w_sqrt: NDArray[np.float64],
        w_sq: NDArray[np.float64],
    ) -> None:
        """Pre-filter full signal: bypass low band, SVD on high band, recombine."""
        bypass_freq = self.bypass_freq
        assert bypass_freq is not None
        try:
            sr = read_audio_metadata(input_path)["samplerate"]
        except AudioIOError as exc:
            raise AudioIOError(audio_io_failed_pair(input_path, output_path)) from exc
        self.logger.info(
            "Bandpass: bypass <%.0f Hz, SVD on >%.0f Hz (sr=%d)",
            bypass_freq,
            bypass_freq,
            sr,
        )

        signal, file_sr = sf.read(input_path, dtype="float64")
        assert file_sr == sr

        low_band, high_band = split_signal(signal, bypass_freq, sr)

        self.logger.info(
            "Low band RMS: %.6f | High band RMS: %.6f",
            np.sqrt(np.mean(low_band**2)),
            np.sqrt(np.mean(high_band**2)),
        )

        if self.highband_whiten:
            output = self._run_with_whitened_highband(
                input_path=input_path,
                output_path=output_path,
                signal=signal,
                low_band=low_band,
                high_band=high_band,
                samplerate=sr,
                denoise_frame=denoise_frame,
                f_size=f_size,
                hop=hop,
                w_sqrt=w_sqrt,
                w_sq=w_sq,
            )
        else:
            processed_high = self._process_high_band_tempfile(
                high_band,
                sr,
                denoise_frame,
                f_size,
                hop,
                w_sqrt,
                w_sq,
                write_float=True,
            )
            output = low_band + processed_high[: len(low_band)]
            np.clip(output, -1.0, 1.0, out=output)

        wkwargs = soundfile_write_kwargs(output_path)
        sf.write(output_path, output, sr, **wkwargs)

    def _process_high_band_tempfile(
        self,
        high_band: NDArray[np.float64],
        samplerate: int,
        denoise_frame: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        f_size: int,
        hop: int,
        w_sqrt: NDArray[np.float64],
        w_sq: NDArray[np.float64],
        *,
        write_float: bool,
    ) -> NDArray[np.float64]:
        """Write a high-band signal to temp WAV, run OLA+MSSA, and read it back."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_in = tmp.name
        tmp_out = tmp_in + "_out.wav"
        try:
            if write_float:
                sf.write(tmp_in, high_band, samplerate, format="WAV", subtype="FLOAT")
            else:
                sf.write(tmp_in, high_band, samplerate)
            self._ola_engine.run_soundfile_ola(
                tmp_in,
                tmp_out,
                denoise_frame,
                f_size,
                hop,
                w_sqrt,
                w_sq,
            )
            processed_high, _ = sf.read(tmp_out, dtype="float64", always_2d=True)
            return np.asarray(processed_high, dtype=np.float64)
        finally:
            for p in [tmp_in, tmp_out]:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def _run_with_whitened_highband(
        self,
        *,
        input_path: str,
        output_path: str,
        signal: NDArray[np.float64],
        low_band: NDArray[np.float64],
        high_band: NDArray[np.float64],
        samplerate: int,
        denoise_frame: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        f_size: int,
        hop: int,
        w_sqrt: NDArray[np.float64],
        w_sq: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        profile = estimate_noise_profile(high_band, samplerate)
        whitened_high = whiten_signal(
            high_band,
            profile,
            alpha=self.whiten_alpha,
        )
        roundtrip_high = roundtrip_whiten_signal(
            high_band,
            profile,
            alpha=self.whiten_alpha,
        )

        processed_whitened_high = self._process_high_band_tempfile(
            whitened_high,
            samplerate,
            denoise_frame,
            f_size,
            hop,
            w_sqrt,
            w_sq,
            write_float=True,
        )
        processed_high = unwhiten_signal(
            processed_whitened_high[: len(high_band)],
            profile,
            alpha=self.whiten_alpha,
        )
        output = low_band + processed_high[: len(low_band)]
        np.clip(output, -1.0, 1.0, out=output)

        if self.whitening_artifact_dir is not None:
            baseline_high = self._process_high_band_tempfile(
                high_band,
                samplerate,
                self._make_denoise_frame_fn(),
                f_size,
                hop,
                w_sqrt,
                w_sq,
                write_float=True,
            )
            baseline_output = low_band + baseline_high[: len(low_band)]
            np.clip(baseline_output, -1.0, 1.0, out=baseline_output)
            roundtrip = low_band + roundtrip_high[: len(low_band)]
            self._write_whitening_artifacts(
                input_path=input_path,
                output_path=output_path,
                signal=signal,
                baseline_output=baseline_output,
                whitened_output=output,
                roundtrip=roundtrip,
                profile=profile,
                samplerate=samplerate,
            )
        return output

    def _write_float_wav(
        self,
        path: Path,
        data: NDArray[np.float64],
        samplerate: int,
    ) -> None:
        sf.write(str(path), data, samplerate, format="WAV", subtype="FLOAT")

    def _write_whitening_artifacts(
        self,
        *,
        input_path: str,
        output_path: str,
        signal: NDArray[np.float64],
        baseline_output: NDArray[np.float64],
        whitened_output: NDArray[np.float64],
        roundtrip: NDArray[np.float64],
        profile: WhiteningProfile,
        samplerate: int,
    ) -> None:
        artifact_dir = self.whitening_artifact_dir
        assert artifact_dir is not None
        try:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            self._write_float_wav(
                artifact_dir / "roundtrip.wav",
                roundtrip,
                samplerate,
            )
            self._write_float_wav(
                artifact_dir / "baseline_no_whiten.wav",
                baseline_output,
                samplerate,
            )
            self._write_float_wav(
                artifact_dir / "whitened_output.wav",
                whitened_output,
                samplerate,
            )
            self._write_float_wav(
                artifact_dir / "diff_baseline_vs_whiten.wav",
                baseline_output - whitened_output,
                samplerate,
            )
            self._write_float_wav(
                artifact_dir / "diff_original_vs_whiten.wav",
                signal - whitened_output,
                samplerate,
            )
            self._write_float_wav(
                artifact_dir / "diff_original_vs_roundtrip.wav",
                signal - roundtrip,
                samplerate,
            )
            metrics = {
                "input_path": input_path,
                "output_path": output_path,
                "samplerate": samplerate,
                "bypass_freq": self.bypass_freq,
                "whiten_alpha": self.whiten_alpha,
                "roundtrip_snr_db": snr_db(signal, roundtrip),
                "roundtrip_diff_rms": rms(signal - roundtrip),
                "baseline_vs_whiten_snr_db": snr_db(
                    baseline_output,
                    whitened_output,
                ),
                "baseline_vs_whiten_diff_rms": rms(
                    baseline_output - whitened_output,
                ),
                "original_vs_whiten_snr_db": snr_db(signal, whitened_output),
                "original_vs_whiten_diff_rms": rms(signal - whitened_output),
                "profile": {
                    "nperseg": profile.nperseg,
                    "noverlap": profile.noverlap,
                    "percentile": profile.percentile,
                    "eps": profile.eps,
                    "min": float(np.min(profile.scale)),
                    "median": float(np.median(profile.scale)),
                    "max": float(np.max(profile.scale)),
                },
            }
            with (artifact_dir / "metrics.json").open("w", encoding="utf-8") as fp:
                json.dump(metrics, fp, indent=2, ensure_ascii=False)
        except (OSError, sf.LibsndfileError) as exc:
            raise AudioIOError(
                f"Failed to write whitening artifacts to {artifact_dir}."
            ) from exc
