"""Run synthetic failure-mode experiments for BPW + MSSA."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import chirp  # type: ignore[import-untyped]

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.prepare_whitening_listening_eval import (  # noqa: E402
    read_audio,
    validate_same_shape,
    write_float_wav,
)
from src.core.process_frame import process_frame  # noqa: E402
from src.core.stages.filter import split_signal  # noqa: E402
from src.core.stages.svd import make_svd_step  # noqa: E402
from src.core.stages.whitening import rms, snr_db  # noqa: E402
from src.core.strategies.truncation import (  # noqa: E402
    EnergyThresholdStrategy,
    FixedRankStrategy,
    TruncationStrategy,
)
from src.facade.purifier import AudioPurifier  # noqa: E402

DEFAULT_OUTPUT_ROOT = Path("data/processed/failure_modes")
DEFAULT_SAMPLE_RATE = 8_000
DEFAULT_DURATION_SECONDS = 0.75
DEFAULT_SEED = 20260511
DEFAULT_WINDOW_LENGTH = 32
DEFAULT_FRAME_SIZE = 128
DEFAULT_ENERGY_FRACTION = 0.9
DEFAULT_BYPASS_FREQ = 2_000.0
DEFAULT_WHITEN_ALPHA = 0.75
DEFAULT_CASES = (
    "transient_attack",
    "stereo_decorrelation",
    "nonstationary_noise",
    "low_snr",
)
VARIANT_NAMES = (
    "fullband_energy_mssa",
    "bandpass_no_whiten",
    "bpw_default",
)
EPS = 1e-30


@dataclass(frozen=True)
class ExperimentConfig:
    sample_rate: int = DEFAULT_SAMPLE_RATE
    duration_seconds: float = DEFAULT_DURATION_SECONDS
    seed: int = DEFAULT_SEED
    window_length: int = DEFAULT_WINDOW_LENGTH
    frame_size: int = DEFAULT_FRAME_SIZE
    energy_fraction: float = DEFAULT_ENERGY_FRACTION
    bypass_freq: float = DEFAULT_BYPASS_FREQ
    whiten_alpha: float = DEFAULT_WHITEN_ALPHA

    @property
    def num_samples(self) -> int:
        return max(1, int(round(self.sample_rate * self.duration_seconds)))


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    clean: NDArray[np.float64]
    noise: NDArray[np.float64]
    sample_rate: int
    description: str

    @property
    def noisy(self) -> NDArray[np.float64]:
        return np.asarray(self.clean + self.noise, dtype=np.float64)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    output_filename: str
    bypass_freq: float | None
    highband_whiten: bool


class RankTracingAudioPurifier(AudioPurifier):
    """Experiment-local purifier that records energy rank after each SVD frame."""

    rank_trace: list[int]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rank_trace = []

    def _make_denoise_frame_fn(
        self,
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        strat: TruncationStrategy
        if self.energy_fraction is not None:
            strat = EnergyThresholdStrategy(self.energy_fraction)
        else:
            strat = FixedRankStrategy(self.truncation_rank)
        svd_step = make_svd_step(strat, use_cuda=self.use_cuda)

        def denoise_frame(frame: NDArray[np.float64]) -> NDArray[np.float64]:
            out = process_frame(
                frame,
                window_length=self.window_length,
                svd_step=svd_step,
            )
            state = getattr(svd_step, "state", None)
            rank = getattr(state, "energy_k_prev", None)
            if rank is not None:
                self.rank_trace.append(int(rank))
            return out

        return denoise_frame


def _time_axis(config: ExperimentConfig) -> NDArray[np.float64]:
    return np.arange(config.num_samples, dtype=np.float64) / config.sample_rate


def _rng(config: ExperimentConfig, salt: int) -> np.random.Generator:
    return np.random.default_rng(int(config.seed) + int(salt))


def _stereo(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.column_stack((left, right)).astype(np.float64)


def _set_rms(x: NDArray[np.float64], target: float) -> NDArray[np.float64]:
    value = rms(x)
    if value <= 0.0:
        return np.asarray(x, dtype=np.float64)
    return np.asarray(x * (target / value), dtype=np.float64)


def _fit_clean_noise(
    clean: NDArray[np.float64],
    noise: NDArray[np.float64],
    *,
    peak: float = 0.9,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    noisy = clean + noise
    max_abs = float(np.max(np.abs(noisy))) if noisy.size else 0.0
    if max_abs <= peak:
        return (
            clean.astype(np.float64, copy=False),
            noise.astype(np.float64, copy=False),
        )
    scale = peak / max_abs
    return (
        np.asarray(clean * scale, dtype=np.float64),
        np.asarray(noise * scale, dtype=np.float64),
    )


def generate_transient_attack(config: ExperimentConfig) -> SyntheticCase:
    t = _time_axis(config)
    rng = _rng(config, 101)
    clean_left = np.zeros_like(t)
    clean_right = np.zeros_like(t)
    onsets = (0.04, 0.28, 0.52)
    for idx, onset in enumerate(onsets):
        active = np.maximum(t - onset, 0.0)
        env = np.exp(-9.0 * active) * (t >= onset)
        clean_left += 0.12 * env * np.sin(2.0 * np.pi * (330.0 + 35.0 * idx) * t)
        clean_right += 0.11 * env * np.sin(2.0 * np.pi * (495.0 + 45.0 * idx) * t)
        attack = np.exp(-np.square((t - onset) / 0.0035))
        clean_left += 0.035 * attack * np.sin(2.0 * np.pi * 1_850.0 * t)
        clean_right += 0.032 * attack * np.sin(2.0 * np.pi * 2_250.0 * t)
    clean = _stereo(clean_left, clean_right)
    hiss = rng.standard_normal(clean.shape)
    noise = _set_rms(hiss, 0.025)
    clean, noise = _fit_clean_noise(clean, noise)
    return SyntheticCase(
        name="transient_attack",
        clean=clean,
        noise=noise,
        sample_rate=config.sample_rate,
        description="Decaying harmonic notes with short high-frequency attacks.",
    )


def generate_stereo_decorrelation(config: ExperimentConfig) -> SyntheticCase:
    t = _time_axis(config)
    rng = _rng(config, 202)
    left = (
        0.09 * np.sin(2.0 * np.pi * 310.0 * t)
        + 0.055 * np.sin(2.0 * np.pi * 620.0 * t + 0.2)
        + 0.025 * np.sin(2.0 * np.pi * 1_240.0 * t)
    )
    right = (
        0.08 * np.sin(2.0 * np.pi * 470.0 * t + 0.7)
        + 0.05 * np.sin(2.0 * np.pi * 830.0 * t)
        + 0.02 * np.sin(2.0 * np.pi * 1_660.0 * t + 1.1)
    )
    clean = _stereo(left, right)
    shared = rng.standard_normal((config.num_samples, 1))
    independent = rng.standard_normal(clean.shape)
    noise = _set_rms(0.65 * shared + 0.35 * independent, 0.02)
    clean, noise = _fit_clean_noise(clean, noise)
    return SyntheticCase(
        name="stereo_decorrelation",
        clean=clean,
        noise=noise,
        sample_rate=config.sample_rate,
        description=(
            "Different left/right harmonic structures stress joint stereo MSSA."
        ),
    )


def generate_nonstationary_noise(config: ExperimentConfig) -> SyntheticCase:
    t = _time_axis(config)
    clean = _stereo(
        0.08 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.035 * np.sin(2.0 * np.pi * 880.0 * t),
        0.075 * np.sin(2.0 * np.pi * 442.0 * t + 0.25)
        + 0.032 * np.sin(2.0 * np.pi * 884.0 * t + 0.1),
    )
    sweep = chirp(
        t,
        f0=450.0,
        f1=min(3_200.0, 0.8 * config.sample_rate / 2.0),
        t1=max(float(t[-1]), 1.0 / config.sample_rate),
        method="linear",
    )
    burst_env = 0.3 + 0.7 * np.exp(-np.square((t - 0.55 * t[-1]) / 0.13))
    left_noise = burst_env * sweep
    right_noise = burst_env * np.roll(sweep, max(1, config.num_samples // 80))
    noise = _set_rms(_stereo(left_noise, right_noise), 0.045)
    clean, noise = _fit_clean_noise(clean, noise)
    return SyntheticCase(
        name="nonstationary_noise",
        clean=clean,
        noise=noise,
        sample_rate=config.sample_rate,
        description="Time-varying noise sweeps across the 2kHz BPW split.",
    )


def generate_low_snr(config: ExperimentConfig) -> SyntheticCase:
    t = _time_axis(config)
    rng = _rng(config, 404)
    clean = _stereo(
        0.04 * np.sin(2.0 * np.pi * 360.0 * t)
        + 0.018 * np.sin(2.0 * np.pi * 720.0 * t),
        0.038 * np.sin(2.0 * np.pi * 365.0 * t + 0.25)
        + 0.016 * np.sin(2.0 * np.pi * 730.0 * t + 0.5),
    )
    broadband = rng.standard_normal(clean.shape)
    hum = _stereo(
        np.sin(2.0 * np.pi * 120.0 * t),
        np.sin(2.0 * np.pi * 123.0 * t + 0.3),
    )
    noise = _set_rms(0.8 * broadband + 0.2 * hum, 0.12)
    clean, noise = _fit_clean_noise(clean, noise)
    return SyntheticCase(
        name="low_snr",
        clean=clean,
        noise=noise,
        sample_rate=config.sample_rate,
        description="Weak harmonic signal buried below stronger broadband noise.",
    )


CASE_GENERATORS: dict[str, Callable[[ExperimentConfig], SyntheticCase]] = {
    "transient_attack": generate_transient_attack,
    "stereo_decorrelation": generate_stereo_decorrelation,
    "nonstationary_noise": generate_nonstationary_noise,
    "low_snr": generate_low_snr,
}


def generate_case(name: str, config: ExperimentConfig) -> SyntheticCase:
    try:
        return CASE_GENERATORS[name](config)
    except KeyError as exc:
        known = ", ".join(DEFAULT_CASES)
        raise ValueError(f"Unknown failure-mode case {name!r}. Known: {known}") from exc


def variant_specs(config: ExperimentConfig) -> tuple[VariantSpec, ...]:
    return (
        VariantSpec(
            name="fullband_energy_mssa",
            output_filename="fullband_energy_mssa.wav",
            bypass_freq=None,
            highband_whiten=False,
        ),
        VariantSpec(
            name="bandpass_no_whiten",
            output_filename="bandpass_no_whiten.wav",
            bypass_freq=config.bypass_freq,
            highband_whiten=False,
        ),
        VariantSpec(
            name="bpw_default",
            output_filename="bpw_default.wav",
            bypass_freq=config.bypass_freq,
            highband_whiten=True,
        ),
    )


def finite_snr_db(
    reference: NDArray[np.float64],
    candidate: NDArray[np.float64],
) -> float:
    value = float(snr_db(reference, candidate))
    if np.isposinf(value):
        return 300.0
    if np.isneginf(value):
        return -300.0
    if np.isnan(value):
        return 0.0
    return value


def stereo_corr(x: NDArray[np.float64]) -> float:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected stereo array, got shape {arr.shape}")
    left = arr[:, 0] - float(np.mean(arr[:, 0]))
    right = arr[:, 1] - float(np.mean(arr[:, 1]))
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= EPS:
        return 0.0
    return float(np.dot(left, right) / denom)


def mid_side_ratio(x: NDArray[np.float64]) -> float:
    arr = np.asarray(x, dtype=np.float64)
    mid = 0.5 * (arr[:, 0] + arr[:, 1])
    side = 0.5 * (arr[:, 0] - arr[:, 1])
    return rms(side) / max(rms(mid), EPS)


def projection_ratio(
    residual: NDArray[np.float64],
    clean: NDArray[np.float64],
) -> float:
    r = np.asarray(residual, dtype=np.float64).ravel()
    c = np.asarray(clean, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(r) * np.linalg.norm(c))
    if denom <= EPS:
        return 0.0
    return float(abs(np.dot(r, c)) / denom)


def rank_trace_summary(rank_trace: Iterable[int]) -> dict[str, float | int]:
    ranks = np.asarray(list(rank_trace), dtype=np.float64)
    if ranks.size == 0:
        return {
            "rank_count": 0,
            "rank_mean": 0.0,
            "rank_std": 0.0,
            "rank_max_delta": 0.0,
        }
    deltas = np.abs(np.diff(ranks)) if ranks.size > 1 else np.zeros(1)
    return {
        "rank_count": int(ranks.size),
        "rank_mean": float(np.mean(ranks)),
        "rank_std": float(np.std(ranks)),
        "rank_max_delta": float(np.max(deltas)) if deltas.size else 0.0,
    }


def variant_metrics(
    *,
    clean: NDArray[np.float64],
    noisy: NDArray[np.float64],
    output: NDArray[np.float64],
    sample_rate: int,
    bypass_freq: float,
    rank_trace: Iterable[int],
) -> dict[str, float | int]:
    clean_high = split_signal(clean, bypass_freq, sample_rate)[1]
    output_high = split_signal(output, bypass_freq, sample_rate)[1]
    residual = noisy - output
    input_snr = finite_snr_db(clean, noisy)
    output_snr = finite_snr_db(clean, output)
    metrics: dict[str, float | int] = {
        "input_snr_db": input_snr,
        "output_snr_db": output_snr,
        "snr_improvement_db": output_snr - input_snr,
        "clean_error_rms": rms(clean - output),
        "residual_rms": rms(residual),
        "residual_clean_projection_ratio": projection_ratio(residual, clean),
        "highband_retention_ratio": rms(output_high) / max(rms(clean_high), EPS),
        "stereo_corr_delta": stereo_corr(output) - stereo_corr(clean),
        "mid_side_ratio_delta": mid_side_ratio(output) - mid_side_ratio(clean),
    }
    metrics.update(rank_trace_summary(rank_trace))
    return metrics


def process_variant(
    input_path: Path,
    output_path: Path,
    spec: VariantSpec,
    config: ExperimentConfig,
) -> tuple[float, list[int]]:
    purifier = RankTracingAudioPurifier(
        config.window_length,
        energy_fraction=config.energy_fraction,
        frame_size=config.frame_size,
        bypass_freq=spec.bypass_freq,
        highband_whiten=spec.highband_whiten,
        whiten_alpha=config.whiten_alpha,
    )
    wall0 = time.perf_counter()
    purifier.process_file(str(input_path), str(output_path))
    return time.perf_counter() - wall0, purifier.rank_trace


def _assert_finite_metrics(metrics: dict[str, float | int]) -> None:
    for key, value in metrics.items():
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"Non-finite metric {key}: {value!r}")


def write_case_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# Failure Mode: {summary['case']}",
        "",
        summary["description"],
        "",
        "## Input",
        "",
        f"- Sample rate: {summary['sample_rate']}",
        f"- Duration: {summary['duration_seconds']:.3f}s",
        f"- Input SNR: {summary['input_snr_db']:.2f} dB",
        "",
        "## Variant Metrics",
        "",
        "| Variant | Output SNR | Improvement | Clean error RMS | Projection | "
        "Highband retention | Rank mean | Rank max delta |",
        "|---------|------------|-------------|-----------------|------------|"
        "--------------------|-----------|----------------|",
    ]
    for name in VARIANT_NAMES:
        item = summary["variants"][name]
        lines.append(
            f"| {name} | "
            f"{item['output_snr_db']:.2f} dB | "
            f"{item['snr_improvement_db']:.2f} dB | "
            f"{item['clean_error_rms']:.8g} | "
            f"{item['residual_clean_projection_ratio']:.4f} | "
            f"{item['highband_retention_ratio']:.4f} | "
            f"{item['rank_mean']:.2f} | "
            f"{item['rank_max_delta']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Clean: `{summary['paths']['clean']}`",
            f"- Noise: `{summary['paths']['noise']}`",
            f"- Input: `{summary['paths']['input']}`",
        ]
    )
    for name in VARIANT_NAMES:
        paths = summary["variant_paths"][name]
        lines.append(f"- {name}: `{paths['output']}` / residual `{paths['residual']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_root_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Synthetic Failure-Mode Experiment",
        "",
        f"- Output root: `{summary['output_root']}`",
        f"- Cases: {', '.join(summary['cases'])}",
        f"- Seed: {summary['config']['seed']}",
        "",
        "| Case | Variant | Output SNR | Improvement | Projection | Highband | "
        "Rank delta |",
        "|------|---------|------------|-------------|------------|----------|"
        "------------|",
    ]
    for case_name, case_summary in summary["case_summaries"].items():
        for variant_name in VARIANT_NAMES:
            item = case_summary["variants"][variant_name]
            lines.append(
                f"| {case_name} | {variant_name} | "
                f"{item['output_snr_db']:.2f} dB | "
                f"{item['snr_improvement_db']:.2f} dB | "
                f"{item['residual_clean_projection_ratio']:.4f} | "
                f"{item['highband_retention_ratio']:.4f} | "
                f"{item['rank_max_delta']:.2f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_case(
    name: str,
    output_root: Path,
    config: ExperimentConfig,
) -> dict[str, Any]:
    case = generate_case(name, config)
    case_dir = output_root / name
    case_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "clean": case_dir / "clean.wav",
        "noise": case_dir / "noise.wav",
        "input": case_dir / "input.wav",
    }
    write_float_wav(paths["clean"], case.clean, case.sample_rate)
    write_float_wav(paths["noise"], case.noise, case.sample_rate)
    write_float_wav(paths["input"], case.noisy, case.sample_rate)

    input_snr = finite_snr_db(case.clean, case.noisy)
    variant_items: dict[str, Any] = {}
    variant_paths: dict[str, dict[str, str]] = {}
    for spec in variant_specs(config):
        output_path = case_dir / spec.output_filename
        wall_time, ranks = process_variant(paths["input"], output_path, spec, config)
        output, sr = read_audio(output_path)
        validate_same_shape(
            {
                "clean": (case.clean, case.sample_rate),
                spec.name: (output, sr),
            }
        )
        residual = case.noisy - output
        clean_error = case.clean - output
        residual_path = case_dir / f"residual_{spec.name}.wav"
        clean_error_path = case_dir / f"clean_error_{spec.name}.wav"
        write_float_wav(residual_path, residual, case.sample_rate)
        write_float_wav(clean_error_path, clean_error, case.sample_rate)
        metrics = variant_metrics(
            clean=case.clean,
            noisy=case.noisy,
            output=output,
            sample_rate=case.sample_rate,
            bypass_freq=config.bypass_freq,
            rank_trace=ranks,
        )
        metrics["wall_time_seconds"] = wall_time
        _assert_finite_metrics(metrics)
        variant_items[spec.name] = metrics
        variant_paths[spec.name] = {
            "output": str(output_path),
            "residual": str(residual_path),
            "clean_error": str(clean_error_path),
        }

    summary: dict[str, Any] = {
        "case": case.name,
        "description": case.description,
        "sample_rate": case.sample_rate,
        "duration_seconds": float(case.clean.shape[0] / case.sample_rate),
        "input_snr_db": input_snr,
        "paths": {key: str(path) for key, path in paths.items()},
        "variant_paths": variant_paths,
        "variants": variant_items,
        "config": config_to_json(config),
    }
    with (case_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    write_case_summary_md(case_dir / "summary.md", summary)
    return summary


def config_to_json(config: ExperimentConfig) -> dict[str, float | int]:
    return {
        "sample_rate": config.sample_rate,
        "duration_seconds": config.duration_seconds,
        "seed": config.seed,
        "window_length": config.window_length,
        "frame_size": config.frame_size,
        "energy_fraction": config.energy_fraction,
        "bypass_freq": config.bypass_freq,
        "whiten_alpha": config.whiten_alpha,
    }


def parse_cases(value: str) -> tuple[str, ...]:
    cases = tuple(item.strip() for item in value.split(",") if item.strip())
    if not cases:
        raise argparse.ArgumentTypeError("At least one case is required.")
    unknown = sorted(set(cases) - set(DEFAULT_CASES))
    if unknown:
        known = ", ".join(DEFAULT_CASES)
        raise argparse.ArgumentTypeError(
            f"Unknown case(s): {', '.join(unknown)}. Known: {known}"
        )
    return cases


def run_experiment(
    output_root: Path,
    *,
    cases: Iterable[str] = DEFAULT_CASES,
    config: ExperimentConfig = ExperimentConfig(),
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    case_summaries = {
        case_name: run_case(case_name, output_root, config) for case_name in cases
    }
    summary: dict[str, Any] = {
        "output_root": str(output_root),
        "cases": list(case_summaries.keys()),
        "config": config_to_json(config),
        "case_summaries": case_summaries,
    }
    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    write_root_summary_md(output_root / "summary.md", summary)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cases", type=parse_cases, default=DEFAULT_CASES)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--window-length", type=int, default=DEFAULT_WINDOW_LENGTH)
    parser.add_argument("--frame-size", type=int, default=DEFAULT_FRAME_SIZE)
    parser.add_argument(
        "--energy-fraction",
        type=float,
        default=DEFAULT_ENERGY_FRACTION,
    )
    parser.add_argument("--bypass-freq", type=float, default=DEFAULT_BYPASS_FREQ)
    parser.add_argument("--whiten-alpha", type=float, default=DEFAULT_WHITEN_ALPHA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(
        sample_rate=args.sample_rate,
        duration_seconds=args.duration,
        seed=args.seed,
        window_length=args.window_length,
        frame_size=args.frame_size,
        energy_fraction=args.energy_fraction,
        bypass_freq=args.bypass_freq,
        whiten_alpha=args.whiten_alpha,
    )
    summary_path = run_experiment(
        args.output_root,
        cases=args.cases,
        config=config,
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
