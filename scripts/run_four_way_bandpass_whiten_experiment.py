"""Run the four-way bandpass/whitening comparison experiment."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.prepare_whitening_listening_eval import (  # noqa: E402
    TARGET_PEAK_DBFS,
    apply_gain,
    audio_stats,
    dbfs_to_amplitude,
    read_audio,
    validate_same_shape,
    write_float_wav,
)
from src.core.stages.filter import split_signal  # noqa: E402
from src.core.stages.whitening import rms, snr_db  # noqa: E402
from src.facade.purifier import AudioPurifier  # noqa: E402

DEFAULT_RAW = Path(
    "data/raw/Brendel_Beethoven_Piano_Music_Vol9/09_Op27_No2_I_Adagio_sostenuto.mp3"
)
DEFAULT_OUTPUT_ROOT = Path("data/processed/fw09")
DEFAULT_START_SECONDS = 0.0
DEFAULT_DURATION_SECONDS = 10.0
DEFAULT_SAMPLE_RATE = 44_100
DEFAULT_WINDOW_LENGTH = 256
DEFAULT_ENERGY_FRACTION = 0.9
DEFAULT_FRAME_SIZE = 1024
DEFAULT_BYPASS_FREQ = 2_000.0
DEFAULT_SEED = 20260510


@dataclass(frozen=True)
class Variant:
    name: str
    output_path: Path
    temp_output_path: Path
    purifier: AudioPurifier


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def cut_input(raw_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(DEFAULT_START_SECONDS),
            "-i",
            str(raw_path),
            "-t",
            str(DEFAULT_DURATION_SECONDS),
            "-ar",
            str(DEFAULT_SAMPLE_RATE),
            "-c:a",
            "pcm_f32le",
            str(output_path),
        ]
    )


def process_variant(input_path: Path, variant: Variant) -> float:
    wall0 = time.perf_counter()
    variant.purifier.process_file(str(input_path), str(variant.temp_output_path))
    data, samplerate = read_audio(variant.temp_output_path)
    write_float_wav(variant.output_path, data, samplerate)
    try:
        variant.temp_output_path.unlink()
    except OSError:
        pass
    return time.perf_counter() - wall0


def audio_metrics(
    original: NDArray[np.float64],
    output: NDArray[np.float64],
    samplerate: int,
    cutoff_hz: float,
) -> dict[str, float]:
    low, high = split_signal(output, cutoff_hz, samplerate)
    orig_low, orig_high = split_signal(original, cutoff_hz, samplerate)
    diff = original - output
    return {
        "rms": rms(output),
        "peak": float(np.max(np.abs(output))) if output.size else 0.0,
        "low_rms": rms(low),
        "high_rms": rms(high),
        "original_low_rms": rms(orig_low),
        "original_high_rms": rms(orig_high),
        "high_rms_ratio_vs_original": rms(high) / max(rms(orig_high), 1e-30),
        "diff_rms": rms(diff),
        "snr_vs_original_db": snr_db(original, output),
    }


def write_residual(
    path: Path,
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    samplerate: int,
) -> dict[str, float | str]:
    diff = left - right
    write_float_wav(path, diff, samplerate)
    return {
        "path": str(path),
        "rms": rms(diff),
        "peak": float(np.max(np.abs(diff))) if diff.size else 0.0,
    }


def rms_match_pair(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, float]]:
    stats = {"first": audio_stats(first), "second": audio_stats(second)}
    target_peak = dbfs_to_amplitude(TARGET_PEAK_DBFS)
    rms_values = [v["rms"] for v in stats.values() if v["rms"] > 0.0]
    if not rms_values:
        raise ValueError("Cannot RMS-match two silent signals.")
    target_rms = min(rms_values)
    for item in stats.values():
        if item["peak"] > 0.0 and item["rms"] > 0.0:
            target_rms = min(target_rms, (target_peak / item["peak"]) * item["rms"])
    first_gain = target_rms / stats["first"]["rms"]
    second_gain = target_rms / stats["second"]["rms"]
    return (
        apply_gain(first, first_gain),
        apply_gain(second, second_gain),
        {
            "first_gain": float(first_gain),
            "second_gain": float(second_gain),
            "target_rms": float(target_rms),
        },
    )


def write_ab_score_sheet(path: Path, pair_name: str) -> None:
    path.write_text(
        f"""# Blind A/B Score Sheet: {pair_name}

Do not open answer_key.json before listening.

| Trial | File | Hiss level | Piano tone/tails | Artifacts | Preference |
|-------|------|------------|------------------|-----------|------------|
| 1 | A |  |  |  |  |
| 1 | B |  |  |  |  |
| 2 | A |  |  |  |  |
| 2 | B |  |  |  |  |
| 3 | A |  |  |  |  |
| 3 | B |  |  |  |  |

## Final Judgment

- Quieter high-frequency hiss:
- Less musical damage:
- Overall preference:
- Confidence:
""",
        encoding="utf-8",
    )


def prepare_blind_ab_pair(
    *,
    pair_name: str,
    first_name: str,
    first_path: Path,
    second_name: str,
    second_path: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    first, first_sr = read_audio(first_path)
    second, second_sr = read_audio(second_path)
    validate_same_shape(
        {
            first_name: (first, first_sr),
            second_name: (second, second_sr),
        }
    )
    first_matched, second_matched, gains = rms_match_pair(first, second)
    normalized = {
        first_name: first_matched,
        second_name: second_matched,
    }
    labels = [first_name, second_name]
    random.Random(seed).shuffle(labels)
    mapping = {"A": labels[0], "B": labels[1]}

    output_dir.mkdir(parents=True, exist_ok=True)
    for label, source_name in mapping.items():
        write_float_wav(output_dir / f"{label}.wav", normalized[source_name], first_sr)

    answer_key = {
        "pair_name": pair_name,
        "seed": seed,
        "mapping": mapping,
        "sources": {
            first_name: str(first_path),
            second_name: str(second_path),
        },
        "gains": {
            first_name: gains["first_gain"],
            second_name: gains["second_gain"],
        },
        "target_rms": gains["target_rms"],
        "target_peak_dbfs": TARGET_PEAK_DBFS,
        "processing": "constant gain only; no EQ, compression, limiting, or filtering",
    }
    with (output_dir / "answer_key.json").open("w", encoding="utf-8") as fp:
        json.dump(answer_key, fp, indent=2, ensure_ascii=False)
    write_ab_score_sheet(output_dir / "score_sheet.md", pair_name)
    return {
        "directory": str(output_dir),
        "answer_key": str(output_dir / "answer_key.json"),
        "score_sheet": str(output_dir / "score_sheet.md"),
        "mapping": mapping,
        "gains": answer_key["gains"],
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Four-Way Bandpass Whitening Experiment",
        "",
        f"- Input: `{summary['input_path']}`",
        f"- Sample rate: {summary['samplerate']}",
        f"- Duration: {summary['duration_seconds']:.3f}s",
        f"- Cutoff: {summary['bypass_freq']} Hz",
        "",
        "## Variant Metrics",
        "",
        "| Variant | high RMS | high/orig | diff RMS | SNR vs original |",
        "|---------|----------|-----------|----------|-----------------|",
    ]
    for name, metrics in summary["variants"].items():
        lines.append(
            "| "
            f"{name} | "
            f"{metrics['high_rms']:.8g} | "
            f"{metrics['high_rms_ratio_vs_original']:.4g} | "
            f"{metrics['diff_rms']:.8g} | "
            f"{metrics['snr_vs_original_db']:.2f} dB |"
        )
    lines.extend(
        [
            "",
            "## Residuals",
            "",
        ]
    )
    for name, item in summary["residuals"].items():
        lines.append(f"- `{name}`: RMS={item['rms']:.8g}, path=`{item['path']}`")
    lines.extend(
        [
            "",
            "## Blind A/B",
            "",
            "- Listen before opening each `answer_key.json`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def run_experiment(raw_path: Path, output_root: Path, seed: int) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    original_path = output_root / "orig.wav"
    cut_input(raw_path, original_path)
    original, samplerate = read_audio(original_path)

    variants = [
        Variant(
            name="fullband_energy_mssa",
            output_path=output_root / "full.wav",
            temp_output_path=output_root / "_tmp_full.wav",
            purifier=AudioPurifier(
                DEFAULT_WINDOW_LENGTH,
                energy_fraction=DEFAULT_ENERGY_FRACTION,
                frame_size=DEFAULT_FRAME_SIZE,
            ),
        ),
        Variant(
            name="bandpass_no_whiten",
            output_path=output_root / "bp.wav",
            temp_output_path=output_root / "_tmp_bp.wav",
            purifier=AudioPurifier(
                DEFAULT_WINDOW_LENGTH,
                energy_fraction=DEFAULT_ENERGY_FRACTION,
                frame_size=DEFAULT_FRAME_SIZE,
                bypass_freq=DEFAULT_BYPASS_FREQ,
            ),
        ),
        Variant(
            name="bandpass_whiten",
            output_path=output_root / "bpw.wav",
            temp_output_path=output_root / "_tmp_bpw.wav",
            purifier=AudioPurifier(
                DEFAULT_WINDOW_LENGTH,
                energy_fraction=DEFAULT_ENERGY_FRACTION,
                frame_size=DEFAULT_FRAME_SIZE,
                bypass_freq=DEFAULT_BYPASS_FREQ,
                highband_whiten=True,
                whitening_artifact_dir=output_root / "art",
            ),
        ),
    ]

    wall_times: dict[str, float] = {}
    for variant in variants:
        wall_times[variant.name] = process_variant(original_path, variant)

    outputs = {
        "original": original,
        **{variant.name: read_audio(variant.output_path)[0] for variant in variants},
    }
    validate_same_shape({name: (data, samplerate) for name, data in outputs.items()})

    diffs_dir = output_root / "d"
    residuals = {
        "diff_original_vs_fullband_energy_mssa": write_residual(
            diffs_dir / "orig_full.wav",
            original,
            outputs["fullband_energy_mssa"],
            samplerate,
        ),
        "diff_original_vs_bandpass_no_whiten": write_residual(
            diffs_dir / "orig_bp.wav",
            original,
            outputs["bandpass_no_whiten"],
            samplerate,
        ),
        "diff_original_vs_bandpass_whiten": write_residual(
            diffs_dir / "orig_bpw.wav",
            original,
            outputs["bandpass_whiten"],
            samplerate,
        ),
        "diff_bandpass_no_whiten_vs_bandpass_whiten": write_residual(
            diffs_dir / "bp_bpw.wav",
            outputs["bandpass_no_whiten"],
            outputs["bandpass_whiten"],
            samplerate,
        ),
    }

    variant_metrics = {
        name: audio_metrics(original, data, samplerate, DEFAULT_BYPASS_FREQ)
        for name, data in outputs.items()
    }
    for name, wall_time in wall_times.items():
        variant_metrics[name]["wall_time_seconds"] = wall_time

    listening_dir = output_root / "ab"
    blind_ab = {
        "fullband_vs_bandpass_whiten": prepare_blind_ab_pair(
            pair_name="fullband_vs_bandpass_whiten",
            first_name="fullband_energy_mssa",
            first_path=output_root / "full.wav",
            second_name="bandpass_whiten",
            second_path=output_root / "bpw.wav",
            output_dir=listening_dir / "full_bpw",
            seed=seed,
        ),
        "bandpass_no_whiten_vs_bandpass_whiten": prepare_blind_ab_pair(
            pair_name="bandpass_no_whiten_vs_bandpass_whiten",
            first_name="bandpass_no_whiten",
            first_path=output_root / "bp.wav",
            second_name="bandpass_whiten",
            second_path=output_root / "bpw.wav",
            output_dir=listening_dir / "bp_bpw",
            seed=seed + 1,
        ),
    }

    whitening_metrics_path = output_root / "art" / "metrics.json"
    whitening_metrics: dict[str, Any] | None = None
    if whitening_metrics_path.exists():
        whitening_metrics = json.loads(whitening_metrics_path.read_text())

    summary: dict[str, Any] = {
        "input_path": str(raw_path),
        "output_root": str(output_root),
        "samplerate": samplerate,
        "duration_seconds": float(original.shape[0] / samplerate),
        "bypass_freq": DEFAULT_BYPASS_FREQ,
        "mssa": {
            "window_length": DEFAULT_WINDOW_LENGTH,
            "energy_fraction": DEFAULT_ENERGY_FRACTION,
            "frame_size": DEFAULT_FRAME_SIZE,
        },
        "paths": {
            "original": str(original_path),
            **{variant.name: str(variant.output_path) for variant in variants},
        },
        "variants": variant_metrics,
        "residuals": residuals,
        "blind_ab": blind_ab,
        "whitening_metrics": whitening_metrics,
    }
    summary_path = output_root / "sum.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    write_summary_md(output_root / "sum.md", summary)
    return summary_path


def main() -> None:
    args = build_parser().parse_args()
    summary_path = run_experiment(args.input, args.output_root, args.seed)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
