"""Prepare normalized diff and blind A/B artifacts for whitening experiments."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

DEFAULT_EXPERIMENT_DIR = Path("data/processed/whiten_exp_09_adagio_10s")
DEFAULT_SEED = 20260510
TARGET_PEAK_DBFS = -1.0
TARGET_RMS_DBFS = -38.0
NUMERICAL_RMS_THRESHOLD = 1e-12
NUMERICAL_PEAK_THRESHOLD = 1e-9

DIFF_FILES = {
    "diff_baseline_vs_whiten": Path("artifacts/diff_baseline_vs_whiten.wav"),
    "diff_original_vs_whiten": Path("artifacts/diff_original_vs_whiten.wav"),
    "diff_original_vs_roundtrip": Path("artifacts/diff_original_vs_roundtrip.wav"),
}

AB_FILES = {
    "baseline_no_whiten": Path("artifacts/baseline_no_whiten.wav"),
    "whitened_output": Path("artifacts/whitened_output.wav"),
}


def dbfs_to_amplitude(dbfs: float) -> float:
    return float(10.0 ** (dbfs / 20.0))


def read_audio(path: Path) -> tuple[NDArray[np.float64], int]:
    data, samplerate = sf.read(path, dtype="float64", always_2d=True)
    arr = np.asarray(data, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite audio samples in {path}")
    return arr, int(samplerate)


def write_float_wav(path: Path, data: NDArray[np.float64], samplerate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, samplerate, format="WAV", subtype="FLOAT")


def audio_stats(data: NDArray[np.float64]) -> dict[str, float]:
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    rms = float(np.sqrt(np.mean(data * data))) if data.size else 0.0
    return {"peak": peak, "rms": rms}


def apply_gain(data: NDArray[np.float64], gain: float) -> NDArray[np.float64]:
    return np.asarray(data * float(gain), dtype=np.float64)


def normalize_to_peak(
    data: NDArray[np.float64],
    target_peak: float,
) -> tuple[NDArray[np.float64], float]:
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= 0.0:
        return data.copy(), 1.0
    gain = target_peak / peak
    return apply_gain(data, gain), gain


def normalize_to_rms_with_peak_ceiling(
    data: NDArray[np.float64],
    target_rms: float,
    peak_ceiling: float,
) -> tuple[NDArray[np.float64], float, bool]:
    stats = audio_stats(data)
    if stats["rms"] <= 0.0:
        return data.copy(), 1.0, False
    rms_gain = target_rms / stats["rms"]
    peak_gain = peak_ceiling / stats["peak"] if stats["peak"] > 0.0 else float("inf")
    gain = min(rms_gain, peak_gain)
    return apply_gain(data, gain), gain, gain < rms_gain


def validate_same_shape(
    entries: dict[str, tuple[NDArray[np.float64], int]],
) -> tuple[tuple[int, int], int]:
    shapes = {name: data.shape for name, (data, _) in entries.items()}
    samplerates = {name: sr for name, (_, sr) in entries.items()}
    shape_values = set(shapes.values())
    samplerate_values = set(samplerates.values())
    if len(shape_values) != 1:
        raise ValueError(f"Audio shapes do not match: {shapes}")
    if len(samplerate_values) != 1:
        raise ValueError(f"Sample rates do not match: {samplerates}")
    shape = next(iter(shape_values))
    if len(shape) != 2:
        raise ValueError(f"Expected 2D audio shape, got {shape}")
    return (shape[0], shape[1]), next(iter(samplerate_values))


def prepare_diff_norms(
    experiment_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], tuple[int, int], int]:
    target_peak = dbfs_to_amplitude(TARGET_PEAK_DBFS)
    target_rms = dbfs_to_amplitude(TARGET_RMS_DBFS)
    diff_dir = output_dir / "diff_norm"
    summary: dict[str, Any] = {}
    loaded: dict[str, tuple[NDArray[np.float64], int]] = {}

    for name, relative_path in DIFF_FILES.items():
        path = experiment_dir / relative_path
        data, samplerate = read_audio(path)
        loaded[name] = (data, samplerate)

    shape, samplerate = validate_same_shape(loaded)

    for name, (data, sr) in loaded.items():
        stats = audio_stats(data)
        stats["duration_seconds"] = float(data.shape[0] / sr)
        below_threshold = (
            stats["rms"] < NUMERICAL_RMS_THRESHOLD
            or stats["peak"] < NUMERICAL_PEAK_THRESHOLD
        )
        entry: dict[str, Any] = {
            "source": str(experiment_dir / DIFF_FILES[name]),
            "original": stats,
            "below_numerical_threshold": bool(below_threshold),
            "outputs": {},
        }

        if below_threshold:
            raw_path = diff_dir / f"{name}_unmodified.wav"
            write_float_wav(raw_path, data, sr)
            entry["note"] = "roundtrip diff below numerical threshold"
            entry["outputs"]["unmodified"] = str(raw_path)
        else:
            peak_data, peak_gain = normalize_to_peak(data, target_peak)
            peak_path = diff_dir / f"{name}_peak_norm.wav"
            write_float_wav(peak_path, peak_data, sr)

            rms_data, rms_gain, peak_limited = normalize_to_rms_with_peak_ceiling(
                data,
                target_rms,
                target_peak,
            )
            rms_path = diff_dir / f"{name}_rms_norm.wav"
            write_float_wav(rms_path, rms_data, sr)

            entry["outputs"]["peak_norm"] = {
                "path": str(peak_path),
                "gain": peak_gain,
                "target_peak_dbfs": TARGET_PEAK_DBFS,
            }
            entry["outputs"]["rms_norm"] = {
                "path": str(rms_path),
                "gain": rms_gain,
                "target_rms_dbfs": TARGET_RMS_DBFS,
                "peak_ceiling_dbfs": TARGET_PEAK_DBFS,
                "peak_ceiling_limited": bool(peak_limited),
            }
        summary[name] = entry

    return summary, shape, samplerate


def write_score_sheet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Whitening Blind A/B Score Sheet

Do not open answer_key.json before listening.

## Setup

- Use matched playback level for A.wav and B.wav.
- Loop short passages if needed, but avoid changing EQ or loudness per file.
- Listen for hiss reduction, piano tone, tails, and artifacts.

## Trial Notes

| Trial | File | Hiss level | Piano tone/tails | Artifacts | Preference |
|-------|------|------------|------------------|-----------|------------|
| 1 | A |  |  |  |  |
| 1 | B |  |  |  |  |
| 2 | A |  |  |  |  |
| 2 | B |  |  |  |  |
| 3 | A |  |  |  |  |
| 3 | B |  |  |  |  |

## Diff Re-listening

| File | Hiss/sand only? | Stable pitch? | Piano tail/melody? | Notes |
|------|-----------------|---------------|--------------------|-------|
| diff_baseline_vs_whiten_peak_norm.wav |  |  |  |  |
| diff_baseline_vs_whiten_rms_norm.wav |  |  |  |  |
| diff_original_vs_whiten_peak_norm.wav |  |  |  |  |
| diff_original_vs_whiten_rms_norm.wav |  |  |  |  |
| diff_original_vs_roundtrip_unmodified.wav |  |  |  |  |

## Final Judgment

- More natural:
- Quieter high-frequency hiss:
- Less musical damage:
- Overall preference:
- Confidence:
""",
        encoding="utf-8",
    )


def prepare_blind_ab(
    experiment_dir: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    ab_dir = output_dir / "ab_blind"
    loaded = {
        name: read_audio(experiment_dir / relative_path)
        for name, relative_path in AB_FILES.items()
    }
    _, samplerate = validate_same_shape(loaded)
    stats = {name: audio_stats(data) for name, (data, _) in loaded.items()}
    target_peak = dbfs_to_amplitude(TARGET_PEAK_DBFS)
    rms_candidates = [stat["rms"] for stat in stats.values() if stat["rms"] > 0.0]
    if not rms_candidates:
        raise ValueError("Cannot RMS-match silent A/B inputs.")

    target_rms = min(rms_candidates)
    for stat in stats.values():
        if stat["peak"] > 0.0 and stat["rms"] > 0.0:
            target_rms = min(target_rms, (target_peak / stat["peak"]) * stat["rms"])

    normalized: dict[str, NDArray[np.float64]] = {}
    gains: dict[str, float] = {}
    for name, (data, _) in loaded.items():
        rms = stats[name]["rms"]
        gain = target_rms / rms if rms > 0.0 else 1.0
        normalized[name] = apply_gain(data, gain)
        gains[name] = float(gain)

    rng = random.Random(seed)
    sources = list(normalized.keys())
    rng.shuffle(sources)
    labels = {"A": sources[0], "B": sources[1]}

    for label, source_name in labels.items():
        write_float_wav(ab_dir / f"{label}.wav", normalized[source_name], samplerate)

    answer_key = {
        "seed": seed,
        "mapping": labels,
        "sources": {
            name: str(experiment_dir / path) for name, path in AB_FILES.items()
        },
        "gains": gains,
        "target_rms": target_rms,
        "target_peak_dbfs": TARGET_PEAK_DBFS,
        "processing": "constant gain only; no EQ, compression, limiting, or filtering",
    }
    ab_dir.mkdir(parents=True, exist_ok=True)
    with (ab_dir / "answer_key.json").open("w", encoding="utf-8") as fp:
        json.dump(answer_key, fp, indent=2, ensure_ascii=False)
    write_score_sheet(ab_dir / "score_sheet.md")

    public_mapping = {label: f"{label}.wav" for label in labels}
    return {
        "seed": seed,
        "directory": str(ab_dir),
        "public_files": public_mapping,
        "answer_key": str(ab_dir / "answer_key.json"),
        "score_sheet": str(ab_dir / "score_sheet.md"),
        "source_stats": stats,
        "gains": gains,
        "target_rms": target_rms,
        "target_peak_dbfs": TARGET_PEAK_DBFS,
        "processing": "constant gain only; no EQ, compression, limiting, or filtering",
    }


def prepare_listening_eval(experiment_dir: Path, seed: int) -> Path:
    output_dir = experiment_dir / "listening_eval"
    diff_summary, shape, samplerate = prepare_diff_norms(experiment_dir, output_dir)
    ab_summary = prepare_blind_ab(experiment_dir, output_dir, seed)
    summary = {
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_dir),
        "samplerate": samplerate,
        "shape": {"samples": shape[0], "channels": shape[1]},
        "duration_seconds": float(shape[0] / samplerate),
        "normalization": {
            "target_peak_dbfs": TARGET_PEAK_DBFS,
            "target_rms_dbfs": TARGET_RMS_DBFS,
            "numerical_rms_threshold": NUMERICAL_RMS_THRESHOLD,
            "numerical_peak_threshold": NUMERICAL_PEAK_THRESHOLD,
        },
        "diffs": diff_summary,
        "blind_ab": ab_summary,
        "manual_status": {
            "normalized_diff_relisten": "pending",
            "blind_ab_preference": "pending",
            "experiment_log_subjective_conclusion": "pending",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare whitening diff normalization and blind A/B artifacts.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Whitening experiment directory containing artifacts/.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for reproducible A/B randomization.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary_path = prepare_listening_eval(args.experiment_dir, args.seed)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
