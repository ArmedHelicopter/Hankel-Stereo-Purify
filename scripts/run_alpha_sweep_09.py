"""Run high-band whitening alpha sweep on the 10s Adagio slice."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.prepare_whitening_listening_eval import (  # noqa: E402
    read_audio,
    validate_same_shape,
    write_float_wav,
)
from src.core.stages.filter import split_signal  # noqa: E402
from src.core.stages.whitening import rms, snr_db  # noqa: E402
from src.facade.purifier import AudioPurifier  # noqa: E402

RAW = Path(
    "data/raw/Brendel_Beethoven_Piano_Music_Vol9/09_Op27_No2_I_Adagio_sostenuto.mp3"
)
OUT = Path("data/processed/a09")
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SR = 44_100
START_SECONDS = 0.0
DURATION_SECONDS = 10.0
L = 256
ENERGY = 0.9
FRAME = 1024
CUT = 2_000.0


def alpha_name(alpha: float) -> str:
    return f"a{int(round(alpha * 100)):g}"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def cut_input(
    raw: Path,
    out: Path,
    *,
    start_seconds: float = START_SECONDS,
    duration_seconds: float = DURATION_SECONDS,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_seconds),
            "-i",
            str(raw),
            "-t",
            str(duration_seconds),
            "-ar",
            str(SR),
            "-c:a",
            "pcm_f32le",
            str(out),
        ]
    )


def stats(
    original: NDArray[np.float64],
    output: NDArray[np.float64],
    samplerate: int,
) -> dict[str, float]:
    _, high = split_signal(output, CUT, samplerate)
    _, original_high = split_signal(original, CUT, samplerate)
    diff = original - output
    return {
        "rms": rms(output),
        "peak": float(np.max(np.abs(output))) if output.size else 0.0,
        "high_rms": rms(high),
        "high_rms_ratio_vs_original": rms(high) / max(rms(original_high), 1e-30),
        "diff_rms": rms(diff),
        "snr_vs_original_db": snr_db(original, output),
    }


def process_alpha(original_path: Path, out_dir: Path, alpha: float) -> float:
    name = alpha_name(alpha)
    tmp = out_dir / f"_{name}.wav"
    wall0 = time.perf_counter()
    AudioPurifier(
        L,
        energy_fraction=ENERGY,
        frame_size=FRAME,
        bypass_freq=CUT,
        highband_whiten=True,
        whiten_alpha=alpha,
        whitening_artifact_dir=out_dir / f"art_{name}",
    ).process_file(str(original_path), str(tmp))
    data, samplerate = read_audio(tmp)
    write_float_wav(out_dir / f"{name}.wav", data, samplerate)
    try:
        tmp.unlink()
    except OSError:
        pass
    return time.perf_counter() - wall0


def run_sweep(
    raw: Path,
    out_dir: Path,
    *,
    start_seconds: float = START_SECONDS,
    duration_seconds: float = DURATION_SECONDS,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    original_path = out_dir / "orig.wav"
    cut_input(
        raw,
        original_path,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
    )
    original, samplerate = read_audio(original_path)

    wall_times: dict[str, float] = {}
    for alpha in ALPHAS:
        wall_times[alpha_name(alpha)] = process_alpha(original_path, out_dir, alpha)

    outputs = {"orig": original}
    for alpha in ALPHAS:
        name = alpha_name(alpha)
        outputs[name] = read_audio(out_dir / f"{name}.wav")[0]
    validate_same_shape({name: (data, samplerate) for name, data in outputs.items()})

    diff_dir = out_dir / "d"
    metrics: dict[str, Any] = {
        "orig": stats(original, original, samplerate),
    }
    residuals: dict[str, Any] = {}
    for alpha in ALPHAS:
        name = alpha_name(alpha)
        data = outputs[name]
        diff = original - data
        diff_path = diff_dir / f"orig_{name}.wav"
        write_float_wav(diff_path, diff, samplerate)
        metrics[name] = {
            **stats(original, data, samplerate),
            "alpha": alpha,
            "wall_time_seconds": wall_times[name],
        }
        residuals[f"orig_{name}"] = {
            "path": str(diff_path),
            "rms": rms(diff),
            "peak": float(np.max(np.abs(diff))) if diff.size else 0.0,
        }

    summary = {
        "input_path": str(raw),
        "output_root": str(out_dir),
        "start_seconds": start_seconds,
        "samplerate": samplerate,
        "duration_seconds": float(original.shape[0] / samplerate),
        "bypass_freq": CUT,
        "mssa": {
            "window_length": L,
            "energy_fraction": ENERGY,
            "frame_size": FRAME,
        },
        "alphas": list(ALPHAS),
        "paths": {
            "orig": str(original_path),
            **{
                alpha_name(alpha): str(out_dir / f"{alpha_name(alpha)}.wav")
                for alpha in ALPHAS
            },
        },
        "variants": metrics,
        "residuals": residuals,
    }
    summary_path = out_dir / "sum.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    write_summary_md(out_dir / "sum.md", summary)
    return summary_path


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Alpha Sweep 09",
        "",
        f"- Input: `{summary['input_path']}`",
        f"- Output: `{summary['output_root']}`",
        f"- Cutoff: {summary['bypass_freq']} Hz",
        "",
        "| Variant | alpha | high/orig | diff RMS | SNR |",
        "|---------|-------|-----------|----------|-----|",
    ]
    for name, item in summary["variants"].items():
        alpha = item.get("alpha", "-")
        snr = item["snr_vs_original_db"]
        snr_text = "inf" if np.isinf(snr) else f"{snr:.2f}dB"
        lines.append(
            f"| {name} | {alpha} | "
            f"{item['high_rms_ratio_vs_original']:.4g} | "
            f"{item['diff_rms']:.8g} | {snr_text} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW)
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--start", type=float, default=START_SECONDS)
    parser.add_argument("--duration", type=float, default=DURATION_SECONDS)
    args = parser.parse_args()
    summary_path = run_sweep(
        args.input,
        args.output_root,
        start_seconds=args.start,
        duration_seconds=args.duration,
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
