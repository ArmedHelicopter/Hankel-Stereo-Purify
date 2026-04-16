"""In-process CLI tests (improves coverage of ``src/cli.py``)."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.cli import build_parser, main, parse_args
from src.core.exceptions import ConfigurationError
from src.facade.purifier import MSSAPurifierBuilder


def test_parse_args_defaults() -> None:
    ns = parse_args(["in.flac", "out.flac"])
    assert ns.input_path == "in.flac"
    assert ns.output_path == "out.flac"
    assert ns.window_length == 256
    assert ns.rank is None
    assert ns.energy_fraction is None
    assert ns.w_corr_threshold is None


def test_parse_args_w_corr_threshold() -> None:
    ns = parse_args(["a.flac", "b.flac", "--w-corr-threshold", "0.25"])
    assert ns.w_corr_threshold == 0.25


def test_build_parser_rejects_nonpositive_window() -> None:
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["-L", "0", "a.flac", "b.flac"])


def test_main_rejects_energy_fraction_out_of_range(tmp_path: Path) -> None:
    inp = tmp_path / "i.flac"
    out = tmp_path / "o.flac"
    sf.write(
        inp,
        np.zeros((50, 2), dtype=np.float64),
        48_000,
        format="FLAC",
        subtype="PCM_24",
    )
    with pytest.raises(SystemExit) as e:
        main([str(inp), str(out), "--energy-fraction", "1.5"])
    assert e.value.code == 2


def test_main_runs_on_short_flac_with_w_corr_threshold(tmp_path: Path) -> None:
    """``--w-corr-threshold 0`` is permissive (W entries are in [0,1])."""
    inp = tmp_path / "iw.flac"
    out = tmp_path / "ow.flac"
    rng = np.random.default_rng(42)
    stereo = (0.01 * rng.standard_normal((120, 2))).astype(np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")
    main(
        [
            str(inp),
            str(out),
            "-L",
            "16",
            "-k",
            "8",
            "--frame-size",
            "64",
            "--hop",
            "32",
            "--max-memory-mb",
            "500",
            "--w-corr-threshold",
            "0",
        ]
    )
    assert out.is_file()


def test_main_rejects_w_corr_threshold_out_of_range(tmp_path: Path) -> None:
    inp = tmp_path / "iw.flac"
    out = tmp_path / "ow.flac"
    sf.write(
        inp,
        np.zeros((120, 2), dtype=np.float64),
        48_000,
        format="FLAC",
        subtype="PCM_24",
    )
    with pytest.raises(SystemExit) as e:
        main(
            [
                str(inp),
                str(out),
                "-L",
                "16",
                "-k",
                "8",
                "--frame-size",
                "64",
                "--hop",
                "32",
                "--w-corr-threshold",
                "1.5",
            ]
        )
    assert e.value.code == 2


def test_main_runs_on_short_flac(tmp_path: Path) -> None:
    inp = tmp_path / "i.flac"
    out = tmp_path / "o.flac"
    rng = np.random.default_rng(42)
    stereo = (0.01 * rng.standard_normal((120, 2))).astype(np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")
    main(
        [
            str(inp),
            str(out),
            "-L",
            "16",
            "-k",
            "8",
            "--frame-size",
            "64",
            "--hop",
            "32",
            "--max-memory-mb",
            "500",
        ]
    )
    assert out.is_file()


def test_main_hankel_purify_error_exits_1(tmp_path: Path) -> None:
    missing = tmp_path / "missing.flac"
    out = tmp_path / "o.flac"
    with pytest.raises(SystemExit) as e:
        main([str(missing), str(out)])
    assert e.value.code == 1


def test_main_configuration_error_exits_2_same_paths(tmp_path: Path) -> None:
    p = tmp_path / "same.flac"
    sf.write(
        p,
        np.zeros((80, 2), dtype=np.float64),
        48_000,
        format="FLAC",
        subtype="PCM_24",
    )
    with pytest.raises(SystemExit) as e:
        main(
            [
                str(p),
                str(p),
                "-L",
                "16",
                "-k",
                "8",
                "--frame-size",
                "64",
                "--hop",
                "32",
            ]
        )
    assert e.value.code == 2


def test_main_max_samples_exits_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HSP_MAX_SAMPLES", raising=False)
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    sf.write(
        inp,
        np.zeros((500, 2), dtype=np.float64),
        48_000,
        format="FLAC",
        subtype="PCM_24",
    )
    with pytest.raises(SystemExit) as e:
        main(
            [
                str(inp),
                str(out),
                "--max-samples",
                "100",
                "-L",
                "16",
                "-k",
                "8",
                "--frame-size",
                "64",
                "--hop",
                "32",
            ]
        )
    assert e.value.code == 2


def test_invalid_hsp_max_samples_env_raises_on_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HSP_MAX_SAMPLES", "not_an_int")
    with pytest.raises(ConfigurationError, match="HSP_MAX_SAMPLES"):
        (MSSAPurifierBuilder().set_window_length(16).set_truncation_rank(8).build())
