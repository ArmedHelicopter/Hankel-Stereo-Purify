"""Command line interface entrypoint for Hankel-Stereo-Purify.

Exit codes: 0 success; 2 ``ConfigurationError``; 1 other ``HankelPurifyError`` or bare
``Exception`` (logged with ``logger.exception`` for unexpected non-Hankel errors).
"""

import argparse
import os
import sys
from collections.abc import Callable

from src.core.exceptions import (
    ConfigurationError,
    HankelPurifyError,
    ProcessingError,
    format_exception_origin,
)
from src.facade.purifier import AudioPurifier
from src.utils.logger import get_logger

_CLI_VERSION = "0.1.0"
DEFAULT_BYPASS_FREQ = 2_000.0
DEFAULT_HIGHBAND_WHITEN = True
DEFAULT_WHITEN_ALPHA = 0.75


def _positive_int(name: str) -> Callable[[str], int]:
    """argparse type: strictly positive int."""

    def _coerce(value: str) -> int:
        try:
            n = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be an integer") from exc
        if n <= 0:
            raise argparse.ArgumentTypeError(f"{name} must be positive")
        return n

    return _coerce


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hankel-Stereo-Purify: MSSA denoising for stereo PCM via libsndfile "
            "(input/output: .flac .wav .aiff/.aif .ogg — see README)."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_CLI_VERSION}",
    )
    parser.add_argument(
        "input_path",
        help="Input audio path (.flac .wav .aiff .aif .ogg)",
    )
    parser.add_argument(
        "output_path",
        help="Output audio path (same extension set; e.g. .wav or .flac)",
    )
    parser.add_argument(
        "-L",
        "--window-length",
        type=_positive_int("window-length"),
        default=256,
        help="Hankel window length L (default: 256, must be positive)",
    )
    trunc = parser.add_mutually_exclusive_group()
    trunc.add_argument(
        "-k",
        "--rank",
        type=_positive_int("rank"),
        default=None,
        metavar="K",
        help="Fixed SVD truncation rank (default: 64 when not using --energy-fraction)",
    )
    trunc.add_argument(
        "--energy-fraction",
        type=float,
        default=None,
        metavar="F",
        help=(
            "Cumulative singular-value energy threshold in (0, 1]; "
            "adaptive rank per frame (cannot combine with -k)"
        ),
    )
    parser.add_argument(
        "--frame-size",
        type=_positive_int("frame-size"),
        default=None,
        help=("OLA frame size in samples (default: derived from L; positive if set)"),
    )
    parser.add_argument(
        "--max-memory-mb",
        type=_positive_int("max-memory-mb"),
        default=1500,
        help=(
            "RAM budget for OLA accumulators in MiB (default: 1500); "
            "spill to temp if exceeded"
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=_positive_int("max-samples"),
        default=None,
        metavar="N",
        help=(
            "Reject input if it has more than N samples per channel "
            "(optional; overrides HSP_MAX_SAMPLES when set)"
        ),
    )
    parser.add_argument(
        "--bypass-freq",
        type=float,
        default=DEFAULT_BYPASS_FREQ,
        metavar="F",
        help=(
            "Bandpass filter cutoff in Hz: signals below F bypass SVD (bypass), "
            "signals above F go through MSSA denoising. "
            "Based on noise analysis: noise is in high frequencies, low-mid is clean. "
            f"Default: {DEFAULT_BYPASS_FREQ:g}."
        ),
    )
    parser.add_argument(
        "--fullband",
        action="store_true",
        default=False,
        help="Disable the default bandpass/BPW path and run full-band MSSA.",
    )
    parser.add_argument(
        "--highband-whiten",
        dest="highband_whiten",
        action="store_true",
        default=None,
        help=(
            "Whiten the high-band branch before MSSA and unwhiten afterward "
            "(default: enabled; requires bandpass)."
        ),
    )
    parser.add_argument(
        "--no-highband-whiten",
        dest="highband_whiten",
        action="store_false",
        help=("Disable high-band whitening while keeping the bandpass split enabled."),
    )
    parser.add_argument(
        "--whiten-artifact-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory for high-band whitening roundtrip/baseline/diff artifacts "
            "(only with --highband-whiten)."
        ),
    )
    parser.add_argument(
        "--whiten-alpha",
        type=float,
        default=DEFAULT_WHITEN_ALPHA,
        metavar="A",
        help=(
            "Experimental high-band whitening strength in [0, 1] "
            f"(default: {DEFAULT_WHITEN_ALPHA:g}; only with --highband-whiten)."
        ),
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help=(
            "Use GPU-accelerated SVD via cuSOLVER (requires compiled CUDA library). "
            "Only works with --rank (fixed-rank truncation)."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    if args.fullband:
        if args.bypass_freq != DEFAULT_BYPASS_FREQ:
            parser.error("--fullband cannot be combined with --bypass-freq.")
        if "--highband-whiten" in raw_argv:
            parser.error("--fullband cannot be combined with --highband-whiten.")
        args.bypass_freq = None
        args.highband_whiten = False
    elif args.highband_whiten is None:
        args.highband_whiten = DEFAULT_HIGHBAND_WHITEN
    return args


def main(argv: list[str] | None = None) -> None:
    logger = get_logger(__name__)

    args = parse_args(argv)

    if args.energy_fraction is not None:
        ef = args.energy_fraction
        if not (0.0 < ef <= 1.0):
            print(
                "error: --energy-fraction must be in (0, 1].",
                file=sys.stderr,
            )
            sys.exit(2)

    try:
        mem_b = args.max_memory_mb * 1024 * 1024
        if args.energy_fraction is not None:
            purifier = AudioPurifier(
                args.window_length,
                energy_fraction=args.energy_fraction,
                frame_size=args.frame_size,
                max_working_memory_bytes=mem_b,
                max_input_samples=args.max_samples,
                bypass_freq=args.bypass_freq,
                highband_whiten=args.highband_whiten,
                whiten_alpha=args.whiten_alpha,
                whitening_artifact_dir=args.whiten_artifact_dir,
                use_cuda=args.cuda,
            )
        else:
            purifier = AudioPurifier(
                args.window_length,
                truncation_rank=args.rank if args.rank is not None else 64,
                frame_size=args.frame_size,
                max_working_memory_bytes=mem_b,
                max_input_samples=args.max_samples,
                bypass_freq=args.bypass_freq,
                highband_whiten=args.highband_whiten,
                whiten_alpha=args.whiten_alpha,
                whitening_artifact_dir=args.whiten_artifact_dir,
                use_cuda=args.cuda,
            )
        mode = (
            f"energy={purifier.energy_fraction}"
            if purifier.energy_fraction is not None
            else f"k={purifier.truncation_rank}"
        )
        logger.info(
            "Processing L=%s %s frame=%s hop=%s",
            purifier.window_length,
            mode,
            purifier.frame_size,
            purifier.hop_size,
        )
        purifier.process_file(args.input_path, args.output_path)
        logger.info("Wrote %s", args.output_path)
    except ConfigurationError as exc:
        logger.error("%s", exc)
        sys.exit(2)
    except HankelPurifyError as exc:
        logger.error("%s", exc)
        if isinstance(exc, ProcessingError) and exc.origin_exception_type is not None:
            logger.error("Origin exception type: %s", exc.origin_exception_type)
        if isinstance(exc, ProcessingError) and exc.__cause__ is not None:
            logger.error(
                "Caused by [%s]: %s",
                format_exception_origin(exc.__cause__),
                exc.__cause__,
            )
        if (
            isinstance(exc, ProcessingError)
            and exc.code is not None
            and os.environ.get("HSP_DEBUG", "").strip().lower() in ("1", "true", "yes")
        ):
            logger.info("ProcessingFailureCode: %s", exc.code.value)
        sys.exit(1)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        if os.environ.get("HSP_DEBUG", "").strip().lower() in ("1", "true", "yes"):
            logger.exception("CLI failed (%s)", type(exc).__name__)
        else:
            msg = repr(exc)
            if len(msg) > 500:
                msg = msg[:500] + "..."
            logger.error(
                "CLI failed: %s: %s",
                f"{type(exc).__module__}.{type(exc).__name__}",
                msg,
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
