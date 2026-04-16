"""Command line interface entrypoint for Hankel-Stereo-Purify.

Exit codes: 0 success; 2 ``ConfigurationError``; 1 other ``HankelPurifyError`` or bare
``Exception`` (logged with ``logger.exception`` for unexpected non-Hankel errors).
"""

import argparse
import os
import sys
from collections.abc import Callable

from src.core.exceptions import ConfigurationError, HankelPurifyError
from src.facade.purifier import MSSAPurifierBuilder
from src.utils.logger import get_logger

_CLI_VERSION = "0.1.0"


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
        "--hop",
        type=_positive_int("hop"),
        default=None,
        help="OLA hop in samples (default: frame_size // 2; positive if set)",
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
        "--w-corr-threshold",
        type=float,
        default=None,
        metavar="T",
        help=(
            "Optional MSSA W-correlation filter on singular values after SVD "
            "(uses -L as window length; adds per-frame cost when set)"
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


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
        builder = (
            MSSAPurifierBuilder()
            .set_window_length(args.window_length)
            .set_max_working_memory_bytes(args.max_memory_mb * 1024 * 1024)
        )
        if args.energy_fraction is not None:
            builder = builder.set_energy_fraction(args.energy_fraction)
        else:
            rank = args.rank if args.rank is not None else 64
            builder = builder.set_truncation_rank(rank)
        if args.frame_size is not None:
            builder = builder.set_frame_size(args.frame_size)
        if args.hop is not None:
            builder = builder.set_hop_size(args.hop)
        if args.max_samples is not None:
            builder = builder.set_max_input_samples(args.max_samples)
        if args.w_corr_threshold is not None:
            builder = builder.set_w_corr_threshold(float(args.w_corr_threshold))
        purifier = builder.build()
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
