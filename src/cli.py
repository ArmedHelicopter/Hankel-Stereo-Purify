"""Command line interface entrypoint for Hankel-Stereo-Purify."""

import argparse
import sys

from src.core.exceptions import HankelPurifyError
from src.facade.purifier import MSSAPurifierBuilder
from src.utils.logger import get_logger

_CLI_VERSION = "0.1.0"


def main() -> None:
    logger = get_logger(__name__)

    parser = argparse.ArgumentParser(
        description="Hankel-Stereo-Purify: MSSA denoising for stereo FLAC (data plane)."
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_CLI_VERSION}",
    )
    parser.add_argument("input_path", help="Path to input FLAC file")
    parser.add_argument("output_path", help="Path to output FLAC file")
    parser.add_argument(
        "-L",
        "--window-length",
        type=int,
        default=256,
        help="Hankel window length L (default: 256)",
    )
    trunc = parser.add_mutually_exclusive_group()
    trunc.add_argument(
        "-k",
        "--rank",
        type=int,
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
        type=int,
        default=None,
        help="OLA frame size in samples (default: derived from L)",
    )
    parser.add_argument(
        "--hop",
        type=int,
        default=None,
        help="OLA hop in samples (default: frame_size // 2)",
    )
    parser.add_argument(
        "--max-memory-mb",
        type=int,
        default=1500,
        help=(
            "RAM budget for OLA accumulators in MiB (default: 1500); "
            "spill to temp if exceeded"
        ),
    )

    args = parser.parse_args()

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
    except HankelPurifyError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except Exception:
        logger.exception("CLI failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
