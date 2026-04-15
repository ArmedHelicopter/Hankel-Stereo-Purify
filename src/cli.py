"""Command line interface entrypoint for Hankel-Stereo-Purify."""

import argparse

from src.utils.logger import get_logger


def main() -> None:
    logger = get_logger(__name__)

    parser = argparse.ArgumentParser(description="Hankel-Stereo-Purify CLI")
    parser.add_argument("input_path", help="Path to input audio file")
    parser.add_argument("output_path", help="Path to output audio file")
    args = parser.parse_args()

    try:
        logger.info("Input path: %s", args.input_path)
        logger.info("Output path: %s", args.output_path)
        # TODO: wire this CLI into the facade layer
    except Exception:
        logger.exception("Unhandled exception in CLI")


if __name__ == "__main__":
    main()
