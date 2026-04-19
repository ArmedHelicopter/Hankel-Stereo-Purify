"""Phase0 CLI placeholder. Full MSSA implementation lives on branch ``tutorial``."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description=(
            "Hankel-Stereo-Purify Phase0 skeleton. "
            "Checkout branch `tutorial` for the deliverable MSSA implementation."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.0.0",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
