"""Command line interface entrypoint for Hankel-Stereo-Purify."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hankel-Stereo-Purify CLI")
    parser.add_argument("input_path", help="Path to input audio file")
    parser.add_argument("output_path", help="Path to output audio file")
    args = parser.parse_args()

    # TODO: wire this CLI into the facade layer
    print(f"Input path: {args.input_path}")
    print(f"Output path: {args.output_path}")


if __name__ == "__main__":
    main()
