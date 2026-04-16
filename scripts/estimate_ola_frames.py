#!/usr/bin/env python3
"""Print OLA frame count for given N, frame_size, hop (uses ``list_frame_starts``).

Optionally pass ``--window-length L`` to print ``min(L, 2*K)`` with
``K = frame_size - L + 1`` (Hankel columns per channel), a coarse indicator
of per-frame SVD problem size (see README "单帧 SVD 规模").

Example::

    PYTHONPATH=src python scripts/estimate_ola_frames.py 480000 512 256
    PYTHONPATH=src python scripts/estimate_ola_frames.py 480000 512 256 -L 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.facade.ola import list_frame_starts  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Estimate OLA frame starts count.")
    p.add_argument("num_samples", type=int, help="Total PCM samples per channel")
    p.add_argument("frame_size", type=int, help="OLA frame length F")
    p.add_argument("hop_size", type=int, help="OLA hop H")
    p.add_argument(
        "-L",
        "--window-length",
        type=int,
        default=None,
        metavar="L",
        help=("If set, also print min(L, 2*K) for K=F-L+1 (SVD inner dimension hint)."),
    )
    args = p.parse_args()
    starts = list_frame_starts(args.num_samples, args.frame_size, args.hop_size)
    print(len(starts))
    if args.window_length is not None:
        wl = args.window_length
        k_h = args.frame_size - wl + 1
        if k_h <= 0:
            print(
                f"# window_length L={wl} too large for frame_size F={args.frame_size} "
                f"(need K=F-L+1>0)",
                file=sys.stderr,
            )
        else:
            inner = min(wl, 2 * k_h)
            print(f"# min(L,2K)={inner}  (L={wl}, K={k_h}, F={args.frame_size})")


if __name__ == "__main__":
    main()
