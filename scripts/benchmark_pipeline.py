#!/usr/bin/env python3
"""Micro-benchmark: A→B→C→D per-frame wall time (perf_counter). No changes to src/.

When ``--w-corr-threshold`` is set, stage C includes W-correlation work; expect
noticeably higher per-frame time than the default (no W filter), especially in
energy mode where the first frame pays full W setup.

Use ``--diag-split`` to print what fraction of stage D wall time is
``batched_diagonal_average`` vs full ``diagonal_reconstruct`` (same tensor).
There is no repo-wide fixed threshold: use the ratio on your target ``L``, ``F``
to judge whether the diagonal kernel is worth further work.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.core.array_types import FloatArray  # noqa: E402
from src.core.exceptions import validate_w_corr_threshold  # noqa: E402
from src.core.stages.a_hankel import hankel_embed  # noqa: E402
from src.core.stages.b_multichannel import combine_hankel_blocks  # noqa: E402
from src.core.stages.c_svd import make_svd_step  # noqa: E402
from src.core.stages.d_diagonal import (  # noqa: E402
    batched_diagonal_average,
    diagonal_reconstruct,
)
from src.core.strategies.truncation import (  # noqa: E402
    EnergyThresholdStrategy,
    FixedRankStrategy,
)


def _median(xs: list[float]) -> float:
    ys = sorted(xs)
    m = len(ys) // 2
    return ys[m] if len(ys) % 2 else 0.5 * (ys[m - 1] + ys[m])


def main() -> None:
    p = argparse.ArgumentParser(description="MSSA core stages micro-benchmark")
    p.add_argument(
        "--frames",
        type=int,
        default=512,
        help="OLA frame length F (samples per ch)",
    )
    p.add_argument("--window-length", type=int, default=256, help="Hankel L")
    p.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Truncation k (default: min(L,2K))",
    )
    p.add_argument(
        "--energy-fraction",
        type=float,
        default=None,
        help="If set, use energy mode",
    )
    p.add_argument(
        "--w-corr-threshold",
        type=float,
        default=None,
        help="If set, measure W-corr cold/hot on same joint tensor",
    )
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument(
        "--cprofile",
        action="store_true",
        help="Dump cProfile top functions for one C+D run",
    )
    p.add_argument(
        "--diag-split",
        action="store_true",
        help=(
            "Report median time share of batched_diagonal_average inside "
            "diagonal_reconstruct (stage D)"
        ),
    )
    args = p.parse_args()
    if args.w_corr_threshold is not None:
        validate_w_corr_threshold(float(args.w_corr_threshold))

    rng = np.random.default_rng(2026)
    f_samples, win_len = args.frames, args.window_length
    if f_samples < win_len:
        raise SystemExit("frames must be >= window-length")
    k_h = f_samples - win_len + 1
    k_default = min(win_len, 2 * k_h)
    k_trunc = int(args.rank) if args.rank is not None else k_default

    stereo = np.ascontiguousarray(rng.standard_normal((f_samples, 2)), dtype=np.float64)

    def run_stages(
        svd_step: Callable[[FloatArray], FloatArray],
    ) -> tuple[float, float, float, float]:
        t0 = time.perf_counter()
        hl, hr = hankel_embed(win_len, stereo)
        t_a = time.perf_counter() - t0
        t0 = time.perf_counter()
        joint = combine_hankel_blocks(hl, hr)
        t_b = time.perf_counter() - t0
        t0 = time.perf_counter()
        mid = svd_step(joint)
        t_c = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = diagonal_reconstruct(mid)
        t_d = time.perf_counter() - t0
        return t_a, t_b, t_c, t_d

    if args.energy_fraction is not None:
        strat: EnergyThresholdStrategy | FixedRankStrategy = EnergyThresholdStrategy(
            args.energy_fraction,
        )
    else:
        strat = FixedRankStrategy(k_trunc)

    c_base = make_svd_step(strat)
    times_a: list[float] = []
    times_b: list[float] = []
    times_c: list[float] = []
    times_d: list[float] = []
    for _ in range(args.warmup):
        run_stages(c_base)
    for _ in range(args.repeats):
        ta, tb, tc, td = run_stages(c_base)
        times_a.append(ta)
        times_b.append(tb)
        times_c.append(tc)
        times_d.append(td)

    ma, mb, mc, md = map(_median, (times_a, times_b, times_c, times_d))
    tot = ma + mb + mc + md

    print(f"F={f_samples} L={win_len} K={k_h} k_trunc={k_trunc} repeats={args.repeats}")
    print(f"Stage A (Hankel view+): {ma * 1e3:.3f} ms  ({100 * ma / tot:.1f}%)")
    print(f"Stage B (hstack):       {mb * 1e3:.3f} ms  ({100 * mb / tot:.1f}%)")
    print(f"Stage C (SVD+recon):    {mc * 1e3:.3f} ms  ({100 * mc / tot:.1f}%)")
    print(f"Stage D (diag avg):     {md * 1e3:.3f} ms  ({100 * md / tot:.1f}%)")
    print(f"Total (median):         {tot * 1e3:.3f} ms")

    if args.diag_split:

        def _d_split_times() -> tuple[float, float]:
            hl, hr = hankel_embed(win_len, stereo)
            joint = combine_hankel_blocks(hl, hr)
            mid = c_base(joint)
            t0 = time.perf_counter()
            _ = diagonal_reconstruct(mid)
            t_full = time.perf_counter() - t0
            _, ncols = mid.shape
            kd = ncols // 2
            both = np.stack((mid[:, :kd], mid[:, kd:]), axis=0)
            t0 = time.perf_counter()
            _ = batched_diagonal_average(both)
            t_bat = time.perf_counter() - t0
            return t_full, t_bat

        for _ in range(args.warmup):
            _d_split_times()
        tfs: list[float] = []
        tbs: list[float] = []
        for _ in range(args.repeats):
            tf, tb = _d_split_times()
            tfs.append(tf)
            tbs.append(tb)
        mf, mb_ = map(_median, (tfs, tbs))
        if mf > 1e-18:
            print(
                f"Stage D split: batched_diagonal_average {100 * mb_ / mf:.1f}% "
                f"of diagonal_reconstruct wall time (median, same mid tensor)"
            )

    if args.w_corr_threshold is not None:
        c_w = make_svd_step(
            strat,
            w_corr_threshold=float(args.w_corr_threshold),
            window_length=win_len,
        )
        hl, hr = hankel_embed(win_len, stereo)
        joint = combine_hankel_blocks(hl, hr)
        t0 = time.perf_counter()
        _ = c_w(joint)
        cold = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = c_w(joint)
        hot = time.perf_counter() - t0
        print(f"Stage C W-corr cold (1st): {cold * 1e3:.3f} ms")
        print(f"Stage C W-corr hot  (2nd): {hot * 1e3:.3f} ms")
        if cold > 1e-9:
            print(f"Hot/Cold ratio: {hot / cold:.3f}")

    if args.cprofile:
        c_prof = make_svd_step(strat)
        hl, hr = hankel_embed(win_len, stereo)
        joint = combine_hankel_blocks(hl, hr)
        pr = cProfile.Profile()
        pr.enable()
        recon = c_prof(joint)
        _ = diagonal_reconstruct(recon)
        pr.disable()
        sio = io.StringIO()
        pstats.Stats(pr, stream=sio).sort_stats("cumulative").print_stats(25)
        print("--- cProfile (cumulative, top 25) ---")
        print(sio.getvalue())


if __name__ == "__main__":
    main()
