"""Run fine high-band whitening alpha sweep on the 10s Adagio slice."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_alpha_sweep_09 import run_sweep  # noqa: E402

RAW = Path(
    "data/raw/Brendel_Beethoven_Piano_Music_Vol9/09_Op27_No2_I_Adagio_sostenuto.mp3"
)
OUT = Path("data/processed/a09b")
DURATION_SECONDS = 3.0


def main() -> None:
    # Reuse the original runner after overriding its module-level sweep settings.
    import scripts.run_alpha_sweep_09 as sweep

    sweep.ALPHAS = (0.75, 0.8, 0.85, 0.9, 0.95, 1.0)  # type: ignore[assignment]
    print(f"Wrote {run_sweep(RAW, OUT, duration_seconds=DURATION_SECONDS)}")


if __name__ == "__main__":
    main()
