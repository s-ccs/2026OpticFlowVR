from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

STEPS = [
    (
        "Fit 1 + condition + spl(speed, 5) for every participant",
        SCRIPT_DIR / "run_unfold_speed_main_effect.jl",
    ),
    (
        "Hotelling T² + cluster-depth correction on posterior ROI",
        SCRIPT_DIR / "hotelling_clusterdepth_speed.jl",
    ),
]


def main() -> int:
    for i, (name, script) in enumerate(STEPS, start=1):
        print("\n" + "=" * 88)
        print(f"Step {i}/{len(STEPS)}: {name}")
        print("=" * 88)
        subprocess.run(
            ["julia", "--project=.", str(script)],
            cwd=PROJECT_ROOT,
            check=True,
        )

    print("\nCompleted spline inference analysis for speed main effect.")
    print(
        "Results: "
        + str(
            PROJECT_ROOT
            / "output-iclabel"
            / "unfold_results"
            / "speed_main_effect"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
