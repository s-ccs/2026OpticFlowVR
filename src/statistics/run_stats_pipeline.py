#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

STEPS = [
    (
        "Export fixation-cleaned epochs for Unfold",
        [sys.executable, SCRIPT_DIR / "prepare_for_unfold.py"],
    ),
    (
        "Check Unfold export",
        [sys.executable, SCRIPT_DIR / "check_unfold_export.py"],
    ),
    (
        "Fit subject-level Unfold models",
        [
            "julia",
            "--project=.",
            SCRIPT_DIR / "run_unfold_all_subjects.jl",
        ],
    ),
    (
        "Run group-level uncorrected tests",
        [
            "julia",
            "--project=.",
            SCRIPT_DIR / "unfold_group_statistics.jl",
        ],
    ),
    (
        "Run cluster-based permutation tests",
        [
            "julia",
            "--project=.",
            SCRIPT_DIR / "cluster_correct_unfold_betas.jl",
        ],
    ),
    (
        "Plot group beta time courses",
        [
            "julia",
            "--project=.",
            SCRIPT_DIR / "plot_unfold_group_betas.jl",
        ],
    ),
    (
        "Plot significant beta topographies",
        [
            "julia",
            "--project=.",
            SCRIPT_DIR / "plot_unfold_topographies.jl",
        ],
    ),
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete Unfold statistical-analysis pipeline"
        )
    )

    parser.add_argument(
        "--skip-export",
        action="store_true",
        help=(
            "Skip prepare_for_unfold.py and check_unfold_export.py"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without running them",
    )

    return parser.parse_args()

def format_duration(seconds: float) -> str:
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours} h {minutes} min {seconds} s"

    if minutes:
        return f"{minutes} min {seconds} s"

    return f"{seconds} s"

def validate_environment() -> None:
    if shutil.which("julia") is None:
        raise RuntimeError(
            "Julia was not found on PATH. Check with `julia --version`"
        )

    missing_scripts = [
        str(command[-1])
        for _, command in STEPS
        if isinstance(command[-1], Path) and not command[-1].exists()
    ]

    if missing_scripts:
        formatted = "\n".join(f"  - {path}" for path in missing_scripts)
        raise FileNotFoundError(
            "The following pipeline scripts were not found:\n"
            f"{formatted}"
        )

    project_file = PROJECT_ROOT / "Project.toml"

    if not project_file.exists():
        raise FileNotFoundError(
            f"Julia Project.toml not found at: {project_file}"
        )

def command_as_strings(command: list[str | Path]) -> list[str]:
    return [str(part) for part in command]

def select_steps(
    *,
    skip_export: bool,
) -> list[tuple[str, list[str | Path]]]:
    selected = []

    for name, command in STEPS:
        script_name = Path(command[-1]).name

        if skip_export and script_name in {
            "prepare_for_unfold.py",
            "check_unfold_export.py",
        }:
            continue

        selected.append((name, command))

    return selected

def run_step(
    step_number: int,
    total_steps: int,
    name: str,
    command: list[str | Path],
    *,
    dry_run: bool,
) -> float:
    command = command_as_strings(command)

    print("\n" + "=" * 88)
    print(f"Step {step_number}/{total_steps}: {name}")
    print("=" * 88)
    print("Command:")
    print("  " + " ".join(command))
    print(f"Working directory:\n  {PROJECT_ROOT}")

    if dry_run:
        print("[DRY RUN] Command not executed")
        return 0.0

    start = time.perf_counter()

    try:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as error:
        duration = time.perf_counter() - start
        print("\n" + "-" * 88, file=sys.stderr)
        print(f"PIPELINE FAILED during step {step_number}: {name}", file=sys.stderr)
        print(f"Exit code: {error.returncode}", file=sys.stderr)
        print(f"Elapsed time: {format_duration(duration)}", file=sys.stderr)
        print("-" * 88, file=sys.stderr)

        raise

    duration = time.perf_counter() - start
    print(f"\n[OK] Completed in {format_duration(duration)}")
    return duration


def main() -> int:
    args = parse_args()

    try:
        validate_environment()
    except (RuntimeError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    selected_steps = select_steps(skip_export=args.skip_export)

    print("=" * 88)
    print("UNFOLD STATISTICAL ANALYSIS PIPELINE")
    print("=" * 88)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Scripts:      {SCRIPT_DIR}")
    print(f"Steps:        {len(selected_steps)}")

    if args.skip_export:
        print("Export:       skipped")
    if args.dry_run:
        print("Mode:         dry run")

    pipeline_start = time.perf_counter()
    durations: list[tuple[str, float]] = []

    try:
        for index, (name, command) in enumerate(
            selected_steps,
            start=1,
        ):
            duration = run_step(
                index,
                len(selected_steps),
                name,
                command,
                dry_run=args.dry_run,
            )
            durations.append((name, duration))

    except subprocess.CalledProcessError:
        return 1

    total_duration = time.perf_counter() - pipeline_start

    print("\n" + "=" * 88)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 88)

    if not args.dry_run:
        for name, duration in durations:
            print(
                f"{format_duration(duration):>16}  {name}"
            )
        print("-" * 88)
        print(
            f"{format_duration(total_duration):>16}  Total"
        )

    print("\nMain output directories:")
    print(
        "  "
        + str(
            PROJECT_ROOT
            / "output-iclabel"
            / "unfold_export"
        )
    )
    print(
        "  "
        + str(
            PROJECT_ROOT
            / "output-iclabel"
            / "unfold_results"
        )
    )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
