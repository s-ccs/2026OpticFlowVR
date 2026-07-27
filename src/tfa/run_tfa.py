from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def run(script: str, subject: str | None, overwrite: bool):
    command = [sys.executable, str(SCRIPT_DIR / script)]
    if subject is not None:
        command += ["--sub", subject]
    if overwrite and script != "make_group_tfrs.py":
        command += ["--overwrite"]

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SCRIPT_DIR), str(PROJECT_ROOT / "src"), env.get("PYTHONPATH", "")]
    )
    print("\n" + "=" * 80)
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--step",
        choices=["epochs", "tfr", "group", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.step in {"epochs", "all"}:
        run("make_long_epochs.py", args.sub, args.overwrite)
    if args.step in {"tfr", "all"}:
        run("compute_subject_tfr.py", args.sub, args.overwrite)
    if args.step in {"group", "all"}:
        run("make_group_tfrs.py", args.sub, False)

if __name__ == "__main__":
    main()
