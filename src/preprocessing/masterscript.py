import os
from pathlib import Path
import argparse
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def normalize_subject(sub):
    if sub is None:
        return None

    sub = str(sub).replace("sub-", "")
    return sub.zfill(3)


def run_command(command, description):
    print("\n" + "=" * 80)
    print(description)
    print("=" * 80)
    print(" ".join(command))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        env=env,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run full EEG analysis pipeline step by step"
    )

    parser.add_argument(
        "--sub",
        type=str,
        default=None,
        help="Subject number, e.g. 15 or 015. If omitted, runs all subjects",
    )

    args = parser.parse_args()
    subject = normalize_subject(args.sub)

    # 1. Clean BIDS dataset
    clean_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "cleanup_bids_for_mne_pipeline.py"),
    ]

    if subject is not None:
        clean_cmd += ["--sub", subject]

    run_command(clean_cmd, "Step 1: Cleaning BIDS dataset")

    # 2. Run MNE-BIDS-Pipeline
    mne_cmd = [
        "mne_bids_pipeline",
        f"--config={SCRIPT_DIR / 'config_mne_bids_pipeline.py'}",
        "--steps=preprocessing",
    ]

    if subject is not None:
        mne_cmd += ["--subject", subject]

    run_command(mne_cmd, "Step 2: Running MNE-BIDS-Pipeline")

    # 3. Reject trials with poor fixation
    fixation_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "fixation_check.py"),
    ]

    if subject is not None:
        fixation_cmd += ["--sub", subject]

    run_command(fixation_cmd, "Step 3: Running fixation check")

    # 4. Plot ERPs and psychometric curves
    condition_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "plot_condition_erps.py"),
    ]

    group_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "plot_group_erps.py"),
    ]

    psychometric_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "plot_psychometric_curves.py"),
    ]

    if subject is not None:
        condition_cmd += ["--sub", subject]
        group_cmd += ["--sub", subject]
        psychometric_cmd += ["--sub", subject]

    run_command(condition_cmd, "Step 4: Plotting subject condition ERPs")
    run_command(group_cmd, "Step 5: Plotting group ERPs")
    run_command(psychometric_cmd, "Step 6: Plotting psychometric curves")

    print("\nDone.")


if __name__ == "__main__":
    main()
