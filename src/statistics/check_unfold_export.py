from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORT_ROOT = PROJECT_ROOT / "output-iclabel" / "unfold_export-n23"

EXPECTED_CONDITIONS = [
    "Forward",
    "Random",
    "Rotation",
    "Spiral",
]

EXPECTED_SPEEDS = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]


def check_subject(subject_dir: Path) -> dict:
    subject = subject_dir.name

    data_file = subject_dir / f"{subject}_data.npy"
    events_file = subject_dir / f"{subject}_events.csv"
    channels_file = subject_dir / f"{subject}_channels.csv"
    times_file = subject_dir / f"{subject}_times.csv"

    for file in [data_file, events_file, channels_file, times_file]:
        if not file.exists():
            raise FileNotFoundError(f"Missing file: {file}")

    data = np.load(data_file, mmap_mode="r")
    events = pd.read_csv(events_file)
    channels = pd.read_csv(channels_file)
    times = pd.read_csv(times_file)

    n_channels, n_times, n_trials = data.shape

    problems = []

    if len(events) != n_trials:
        problems.append(f"events rows ({len(events)}) != data trials ({n_trials})")

    if len(channels) != n_channels:
        problems.append(f"channels rows ({len(channels)}) != data channels ({n_channels})")

    if len(times) != n_times:
        problems.append(f"times rows ({len(times)}) != data times ({n_times})")

    required_cols = [
        "subject",
        "trial",
        "event_name",
        "condition",
        "base_condition",
        "direction",
        "speed_idx",
        "speed",
    ]

    for col in required_cols:
        if col not in events.columns:
            problems.append(f"missing events column: {col}")

    if events[required_cols].isna().any().any():
        bad_cols = events[required_cols].columns[
            events[required_cols].isna().any()
        ].tolist()
        problems.append(f"NaNs in required event columns: {bad_cols}")

    observed_conditions = sorted(events["condition"].unique().tolist())
    missing_conditions = sorted(set(EXPECTED_CONDITIONS) - set(observed_conditions))
    extra_conditions = sorted(set(observed_conditions) - set(EXPECTED_CONDITIONS))

    if missing_conditions:
        problems.append(f"missing conditions: {missing_conditions}")

    if extra_conditions:
        problems.append(f"extra conditions: {extra_conditions}")

    observed_speeds = sorted(events["speed"].unique().tolist())
    missing_speeds = sorted(set(EXPECTED_SPEEDS) - set(observed_speeds))
    extra_speeds = sorted(set(observed_speeds) - set(EXPECTED_SPEEDS))

    if missing_speeds:
        problems.append(f"missing speeds: {missing_speeds}")

    if extra_speeds:
        problems.append(f"extra speeds: {extra_speeds}")

    condition_counts = (
        events["condition"]
        .value_counts()
        .reindex(EXPECTED_CONDITIONS, fill_value=0)
    )

    speed_counts = (
        events["speed"]
        .value_counts()
        .reindex(EXPECTED_SPEEDS, fill_value=0)
    )

    condition_speed_counts = (
        events.groupby(["condition", "speed"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=EXPECTED_CONDITIONS, columns=EXPECTED_SPEEDS, fill_value=0)
    )

    min_cell_count = int(condition_speed_counts.min().min())

    if min_cell_count == 0:
        problems.append("at least one condition × speed cell has 0 trials")

    return {
        "subject": subject,
        "n_channels": n_channels,
        "n_times": n_times,
        "n_trials": n_trials,
        "time_start": float(times["time"].iloc[0]),
        "time_end": float(times["time"].iloc[-1]),
        "min_condition_count": int(condition_counts.min()),
        "min_speed_count": int(speed_counts.min()),
        "min_condition_speed_count": min_cell_count,
        "problems": "; ".join(problems) if problems else "OK",
        "channels": channels["channel"].tolist(),
        "times": times["time"].to_numpy(),
        "condition_counts": condition_counts,
        "speed_counts": speed_counts,
        "condition_speed_counts": condition_speed_counts,
    }


def main():
    subject_dirs = sorted(
        d for d in EXPORT_ROOT.glob("sub-*")
        if d.is_dir()
    )

    if not subject_dirs:
        raise RuntimeError(f"No subject folders found in {EXPORT_ROOT}")

    summaries = []
    all_condition_counts = []
    all_speed_counts = []
    all_condition_speed_counts = []

    reference_channels = None
    reference_times = None

    for subject_dir in subject_dirs:
        result = check_subject(subject_dir)

        if reference_channels is None:
            reference_channels = result["channels"]
            reference_times = result["times"]
        else:
            if result["channels"] != reference_channels:
                result["problems"] += "; channel list/order differs from first subject"

            if not np.allclose(result["times"], reference_times):
                result["problems"] += "; time vector differs from first subject"

        summaries.append({
            "subject": result["subject"],
            "n_channels": result["n_channels"],
            "n_times": result["n_times"],
            "n_trials": result["n_trials"],
            "time_start": result["time_start"],
            "time_end": result["time_end"],
            "min_condition_count": result["min_condition_count"],
            "min_speed_count": result["min_speed_count"],
            "min_condition_speed_count": result["min_condition_speed_count"],
            "problems": result["problems"],
        })

        cc = result["condition_counts"].rename(result["subject"])
        all_condition_counts.append(cc)

        sc = result["speed_counts"].rename(result["subject"])
        all_speed_counts.append(sc)

        csc = result["condition_speed_counts"].copy()
        csc.index = [f"{result['subject']}__{idx}" for idx in csc.index]
        all_condition_speed_counts.append(csc)

    summary_df = pd.DataFrame(summaries)

    condition_counts_df = pd.DataFrame(all_condition_counts)
    speed_counts_df = pd.DataFrame(all_speed_counts)
    condition_speed_counts_df = pd.concat(all_condition_speed_counts)

    qc_dir = EXPORT_ROOT / "qc"
    qc_dir.mkdir(exist_ok=True)

    summary_df.to_csv(qc_dir / "summary.csv", index=False)
    condition_counts_df.to_csv(qc_dir / "condition_counts.csv")
    speed_counts_df.to_csv(qc_dir / "speed_counts.csv")
    condition_speed_counts_df.to_csv(qc_dir / "condition_speed_counts.csv")

    print("\nSummary:")
    print(summary_df.to_string(index=False))

    print("\nCondition counts:")
    print(condition_counts_df.to_string())

    print("\nSpeed counts:")
    print(speed_counts_df.to_string())

    bad = summary_df[summary_df["problems"] != "OK"]

    print("\nQC result:")
    if len(bad) == 0:
        print("All exported subjects passed basic QC.")
    else:
        print("Some subjects need attention:")
        print(bad[["subject", "problems"]].to_string(index=False))

    print(f"\nSaved QC files to: {qc_dir}")


if __name__ == "__main__":
    main()
