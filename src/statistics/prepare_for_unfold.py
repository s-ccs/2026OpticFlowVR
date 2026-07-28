from pathlib import Path
import re

import mne
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EPOCH_ROOT = PROJECT_ROOT / "output" / "derivatives" / "mne-bids-pipeline-iclabel"
OUT_ROOT = PROJECT_ROOT / "output-iclabel" / "unfold_export"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SUBJECTS = [
    "sub-002", 
    "sub-003", 
    "sub-004", 
    "sub-005", 
    "sub-006", 
    # "sub-007", # exluded: abnormal PSD signal
    "sub-008", 
    # "sub-009", # excluded: missing condition x speed cells
    "sub-010", 
    "sub-011", 
    "sub-012", 
    "sub-013", 
    "sub-014", 
    "sub-015", 
    "sub-016",
    # "sub-018", # excluded: ERP not reproducable across trials
    "sub-019", 
    # "sub-020", # excluded: all posterior channels are interpolated
    "sub-021", 
    "sub-022", 
    "sub-023", 
    "sub-024",
    "sub-026", 
    "sub-027", 
    "sub-029",
    "sub-030", 
    "sub-031", 
    "sub-032",
    "sub-034",
    "sub-035",
    "sub-036",
    "sub-037",
    "sub-038",
    "sub-039",
    "sub-040",
]

SPEED_MAP = {
    0: 0.8,
    1: 1.0,
    2: 1.2,
    3: 1.4,
    4: 1.6,
    5: 1.8,
    6: 2.0,
}

BASELINE = (-0.2, 0.0)


def find_epochs_file(subject: str) -> Path | None:
    eeg_dir = EPOCH_ROOT / subject / "ses-001" / "eeg"

    preferred = eeg_dir / f"{subject}_ses-001_task-compareSpeed_proc-clean_fixation-epo.fif"
    fallback = eeg_dir / f"{subject}_ses-001_task-compareSpeed_proc-clean_epo.fif"

    if preferred.exists():
        return preferred

    if fallback.exists():
        return fallback

    return None


def parse_event_name(event_name: str) -> dict:
    event_name = str(event_name)

    speed_match = re.search(r"speed-(\d+)", event_name)
    if speed_match is None:
        raise ValueError(f"Could not parse speed_idx from event_name: {event_name}")

    speed_idx = int(speed_match.group(1))
    parts = event_name.split("/")
    base_condition = parts[0]

    if base_condition in ["Rotation", "Spiral"]:
        direction = parts[1]
    else:
        direction = "none"
    condition = base_condition

    if speed_idx not in SPEED_MAP:
        raise ValueError(f"Unexpected speed_idx={speed_idx} in {event_name}")

    return {
        "event_name": event_name,
        "base_condition": base_condition,
        "direction": direction,
        "condition": condition,
        "speed_idx": speed_idx,
        "speed": SPEED_MAP[speed_idx],
    }


def export_subject(subject: str) -> bool:
    epochs_file = find_epochs_file(subject)

    if epochs_file is None:
        print(f"[SKIP] {subject}: no epoch file found")
        return False

    print(f"\n[LOAD] {subject}: {epochs_file}")

    epochs = mne.read_epochs(epochs_file, preload=True)

    # Exclude EOG channels
    epochs.pick("eeg")
    # Apply average-reference projection
    epochs.apply_proj()
    epochs.apply_baseline(BASELINE)

    if epochs.metadata is None or "event_name" not in epochs.metadata.columns:
        raise RuntimeError(f"{subject}: epochs.metadata['event_name'] not found")

    parsed_rows = []
    for trial_idx, event_name in enumerate(epochs.metadata["event_name"].to_list()):
        row = parse_event_name(event_name)
        row["trial"] = trial_idx + 1
        row["subject"] = subject
        parsed_rows.append(row)

    events = pd.DataFrame(parsed_rows)

    # MNE gives data as trials x channels x time
    # Transform to Unfold-style as channels x time x trials
    data = epochs.get_data(units=dict(eeg="uV",eog="uV"), copy=True)
    data = np.transpose(data, (1, 2, 0))

    times = epochs.times
    channels = epochs.ch_names

    subject_out = OUT_ROOT / subject
    subject_out.mkdir(parents=True, exist_ok=True)

    np.save(subject_out / f"{subject}_data.npy", data)

    events.to_csv(subject_out / f"{subject}_events.csv", index=False)

    pd.DataFrame({"channel": channels}).to_csv(
        subject_out / f"{subject}_channels.csv",
        index=False,
    )

    pd.DataFrame({"time": times}).to_csv(
        subject_out / f"{subject}_times.csv",
        index=False,
    )

    print(f"[OK] {subject}")
    print(f"     data shape: {data.shape} = channels x times x trials")
    print(f"     events: {events.shape}")
    print(f"     channels: {len(channels)}")
    print(f"     times: {times[0]:.3f} to {times[-1]:.3f} s")

    return True


def main():
    exported = []

    for subject in SUBJECTS:
        ok = export_subject(subject)
        if ok:
            exported.append(subject)

    summary = pd.DataFrame({"subject": exported})
    summary.to_csv(OUT_ROOT / "exported_subjects.csv", index=False)

    print("\nDone.")
    print(f"Exported {len(exported)} subjects:")
    print(exported)
    print(f"Output folder: {OUT_ROOT}")


if __name__ == "__main__":
    main()
