"""
Clean BIDS EEGLAB .set files before running MNE-BIDS-Pipeline
Creates a cleaned copy of data/ in data_clean/
"""

from pathlib import Path
import shutil
import re
import pandas as pd
import mne
import numpy as np

from bad_channels import BAD_CHANNELS


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BIDS_ROOT = PROJECT_ROOT / "data"
CLEAN_ROOT = PROJECT_ROOT / "data_clean"

EOG_CHANNELS = ["HEOGR", "HEOGL", "VEOGU", "VEOGL"]

DROP_EXACT = ["sampleNumber"]

DROP_KEYWORDS = [
    "Pupil",
    "Eyeball",
    "Eyelid",
    "OpticalAxis",
    "Gaze",
    "Blink",
    "Velocity",
    "Confidence",
]


def interpolate_nans(raw):
    data = raw.get_data()

    for ch_idx, ch_name in enumerate(raw.ch_names):
        channel = data[ch_idx]
        bad = ~np.isfinite(channel)

        if bad.all():
            print(f"  Warning: {ch_name} is all NaN/Inf; filling with zeros.")
            channel[:] = 0.0

        elif bad.any():
            print(f"  Interpolating {bad.sum()} NaN/Inf samples in {ch_name}")
            good_idx = np.where(~bad)[0]
            bad_idx = np.where(bad)[0]
            channel[bad_idx] = np.interp(bad_idx, good_idx, channel[good_idx])

        data[ch_idx] = channel

    raw._data = data
    return raw


def clean_raw(raw):
    channels_to_drop = []

    for ch in raw.ch_names:
        if ch in DROP_EXACT:
            channels_to_drop.append(ch)

        if ch.lower() in ["x", "y"]:
            channels_to_drop.append(ch)

        if any(keyword.lower() in ch.lower() for keyword in DROP_KEYWORDS):
            channels_to_drop.append(ch)

    channels_to_drop = sorted(set(channels_to_drop))

    if channels_to_drop:
        print(f"  Dropping channels: {channels_to_drop}")
        raw.drop_channels(channels_to_drop)

    available_eog = [ch for ch in EOG_CHANNELS if ch in raw.ch_names]

    if available_eog:
        print(f"  Marking EOG channels: {available_eog}")
        raw.set_channel_types({ch: "eog" for ch in available_eog})

    raw = interpolate_nans(raw)

    print(f"  Final channels: {raw.info['nchan']}")
    print(f"  EEG channels: {len(raw.copy().pick('eeg').ch_names)}")
    print(f"  EOG channels: {len(raw.copy().pick('eog').ch_names)}")

    return raw


def copy_bids_sidecars():
    """Copy BIDS dataset, excluding pilot subjects"""

    if CLEAN_ROOT.exists():
        print(f"Removing existing clean dataset: {CLEAN_ROOT}")
        shutil.rmtree(CLEAN_ROOT)

    print(f"Copying BIDS dataset:\n  {BIDS_ROOT}\n→ {CLEAN_ROOT}")

    shutil.copytree(
        BIDS_ROOT,
        CLEAN_ROOT,
        ignore=shutil.ignore_patterns("sub-001"),
    )


def clean_all_set_files():
    set_files = sorted(CLEAN_ROOT.glob("sub-*/ses-*/eeg/*_eeg.set"))

    if not set_files:
        raise RuntimeError(f"No .set files found in {CLEAN_ROOT}")

    print(f"Found {len(set_files)} EEGLAB .set files")

    for set_file in set_files:
        print("\n" + "=" * 80)
        print(f"Cleaning: {set_file.relative_to(CLEAN_ROOT)}")

        subject = set_file.name.split("_")[0].replace("sub-", "")

        raw = mne.io.read_raw_eeglab(set_file, preload=True)

        manual_bads = BAD_CHANNELS.get(subject, [])

        if manual_bads:
            available_bads = [ch for ch in manual_bads if ch in raw.ch_names]
            print(f"  Marking bad channels for sub-{subject}: {available_bads}")
            raw.info["bads"] = available_bads

        raw = clean_raw(raw)

        if raw.info["bads"]:
            print(f"  Interpolating bad EEG channels: {raw.info['bads']}")
            montage = mne.channels.make_standard_montage("standard_1020")
            raw.set_montage(montage, on_missing="ignore")
            raw.interpolate_bads(reset_bads=True)

        raw.export(set_file, fmt="eeglab", overwrite=True)
        print(f"Saved cleaned file: {set_file}")


def parse_trial_info(event_text):
    """Extract condition, direction, and speedIdx from a TRIAL_INFO event"""

    cond_match = re.search(r"cond=([A-Za-z]+)", event_text)
    dir_match = re.search(r"dir=([A-Za-z]+)", event_text)
    speed_match = re.search(r"speedIdx=([0-9]+)", event_text)

    cond = cond_match.group(1) if cond_match else None
    direction = dir_match.group(1) if dir_match else None
    speed_idx = int(speed_match.group(1)) if speed_match else None

    return cond, direction, speed_idx


def make_clean_trial_type(cond, direction):
    """Create clean condition labels"""

    if cond in ["Rotation", "Spiral"] and direction is not None:
        return f"{cond}/{direction}"

    return cond


def strip_marker_prefix(text):
    text = str(text)
    text = text.split("|")[0]
    text = re.sub(r"^\d+-", "", text)
    return text


def clean_events_file(events_file):
    print(f"\nCleaning events file: {events_file.relative_to(CLEAN_ROOT)}")

    events = pd.read_csv(events_file, sep="\t")

    marker_col = "event" if "event" in events.columns else "trial_type"

    clean_rows = []

    last_cond = None
    last_direction = None
    last_speed_idx = None

    for _, row in events.iterrows():
        event_text = strip_marker_prefix(row[marker_col])

        if event_text.startswith("TRIAL_INFO"):
            last_cond, last_direction, last_speed_idx = parse_trial_info(event_text)
            continue

        # Prefer photo_ON over stimOnset
        if event_text == "photo_ON":
            if last_cond is None:
                continue

            clean_row = row.copy()
            clean_row["trial_type"] = make_clean_trial_type(last_cond, last_direction)
            clean_row["condition"] = last_cond
            clean_row["direction"] = last_direction if last_direction else "n/a"
            clean_row["speed_idx"] = last_speed_idx
            clean_rows.append(clean_row)

    if not clean_rows:
        print("  Warning: no photo_ON events found. Falling back to stimOnset.")

        last_cond = None
        last_direction = None
        last_speed_idx = None

        for _, row in events.iterrows():
            event_text = strip_marker_prefix(row[marker_col])

            if event_text.startswith("TRIAL_INFO"):
                last_cond, last_direction, last_speed_idx = parse_trial_info(event_text)
                continue

            if event_text == "stimOnset":
                if last_cond is None:
                    continue

                clean_row = row.copy()
                clean_row["trial_type"] = make_clean_trial_type(last_cond, last_direction)
                clean_row["condition"] = last_cond
                clean_row["direction"] = last_direction if last_direction else "n/a"
                clean_row["speed_idx"] = last_speed_idx
                clean_rows.append(clean_row)

    if not clean_rows:
        raise RuntimeError(f"No clean stimulus-onset events found in {events_file}")

    clean_events = pd.DataFrame(clean_rows)

    keep_cols = [
        "onset",
        "duration",
        "trial_type",
        "condition",
        "direction",
        "speed_idx",
        "sample",
    ]

    keep_cols = [col for col in keep_cols if col in clean_events.columns]
    clean_events = clean_events[keep_cols]

    clean_events.to_csv(events_file, sep="\t", index=False, na_rep="n/a")

    check = pd.read_csv(events_file, sep="\t")
    bad_mask = check["trial_type"].astype(str).str.contains(
        "TRIAL_INFO|OPTIC_FLOW|COMPARE|TRIAL_START|TRIAL_END|PREVIEW|photo_OFF|stimOffset|BAD_NAN",
        regex=True,
    )

    if bad_mask.any():
        bad_examples = check.loc[bad_mask, "trial_type"].head(10).tolist()
        raise RuntimeError(
            f"Event cleaning failed for {events_file}. "
            f"Still found verbose rows: {bad_examples}"
        )

    print(f"  Kept {len(clean_events)} clean stimulus-onset events")
    print(clean_events["trial_type"].value_counts().sort_index())


def clean_all_events_files():
    events_files = sorted(CLEAN_ROOT.glob("sub-*/ses-*/eeg/*_events.tsv"))

    if not events_files:
        print("No events.tsv files found.")
        return

    print(f"\nFound {len(events_files)} events.tsv files")

    for events_file in events_files:
        clean_events_file(events_file)


def main():
    copy_bids_sidecars()
    clean_all_set_files()
    clean_all_events_files()

    print("\nDone.")
    print(f"Cleaned BIDS dataset written to: {CLEAN_ROOT}")


if __name__ == "__main__":
    main()
