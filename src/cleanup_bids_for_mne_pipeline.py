"""
Clean BIDS EEGLAB .set files before running MNE-BIDS-Pipeline
Creates a cleaned copy of data/ in data_clean/
"""

import argparse
from pathlib import Path
from pyprep.find_noisy_channels import NoisyChannels
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

VISUAL_ONSET_OFFSET_S = 0.030  # Unity/LSL marker to photodiode onset
EEG_SFREQ = 1000


def normalize_subject(subject):
    if subject is None:
        return None

    subject = str(subject).replace("sub-", "")
    return subject.zfill(3)


def interpolate_nans(raw, log_file=None):
    data = raw.get_data()

    for ch_idx, ch_name in enumerate(raw.ch_names):
        channel = data[ch_idx]
        bad = ~np.isfinite(channel)

        if bad.all():
            log(f"  Warning: {ch_name} is all NaN/Inf; filling with zeros.", log_file)
            channel[:] = 0.0

        elif bad.any():
            log(f"  Interpolating {bad.sum()} NaN/Inf samples in {ch_name}", log_file)
            good_idx = np.where(~bad)[0]
            bad_idx = np.where(bad)[0]
            channel[bad_idx] = np.interp(bad_idx, good_idx, channel[good_idx])

        data[ch_idx] = channel

    raw._data = data
    return raw


def clean_raw(raw, log_file=None):
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
        log(f"  Dropping channels: {channels_to_drop}", log_file)
        raw.drop_channels(channels_to_drop)

    available_eog = [ch for ch in EOG_CHANNELS if ch in raw.ch_names]

    if available_eog:
        log(f"  Marking EOG channels: {available_eog}", log_file)
        raw.set_channel_types({ch: "eog" for ch in available_eog})

    raw = interpolate_nans(raw, log_file)

    log(f"  Final channels: {raw.info['nchan']}", log_file)
    log(f"  EEG channels: {len(raw.copy().pick('eeg').ch_names)}", log_file)
    log(f"  EOG channels: {len(raw.copy().pick('eog').ch_names)}", log_file)

    return raw


def detect_bad_channels_with_pyprep(raw, subject, log_file=None):
    """Detect bad EEG channels using PyPREP"""

    log(f"  Running PyPREP bad-channel detection for sub-{subject}", log_file)

    raw_eeg = raw.copy().pick("eeg")

    if len(raw_eeg.ch_names) == 0:
        log("  No EEG channels found for PyPREP.", log_file)
        return []

    montage = mne.channels.make_standard_montage("standard_1020")
    raw_eeg.set_montage(montage, on_missing="ignore")

    noisy = NoisyChannels(
        raw_eeg,
        do_detrend=True,
        random_state=42,
    )

    noisy.find_all_bads()

    pyprep_bads = noisy.get_bads()

    log(f"  PyPREP bad channels: {pyprep_bads}", log_file)

    return pyprep_bads


def create_clean_dataset_copy(subject=None, log_file=None):
    """Copy BIDS dataset, excluding pilot subjects"""

    # --------------------------------------------------
    # Full dataset
    # --------------------------------------------------
    if subject is None:
        if CLEAN_ROOT.exists():
            log(f"Removing existing clean dataset: {CLEAN_ROOT}", log_file)
            shutil.rmtree(CLEAN_ROOT)

        log(
            f"Copying full BIDS dataset:\n"
            f"  {BIDS_ROOT}\n"
            f"-> {CLEAN_ROOT}",
            log_file,
        )

        shutil.copytree(
            BIDS_ROOT,
            CLEAN_ROOT,
            ignore=shutil.ignore_patterns("sub-001"),
        )

        return

    # --------------------------------------------------
    # Single subject
    # --------------------------------------------------
    src_subject = BIDS_ROOT / f"sub-{subject}"
    dst_subject = CLEAN_ROOT / f"sub-{subject}"

    if not src_subject.exists():
        raise FileNotFoundError(
            f"Subject not found: {src_subject}"
        )

    CLEAN_ROOT.mkdir(parents=True, exist_ok=True)

    if dst_subject.exists():
        log(
            f"Removing existing clean copy: {dst_subject}",
            log_file,
        )
        shutil.rmtree(dst_subject)

    log(
        f"Copying subject:\n"
        f"  {src_subject}\n"
        f"-> {dst_subject}",
        log_file,
    )

    shutil.copytree(src_subject, dst_subject)

def clean_all_set_files(subject=None):
    if subject is None:
        set_files = sorted(CLEAN_ROOT.glob("sub-*/ses-*/eeg/*_eeg.set"))
    else:
        set_files = sorted(CLEAN_ROOT.glob(f"sub-{subject}/ses-*/eeg/*_eeg.set"))

    if not set_files:
        raise RuntimeError(f"No .set files found in {CLEAN_ROOT}")

    print(f"Found {len(set_files)} EEGLAB .set files")

    for set_file in set_files:
        subject = set_file.name.split("_")[0].replace("sub-", "")

        output_dir = PROJECT_ROOT / "output" / f"sub-{subject}"
        output_dir.mkdir(parents=True, exist_ok=True)

        log_file = output_dir / "clean_bids.log"
        log_file.write_text("", encoding="utf-8")

        log("\n" + "=" * 80, log_file)
        log(f"Cleaning: {set_file.relative_to(CLEAN_ROOT)}", log_file)

        raw = mne.io.read_raw_eeglab(set_file, preload=True)
        raw = clean_raw(raw, log_file)

        manual_bads = BAD_CHANNELS.get(subject, [])
        manual_bads = [ch for ch in manual_bads if ch in raw.ch_names]

        pyprep_bads = detect_bad_channels_with_pyprep(raw, subject, log_file)
        pyprep_bads = [ch for ch in pyprep_bads if ch in raw.ch_names]

        all_bads = sorted(set(manual_bads + pyprep_bads))

        if all_bads:
            log(f"  Manual bad channels: {manual_bads}", log_file)
            log(f"  PyPREP bad channels: {pyprep_bads}", log_file)
            log(f"  Final bad channels for interpolation: {all_bads}", log_file)

            raw.info["bads"] = all_bads

            montage = mne.channels.make_standard_montage("standard_1020")
            raw.set_montage(montage, on_missing="ignore")

            raw.interpolate_bads(reset_bads=True)
        else:
            log("  No bad EEG channels marked.", log_file)

        raw.export(set_file, fmt="eeglab", overwrite=True)
        log(f"Saved cleaned file: {set_file}", log_file)


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
    text = re.sub(r"^\d+-", "", str(text))
    return text


def clean_events_file(events_file):
    subject = events_file.name.split("_")[0].replace("sub-", "")
    log_file = PROJECT_ROOT / "output" / f"sub-{subject}" / "clean_bids.log"

    log(f"\nCleaning events file: {events_file.relative_to(CLEAN_ROOT)}", log_file)

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

    # Correct Unity/LSL visual onset markers to estimated physical screen onset
    # Based on photodiode timing test: physical onset occurs ~30 ms after marker
    clean_events["onset"] = clean_events["onset"].astype(float) + VISUAL_ONSET_OFFSET_S

    if "sample" in clean_events.columns:
        clean_events["sample"] = (
            clean_events["sample"].astype(float) + VISUAL_ONSET_OFFSET_S * EEG_SFREQ
        ).round().astype(int)

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
        "TRIAL_INFO|OPTIC_FLOW|COMPARE|TRIAL_START|TRIAL_END|PREVIEW|photo_ON|photo_OFF|stimOffset|BAD_NAN",
        regex=True,
    )

    if bad_mask.any():
        bad_examples = check.loc[bad_mask, "trial_type"].head(10).tolist()
        raise RuntimeError(
            f"Event cleaning failed for {events_file}. "
            f"Still found verbose rows: {bad_examples}"
        )

    log(f"  Kept {len(clean_events)} clean stimulus-onset events", log_file)
    log(clean_events["trial_type"].value_counts().sort_index().to_string(), log_file)


def clean_all_events_files(subject=None):
    if subject is None:
        events_files = sorted(CLEAN_ROOT.glob("sub-*/ses-*/eeg/*_events.tsv"))
    else:
        events_files = sorted(CLEAN_ROOT.glob(f"sub-{subject}/ses-*/eeg/*_events.tsv"))

    if not events_files:
        print("No events.tsv files found.")
        return

    print(f"\nFound {len(events_files)} events.tsv files")

    for events_file in events_files:
        clean_events_file(events_file)


def log(msg, log_file=None):
    print(msg)

    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sub",
        type=str,
        default=None,
        help="Subject to process, e.g. 15 or 015",
    )

    args = parser.parse_args()
    subject = normalize_subject(args.sub)

    create_clean_dataset_copy(subject)
    clean_all_set_files(subject)
    clean_all_events_files(subject)

    print("\nDone.")

    if subject:
        print(f"Processed sub-{subject}")
    else:
        print(f"Processed all subjects")

    print(f"Cleaned BIDS dataset written to: {CLEAN_ROOT}")


if __name__ == "__main__":
    main()
