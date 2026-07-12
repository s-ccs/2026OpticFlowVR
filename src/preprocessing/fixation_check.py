import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_BIDS_ROOT = PROJECT_ROOT / "data"
CLEAN_BIDS_ROOT = PROJECT_ROOT / "data_clean"

DERIV_ROOT = (
    PROJECT_ROOT
    / "output"
    / "derivatives"
    / "mne-bids-pipeline"
)

OUT_ROOT = PROJECT_ROOT / "output" / "fixation"

SESSION = "001"
TASK = "compareSpeed"
RUN = "1"

STIM_START_S = 0.0
STIM_END_S = 0.8

BASELINE_START_S = -0.2
BASELINE_END_S = 0.0

SCENE_WIDTH_PX = 1600
SCENE_HEIGHT_PX = 1200
SCENE_HFOV_DEG = 103
SCENE_VFOV_DEG = 77

DEFAULT_THRESHOLD_DEG = 10.0
DEFAULT_MIN_PROP_INSIDE = 0.80
DEFAULT_MAX_PROP_MISSING = 0.30


def normalize_subject(subject):
    if subject is None:
        return None
    subject = str(subject).replace("sub-", "")
    return subject.zfill(3)


def find_subjects():
    subjects = sorted(
        p.name.replace("sub-", "")
        for p in DERIV_ROOT.glob("sub-*")
        if p.is_dir()
    )
    return subjects


def find_one(pattern, description):
    matches = sorted(pattern)
    if not matches:
        raise FileNotFoundError(f"Could not find {description}")
    if len(matches) > 1:
        print(f"Warning: found multiple files for {description}; using:")
        for match in matches:
            print(f"  {match}")
    return matches[0]


def guess_gaze_channels(raw, x_name=None, y_name=None):
    ch_names = raw.ch_names
    lower_to_original = {ch.lower(): ch for ch in ch_names}

    if x_name is not None and y_name is not None:
        if x_name not in ch_names:
            raise ValueError(f"Requested x channel not found: {x_name}")
        if y_name not in ch_names:
            raise ValueError(f"Requested y channel not found: {y_name}")
        return x_name, y_name

    exact_pairs = [
        ("x", "y"),
        ("gaze_x", "gaze_y"),
        ("gazex", "gazey"),
        ("norm_pos_x", "norm_pos_y"),
        ("gaze position x", "gaze position y"),
    ]

    for x_lower, y_lower in exact_pairs:
        if x_lower in lower_to_original and y_lower in lower_to_original:
            return lower_to_original[x_lower], lower_to_original[y_lower]

    # Fallback: look for channels containing gaze + x/y
    x_candidates = [
        ch for ch in ch_names
        if "gaze" in ch.lower() and (
            ch.lower().endswith("x")
            or "_x" in ch.lower()
            or " x" in ch.lower()
        )
    ]
    y_candidates = [
        ch for ch in ch_names
        if "gaze" in ch.lower() and (
            ch.lower().endswith("y")
            or "_y" in ch.lower()
            or " y" in ch.lower()
        )
    ]

    if x_candidates and y_candidates:
        return x_candidates[0], y_candidates[0]

    raise RuntimeError(
        "Could not automatically identify gaze x/y channels\n"
    )


def load_files(subject):
    sub = f"sub-{subject}"

    raw_file = find_one(
        RAW_BIDS_ROOT.glob(
            f"{sub}/ses-{SESSION}/eeg/"
            f"{sub}_ses-{SESSION}_task-{TASK}_run-{RUN}_eeg.set"
        ),
        f"original raw EEGLAB file for {sub}",
    )

    events_file = find_one(
        CLEAN_BIDS_ROOT.glob(
            f"{sub}/ses-{SESSION}/eeg/"
            f"{sub}_ses-{SESSION}_task-{TASK}_run-{RUN}_events.tsv"
        ),
        f"cleaned events.tsv for {sub}",
    )

    epochs_file = find_one(
        DERIV_ROOT.glob(
            f"{sub}/ses-{SESSION}/eeg/"
            f"{sub}_ses-{SESSION}_task-{TASK}_proc-clean_epo.fif"
        ),
        f"MNE-BIDS-Pipeline clean epochs for {sub}",
    )

    return raw_file, events_file, epochs_file


def get_gaze_segments(raw, onsets, x_channel, y_channel, tmin, tmax):
    """Return arrays shaped n_trials x n_times for x and y"""
    sfreq = raw.info["sfreq"]
    n_times = int(round((tmax - tmin) * sfreq)) + 1

    x_idx = raw.ch_names.index(x_channel)
    y_idx = raw.ch_names.index(y_channel)

    x_all = np.full((len(onsets), n_times), np.nan)
    y_all = np.full((len(onsets), n_times), np.nan)

    raw_n_times = raw.n_times

    for trial_idx, onset in enumerate(onsets):
        start = int(round((onset + tmin) * sfreq))
        stop = start + n_times

        src_start = max(start, 0)
        src_stop = min(stop, raw_n_times)

        if src_start >= src_stop:
            continue

        dst_start = src_start - start
        dst_stop = dst_start + (src_stop - src_start)

        data = raw.get_data(
            picks=[x_idx, y_idx],
            start=src_start,
            stop=src_stop,
        )

        x_all[trial_idx, dst_start:dst_stop] = data[0]
        y_all[trial_idx, dst_start:dst_stop] = data[1]

    times = np.arange(n_times) / sfreq + tmin
    return x_all, y_all, times

def gaze_px_to_deg(x, y, x0, y0):
    deg_per_px_x = SCENE_HFOV_DEG / SCENE_WIDTH_PX
    deg_per_px_y = SCENE_VFOV_DEG / SCENE_HEIGHT_PX

    dx_deg = (x - x0) * deg_per_px_x
    dy_deg = (y - y0) * deg_per_px_y

    dist_deg = np.sqrt(dx_deg ** 2 + dy_deg ** 2)

    return dx_deg, dy_deg, dist_deg

def compute_fixation_qc(
    x,
    y,
    times,
    threshold,
    min_prop_inside,
    max_prop_missing,
):
    baseline_mask = (times >= BASELINE_START_S) & (times < BASELINE_END_S)
    stim_mask = (times >= STIM_START_S) & (times <= STIM_END_S)

    if not baseline_mask.any():
        raise RuntimeError("No baseline samples available for fixation center")
    if not stim_mask.any():
        raise RuntimeError("No stimulus samples available for fixation check")

    x0 = np.nanmedian(x[:, baseline_mask], axis=1, keepdims=True)
    y0 = np.nanmedian(y[:, baseline_mask], axis=1, keepdims=True)

    x_stim = x[:, stim_mask]
    y_stim = y[:, stim_mask]

    dx_deg, dy_deg, dist = gaze_px_to_deg(
        x_stim,
        y_stim,
        x0,
        y0,
    )

    valid = np.isfinite(dist)
    prop_missing = 1.0 - np.mean(valid, axis=1)

    inside = dist <= threshold
    inside[~valid] = False
    prop_inside = np.mean(inside, axis=1)

    with np.errstate(all="ignore"):
        median_dist = np.nanmedian(dist, axis=1)
        p95_dist = np.nanpercentile(dist, 95, axis=1)
        max_dist = np.nanmax(dist, axis=1)

    bad_fixation = (
        (prop_inside < min_prop_inside)
        | (prop_missing > max_prop_missing)
        | ~np.isfinite(x0[:, 0])
        | ~np.isfinite(y0[:, 0])
    )

    qc = pd.DataFrame(
        {
            "trial_index": np.arange(len(prop_inside)),
            "fix_x": x0[:, 0],
            "fix_y": y0[:, 0],
            "prop_inside": prop_inside,
            "prop_missing": prop_missing,
            "median_dist_deg": median_dist,
            "p95_dist_deg": p95_dist,
            "max_dist_deg": max_dist,
            "bad_fixation": bad_fixation,
        }
    )

    return qc


def process_subject(
    subject,
    threshold,
    min_prop_inside,
    max_prop_missing,
    dry_run=False,
):
    sub = f"sub-{subject}"
    print("\n" + "=" * 80)
    print(f"Fixation check: {sub}")
    print("=" * 80)

    raw_file, events_file, epochs_file = load_files(subject)

    print(f"Reading raw gaze source:\n  {raw_file}")
    raw = mne.io.read_raw_eeglab(raw_file, preload=False, verbose="ERROR")

    x_channel, y_channel = guess_gaze_channels(raw)
    print(f"Using gaze channels: x={x_channel}, y={y_channel}")

    print(f"Reading events:\n  {events_file}")
    events = pd.read_csv(events_file, sep="\t")
    onsets = events["onset"].astype(float).to_numpy()

    print(f"Reading epochs:\n  {epochs_file}")
    epochs = mne.read_epochs(epochs_file, preload=True, verbose="ERROR")

    if len(events) != len(epochs):
        selection = np.asarray(epochs.selection, dtype=int)

        if len(selection) != len(epochs):
            raise RuntimeError(
                f"Mismatch between events.tsv rows ({len(events)}), "
                f"epochs ({len(epochs)}), and epochs.selection ({len(selection)})"
            )

        if selection.max() >= len(events):
            raise RuntimeError(
                f"epochs.selection refers to index {selection.max()}, "
                f"but events.tsv has only {len(events)} rows."
            )

        print(
            f"events.tsv has {len(events)} rows but epochs has {len(epochs)} rows\n"
            f"Using epochs.selection to keep the events retained by MNE"
        )

        events = events.iloc[selection].reset_index(drop=True)
        onsets = events["onset"].astype(float).to_numpy()

    x, y, times = get_gaze_segments(
        raw=raw,
        onsets=onsets,
        x_channel=x_channel,
        y_channel=y_channel,
        tmin=epochs.tmin,
        tmax=epochs.tmax,
    )

    qc = compute_fixation_qc(
        x=x,
        y=y,
        times=times,
        threshold=threshold,
        min_prop_inside=min_prop_inside,
        max_prop_missing=max_prop_missing,
    )

    # Add metadata/event columns to QC output
    qc = pd.concat(
        [
            qc,
            events.reset_index(drop=True).add_prefix("event_"),
        ],
        axis=1,
    )

    out_dir = OUT_ROOT / sub / "fixation"
    out_dir.mkdir(parents=True, exist_ok=True)

    qc_file = out_dir / (
        f"{sub}_ses-{SESSION}_task-{TASK}_run-{RUN}_fixation-qc.csv"
    )
    qc.to_csv(qc_file, index=False)

    n_bad = int(qc["bad_fixation"].sum())
    n_total = len(qc)
    print(f"Bad fixation trials: {n_bad}/{n_total} ({n_bad / n_total:.1%})")
    print(f"Saved QC table:\n  {qc_file}")
    summary_row = {
        "subject": sub,
        "trials": n_total,
        "rejected": n_bad,
        "percent_rejected": 100 * n_bad / n_total,
        "median_p95_deg": qc["p95_dist_deg"].median(),
    }

    if dry_run:
        print("Dry run only: epochs were not modified")
        return summary_row

    bad_mask = qc["bad_fixation"].to_numpy(dtype=bool)

    epochs_fix = epochs.copy()
    epochs_fix.drop(bad_mask, reason="BAD_fixation")

    out_epochs_file = epochs_file.with_name(
        epochs_file.name.replace("_proc-clean_epo.fif", "_proc-clean_fixation-epo.fif")
    )
    epochs_fix.save(out_epochs_file, overwrite=True)

    print(f"Saved fixation-cleaned epochs:\n  {out_epochs_file}")
    return summary_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sub",
        type=str,
        default=None,
        help="Subject to process, e.g. 15 or 015. If omitted, runs all derivative subjects",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write QC CSV; do not drop epochs or save fixation-cleaned epochs",
    )

    args = parser.parse_args()
    subject = normalize_subject(args.sub)

    if subject is None:
        subjects = find_subjects()
    else:
        subjects = [subject]

    if not subjects:
        raise RuntimeError(f"No subjects found in {DERIV_ROOT}")
    
    summary_rows = []

    for sub in subjects:
        summary_row = process_subject(
            subject=sub,
            threshold=DEFAULT_THRESHOLD_DEG,
            min_prop_inside=DEFAULT_MIN_PROP_INSIDE,
            max_prop_missing=DEFAULT_MAX_PROP_MISSING,
            dry_run=args.dry_run,
        )

        if summary_row is not None:
            summary_rows.append(summary_row)
    
    if summary_rows:
        summary = pd.DataFrame(summary_rows)

        summary_file = OUT_ROOT / "fixation_qc_summary.csv"
        summary.to_csv(summary_file, index=False)

        print(f"\nSaved fixation QC summary:\n  {summary_file}")

    print("\nDone.")


if __name__ == "__main__":
    main()
