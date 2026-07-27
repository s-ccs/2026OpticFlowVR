from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import mne

from config import (
    CLEAN_BIDS_ROOT,
    DERIV_ROOT,
    SESSION,
    TASK,
    RUN,
)

def normalize_subject(subject: str) -> str:
    return str(subject).replace("sub-", "").zfill(3)

def eeg_dir(subject: str) -> Path:
    subject = normalize_subject(subject)
    return DERIV_ROOT / f"sub-{subject}" / f"ses-{SESSION}" / "eeg"

def events_file(subject: str) -> Path:
    subject = normalize_subject(subject)
    path = (
        CLEAN_BIDS_ROOT
        / f"sub-{subject}"
        / f"ses-{SESSION}"
        / "eeg"
        / f"sub-{subject}_ses-{SESSION}_task-{TASK}_run-{RUN}_events.tsv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing cleaned events file: {path}")
    return path

def fixation_epochs_file(subject: str) -> Path:
    subject = normalize_subject(subject)
    path = (
        eeg_dir(subject)
        / f"sub-{subject}_ses-{SESSION}_task-{TASK}_proc-clean_fixation-epo.fif"
    )
    if not path.exists():
        raise FileNotFoundError(
            "Missing fixation-cleaned epochs. Run fixation_check.py first:\n"
            f"{path}"
        )
    return path

def find_clean_raw(subject: str) -> Path:
    """Find the ICA-cleaned continuous FIF produced by MNE-BIDS-Pipeline"""
    subject = normalize_subject(subject)
    directory = eeg_dir(subject)

    exact_candidates = [
        directory / (
            f"sub-{subject}_ses-{SESSION}_task-{TASK}_run-{RUN}"
            "_proc-clean_raw.fif"
        ),
        directory / (
            f"sub-{subject}_ses-{SESSION}_task-{TASK}"
            "_proc-clean_raw.fif"
        ),
    ]
    for path in exact_candidates:
        if path.exists():
            return path

    matches = sorted(directory.glob("*proc-clean*raw.fif"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        available = "\n".join(f"  {p.name}" for p in sorted(directory.glob("*.fif")))
        raise FileNotFoundError(
            "Could not find an ICA-cleaned continuous raw FIF file in:\n"
            f"  {directory}\n"
            "Expected a file containing 'proc-clean' and ending '_raw.fif'.\n"
            f"Available FIF files:\n{available or '  <none>'}\n\n"
        )
    raise RuntimeError(
        "Found multiple possible cleaned raw files;\n"
        + "\n".join(f"  {p}" for p in matches)
    )

def read_clean_events(subject: str) -> pd.DataFrame:
    events = pd.read_csv(events_file(subject), sep="\t")
    required = {"onset", "trial_type", "condition", "speed_idx"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Events table is missing columns: {sorted(missing)}")

    events = events.reset_index(drop=True)
    events["source_event_index"] = np.arange(len(events), dtype=int)
    events["onset"] = events["onset"].astype(float)
    events["speed_idx"] = events["speed_idx"].astype(int)
    events["speed_m_s"] = 0.8 + 0.2 * events["speed_idx"]
    # Collapse directions for Rotation and Spiral
    events["condition_family"] = events["condition"].astype(str)
    events.loc[events["condition_family"] == "Rotation", "condition_family"] = "Rotation"
    events.loc[events["condition_family"] == "Spiral", "condition_family"] = "Spiral"

    return events

def retained_source_indices(subject: str) -> np.ndarray:
    """Original event-row indices retained by ERP + fixation rejection"""
    epochs = mne.read_epochs(fixation_epochs_file(subject), preload=False, verbose="ERROR")
    selection = np.asarray(epochs.selection, dtype=int)

    if len(selection) != len(epochs):
        raise RuntimeError(
            f"Unexpected selection length for sub-{subject}: "
            f"{len(selection)} selection rows versus {len(epochs)} epochs."
        )
    return selection

def make_mne_events(raw: mne.io.BaseRaw, onsets: np.ndarray) -> np.ndarray:
    # BIDS event onset is relative to raw start; MNE event samples include first_samp
    samples = raw.time_as_index(onsets, use_rounding=True) + raw.first_samp
    event_codes = np.arange(1, len(samples) + 1, dtype=int)
    return np.column_stack(
        [
            samples.astype(int),
            np.zeros(len(samples), dtype=int),
            event_codes,
        ]
    )

def event_id_from_rows(events: pd.DataFrame) -> dict[str, int]:
    return {
        str(label): idx + 1
        for idx, label in enumerate(events["trial_type"].astype(str))
    }
