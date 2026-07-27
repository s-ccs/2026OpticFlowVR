from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import mne

from config import (SUBJECTS, OUT_ROOT, EPOCH_TMIN, EPOCH_TMAX, REJECT)
from common import (normalize_subject, find_clean_raw, read_clean_events, retained_source_indices, make_mne_events)

def process_subject(subject: str, overwrite: bool = False) -> Path:
    subject = normalize_subject(subject)
    out_dir = OUT_ROOT / f"sub-{subject}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"sub-{subject}_tfa-long-epo.fif"

    if out_file.exists() and not overwrite:
        print(f"Exists, skipping sub-{subject}: {out_file}")
        return out_file

    raw_file = find_clean_raw(subject)
    print(f"\nsub-{subject}: reading cleaned raw\n  {raw_file}")
    raw = mne.io.read_raw_fif(raw_file, preload=False, verbose="ERROR")
    raw.pick("eeg", exclude="bads") # Remove non-EEG channels

    all_events = read_clean_events(subject)
    keep = retained_source_indices(subject)

    if keep.size == 0:
        raise RuntimeError(f"No fixation-cleaned trials remain for sub-{subject}")
    if keep.max() >= len(all_events):
        raise RuntimeError(
            f"sub-{subject}: epochs.selection reaches {keep.max()}, "
            f"but events.tsv has {len(all_events)} rows."
        )

    metadata = all_events.iloc[keep].copy().reset_index(drop=True)
    mne_events = make_mne_events(raw, metadata["onset"].to_numpy())

    # Each retained row gets a unique code
    event_id = {
        f"trial-{row.source_event_index:03d}": idx + 1
        for idx, row in metadata.iterrows()
    }
    epochs = mne.Epochs(
        raw,
        events=mne_events,
        event_id=event_id,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=None,  # do not voltage-baseline before TFR
        picks="eeg",
        preload=True,
        reject=REJECT,
        reject_by_annotation=True,
        metadata=metadata,
        event_repeated="error",
        verbose=True,
    )

    # TFR baseline normalization is performed later in the power domain
    epochs.save(out_file, overwrite=True)
    retained_after_long_epoch_reject = set(
        epochs.metadata["source_event_index"].astype(int)
    )
    dropped_long = sorted(set(keep.tolist()) - retained_after_long_epoch_reject)

    print(
        f"sub-{subject}: fixation-retained={len(keep)}, "
        f"long-epoch retained={len(epochs)}, "
        f"additional long-window drops={len(dropped_long)}"
    )
    if dropped_long:
        np.savetxt(
            out_dir / f"sub-{subject}_additional-long-epoch-drops.txt",
            np.asarray(dropped_long, dtype=int),
            fmt="%d",
        )
    return out_file

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    subjects = (
        [normalize_subject(args.sub)]
        if args.sub is not None
        else SUBJECTS
    )
    for subject in subjects:
        try:
            process_subject(subject, overwrite=args.overwrite)
        except FileNotFoundError as exc:
            print(f"Skipping sub-{subject}: {exc}")

    print("\nLong-epoch creation complete.")

if __name__ == "__main__":
    main()
