from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import mne

from config import (
    SUBJECTS,
    OUT_ROOT,
    POSTERIOR_ROI,
    FREQS,
    N_CYCLES,
    TFR_DECIM,
    BASELINE,
    BASELINE_MODE,
    ANALYSIS_TMIN,
    ANALYSIS_TMAX,
)
from common import normalize_subject


def process_subject(subject: str, overwrite: bool = False) -> Path:
    subject = normalize_subject(subject)
    sub_dir = OUT_ROOT / f"sub-{subject}"
    epochs_file = sub_dir / f"sub-{subject}_tfa-long-epo.fif"
    out_file = sub_dir / f"sub-{subject}_posterior-single-trial-tfr.npz"
    metadata_file = sub_dir / f"sub-{subject}_posterior-single-trial-metadata.csv"

    if out_file.exists() and metadata_file.exists() and not overwrite:
        print(f"Exists, skipping sub-{subject}")
        return out_file
    if not epochs_file.exists():
        raise FileNotFoundError(
            f"Missing long epochs for sub-{subject}: {epochs_file}"
        )

    epochs = mne.read_epochs(epochs_file, preload=True, verbose="ERROR")
    available_roi = [ch for ch in POSTERIOR_ROI if ch in epochs.ch_names]
    missing_roi = sorted(set(POSTERIOR_ROI) - set(available_roi))
    if not available_roi:
        raise RuntimeError(f"sub-{subject}: no posterior ROI channels available")
    if missing_roi:
        print(f"sub-{subject}: missing ROI channels: {missing_roi}")

    roi_epochs = epochs.copy().pick(available_roi)

    print(
        f"\nsub-{subject}: computing Morlet power\n"
        f"  epochs={len(roi_epochs)}, channels={len(available_roi)}, "
        f"freqs={len(FREQS)}, sfreq={roi_epochs.info['sfreq']}"
    )

    power = roi_epochs.compute_tfr(
        method="morlet",
        freqs=FREQS,
        n_cycles=N_CYCLES,
        output="power",
        average=False,
        return_itc=False,
        decim=TFR_DECIM,
        n_jobs=-1,
        use_fft=True,
        zero_mean=True,
        verbose=True,
    )

    # Baseline each trial, channel, and frequency independently.
    power.apply_baseline(BASELINE, mode=BASELINE_MODE)
    power.crop(tmin=ANALYSIS_TMIN, tmax=ANALYSIS_TMAX)

    # Crucial order: calculate channel-level power first, then average ROI power.
    # Shape before mean: epochs × channels × frequencies × times.
    roi_power = power.get_data().mean(axis=1).astype(np.float32)

    metadata = power.metadata.copy().reset_index(drop=True)
    metadata.to_csv(metadata_file, index=False)

    np.savez_compressed(
        out_file,
        power=roi_power,
        freqs=power.freqs.astype(np.float32),
        times=power.times.astype(np.float32),
        roi_channels=np.asarray(available_roi),
        subject=np.asarray(subject),
        baseline=np.asarray(BASELINE, dtype=np.float32),
        baseline_mode=np.asarray(BASELINE_MODE),
        sfreq=np.asarray(power.info["sfreq"], dtype=np.float32),
    )

    print(
        f"Saved sub-{subject}: {roi_power.shape} "
        "(trials × frequencies × times)\n"
        f"  {out_file}\n  {metadata_file}"
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

    print("\nSubject-level TFR computation complete.")


if __name__ == "__main__":
    main()
