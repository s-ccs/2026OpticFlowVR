from pathlib import Path

import mne
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

eeg_dir = (PROJECT_ROOT
    / "output"
    / "derivatives"
    / "mne-bids-pipeline"
    / "sub-024"
    / "ses-001"
    / "eeg"
)

files = [
    eeg_dir / "sub-024_ses-001_task-compareSpeed_proc-clean_epo.fif",
    eeg_dir / "sub-024_ses-001_task-compareSpeed_proc-clean_fixation-epo.fif",
]

for epochs_file in files:
    if not epochs_file.exists():
        continue

    epochs = mne.read_epochs(epochs_file, preload=True, verbose="ERROR")
    data = epochs.copy().pick("eeg").get_data()
    max_mean_uv = abs(data.mean(axis=1)).max() * 1e6

    print(f"{epochs_file.name}")
    print(f"  Maximum channel mean: {max_mean_uv} µV")
    print(f"  Maximum channel mean: {max_mean_uv:.16e} µV")