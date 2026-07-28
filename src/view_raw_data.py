import mne
import os
from pathlib import Path

sample_data_folder = Path("./data/")
raw_file = sample_data_folder / "sub-040" / "ses-001" / "eeg" / \
    "sub-040_ses-001_task-compareSpeed_run-1_eeg.set"

annot_file = raw_file.with_name(
    "sub-040_ses-001_task-compareSpeed_run-1_manual-annot.fif"
)

raw = mne.io.read_raw_eeglab(raw_file, preload=True)

# Load previous annotations if they exist
if annot_file.exists():
    raw.set_annotations(mne.read_annotations(annot_file))
    print(f"Loaded annotations from {annot_file}")

raw.plot(block=True)

print(raw.annotations)

raw.annotations.save(annot_file, overwrite=True)
print(f"Saved annotations to {annot_file}")
