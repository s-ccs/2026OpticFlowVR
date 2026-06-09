import mne

raw = mne.io.read_raw_eeglab(
    "data/sub-001/ses-001/eeg/sub-001_ses-001_task-Default_run-1_eeg.set",
    preload=True
)

print(raw.ch_names) 