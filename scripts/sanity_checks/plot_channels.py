import mne
import matplotlib.pyplot as plt
import numpy as np

raw = mne.io.read_raw_eeglab(
    "data/sub-001/ses-001/eeg/sub-001_ses-001_task-Default_run-1_eeg.set",
    preload=True
)

channels_to_plot = ["x", "HEOGR", "HEOGL", "VEOGU"]

data = raw.copy().pick(channels_to_plot).get_data()

plt.figure(figsize=(14, 6))

for ch_idx, ch in enumerate(channels_to_plot):
    sig = data[ch_idx][:10000]
    sig = (sig - np.nanmean(sig)) / np.nanstd(sig)
    plt.plot(sig + ch_idx * 5, label=ch)

plt.legend()
plt.title("Normalized gaze and EOG channels")
plt.xlabel("Samples")
plt.ylabel("Normalized amplitude + offset")
plt.tight_layout()
plt.show()