import mne

mne.viz.set_browser_backend("qt")

raw = mne.io.read_raw_eeglab(
    "data/sub-001/ses-001/eeg/sub-001_ses-001_task-Default_run-1_eeg.set",
    preload=True
)

raw.set_channel_types({
    "x": "misc",
    "y": "misc",
    "HEOGR": "eog",
    "HEOGL": "eog",
    "VEOGU": "eog",
    "VEOGL": "eog",
})

raw.pick(["x", "HEOGR", "HEOGL", "VEOGU"])

# Remove annotations for faster interactive viewing
raw.set_annotations(None)

raw.crop(tmax=60)

raw.plot(
    duration=20,
    n_channels=4,
    scalings={
        "misc": 5,
        "eog": 300e-6,
    },
    block=True
)
