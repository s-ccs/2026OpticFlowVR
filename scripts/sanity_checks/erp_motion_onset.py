from pathlib import Path
import mne
import matplotlib.pyplot as plt

SUBJECT = "sub-001"
SESSION = "ses-001"

eeg_path = f"data/{SUBJECT}/{SESSION}/eeg/{SUBJECT}_{SESSION}_task-Default_run-1_eeg.set"
out_dir = Path("output") / SUBJECT
out_dir.mkdir(parents=True, exist_ok=True)

raw = mne.io.read_raw_eeglab(eeg_path, preload=True)

# Mark non-EEG channels
eye_misc_channels = [
    ch for ch in raw.ch_names
    if (
        ch in ["x", "y", "sampleNumber"]
        or "PupilDiameter" in ch
        or "EyeballCenter" in ch
        or "OpticalAxis" in ch
        or "Eyelid" in ch
    )
]

raw.set_channel_types({ch: "misc" for ch in eye_misc_channels})

raw.set_channel_types({
    "HEOGR": "eog",
    "HEOGL": "eog",
    "VEOGU": "eog",
    "VEOGL": "eog",
})

montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage, on_missing="ignore")

raw.info["bads"] = ["AF7"]

raw.notch_filter(50)
raw.filter(0.1, 30)

events, event_id = mne.events_from_annotations(raw)

stim_onset_event_ids = {
    name: code
    for name, code in event_id.items()
    if name.startswith("4-stimOnset")
}

print(f"Found {len(stim_onset_event_ids)} stimOnset event types")

raw_eeg = raw.copy().pick("eeg")

epochs = mne.Epochs(
    raw_eeg,
    events,
    event_id=stim_onset_event_ids,
    tmin=-0.2,
    tmax=0.8,
    baseline=(-0.2, 0),
    reject=None,
    preload=True,
    reject_by_annotation=True,
)

print(epochs)

evoked = epochs.average()

posterior_channels = ["O1", "O2", "POz", "PO3", "PO4", "PO7", "PO8"]

evoked_posterior = evoked.copy().pick(posterior_channels)

fig = evoked_posterior.plot(
    spatial_colors=False,
    show=False,
    titles=f"{SUBJECT} motion-onset VEP"
)

fig.savefig(out_dir / f"{SUBJECT}_{SESSION}_motion-onset-vep.png", dpi=300)

topomap_fig = evoked.plot_topomap(
    times=[0.10, 0.15, 0.18, 0.20, 0.25, 0.30],
    ch_type="eeg",
    show=False,
    time_unit="s",
)

topomap_fig.savefig(out_dir / f"{SUBJECT}_{SESSION}_motion-onset-topomap.png", dpi=300)

evoked.save(out_dir / f"{SUBJECT}_{SESSION}_motion-onset-ave.fif", overwrite=True)

plt.show()