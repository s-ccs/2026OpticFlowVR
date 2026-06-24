from pathlib import Path
import numpy as np
import pandas as pd
import mne
import pyxdf
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

XDF_PATH = Path(
    "data/sub-023/ses-001/eeg/sub-023_ses-001_task-EEGnoVR_run-001_eeg.xdf"
)

OUT_DIR = Path("output/sanity_checks/psd_peaks/eeg_no_vr_first_5min")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEGMENT_START_S = 0
SEGMENT_END_S = 5 * 60  # duration is 5 minutes

streams, _ = pyxdf.load_xdf(str(XDF_PATH))

eeg_stream = None
for stream in streams:
    name = stream["info"]["name"][0].lower()
    stype = stream["info"]["type"][0].lower()
    sfreq = float(stream["info"]["nominal_srate"][0])

    print(name, stype, sfreq)

    if name == "eegosport" and stype == "eeg" and sfreq > 0:
        eeg_stream = stream
        break

if eeg_stream is None:
    raise RuntimeError("No eegosport EEG stream found.")

data = np.asarray(eeg_stream["time_series"]).T
data = data * 1e-6 
times = np.asarray(eeg_stream["time_stamps"])
sfreq = float(eeg_stream["info"]["nominal_srate"][0])

try:
    channels = eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
    ch_names = [ch["label"][0] for ch in channels]
except Exception:
    ch_names = [f"EEG{i+1}" for i in range(data.shape[0])]

if len(ch_names) != data.shape[0]:
    ch_names = [f"EEG{i+1}" for i in range(data.shape[0])]

# Select first 5 minutes relative to XDF EEG stream start
t_rel = times - times[0]
mask = (t_rel >= SEGMENT_START_S) & (t_rel < SEGMENT_END_S)

data = data[:, mask]

info = mne.create_info(
    ch_names=ch_names,
    sfreq=sfreq,
    ch_types=["eeg"] * len(ch_names),
)

raw = mne.io.RawArray(data, info)

drop_channels = ["sampleNumber", "HEOGR", "HEOGL", "VEOGU", "VEOGL"]
drop_channels = [ch for ch in drop_channels if ch in raw.ch_names]
raw.drop_channels(drop_channels)

raw.pick("eeg")
raw.set_montage("standard_1005", on_missing="ignore")

# Match pipeline sampling rate
raw.resample(250)

# Demean to avoid DC/drift leakage into Welch PSD
raw.apply_function(
    lambda x: x - np.mean(x),
    picks="eeg",
    channel_wise=True,
)

psd = raw.compute_psd(
    method="welch",
    fmin=1,
    fmax=100,
    n_fft=4096,
    n_overlap=2048,
    reject_by_annotation=False,
)

freqs = psd.freqs
data_psd = psd.get_data()  # channels x freqs
data_db = 10 * np.log10(data_psd)
mean_psd_db = data_db.mean(axis=0)

# Smooth background / 1/f trend
baseline = (
    pd.Series(mean_psd_db)
    .rolling(window=101, center=True, min_periods=1)
    .median()
    .to_numpy()
)

residual = mean_psd_db - baseline

target_freqs = np.arange(4.8, 100, 4.8)

fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(freqs, residual, color="black", linewidth=1)

for f in target_freqs:
    ax.axvline(f, color="red", alpha=0.25, linewidth=0.8)

ax.axhline(0, color="gray", linewidth=1)
ax.set_xlim(1, 100)
ax.set_xticks(range(0, 101, 5))
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Residual PSD (dB)")
ax.set_title("PSD residual after removing smooth 1/f background")
ax.grid(True, alpha=0.3)

plt.show()

# --------------------------------------------------
# Plot PSD
# --------------------------------------------------

fig = psd.plot(
    picks="eeg",
    average=False,
    amplitude=False,
    show=False,
)

ax = fig.axes[0]
ax.set_title("First 5 min: EEG no VR")
ax.set_xticks(range(0, 101, 5))
ax.set_xlim(0, 100)
ax.grid(True, alpha=0.3)

fig.savefig(
    OUT_DIR / "first_5min_eeg_no_vr_psd.png",
    dpi=300,
    bbox_inches="tight",
)

# --------------------------------------------------
# Automatic peak detection
# --------------------------------------------------

baseline = (
    pd.Series(mean_psd_db)
    .rolling(window=101, center=True, min_periods=1)
    .median()
    .to_numpy()
)

residual = mean_psd_db - baseline

peaks, props = find_peaks(
    residual,
    prominence=1.5,
    distance=30,
)

peak_table = pd.DataFrame({
    "frequency_hz": freqs[peaks],
    "prominence_db": props["prominences"],
    "psd_db": mean_psd_db[peaks],
})

peak_table = peak_table[
    (peak_table["frequency_hz"] >= 1)
    & (peak_table["frequency_hz"] <= 100)
].sort_values("frequency_hz")

peak_table.to_csv(
    OUT_DIR / "first_5min_eeg_no_vr_detected_peaks.csv",
    index=False,
)

print("\nDetected peaks:")
print(peak_table.to_string(index=False))

# --------------------------------------------------
# Explicit 4.8 Hz harmonic check
# --------------------------------------------------

target_freqs = np.arange(4.8, 100, 4.8)

rows = []

for target in target_freqs:
    idx = np.argmin(np.abs(freqs - target))

    local_mask = (
        (freqs >= target - 1.0)
        & (freqs <= target + 1.0)
        & ~((freqs >= target - 0.2) & (freqs <= target + 0.2))
    )

    local_mean = mean_psd_db[local_mask].mean()
    peak_height = mean_psd_db[idx] - local_mean

    rows.append({
        "target_hz": target,
        "nearest_bin_hz": freqs[idx],
        "psd_db": mean_psd_db[idx],
        "local_mean_db": local_mean,
        "peak_height_db": peak_height,
    })

harmonics = pd.DataFrame(rows)

harmonics.to_csv(
    OUT_DIR / "first_5min_eeg_no_vr_4p8hz_harmonic_check.csv",
    index=False,
)

print("\n4.8 Hz harmonic check:")
print(harmonics.to_string(index=False))

plt.show()