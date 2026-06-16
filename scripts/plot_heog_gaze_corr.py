import numpy as np
import pyxdf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

XDF_PATH = "data/sub-001/ses-001/eeg/sub-001_ses-001_task-Default_run-001_eeg_raw.xdf"

streams, _ = pyxdf.load_xdf(XDF_PATH, dejitter_timestamps=False)

gaze = streams[3]
eeg = streams[4]

gaze_x = np.asarray(gaze["time_series"])[:, 0]
gaze_t = np.asarray(gaze["time_stamps"])

eeg_data = np.asarray(eeg["time_series"])
eeg_t = np.asarray(eeg["time_stamps"])

heogr = eeg_data[:, 31]
heogl = eeg_data[:, 53]
heog = heogl - heogr 

# Time window
t_min = max(gaze_t[0], eeg_t[0])
t_max = min(gaze_t[-1], eeg_t[-1])

fs = 1 / np.median(np.diff(gaze_t))
print(f"Using resampling fs = {fs:.2f} Hz")
t = np.arange(t_min, t_max, 1 / fs)

gaze_interp = interp1d(gaze_t, gaze_x, bounds_error=False, fill_value="extrapolate")
heog_interp = interp1d(eeg_t, heog, bounds_error=False, fill_value="extrapolate")

gaze_rs = gaze_interp(t)
heog_rs = heog_interp(t)

# Remove slow drift
gaze_rs = gaze_rs - np.nanmean(gaze_rs)
heog_rs = heog_rs - np.nanmean(heog_rs)

# Bandpass filter around typical eye movement frequencies
b, a = butter(2, [0.1, 10], btype="bandpass", fs=fs)
gaze_f = filtfilt(b, a, gaze_rs)
heog_f = filtfilt(b, a, heog_rs)

# Normalize
gaze_f = (gaze_f - np.mean(gaze_f)) / np.std(gaze_f)
heog_f = (heog_f - np.mean(heog_f)) / np.std(heog_f)

max_lag_sec = 1.0
max_lag = int(max_lag_sec * fs)
lags = np.arange(-max_lag, max_lag + 1)
corrs = []

for lag in lags:
    if lag < 0:
        c = np.corrcoef(heog_f[:lag], gaze_f[-lag:])[0, 1]
    elif lag > 0:
        c = np.corrcoef(heog_f[lag:], gaze_f[:-lag])[0, 1]
    else:
        c = np.corrcoef(heog_f, gaze_f)[0, 1]
    corrs.append(c)

corrs = np.asarray(corrs)
lag_ms = lags / fs * 1000

best_idx = np.nanargmax(np.abs(corrs))
print(f"Best lag: {lag_ms[best_idx]:.1f} ms")
print(f"Correlation at best lag: {corrs[best_idx]:.3f}")

plt.figure(figsize=(8, 4))
plt.plot(lag_ms, corrs)
plt.axvline(0, linestyle="--")
plt.axvline(lag_ms[best_idx], linestyle=":")
plt.xlabel("Lag (ms)")
plt.ylabel("Correlation: HEOG vs gaze X")
plt.title("Cross-correlation between HEOG and Pupil Labs gaze X")
plt.tight_layout()
plt.show()
