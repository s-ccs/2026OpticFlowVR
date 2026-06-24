from pathlib import Path
import numpy as np
import pandas as pd
import mne
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DERIV_ROOT = (
    PROJECT_ROOT
    / "output"
    / "derivatives"
    / "mne-bids-pipeline"
)

SUBJECTS = [
    "002", "003", "004", "005", "006", "007",
    "008", "009", "010", "011", "012", "013", "014", "015",
]

OUT_DIR = PROJECT_ROOT / "output" / "sanity_checks" / "psd_peaks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FMIN = 1
FMAX = 100

# Ignore expected notch edge areas
IGNORE_AROUND = [50]     # Hz
IGNORE_WIDTH = 1.0       # +/- 1 Hz

all_rows = []

for subject in SUBJECTS:
    sub = f"sub-{subject}"

    raw_path = (
        DERIV_ROOT
        / sub
        / "ses-001"
        / "eeg"
        / f"{sub}_ses-001_task-compareSpeed_run-1_proc-clean_raw.fif"
    )

    if not raw_path.exists():
        raw_path = (
            DERIV_ROOT
            / sub
            / "ses-001"
            / "eeg"
            / f"{sub}_ses-001_task-compareSpeed_run-1_proc-filtered_raw.fif"
        )

    if not raw_path.exists():
        print(f"Missing raw file for {sub}")
        continue

    print(f"Loading {sub}: {raw_path}")
    raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)
    raw.pick("eeg")

    psd = raw.compute_psd(
        method="welch",
        fmin=FMIN,
        fmax=FMAX,
        n_fft=4096,
        n_overlap=2048,
        average="mean",
        verbose=False,
    )

    data = psd.get_data()  # shape: channels x frequencies
    freqs = psd.freqs
    ch_names = psd.ch_names

    # Convert to dB
    data_db = 10 * np.log10(data)

    for ch_idx, ch_name in enumerate(ch_names):
        y = data_db[ch_idx]

        # Remove slow 1/f trend using rolling median
        baseline = pd.Series(y).rolling(window=21, center=True, min_periods=1).median()
        residual = y - baseline.to_numpy()

        peaks, props = find_peaks(
            residual,
            prominence=2.0,
            distance=5,
        )

        for peak_idx in peaks:
            peak_freq = freqs[peak_idx]

            if any(abs(peak_freq - bad) <= IGNORE_WIDTH for bad in IGNORE_AROUND):
                continue

            all_rows.append({
                "subject": sub,
                "channel": ch_name,
                "frequency_hz": round(float(peak_freq), 2),
                "prominence_db": round(float(props["prominences"][list(peaks).index(peak_idx)]), 2),
                "power_db": round(float(y[peak_idx]), 2),
            })

df = pd.DataFrame(all_rows)
df.to_csv(OUT_DIR / "psd_peaks_by_subject_channel.csv", index=False)

summary = (
    df.assign(freq_rounded=df["frequency_hz"].round())
      .groupby(["freq_rounded", "channel"])
      .agg(
          n_subjects=("subject", "nunique"),
          subjects=("subject", lambda x: ", ".join(sorted(set(x)))),
          mean_prominence_db=("prominence_db", "mean"),
      )
      .reset_index()
      .sort_values(["freq_rounded", "n_subjects"], ascending=[True, False])
)

summary.to_csv(OUT_DIR / "psd_peaks_summary.csv", index=False)

print("\nSaved:")
print(OUT_DIR / "psd_peaks_by_subject_channel.csv")
print(OUT_DIR / "psd_peaks_summary.csv")

print("\nMost common peaks:")
print(summary[summary["n_subjects"] >= 5].to_string(index=False))
