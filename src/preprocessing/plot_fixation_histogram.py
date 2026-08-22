from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIX_ROOT = PROJECT_ROOT / "output-iclabel" / "fixation-iclabel"

all_data = []

for file in FIX_ROOT.glob(
    "sub-*/fixation/*_fixation-qc.csv"
):
    sub = file.parts[-3]
    df = pd.read_csv(file)
    df["subject"] = sub
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)
print(data["p95_dist_deg"].describe())

plt.figure(figsize=(8, 5))
plt.hist(
    data["p95_dist_deg"].dropna(),
    bins=100,
)
plt.axvline(
    2,
    linestyle="--",
    label="2 deg",
    c="orange"
)
plt.axvline(
    5,
    linestyle="--",
    label="5 deg",
    c="red"
)
plt.axvline(
    10,
    linestyle="--",
    label="10 deg",
    c="purple"
)
plt.xlabel("95th percentile gaze deviation (deg)")
plt.ylabel("Number of trials")
plt.title("Fixation deviation distribution")
plt.legend()

out = (
    PROJECT_ROOT
    / "output-iclabel"
    / "fixation-iclabel"
    / "fixation_p95_histogram.png"
)

plt.savefig(out, dpi=300)
plt.close()

print(f"Saved: {out}")
