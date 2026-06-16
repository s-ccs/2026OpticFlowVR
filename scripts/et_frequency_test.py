import numpy as np
import pyxdf
import matplotlib.pyplot as plt
from pathlib import Path

XDF_PATH = Path("../../data/sub-001/ses-001/eeg/sub-001_ses-001_task-ETTest_run-002_eeg.xdf")

streams, _ = pyxdf.load_xdf(XDF_PATH)

for i, s in enumerate(streams):
    name = s["info"]["name"][0]
    stype = s["info"]["type"][0]
    nominal = s["info"]["nominal_srate"][0]
    print(f"{i}: {name} | type={stype} | nominal={nominal}")

GAZE_STREAM_INDEX = 2

gaze = streams[GAZE_STREAM_INDEX]
t = np.asarray(gaze["time_stamps"])

name = gaze["info"]["name"][0]
nominal = float(gaze["info"]["nominal_srate"][0])

t_rel = t - t[0]
dt = np.diff(t)

duration = t[-1] - t[0]
eff_mean = len(t) / duration
eff_median = 1 / np.median(dt)

print("\nSelected stream:", name)
print("Samples:", len(t))
print("Duration seconds:", duration)
print("Effective Hz from duration:", eff_mean)
print("Effective Hz from median dt:", eff_median)
print("Median dt ms:", np.median(dt) * 1000)

print("\ndt percentiles in ms:")
for p in [50, 75, 90, 95, 99, 99.9]:
    print(p, np.percentile(dt, p) * 1000)

print("Gaps >20 ms:", np.sum(dt > 0.020))
print("Gaps >50 ms:", np.sum(dt > 0.050))

# Samples per second
bins = np.arange(0, t_rel[-1] + 1, 1)
counts, edges = np.histogram(t_rel, bins=bins)

plt.figure(figsize=(10, 4))
plt.plot(edges[:-1] / 60, counts, linewidth=0.8)

plt.axhline(nominal, linestyle="--", label=f"{nominal:.0f} Hz nominal")
plt.xlabel("Time in recording (min)")
plt.ylabel("Samples per second")
plt.title(f"XDF stream sampling rate over time: {name}")
plt.legend()
plt.tight_layout()
plt.show()
