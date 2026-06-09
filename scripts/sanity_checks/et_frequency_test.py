import numpy as np
import pyxdf

XDF_PATH = "data/sub-001/ses-001/eeg/sub-001_ses-001_task-ETTest_run-001_eeg.xdf"

streams, header = pyxdf.load_xdf(XDF_PATH)

for i, s in enumerate(streams):
    name = s["info"]["name"][0]
    stype = s["info"]["type"][0]
    nominal = s["info"]["nominal_srate"][0]
    t = np.asarray(s["time_stamps"])

    if len(t) > 1:
        duration = t[-1] - t[0]
        eff_mean = len(t) / duration
        eff_median = 1 / np.median(np.diff(t))
        med_dt_ms = np.median(np.diff(t)) * 1000
    else:
        eff_mean = np.nan
        eff_median = np.nan
        med_dt_ms = np.nan

    print(f"\nStream {i}")
    print(f"Name: {name}")
    print(f"Type: {stype}")
    print(f"Nominal srate: {nominal}")
    print(f"Samples: {len(t)}")
    print(f"Effective Hz from duration: {eff_mean:.3f}")
    print(f"Effective Hz from median dt: {eff_median:.3f}")
    print(f"Median dt: {med_dt_ms:.3f} ms")
