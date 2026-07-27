from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (SUBJECTS, OUT_ROOT, CONDITION_ORDER, CONTRASTS, BANDS, TIME_WINDOWS)
from common import normalize_subject

def load_subject(subject: str):
    subject = normalize_subject(subject)
    sub_dir = OUT_ROOT / f"sub-{subject}"
    tfr_file = sub_dir / f"sub-{subject}_posterior-single-trial-tfr.npz"
    metadata_file = sub_dir / f"sub-{subject}_posterior-single-trial-metadata.csv"

    if not tfr_file.exists() or not metadata_file.exists():
        return None

    z = np.load(tfr_file, allow_pickle=False)
    metadata = pd.read_csv(metadata_file)
    power = z["power"]

    if len(metadata) != power.shape[0]:
        raise RuntimeError(
            f"sub-{subject}: metadata rows={len(metadata)}, trials={power.shape[0]}"
        )
    return {
        "subject": subject,
        "power": power,
        "freqs": z["freqs"],
        "times": z["times"],
        "metadata": metadata,
    }

def condition_family(metadata: pd.DataFrame) -> pd.Series:
    if "condition_family" in metadata:
        return metadata["condition_family"].astype(str)
    return metadata["condition"].astype(str)

def subject_condition_means(item):
    labels = condition_family(item["metadata"])
    result = {}
    counts = {}
    for condition in CONDITION_ORDER:
        mask = labels.eq(condition).to_numpy()
        counts[condition] = int(mask.sum())
        if mask.any():
            result[condition] = item["power"][mask].mean(axis=0)
    return result, counts

def symmetric_limit(data_arrays, percentile=99.0):
    values = np.concatenate([np.ravel(x) for x in data_arrays])
    return float(np.nanpercentile(np.abs(values), percentile))

def plot_tfr(data, freqs, times, title, out_file, vlim=None):
    fig, ax = plt.subplots(figsize=(11, 6))
    if vlim is None:
        mesh = ax.pcolormesh(times, freqs, data, shading="auto")
    else:
        mesh = ax.pcolormesh(
            times, freqs, data, shading="auto", vmin=-vlim, vmax=vlim
        )
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.axvline(0.5, linestyle=":", linewidth=1)
    ax.axvline(2.0, linestyle=":", linewidth=1)
    ax.set_xlabel("Time from optic-flow onset (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Log power ratio to baseline")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

def extract_band_window_rows(subject, means, counts, freqs, times):
    rows = []
    for condition, matrix in means.items():
        for band, (fmin, fmax) in BANDS.items():
            fmask = (freqs >= fmin) & (freqs <= fmax)

            for window, (tmin, tmax) in TIME_WINDOWS.items():
                tmask = (times >= tmin) & (times <= tmax)
                values = matrix[np.ix_(fmask, tmask)]
                rows.append(
                    {
                        "subject": subject,
                        "condition": condition,
                        "band": band,
                        "window": window,
                        "fmin_hz": fmin,
                        "fmax_hz": fmax,
                        "tmin_s": tmin,
                        "tmax_s": tmax,
                        "n_trials": counts[condition],
                        "mean_logratio": float(
                            np.nanmean(values)
                        ),
                    }
                )
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", type=str, default=None)
    args = parser.parse_args()
    subjects = (
        [normalize_subject(args.sub)]
        if args.sub is not None
        else SUBJECTS
    )

    by_condition = defaultdict(list)
    included = []
    count_rows = []
    measure_rows = []
    freqs = times = None

    for subject in subjects:
        item = load_subject(subject)
        if item is None:
            print(f"Skipping sub-{subject}: subject TFR missing")
            continue

        if freqs is None:
            freqs = item["freqs"]
            times = item["times"]
        elif not (
            np.allclose(freqs, item["freqs"])
            and np.allclose(times, item["times"])
        ):
            raise RuntimeError(f"sub-{subject}: TFR grid differs from earlier subjects")

        means, counts = subject_condition_means(item)
        if not all(c in means for c in CONDITION_ORDER):
            print(f"Skipping sub-{subject}: not all condition families available")
            continue

        for condition, matrix in means.items():
            by_condition[condition].append(matrix)
        for condition, count in counts.items():
            count_rows.append(
                {"subject": subject, "condition": condition, "n_trials": count}
            )
        measure_rows.extend(extract_band_window_rows(subject, means, counts, freqs, times))
        included.append(subject)

    if not included:
        raise RuntimeError("No complete subject TFRs found.")

    group_dir = OUT_ROOT / "group"
    group_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(count_rows).to_csv(group_dir / "tfr_trial_counts.csv", index=False)
    pd.DataFrame(measure_rows).to_csv(group_dir / "tfr_subject_band_window_measures.csv", index=False)

    group_means = {
        condition: np.stack(matrices).mean(axis=0)
        for condition, matrices in by_condition.items()
    }
    group_sem = {
        condition: np.stack(matrices).std(axis=0, ddof=1) / np.sqrt(len(matrices))
        for condition, matrices in by_condition.items()
    }

    condition_vlim = symmetric_limit(list(group_means.values()))
    for condition in CONDITION_ORDER:
        plot_tfr(
            group_means[condition],
            freqs,
            times,
            f"{condition}: posterior TFR, N={len(included)}",
            group_dir / f"group_{condition.lower()}_posterior-tfr.png",
            vlim=condition_vlim,
        )

    contrasts = {}
    for contrast_name, (a, b) in CONTRASTS.items():
        # Paired subject-level subtraction then group average
        subject_diffs = (
            np.stack(by_condition[a]) - np.stack(by_condition[b])
        )
        contrasts[contrast_name] = subject_diffs.mean(axis=0)

    contrast_vlim = symmetric_limit(list(contrasts.values()))
    for contrast_name, matrix in contrasts.items():
        plot_tfr(
            matrix,
            freqs,
            times,
            f"{contrast_name}: posterior TFR, N={len(included)}",
            group_dir / f"group_{contrast_name.lower()}_posterior-tfr.png",
            vlim=contrast_vlim,
        )

    np.savez_compressed(
        group_dir / "group_posterior_tfrs.npz",
        freqs=freqs,
        times=times,
        included_subjects=np.asarray(included),
        **{
            f"condition__{name}": matrix.astype(np.float32)
            for name, matrix in group_means.items()
        },
        **{
            f"contrast__{name}": matrix.astype(np.float32)
            for name, matrix in contrasts.items()
        },
    )
    print(f"\nIncluded N={len(included)}: {included}")
    print(f"Group outputs: {group_dir}")

if __name__ == "__main__":
    main()
