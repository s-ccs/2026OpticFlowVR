from pathlib import Path
from collections import defaultdict

import mne
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SESSION = "001"
TASK = "compareSpeed"

DERIV_ROOT = PROJECT_ROOT / "output" / "derivatives" / "mne-bids-pipeline"
OUT_DIR = PROJECT_ROOT / "output" / "plots" / "group"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = [
    "002",
    "003",
    "004",
    "005",
    "006",
    "007",
    "008",
    "009",
    "010",
    "011",
    "012",
    "013",
    "014",
    "015",
]

CONDITIONS = [
    "Forward",
    "Random",
    "Rotation/Left",
    "Rotation/Right",
    "Spiral/Left",
    "Spiral/Right",
]

CHANNEL_GROUPS = {
    "occipital": ["O1", "O2"],
    "parieto_occipital": ["POz", "PO3", "PO4", "PO7", "PO8"],
    "posterior": ["O1", "O2", "POz", "PO3", "PO4", "PO7", "PO8"],
    "central": ["FCz", "Cz", "CP1", "CP2", "Pz"],
    "fcz": ["FCz"],
}


def get_epochs_file(subject):
    return (
        DERIV_ROOT
        / f"sub-{subject}"
        / f"ses-{SESSION}"
        / "eeg"
        / f"sub-{subject}_ses-{SESSION}_task-{TASK}_proc-clean_epo.fif"
    )


def load_subject_epochs(subject):
    epochs_file = get_epochs_file(subject)

    if not epochs_file.exists():
        print(f"Skipping sub-{subject}: missing {epochs_file}")
        return None

    print(f"\nLoading sub-{subject}: {epochs_file}")

    epochs = mne.read_epochs(epochs_file, preload=True)
    epochs.apply_baseline((-0.2, 0.0))

    # Visualization-only low-pass
    epochs = epochs.copy().filter(l_freq=None, h_freq=30.0)

    print(epochs)

    for condition in CONDITIONS:
        if condition in epochs.event_id:
            print(f"  {condition}: {len(epochs[condition])}")
        else:
            print(f"  {condition}: missing")

    return epochs


def collect_subject_evokeds():
    evokeds_by_condition = defaultdict(list)
    included_subjects = []

    for subject in SUBJECTS:
        epochs = load_subject_epochs(subject)

        if epochs is None:
            continue

        missing_conditions = [
            condition
            for condition in CONDITIONS
            if condition not in epochs.event_id
        ]

        if missing_conditions:
            print(f"Skipping sub-{subject}: missing {missing_conditions}")
            continue

        for condition in CONDITIONS:
            evoked = epochs[condition].average()
            evoked.comment = f"sub-{subject}"
            evokeds_by_condition[condition].append(evoked)

        included_subjects.append(subject)

    print("\nIncluded subjects:")
    print(included_subjects)

    return evokeds_by_condition, included_subjects


def compute_grand_averages(evokeds_by_condition):
    grand_averages = {}

    for condition, evokeds in evokeds_by_condition.items():
        if not evokeds:
            continue

        grand_averages[condition] = mne.grand_average(evokeds)
        grand_averages[condition].comment = condition

        print(f"{condition}: N={len(evokeds)} subjects")

    return grand_averages


def plot_group_condition_erps(grand_averages, included_subjects):
    for group_name, channels in CHANNEL_GROUPS.items():
        available_channels = [
            ch
            for ch in channels
            if ch in next(iter(grand_averages.values())).ch_names
        ]

        if not available_channels:
            print(f"Skipping {group_name}: no available channels")
            continue

        fig = mne.viz.plot_compare_evokeds(
            grand_averages,
            picks=available_channels,
            combine="mean",
            show=False,
            title=f"Group {group_name} ERP, N={len(included_subjects)}",
        )

        fig[0].savefig(
            OUT_DIR / f"group_{group_name}_condition-erps.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig[0])


def plot_group_condition_subplots(grand_averages, included_subjects, group_name):
    channels = CHANNEL_GROUPS[group_name]

    available_channels = [
        ch
        for ch in channels
        if ch in next(iter(grand_averages.values())).ch_names
    ]

    if not available_channels:
        print(f"Skipping {group_name}: no available channels")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, condition in zip(axes, CONDITIONS):
        if condition not in grand_averages:
            ax.set_title(f"{condition} missing")
            ax.axis("off")
            continue

        evoked = grand_averages[condition].copy().pick(available_channels)
        data_uv = evoked.data.mean(axis=0) * 1e6

        ax.plot(evoked.times, data_uv)
        ax.axvline(0, linestyle="--", color="black", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(condition)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("µV")

    fig.suptitle(
        f"Group {group_name} ERP by condition, N={len(included_subjects)}",
        fontsize=16,
    )
    fig.tight_layout()

    fig.savefig(
        OUT_DIR / f"group_{group_name}_condition-subplots.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_group_topomaps(grand_averages, included_subjects):
    # Collapsed across conditions
    all_evokeds = list(grand_averages.values())
    evoked_all = mne.grand_average(all_evokeds)
    evoked_all.comment = "All conditions"

    fig = evoked_all.plot_topomap(
        times=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
        ch_type="eeg",
        show=False,
        time_unit="s",
    )

    fig.savefig(
        OUT_DIR / f"group_all-conditions_topomaps_N-{len(included_subjects)}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_group_evokeds(grand_averages, included_subjects):
    evokeds = list(grand_averages.values())

    out_file = OUT_DIR / f"group_condition-grand-averages_N-{len(included_subjects)}_ave.fif"

    mne.write_evokeds(out_file, evokeds, overwrite=True)

    print(f"Saved group evokeds to: {out_file}")


def main():
    evokeds_by_condition, included_subjects = collect_subject_evokeds()

    if not included_subjects:
        raise RuntimeError("No subjects included. Check subject list and epoch files.")

    grand_averages = compute_grand_averages(evokeds_by_condition)

    plot_group_condition_erps(grand_averages, included_subjects)

    plot_group_condition_subplots(
        grand_averages,
        included_subjects,
        "posterior",
    )

    plot_group_condition_subplots(
        grand_averages,
        included_subjects,
        "parieto_occipital",
    )

    plot_group_condition_subplots(
        grand_averages,
        included_subjects,
        "fcz",
    )

    plot_group_topomaps(grand_averages, included_subjects)
    save_group_evokeds(grand_averages, included_subjects)

    print(f"\nSaved group plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()