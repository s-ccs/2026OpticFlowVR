from pathlib import Path

import mne
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SESSION = "001"
TASK = "compareSpeed"

DERIV_ROOT = PROJECT_ROOT / "output" / "derivatives" / "mne-bids-pipeline"
PLOTS_ROOT = PROJECT_ROOT / "output" / "plots"

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


def load_epochs(subject):
    epochs_file = get_epochs_file(subject)

    if not epochs_file.exists():
        print(f"Skipping sub-{subject}: missing file {epochs_file}")
        return None

    print(f"\nLoading epochs for sub-{subject}:\n{epochs_file}")

    epochs = mne.read_epochs(epochs_file, preload=True)
    epochs.apply_baseline((-0.2, 0.0))

    # low-pass filter for plotting only
    epochs_plot = epochs.copy().filter(l_freq=None, h_freq=30.0)

    print(f"Epochs have baseline: {epochs.baseline}")
    print(epochs_plot)
    print("Available event IDs:")
    print(epochs_plot.event_id)

    return epochs_plot


def create_evokeds(epochs):
    evokeds = {
        condition: epochs[condition].average()
        for condition in CONDITIONS
        if condition in epochs.event_id
    }

    print(f"Created evokeds for: {list(evokeds.keys())}")

    return evokeds


def plot_condition_subplots(subject, out_dir, evokeds, channels, group_name):
    available_channels = [
        ch for ch in channels
        if ch in next(iter(evokeds.values())).ch_names
    ]

    if not available_channels:
        print(f"Skipping {group_name}: no available channels")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, condition in zip(axes, CONDITIONS):
        evoked = evokeds[condition].copy().pick(available_channels)

        data_uv = evoked.data.mean(axis=0) * 1e6

        ax.plot(evoked.times, data_uv)
        ax.axvline(0, linestyle="--", color="black", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(condition)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("µV")

    fig.suptitle(f"Sub-{subject} {group_name} ERP by condition", fontsize=16)
    fig.tight_layout()

    fig.savefig(
        out_dir / f"sub-{subject}_{group_name}_condition-subplots.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_condition_erps(subject, out_dir, evokeds, epochs):
    for group_name, channels in CHANNEL_GROUPS.items():
        available_channels = [ch for ch in channels if ch in epochs.ch_names]

        if not available_channels:
            print(f"Skipping {group_name}: no available channels")
            continue

        fig = mne.viz.plot_compare_evokeds(
            evokeds,
            picks=available_channels,
            combine="mean",
            show=False,
            title=f"Sub-{subject} {group_name} ERP",
        )

        fig[0].savefig(
            out_dir / f"sub-{subject}_{group_name}_condition-erps.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig[0])


def plot_condition_subplots_for_selected_groups(subject, out_dir, evokeds):
    selected_groups = ["posterior", "parieto_occipital", "fcz"]

    for group_name in selected_groups:
        plot_condition_subplots(
            subject,
            out_dir,
            evokeds,
            CHANNEL_GROUPS[group_name],
            group_name,
        )


def plot_topomaps(subject, out_dir, epochs):
    evoked_all = epochs.average()

    fig = evoked_all.plot_topomap(
        times=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
        ch_type="eeg",
        show=False,
        time_unit="s",
    )

    fig.savefig(
        out_dir / f"sub-{subject}_all-conditions_topomaps.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def process_subject(subject):
    out_dir = PLOTS_ROOT / f"sub-{subject}"
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = load_epochs(subject)

    if epochs is None:
        return

    evokeds = create_evokeds(epochs)

    if not evokeds:
        print(f"Skipping sub-{subject}: no evokeds created")
        return

    plot_condition_erps(subject, out_dir, evokeds, epochs)
    plot_condition_subplots_for_selected_groups(subject, out_dir, evokeds)
    plot_topomaps(subject, out_dir, epochs)

    print(f"Saved plots for sub-{subject} to: {out_dir}")


def main():
    for subject in SUBJECTS:
        process_subject(subject)


if __name__ == "__main__":
    main()
