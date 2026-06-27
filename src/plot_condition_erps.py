import argparse
from pathlib import Path

import mne
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SESSION = "001"
TASK = "compareSpeed"

DERIV_ROOT = PROJECT_ROOT / "output" / "derivatives" / "mne-bids-pipeline"
PLOTS_ROOT = PROJECT_ROOT / "output" 

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
    "016",
    # "017",
    "018",
    "019",
    "020",
    "021",
    "022",
    "023",
]

CONDITIONS = [
    "Forward",
    "Rotation/Left",
    "Rotation/Right",
    "Spiral/Left",
    "Spiral/Right",
    "Random",
]

SPEED_LEVELS = {
    "speed-0": "0.8 m/s",
    "speed-1": "1.0 m/s",
    "speed-2": "1.2 m/s",
    "speed-3": "1.4 m/s",
    "speed-4": "1.6 m/s",
    "speed-5": "1.8 m/s",
    "speed-6": "2.0 m/s",
}

CHANNEL_GROUPS = {
    "occipital": ["O1", "O2"],
    "parieto_occipital": ["POz", "PO3", "PO4", "PO7", "PO8"],
    "posterior": ["O1", "O2", "POz", "PO3", "PO4", "PO7", "PO8"],
    "central": ["FCz", "Cz", "CP1", "CP2", "Pz"],
    "fcz": ["FCz"],
}

TIME_WINDOWS = {
    "early_100_200": (0.100, 0.200),
    "mid_200_350": (0.200, 0.350),
    "late_350_600": (0.350, 0.600),
}

FONT_SIZES = {
    "title": 24,
    "subtitle": 20,
    "axis_label": 20,
    "tick_label": 18,
    "legend": 18,
}


def get_epochs_file(subject):
    eeg_dir = (
        DERIV_ROOT
        / f"sub-{subject}"
        / f"ses-{SESSION}"
        / "eeg"
    )

    fixation_file = (
        eeg_dir
        / f"sub-{subject}_ses-{SESSION}_task-{TASK}_proc-clean_fixation-epo.fif"
    )

    clean_file = (
        eeg_dir
        / f"sub-{subject}_ses-{SESSION}_task-{TASK}_proc-clean_epo.fif"
    )

    if fixation_file.exists():
        print(f"Using fixation-cleaned epochs for sub-{subject}")
        return fixation_file

    return clean_file


def load_epochs(subject):
    epochs_file = get_epochs_file(subject)

    if not epochs_file.exists():
        print(f"Skipping sub-{subject}: missing file {epochs_file}")
        return None

    print(f"\nLoading epochs for sub-{subject}:\n{epochs_file}")

    epochs = mne.read_epochs(epochs_file, preload=True)
    epochs.apply_baseline((-0.2, 0.0))

    print(f"Epochs have baseline: {epochs.baseline}")
    print(epochs)
    print("Available event IDs:")
    print(epochs.event_id)

    return epochs


def create_condition_evokeds(epochs):
    evokeds = {}

    for condition in CONDITIONS:
        try:
            condition_epochs = epochs[condition]
        except KeyError:
            print(f"Missing {condition}")
            continue

        if len(condition_epochs) == 0:
            print(f"No epochs for {condition}")
            continue

        evoked = condition_epochs.average()
        evoked.comment = condition
        evokeds[condition] = evoked

        print(f"{condition}: {len(condition_epochs)} epochs")

    print(f"Created condition evokeds for: {list(evokeds.keys())}")
    return evokeds


def create_speed_evokeds(epochs):
    evokeds = {}

    for speed_key, speed_label in SPEED_LEVELS.items():
        try:
            speed_epochs = epochs[speed_key]
        except KeyError:
            print(f"Missing {speed_key}")
            continue

        if len(speed_epochs) == 0:
            print(f"No epochs for {speed_key}")
            continue

        evoked = speed_epochs.average()
        evoked.comment = speed_label
        evokeds[speed_label] = evoked

        print(f"{speed_label}: {len(speed_epochs)} epochs")

    print(f"Created speed evokeds for: {list(evokeds.keys())}")
    return evokeds


def create_condition_speed_evokeds(epochs):
    evokeds = {}

    for condition in CONDITIONS:
        for speed_key, speed_label in SPEED_LEVELS.items():
            event_name = f"{condition}/{speed_key}"

            try:
                selected_epochs = epochs[event_name]
            except KeyError:
                print(f"Missing {event_name}")
                continue

            if len(selected_epochs) == 0:
                continue

            label = f"{condition} | {speed_label}"
            evoked = selected_epochs.average()
            evoked.comment = label
            evokeds[label] = evoked

            print(f"{label}: {len(selected_epochs)} epochs")

    return evokeds


def extract_erp_measures(subject, evokeds, analysis_name):
    rows = []

    for level_name, evoked in evokeds.items():
        condition = None
        speed = None

        if analysis_name == "condition_speed":
            condition, speed = level_name.split(" | ")
        elif analysis_name == "condition":
            condition = level_name
        elif analysis_name == "speed":
            speed = level_name

        for group_name, channels in CHANNEL_GROUPS.items():
            available_channels = [
                ch for ch in channels
                if ch in evoked.ch_names
            ]

            if not available_channels:
                continue

            evoked_roi = evoked.copy().pick(available_channels)
            data_uv = evoked_roi.data.mean(axis=0) * 1e6

            for window_name, (tmin, tmax) in TIME_WINDOWS.items():
                time_mask = (
                    (evoked_roi.times >= tmin)
                    & (evoked_roi.times <= tmax)
                )

                window_data = data_uv[time_mask]
                window_times = evoked_roi.times[time_mask]

                mean_amp = window_data.mean()

                peak_idx = abs(window_data).argmax()
                peak_amp = window_data[peak_idx]
                peak_latency = window_times[peak_idx] * 1000

                rows.append({
                    "subject": subject,
                    "analysis": analysis_name,
                    "level": level_name,
                    "condition": condition,
                    "speed": speed,
                    "roi": group_name,
                    "time_window": window_name,
                    "tmin_ms": tmin * 1000,
                    "tmax_ms": tmax * 1000,
                    "mean_amplitude_uv": mean_amp,
                    "peak_amplitude_uv": peak_amp,
                    "peak_latency_ms": peak_latency,
                })

    return rows


def plot_erps(subject, out_dir, evokeds, epochs, analysis_name):
    for group_name, channels in CHANNEL_GROUPS.items():
        available_channels = [ch for ch in channels if ch in epochs.ch_names]

        if not available_channels:
            print(f"Skipping {group_name}: no available channels")
            continue

        title = f"Sub-{subject} {group_name} ERP by {analysis_name}"

        fig = mne.viz.plot_compare_evokeds(
            evokeds,
            picks=available_channels,
            combine="mean",
            show=False,
            title=title,
        )

        fig[0].set_size_inches(12, 8)
        style_compare_evokeds_figure(fig, title)

        fig[0].savefig(
            out_dir / f"sub-{subject}_{group_name}_{analysis_name}-erps.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig[0])


def plot_subplots(subject, out_dir, evokeds, labels, channels, group_name, analysis_name):
    available_channels = [
        ch for ch in channels
        if ch in next(iter(evokeds.values())).ch_names
    ]

    if not available_channels:
        print(f"Skipping {group_name}: no available channels")
        return

    n_plots = len(labels)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(20, 4 * n_rows),
        sharex=True,
        sharey=True,
    )

    axes = axes.ravel()

    for ax, label in zip(axes, labels):
        if label not in evokeds:
            ax.set_title(f"{label} missing")
            ax.axis("off")
            continue

        evoked = evokeds[label].copy().pick(available_channels)
        data_uv = evoked.data.mean(axis=0) * 1e6

        ax.plot(evoked.times, data_uv, linewidth=2)
        ax.axvline(0, linestyle="--", color="black", linewidth=1.2)
        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_title(label, fontsize=FONT_SIZES["subtitle"])
        ax.set_xlabel("Time (s)", fontsize=FONT_SIZES["axis_label"])
        ax.set_ylabel("µV", fontsize=FONT_SIZES["axis_label"])
        ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])

    for ax in axes[n_plots:]:
        ax.axis("off")

    fig.suptitle(
        f"Sub-{subject} {group_name} ERP by {analysis_name}",
        fontsize=FONT_SIZES["title"],
    )
    fig.tight_layout()

    fig.savefig(
        out_dir / f"sub-{subject}_{group_name}_{analysis_name}-subplots.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_subplots_for_selected_groups(subject, out_dir, evokeds, labels, analysis_name):
    selected_groups = ["posterior", "parieto_occipital", "fcz"]

    for group_name in selected_groups:
        plot_subplots(
            subject,
            out_dir,
            evokeds,
            labels,
            CHANNEL_GROUPS[group_name],
            group_name,
            analysis_name,
        )


def plot_topomaps(subject, out_dir, epochs):
    evoked_all = epochs.average()

    fig = evoked_all.plot_topomap(
        times=[0.10, 0.13, 0.15, 0.18, 0.20, 0.30, 0.40],
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


def style_compare_evokeds_figure(fig, title):
    ax = fig[0].axes[0]

    ax.set_title(title, fontsize=FONT_SIZES["title"])
    ax.set_xlabel("Time (s)", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("µV", fontsize=FONT_SIZES["axis_label"])

    ax.tick_params(
        axis="both",
        labelsize=FONT_SIZES["tick_label"],
    )

    legend = ax.get_legend()

    if legend is not None:
        legend.set_loc("upper center")
        legend.set_bbox_to_anchor((0.5, -0.15))
        legend.set_ncols(2)

        for text in legend.get_texts():
            text.set_fontsize(FONT_SIZES["legend"])


def process_subject(subject):
    out_dir = PLOTS_ROOT / f"sub-{subject}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    epochs = load_epochs(subject)
    if epochs is None:
        return

    condition_evokeds = create_condition_evokeds(epochs)
    speed_evokeds = create_speed_evokeds(epochs)
    condition_speed_evokeds = create_condition_speed_evokeds(epochs)

    if condition_evokeds:
        all_rows.extend(extract_erp_measures(subject, condition_evokeds, "condition",))
        plot_erps(subject, out_dir, condition_evokeds, epochs, "condition")
        plot_subplots_for_selected_groups(subject, out_dir, condition_evokeds, CONDITIONS, "condition")

    if speed_evokeds:
        all_rows.extend(extract_erp_measures(subject, speed_evokeds, "speed"))
        plot_erps(subject, out_dir, speed_evokeds, epochs, "speed")
        plot_subplots_for_selected_groups(subject, out_dir, speed_evokeds, list(SPEED_LEVELS.values()), "speed")

    if condition_speed_evokeds:
        all_rows.extend(extract_erp_measures(subject, condition_speed_evokeds, "condition_speed"))
    
    if all_rows:
        measures = pd.DataFrame(all_rows)
        measures_file = out_dir / f"sub-{subject}_erp-measures.csv"
        measures.to_csv(measures_file, index=False)
        print(f"Saved ERP measures to: {measures_file}")
    
    plot_topomaps(subject, out_dir, epochs)

    print(f"Saved plots for sub-{subject} to: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", type=str, default=None)
    args = parser.parse_args()

    if args.sub is not None:
        subjects = [args.sub.replace("sub-", "").zfill(3)]
    else:
        subjects = SUBJECTS

    for subject in subjects:
        process_subject(subject)


if __name__ == "__main__":
    main()
