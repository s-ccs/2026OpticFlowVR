from pathlib import Path
from collections import defaultdict

import mne
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SESSION = "001"
TASK = "compareSpeed"

DERIV_ROOT = PROJECT_ROOT / "output" / "derivatives" / "mne-bids-pipeline-iclabel"
OUT_DIR = PROJECT_ROOT / "output-iclabel" / "group"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = [
    "002",
    "003",
    "004",
    "005",
    "006",
    # "007",
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
    # "018",
    "019",
    # "020",
    "021",
    "022",
    "023",
    "024",
    "026",
    "027",
    "029",
    "030",
    "031",
    "032",
    "034",
    "035",
    "036",
    "037",
    "038",
    "039",
    "040",
]

# Left/right directions are collapsed for condition-level analyses
CONDITIONS = [
    "Forward",
    "Rotation",
    "Spiral",
    "Random",
]

CONDITION_EVENT_NAMES = {
    "Forward": ["Forward"],
    "Rotation": ["Rotation/Left", "Rotation/Right"],
    "Spiral": ["Spiral/Left", "Spiral/Right"],
    "Random": ["Random"],
}

SPEED_LEVELS = {
    "speed-0": "0.8 m/s",
    "speed-1": "1.0 m/s",
    "speed-2": "1.2 m/s",
    "speed-3": "1.4 m/s",
    "speed-4": "1.6 m/s",
    "speed-5": "1.8 m/s",
    "speed-6": "2.0 m/s",
}

speed_colors = plt.cm.viridis(
    np.linspace(0.05, 0.95, len(SPEED_LEVELS))
)

SPEED_COLORS = {
    label: color
    for label, color in zip(SPEED_LEVELS.values(), speed_colors)
}

CHANNEL_GROUPS = {
    "occipital": ["O1", "O2"],
    "parieto_occipital": ["POz", "PO3", "PO4", "PO7", "PO8"],
    "posterior": ["O1", "O2", "POz", "PO3", "PO4", "PO7", "PO8"],
    "central": ["FCz", "Cz", "CP1", "CP2", "Pz"],
    "fcz": ["FCz"],
    "pz": ["Pz"],
}

TIME_WINDOWS = {
    "early_100_200": (0.100, 0.200),
    "mid_200_350": (0.200, 0.350),
    "late_350_600": (0.350, 0.600),
}

FONT_SIZES = {
    "title": 16,
    "subtitle": 16,
    "axis_label": 14,
    "tick_label": 12,
    "legend": 12,
}

PLOT_TMIN = -0.2
PLOT_TMAX = 0.8

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

def load_subject_epochs(subject):
    epochs_file = get_epochs_file(subject)

    if not epochs_file.exists():
        print(f"Skipping sub-{subject}: missing {epochs_file}")
        return None

    print(f"\nLoading sub-{subject}: {epochs_file}")

    epochs = mne.read_epochs(epochs_file, preload=True)

    standard_montage = mne.channels.make_standard_montage("standard_1020")

    epochs.set_montage(
        standard_montage,
        on_missing="warn",
    )

    fig = epochs.plot_sensors(
        kind="topomap",
        show_names=True,
        show=False,
        sphere=(0, 0, 0, 0.105),
    )

    fig.savefig(
        OUT_DIR / "montage_check.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    epochs.apply_baseline((-0.2, 0.0))

    print(epochs)

    for condition in CONDITIONS:
        try:
            condition_epochs = select_condition_epochs(epochs, condition)
            print(f"  {condition}: {len(condition_epochs)}")
        except KeyError:
            print(f"  {condition}: missing")

    return epochs

def select_condition_epochs(epochs, condition):
    raw_conditions = CONDITION_EVENT_NAMES[condition]
    matching_event_names = []

    for event_name in epochs.event_id:
        for raw_condition in raw_conditions:
            if (
                event_name == raw_condition
                or event_name.startswith(f"{raw_condition}/")
            ):
                matching_event_names.append(event_name)
                break

    if not matching_event_names:
        raise KeyError(
            f"No events found for condition {condition}. "
            f"Expected prefixes: {raw_conditions}"
        )

    return epochs[matching_event_names]

def select_condition_speed_epochs(epochs, condition, speed_key):
    event_names = [
        f"{event_name}/{speed_key}"
        for event_name in CONDITION_EVENT_NAMES[condition]
        if f"{event_name}/{speed_key}" in epochs.event_id
    ]
    if not event_names:
        raise KeyError(f"No events found for {condition}/{speed_key}")
    return epochs[event_names]

def compute_roi_mean_and_sem(evokeds, channels):
    subject_waveforms = []
    for evoked in evokeds:
        available_channels = [ch for ch in channels if ch in evoked.ch_names]
        if not available_channels:
            continue
        waveform_uv = evoked.copy().pick(available_channels).data.mean(axis=0) * 1e6
        subject_waveforms.append(waveform_uv)

    if not subject_waveforms:
        return None

    data = np.stack(subject_waveforms, axis=0)
    mean = data.mean(axis=0)
    if data.shape[0] < 2:
        sem = np.zeros_like(mean)
    else:
        sem = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])

    return mean, sem, data.shape[0]

def collect_subject_evokeds():
    evokeds_by_condition = defaultdict(list)
    evokeds_by_speed = defaultdict(list)
    evokeds_by_condition_speed = defaultdict(list)
    included_subjects = []
    measure_rows = []

    for subject in SUBJECTS:
        epochs = load_subject_epochs(subject)

        if epochs is None:
            continue

        subject_condition_evokeds = {}
        subject_speed_evokeds = {}
        subject_condition_speed_evokeds = {}

        for condition in CONDITIONS:
            try:
                condition_epochs = select_condition_epochs(epochs, condition)
            except KeyError:
                print(f"Missing {condition} for sub-{subject}")
                continue

            if len(condition_epochs) == 0:
                continue

            evoked = condition_epochs.average()
            evoked.comment = f"sub-{subject}"
            evokeds_by_condition[condition].append(evoked)
            subject_condition_evokeds[condition] = evoked

        if subject_condition_evokeds:
            measure_rows.extend(extract_erp_measures(subject, subject_condition_evokeds, "condition"))

        for speed_key, speed_label in SPEED_LEVELS.items():
            try:
                speed_epochs = epochs[speed_key]
            except KeyError:
                print(f"Missing {speed_key} for sub-{subject}")
                continue

            if len(speed_epochs) == 0:
                continue

            evoked = speed_epochs.average()
            evoked.comment = f"sub-{subject}"
            evokeds_by_speed[speed_label].append(evoked)
            subject_speed_evokeds[speed_label] = evoked

        if subject_speed_evokeds:
            measure_rows.extend(extract_erp_measures(subject, subject_speed_evokeds, "speed"))

        for condition in CONDITIONS:
            for speed_key, speed_label in SPEED_LEVELS.items():
                event_name = f"{condition}/{speed_key}"

                try:
                    selected_epochs = select_condition_speed_epochs(
                        epochs, condition, speed_key
                    )
                except KeyError:
                    print(f"Missing {event_name} for sub-{subject}")
                    continue

                if len(selected_epochs) == 0:
                    continue

                label = f"{condition} | {speed_label}"

                evoked = selected_epochs.average()
                evoked.comment = f"sub-{subject}"

                evokeds_by_condition_speed[label].append(evoked)
                subject_condition_speed_evokeds[label] = evoked

        if subject_condition_speed_evokeds:
            measure_rows.extend(extract_erp_measures(subject, subject_condition_speed_evokeds, "condition_speed"))

        if subject_condition_evokeds or subject_speed_evokeds or subject_condition_speed_evokeds:
            included_subjects.append(subject)

    print("\nIncluded subjects:")
    print(included_subjects)

    return evokeds_by_condition, evokeds_by_speed, evokeds_by_condition_speed, included_subjects, measure_rows

def compute_grand_averages(evokeds_by_level):
    grand_averages = {}

    for level, evokeds in evokeds_by_level.items():
        if not evokeds:
            continue

        grand_averages[level] = mne.grand_average(evokeds)
        grand_averages[level].comment = level

        print(f"{level}: N={len(evokeds)} subjects")

    return grand_averages

def extract_erp_measures(subject, evokeds_by_level, analysis_name):
    rows = []

    for level_name, evoked in evokeds_by_level.items():
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
                peak_latency_ms = window_times[peak_idx] * 1000

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
                    "peak_latency_ms": peak_latency_ms,
                })

    return rows

def plot_group_erps(evokeds_by_level, included_subjects, analysis_name):
    """Plot group ERPs with shaded SEM across participants"""
    for group_name, channels in CHANNEL_GROUPS.items():
        first_evoked = next(
            evoked
            for evokeds in evokeds_by_level.values()
            for evoked in evokeds
        )
        available_channels = [
            ch for ch in channels
            if ch in first_evoked.ch_names
        ]

        if not available_channels:
            print(f"Skipping {group_name}: no available channels")
            continue

        title = f"Group {group_name} ERP by {analysis_name}"
        fig, ax = plt.subplots(figsize=(10, 6))

        for label, evokeds in evokeds_by_level.items():
            result = compute_roi_mean_and_sem(
                evokeds,
                available_channels,
            )

            if result is None:
                continue

            mean_uv, sem_uv, n_subjects = result
            times = evokeds[0].times
            color = SPEED_COLORS.get(label)

            line = ax.plot(
                times,
                mean_uv,
                color=color,
                linewidth=2,
                label=f"{label}",
            )[0]

            ax.fill_between(
                times,
                mean_uv - sem_uv,
                mean_uv + sem_uv,
                color=line.get_color(),
                alpha=0.12,
                linewidth=0,
            )

        ax.axvline(
            0,
            linestyle="--",
            color="black",
            linewidth=1.2,
        )
        ax.axhline(
            0,
            color="black",
            linewidth=1.0,
        )

        ax.set_title(title, fontsize=FONT_SIZES["title"])
        ax.set_xlabel("Time (s)", fontsize=FONT_SIZES["axis_label"])
        ax.set_ylabel("Amplitude (µV)", fontsize=FONT_SIZES["axis_label"])
        ax.set_xlim(PLOT_TMIN, PLOT_TMAX)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        ax.tick_params(
            axis="both",
            labelsize=FONT_SIZES["tick_label"],
        )
        ax.legend(
            loc="upper right",
            fontsize=FONT_SIZES["legend"],
        )

        fig.tight_layout()
        out_file = OUT_DIR / f"group_{group_name}_{analysis_name}-erps.png"
        fig.savefig(
            OUT_DIR / f"group_{group_name}_{analysis_name}-erps.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Saved SEM ERP plot: {out_file.resolve()}")
        plt.close(fig)


def plot_group_subplots(evokeds_by_level, included_subjects, group_name, labels, analysis_name):
    channels = CHANNEL_GROUPS[group_name]
    first_evoked = next(
        evoked for evokeds in evokeds_by_level.values() for evoked in evokeds
    )
    available_channels = [ch for ch in channels if ch in first_evoked.ch_names]
    if not available_channels:
        print(f"Skipping {group_name}: no available channels")
        return

    n_plots = len(labels)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows),
        sharex=True, sharey=True, squeeze=False,
    )
    axes = axes.ravel()

    for ax, label in zip(axes, labels):
        evokeds = evokeds_by_level.get(label, [])
        if not evokeds:
            ax.set_title(f"{label} missing")
            ax.axis("off")
            continue

        result = compute_roi_mean_and_sem(evokeds, available_channels)
        if result is None:
            ax.set_title(f"{label} missing")
            ax.axis("off")
            continue

        mean_uv, sem_uv, n_subjects = result
        times = evokeds[0].times
        line = ax.plot(times, mean_uv, linewidth=2, label=label)[0]
        ax.fill_between(
            times, mean_uv - sem_uv, mean_uv + sem_uv,
            color=line.get_color(), alpha=0.12, linewidth=0,
        )
        ax.axvline(0, linestyle="--", color="black", linewidth=1.2)
        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_title(f"{label} (N={n_subjects})", fontsize=FONT_SIZES["subtitle"])
        ax.set_xlabel("Time (s)", fontsize=FONT_SIZES["axis_label"])
        ax.set_ylabel("Amplitude (µV)", fontsize=FONT_SIZES["axis_label"])
        ax.set_xlim(PLOT_TMIN, PLOT_TMAX)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])

    for ax in axes[n_plots:]:
        ax.axis("off")

    fig.suptitle(
        f"Group {group_name} ERP by {analysis_name} (shading: SEM)",
        fontsize=FONT_SIZES["title"],
    )
    fig.tight_layout()
    fig.savefig(
        OUT_DIR / f"group_{group_name}_{analysis_name}-subplots.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

def plot_group_topomaps(grand_averages, included_subjects):
    # Collapse across the four condition grand averages
    all_evokeds = list(grand_averages.values())
    evoked_all = mne.grand_average(all_evokeds)
    evoked_all.comment = "All conditions"

    topomap_times = [
        0.10,
        0.13,
        0.15,
        0.18,
        0.20,
        0.30,
        0.40,
        0.45,
        0.50,
        0.60,
        0.70,
        0.80,
    ]

    fig = evoked_all.plot_topomap(
        times=topomap_times,
        vlim=(-3, 3),
        ch_type="eeg",
        show=False,
        time_unit="s",
        time_format="%0.2f s",
        nrows=2,
        ncols=6,
        size=2.0,
        sphere=(0, 0, 0, 0.110),
    )

    fig.suptitle(
        f"Grand-average scalp topographies across conditions",
        fontsize=16,
    )

    fig.savefig(
        OUT_DIR
        / f"group_all-conditions_topomaps_N-{len(included_subjects)}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def style_compare_evokeds_figure(fig, title):
    ax = fig[0].axes[0]
    ax.set_title(title, fontsize=FONT_SIZES["title"])
    ax.set_xlabel("Time (s)", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("µV", fontsize=FONT_SIZES["axis_label"])
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])

    legend = ax.get_legend()
    if legend is not None:
        legend.set_loc("upper right")

        for text in legend.get_texts():
            text.set_fontsize(FONT_SIZES["legend"])


def save_group_evokeds(grand_averages, included_subjects, analysis_name):
    evokeds = list(grand_averages.values())
    out_file = OUT_DIR / f"group_{analysis_name}-grand-averages_N-{len(included_subjects)}_ave.fif"
    mne.write_evokeds(out_file, evokeds, overwrite=True)
    print(f"Saved group {analysis_name} evokeds to: {out_file}")


def main():
    evokeds_by_condition, evokeds_by_speed, evokeds_by_condition_speed, included_subjects, measure_rows = collect_subject_evokeds()

    if not included_subjects:
        raise RuntimeError("No subjects included. Check subject list and epoch files.")

    condition_grand_averages = compute_grand_averages(evokeds_by_condition)
    speed_grand_averages = compute_grand_averages(evokeds_by_speed)
    condition_speed_grand_averages = compute_grand_averages(evokeds_by_condition_speed)

    if measure_rows:
        measures = pd.DataFrame(measure_rows)
        measures_file = OUT_DIR / "erp_subject_level_measures.csv"
        measures.to_csv(measures_file, index=False)
        print(f"Saved ERP measures to: {measures_file}")

    if condition_grand_averages:
        plot_group_erps(evokeds_by_condition, included_subjects, "condition")

        for group_name in ["posterior", "parieto_occipital", "fcz"]:
            plot_group_subplots(evokeds_by_condition, included_subjects, group_name, CONDITIONS, "condition")

        plot_group_topomaps(condition_grand_averages, included_subjects)
        save_group_evokeds(condition_grand_averages, included_subjects, "condition")

    if speed_grand_averages:
        plot_group_erps(evokeds_by_speed, included_subjects, "speed")

        for group_name in ["posterior", "parieto_occipital", "fcz"]:
            plot_group_subplots(evokeds_by_speed, included_subjects, group_name, list(SPEED_LEVELS.values()), "speed")

        save_group_evokeds(speed_grand_averages, included_subjects, "speed")

    if condition_speed_grand_averages:
        save_group_evokeds(condition_speed_grand_averages, included_subjects, "condition_speed")

    print(f"\nSaved group plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()