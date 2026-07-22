from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SESSION = "001"
TASK = "compareSpeed"
CONDITIONS = [
    "Forward", "Random", "Rotation/Left", "Rotation/Right",
    "Spiral/Left", "Spiral/Right",
]
POSTERIOR_CHANNELS = ["O1", "O2", "Oz", "POz", "PO3", "PO4", "PO7", "PO8"]
BASELINE_WINDOW = (-0.2, 0.0)
ERP_WINDOW = (0.0, 0.6)


def normalize_subject(value: str) -> str:
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        raise ValueError(f"Could not parse subject from {value!r}")
    return digits.zfill(3)


def discover_subjects(deriv_root: Path) -> list[str]:
    out = []
    for path in sorted(deriv_root.glob("sub-*")):
        match = re.fullmatch(r"sub-(\d+)", path.name)
        if path.is_dir() and match:
            out.append(match.group(1).zfill(3))
    return out


def eeg_dir(deriv_root: Path, subject: str) -> Path:
    return deriv_root / f"sub-{subject}" / f"ses-{SESSION}" / "eeg"


def clean_epochs_path(deriv_root: Path, subject: str) -> Path:
    return eeg_dir(deriv_root, subject) / (
        f"sub-{subject}_ses-{SESSION}_task-{TASK}_proc-clean_epo.fif"
    )


def fixation_epochs_path(deriv_root: Path, subject: str) -> Path:
    return eeg_dir(deriv_root, subject) / (
        f"sub-{subject}_ses-{SESSION}_task-{TASK}_proc-clean_fixation-epo.fif"
    )


def selected_epochs_path(deriv_root: Path, subject: str) -> tuple[Path, bool]:
    fix = fixation_epochs_path(deriv_root, subject)
    clean = clean_epochs_path(deriv_root, subject)
    if fix.exists():
        return fix, True
    if clean.exists():
        return clean, False
    raise FileNotFoundError(f"No cleaned epochs found for sub-{subject}")


def condition_count(epochs: mne.Epochs, condition: str) -> int:
    try:
        return int(len(epochs[condition]))
    except (KeyError, ValueError):
        return 0


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or len(y) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_erp_metrics(epochs: mne.Epochs) -> dict:
    posterior = [ch for ch in POSTERIOR_CHANNELS if ch in epochs.ch_names]
    if len(posterior) < 4:
        raise RuntimeError(f"Only {len(posterior)} posterior channels found: {posterior}")

    data_uv = epochs.get_data(picks=posterior) * 1e6
    roi_trials = data_uv.mean(axis=1)
    times = epochs.times
    baseline = (times >= BASELINE_WINDOW[0]) & (times <= BASELINE_WINDOW[1])
    response = (times >= ERP_WINDOW[0]) & (times <= ERP_WINDOW[1])

    all_erp = roi_trials.mean(axis=0)
    signal_rms = float(np.sqrt(np.mean(all_erp[response] ** 2)))
    sem_wave = np.std(roi_trials[:, response], axis=0, ddof=1) / np.sqrt(len(epochs))
    noise_rms = float(np.sqrt(np.mean(sem_wave ** 2)))
    snr_linear = signal_rms / noise_rms if noise_rms > 0 else float("nan")
    snr_db = 20 * np.log10(snr_linear) if snr_linear > 0 else float("nan")

    baseline_rms = np.sqrt(np.mean(roi_trials[:, baseline] ** 2, axis=1))
    idx = np.arange(len(epochs))
    odd = roi_trials[idx % 2 == 1].mean(axis=0)
    even = roi_trials[idx % 2 == 0].mean(axis=0)
    half = len(epochs) // 2
    first = roi_trials[:half].mean(axis=0)
    second = roi_trials[half:].mean(axis=0)

    return {
        "posterior_channels_used": ",".join(posterior),
        "median_posterior_baseline_rms_uv": float(np.median(baseline_rms)),
        "p95_posterior_baseline_rms_uv": float(np.percentile(baseline_rms, 95)),
        "all_trial_erp_signal_rms_uv": signal_rms,
        "all_trial_erp_noise_sem_rms_uv": noise_rms,
        "all_trial_erp_snr_linear": float(snr_linear),
        "all_trial_erp_snr_db": float(snr_db),
        "odd_even_r": safe_corr(odd[response], even[response]),
        "first_second_r": safe_corr(first[response], second[response]),
    }


def find_fixation_qc_file(project_root: Path, fixation_root: Path, subject: str) -> Path | None:
    expected = fixation_root / f"sub-{subject}" / "fixation" / (
        f"sub-{subject}_ses-{SESSION}_task-{TASK}_run-1_fixation-qc.csv"
    )
    if expected.exists():
        return expected
    for pattern in [f"**/sub-{subject}*fixation-qc.csv", f"**/sub-{subject}*fixation_qc.csv"]:
        matches = sorted(project_root.glob(pattern))
        if matches:
            return max(matches, key=lambda p: p.stat().st_mtime)
    return None


def fixation_metrics(project_root: Path, fixation_root: Path, subject: str,
                     n_before: int | None, n_after: int) -> dict:
    qc_file = find_fixation_qc_file(project_root, fixation_root, subject)
    if qc_file is not None:
        qc = pd.read_csv(qc_file)
        if "bad_fixation" in qc.columns:
            if qc["bad_fixation"].dtype == bool:
                bad = qc["bad_fixation"]
            else:
                bad = qc["bad_fixation"].astype(str).str.lower().eq("true")
            total = int(len(qc))
            rejected = int(bad.sum())
            return {
                "trials_entering_fixation_qc": total,
                "fixation_rejected_trials": rejected,
                "fixation_rejection_percent": 100 * rejected / total if total else float("nan"),
            }

    rejected = max(0, n_before - n_after) if n_before is not None else 0
    return {
        "trials_entering_fixation_qc": n_before,
        "fixation_rejected_trials": rejected,
        "fixation_rejection_percent": (
            100 * rejected / n_before if n_before else float("nan")
        ),
    }


def parse_list_from_log_line(line: str) -> list[str]:
    value = line.partition(":")[2].strip()
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple, set)):
            return sorted({str(item) for item in parsed})
    except (ValueError, SyntaxError):
        pass
    return sorted(set(re.findall(r"['\"]([^'\"]+)['\"]", value)))


def interpolated_channels(project_root: Path, subject: str) -> dict:
    candidates = [
        project_root / "output" / f"sub-{subject}" / "clean_bids.log",
        project_root / "output-iclabel" / f"sub-{subject}" / "clean_bids.log",
    ]
    log_file = next((p for p in candidates if p.exists()), None)
    channels = []
    if log_file is not None:
        for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if ("Bad EEG channels for interpolation:" in line or
                    "Final bad channels for interpolation:" in line):
                parsed = parse_list_from_log_line(line)
                if parsed:
                    channels = parsed
    return {
        "n_interpolated_eeg_channels": len(channels),
        "interpolated_eeg_channels": ",".join(channels),
    }


def find_ica_components_tsv(deriv_root: Path, subject: str) -> Path | None:
    root = eeg_dir(deriv_root, subject)
    for pattern in [
        f"sub-{subject}_ses-{SESSION}_proc-ica_components.tsv",
        f"sub-{subject}*proc-ica_components.tsv",
        f"sub-{subject}*ica_components.tsv",
    ]:
        matches = sorted(root.glob(pattern))
        if matches:
            return max(matches, key=lambda p: p.stat().st_mtime)
    return None


def normalize_component_name(value) -> str:
    text = str(value).strip()
    if re.fullmatch(r"\d+", text):
        return f"ICA{int(text):03d}"
    match = re.search(r"ICA\s*0*(\d+)", text, flags=re.IGNORECASE)
    return f"ICA{int(match.group(1)):03d}" if match else text


def rejected_ica_components(deriv_root: Path, subject: str) -> dict:
    path = find_ica_components_tsv(deriv_root, subject)
    if path is None:
        return {
            "n_rejected_ica_components": np.nan,
            "rejected_ica_components": "",
        }

    table = pd.read_csv(path, sep="\t")
    status_col = next((c for c in table.columns if str(c).lower() == "status"), None)
    if status_col is None:
        return {
            "n_rejected_ica_components": np.nan,
            "rejected_ica_components": "",
        }

    component_col = next(
        (c for c in table.columns if str(c).lower() in {
            "component", "name", "component_name", "ica_component"
        }),
        table.columns[0],
    )
    bad = table[status_col].astype(str).str.strip().str.lower().isin(
        {"bad", "exclude", "excluded", "reject", "rejected"}
    )
    rejected = sorted({normalize_component_name(v) for v in table.loc[bad, component_col]})
    return {
        "n_rejected_ica_components": len(rejected),
        "rejected_ica_components": ",".join(rejected),
    }


def process_subject(project_root: Path, deriv_root: Path, fixation_root: Path,
                    subject: str) -> tuple[dict | None, dict | None]:
    try:
        selected_path, used_fixation = selected_epochs_path(deriv_root, subject)
        epochs = mne.read_epochs(selected_path, preload=True, verbose="error")

        clean_path = clean_epochs_path(deriv_root, subject)
        n_before = None
        if clean_path.exists():
            n_before = int(len(mne.read_epochs(clean_path, preload=False, verbose="error")))

        row = {
            "subject": f"sub-{subject}",
            "used_fixation_cleaned_epochs": used_fixation,
            "retained_total_trials": int(len(epochs)),
            "trials_before_fixation": n_before,
        }
        for condition in CONDITIONS:
            row["trials_" + condition.lower().replace("/", "_")] = condition_count(epochs, condition)
        row["minimum_trials_across_conditions"] = min(
            row["trials_" + c.lower().replace("/", "_")] for c in CONDITIONS
        )

        row.update(fixation_metrics(project_root, fixation_root, subject, n_before, len(epochs)))
        row.update(compute_erp_metrics(epochs))
        row.update(interpolated_channels(project_root, subject))
        row.update(rejected_ica_components(deriv_root, subject))
        return row, None
    except Exception as exc:
        return None, {
            "subject": f"sub-{subject}",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def main() -> None:
    project_root = PROJECT_ROOT
    deriv_root = project_root / "output" / "derivatives" / "mne-bids-pipeline-iclabel"
    fixation_root = project_root / "output-iclabel" / "fixation-iclabel"
    output_dir = project_root / "output-iclabel" / "qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = discover_subjects(deriv_root)
    if not subjects:
        raise RuntimeError(f"No subjects found under {deriv_root}")

    print("=" * 80)
    print(f"Derivative root: {deriv_root}")
    print(f"Fixation root:   {fixation_root}")
    print(f"Subjects:        {len(subjects)}")
    print(f"Output folder:   {output_dir}")
    print("=" * 80)

    rows, errors = [], []
    for subject in subjects:
        row, error = process_subject(project_root, deriv_root, fixation_root, subject)
        if row is not None:
            rows.append(row)
            print(
                f"sub-{subject}: retained={row['retained_total_trials']}, "
                f"fix_rej={row['fixation_rejection_percent']:.1f}%, "
                f"SNR={row['all_trial_erp_snr_db']:.2f} dB, "
                f"interp={row['n_interpolated_eeg_channels']}, "
                f"ICA={row['n_rejected_ica_components']}"
            )
        else:
            errors.append(error)
            print(f"sub-{subject}: ERROR: {error['error']}")

    qc = pd.DataFrame(rows).sort_values("subject") if rows else pd.DataFrame()
    error_table = pd.DataFrame(errors)

    csv_file = output_dir / "common_subject_qc_table.csv"
    xlsx_file = output_dir / "common_subject_qc_table.xlsx"
    errors_file = output_dir / "common_subject_qc_errors.csv"

    qc.to_csv(csv_file, index=False)
    error_table.to_csv(errors_file, index=False)
    try:
        qc.to_excel(xlsx_file, index=False)
        wrote_excel = True
    except ImportError:
        wrote_excel = False

    print("\nSaved")
    print("-----")
    print(csv_file)
    if wrote_excel:
        print(xlsx_file)
    print(errors_file)


if __name__ == "__main__":
    main()