from pathlib import Path
import re

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data"

GROUP_OUT = PROJECT_ROOT / "output-iclabel" / "group" / "psychometric_plots"
GROUP_OUT.mkdir(parents=True, exist_ok=True)

SUBJECTS = [
    "sub-001", 
    "sub-002", 
    "sub-003", 
    "sub-004",
    "sub-005",
    "sub-006",
    "sub-007",
    "sub-008",
    "sub-009",
    "sub-010",
    "sub-011",
    "sub-012",
    "sub-013",
    "sub-014",
    "sub-015",
    "sub-016",
    "sub-017",
    "sub-018",
    "sub-019",
    "sub-020",
    "sub-021",
    "sub-022",
    "sub-023",
    "sub-024",
    "sub-026",
    "sub-027",
    "sub-030",
    "sub-031",
    "sub-032",
    "sub-034",
    "sub-035",
    "sub-036",
    "sub-037",
    "sub-038",
    "sub-039",
    "sub-040",
]


def normalize_subject(subject):
    if subject is None:
        return None

    subject = str(subject).replace("sub-", "")
    return f"sub-{subject.zfill(3)}"


def parse_compare_event(event: str) -> dict:
    pattern = (
        r"cond=(?P<cond>\w+)_"
        r"refSpeedIdx=(?P<refSpeedIdx>\d+)_"
        r"refSpeedVal=(?P<refSpeedVal>[\d.]+)_"
        r"currSpeedIdx=(?P<currSpeedIdx>\d+)_"
        r"currSpeedVal=(?P<currSpeedVal>[\d.]+)_"
        r"response=(?P<response>\w+)_"
        r"actual=(?P<actual>\w+)_"
        r"accuracy=(?P<accuracy>\w+)_"
        r"rt=(?P<rt>[\d.]+)"
    )

    match = re.search(pattern, event)
    if match is None:
        return {}

    d = match.groupdict()
    d["refSpeedIdx"] = int(d["refSpeedIdx"])
    d["currSpeedIdx"] = int(d["currSpeedIdx"])
    d["refSpeedVal"] = float(d["refSpeedVal"])
    d["currSpeedVal"] = float(d["currSpeedVal"])
    d["rt"] = float(d["rt"])
    return d


def logistic_from_params(x, intercept, slope):
    z = intercept + slope * x
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def fit_binomial_logistic(summary):
    x = summary["currSpeedVal"].to_numpy()
    k = summary["n_faster"].to_numpy()
    n = summary["n"].to_numpy()

    def neg_log_likelihood(params):
        p = logistic_from_params(x, params[0], params[1])
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.sum(k * np.log(p) + (n - k) * np.log(1 - p))

    result = minimize(
        neg_log_likelihood,
        x0=np.array([-2.0, 2.0]),
        method="Nelder-Mead",
    )

    if not result.success:
        return None

    return result.x

def wilson_interval(k, n, z=1.96):
    """Wilson 95% confidence interval for a binomial proportion."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half_width = (
        z
        * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))
        / denom
    )
    return centre - half_width, centre + half_width

def make_summary(df):
    summary = (
        df.groupby("currSpeedVal")
        .agg(
            p_faster=("answered_faster", "mean"),
            n_faster=("answered_faster", "sum"),
            n=("answered_faster", "size"),
        )
        .reset_index()
        .sort_values("currSpeedVal")
    )
    summary["ci_low"], summary["ci_high"] = wilson_interval(
        summary["n_faster"], summary["n"]
    )
    return summary

def plot_psychometric(plot_df, title, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    condition_order = ["Forward", "Random", "Rotation", "Spiral"]

    plot_df["cond"] = pd.Categorical(
        plot_df["cond"],
        categories=condition_order,
        ordered=True,
    )

    for cond, cond_df in plot_df.groupby("cond", sort=False):
        summary = make_summary(cond_df)

        x = summary["currSpeedVal"].to_numpy()
        y = summary["p_faster"].to_numpy()
        ci_low = summary["ci_low"].to_numpy()
        ci_high = summary["ci_high"].to_numpy()

        points = ax.scatter(
            x,
            y,
            label=f"{cond} observed",
        )

        point_color = points.get_facecolor()[0]

        if len(summary) >= 4 and summary["p_faster"].nunique() > 1:
            params = fit_binomial_logistic(summary)

            if params is not None:
                x_fit = np.linspace(x.min(), x.max(), 200)
                y_fit = logistic_from_params(x_fit, params[0], params[1])
                ax.plot(
                    x_fit,
                    y_fit,
                    color=point_color,
                    label=f"{cond} logistic fit",
                )
            else:
                print(f"Fit failed for {title}, {cond}")
    
    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1,
        color="0.35",
        label="Equal response probability (0.5)",
    )

    handles, labels = ax.get_legend_handles_labels()
    lookup = dict(zip(labels, handles))

    desired = [
        "Forward observed",
        "Forward logistic fit",
        "Random observed",
        "Random logistic fit",
        "Rotation observed",
        "Rotation logistic fit",
        "Spiral observed",
        "Spiral logistic fit",
        "Equal response probability (0.5)",
    ]

    ordered_labels = [label for label in desired if label in lookup]
    ordered_handles = [lookup[label] for label in ordered_labels]

    ax.legend(
        ordered_handles,
        ordered_labels,
        ncol=2,
        fontsize=9,
        loc="upper left",
    )

    ax.set_xlabel("Stimulus speed (m/s)")
    ax.set_ylabel("Probability of 'Faster' response")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def canonicalize_condition(cond):
    """Combine left/right rotation and spiral directions"""
    cond = str(cond)
    lower = cond.lower()
    if lower.startswith("rotation"):
        return "Rotation"
    if lower.startswith("spiral"):
        return "Spiral"
    if lower.startswith("forward"):
        return "Forward"
    if lower.startswith("random"):
        return "Random"
    return cond

def load_subject_data(subject):
    file_path = (
        DATA_ROOT
        / subject
        / "ses-001"
        / "misc"
        / f"{subject}_ses-001_task-compareSpeed_events.csv"
    )

    if not file_path.exists():
        print(f"Missing file, skipping: {file_path}")
        return None

    raw = pd.read_csv(file_path)
    result_mask = raw["event"].astype(str).str.startswith(
        "COMPARE_TO_MEAN_RESULT"
    )
    df = raw.loc[result_mask].copy()
    n_result_events = len(df)

    parsed = df["event"].apply(parse_compare_event)
    parsed_df = pd.DataFrame(parsed.tolist())
    n_parse_failed = int(parsed_df.get("response", pd.Series(index=df.index, dtype=object)).isna().sum())

    df = pd.concat([df.reset_index(drop=True), parsed_df.reset_index(drop=True)], axis=1)
    before_drop = len(df)
    df = df.dropna(subset=["response", "currSpeedVal", "cond"]).copy()
    n_dropped_required = before_drop - len(df)

    valid_response = df["response"].isin(["FASTER", "SLOWER"])
    n_invalid_response = int((~valid_response).sum())
    df = df.loc[valid_response].copy()

    df["condition_raw"] = df["cond"]
    df["cond"] = df["cond"].map(canonicalize_condition)
    df["subject"] = subject
    df["answered_faster"] = (df["response"] == "FASTER").astype(int)

    print(
        f"{subject}: result_events={n_result_events}, "
        f"parse_failures={n_parse_failed}, "
        f"dropped_missing_fields={n_dropped_required}, "
        f"invalid_responses={n_invalid_response}, retained={len(df)}"
    )

    return df

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sub",
        type=str,
        default=None,
        help="Subject to process, e.g. 15, 015, or sub-015",
    )

    args = parser.parse_args()
    subject = normalize_subject(args.sub)

    if subject is None:
        subjects = SUBJECTS
    else:
        subjects = [subject]

    all_rows = []

    for subject_name in subjects:
        df = load_subject_data(subject_name)

        if df is not None:
            all_rows.append(df)

    if not all_rows:
        raise FileNotFoundError(f"No participant files found in: {DATA_ROOT}")

    data = pd.concat(all_rows, ignore_index=True)

    print("=== Overall response counts ===")
    print(data["response"].value_counts())

    print("\n=== Reference speed value counts ===")
    print(data["refSpeedVal"].value_counts())

    print("\n=== Response counts by speed and condition ===")

    summary_cond = (
        data.groupby(["cond", "currSpeedVal"])
        .agg(
            n_trials=("answered_faster", "size"),
            p_faster=("answered_faster", "mean"),
        )
        .reset_index()
        .sort_values(["cond", "currSpeedVal"])
    )

    print(summary_cond)

    print("\n=== Total retained trials by condition ===")
    condition_totals = (
        data.groupby("cond")
        .agg(
            n_trials=("answered_faster", "size"),
            n_subjects=("subject", "nunique"),
        )
        .sort_index()
    )
    print(condition_totals)

    print("\n=== Retained trials by subject and condition ===")
    subject_condition_counts = pd.crosstab(data["subject"], data["cond"])
    print(subject_condition_counts.to_string())
    subject_condition_counts.to_csv(
        GROUP_OUT / "trial_counts_by_subject_and_condition.csv"
    )

    slowest = data["currSpeedVal"].min()

    print(f"\n=== Slowest speed ({slowest}) ===")

    print(
        data[data["currSpeedVal"] == slowest]
        .groupby("cond")
        .agg(
            n_trials=("answered_faster", "size"),
            p_faster=("answered_faster", "mean"),
        )
    )

    data = data.sort_values(["subject", "onset"]).reset_index(drop=True)

    if data.empty:
        raise ValueError("No trials remain.")

    if subject is None:
        parsed_out = GROUP_OUT / "compare_speed_parsed_trials.csv"
    else:
        parsed_out = (
            PROJECT_ROOT
            / "output"
            / subject
            / f"{subject}_compare_speed_parsed_trials.csv"
        )
        parsed_out.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(parsed_out, index=False)
    print(f"\nSaved parsed trials to: {parsed_out}")

    for subject_name, sub_df in data.groupby("subject"):
        subject_out = PROJECT_ROOT / "output" / subject_name
        subject_out.mkdir(parents=True, exist_ok=True)

        plot_psychometric(
            sub_df,
            title=f"Psychometric curve: {subject_name}",
            out_path=subject_out / f"{subject_name}_psychometric_curve.png",
        )

    if subject is None:
        plot_psychometric(
            data,
            title="Group speed judgments by optic flow condition",
            out_path=GROUP_OUT / "group_psychometric_curve.png",
        )

        print(f"Group outputs saved to: {GROUP_OUT}")

    print("Done.")
    print(f"Subject outputs saved to: {PROJECT_ROOT / 'output' / 'sub-XXX'}")

if __name__ == "__main__":
    main()
