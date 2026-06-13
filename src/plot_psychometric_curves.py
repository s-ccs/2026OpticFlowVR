from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data"

GROUP_OUT = PROJECT_ROOT / "output" / "group" / "psychometric_plots"
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
]


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


def make_summary(df):
    return (
        df.groupby("currSpeedVal")
        .agg(
            p_faster=("answered_faster", "mean"),
            n_faster=("answered_faster", "sum"),
            n=("answered_faster", "size"),
        )
        .reset_index()
        .sort_values("currSpeedVal")
    )


def plot_psychometric(plot_df, title, out_path):
    plt.figure(figsize=(8, 5))

    for cond, cond_df in plot_df.groupby("cond"):
        summary = make_summary(cond_df)

        x = summary["currSpeedVal"].to_numpy()
        y = summary["p_faster"].to_numpy()

        plt.scatter(x, y, label=f"{cond} data")

        if len(summary) >= 4 and summary["p_faster"].nunique() > 1:
            params = fit_binomial_logistic(summary)

            if params is not None:
                x_fit = np.linspace(x.min(), x.max(), 200)
                y_fit = logistic_from_params(x_fit, params[0], params[1])
                plt.plot(x_fit, y_fit, label=f"{cond} fit")
            else:
                print(f"Fit failed for {title}, {cond}")

    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.xlabel("Current speed")
    plt.ylabel("P(answered FASTER)")
    plt.title(title)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


all_rows = []

for subject in SUBJECTS:
    file_path = DATA_ROOT / subject / "ses-001" / "misc" / f"{subject}_ses-001_task-compareSpeed_events.csv"

    if not file_path.exists():
        print(f"Missing file, skipping: {file_path}")
        continue

    df = pd.read_csv(file_path)
    df = df[df["event"].astype(str).str.startswith("COMPARE_TO_MEAN_RESULT")].copy()

    parsed = df["event"].apply(parse_compare_event)
    parsed_df = pd.DataFrame(parsed.tolist())

    df = pd.concat([df.reset_index(drop=True), parsed_df], axis=1)
    df = df.dropna(subset=["response", "currSpeedVal", "cond"]).copy()

    df["subject"] = subject
    df["answered_faster"] = (df["response"] == "FASTER").astype(int)

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

data.to_csv(GROUP_OUT / "compare_speed_parsed_trials.csv", index=False)

for subject, sub_df in data.groupby("subject"):
    subject_out = PROJECT_ROOT / "output" / "plots" / subject
    subject_out.mkdir(parents=True, exist_ok=True)

    plot_psychometric(
        sub_df,
        title=f"Psychometric curve: {subject}",
        out_path=subject_out / f"{subject}_psychometric_curve.png",
    )

plot_psychometric(
    data,
    title="Group psychometric curve",
    out_path=GROUP_OUT / "group_psychometric_curve.png",
)

print("Done.")
print(f"Group outputs saved to: {GROUP_OUT}")
print(f"Subject outputs saved to: {PROJECT_ROOT / 'output' / 'plots' / 'sub-XXX'}")
