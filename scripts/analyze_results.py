"""
Aggregate ablation results and produce summary table + figures.

Usage:
    python scripts/analyze_results.py --results results/ablation_results.json
    python scripts/analyze_results.py --results results/ablation_results.json \
                                      --output_dir results/figures/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


SCHEDULE_ORDER = [
    "hormetic_sigmoid",
    "hormetic_cosine",
    "linear",
    "reverse",
    "random_permutation",
    "fixed_beta",
]

SCHEDULE_LABELS = {
    "hormetic_sigmoid": "Hormetic-Sigmoid",
    "hormetic_cosine": "Hormetic-Cosine",
    "linear": "Linear",
    "reverse": "Reverse",
    "random_permutation": "Random Perm.",
    "fixed_beta": "Fixed-β",
}

COLORS = {
    "hormetic_sigmoid": "#1f77b4",
    "hormetic_cosine": "#aec7e8",
    "linear": "#ff7f0e",
    "reverse": "#d62728",
    "random_permutation": "#9467bd",
    "fixed_beta": "#7f7f7f",
}


def load_results(path: str) -> dict:
    """Load and group results by schedule name."""
    with open(path) as f:
        raw = json.load(f)

    grouped = {}
    for entry in raw:
        name = entry["schedule"]
        grouped.setdefault(name, []).append(entry)
    return grouped


def compute_stats(values):
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(arr),
    }


def build_summary_table(grouped: dict) -> str:
    """Produce a Markdown table of mean ± std identity retention accuracy."""
    occlusion_keys = None
    for runs in grouped.values():
        for r in runs:
            for dataset in ("clevrer", "adept"):
                if f"{dataset}_identity_retention" in r:
                    occlusion_keys = sorted(
                        r[f"{dataset}_identity_retention"].keys(), key=int
                    )
                    break
            if occlusion_keys:
                break
        if occlusion_keys:
            break

    if occlusion_keys is None:
        return "No identity retention metrics found in results."

    header = "| Schedule | " + " | ".join(f"k={k}" for k in occlusion_keys) + " | Mean Acc |"
    divider = "|" + "-|" * (len(occlusion_keys) + 2)

    rows = [header, divider]
    summary_data = {}

    for schedule in SCHEDULE_ORDER:
        if schedule not in grouped:
            continue
        runs = grouped[schedule]
        acc_by_k = {k: [] for k in occlusion_keys}
        for run in runs:
            for dataset in ("clevrer", "adept"):
                key = f"{dataset}_identity_retention"
                if key in run:
                    for k in occlusion_keys:
                        if k in run[key]:
                            acc_by_k[k].append(run[key][k])

        if not any(acc_by_k.values()):
            continue

        cell_strs = []
        means = []
        for k in occlusion_keys:
            vals = acc_by_k[k]
            if vals:
                m, s = np.mean(vals), np.std(vals)
                cell_strs.append(f"{m:.3f}±{s:.3f}")
                means.append(m)
            else:
                cell_strs.append("—")

        overall_mean = np.mean(means) if means else 0.0
        label = SCHEDULE_LABELS.get(schedule, schedule)
        row = f"| {label} | " + " | ".join(cell_strs) + f" | **{overall_mean:.3f}** |"
        rows.append(row)
        summary_data[schedule] = {"by_k": {k: np.mean(v) for k, v in acc_by_k.items() if v},
                                   "mean": overall_mean}

    return "\n".join(rows), summary_data


def plot_identity_retention_by_duration(summary_data: dict, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    for schedule in SCHEDULE_ORDER:
        if schedule not in summary_data:
            continue
        by_k = summary_data[schedule]["by_k"]
        ks = sorted(by_k.keys(), key=int)
        accs = [by_k[k] for k in ks]
        label = SCHEDULE_LABELS.get(schedule, schedule)
        color = COLORS.get(schedule, "black")
        linestyle = "-" if "hormetic" in schedule else "--"
        ax.plot(ks, accs, marker="o", label=label, color=color, linestyle=linestyle, linewidth=2)

    ax.set_xlabel("Occlusion Duration (frames)", fontsize=12)
    ax.set_ylabel("Identity Retention Accuracy", fontsize=12)
    ax.set_title("Object Identity Retention vs Occlusion Duration\nby β Schedule", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "identity_retention_by_duration.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_mean_accuracy_bar(summary_data: dict, output_dir: Path):
    schedules = [s for s in SCHEDULE_ORDER if s in summary_data]
    means = [summary_data[s]["mean"] for s in schedules]
    labels = [SCHEDULE_LABELS.get(s, s) for s in schedules]
    colors = [COLORS.get(s, "gray") for s in schedules]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, means, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mean Identity Retention Accuracy", fontsize=12)
    ax.set_title("Mean Identity Retention by β Schedule", fontsize=13)
    ax.set_ylim(0, 1)
    ax.axhline(y=means[0] if means else 0, color=COLORS["hormetic_sigmoid"],
               linestyle="--", alpha=0.3, label="Hormetic-Sigmoid level")

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    path = output_dir / "mean_accuracy_bar.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = load_results(str(results_path))
    print(f"Loaded results for schedules: {list(grouped.keys())}")

    table_str, summary_data = build_summary_table(grouped)
    print("\n=== Identity Retention Summary ===")
    print(table_str)

    table_path = output_dir / "summary_table.md"
    with open(table_path, "w") as f:
        f.write("# Schedule Ablation: Identity Retention Accuracy\n\n")
        f.write(table_str)
    print(f"\nSaved: {table_path}")

    if summary_data:
        plot_identity_retention_by_duration(summary_data, output_dir)
        plot_mean_accuracy_bar(summary_data, output_dir)

    print(f"\nAll outputs in: {output_dir}")


if __name__ == "__main__":
    main()
