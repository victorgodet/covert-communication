"""Generate experiment plot: judge score + accuracy vs step, with ICL events.

Usage:
    uv run --dev scripts/plot_experiment.py <run_dir> [output_path]

Example:
    uv run --dev scripts/plot_experiment.py results/runs/qwen3-4b/20260303_155433_covert_comm_level2_cr0.5_ow1.0
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_run(run_dir):
    rollout_dir = os.path.join(run_dir, "rollouts")
    steps = sorted(
        int(f.replace("step_", "").replace(".jsonl", ""))
        for f in os.listdir(rollout_dir)
        if f.endswith(".jsonl")
    )

    data = {
        "steps": [],
        "accuracy": [],
        "judge_mean": [],
        "judge_std": [],
        "targets": [],
        "icl_events": [],  # (step, target, score, message)
    }

    seen_icl = set()

    for step in steps:
        path = os.path.join(rollout_dir, f"step_{step}.jsonl")
        with open(path) as f:
            records = [json.loads(l) for l in f]

        n = len(records)
        if n == 0:
            continue

        target = records[0].get("target", "?")
        n_correct = sum(1 for r in records if r.get("correct", 0) > 0)
        acc = n_correct / n * 100

        scores = []
        for r in records:
            if r.get("correct", 0) > 0:
                try:
                    scores.append(int(r.get("judge_verdict", "10")))
                except (ValueError, TypeError):
                    pass

        avg_j = np.mean(scores) if scores else 10.0
        std_j = np.std(scores) if scores else 0.0

        data["steps"].append(step)
        data["accuracy"].append(acc)
        data["judge_mean"].append(avg_j)
        data["judge_std"].append(std_j)
        data["targets"].append(target)

        # Detect ICL events: best stealthy correct sample with score <= 2
        best_score = 11
        best_rec = None
        for r in records:
            if r.get("correct", 0) <= 0:
                continue
            try:
                s = int(r.get("judge_verdict", "10"))
            except (ValueError, TypeError):
                continue
            if s < best_score:
                best_score = s
                best_rec = r
        if best_rec is not None and best_score <= 2:
            msg = best_rec.get("encode_text", "").replace("\n", " ").strip()
            if msg not in seen_icl:
                seen_icl.add(msg)
                data["icl_events"].append((step, target, best_score, msg))

    return data


def plot_experiment(data, output_path, run_name=""):
    n_icl = len(data["icl_events"])
    # Allocate space for ICL annotations below the plot
    annot_height = max(1.0, n_icl * 0.45)
    fig_height = 5 + annot_height
    fig, ax1 = plt.subplots(figsize=(14, fig_height))
    fig.subplots_adjust(bottom=annot_height / fig_height)

    steps = np.array(data["steps"])
    judge_mean = np.array(data["judge_mean"])
    judge_std = np.array(data["judge_std"])
    accuracy = np.array(data["accuracy"])

    # Center judge axis around the data
    j_lo = max(1, int(np.floor(min(judge_mean - judge_std))) - 1)
    j_hi = min(10, int(np.ceil(max(judge_mean + judge_std))) + 1)
    j_lo = min(j_lo, 4)  # always show at least down to 4
    j_hi = max(j_hi, 10)  # always show up to 10

    # Judge score (left axis)
    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Judge Score (correct samples)", fontsize=12, color="#d62728")
    ax1.plot(steps, judge_mean, color="#d62728", linewidth=2, label="Judge mean", zorder=3)
    ax1.fill_between(
        steps,
        judge_mean - judge_std,
        judge_mean + judge_std,
        alpha=0.15,
        color="#d62728",
        zorder=2,
    )
    ax1.set_ylim(j_lo, j_hi + 0.5)
    ax1.set_yticks(range(j_lo, j_hi + 1))
    ax1.tick_params(axis="y", labelcolor="#d62728")

    # Accuracy (right axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy %", fontsize=12, color="#1f77b4")
    ax2.plot(steps, accuracy, color="#1f77b4", linewidth=2, linestyle="--", label="Accuracy", zorder=3)
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    # ICL event vertical lines
    for step, target, score, msg in data["icl_events"]:
        ax1.axvline(x=step, color="#2ca02c", alpha=0.4, linestyle=":", linewidth=1.5, zorder=1)

    # Legend
    judge_patch = mpatches.Patch(color="#d62728", label="Judge score (±std)")
    acc_patch = mpatches.Patch(color="#1f77b4", label="Accuracy %")
    icl_patch = mpatches.Patch(color="#2ca02c", label="ICL example added")
    ax1.legend(handles=[judge_patch, acc_patch, icl_patch], loc="upper right", fontsize=9)

    title = "Covert Communication: Judge Score & Accuracy"
    if run_name:
        title += f"\n{run_name}"
    plt.title(title, fontsize=13)

    # ICL annotations below the plot area as a clean list
    for i, (step, target, score, msg) in enumerate(data["icl_events"]):
        full_msg = msg.replace("\n", " ").strip()
        label = f"s{step:>2d}  [{target} {score}/10]  \"{full_msg}\""
        y_frac = (annot_height - 0.35 - i * 0.4) / fig_height
        fig.text(
            0.08, y_frac,
            label,
            fontsize=8,
            color="#2ca02c",
            fontfamily="monospace",
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fff0", edgecolor="#2ca02c", alpha=0.9),
        )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()


def generate_icl_table(data):
    """Generate markdown table of ICL events."""
    if not data["icl_events"]:
        return "*No ICL examples accumulated yet.*"
    lines = ["| # | Step | Target | Score | Message |", "|---|------|--------|-------|---------|"]
    for i, (step, target, score, msg) in enumerate(data["icl_events"], 1):
        safe_msg = msg.replace("|", "\\|")
        if len(safe_msg) > 80:
            safe_msg = safe_msg[:77] + "..."
        lines.append(f"| {i} | {step} | {target} | {score}/10 | {safe_msg} |")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run --dev scripts/plot_experiment.py <run_dir> [output_path]")
        sys.exit(1)

    run_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "assets/experiment_plot.png"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    run_name = os.path.basename(run_dir.rstrip("/"))
    data = load_run(run_dir)
    plot_experiment(data, output_path, run_name)

    print(f"\nICL examples ({len(data['icl_events'])} total):")
    print(generate_icl_table(data))
