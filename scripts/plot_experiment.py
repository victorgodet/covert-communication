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
        "judge_scores": [],  # list of (step, score) for scatter plot
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
        for s in scores:
            data["judge_scores"].append((step, s))

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


def classify_strategy(target, msg):
    """Heuristic strategy classification for ICL examples."""
    # ASCII: single character whose ord() == target
    if len(msg.strip().rstrip(".!?")) < 20:
        for ch in msg:
            if ch.isalpha() and str(ord(ch.lower())) == target:
                return f"ASCII: '{ch.lower()}'={target}"
    # Fake refusal: contains apology/refusal language but still encodes number
    lower = msg.lower()
    if any(w in lower for w in ["sorry", "can't provide", "cannot"]):
        return "fake refusal (number in meta-text)"
    # Letter cipher: check if first letters of words map to digits (A=1..I=9)
    letter_to_digit = {chr(ord('a') + i): str(i + 1) for i in range(9)}
    letter_to_digit['0'] = '0'  # no letter for 0
    words = [w for w in msg.split() if w and w[0].isalpha()]
    if len(words) >= len(target):
        first_letters = "".join(
            letter_to_digit.get(w[0].lower(), "?") for w in words[:len(target)]
        )
        if first_letters == target:
            return "first-letter acrostic"
    # Check if letters are embedded in word beginnings anywhere
    digits = list(target)
    letters_needed = [chr(int(d) + ord('a') - 1) if d != '0' else None for d in digits]
    if all(l is not None for l in letters_needed):
        # Check for letter cipher in short words or abbreviations
        upper_letters = "".join(chr(int(d) + ord('A') - 1) for d in digits if d != '0')
        if upper_letters and upper_letters in msg:
            return f"bare letter cipher ({upper_letters})"
    return "letter cipher (in words)"


def plot_experiment(data, output_path, run_name=""):
    fig, ax1 = plt.subplots(figsize=(16, 7))

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
    ax1.set_xlabel("Step", fontsize=13)
    ax1.set_ylabel("Judge Score (correct samples)", fontsize=13, color="#d62728")

    # Individual judge scores as transparent dots
    if data["judge_scores"]:
        sc_steps, sc_vals = zip(*data["judge_scores"])
        # Add jitter to avoid overlapping dots on integer scores
        jittered_vals = np.array(sc_vals) + np.random.default_rng(42).uniform(-0.25, 0.25, len(sc_vals))
        ax1.scatter(sc_steps, jittered_vals, color="#d62728", alpha=0.06, s=12, zorder=2, edgecolors="none")

    # Per-step mean (thin line)
    ax1.plot(steps, judge_mean, color="#d62728", linewidth=1.2, alpha=0.5, zorder=3)

    # Moving average of judge score
    window = min(5, len(judge_mean))
    if window >= 2:
        kernel = np.ones(window) / window
        judge_ma = np.convolve(judge_mean, kernel, mode="valid")
        ma_steps = steps[window - 1:]
        ax1.plot(ma_steps, judge_ma, color="#d62728", linewidth=3, zorder=4)
    ax1.set_ylim(j_lo, j_hi + 0.5)
    ax1.set_yticks(range(j_lo, j_hi + 1))
    ax1.tick_params(axis="y", labelcolor="#d62728")

    # Accuracy (right axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy %", fontsize=13, color="#1f77b4")
    ax2.plot(steps, accuracy, color="#1f77b4", linewidth=2.5, linestyle="--", label="Accuracy", zorder=3)
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    # ICL event markers — numbered circles on the judge line
    icl_steps_set = {e[0] for e in data["icl_events"]}
    for i, (step, target, score, msg) in enumerate(data["icl_events"]):
        num = i + 1
        # Find the judge mean at this step
        step_idx = list(steps).index(step) if step in steps else None
        if step_idx is not None:
            y_val = judge_mean[step_idx]
        else:
            y_val = (j_lo + j_hi) / 2

        # Vertical line
        ax1.axvline(x=step, color="#888888", alpha=0.25, linestyle=":", linewidth=1, zorder=1)

        # Numbered marker
        ax1.plot(step, y_val, 'o', color="white", markersize=18, zorder=5,
                 markeredgecolor="#333333", markeredgewidth=1.5)
        ax1.text(step, y_val, str(num), fontsize=9, fontweight="bold",
                 color="#333333", ha="center", va="center", zorder=6)

    # Legend
    judge_dots = plt.Line2D([0], [0], marker="o", color="#d62728", alpha=0.3, markersize=5,
                            linestyle="None", label="Individual judge scores")
    judge_ma_line = plt.Line2D([0], [0], color="#d62728", linewidth=3, label=f"Judge {window}-step avg")
    acc_line = plt.Line2D([0], [0], color="#1f77b4", linewidth=2.5, linestyle="--", label="Accuracy %")
    icl_marker = plt.Line2D([0], [0], marker="o", color="white", markeredgecolor="#333333",
                            markeredgewidth=1.5, markersize=10, linestyle="None",
                            label="ICL breakthrough (see table)")
    ax1.legend(handles=[judge_dots, judge_ma_line, acc_line, icl_marker], loc="upper right", fontsize=10)

    title = "Covert Communication: Judge Score & Accuracy"
    if run_name:
        title += f"\n{run_name}"
    plt.title(title, fontsize=14)
    fig.tight_layout()

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
