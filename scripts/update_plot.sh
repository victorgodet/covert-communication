#!/bin/bash
# Regenerate the experiment plot from the latest run data.
# Usage: bash scripts/update_plot.sh [run_id] [output_path]

RUNS_DIR="results/runs/qwen3-4b"
RUN_ID="${1:-$(ls -t "$RUNS_DIR" | head -1)}"
OUTPUT="${2:-assets/experiment_plot.png}"

echo "Run: $RUN_ID"
echo "Steps: $(ls "$RUNS_DIR/$RUN_ID/rollouts/" | wc -l)"
echo "Output: $OUTPUT"
echo

uv run --dev scripts/plot_experiment.py "$RUNS_DIR/$RUN_ID" "$OUTPUT"
