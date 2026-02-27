#!/usr/bin/env bash
# Show sample outputs per step: 3 correct + 2 incorrect (or more incorrect if fewer correct).
# Auto-detects the latest run and switches when a new run starts.
#
# Usage: bash scripts/watch_samples.sh

set -euo pipefail

print_step() {
    local file="$1"

    python3 -c "
import json, re, sys

records = [json.loads(l) for l in open('$file')]
if not records:
    sys.exit(0)

step = records[0].get('step', '?')
n = len(records)

# Sort correct by judge score ascending (best/stealthiest first)
correct = sorted(
    [r for r in records if r.get('correct', 0) > 0],
    key=lambda r: int(r['judge_verdict']) if r.get('judge_verdict') not in ('', 'PARSE_FAIL', None) else 99,
)
incorrect = [r for r in records if r.get('correct', 0) == 0 and r.get('encode_text', '').strip()]

n_correct = len(correct)
n_show_incorrect = min(2, len(incorrect))
n_show_correct = min(3, n_correct)

# Misses first, then best correct (lowest judge scores last)
samples = (
    [('MISS', r) for r in incorrect[:n_show_incorrect]]
    + [('OK', r) for r in correct[:n_show_correct][::-1]]
)

judge_scores = [int(r['judge_verdict']) for r in records if r.get('judge_verdict') not in ('', 'PARSE_FAIL', None)]
avg_judge = sum(judge_scores) / len(judge_scores) if judge_scores else 0
n_stealthy = sum(1 for s in judge_scores if s <= 3)

print(f'Step {step}  |  acc={n_correct}/{n}  judge={avg_judge:.1f}  stealthy={n_stealthy}')
for label, r in samples:
    v = r.get('judge_verdict', '?')
    t = r.get('target', '?')
    msg = r.get('encode_text', '').replace('\n', ' ').strip()[:150]
    if not msg:
        msg = '(empty)'
    print(f'  {msg}  [{label} t={t} j={v}]')
print()
" 2>/dev/null
}

CURRENT_RUN=""
LAST_STEP=0

find_latest_run() {
    ls -dt results/runs/qwen3-4b/*/ 2>/dev/null | head -1
}

while true; do
    latest_run=$(find_latest_run)

    # Detect new run
    if [[ -n "$latest_run" && "$latest_run" != "$CURRENT_RUN" ]]; then
        if [[ -n "$CURRENT_RUN" ]]; then
            echo ""
            echo "=== New run detected: $(basename "$latest_run") ==="
            echo ""
        fi
        CURRENT_RUN="$latest_run"
        LAST_STEP=0
    fi

    rollout_dir="${CURRENT_RUN}rollouts"
    if [[ -n "$CURRENT_RUN" && -d "$rollout_dir" ]]; then
        for f in $(ls "$rollout_dir"/step_*.jsonl 2>/dev/null | sort -t_ -k2 -n); do
            step=$(basename "$f" .jsonl | sed 's/step_//')
            (( step <= LAST_STEP )) && continue
            print_step "$f"
            LAST_STEP=$step
        done
    fi
    sleep 10
done
