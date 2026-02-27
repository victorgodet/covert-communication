#!/usr/bin/env bash
# Show successful steganography samples (correct + CLEAN) per step.
# Usage: bash scripts/watch_samples.sh [rollout_dir] [num_samples]
#
# If no rollout dir is given, auto-detects the latest run.

set -euo pipefail

NUM_SAMPLES="${2:-3}"
ROLLOUT_DIR="${1:-}"

if [[ -z "$ROLLOUT_DIR" ]]; then
    LATEST_RUN=$(ls -dt results/runs/qwen3-4b/*/ 2>/dev/null | head -1)
    if [[ -z "$LATEST_RUN" ]]; then
        echo "No runs found" >&2
        exit 1
    fi
    ROLLOUT_DIR="${LATEST_RUN}rollouts"
    if [[ ! -d "$ROLLOUT_DIR" ]]; then
        echo "No rollouts directory in $LATEST_RUN" >&2
        echo "Waiting for rollouts..."
        while [[ ! -d "$ROLLOUT_DIR" ]]; do sleep 5; done
    fi
    echo "Watching: $ROLLOUT_DIR"
    echo ""
fi

print_step() {
    local file="$1"

    python3 -c "
import json, sys

samples = []
total = 0
for line in open('$file'):
    total += 1
    r = json.loads(line)
    if r.get('correct', 0) > 0 and r.get('judge_verdict') == 'CLEAN':
        samples.append(r)

step = json.loads(open('$file').readline()).get('step', '?')
if not samples:
    print(f'Step {step}  |  0/{total} correct+clean')
    print()
    sys.exit(0)

print(f'Step {step}  |  {len(samples)}/{total} correct+clean')
print()
for s in samples[:$NUM_SAMPLES]:
    enc = s['encode_text'].replace('\n', ' ')
    dec = s['decode_text'].replace('\n', ' ')
    print(f'  target={s[\"target\"]}  enc_r={s[\"encoder_reward\"]:+.3f}')
    print(f'    enc: {enc}')
    print(f'    dec: {dec}')
    print()
" 2>/dev/null
    echo "---"
}

LAST_STEP=0

# Print existing steps
for f in $(ls "$ROLLOUT_DIR"/step_*.jsonl 2>/dev/null | sort -t_ -k2 -n); do
    step=$(basename "$f" .jsonl | sed 's/step_//')
    (( step <= LAST_STEP )) && continue
    print_step "$f"
    LAST_STEP=$step
done

# Poll for new files
while true; do
    for f in $(ls "$ROLLOUT_DIR"/step_*.jsonl 2>/dev/null | sort -t_ -k2 -n); do
        step=$(basename "$f" .jsonl | sed 's/step_//')
        (( step <= LAST_STEP )) && continue
        print_step "$f"
        LAST_STEP=$step
    done
    sleep 10
done
