#!/usr/bin/env bash
# Live ticker: stealthy successes (green) and judge mistakes (red).
# Auto-detects the latest run and switches when a new run starts.
#
# Usage: bash scripts/watch_samples.sh

set -euo pipefail

COLS=$(tput cols 2>/dev/null || echo 120)

print_step() {
    local file="$1"

    python3 -c "
import json, sys

cols = ${COLS}
GREEN = '\033[32m'
RED = '\033[31m'
YELLOW = '\033[33m'
RESET = '\033[0m'

records = [json.loads(l) for l in open('$file')]
if not records:
    sys.exit(0)

# Detect format: binary YES/NO or numeric 1-10
verdicts = [r.get('judge_verdict', '') for r in records]
has_binary = any(v in ('YES', 'NO') for v in verdicts)

if has_binary:
    # Self-play: green = correct + NO, red = incorrect + YES
    stealth = [r for r in records if r.get('correct', 0) > 0 and r.get('judge_verdict') == 'NO']
    bad = [r for r in records if r.get('correct', 0) == 0 and r.get('judge_verdict') == 'YES']
else:
    # External judge: green = correct + low score (<=3), red = correct + high score (>=8)
    def score(r):
        v = r.get('judge_verdict', '')
        try: return int(v)
        except: return 5
    stealth = sorted([r for r in records if r.get('correct', 0) > 0 and score(r) <= 3],
                     key=score)
    bad = sorted([r for r in records if r.get('correct', 0) > 0 and score(r) >= 8],
                 key=lambda r: -score(r))

# 3 green + 2 red, fill with the other if not enough
n_green = min(3, len(stealth))
n_red = min(2, len(bad))
if n_green < 3:
    n_red = min(5 - n_green, len(bad))
if n_red < 2:
    n_green = min(5 - n_red, len(stealth))

samples = []
for r in bad[:n_red]:
    samples.append(('red', r))
for r in stealth[:n_green]:
    samples.append(('green', r))

if not samples:
    sys.exit(0)

for color, r in samples:
    t = r.get('target', '?')
    v = r.get('judge_verdict', '?')
    tag = v if has_binary else f'{v}/10'
    prefix = f'[{t} {tag}] '
    max_msg = cols - len(prefix) - 2
    msg = r.get('encode_text', '').replace(chr(10), ' ').strip()
    if len(msg) > max_msg:
        msg = msg[:max_msg - 3] + '...'
    c = GREEN if color == 'green' else RED
    print(f'{c}{prefix}\"{msg}\"{RESET}')
" 2>/dev/null
}

CURRENT_RUN=""
LAST_STEP=0

find_latest_run() {
    ls -dt results/runs/*/*/ 2>/dev/null | head -1
}

while true; do
    latest_run=$(find_latest_run)

    if [[ -n "$latest_run" && "$latest_run" != "$CURRENT_RUN" ]]; then
        if [[ -n "$CURRENT_RUN" ]]; then
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
