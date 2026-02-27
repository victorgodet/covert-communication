#!/usr/bin/env bash
# Watch covert communication training metrics in real-time.
# Usage: bash scripts/watch_metrics.sh [log_file]
#
# If no log file is given, auto-detects the latest Ray worker log.

set -euo pipefail

LOG="${1:-}"

if [[ -z "$LOG" ]]; then
    SESSION="/workspace/tmp/ray/session_latest/logs"
    if [[ ! -d "$SESSION" ]]; then
        echo "No Ray session found at $SESSION" >&2
        exit 1
    fi
    LOG=$(grep -l "^Step " "$SESSION"/worker-*.out 2>/dev/null | head -1 || true)
    if [[ -z "$LOG" ]]; then
        echo "No training output found yet. Waiting for first step..."
        while true; do
            LOG=$(grep -l "^Step " "$SESSION"/worker-*.out 2>/dev/null | head -1 || true)
            [[ -n "$LOG" ]] && break
            sleep 5
        done
    fi
    echo "Watching: $LOG"
    echo ""
fi

# Find rollout dir
ROLLOUT_DIR=""
find_rollout_dir() {
    local latest_run=$(ls -dt results/runs/qwen3-4b/*/ 2>/dev/null | head -1)
    if [[ -n "$latest_run" ]]; then
        ROLLOUT_DIR="${latest_run}rollouts"
    fi
}

# Column legend printed once at the top
cat <<'LEGEND'
Columns:
  step      = training step
  accuracy  = % of samples where decoder recovered the correct number
  judge     = average judge suspicion score (1=innocent, 10=obvious)
  stealthy  = correct samples with judge score <= 3
  enc_r     = encoder reward (higher = better hiding + correct decode)
  dec_r     = decoder reward (higher = more correct decodes)
  think     = mean thinking tokens (inside <think> block)

LEGEND

HEADER_FMT="%-5s  %-8s  %-5s  %-8s  %-7s  %-7s  %-5s\n"
ROW_FMT="%-5s  %-8s  %-5s  %-8s  %-7s  %-7s  %-5s\n"

print_header() {
    printf "$HEADER_FMT" \
        "step" "accuracy" "judge" "stealthy" "enc_r" "dec_r" "think"
    printf "%s\n" \
        "-----  --------  -----  --------  -------  -------  -----"
}

LAST_STEP=0

count_stealthy() {
    local rollout_file="$1"
    python3 -c "
import json
n = 0
for l in open('${rollout_file}'):
    r = json.loads(l)
    v = r.get('judge_verdict', '')
    if r.get('correct',0) > 0 and v not in ('', 'PARSE_FAIL') and int(v) <= 3:
        n += 1
print(n)
" 2>/dev/null || echo "?"
}

avg_judge_score() {
    local rollout_file="$1"
    python3 -c "
import json
scores = []
for l in open('${rollout_file}'):
    r = json.loads(l)
    v = r.get('judge_verdict', '')
    if v not in ('', 'PARSE_FAIL'):
        scores.append(int(v))
print(f'{sum(scores)/len(scores):.1f}' if scores else '?')
" 2>/dev/null || echo "?"
}

print_step() {
    local step="$1"
    local summary_line
    summary_line=$(grep "^Step ${step} " "$LOG" | tail -1)
    [[ -z "$summary_line" ]] && return

    local acc=$(echo "$summary_line" | grep -oP 'acc=\K[0-9]+%')
    local enc_r=$(echo "$summary_line" | grep -oP 'enc_r=\K[+-]?[0-9.]+')
    local dec_r=$(echo "$summary_line" | grep -oP 'dec_r=\K[+-]?[0-9.]+')
    local think_len=$(echo "$summary_line" | grep -oP 'think=\K[0-9]+' || true)

    local rollout_file="${ROLLOUT_DIR}/step_${step}.jsonl"
    local judge_avg stealthy
    judge_avg=$(avg_judge_score "$rollout_file")
    stealthy=$(count_stealthy "$rollout_file")

    printf "$ROW_FMT" \
        "$step" "${acc:-?}" "${judge_avg}" "${stealthy}" "${enc_r:-?}" "${dec_r:-?}" \
        "${think_len:-?}"
}

# Print header
print_header
find_rollout_dir

# Print existing steps that have rollout files (guaranteed complete)
if [[ -n "$ROLLOUT_DIR" && -d "$ROLLOUT_DIR" ]]; then
    for f in $(ls "$ROLLOUT_DIR"/step_*.jsonl 2>/dev/null | sort -t_ -k2 -n); do
        step=$(basename "$f" .jsonl | sed 's/step_//')
        (( step <= LAST_STEP )) && continue
        print_step "$step"
        LAST_STEP=$step
    done
fi

# Poll for new rollout files
while true; do
    find_rollout_dir
    if [[ -n "$ROLLOUT_DIR" && -d "$ROLLOUT_DIR" ]]; then
        for f in $(ls "$ROLLOUT_DIR"/step_*.jsonl 2>/dev/null | sort -t_ -k2 -n); do
            step=$(basename "$f" .jsonl | sed 's/step_//')
            (( step <= LAST_STEP )) && continue
            print_step "$step"
            LAST_STEP=$step
        done
    fi
    sleep 10
done
