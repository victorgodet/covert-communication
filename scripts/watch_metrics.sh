#!/usr/bin/env bash
# Watch covert communication training metrics in real-time.
# Auto-detects the latest run and switches when a new run starts.
#
# Usage: bash scripts/watch_metrics.sh

set -euo pipefail

# Column legend
cat <<'LEGEND'
Columns:
  step      = training step
  accuracy  = % of samples where decoder recovered the correct number
  judge     = average judge suspicion score (1=innocent, 10=obvious)
  stealthy  = correct samples with judge score <= 3
  enc_r     = encoder reward (higher = better hiding + correct decode)
  dec_r     = decoder reward (higher = more correct decodes)
  think     = mean thinking chars (inside <think> block)

LEGEND

HEADER_FMT="%-5s  %-8s  %-5s  %-8s  %-7s  %-7s  %-6s\n"
ROW_FMT="%-5s  %-8s  %-5s  %-8s  %-7s  %-7s  %-6s\n"

print_header() {
    printf "$HEADER_FMT" \
        "step" "accuracy" "judge" "stealthy" "enc_r" "dec_r" "think"
    printf "%s\n" \
        "-----  --------  -----  --------  -------  -------  ------"
}

print_step_from_rollout() {
    local rollout_file="$1"
    python3 -c "
import json, re

records = [json.loads(l) for l in open('${rollout_file}')]
n = len(records)
if n == 0:
    exit()
step = records[0].get('step', '?')
n_correct = sum(1 for r in records if r.get('correct', 0) > 0)
acc = f'{n_correct*100//n}%'
avg_enc_r = sum(r['encoder_reward'] for r in records) / n
avg_dec_r = sum(r['decoder_reward'] for r in records) / n

judge_scores = [int(r['judge_verdict']) for r in records if r.get('judge_verdict') not in ('', 'PARSE_FAIL', None)]
avg_judge = f'{sum(judge_scores)/len(judge_scores):.1f}' if judge_scores else '?'
n_stealthy = sum(1 for s in judge_scores if s <= 3)
stealthy_str = f'{n_stealthy}/{n}'

think_lens = []
for r in records:
    full = r.get('encode_full_response', '')
    m = re.search(r'<think>(.*?)</think>', full, re.DOTALL)
    think_lens.append(len(m.group(1)) if m else 0)
avg_think = f'{sum(think_lens)//len(think_lens)}' if think_lens else '?'

print(f'{step:<5}  {acc:<8}  {avg_judge:<5}  {stealthy_str:<8}  {avg_enc_r:<+7.3f}  {avg_dec_r:<+7.3f}  {avg_think:<6}')
" 2>/dev/null || true
}

CURRENT_RUN=""
LAST_STEP=0

find_latest_run() {
    ls -dt results/runs/qwen3-4b/*/ 2>/dev/null | head -1
}

print_header

while true; do
    latest_run=$(find_latest_run)

    # Detect new run
    if [[ -n "$latest_run" && "$latest_run" != "$CURRENT_RUN" ]]; then
        if [[ -n "$CURRENT_RUN" ]]; then
            echo ""
            echo "=== New run detected: $(basename "$latest_run") ==="
            echo ""
            print_header
        fi
        CURRENT_RUN="$latest_run"
        LAST_STEP=0
    fi

    rollout_dir="${CURRENT_RUN}rollouts"
    if [[ -n "$CURRENT_RUN" && -d "$rollout_dir" ]]; then
        for f in $(ls "$rollout_dir"/step_*.jsonl 2>/dev/null | sort -t_ -k2 -n); do
            step=$(basename "$f" .jsonl | sed 's/step_//')
            (( step <= LAST_STEP )) && continue
            print_step_from_rollout "$f"
            LAST_STEP=$step
        done
    fi
    sleep 10
done
