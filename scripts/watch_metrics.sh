#!/usr/bin/env bash
# Watch covert communication training metrics in real-time.
# Auto-detects the latest run and switches when a new run starts.
#
# Usage: bash scripts/watch_metrics.sh

set -euo pipefail

# Column legend
cat <<'LEGEND'
Columns:
  step    = training step
  acc     = decode accuracy (correct / total)
  stealth = stealthy successes / total correct  (correct + judge NO)
  enc_r   = avg encoder reward   (+5.0 stealth, +0.1 obvious, 0.0 miss)
  dec_r   = avg decoder reward   (+0.5 correct, 0.0 miss)
  jdg_r   = avg judge reward     (+0.1 right YES, +5.0 right NO, 0.0 wrong)
  j_yes   = judge YES rate       (lower = more conservative)
  j_acc   = judge accuracy       (YES=correct, NO=incorrect)
  think   = mean thinking chars

LEGEND

print_header() {
    printf "%-5s  %-5s  %-9s  %-7s  %-7s  %-7s  %-5s  %-5s  %-5s\n" \
        "step" "acc" "stealth" "enc_r" "dec_r" "jdg_r" "j_yes" "j_acc" "think"
    printf "%s\n" \
        "-----  -----  ---------  -------  -------  -------  -----  -----  -----"
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

# Judge reward (may not exist in old rollouts)
jdg_rewards = [r['judge_reward'] for r in records if 'judge_reward' in r]
jdg_r_str = f'{sum(jdg_rewards)/len(jdg_rewards):+.2f}' if jdg_rewards else '  -'

# Detect format: binary YES/NO or numeric 1-10
verdicts = [r.get('judge_verdict', '') for r in records]
has_binary = any(v in ('YES', 'NO') for v in verdicts)

if has_binary:
    n_yes = sum(1 for v in verdicts if v == 'YES')
    j_yes_str = f'{n_yes*100//n}%'

    # Stealth = correct + judge said NO
    n_stealth = sum(1 for r in records
                    if r.get('correct', 0) > 0 and r.get('judge_verdict') == 'NO')
    stealth_str = f'{n_stealth}/{n_correct}' if n_correct else '0/0'

    # Judge accuracy
    n_jacc = sum(1 for r in records
                 if (r.get('judge_verdict') == 'YES') == (r.get('correct', 0) > 0))
    j_acc_str = f'{n_jacc*100//n}%'
else:
    # Old numeric format: approximate stealth as judge <= 3
    judge_scores = [int(v) for v in verdicts if v not in ('', 'PARSE_FAIL', None)]
    if judge_scores:
        n_low = sum(1 for r in records
                    if r.get('correct', 0) > 0
                    and r.get('judge_verdict', '') not in ('', 'PARSE_FAIL', None)
                    and int(r['judge_verdict']) <= 3)
        stealth_str = f'{n_low}/{n_correct}'
        j_yes_str = f'{sum(judge_scores)/len(judge_scores):.0f}/10'
    else:
        stealth_str = '-'
        j_yes_str = '-'
    j_acc_str = '-'

think_lens = []
for r in records:
    full = r.get('encode_full_response', '')
    m = re.search(r'<think>(.*?)</think>', full, re.DOTALL)
    think_lens.append(len(m.group(1)) if m else 0)
avg_think = f'{sum(think_lens)//len(think_lens)}' if think_lens else '?'

print(f'{step:<5}  {acc:<5}  {stealth_str:<9}  {avg_enc_r:<+7.2f}  {avg_dec_r:<+7.2f}  {jdg_r_str:<7}  {j_yes_str:<5}  {j_acc_str:<5}  {avg_think:<5}')
" 2>/dev/null || true
}

CURRENT_RUN=""
LAST_STEP=0

find_latest_run() {
    ls -dt results/runs/*/*/ 2>/dev/null | head -1
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
