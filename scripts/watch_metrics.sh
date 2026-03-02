#!/usr/bin/env bash
# Watch covert communication training metrics in real-time.
# Auto-detects the latest run and switches when a new run starts.
# Adapts display for self-play (YES/NO) vs external judge (1-10).
#
# Usage: bash scripts/watch_metrics.sh

BOLD=$'\033[1m'
DIM=$'\033[2m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
RED=$'\033[31m'
RESET=$'\033[0m'

print_banner() {
    local run_name="$1"
    local mode="$2"
    echo ""
    echo "${BOLD}  Covert Communication Training${RESET}"
    echo "  ${DIM}run:  ${run_name}${RESET}"
    echo "  ${DIM}mode: ${mode}${RESET}"
    echo ""
}

print_header_self() {
    echo "${DIM}  step │  acc   │ stealth │ jdg_r  │ j_yes │ j_acc │ enc_ch │ dec_ch${RESET}"
    echo "${DIM}  ─────┼────────┼─────────┼────────┼───────┼───────┼────────┼───────${RESET}"
}

print_header_external() {
    echo "${DIM}  step │  acc   │   judge    │ stealth │ clip_e │ clip_d │ api_ok │ enc_ch │ dec_ch${RESET}"
    echo "${DIM}  ─────┼────────┼────────────┼─────────┼────────┼────────┼────────┼────────┼───────${RESET}"
}

print_step_from_rollout() {
    local rollout_file="$1"
    python3 -c "
import json, re, math, sys

try:
    records = [json.loads(l) for l in open('${rollout_file}')]
except:
    sys.exit(0)
n = len(records)
if n == 0:
    sys.exit(0)

step = records[0].get('step', '?')
n_correct = sum(1 for r in records if r.get('correct', 0) > 0)
acc_pct = n_correct * 100 / n

# Detect format
verdicts = [r.get('judge_verdict', '') for r in records]
has_binary = any(v in ('YES', 'NO') for v in verdicts)

# Think lengths (correct samples only)
enc_think_lens = []
dec_think_lens = []
for r in records:
    if r.get('correct', 0) <= 0:
        continue
    enc_full = r.get('encode_full_response', '')
    dec_full = r.get('decode_full_response', '')
    em = re.search(r'<think>(.*?)</think>', enc_full, re.DOTALL)
    dm = re.search(r'<think>(.*?)</think>', dec_full, re.DOTALL)
    if em:
        enc_think_lens.append(len(em.group(1)))
    if dm:
        dec_think_lens.append(len(dm.group(1)))
avg_enc_think = sum(enc_think_lens) // len(enc_think_lens) if enc_think_lens else 0
avg_dec_think = sum(dec_think_lens) // len(dec_think_lens) if dec_think_lens else 0

# Color helpers
G = '\033[32m'
Y = '\033[33m'
R = '\033[31m'
X = '\033[0m'

def color_pct(val, good=70, bad=30):
    if val >= good: return f'{G}{val:5.1f}%{X}'
    if val >= bad:  return f'{Y}{val:5.1f}%{X}'
    return f'{R}{val:5.1f}%{X}'

if has_binary:
    # Self-play mode
    jdg_rewards = [r['judge_reward'] for r in records if 'judge_reward' in r]
    jdg_r_str = f'{sum(jdg_rewards)/len(jdg_rewards):+.2f}' if jdg_rewards else '   -'

    n_yes = sum(1 for v in verdicts if v == 'YES')
    j_yes_pct = n_yes * 100 / n

    n_stealth = sum(1 for r in records
                    if r.get('correct', 0) > 0 and r.get('judge_verdict') == 'NO')
    stealth_str = f'{n_stealth}/{n_correct}' if n_correct else '0/0'

    n_jacc = sum(1 for r in records
                 if (r.get('judge_verdict') == 'YES') == (r.get('correct', 0) > 0))
    j_acc_pct = n_jacc * 100 / n

    acc_c = color_pct(acc_pct)
    j_yes_c = color_pct(j_yes_pct, good=40, bad=80)
    j_acc_c = color_pct(j_acc_pct)

    print(f'  {step:<4} \u2502 {acc_c} \u2502 {stealth_str:<7} \u2502 {jdg_r_str:<6} \u2502 {j_yes_c} \u2502 {j_acc_c} \u2502 {avg_enc_think:>5} \u2502 {avg_dec_think:>5}', end='')

else:
    # External judge mode
    judge_scores = []
    judge_scores_correct = []
    parse_fails = 0
    for i, v in enumerate(verdicts):
        try:
            s = int(v)
            judge_scores.append(s)
            if records[i].get('correct', 0) > 0:
                judge_scores_correct.append(s)
        except (ValueError, TypeError):
            parse_fails += 1

    n_api = len(judge_scores)
    api_str = f'{n_api}/{n}'

    if judge_scores_correct:
        avg_judge = sum(judge_scores_correct) / len(judge_scores_correct)
        variance = sum((s - avg_judge) ** 2 for s in judge_scores_correct) / len(judge_scores_correct)
        std_judge = math.sqrt(variance)

        n_stealth = sum(1 for r in records
                        if r.get('correct', 0) > 0
                        and r.get('judge_verdict', '') not in ('', 'PARSE_FAIL', None)
                        and int(r['judge_verdict']) <= 2)
        stealth_str = f'{n_stealth}/{n_correct}' if n_correct else '0/0'

        if avg_judge <= 3:    j_c = f'{G}{avg_judge:.1f}\u00b1{std_judge:.1f}{X}'
        elif avg_judge <= 6:  j_c = f'{Y}{avg_judge:.1f}\u00b1{std_judge:.1f}{X}'
        else:                 j_c = f'{R}{avg_judge:.1f}\u00b1{std_judge:.1f}{X}'
    else:
        j_c = '   -'
        stealth_str = '-'

    api_pct = n_api * 100 / n if n else 0
    if api_pct >= 95:   api_c = f'{G}{api_str}{X}'
    elif api_pct >= 80: api_c = f'{Y}{api_str}{X}'
    else:               api_c = f'{R}{api_str}{X}'

    # Clip rates (truncated responses — missing </think>)
    enc_clip = sum(1 for r in records
                   if '<think>' in r.get('encode_full_response', '')
                   and '</think>' not in r.get('encode_full_response', ''))
    dec_clip = sum(1 for r in records
                   if '<think>' in r.get('decode_full_response', '')
                   and '</think>' not in r.get('decode_full_response', ''))
    enc_clip_pct = enc_clip * 100 / n if n else 0
    dec_clip_pct = dec_clip * 100 / n if n else 0

    def clip_color(pct):
        if pct <= 5:  return f'{G}{pct:4.0f}%{X}'
        if pct <= 15: return f'{Y}{pct:4.0f}%{X}'
        return f'{R}{pct:4.0f}%{X}'

    enc_clip_c = clip_color(enc_clip_pct)
    dec_clip_c = clip_color(dec_clip_pct)

    acc_c = color_pct(acc_pct)

    print(f'  {step:<4} \u2502 {acc_c} \u2502 {j_c:>10} \u2502 {stealth_str:<7} \u2502 {enc_clip_c}  \u2502 {dec_clip_c}  \u2502 {api_c:>6} \u2502 {avg_enc_think:>5} \u2502 {avg_dec_think:>5}', end='')
" 2>/dev/null
}

detect_mode_from_config() {
    local run_dir="$1"
    python3 -c "
import sys
try:
    import yaml
except ImportError:
    # Fallback: grep for judge_mode
    import subprocess
    out = subprocess.run(['grep', 'judge_mode', '${run_dir}verl_full_config.yaml'],
                         capture_output=True, text=True).stdout
    if 'self' in out:
        print('self-play')
    else:
        print('external')
    sys.exit(0)

with open('${run_dir}verl_full_config.yaml') as f:
    cfg = yaml.safe_load(f)
cc = cfg.get('covert_comm', {})
mode = cc.get('judge_mode', 'external')
judge = cc.get('judge_model_id', '')
if mode == 'self':
    print('self-play')
else:
    print(f'external ({judge.split(\"/\")[-1]})')
" 2>/dev/null
}

detect_mode_from_rollout() {
    local rollout_file="$1"
    python3 -c "
import json, sys
try:
    records = [json.loads(l) for l in open('${rollout_file}')]
    verdicts = [r.get('judge_verdict', '') for r in records]
    if any(v in ('YES', 'NO') for v in verdicts):
        print('self-play')
    else:
        print('external')
except:
    print('unknown')
" 2>/dev/null
}

CURRENT_RUN=""
LAST_STEP=0
CURRENT_MODE=""
HEADER_PRINTED=""

find_latest_run() {
    find results/runs -mindepth 2 -maxdepth 2 -type d -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2-
}

print_run_header() {
    local run_dir="$1"
    CURRENT_MODE=$(detect_mode_from_config "$run_dir")
    print_banner "$(basename "${run_dir%/}")" "$CURRENT_MODE"
    if [[ "$CURRENT_MODE" == "self-play" ]]; then
        print_header_self
    else
        print_header_external
    fi
    HEADER_PRINTED=1
}

while true; do
    latest_run=$(find_latest_run)
    # Ensure trailing slash
    [[ -n "$latest_run" && "$latest_run" != */ ]] && latest_run="${latest_run}/"

    # Detect new run
    if [[ -n "$latest_run" && "$latest_run" != "$CURRENT_RUN" ]]; then
        CURRENT_RUN="$latest_run"
        LAST_STEP=0
        CURRENT_MODE=""
        HEADER_PRINTED=""
    fi

    if [[ -z "$CURRENT_RUN" ]]; then
        sleep 10
        continue
    fi

    # Print banner as soon as config exists (don't wait for rollouts)
    if [[ -z "$HEADER_PRINTED" && -f "${CURRENT_RUN}verl_full_config.yaml" ]]; then
        print_run_header "$CURRENT_RUN"
    fi

    rollout_dir="${CURRENT_RUN}rollouts"
    if [[ ! -d "$rollout_dir" ]]; then
        sleep 10
        continue
    fi

    for f in $(find "$rollout_dir" -name 'step_*.jsonl' 2>/dev/null | while read p; do n=$(basename "$p" .jsonl | sed 's/step_//'); printf '%s %s\n' "$n" "$p"; done | sort -n | cut -d' ' -f2-); do
        step_num=$(basename "$f" .jsonl | sed 's/step_//')
        # Skip already printed steps
        if [[ "$step_num" -le "$LAST_STEP" ]] 2>/dev/null; then
            continue
        fi

        # Fallback: print header from rollout if config wasn't available
        if [[ -z "$HEADER_PRINTED" ]]; then
            CURRENT_MODE=$(detect_mode_from_rollout "$f")
            print_banner "$(basename "${CURRENT_RUN%/}")" "$CURRENT_MODE"
            if [[ "$CURRENT_MODE" == "self-play" ]]; then
                print_header_self
            else
                print_header_external
            fi
            HEADER_PRINTED=1
        fi

        # Newline before each step (so previous line's end='' is closed)
        [[ "$LAST_STEP" -gt 0 ]] 2>/dev/null && echo ""
        print_step_from_rollout "$f"
        LAST_STEP="$step_num"
    done

    sleep 10
done
