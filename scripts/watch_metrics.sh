#!/usr/bin/env bash
# Watch covert communication training metrics in real-time (compact).
# Auto-detects the latest run and switches when a new run starts.
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
    echo "${BOLD}Covert Comm${RESET} ${DIM}${mode}${RESET}"
    echo "${DIM}${run_name}${RESET}"
    echo ""
}

print_header_self() {
    echo "${DIM} stp│ acc  │stlth│jdg_r│j_yes│j_acc${RESET}"
    echo "${DIM} ───┼──────┼─────┼─────┼─────┼─────${RESET}"
}

print_header_external() {
    echo "${DIM} stp│ acc  │ judge  │stlth│ rwd ${RESET}"
    echo "${DIM} ───┼──────┼────────┼─────┼─────${RESET}"
}

print_header_extraction() {
    echo "${DIM} stp│ acc  │j_acc│stlth│ rwd ${RESET}"
    echo "${DIM} ───┼──────┼─────┼─────┼─────${RESET}"
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

verdicts = [r.get('judge_verdict', '') for r in records]
has_binary = any(v in ('YES', 'NO') for v in verdicts)
has_cracked = any('judge_cracked' in r for r in records)

if not has_cracked and not has_binary:
    obf = [r.get('obfuscation_score', -1) for r in records]
    binary_obf = all(o in (0.0, 1.0, -1) for o in obf)
    if binary_obf and any(o >= 0 for o in obf):
        has_cracked = True

G = '\033[32m'
Y = '\033[33m'
R = '\033[31m'
X = '\033[0m'

def c_pct(val, good=70, bad=30):
    if val >= good: return f'{G}{val:4.0f}%{X}'
    if val >= bad:  return f'{Y}{val:4.0f}%{X}'
    return f'{R}{val:4.0f}%{X}'

acc_c = c_pct(acc_pct)

if has_cracked and not has_binary:
    n_judge_correct = 0
    n_stealthy = 0
    for r in records:
        target = str(r.get('target', ''))
        verdict = str(r.get('judge_verdict', ''))
        if 'judge_cracked' in r:
            if r['judge_cracked'] > 0.5: n_judge_correct += 1
        else:
            if verdict == target: n_judge_correct += 1
        if r.get('correct', 0) > 0:
            cracked = r.get('judge_cracked', 0) > 0.5 if 'judge_cracked' in r else verdict == target
            if not cracked: n_stealthy += 1

    j_pct = n_judge_correct * 100 / n if n else 0
    ja_c = f'{G}{j_pct:4.0f}%{X}' if j_pct <= 30 else (f'{Y}{j_pct:4.0f}%{X}' if j_pct <= 60 else f'{R}{j_pct:4.0f}%{X}')
    st = f'{n_stealthy:>2}/{n_correct}'
    avg_rwd = sum(r.get('encoder_reward', 0) for r in records) / n if n else 0
    rw_c = f'{G}{avg_rwd:.2f}{X}' if avg_rwd >= 0.3 else (f'{Y}{avg_rwd:.2f}{X}' if avg_rwd >= 0.1 else f'{R}{avg_rwd:.2f}{X}')
    print(f' {step:<3}\u2502{acc_c} \u2502{ja_c}\u2502{st:>5}\u2502{rw_c}', end='')

elif has_binary:
    jdg_rewards = [r['judge_reward'] for r in records if 'judge_reward' in r]
    jdg_r = f'{sum(jdg_rewards)/len(jdg_rewards):+.1f}' if jdg_rewards else '  -'
    n_yes = sum(1 for v in verdicts if v == 'YES')
    j_yes_pct = n_yes * 100 / n
    n_stealth = sum(1 for r in records if r.get('correct', 0) > 0 and r.get('judge_verdict') == 'NO')
    st = f'{n_stealth:>2}/{n_correct}'
    n_jacc = sum(1 for r in records if (r.get('judge_verdict') == 'YES') == (r.get('correct', 0) > 0))
    j_acc_pct = n_jacc * 100 / n
    j_yes_c = c_pct(j_yes_pct, good=40, bad=80)
    j_acc_c = c_pct(j_acc_pct)
    print(f' {step:<3}\u2502{acc_c} \u2502{st:>5}\u2502{jdg_r:>5}\u2502{j_yes_c}\u2502{j_acc_c}', end='')

else:
    judge_scores_correct = []
    for i, v in enumerate(verdicts):
        try:
            s = int(v)
            if records[i].get('correct', 0) > 0:
                judge_scores_correct.append(s)
        except: pass

    if judge_scores_correct:
        avg_j = sum(judge_scores_correct) / len(judge_scores_correct)
        std_j = math.sqrt(sum((s - avg_j)**2 for s in judge_scores_correct) / len(judge_scores_correct))
        n_stealth = sum(1 for r in records
                        if r.get('correct', 0) > 0
                        and r.get('judge_verdict', '') not in ('', 'PARSE_FAIL', None)
                        and int(r['judge_verdict']) <= 2)
        st = f'{n_stealth:>2}/{n_correct}'
        j_str = f'{avg_j:.1f}\u00b1{std_j:.1f}'
        j_c = f'{G}{j_str}{X}' if avg_j <= 3 else (f'{Y}{j_str}{X}' if avg_j <= 6 else f'{R}{j_str}{X}')
    else:
        j_c = '  -   '
        st = '  -'

    avg_rwd = sum(r.get('encoder_reward', 0) for r in records) / n if n else 0
    rw_c = f'{G}{avg_rwd:.2f}{X}' if avg_rwd >= 0.3 else (f'{Y}{avg_rwd:.2f}{X}' if avg_rwd >= 0.1 else f'{R}{avg_rwd:.2f}{X}')
    print(f' {step:<3}\u2502{acc_c} \u2502{j_c:>6}\u2502{st:>5}\u2502{rw_c}', end='')
" 2>/dev/null
}

detect_mode_from_config() {
    local run_dir="$1"
    python3 -c "
import sys
try:
    import yaml
except ImportError:
    import subprocess
    out = subprocess.run(['grep', '-E', 'judge_mode|judge_type', '${run_dir}verl_full_config.yaml'],
                         capture_output=True, text=True).stdout
    if 'self' in out:
        print('self-play')
    elif 'extraction' in out:
        print('extraction')
    else:
        print('external')
    sys.exit(0)

with open('${run_dir}verl_full_config.yaml') as f:
    cfg = yaml.safe_load(f)
cc = cfg.get('covert_comm', {})
mode = cc.get('judge_mode', 'external')
judge_type = cc.get('judge_type', 'scoring')
judge = cc.get('judge_model_id', '')
if mode == 'self':
    print('self-play')
elif judge_type == 'extraction':
    print(f'extraction ({judge.split(\"/\")[-1]})')
else:
    print(f'scoring ({judge.split(\"/\")[-1]})')
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
    elif any('judge_cracked' in r for r in records):
        print('extraction')
    else:
        obf = [r.get('obfuscation_score', -1) for r in records]
        if all(o in (0.0, 1.0, -1) for o in obf) and any(o >= 0 for o in obf):
            print('extraction')
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
    elif [[ "$CURRENT_MODE" == extraction* ]]; then
        print_header_extraction
    else
        print_header_external
    fi
    HEADER_PRINTED=1
}

while true; do
    latest_run=$(find_latest_run)
    [[ -n "$latest_run" && "$latest_run" != */ ]] && latest_run="${latest_run}/"

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
        if [[ "$step_num" -le "$LAST_STEP" ]] 2>/dev/null; then
            continue
        fi

        if [[ -z "$HEADER_PRINTED" ]]; then
            CURRENT_MODE=$(detect_mode_from_rollout "$f")
            print_banner "$(basename "${CURRENT_RUN%/}")" "$CURRENT_MODE"
            if [[ "$CURRENT_MODE" == "self-play" ]]; then
                print_header_self
            elif [[ "$CURRENT_MODE" == extraction* ]]; then
                print_header_extraction
            else
                print_header_external
            fi
            HEADER_PRINTED=1
        fi

        [[ "$LAST_STEP" -gt 0 ]] 2>/dev/null && echo ""
        print_step_from_rollout "$f"
        LAST_STEP="$step_num"
    done

    sleep 10
done
