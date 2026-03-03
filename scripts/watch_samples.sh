#!/usr/bin/env bash
# Live sample viewer — shows only stealthy/ICL examples (single column).
# Compact for quarter-screen windows.
#
# Usage: bash scripts/watch_samples.sh

BOLD=$'\033[1m'
DIM=$'\033[2m'
RESET=$'\033[0m'
COLS=$(tput cols 2>/dev/null || echo 60)

print_step() {
    local file="$1"
    local all_files="$2"

    python3 -c "
import json, sys, re as _re

cols = ${COLS}
DIM = '\033[2m'
BOLD = '\033[1m'
X = '\033[0m'

def score_color(score):
    if score <= 1:   return '\033[38;5;46m'
    elif score <= 2: return '\033[38;5;40m'
    elif score <= 3: return '\033[38;5;34m'
    elif score <= 4: return '\033[38;5;226m'
    elif score <= 5: return '\033[38;5;220m'
    elif score <= 6: return '\033[38;5;214m'
    elif score <= 7: return '\033[38;5;208m'
    elif score <= 8: return '\033[38;5;202m'
    elif score <= 9: return '\033[38;5;196m'
    else:            return '\033[38;5;160m'

STEALTHY_COLOR = '\033[38;5;46m'
ICL_COLOR = '\033[38;5;117m'

try:
    records = [json.loads(l) for l in open('$file')]
except:
    sys.exit(0)
if not records:
    sys.exit(0)

step = records[0].get('step', '?')
n = len(records)

verdicts = [r.get('judge_verdict', '') for r in records]
has_binary = any(v in ('YES', 'NO') for v in verdicts)
has_cracked = any('judge_cracked' in r for r in records)
if not has_cracked and not has_binary:
    obf = [r.get('obfuscation_score', -1) for r in records]
    if all(o in (0.0, 1.0, -1) for o in obf) and any(o >= 0 for o in obf):
        has_cracked = True

def is_correct(r):
    return r.get('correct', 0) > 0
def is_cracked(r):
    if 'judge_cracked' in r:
        return r['judge_cracked'] > 0.5
    return str(r.get('judge_verdict', '')) == str(r.get('target', ''))

# Collect stealthy samples from this step
items = []
if has_cracked and not has_binary:
    # Extraction mode: correct + judge failed
    seen = set()
    for r in records:
        if is_correct(r) and not is_cracked(r):
            msg = r.get('encode_text', '').replace(chr(10), ' ').strip()
            if msg not in seen:
                seen.add(msg)
                t = r.get('target', '?')
                v = r.get('judge_verdict', '?')
                items.append((f'[{t} j={v}]', msg, STEALTHY_COLOR))
            if len(items) >= 5:
                break
else:
    # Scoring mode: correct + score <= 3
    seen = set()
    pool = []
    for r in records:
        if not is_correct(r):
            continue
        try:
            s = int(r.get('judge_verdict', '10'))
        except:
            continue
        if s <= 3:
            msg = r.get('encode_text', '').replace(chr(10), ' ').strip()
            if msg not in seen:
                seen.add(msg)
                pool.append((s, r.get('target', '?'), msg))
    pool.sort()
    for s, t, msg in pool[:5]:
        c = score_color(s)
        items.append((f'[{t} {s}/10]', msg, c))

# Also show accumulated ICL examples
all_files = '$all_files'.split()
icl = []
seen_icl = set()

if has_cracked and not has_binary:
    for af in all_files:
        try:
            recs = [json.loads(l) for l in open(af)]
        except:
            continue
        for r in recs:
            if r.get('correct', 0) <= 0:
                continue
            target = str(r.get('target', ''))
            verdict = str(r.get('judge_verdict', ''))
            cracked = r.get('judge_cracked', 0) > 0.5 if 'judge_cracked' in r else verdict == target
            if cracked:
                continue
            msg = r.get('encode_text', '').replace(chr(10), ' ').strip()
            if msg not in seen_icl:
                seen_icl.add(msg)
                icl.append((r.get('step', '?'), target, verdict, msg))
                break
else:
    for af in all_files:
        try:
            recs = [json.loads(l) for l in open(af)]
        except:
            continue
        best_s = 11
        best_rec = None
        for r in recs:
            if r.get('correct', 0) <= 0:
                continue
            try:
                s = int(r.get('judge_verdict', '10'))
            except:
                continue
            if s < best_s:
                best_s = s
                best_rec = r
        if best_rec is not None and best_s <= 2:
            msg = best_rec.get('encode_text', '').replace(chr(10), ' ').strip()
            if msg not in seen_icl:
                seen_icl.add(msg)
                icl.append((best_rec.get('step', '?'), best_rec.get('target', '?'), str(best_s), msg))

w = cols - 2

# Print this step's stealthy samples
if items:
    for i, (prefix, msg, c) in enumerate(items):
        label = f's{step} ' if i == 0 else '    '
        max_msg = w - len(label) - len(prefix) - 3
        if len(msg) > max_msg:
            msg = msg[:max_msg - 2] + '..'
        print(f'{DIM}{label}{X}{c}{prefix} \"{msg}\"{X}')
else:
    print(f'{DIM}s{step} (no stealthy samples){X}')

# Print ICL separator + accumulated examples (last 7)
if icl:
    print(f'{DIM}  icl ({len(icl)} total):{X}')
    for st, t, v, msg in icl[-7:]:
        if has_cracked and not has_binary:
            prefix = f's{st} [{t} j={v}]'
        else:
            prefix = f's{st} [{t} {v}/10]'
        max_msg = w - len(prefix) - 5
        if len(msg) > max_msg:
            msg = msg[:max_msg - 2] + '..'
        print(f'  {ICL_COLOR}{prefix} \"{msg}\"{X}')
" 2>/dev/null
}

CURRENT_RUN=""
LAST_STEP=0

find_latest_run() {
    find results/runs -mindepth 2 -maxdepth 2 -type d -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2-
}

while true; do
    latest_run=$(find_latest_run)
    [[ -n "$latest_run" && "$latest_run" != */ ]] && latest_run="${latest_run}/"

    if [[ -n "$latest_run" && "$latest_run" != "$CURRENT_RUN" ]]; then
        CURRENT_RUN="$latest_run"
        LAST_STEP=0
        echo ""
        echo "${BOLD}Covert Comm Samples${RESET}"
        echo "${DIM}$(basename "${CURRENT_RUN%/}")${RESET}"
        echo ""
    fi

    if [[ -z "$CURRENT_RUN" ]]; then
        sleep 10
        continue
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

        all_files=$(find "$rollout_dir" -name 'step_*.jsonl' 2>/dev/null | while read p; do n=$(basename "$p" .jsonl | sed 's/step_//'); printf '%s %s\n' "$n" "$p"; done | sort -n | cut -d' ' -f2- | while read af; do
            af_num=$(basename "$af" .jsonl | sed 's/step_//')
            if [[ "$af_num" -le "$step_num" ]] 2>/dev/null; then
                printf '%s ' "$af"
            fi
        done)

        print_step "$f" "$all_files"
        LAST_STEP="$step_num"
    done

    sleep 10
done
