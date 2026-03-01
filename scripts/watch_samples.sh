#!/usr/bin/env bash
# Live sample viewer for covert communication training.
#
# Default: left=correct samples (diverse scores), right=accumulated missed
# examples (1-2/10 + correct across all steps) fed to the judge via ICL.
#
# --show-incorrect: right column shows incorrect samples instead.
#
# Auto-detects the latest run and switches when a new run starts.
#
# Usage:
#   bash scripts/watch_samples.sh                    # accumulated misses
#   bash scripts/watch_samples.sh --show-incorrect   # correct/incorrect columns

MODE="accumulated"
if [[ "$1" == "--show-incorrect" ]]; then
    MODE="incorrect"
fi

BOLD=$'\033[1m'
DIM=$'\033[2m'
RESET=$'\033[0m'
COLS=$(tput cols 2>/dev/null || echo 120)

print_step() {
    local file="$1"
    local all_files="$2"
    local mode="$3"

    python3 -c "
import json, sys, re as _re

cols = ${COLS}
DIM = '\033[2m'
BOLD = '\033[1m'
X = '\033[0m'
mode = '$mode'

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

def get_score(r):
    v = r.get('judge_verdict', '')
    if has_binary:
        return 10 if v == 'YES' else (1 if v == 'NO' else 5)
    try:
        return int(v)
    except:
        return -1

def is_correct(r):
    return r.get('correct', 0) > 0

valid = [r for r in records if get_score(r) >= 0]
if not valid:
    valid = records

def pick_7(pool):
    if not pool:
        return []
    deduped = []
    seen = set()
    for r in pool:
        msg = r.get('encode_text', '').strip()
        if msg not in seen:
            seen.add(msg)
            deduped.append(r)
    if len(deduped) <= 7:
        return deduped
    bottom2 = deduped[:2]
    top2 = deduped[-2:]
    used_idx = {0, 1, len(deduped)-2, len(deduped)-1}
    middle = [i for i in range(len(deduped)) if i not in used_idx]
    mid_picks = []
    if middle:
        step_size = max(1, len(middle) // 4)
        for j in range(0, len(middle), step_size):
            if len(mid_picks) >= 3:
                break
            mid_picks.append(deduped[middle[j]])
    return top2 + mid_picks + bottom2

# Left column: correct samples (diverse scores), always the same
correct_by_score = sorted([r for r in valid if is_correct(r)], key=get_score)
left_col = sorted(pick_7(correct_by_score), key=get_score, reverse=True)

def make_item(r):
    t = r.get('target', '?')
    v = r.get('judge_verdict', '?')
    raw_dec = r.get('decode_text', r.get('decoded_value', '?'))
    _m = _re.search(r'\d+', str(raw_dec))
    decoded = _m.group(0) if _m else '?'
    score = get_score(r)
    if has_binary:
        tag = v
    else:
        tag = f'{v}/10' if v not in ('PARSE_FAIL', '') else '?/10'
    c = score_color(score) if score >= 0 else DIM
    status = '\u2713' if is_correct(r) else '\u2717'
    msg = r.get('encode_text', '').replace(chr(10), ' ').strip()
    if is_correct(r):
        prefix = f'{status} [{t} {tag}] '
    else:
        prefix = f'{status} [{t}\u2192{decoded} {tag}] '
    return (prefix, msg, c)

# Right column
if mode == 'accumulated':
    # Mirror the code logic: 1 best correct sample per step, only if <=3
    all_files = '$all_files'.split()
    all_misses = []
    seen_msgs = set()
    for af in all_files:
        try:
            step_records = [json.loads(l) for l in open(af)]
        except:
            continue
        # Find the single lowest-scoring correct sample in this step
        best_score = 11
        best_rec = None
        for r in step_records:
            if r.get('correct', 0) <= 0:
                continue
            v = r.get('judge_verdict', '')
            try:
                s = int(v)
            except:
                continue
            if s < best_score:
                best_score = s
                best_rec = r
        if best_rec is not None and best_score <= 3:
            msg = best_rec.get('encode_text', '').replace(chr(10), ' ').strip()
            if msg not in seen_msgs:
                seen_msgs.add(msg)
                all_misses.append({
                    'step': best_rec.get('step', '?'),
                    'target': best_rec.get('target', '?'),
                    'judge_verdict': str(best_score),
                    'encode_text': msg,
                })

    # Make right column items from accumulated misses
    right_col_items = []
    for r in all_misses:
        s = int(r['judge_verdict'])
        c = score_color(s)
        t = r['target']
        st = r['step']
        tag = f'{s}/10'
        prefix = f's{st} [{t} {tag}] '
        msg = r['encode_text']
        right_col_items.append((prefix, msg, c))
else:
    # Incorrect samples (old behavior)
    incorrect_by_score = sorted([r for r in valid if not is_correct(r)], key=get_score)
    right_col_raw = sorted(pick_7(incorrect_by_score), key=get_score, reverse=True)
    right_col_items = [make_item(r) for r in right_col_raw]

# Make left column items
left_col_items = [make_item(r) for r in left_col]

# Display: 2 columns
margin_w = 5
sep = ' '
half_w = (cols - margin_w - len(sep)) // 2
n_rows = max(len(left_col_items), len(right_col_items))

for row in range(n_rows):
    margin = f'{DIM}{step:>4}{X} ' if row == 0 else '     '

    # Left cell
    if row < len(left_col_items):
        p1, m1, c1 = left_col_items[row]
        max1 = half_w - len(p1) - 2
        if len(m1) > max1:
            m1 = m1[:max1 - 2] + '..'
        cell1 = f'{c1}{p1}\"{m1}\"{X}'
        pad = half_w - len(p1) - len(m1) - 2
    else:
        cell1 = ''
        pad = half_w

    # Right cell
    if row < len(right_col_items):
        p2, m2, c2 = right_col_items[row]
        max2 = half_w - len(p2) - 2
        if len(m2) > max2:
            m2 = m2[:max2 - 2] + '..'
        cell2 = f'{c2}{p2}\"{m2}\"{X}'
    else:
        cell2 = ''

    is_last = row == n_rows - 1
    end = '' if is_last else '\n'
    if cell2:
        print(f'{margin}{cell1}{\" \" * pad}{sep}{cell2}', end=end)
    else:
        print(f'{margin}{cell1}', end=end)

if n_rows == 0:
    print(f'{DIM}{step:>4}{X} {DIM}(no samples){X}', end='')
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

    for f in $(find "$rollout_dir" -name 'step_*.jsonl' 2>/dev/null | sort -t_ -k2 -n); do
        step_num=$(basename "$f" .jsonl | sed 's/step_//')
        if [[ "$step_num" -le "$LAST_STEP" ]] 2>/dev/null; then
            continue
        fi
        # Newline to close previous step's end=''
        [[ "$LAST_STEP" -gt 0 ]] 2>/dev/null && echo ""

        # Collect all rollout files up to and including this step (space-separated)
        all_files=$(find "$rollout_dir" -name 'step_*.jsonl' 2>/dev/null | sort -t_ -k2 -n | while read af; do
            af_num=$(basename "$af" .jsonl | sed 's/step_//')
            if [[ "$af_num" -le "$step_num" ]] 2>/dev/null; then
                printf '%s ' "$af"
            fi
        done)

        print_step "$f" "$all_files" "$MODE"
        LAST_STEP="$step_num"
    done

    sleep 10
done
