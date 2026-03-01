#!/usr/bin/env bash
# Live sample viewer for covert communication training.
# Shows 7 diverse samples per step:
#   ✓/✗ for correct/incorrect, color gradient by judge score
#   (green=stealthy, yellow=mid, red=obvious).
# Picks samples across the judge score range for a good overview.
# Auto-detects the latest run and switches when a new run starts.
#
# Usage: bash scripts/watch_samples.sh

BOLD=$'\033[1m'
DIM=$'\033[2m'
RESET=$'\033[0m'
COLS=$(tput cols 2>/dev/null || echo 120)

print_step() {
    local file="$1"

    python3 -c "
import json, sys, random

cols = ${COLS}
DIM = '\033[2m'
BOLD = '\033[1m'
X = '\033[0m'

def score_color(score):
    \"\"\"Return ANSI color for judge score 1-10: green->yellow->red gradient.\"\"\"
    # Use 256-color mode for smooth gradient
    if score <= 1:   return '\033[38;5;46m'   # bright green
    elif score <= 2: return '\033[38;5;40m'   # green
    elif score <= 3: return '\033[38;5;34m'   # dark green
    elif score <= 4: return '\033[38;5;226m'  # yellow
    elif score <= 5: return '\033[38;5;220m'  # gold
    elif score <= 6: return '\033[38;5;214m'  # orange
    elif score <= 7: return '\033[38;5;208m'  # dark orange
    elif score <= 8: return '\033[38;5;202m'  # red-orange
    elif score <= 9: return '\033[38;5;196m'  # red
    else:            return '\033[38;5;160m'  # dark red

try:
    records = [json.loads(l) for l in open('$file')]
except:
    sys.exit(0)
if not records:
    sys.exit(0)

step = records[0].get('step', '?')
n = len(records)

# Detect format: binary YES/NO or numeric 1-10
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

# Filter out parse fails for external mode
valid = [r for r in records if get_score(r) >= 0]
if not valid:
    valid = records

# Sort by judge score
sorted_records = sorted(valid, key=get_score)

import re as _re

# Pick 7 correct + 7 incorrect, deduplicated, diverse scores
# Left column = correct (obvious→stealthy), Right column = incorrect (obvious→stealthy)

def pick_7(pool):
    \"\"\"Pick up to 7 from pool (sorted by score asc), deduped by message.
    Top 2 (highest score) + bottom 2 (lowest score) + 3 diverse from middle.\"\"\"
    if not pool:
        return []
    # Deduplicate by message, preserving score order
    deduped = []
    seen = set()
    for r in pool:
        msg = r.get('encode_text', '').strip()
        if msg not in seen:
            seen.add(msg)
            deduped.append(r)
    if len(deduped) <= 7:
        return deduped
    # Bottom 2 (lowest scores) and top 2 (highest scores)
    bottom2 = deduped[:2]
    top2 = deduped[-2:]
    used_idx = {0, 1, len(deduped)-2, len(deduped)-1}
    # 3 from the middle, evenly spaced
    middle = [i for i in range(len(deduped)) if i not in used_idx]
    mid_picks = []
    if middle:
        step_size = max(1, len(middle) // 4)
        for j in range(0, len(middle), step_size):
            if len(mid_picks) >= 3:
                break
            mid_picks.append(deduped[middle[j]])
    return top2 + mid_picks + bottom2

correct_by_score = sorted([r for r in valid if is_correct(r)], key=get_score)
incorrect_by_score = sorted([r for r in valid if not is_correct(r)], key=get_score)

left_col = sorted(pick_7(correct_by_score), key=get_score, reverse=True)
right_col = sorted(pick_7(incorrect_by_score), key=get_score, reverse=True)

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

# Display: left=correct, right=incorrect, 7 rows
margin_w = 5
sep = ' '
half_w = (cols - margin_w - len(sep)) // 2
n_rows = max(len(left_col), len(right_col))

for row in range(n_rows):
    margin = f'{DIM}{step:>4}{X} ' if row == 0 else '     '

    # Left cell (correct)
    if row < len(left_col):
        p1, m1, c1 = make_item(left_col[row])
        max1 = half_w - len(p1) - 2
        if len(m1) > max1:
            m1 = m1[:max1 - 2] + '..'
        cell1 = f'{c1}{p1}\"{m1}\"{X}'
        pad = half_w - len(p1) - len(m1) - 2
    else:
        cell1 = ''
        pad = half_w

    # Right cell (incorrect)
    if row < len(right_col):
        p2, m2, c2 = make_item(right_col[row])
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
        print_step "$f"
        LAST_STEP="$step_num"
    done

    sleep 10
done
