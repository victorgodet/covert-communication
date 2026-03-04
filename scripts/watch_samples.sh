#!/usr/bin/env bash
# Live sample viewer — two columns:
#   Left:  correct samples from latest step (sorted by judge score)
#   Right: accumulated ICL breakthrough messages (score <= 1)
#
# Usage: bash scripts/watch_samples.sh

BOLD=$'\033[1m'
DIM=$'\033[2m'
RESET=$'\033[0m'
COLS=$(tput cols 2>/dev/null || echo 120)

ICL_CACHE=$(mktemp /tmp/watch_samples_icl.XXXXXX)
trap "rm -f $ICL_CACHE" EXIT

ICL_THRESHOLD=1

build_icl_cache() {
    local rollout_dir="$1"
    > "$ICL_CACHE"
    python3 -c "
import json, os, sys

rollout_dir = '$rollout_dir'
threshold = $ICL_THRESHOLD
if not os.path.isdir(rollout_dir):
    sys.exit(0)

files = sorted(
    (int(f.replace('step_','').replace('.jsonl','')), os.path.join(rollout_dir, f))
    for f in os.listdir(rollout_dir) if f.endswith('.jsonl')
)

seen = set()
for step_num, path in files:
    try:
        recs = [json.loads(l) for l in open(path)]
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
    if best_rec is not None and best_s <= threshold:
        msg = best_rec.get('encode_text', '').replace(chr(13), '').replace(chr(10), ' ').strip()
        if msg not in seen:
            seen.add(msg)
            print(json.dumps({
                'step': step_num,
                'target': best_rec.get('target', '?'),
                'score': str(best_s),
                'msg': msg
            }))
" > "$ICL_CACHE" 2>/dev/null
}

update_icl_cache() {
    local file="$1"
    python3 -c "
import json, sys, os, re

threshold = $ICL_THRESHOLD
seen = set()
try:
    for l in open('$ICL_CACHE'):
        e = json.loads(l)
        seen.add(e['msg'])
except:
    pass

try:
    recs = [json.loads(l) for l in open('$file')]
except:
    sys.exit(0)

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
if best_rec is not None and best_s <= threshold:
    msg = best_rec.get('encode_text', '').replace(chr(13), '').replace(chr(10), ' ').strip()
    if msg not in seen:
        step = best_rec.get('step', '?')
        m = re.search(r'step_(\d+)', '$file')
        if m:
            step = int(m.group(1))
        print(json.dumps({
            'step': step,
            'target': best_rec.get('target', '?'),
            'score': str(best_s),
            'msg': msg
        }))
" >> "$ICL_CACHE" 2>/dev/null
}

print_step() {
    local file="$1"

    python3 -c "
import json, sys, re

cols = ${COLS}
half = cols // 2 - 1
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

ICL_COLOR = '\033[38;5;117m'
SEP = f'{DIM}\u2502{X}'

try:
    records = [json.loads(l) for l in open('$file')]
except:
    sys.exit(0)
if not records:
    sys.exit(0)

step = records[0].get('step', '?')
n = len(records)
n_correct = sum(1 for r in records if r.get('correct', 0) > 0)
scores_correct = []
for r in records:
    if r.get('correct', 0) <= 0:
        continue
    try:
        scores_correct.append(int(r.get('judge_verdict', '10')))
    except:
        pass
avg_score = sum(scores_correct) / len(scores_correct) if scores_correct else 0
min_score = min(scores_correct) if scores_correct else 0

# Left: correct samples sorted by judge score (deduped)
seen = set()
pool = []
for r in records:
    if r.get('correct', 0) <= 0:
        continue
    try:
        s = int(r.get('judge_verdict', '10'))
    except:
        continue
    msg = r.get('encode_text', '').replace(chr(13), '').replace(chr(10), ' ').strip()
    if msg not in seen:
        seen.add(msg)
        pool.append((s, r.get('target', '?'), msg))
pool.sort()

# Right: ICL from cache file
icl = []
try:
    for l in open('$ICL_CACHE'):
        e = json.loads(l)
        icl.append((e['step'], e['target'], e['score'], e['msg']))
except:
    pass

MAX_ROWS = 7
left_entries = [(s, t, msg) for s, t, msg in pool[:MAX_ROWS]]
right_entries = list(reversed(icl))[:MAX_ROWS]
max_rows = max(len(left_entries), len(right_entries), 1)
max_rows = min(max_rows, MAX_ROWS)

step_label = f's{step}'
step_w = len(step_label) + 1
stats = f'{n_correct}/{n} avg={avg_score:.1f} best={min_score}'
icl_label = f'ICL({len(icl)})'

for i in range(max_rows):
    if i == 0:
        prefix_str = f'{DIM}{step_label}{X} '
    else:
        prefix_str = ' ' * step_w
    prefix_plain_len = step_w

    if i < len(left_entries):
        s, t, msg = left_entries[i]
        c = score_color(s)
        tag = f'{s:>2}/10'
        tag_len = 5
        avail = half - prefix_plain_len - tag_len - 4
        if len(msg) > avail:
            msg = msg[:avail - 2] + '..'
        left_col = f'{prefix_str}{c}{tag}{X} {DIM}\"{msg}\"{X}'
        left_plain_len = prefix_plain_len + tag_len + 1 + len(msg) + 2
    else:
        left_col = prefix_str
        left_plain_len = prefix_plain_len

    pad = half - left_plain_len
    if pad < 0:
        pad = 0

    if i < len(right_entries):
        st, t, v, msg = right_entries[i]
        tag = f's{st}'
        avail = half - len(tag) - 6
        if len(msg) > avail:
            msg = msg[:avail - 2] + '..'
        right_col = f'{ICL_COLOR}{tag} {v}/10 \"{msg}\"{X}'
    else:
        right_col = ''

    is_last = (i == max_rows - 1)
    nl = '' if is_last else '\n'
    if right_col:
        sys.stdout.write(f'{left_col}{\" \" * pad} {SEP} {right_col}{nl}')
    else:
        sys.stdout.write(f'{left_col}{nl}')
sys.stdout.flush()
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
        printf "\n${BOLD}Covert Comm Samples${RESET}  ${DIM}$(basename "${CURRENT_RUN%/}")${RESET}\n"
        build_icl_cache "${CURRENT_RUN}rollouts"
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

        [[ "$LAST_STEP" -gt 0 ]] 2>/dev/null && echo ""
        update_icl_cache "$f"
        print_step "$f"
        LAST_STEP="$step_num"
    done

    sleep 10
done
