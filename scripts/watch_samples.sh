#!/usr/bin/env bash
# Live sample viewer — two columns:
#   Left:  correct samples from latest step (sorted by judge score)
#   Right: accumulated ICL breakthrough messages
#
# Usage: bash scripts/watch_samples.sh

BOLD=$'\033[1m'
DIM=$'\033[2m'
RESET=$'\033[0m'
COLS=$(tput cols 2>/dev/null || echo 120)

# ICL cache file (rebuilt on run change, appended per step)
ICL_CACHE=$(mktemp /tmp/watch_samples_icl.XXXXXX)
trap "rm -f $ICL_CACHE" EXIT

# Build ICL cache from all existing rollout files (run once on launch/run change)
build_icl_cache() {
    local rollout_dir="$1"
    > "$ICL_CACHE"
    python3 -c "
import json, os, sys

rollout_dir = '$rollout_dir'
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
    if best_rec is not None and best_s <= 2:
        msg = best_rec.get('encode_text', '').replace(chr(13), '').replace(chr(10), ' ').strip()
        if msg not in seen:
            seen.add(msg)
            print(json.dumps({
                'step': best_rec.get('step', '?'),
                'target': best_rec.get('target', '?'),
                'score': str(best_s),
                'msg': msg
            }))
" > "$ICL_CACHE" 2>/dev/null
}

# Try to add ICL entry from a single new step file
update_icl_cache() {
    local file="$1"
    python3 -c "
import json, sys

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
if best_rec is not None and best_s <= 2:
    msg = best_rec.get('encode_text', '').replace(chr(13), '').replace(chr(10), ' ').strip()
    if msg not in seen:
        print(json.dumps({
            'step': best_rec.get('step', '?'),
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
half = cols // 2 - 2
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

try:
    records = [json.loads(l) for l in open('$file')]
except:
    sys.exit(0)
if not records:
    sys.exit(0)

step = records[0].get('step', '?')
target = records[0].get('target', '?')
n = len(records)
n_correct = sum(1 for r in records if r.get('correct', 0) > 0)

# Left: correct samples sorted by judge score
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

left_items = []
for s, t, msg in pool[:8]:
    c = score_color(s)
    left_items.append((f'[{t} {s}/10]', msg, c))

# Right: ICL from cache file
icl = []
try:
    for l in open('$ICL_CACHE'):
        e = json.loads(l)
        icl.append((e['step'], e['target'], e['score'], e['msg']))
except:
    pass

# Header
print(f'{BOLD}s{step} target={target} ({n_correct}/{n} correct){X}')
print(f'{DIM}{\"-\" * half}{X}  {DIM}{\"-\" * half}{X}')
lh = f'  Correct samples (best first)'
rh = f'  ICL breakthroughs ({len(icl)} total)'
print(f'{DIM}{lh:<{half}}{X}  {DIM}{rh}{X}')
print(f'{DIM}{\"-\" * half}{X}  {DIM}{\"-\" * half}{X}')

# Build columns
left_lines = []
for prefix, msg, c in left_items:
    max_msg = half - len(prefix) - 5
    if len(msg) > max_msg:
        msg = msg[:max_msg - 2] + '..'
    left_lines.append(f'{c}{prefix} \"{msg}\"{X}')

right_lines = []
for st, t, v, msg in reversed(icl):
    prefix = f's{st} [{t} {v}/10]'
    max_msg = half - len(prefix) - 5
    if len(msg) > max_msg:
        msg = msg[:max_msg - 2] + '..'
    right_lines.append(f'{ICL_COLOR}{prefix} \"{msg}\"{X}')

max_rows = max(len(left_lines), len(right_lines))
for i in range(max_rows):
    l = left_lines[i] if i < len(left_lines) else ''
    r = right_lines[i] if i < len(right_lines) else ''
    l_plain = re.sub(r'\033\[[^m]*m', '', l)
    pad = half - len(l_plain)
    if pad < 0: pad = 0
    print(f'  {l}{\" \" * pad}  {r}')

print()
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
        # Build ICL cache once from all existing files
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

        # Incrementally update ICL cache with just this new file
        update_icl_cache "$f"
        print_step "$f"
        LAST_STEP="$step_num"
    done

    sleep 10
done
