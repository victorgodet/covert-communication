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

# Column legend printed once at the top
cat <<'LEGEND'
Columns:
  step      = training step
  accuracy  = % of samples where decoder recovered the correct number
  digits    = % of encoder messages containing digit characters (should be 0%)
  enc_r     = encoder reward (higher = better hiding + correct decode)
  dec_r     = decoder reward (higher = more correct decodes)
  obf       = obfuscation score from judge (1.0 = fully hidden, 0.0 = obvious)
  detected  = number of samples the judge flagged as suspicious
  clean     = number of samples the judge found innocent
  think_len = mean thinking tokens (inside <think> block)
  resp_len  = mean total response length in tokens (thinking + message)
  maxed_out = % of responses that hit the token limit (thinking too long)
  pg_loss   = policy gradient loss

LEGEND

HEADER_FMT="%-5s  %-8s  %-6s  %-7s  %-7s  %-5s  %-8s  %-5s  %-9s  %-8s  %-9s  %-7s\n"
ROW_FMT="%-5s  %-8s  %-6s  %-7s  %-7s  %-5s  %-8s  %-5s  %-9s  %-8s  %-9s  %-7s\n"

print_header() {
    printf "$HEADER_FMT" \
        "step" "accuracy" "digits" "enc_r" "dec_r" "obf" "detected" "clean" "think_len" "resp_len" "maxed_out" "pg_loss"
    printf "%s\n" \
        "-----  --------  ------  -------  -------  -----  --------  -----  ---------  --------  ---------  -------"
}

LAST_STEP=0

parse_and_print() {
    local file="$1"

    while IFS= read -r summary_line; do
        step=$(echo "$summary_line" | grep -oP 'Step \K[0-9]+')
        [[ -z "$step" ]] && continue
        (( step <= LAST_STEP )) && continue

        acc=$(echo "$summary_line" | grep -oP 'acc=\K[0-9]+%')
        digits=$(echo "$summary_line" | grep -oP 'digits=\K[0-9]+%')
        enc_r=$(echo "$summary_line" | grep -oP 'enc_r=\K[+-]?[0-9.]+')
        dec_r=$(echo "$summary_line" | grep -oP 'dec_r=\K[+-]?[0-9.]+')
        obf=$(echo "$summary_line" | grep -oP 'obf=\K[0-9.]+')
        think_len=$(echo "$summary_line" | grep -oP 'think=\K[0-9]+' || true)

        metric_line=$(grep "^step:${step} " "$file" 2>/dev/null | tail -1 || true)
        resp_len="" ; clip_ratio="" ; pg_loss=""
        if [[ -n "$metric_line" ]]; then
            resp_len=$(echo "$metric_line" | grep -oP 'response_length/mean:\K[0-9.]+' || true)
            clip_ratio=$(echo "$metric_line" | grep -oP 'response_length/clip_ratio:\K[0-9.]+' || true)
            pg_loss=$(echo "$metric_line" | grep -oP 'actor/pg_loss:\K[0-9.e+-]+' || true)
        fi

        judge_line=$(grep -A1 "^Step ${step} " "$file" | grep "Judge:" | tail -1 || true)
        detected="" ; clean=""
        if [[ -n "$judge_line" ]]; then
            detected=$(echo "$judge_line" | grep -oP '[0-9]+ detected' | grep -oP '[0-9]+' || true)
            clean=$(echo "$judge_line" | grep -oP '[0-9]+ clean' | grep -oP '[0-9]+' || true)
        fi

        # Format numbers
        resp_fmt=""
        [[ -n "$resp_len" ]] && resp_fmt=$(awk "BEGIN {printf \"%.0f\", $resp_len}")
        clip_pct=""
        [[ -n "$clip_ratio" ]] && clip_pct=$(awk "BEGIN {printf \"%.0f%%\", $clip_ratio * 100}")
        pg_fmt=""
        [[ -n "$pg_loss" ]] && pg_fmt=$(awk "BEGIN {printf \"%.5f\", $pg_loss}")

        printf "$ROW_FMT" \
            "$step" "${acc:-?}" "${digits:-?}" "${enc_r:-?}" "${dec_r:-?}" \
            "${obf:-?}" "${detected:-?}" "${clean:-?}" \
            "${think_len:-?}" "${resp_fmt:-?}" "${clip_pct:-?}" "${pg_fmt:-?}"

        LAST_STEP=$step
    done < <(grep "^Step " "$file")
}

# Print header + existing data
print_header
parse_and_print "$LOG"

# Tail for new steps
tail -n 0 -f "$LOG" 2>/dev/null | while IFS= read -r line; do
    if [[ "$line" == Step\ * ]]; then
        sleep 2  # let metric line flush
        parse_and_print "$LOG"
    fi
done
