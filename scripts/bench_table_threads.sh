#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/bench_common.sh"

DEFAULT_OUTPUT_DIR="$ROOT_DIR/benchmark_results/threads_$(date +%Y%m%d_%H%M%S)"

TABLE_SIZE=10000000
LOAD_FACTOR=50
PROBING=0
KEY_DIST=1
REPS=7
THREADS_STRING="1 2 4 8 16 32"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
BENCH_BIN=""

usage() {
    cat <<'EOF'
Usage:
  bench_table_threads.sh [options]

Options:
  --table-size N     Hash table size (default: 10000000)
  --load-factor N    Load factor in percent (default: 50)
  --probing N        0=linear, 1=quadratic (default: 0)
  --key-dist N       0=sequential, 1=random, 2=zipf (default: 1)
  --reps N           Benchmark repetitions (default: 7)
  --threads "..."    Space-separated thread counts (default: "1 2 4 8 16 32")
  --output-dir PATH  Directory for logs and summary output
  --bench-bin PATH   Path to bench_hashmap binary
  --help             Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --table-size)
            TABLE_SIZE="$2"
            shift 2
            ;;
        --load-factor)
            LOAD_FACTOR="$2"
            shift 2
            ;;
        --probing)
            PROBING="$2"
            shift 2
            ;;
        --key-dist)
            KEY_DIST="$2"
            shift 2
            ;;
        --reps)
            REPS="$2"
            shift 2
            ;;
        --threads)
            THREADS_STRING="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --bench-bin)
            BENCH_BIN="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ "$PROBING" != "0" && "$PROBING" != "1" ]]; then
    echo "--probing must be 0 or 1" >&2
    exit 1
fi

if [[ "$KEY_DIST" != "0" && "$KEY_DIST" != "1" && "$KEY_DIST" != "2" ]]; then
    echo "--key-dist must be 0, 1, or 2" >&2
    exit 1
fi

if [[ "$LOAD_FACTOR" -le 0 || "$LOAD_FACTOR" -ge 100 ]]; then
    echo "--load-factor must be between 1 and 99" >&2
    exit 1
fi

if [[ "$REPS" -lt 1 ]]; then
    echo "--reps must be >= 1" >&2
    exit 1
fi

OPS=$((TABLE_SIZE * LOAD_FACTOR / 100))
if [[ "$OPS" -lt 1 || "$OPS" -ge "$TABLE_SIZE" ]]; then
    echo "Computed ops must satisfy 1 <= ops < table_size" >&2
    exit 1
fi

BENCH_BIN="$(resolve_bench_bin "$ROOT_DIR" "$BENCH_BIN")"

mkdir -p "$OUTPUT_DIR/logs"

CONFIG_FILE="$OUTPUT_DIR/run_config.txt"
SUMMARY_FILE="$OUTPUT_DIR/summary.tsv"
SEQUENTIAL_LOG="$OUTPUT_DIR/logs/sequential_baseline.log"

cat > "$CONFIG_FILE" <<EOF
bench_bin=$BENCH_BIN
table_size=$TABLE_SIZE
ops=$OPS
load_factor=$LOAD_FACTOR
probing=$PROBING
key_dist=$KEY_DIST
reps=$REPS
threads=$THREADS_STRING
EOF

printf 'Backend\tThreads\tMean (s)\tStd Dev\tMin (s)\tMax (s)\tSpeedup vs Seq\n' > "$SUMMARY_FILE"

echo "Running sequential baseline..."
"$BENCH_BIN" "$TABLE_SIZE" "$OPS" 0 1 "$PROBING" "$KEY_DIST" "$REPS" | tee "$SEQUENTIAL_LOG"

SEQ_MEAN="$(extract_metric "Mean time" "$SEQUENTIAL_LOG")"
SEQ_STDDEV="$(extract_metric "Std dev" "$SEQUENTIAL_LOG")"
SEQ_MIN="$(extract_metric "Min time" "$SEQUENTIAL_LOG")"
SEQ_MAX="$(extract_metric "Max time" "$SEQUENTIAL_LOG")"

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "Sequential (baseline)" \
    "1" \
    "$SEQ_MEAN" \
    "$SEQ_STDDEV" \
    "$SEQ_MIN" \
    "$SEQ_MAX" \
    "1" >> "$SUMMARY_FILE"

read -r -a THREADS <<< "$THREADS_STRING"

for thread_count in "${THREADS[@]}"; do
    if [[ "$thread_count" -lt 1 ]]; then
        echo "Thread counts must be >= 1: $thread_count" >&2
        exit 1
    fi

    cas_log="$OUTPUT_DIR/logs/cas_t${thread_count}.log"
    mutex_log="$OUTPUT_DIR/logs/mutex_t${thread_count}.log"

    echo "Running CAS benchmark with ${thread_count} thread(s)..."
    "$BENCH_BIN" "$TABLE_SIZE" "$OPS" 1 "$thread_count" "$PROBING" "$KEY_DIST" "$REPS" | tee "$cas_log"

    cas_mean="$(extract_metric "Mean time" "$cas_log")"
    cas_stddev="$(extract_metric "Std dev" "$cas_log")"
    cas_min="$(extract_metric "Min time" "$cas_log")"
    cas_max="$(extract_metric "Max time" "$cas_log")"
    cas_speedup="$(compute_speedup "$SEQ_MEAN" "$cas_mean")"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "CAS" \
        "$thread_count" \
        "$cas_mean" \
        "$cas_stddev" \
        "$cas_min" \
        "$cas_max" \
        "$cas_speedup" >> "$SUMMARY_FILE"
done

for thread_count in "${THREADS[@]}"; do
    mutex_log="$OUTPUT_DIR/logs/mutex_t${thread_count}.log"

    echo "Running mutex benchmark with ${thread_count} thread(s)..."
    "$BENCH_BIN" "$TABLE_SIZE" "$OPS" 2 "$thread_count" "$PROBING" "$KEY_DIST" "$REPS" | tee "$mutex_log"

    mutex_mean="$(extract_metric "Mean time" "$mutex_log")"
    mutex_stddev="$(extract_metric "Std dev" "$mutex_log")"
    mutex_min="$(extract_metric "Min time" "$mutex_log")"
    mutex_max="$(extract_metric "Max time" "$mutex_log")"
    mutex_speedup="$(compute_speedup "$SEQ_MEAN" "$mutex_mean")"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "Mutex" \
        "$thread_count" \
        "$mutex_mean" \
        "$mutex_stddev" \
        "$mutex_min" \
        "$mutex_max" \
        "$mutex_speedup" >> "$SUMMARY_FILE"
done

echo
echo "Saved summary to: $SUMMARY_FILE"
echo "Saved raw logs to: $OUTPUT_DIR/logs"
