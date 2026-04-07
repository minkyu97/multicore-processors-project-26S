#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_OUTPUT_DIR="$ROOT_DIR/benchmark_results/$(date +%Y%m%d_%H%M%S)"

TABLE_SIZE=1000000
NUM_OPS=500000
PROBING=0
KEY_DIST=0
REPS=7
THREADS_STRING="1 2 4 8 16 32"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
BENCH_BIN=""

usage() {
    cat <<'EOF'
Usage:
  run_thread_scaling_bench.sh [options]

Options:
  --table-size N     Hash table size passed to bench_hashmap (default: 1000000)
  --num-ops N        Number of insert/search operations (default: 500000)
  --probing N        0=linear, 1=quadratic (default: 0)
  --key-dist N       0=sequential, 1=random, 2=zipf (default: 0)
  --reps N           Benchmark repetitions (default: 7)
  --threads "..."    Space-separated thread counts (default: "1 2 4 8 16 32")
  --output-dir PATH  Directory for logs and summary output
  --bench-bin PATH   Path to bench_hashmap binary
  --help             Show this help text

Output:
  <output-dir>/summary.tsv
  <output-dir>/logs/sequential.log
  <output-dir>/logs/cas_t{threads}.log
  <output-dir>/logs/mutex_t{threads}.log
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --table-size)
            TABLE_SIZE="$2"
            shift 2
            ;;
        --num-ops)
            NUM_OPS="$2"
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

if [[ "$REPS" -lt 1 ]]; then
    echo "--reps must be >= 1" >&2
    exit 1
fi

resolve_bench_bin() {
    if [[ -n "$BENCH_BIN" ]]; then
        if [[ ! -x "$BENCH_BIN" ]]; then
            echo "bench_hashmap binary is not executable: $BENCH_BIN" >&2
            exit 1
        fi
        return
    fi

    local candidates=(
        "$ROOT_DIR/build/src/bench_hashmap"
        "$ROOT_DIR/build/bench_hashmap"
    )

    local candidate=""
    for candidate in "${candidates[@]}"; do
        if [[ -x "$candidate" ]]; then
            BENCH_BIN="$candidate"
            return
        fi
    done

    if [[ -d "$ROOT_DIR/build" ]]; then
        echo "bench_hashmap binary not found. Building target..." >&2
        cmake --build "$ROOT_DIR/build" --target bench_hashmap >&2
        for candidate in "${candidates[@]}"; do
            if [[ -x "$candidate" ]]; then
                BENCH_BIN="$candidate"
                return
            fi
        done
    fi

    echo "Could not find bench_hashmap. Use --bench-bin or build the target first." >&2
    exit 1
}

extract_metric() {
    local label="$1"
    local log_file="$2"
    awk -v label="$label" '
        index($0, label) == 1 {
            sub(/^[^:]*:[[:space:]]*/, "", $0)
            print $1
            exit
        }
    ' "$log_file"
}

compute_speedup() {
    local baseline="$1"
    local current="$2"
    awk -v baseline="$baseline" -v current="$current" '
        BEGIN {
            if (current == 0) {
                print "0.00"
            } else {
                printf "%.2f", baseline / current
            }
        }
    '
}

resolve_bench_bin

mkdir -p "$OUTPUT_DIR/logs"

CONFIG_FILE="$OUTPUT_DIR/run_config.txt"
SUMMARY_FILE="$OUTPUT_DIR/summary.tsv"
SEQUENTIAL_LOG="$OUTPUT_DIR/logs/sequential.log"

cat > "$CONFIG_FILE" <<EOF
bench_bin=$BENCH_BIN
table_size=$TABLE_SIZE
num_ops=$NUM_OPS
probing=$PROBING
key_dist=$KEY_DIST
reps=$REPS
threads=$THREADS_STRING
EOF

read -r -a THREADS <<< "$THREADS_STRING"

CAS_BASELINE=""

echo "Running sequential benchmark..."
"$BENCH_BIN" "$TABLE_SIZE" "$NUM_OPS" 0 1 "$PROBING" "$KEY_DIST" "$REPS" | tee "$SEQUENTIAL_LOG"

SEQUENTIAL_MEAN="$(extract_metric "Mean time" "$SEQUENTIAL_LOG")"
SEQUENTIAL_STDDEV="$(extract_metric "Std dev" "$SEQUENTIAL_LOG")"
SEQUENTIAL_MIN="$(extract_metric "Min time" "$SEQUENTIAL_LOG")"
SEQUENTIAL_MAX="$(extract_metric "Max time" "$SEQUENTIAL_LOG")"

printf 'Threads\tSequential Mean (s)\tSequential Std Dev\tSequential Min (s)\tSequential Max (s)\tCAS Mean (s)\tCAS Std Dev\tCAS Min (s)\tCAS Max (s)\tMutex Mean (s)\tMutex Std Dev\tMutex Min (s)\tMutex Max (s)\tCAS Speedup vs 1T\n' > "$SUMMARY_FILE"

for thread_count in "${THREADS[@]}"; do
    if [[ "$thread_count" -lt 1 ]]; then
        echo "Thread counts must be >= 1: $thread_count" >&2
        exit 1
    fi

    cas_log="$OUTPUT_DIR/logs/cas_t${thread_count}.log"
    mutex_log="$OUTPUT_DIR/logs/mutex_t${thread_count}.log"

    echo "Running CAS benchmark with ${thread_count} thread(s)..."
    "$BENCH_BIN" "$TABLE_SIZE" "$NUM_OPS" 1 "$thread_count" "$PROBING" "$KEY_DIST" "$REPS" | tee "$cas_log"

    echo "Running mutex benchmark with ${thread_count} thread(s)..."
    "$BENCH_BIN" "$TABLE_SIZE" "$NUM_OPS" 2 "$thread_count" "$PROBING" "$KEY_DIST" "$REPS" | tee "$mutex_log"

    cas_mean="$(extract_metric "Mean time" "$cas_log")"
    cas_stddev="$(extract_metric "Std dev" "$cas_log")"
    cas_min="$(extract_metric "Min time" "$cas_log")"
    cas_max="$(extract_metric "Max time" "$cas_log")"

    mutex_mean="$(extract_metric "Mean time" "$mutex_log")"
    mutex_stddev="$(extract_metric "Std dev" "$mutex_log")"
    mutex_min="$(extract_metric "Min time" "$mutex_log")"
    mutex_max="$(extract_metric "Max time" "$mutex_log")"

    if [[ -z "$CAS_BASELINE" ]]; then
        CAS_BASELINE="$cas_mean"
    fi

    cas_speedup="$(compute_speedup "$CAS_BASELINE" "$cas_mean")"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$thread_count" \
        "$SEQUENTIAL_MEAN" \
        "$SEQUENTIAL_STDDEV" \
        "$SEQUENTIAL_MIN" \
        "$SEQUENTIAL_MAX" \
        "$cas_mean" \
        "$cas_stddev" \
        "$cas_min" \
        "$cas_max" \
        "$mutex_mean" \
        "$mutex_stddev" \
        "$mutex_min" \
        "$mutex_max" \
        "$cas_speedup" >> "$SUMMARY_FILE"
done

echo
echo "Saved summary to: $SUMMARY_FILE"
echo "Saved raw logs to: $OUTPUT_DIR/logs"
