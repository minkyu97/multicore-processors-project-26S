#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/bench_common.sh"

DEFAULT_OUTPUT_DIR="$ROOT_DIR/benchmark_results/key_distribution_$(date +%Y%m%d_%H%M%S)"

TABLE_SIZE=10000000
LOAD_FACTOR=50
PROBING=0
THREADS=16
REPS=7
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
BENCH_BIN=""

usage() {
    cat <<'EOF'
Usage:
  bench_table_key_distribution.sh [options]

Options:
  --table-size N     Hash table size (default: 10000000)
  --load-factor N    Load factor in percent (default: 50)
  --probing N        0=linear, 1=quadratic (default: 0)
  --threads N        Thread count for CAS/mutex rows (default: 16)
  --reps N           Benchmark repetitions (default: 7)
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
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --reps)
            REPS="$2"
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

if [[ "$LOAD_FACTOR" -le 0 || "$LOAD_FACTOR" -ge 100 ]]; then
    echo "--load-factor must be between 1 and 99" >&2
    exit 1
fi

if [[ "$THREADS" -lt 1 ]]; then
    echo "--threads must be >= 1" >&2
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

cat > "$CONFIG_FILE" <<EOF
bench_bin=$BENCH_BIN
table_size=$TABLE_SIZE
ops=$OPS
load_factor=$LOAD_FACTOR
probing=$PROBING
threads=$THREADS
reps=$REPS
EOF

printf 'Key Distribution\tBackend\tMean (s)\tStd Dev\tMin (s)\tMax (s)\tSpeedup vs Seq\n' > "$SUMMARY_FILE"

for key_dist in 0 1 2; do
    label="$(key_dist_label "$key_dist")"
    slug="$(key_dist_slug "$key_dist")"

    seq_log="$OUTPUT_DIR/logs/${slug}_sequential.log"
    cas_log="$OUTPUT_DIR/logs/${slug}_cas_t${THREADS}.log"
    mutex_log="$OUTPUT_DIR/logs/${slug}_mutex_t${THREADS}.log"

    echo "Running sequential benchmark for ${label}..."
    "$BENCH_BIN" "$TABLE_SIZE" "$OPS" 0 1 "$PROBING" "$key_dist" "$REPS" | tee "$seq_log"

    seq_mean="$(extract_metric "Mean time" "$seq_log")"
    seq_stddev="$(extract_metric "Std dev" "$seq_log")"
    seq_min="$(extract_metric "Min time" "$seq_log")"
    seq_max="$(extract_metric "Max time" "$seq_log")"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" \
        "Sequential" \
        "$seq_mean" \
        "$seq_stddev" \
        "$seq_min" \
        "$seq_max" \
        "1" >> "$SUMMARY_FILE"

    echo "Running CAS benchmark for ${label} with ${THREADS} thread(s)..."
    "$BENCH_BIN" "$TABLE_SIZE" "$OPS" 1 "$THREADS" "$PROBING" "$key_dist" "$REPS" | tee "$cas_log"

    cas_mean="$(extract_metric "Mean time" "$cas_log")"
    cas_stddev="$(extract_metric "Std dev" "$cas_log")"
    cas_min="$(extract_metric "Min time" "$cas_log")"
    cas_max="$(extract_metric "Max time" "$cas_log")"
    cas_speedup="$(compute_speedup "$seq_mean" "$cas_mean")"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" \
        "CAS (${THREADS}T)" \
        "$cas_mean" \
        "$cas_stddev" \
        "$cas_min" \
        "$cas_max" \
        "$cas_speedup" >> "$SUMMARY_FILE"

    echo "Running mutex benchmark for ${label} with ${THREADS} thread(s)..."
    "$BENCH_BIN" "$TABLE_SIZE" "$OPS" 2 "$THREADS" "$PROBING" "$key_dist" "$REPS" | tee "$mutex_log"

    mutex_mean="$(extract_metric "Mean time" "$mutex_log")"
    mutex_stddev="$(extract_metric "Std dev" "$mutex_log")"
    mutex_min="$(extract_metric "Min time" "$mutex_log")"
    mutex_max="$(extract_metric "Max time" "$mutex_log")"
    mutex_speedup="$(compute_speedup "$seq_mean" "$mutex_mean")"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" \
        "Mutex (${THREADS}T)" \
        "$mutex_mean" \
        "$mutex_stddev" \
        "$mutex_min" \
        "$mutex_max" \
        "$mutex_speedup" >> "$SUMMARY_FILE"
done

echo
echo "Saved summary to: $SUMMARY_FILE"
echo "Saved raw logs to: $OUTPUT_DIR/logs"
