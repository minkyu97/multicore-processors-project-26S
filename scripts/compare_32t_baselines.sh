#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/bench_common.sh"

DEFAULT_OUTPUT_DIR="$ROOT_DIR/benchmark_results/compare_32t_$(date +%Y%m%d_%H%M%S)"

TABLE_SIZE=10000000
LOAD_FACTOR=50
THREADS=32
PROBING=0
KEY_DIST=1
REPS=7
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
CUSTOM_BENCH_BIN=""
BASELINE_BENCH_BIN=""

usage() {
    cat <<'EOF'
Usage:
  compare_32t_baselines.sh [options]

Options:
  --table-size N       Hash table size (default: 10000000)
  --load-factor N      Load factor in percent (default: 50)
  --threads N          Thread count for custom parallel rows (default: 32)
  --probing N          0=linear, 1=quadratic (default: 0)
  --key-dist N         0=sequential, 1=random, 2=zipf (default: 1)
  --reps N             Benchmark repetitions (default: 7)
  --output-dir PATH    Directory for logs and summary output
  --bench-bin PATH     Path to bench_hashmap binary
  --baseline-bin PATH  Path to bench_map_baselines binary
  --help               Show this help text

Output:
  <output-dir>/summary.tsv
  <output-dir>/logs/custom_sequential.log
  <output-dir>/logs/std_unordered_map.log
  <output-dir>/logs/absl_flat_hash_map.log (if available)
  <output-dir>/logs/custom_cas_t{threads}.log
  <output-dir>/logs/custom_mutex_t{threads}.log
  <output-dir>/logs/segmented_cas_t{threads}.log
  <output-dir>/logs/segmented_mutex_t{threads}.log
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
        --threads)
            THREADS="$2"
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
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --bench-bin)
            CUSTOM_BENCH_BIN="$2"
            shift 2
            ;;
        --baseline-bin)
            BASELINE_BENCH_BIN="$2"
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

if [[ "$LOAD_FACTOR" -le 0 || "$LOAD_FACTOR" -ge 100 ]]; then
    echo "--load-factor must be between 1 and 99" >&2
    exit 1
fi

if [[ "$THREADS" -lt 1 ]]; then
    echo "--threads must be >= 1" >&2
    exit 1
fi

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

OPS=$((TABLE_SIZE * LOAD_FACTOR / 100))
if [[ "$OPS" -lt 1 || "$OPS" -ge "$TABLE_SIZE" ]]; then
    echo "Computed ops must satisfy 1 <= ops < table_size" >&2
    exit 1
fi

CUSTOM_BENCH_BIN="$(resolve_cmake_binary "$ROOT_DIR" "bench_hashmap" "$CUSTOM_BENCH_BIN")"
BASELINE_BENCH_BIN="$(resolve_cmake_binary "$ROOT_DIR" "bench_map_baselines" "$BASELINE_BENCH_BIN")"

mkdir -p "$OUTPUT_DIR/logs"

CONFIG_FILE="$OUTPUT_DIR/run_config.txt"
SUMMARY_FILE="$OUTPUT_DIR/summary.tsv"

cat > "$CONFIG_FILE" <<EOF
custom_bench_bin=$CUSTOM_BENCH_BIN
baseline_bench_bin=$BASELINE_BENCH_BIN
table_size=$TABLE_SIZE
ops=$OPS
load_factor=$LOAD_FACTOR
threads=$THREADS
probing=$PROBING
key_dist=$KEY_DIST
reps=$REPS
EOF

printf 'Implementation\tThreads\tMean (s)\tStd Dev\tMin (s)\tMax (s)\tSpeedup vs std 1T\tSpeedup vs absl 1T\n' > "$SUMMARY_FILE"

CUSTOM_SEQ_LOG="$OUTPUT_DIR/logs/custom_sequential.log"
STD_LOG="$OUTPUT_DIR/logs/std_unordered_map.log"
ABSL_LOG="$OUTPUT_DIR/logs/absl_flat_hash_map.log"
CAS_LOG="$OUTPUT_DIR/logs/custom_cas_t${THREADS}.log"
MUTEX_LOG="$OUTPUT_DIR/logs/custom_mutex_t${THREADS}.log"
SEGMENTED_CAS_LOG="$OUTPUT_DIR/logs/segmented_cas_t${THREADS}.log"
SEGMENTED_MUTEX_LOG="$OUTPUT_DIR/logs/segmented_mutex_t${THREADS}.log"

echo "Running custom sequential baseline..."
"$CUSTOM_BENCH_BIN" "$TABLE_SIZE" "$OPS" 0 1 "$PROBING" "$KEY_DIST" "$REPS" | tee "$CUSTOM_SEQ_LOG"

echo "Running std::unordered_map baseline..."
"$BASELINE_BENCH_BIN" 0 "$TABLE_SIZE" "$OPS" "$KEY_DIST" "$REPS" | tee "$STD_LOG"

echo "Running custom CAS benchmark with ${THREADS} thread(s)..."
"$CUSTOM_BENCH_BIN" "$TABLE_SIZE" "$OPS" 1 "$THREADS" "$PROBING" "$KEY_DIST" "$REPS" | tee "$CAS_LOG"

echo "Running custom mutex benchmark with ${THREADS} thread(s)..."
"$CUSTOM_BENCH_BIN" "$TABLE_SIZE" "$OPS" 2 "$THREADS" "$PROBING" "$KEY_DIST" "$REPS" | tee "$MUTEX_LOG"

echo "Running segmented CAS benchmark with ${THREADS} thread(s)..."
"$CUSTOM_BENCH_BIN" "$TABLE_SIZE" "$OPS" 3 "$THREADS" "$PROBING" "$KEY_DIST" "$REPS" | tee "$SEGMENTED_CAS_LOG"

echo "Running segmented mutex benchmark with ${THREADS} thread(s)..."
"$CUSTOM_BENCH_BIN" "$TABLE_SIZE" "$OPS" 4 "$THREADS" "$PROBING" "$KEY_DIST" "$REPS" | tee "$SEGMENTED_MUTEX_LOG"

HAS_ABSL="$("$BASELINE_BENCH_BIN" --has-absl)"
ABSL_AVAILABLE=0
if [[ "$HAS_ABSL" == "1" ]]; then
    echo "Running absl::flat_hash_map baseline..."
    "$BASELINE_BENCH_BIN" 1 "$TABLE_SIZE" "$OPS" "$KEY_DIST" "$REPS" | tee "$ABSL_LOG"
    ABSL_AVAILABLE=1
fi

STD_MEAN="$(extract_metric "Mean time" "$STD_LOG")"

append_row() {
    local implementation="$1"
    local threads="$2"
    local log_file="$3"
    local reference_std="$4"
    local reference_absl="${5:-}"

    local mean_value
    mean_value="$(extract_metric "Mean time" "$log_file")"
    local stddev_value
    stddev_value="$(extract_metric "Std dev" "$log_file")"
    local min_value
    min_value="$(extract_metric "Min time" "$log_file")"
    local max_value
    max_value="$(extract_metric "Max time" "$log_file")"

    local speedup_std
    speedup_std="$(compute_speedup "$reference_std" "$mean_value")"

    local speedup_absl="N/A"
    if [[ -n "$reference_absl" ]]; then
        speedup_absl="$(compute_speedup "$reference_absl" "$mean_value")"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$implementation" \
        "$threads" \
        "$mean_value" \
        "$stddev_value" \
        "$min_value" \
        "$max_value" \
        "$speedup_std" \
        "$speedup_absl" >> "$SUMMARY_FILE"
}

ABSL_MEAN=""
if [[ "$ABSL_AVAILABLE" == "1" ]]; then
    ABSL_MEAN="$(extract_metric "Mean time" "$ABSL_LOG")"
fi

append_row "Custom sequential" "1" "$CUSTOM_SEQ_LOG" "$STD_MEAN" "$ABSL_MEAN"
append_row "std::unordered_map" "1" "$STD_LOG" "$STD_MEAN" "$ABSL_MEAN"
if [[ "$ABSL_AVAILABLE" == "1" ]]; then
    append_row "absl::flat_hash_map" "1" "$ABSL_LOG" "$STD_MEAN" "$ABSL_MEAN"
else
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "absl::flat_hash_map" \
        "1" \
        "Unavailable" \
        "Unavailable" \
        "Unavailable" \
        "Unavailable" \
        "Unavailable" \
        "Unavailable" >> "$SUMMARY_FILE"
fi
append_row "Custom CAS" "$THREADS" "$CAS_LOG" "$STD_MEAN" "$ABSL_MEAN"
append_row "Custom Mutex" "$THREADS" "$MUTEX_LOG" "$STD_MEAN" "$ABSL_MEAN"
append_row "Segmented CAS" "$THREADS" "$SEGMENTED_CAS_LOG" "$STD_MEAN" "$ABSL_MEAN"
append_row "Segmented Mutex" "$THREADS" "$SEGMENTED_MUTEX_LOG" "$STD_MEAN" "$ABSL_MEAN"

echo
echo "Saved summary to: $SUMMARY_FILE"
echo "Saved raw logs to: $OUTPUT_DIR/logs"
