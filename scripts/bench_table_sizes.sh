#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/bench_common.sh"

DEFAULT_OUTPUT_DIR="$ROOT_DIR/benchmark_results/table_sizes_$(date +%Y%m%d_%H%M%S)"

TABLE_SIZES_STRING="100000 1000000 10000000 50000000"
LOAD_FACTOR=50
THREADS=8
PROBING=0
KEY_DIST=1
REPS=7
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
BENCH_BIN=""

usage() {
    cat <<'EOF'
Usage:
  bench_table_sizes.sh [options]

Options:
  --table-sizes "..." Space-separated table sizes (default: "100000 1000000 10000000 50000000")
  --load-factor N     Load factor in percent (default: 50)
  --threads N         Thread count for CAS rows (default: 8)
  --probing N         0=linear, 1=quadratic (default: 0)
  --key-dist N        0=sequential, 1=random, 2=zipf (default: 1)
  --reps N            Benchmark repetitions (default: 7)
  --output-dir PATH   Directory for logs and summary output
  --bench-bin PATH    Path to bench_hashmap binary
  --help              Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --table-sizes)
            TABLE_SIZES_STRING="$2"
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

BENCH_BIN="$(resolve_bench_bin "$ROOT_DIR" "$BENCH_BIN")"

mkdir -p "$OUTPUT_DIR/logs"

CONFIG_FILE="$OUTPUT_DIR/run_config.txt"
SUMMARY_FILE="$OUTPUT_DIR/summary.tsv"

cat > "$CONFIG_FILE" <<EOF
bench_bin=$BENCH_BIN
table_sizes=$TABLE_SIZES_STRING
load_factor=$LOAD_FACTOR
threads=$THREADS
probing=$PROBING
key_dist=$KEY_DIST
reps=$REPS
throughput_formula=ops/mean_seconds/1e6
EOF

printf 'Table Size\tOps\tBackend\tMean (s)\tStd Dev\tMin (s)\tMax (s)\tThroughput (Mops/s)\tSpeedup vs Seq\n' > "$SUMMARY_FILE"

read -r -a TABLE_SIZES <<< "$TABLE_SIZES_STRING"

for table_size in "${TABLE_SIZES[@]}"; do
    if [[ "$table_size" -lt 2 ]]; then
        echo "Each table size must be >= 2: $table_size" >&2
        exit 1
    fi

    ops=$((table_size * LOAD_FACTOR / 100))
    if [[ "$ops" -lt 1 || "$ops" -ge "$table_size" ]]; then
        echo "Computed ops must satisfy 1 <= ops < table_size for size $table_size" >&2
        exit 1
    fi

    size_label="$(format_compact_size "$table_size")"
    seq_log="$OUTPUT_DIR/logs/size_${table_size}_sequential.log"
    cas_log="$OUTPUT_DIR/logs/size_${table_size}_cas_t${THREADS}.log"

    echo "Running sequential benchmark for table size ${table_size}..."
    "$BENCH_BIN" "$table_size" "$ops" 0 1 "$PROBING" "$KEY_DIST" "$REPS" | tee "$seq_log"

    seq_mean="$(extract_metric "Mean time" "$seq_log")"
    seq_stddev="$(extract_metric "Std dev" "$seq_log")"
    seq_min="$(extract_metric "Min time" "$seq_log")"
    seq_max="$(extract_metric "Max time" "$seq_log")"
    seq_throughput="$(compute_throughput_mops "$ops" "$seq_mean")"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$size_label" \
        "$(format_commas "$ops")" \
        "Sequential" \
        "$seq_mean" \
        "$seq_stddev" \
        "$seq_min" \
        "$seq_max" \
        "$seq_throughput" \
        "1" >> "$SUMMARY_FILE"

    echo "Running CAS benchmark for table size ${table_size} with ${THREADS} thread(s)..."
    "$BENCH_BIN" "$table_size" "$ops" 1 "$THREADS" "$PROBING" "$KEY_DIST" "$REPS" | tee "$cas_log"

    cas_mean="$(extract_metric "Mean time" "$cas_log")"
    cas_stddev="$(extract_metric "Std dev" "$cas_log")"
    cas_min="$(extract_metric "Min time" "$cas_log")"
    cas_max="$(extract_metric "Max time" "$cas_log")"
    cas_throughput="$(compute_throughput_mops "$ops" "$cas_mean")"
    cas_speedup="$(compute_speedup "$seq_mean" "$cas_mean")"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$size_label" \
        "$(format_commas "$ops")" \
        "CAS (${THREADS}T)" \
        "$cas_mean" \
        "$cas_stddev" \
        "$cas_min" \
        "$cas_max" \
        "$cas_throughput" \
        "$cas_speedup" >> "$SUMMARY_FILE"
done

echo
echo "Saved summary to: $SUMMARY_FILE"
echo "Saved raw logs to: $OUTPUT_DIR/logs"
