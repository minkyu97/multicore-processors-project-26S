#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/bench_common.sh"

DEFAULT_OUTPUT_DIR="$ROOT_DIR/benchmark_results/load_factors_$(date +%Y%m%d_%H%M%S)"

TABLE_SIZE=10000000
LOAD_FACTORS_STRING="50 70 85 95"
THREADS=8
KEY_DIST=1
REPS=7
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
BENCH_BIN=""

usage() {
    cat <<'EOF'
Usage:
  bench_table_load_factors.sh [options]

Options:
  --table-size N      Hash table size (default: 10000000)
  --load-factors "..." Space-separated load factors in percent (default: "50 70 85 95")
  --threads N         Thread count for CAS rows (default: 8)
  --key-dist N        0=sequential, 1=random, 2=zipf (default: 1)
  --reps N            Benchmark repetitions (default: 7)
  --output-dir PATH   Directory for logs and summary output
  --bench-bin PATH    Path to bench_hashmap binary
  --help              Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --table-size)
            TABLE_SIZE="$2"
            shift 2
            ;;
        --load-factors)
            LOAD_FACTORS_STRING="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
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

if [[ "$THREADS" -lt 1 ]]; then
    echo "--threads must be >= 1" >&2
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

if [[ "$TABLE_SIZE" -lt 2 ]]; then
    echo "--table-size must be >= 2" >&2
    exit 1
fi

BENCH_BIN="$(resolve_bench_bin "$ROOT_DIR" "$BENCH_BIN")"

mkdir -p "$OUTPUT_DIR/logs"

CONFIG_FILE="$OUTPUT_DIR/run_config.txt"
SUMMARY_FILE="$OUTPUT_DIR/summary.tsv"

cat > "$CONFIG_FILE" <<EOF
bench_bin=$BENCH_BIN
table_size=$TABLE_SIZE
load_factors=$LOAD_FACTORS_STRING
threads=$THREADS
key_dist=$KEY_DIST
reps=$REPS
EOF

printf 'Load Factor\tOps\tProbing\tBackend\tMean (s)\tStd Dev\tMin (s)\tMax (s)\tSpeedup vs Seq\n' > "$SUMMARY_FILE"

read -r -a LOAD_FACTORS <<< "$LOAD_FACTORS_STRING"

for load_factor in "${LOAD_FACTORS[@]}"; do
    if [[ "$load_factor" -le 0 || "$load_factor" -ge 100 ]]; then
        echo "Each load factor must be between 1 and 99: $load_factor" >&2
        exit 1
    fi

    ops=$((TABLE_SIZE * load_factor / 100))
    if [[ "$ops" -lt 1 || "$ops" -ge "$TABLE_SIZE" ]]; then
        echo "Computed ops must satisfy 1 <= ops < table_size for ${load_factor}%" >&2
        exit 1
    fi

    for probing in 0 1; do
        probing_name="$(probing_label "$probing")"
        probing_slug="$(tr '[:upper:]' '[:lower:]' <<< "$probing_name")"

        seq_log="$OUTPUT_DIR/logs/lf${load_factor}_${probing_slug}_sequential.log"
        cas_log="$OUTPUT_DIR/logs/lf${load_factor}_${probing_slug}_cas_t${THREADS}.log"

        echo "Running sequential benchmark at ${load_factor}% load with ${probing_name} probing..."
        "$BENCH_BIN" "$TABLE_SIZE" "$ops" 0 1 "$probing" "$KEY_DIST" "$REPS" | tee "$seq_log"

        seq_mean="$(extract_metric "Mean time" "$seq_log")"
        seq_stddev="$(extract_metric "Std dev" "$seq_log")"
        seq_min="$(extract_metric "Min time" "$seq_log")"
        seq_max="$(extract_metric "Max time" "$seq_log")"

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${load_factor}%" \
            "$(format_commas "$ops")" \
            "$probing_name" \
            "Sequential" \
            "$seq_mean" \
            "$seq_stddev" \
            "$seq_min" \
            "$seq_max" \
            "1" >> "$SUMMARY_FILE"

        echo "Running CAS benchmark at ${load_factor}% load with ${probing_name} probing..."
        "$BENCH_BIN" "$TABLE_SIZE" "$ops" 1 "$THREADS" "$probing" "$KEY_DIST" "$REPS" | tee "$cas_log"

        cas_mean="$(extract_metric "Mean time" "$cas_log")"
        cas_stddev="$(extract_metric "Std dev" "$cas_log")"
        cas_min="$(extract_metric "Min time" "$cas_log")"
        cas_max="$(extract_metric "Max time" "$cas_log")"
        cas_speedup="$(compute_speedup "$seq_mean" "$cas_mean")"

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${load_factor}%" \
            "$(format_commas "$ops")" \
            "$probing_name" \
            "CAS (${THREADS}T)" \
            "$cas_mean" \
            "$cas_stddev" \
            "$cas_min" \
            "$cas_max" \
            "$cas_speedup" >> "$SUMMARY_FILE"
    done
done

echo
echo "Saved summary to: $SUMMARY_FILE"
echo "Saved raw logs to: $OUTPUT_DIR/logs"
