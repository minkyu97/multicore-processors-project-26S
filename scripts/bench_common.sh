#!/usr/bin/env bash

resolve_cmake_binary() {
    local root_dir="$1"
    local target_name="$2"
    local requested_bin="${3:-}"

    if [[ -n "$requested_bin" ]]; then
        if [[ ! -x "$requested_bin" ]]; then
            echo "$target_name binary is not executable: $requested_bin" >&2
            return 1
        fi
        printf '%s\n' "$requested_bin"
        return 0
    fi

    local candidates=(
        "$root_dir/build/src/$target_name"
        "$root_dir/build/$target_name"
    )

    local candidate=""
    for candidate in "${candidates[@]}"; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    if [[ -d "$root_dir/build" ]]; then
        echo "$target_name binary not found. Building target..." >&2
        cmake --build "$root_dir/build" --target "$target_name" >&2
        for candidate in "${candidates[@]}"; do
            if [[ -x "$candidate" ]]; then
                printf '%s\n' "$candidate"
                return 0
            fi
        done
    fi

    echo "Could not find $target_name. Build the target first or pass an explicit path." >&2
    return 1
}

resolve_bench_bin() {
    local root_dir="$1"
    local requested_bin="${2:-}"
    resolve_cmake_binary "$root_dir" "bench_hashmap" "$requested_bin"
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
    local sequential_mean="$1"
    local current_mean="$2"
    awk -v sequential_mean="$sequential_mean" -v current_mean="$current_mean" '
        BEGIN {
            if (current_mean == 0) {
                print "0.00"
            } else {
                printf "%.2f", sequential_mean / current_mean
            }
        }
    '
}

compute_throughput_mops() {
    local ops="$1"
    local mean_seconds="$2"
    awk -v ops="$ops" -v mean_seconds="$mean_seconds" '
        BEGIN {
            if (mean_seconds == 0) {
                print "0.00"
            } else {
                printf "%.2f", ops / mean_seconds / 1000000.0
            }
        }
    '
}

format_commas() {
    awk -v value="$1" '
        BEGIN {
            s = sprintf("%.0f", value)
            out = ""
            while (length(s) > 3) {
                out = "," substr(s, length(s) - 2) out
                s = substr(s, 1, length(s) - 3)
            }
            print s out
        }
    '
}

format_compact_size() {
    awk -v value="$1" '
        BEGIN {
            if (value % 1000000 == 0) {
                printf "%.0fM\n", value / 1000000
            } else if (value % 1000 == 0) {
                printf "%.0fK\n", value / 1000
            } else {
                s = sprintf("%.0f", value)
                out = ""
                while (length(s) > 3) {
                    out = "," substr(s, length(s) - 2) out
                    s = substr(s, 1, length(s) - 3)
                }
                print s out
            }
        }
    '
}

probing_label() {
    case "$1" in
        0) printf 'Linear\n' ;;
        1) printf 'Quadratic\n' ;;
        *)
            echo "Unknown probing mode: $1" >&2
            return 1
            ;;
    esac
}

key_dist_label() {
    case "$1" in
        0) printf 'Sequential Keys\n' ;;
        1) printf 'Random Keys\n' ;;
        2) printf 'Zipf (skewed)\n' ;;
        *)
            echo "Unknown key distribution: $1" >&2
            return 1
            ;;
    esac
}

key_dist_slug() {
    case "$1" in
        0) printf 'sequential\n' ;;
        1) printf 'random\n' ;;
        2) printf 'zipf\n' ;;
        *)
            echo "Unknown key distribution: $1" >&2
            return 1
            ;;
    esac
}
