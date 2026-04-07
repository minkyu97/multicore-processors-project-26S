#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <omp.h>

#if __has_include(<absl/container/flat_hash_map.h>)
#include <absl/container/flat_hash_map.h>
#define HASHMAP_HAS_ABSL 1
#else
#define HASHMAP_HAS_ABSL 0
#endif

namespace {

constexpr int IMPL_STD_UNORDERED = 0;
constexpr int IMPL_ABSL_FLAT = 1;

constexpr int KEYS_SEQUENTIAL = 0;
constexpr int KEYS_RANDOM = 1;
constexpr int KEYS_ZIPF = 2;

constexpr int MAX_REPS = 20;

double mean(const std::array<double, MAX_REPS>& values, int count) {
    double sum = 0.0;
    for (int i = 0; i < count; ++i) {
        sum += values[i];
    }
    return sum / count;
}

double stddev(const std::array<double, MAX_REPS>& values, int count, double avg) {
    double sum = 0.0;
    for (int i = 0; i < count; ++i) {
        const double delta = values[i] - avg;
        sum += delta * delta;
    }
    return std::sqrt(sum / count);
}

double arr_min(const std::array<double, MAX_REPS>& values, int count) {
    double min_value = values[0];
    for (int i = 1; i < count; ++i) {
        if (values[i] < min_value) {
            min_value = values[i];
        }
    }
    return min_value;
}

double arr_max(const std::array<double, MAX_REPS>& values, int count) {
    double max_value = values[0];
    for (int i = 1; i < count; ++i) {
        if (values[i] > max_value) {
            max_value = values[i];
        }
    }
    return max_value;
}

double trimmed_mean(const std::array<double, MAX_REPS>& values, int count) {
    if (count < 3) {
        return mean(values, count);
    }

    const double min_value = arr_min(values, count);
    const double max_value = arr_max(values, count);
    double sum = 0.0;
    int remaining = 0;
    bool dropped_min = false;
    bool dropped_max = false;

    for (int i = 0; i < count; ++i) {
        if (!dropped_min && values[i] == min_value) {
            dropped_min = true;
            continue;
        }
        if (!dropped_max && values[i] == max_value) {
            dropped_max = true;
            continue;
        }
        sum += values[i];
        ++remaining;
    }

    return remaining > 0 ? sum / remaining : mean(values, count);
}

const char* implementation_name(int implementation) {
    switch (implementation) {
        case IMPL_STD_UNORDERED:
            return "std::unordered_map";
        case IMPL_ABSL_FLAT:
            return "absl::flat_hash_map";
        default:
            return "Unknown";
    }
}

const char* dist_name(int key_dist) {
    switch (key_dist) {
        case KEYS_SEQUENTIAL:
            return "Sequential";
        case KEYS_RANDOM:
            return "Random";
        case KEYS_ZIPF:
            return "Zipf (skewed)";
        default:
            return "Unknown";
    }
}

void generate_keys(std::vector<int>& keys, std::vector<int>& values, int table_size, int dist) {
    switch (dist) {
        case KEYS_SEQUENTIAL:
            for (std::size_t i = 0; i < keys.size(); ++i) {
                keys[i] = static_cast<int>(i) + 1;
                values[i] = keys[i] * 10;
            }
            break;

        case KEYS_RANDOM:
            std::srand(42);
            for (std::size_t i = 0; i < keys.size(); ++i) {
                keys[i] = (std::rand() % (table_size * 10)) + 1;
                values[i] = keys[i] * 10;
            }
            break;

        case KEYS_ZIPF: {
            std::srand(42);
            int vocab = static_cast<int>(keys.size()) / 5;
            if (vocab < 1) {
                vocab = 1;
            }

            std::vector<double> cumulative(static_cast<std::size_t>(vocab) + 1, 0.0);
            double total = 0.0;
            for (int rank = 1; rank <= vocab; ++rank) {
                total += 1.0 / static_cast<double>(rank);
            }

            for (int rank = 1; rank <= vocab; ++rank) {
                cumulative[rank] =
                    cumulative[rank - 1] + (1.0 / static_cast<double>(rank)) / total;
            }

            for (std::size_t i = 0; i < keys.size(); ++i) {
                const double u = static_cast<double>(std::rand()) /
                    static_cast<double>(RAND_MAX);
                int lo = 1;
                int hi = vocab;
                int rank = vocab;
                while (lo <= hi) {
                    const int mid = (lo + hi) / 2;
                    if (cumulative[mid] >= u) {
                        rank = mid;
                        hi = mid - 1;
                    } else {
                        lo = mid + 1;
                    }
                }
                keys[i] = rank;
                values[i] = rank * 10;
            }
            break;
        }

        default:
            std::fprintf(stderr, "Unknown key distribution: %d\n", dist);
            std::exit(1);
    }
}

template <typename Map>
void prepare_map(Map& map, std::size_t reserve_count, float load_factor) {
    map.clear();
    if constexpr (requires(Map & current, float factor) { current.max_load_factor(factor); }) {
        map.max_load_factor(load_factor);
    }
    map.reserve(reserve_count);
}

template <typename Map>
int run_map_ops(Map& map, const std::vector<int>& keys, const std::vector<int>& values) {
    for (std::size_t i = 0; i < keys.size(); ++i) {
        map.emplace(keys[i], values[i]);
    }

    int found = 0;
    for (int key : keys) {
        if (map.find(key) != map.end()) {
            ++found;
        }
    }

    return found;
}

template <typename Map>
int benchmark_map(Map& map,
                  const std::vector<int>& keys,
                  const std::vector<int>& values,
                  int reps,
                  std::size_t reserve_count,
                  float load_factor,
                  std::array<double, MAX_REPS>& times) {
    std::printf("Running warm-up...\n");
    prepare_map(map, reserve_count, load_factor);
    int last_found = run_map_ops(map, keys, values);

    std::printf("Running %d timed repetitions...\n", reps);
    for (int rep = 0; rep < reps; ++rep) {
        prepare_map(map, reserve_count, load_factor);
        const double start = omp_get_wtime();
        last_found = run_map_ops(map, keys, values);
        const double end = omp_get_wtime();

        times[rep] = end - start;
        std::printf("  Rep %2d: %.6f seconds\n", rep + 1, times[rep]);
    }

    return last_found;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc == 2 && std::string_view(argv[1]) == "--has-absl") {
        std::printf("%s\n", HASHMAP_HAS_ABSL ? "1" : "0");
        return 0;
    }

    if (argc != 6) {
        std::fprintf(stderr,
                     "usage: bench_map_baselines impl table_size num_ops key_dist reps\n");
        std::fprintf(stderr, "  impl      : 0=std::unordered_map, 1=absl::flat_hash_map\n");
        std::fprintf(stderr, "  table_size: slots in custom table (used to derive target load factor)\n");
        std::fprintf(stderr, "  num_ops   : keys to insert/search\n");
        std::fprintf(stderr, "  key_dist  : 0=sequential, 1=random, 2=zipf\n");
        std::fprintf(stderr, "  reps      : repetitions (recommended: 7)\n");
        return 1;
    }

    const int implementation = std::atoi(argv[1]);
    const int table_size = std::atoi(argv[2]);
    const int num_ops = std::atoi(argv[3]);
    const int key_dist = std::atoi(argv[4]);
    const int reps = std::atoi(argv[5]);

    if (implementation != IMPL_STD_UNORDERED && implementation != IMPL_ABSL_FLAT) {
        std::fprintf(stderr, "impl must be 0 or 1\n");
        return 1;
    }
    if (reps < 1 || reps > MAX_REPS) {
        std::fprintf(stderr, "reps must be between 1 and %d\n", MAX_REPS);
        return 1;
    }
    if (num_ops >= table_size) {
        std::fprintf(stderr, "num_ops must be < table_size\n");
        return 1;
    }
    if (implementation == IMPL_ABSL_FLAT && !HASHMAP_HAS_ABSL) {
        std::fprintf(stderr, "absl::flat_hash_map is not available in this build\n");
        return 2;
    }

    const float load_factor = static_cast<float>(num_ops) /
        static_cast<float>(table_size);

    std::printf("========================================\n");
    std::printf("Table size  : %d\n", table_size);
    std::printf("Operations  : %d\n", num_ops);
    std::printf("Load factor : %.2f (%.0f%%)\n", load_factor, load_factor * 100.0f);
    std::printf("Implementation: %s\n", implementation_name(implementation));
    std::printf("Threads     : 1\n");
    std::printf("Key dist    : %s\n", dist_name(key_dist));
    std::printf("Repetitions : %d (drops min+max, averages rest)\n", reps);
    std::printf("========================================\n");

    std::vector<int> keys(static_cast<std::size_t>(num_ops));
    std::vector<int> values(static_cast<std::size_t>(num_ops));
    generate_keys(keys, values, table_size, key_dist);

    std::array<double, MAX_REPS> times{};
    int last_found_count = 0;
    const std::size_t reserve_count = static_cast<std::size_t>(num_ops);

    if (implementation == IMPL_STD_UNORDERED) {
        std::unordered_map<int, int> map;
        last_found_count = benchmark_map(map, keys, values, reps, reserve_count, load_factor, times);
    }
#if HASHMAP_HAS_ABSL
    else {
        absl::flat_hash_map<int, int> map;
        last_found_count = benchmark_map(map, keys, values, reps, reserve_count, load_factor, times);
    }
#endif

    if (last_found_count == num_ops) {
        std::printf("Result is correct!\n");
    } else {
        std::printf("MISMATCH: found %d / %d keys\n", last_found_count, num_ops);
    }

    const double raw_mean = mean(times, reps);
    const double trimmed = trimmed_mean(times, reps);
    const double deviation = stddev(times, reps, raw_mean);
    const double min_time = arr_min(times, reps);
    const double max_time = arr_max(times, reps);

    std::printf("----------------------------------------\n");
    std::printf("Mean time   : %.6f seconds (trimmed)\n", trimmed);
    std::printf("Std dev     : %.6f seconds\n", deviation);
    std::printf("Min time    : %.6f seconds\n", min_time);
    std::printf("Max time    : %.6f seconds\n", max_time);
    std::printf("CV (%%RSD)   : %.2f%%\n", (deviation / raw_mean) * 100.0);
    std::printf("========================================\n");

    return 0;
}
