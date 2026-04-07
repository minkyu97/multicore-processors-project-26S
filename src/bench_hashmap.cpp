#include "hashtable.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int PROBE_LINEAR = 0;
constexpr int PROBE_QUADRATIC = 1;
constexpr int PROBE_CAS = 2;
constexpr int PROBE_MUTEX = 3;

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

const char* probe_name(int probing) {
    switch (probing) {
        case PROBE_LINEAR:
            return "Linear probing (CAS)";
        case PROBE_QUADRATIC:
            return "Quadratic probing (CAS)";
        case PROBE_CAS:
            return "CAS lock-free (linear)";
        case PROBE_MUTEX:
            return "Mutex locking (linear)";
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

ParallelBackend backend_for_mode(int probing) {
    return probing == PROBE_MUTEX ? ParallelBackend::MUTEX : ParallelBackend::CAS;
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

std::vector<bool> count_finds_sequential(const std::vector<int>& keys,
                                         const std::vector<int>& values,
                                         int table_size,
                                         int num_threads,
                                         int probing) {
    std::vector<bool> found(keys.size(), false);
    const ParallelBackend backend = backend_for_mode(probing);

    if (probing == PROBE_QUADRATIC) {
        ParallelHashTable<int, int, ProbingStrategy::QUADRATIC> table(
            table_size, num_threads, backend);
        for (std::size_t i = 0; i < keys.size(); ++i) {
            table.insert(keys[i], values[i]);
        }
        for (std::size_t i = 0; i < keys.size(); ++i) {
            int out_value = 0;
            found[i] = table.get(keys[i], out_value);
        }
    } else {
        ParallelHashTable<int, int, ProbingStrategy::LINEAR> table(
            table_size, num_threads, backend);
        for (std::size_t i = 0; i < keys.size(); ++i) {
            table.insert(keys[i], values[i]);
        }
        for (std::size_t i = 0; i < keys.size(); ++i) {
            int out_value = 0;
            found[i] = table.get(keys[i], out_value);
        }
    }

    return found;
}

std::vector<bool> count_finds_parallel(const std::vector<int>& keys,
                                       const std::vector<int>& values,
                                       int table_size,
                                       int num_threads,
                                       int probing) {
    std::vector<int> out_values;
    std::vector<bool> found;
    const ParallelBackend backend = backend_for_mode(probing);

    if (probing == PROBE_QUADRATIC) {
        ParallelHashTable<int, int, ProbingStrategy::QUADRATIC> table(
            table_size, num_threads, backend);
        table.insert_batch(keys, values);
        table.get_batch(keys, out_values, found);
    } else {
        ParallelHashTable<int, int, ProbingStrategy::LINEAR> table(
            table_size, num_threads, backend);
        table.insert_batch(keys, values);
        table.get_batch(keys, out_values, found);
    }

    return found;
}

std::vector<bool> run_hash_ops(int which_code,
                               int table_size,
                               int num_threads,
                               int probing,
                               const std::vector<int>& keys,
                               const std::vector<int>& values) {
    if (which_code == 0) {
        return count_finds_sequential(keys, values, table_size, 1, probing);
    }
    return count_finds_parallel(keys, values, table_size, num_threads, probing);
}

void check_result(const std::vector<bool>& found) {
    const int mismatches = static_cast<int>(
        std::count(found.begin(), found.end(), false));
    if (mismatches == 0) {
        std::printf("Result is correct!\n");
    } else {
        std::printf("MISMATCH: %d keys not found\n", mismatches);
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 8) {
        std::fprintf(stderr,
                     "usage: test_hashmap table_size num_ops who threads probing key_dist reps\n");
        std::fprintf(stderr,
                     "  table_size : slots in hash table (e.g. 1000000, 10000000)\n");
        std::fprintf(stderr, "  num_ops    : keys to insert/search\n");
        std::fprintf(stderr, "  who        : 0=sequential, 1=parallel\n");
        std::fprintf(stderr, "  threads    : 1,2,4,8,16,32,64\n");
        std::fprintf(stderr, "  probing    : 0=linear, 1=quadratic, 2=CAS, 3=mutex\n");
        std::fprintf(stderr, "  key_dist   : 0=sequential, 1=random, 2=zipf\n");
        std::fprintf(stderr, "  reps       : repetitions (recommended: 7)\n");
        return 1;
    }

    const int table_size = std::atoi(argv[1]);
    const int num_ops = std::atoi(argv[2]);
    const int which_code = std::atoi(argv[3]);
    const int num_threads = std::atoi(argv[4]);
    const int probing = std::atoi(argv[5]);
    const int key_dist = std::atoi(argv[6]);
    const int reps = std::atoi(argv[7]);

    if (reps < 1 || reps > MAX_REPS) {
        std::fprintf(stderr, "reps must be between 1 and %d\n", MAX_REPS);
        return 1;
    }
    if (num_ops >= table_size) {
        std::fprintf(stderr, "num_ops must be < table_size\n");
        return 1;
    }

    const float load_factor = static_cast<float>(num_ops) /
        static_cast<float>(table_size);

    std::printf("========================================\n");
    std::printf("Table size  : %d\n", table_size);
    std::printf("Operations  : %d\n", num_ops);
    std::printf("Load factor : %.2f (%.0f%%)\n", load_factor, load_factor * 100.0f);
    std::printf("Threads     : %d\n", num_threads);
    std::printf("Probing     : %s\n", probe_name(probing));
    std::printf("Key dist    : %s\n", dist_name(key_dist));
    std::printf("Repetitions : %d (drops min+max, averages rest)\n", reps);
    std::printf("========================================\n");

    std::vector<int> keys(static_cast<std::size_t>(num_ops));
    std::vector<int> values(static_cast<std::size_t>(num_ops));
    generate_keys(keys, values, table_size, key_dist);

    std::array<double, MAX_REPS> times{};
    std::vector<bool> last_found;

    std::printf("Running warm-up...\n");
    last_found = run_hash_ops(which_code, table_size, num_threads, probing, keys, values);

    std::printf("Running %d timed repetitions...\n", reps);
    for (int rep = 0; rep < reps; ++rep) {
        const double start = omp_get_wtime();
        last_found = run_hash_ops(which_code, table_size, num_threads, probing, keys, values);
        const double end = omp_get_wtime();

        times[rep] = end - start;
        std::printf("  Rep %2d: %.6f seconds\n", rep + 1, times[rep]);
    }

    if (which_code == 1) {
        check_result(last_found);
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
