#include "hashtable.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <omp.h>

namespace {

constexpr int EMPTY = EMPTY_KEY;
constexpr int DELETED = DELETED_KEY;

constexpr int KEYS_SEQUENTIAL = 0;
constexpr int KEYS_RANDOM = 1;
constexpr int KEYS_ZIPF = 2;

using LinearSequentialTable = SequentialHashTable<int, int, ProbingStrategy::LINEAR>;
using QuadraticSequentialTable = SequentialHashTable<int, int, ProbingStrategy::QUADRATIC>;
using LinearParallelTable = ParallelHashTable<int, int, ProbingStrategy::LINEAR>;
using QuadraticParallelTable = ParallelHashTable<int, int, ProbingStrategy::QUADRATIC>;

int pass_count = 0;
int fail_count = 0;

#define TEST(name, cond) do { \
    if (cond) { \
        std::printf("  PASS : %s\n", name); \
        pass_count++; \
    } else { \
        std::printf("  FAIL : %s\n", name); \
        fail_count++; \
    } \
} while (0)

#define SECTION(name) do { \
    std::printf("\n════════════════════════════════════════\n"); \
    std::printf("  %s\n", name); \
    std::printf("════════════════════════════════════════\n"); \
} while (0)

double mean_fn(const double* arr, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += arr[i];
    }
    return sum / n;
}

double stddev_fn(const double* arr, int n, double mean) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const double delta = arr[i] - mean;
        sum += delta * delta;
    }
    return std::sqrt(sum / n);
}

double arr_min(const double* arr, int n) {
    double min_value = arr[0];
    for (int i = 1; i < n; ++i) {
        if (arr[i] < min_value) {
            min_value = arr[i];
        }
    }
    return min_value;
}

double arr_max(const double* arr, int n) {
    double max_value = arr[0];
    for (int i = 1; i < n; ++i) {
        if (arr[i] > max_value) {
            max_value = arr[i];
        }
    }
    return max_value;
}

double trimmed_mean(const double* arr, int n) {
    if (n < 3) {
        return mean_fn(arr, n);
    }

    const double min_value = arr_min(arr, n);
    const double max_value = arr_max(arr, n);
    double sum = 0.0;
    int count = 0;
    bool dropped_min = false;
    bool dropped_max = false;

    for (int i = 0; i < n; ++i) {
        if (!dropped_min && arr[i] == min_value) {
            dropped_min = true;
            continue;
        }
        if (!dropped_max && arr[i] == max_value) {
            dropped_max = true;
            continue;
        }
        sum += arr[i];
        ++count;
    }

    return count > 0 ? sum / count : mean_fn(arr, n);
}

void generate_keys(int* keys, int* values, int num_ops, int table_size, int dist) {
    switch (dist) {
        case KEYS_SEQUENTIAL:
            for (int i = 0; i < num_ops; ++i) {
                keys[i] = i + 1;
                values[i] = (i + 1) * 10;
            }
            break;

        case KEYS_RANDOM:
            std::srand(42);
            for (int i = 0; i < num_ops; ++i) {
                keys[i] = (std::rand() % (table_size * 10)) + 1;
                values[i] = keys[i] * 10;
            }
            break;

        case KEYS_ZIPF: {
            std::srand(42);
            int vocab = num_ops / 5;
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

            for (int i = 0; i < num_ops; ++i) {
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
    }
}

template <typename Table>
bool table_contains(Table& table, int key) {
    int value = 0;
    return table.get(key, value);
}

template <typename Table>
int table_get(Table& table, int key) {
    int value = 0;
    return table.get(key, value) ? value : EMPTY;
}

template <typename Table>
int count_occupied(const Table& table) {
    return static_cast<int>(table.size());
}

template <typename Table>
int run_sequential_ops(Table& table,
                       const std::vector<int>& keys,
                       const std::vector<int>& values) {
    for (std::size_t i = 0; i < keys.size(); ++i) {
        table.insert(keys[i], values[i]);
    }

    int found = 0;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        int value = 0;
        if (table.get(keys[i], value)) {
            ++found;
        }
    }

    return found;
}

template <typename Table>
int run_parallel_ops(Table& table,
                     const std::vector<int>& keys,
                     const std::vector<int>& values) {
    table.insert_batch(keys, values);

    std::vector<int> out_values;
    std::vector<bool> found;
    table.get_batch(keys, out_values, found);
    return static_cast<int>(std::count(found.begin(), found.end(), true));
}

template <typename Table>
int count_mismatches(Table& table, const std::vector<int>& keys) {
    int mismatches = 0;
    for (int key : keys) {
        if (!table_contains(table, key)) {
            ++mismatches;
        }
    }
    return mismatches;
}

void test_hash_function() {
    SECTION("1. Hash Function");

    const int capacity = 1000000;
    LinearSequentialTable table(capacity);
    LinearSequentialTable table_a(1000);
    LinearSequentialTable table_b(999);

    int ok = 1;
    for (int key = 1; key <= 10000; ++key) {
        const int hash = static_cast<int>(table.hash(key));
        if (hash < 0 || hash >= capacity) {
            ok = 0;
            break;
        }
    }
    TEST("hash() output always in [0, table_size)", ok);

    TEST("hash() different table sizes give different ranges",
         table_a.hash(42) != table_b.hash(42) || table_a.hash(1) != table_b.hash(1));

    TEST("hash() is deterministic",
         table.hash(12345) == table.hash(12345) &&
         table.hash(99999) == table.hash(99999));

    int collisions = 0;
    for (int i = 1; i <= 1000; ++i) {
        if (table.hash(i) == table.hash(i + 1)) {
            ++collisions;
        }
    }
    TEST("hash() distributes keys (collisions < 5% of 1000 pairs)",
         collisions < 50);
}

void test_next_slot() {
    SECTION("2. Probing Step (next_slot)");

    const int capacity = 1000;

    TEST("Linear: attempt 0 returns start",
         ProbingStrategy::LINEAR::next_slot(100, 0, capacity) == 100);
    TEST("Linear: attempt 1 returns start+1",
         ProbingStrategy::LINEAR::next_slot(100, 1, capacity) == 101);
    TEST("Linear: attempt N-1 wraps around",
         ProbingStrategy::LINEAR::next_slot(1, capacity - 1, capacity) == 0);
    TEST("Linear: attempt 0 at slot 0",
         ProbingStrategy::LINEAR::next_slot(0, 0, capacity) == 0);

    TEST("Quadratic: attempt 0 returns start",
         ProbingStrategy::QUADRATIC::next_slot(100, 0, capacity) == 100);
    TEST("Quadratic: attempt 1 returns start+1",
         ProbingStrategy::QUADRATIC::next_slot(100, 1, capacity) == 101);
    TEST("Quadratic: attempt 2 returns start+4",
         ProbingStrategy::QUADRATIC::next_slot(100, 2, capacity) == 104);
    TEST("Quadratic: attempt 3 returns start+9",
         ProbingStrategy::QUADRATIC::next_slot(100, 3, capacity) == 109);
    TEST("Quadratic: wraps correctly",
         ProbingStrategy::QUADRATIC::next_slot(999, 2, capacity) == (999 + 4) % capacity);

    TEST("CAS mode: same step as linear",
         ProbingStrategy::LINEAR::next_slot(100, 3, capacity) == 103);
}

void test_init_table() {
    SECTION("3. Table Initialization");

    const int capacity = 10000;
    LinearSequentialTable seq_table(capacity);
    LinearParallelTable par_table(capacity, 4, ParallelBackend::CAS);

    for (int i = 0; i < 1000; ++i) {
        seq_table.insert(i + 1, i + 2);
        par_table.insert(i + 1, i + 2);
    }

    seq_table.clear();
    par_table.clear();

    TEST("Sequential clear resets occupied slots to 0", seq_table.size() == 0);
    TEST("Parallel clear resets occupied slots to 0", par_table.size() == 0);
    TEST("clear() removes previously inserted keys",
         !table_contains(seq_table, 1) && !table_contains(par_table, 1));
    TEST("EMPTY sentinel value is -1", EMPTY == -1);
    TEST("DELETED sentinel value is -2", DELETED == -2);
}

void test_generate_keys() {
    SECTION("4. Key Generation");

    const int count = 1000;
    const int table_size = 10000;
    std::vector<int> keys(count);
    std::vector<int> values(count);

    generate_keys(keys.data(), values.data(), count, table_size, KEYS_SEQUENTIAL);
    TEST("Sequential: keys[0] == 1", keys[0] == 1);
    TEST("Sequential: keys[N-1] == N", keys[count - 1] == count);
    TEST("Sequential: values[i] == keys[i]*10", values[5] == keys[5] * 10);
    TEST("Sequential: keys are unique",
         keys[0] != keys[1] && keys[count - 2] != keys[count - 1]);

    generate_keys(keys.data(), values.data(), count, table_size, KEYS_RANDOM);
    int all_positive = 1;
    for (int key : keys) {
        if (key < 1) {
            all_positive = 0;
            break;
        }
    }
    TEST("Random: all keys >= 1", all_positive);
    TEST("Random: values[i] == keys[i]*10", values[10] == keys[10] * 10);

    int is_sequential = 1;
    for (int i = 1; i < count; ++i) {
        if (keys[i] != keys[i - 1] + 1) {
            is_sequential = 0;
            break;
        }
    }
    TEST("Random: keys are not perfectly sequential", !is_sequential);

    generate_keys(keys.data(), values.data(), count, table_size, KEYS_ZIPF);
    int zipf_positive = 1;
    for (int key : keys) {
        if (key < 1) {
            zipf_positive = 0;
            break;
        }
    }
    TEST("Zipf: all keys >= 1", zipf_positive);
    TEST("Zipf: values[i] == keys[i]*10", values[0] == keys[0] * 10);

    int count_1 = 0;
    int count_high = 0;
    const int vocab = count / 5;
    for (int key : keys) {
        if (key == 1) {
            ++count_1;
        }
        if (key == vocab) {
            ++count_high;
        }
    }
    TEST("Zipf: key=1 appears more than key=vocab (skewed distribution)",
         count_1 > count_high);
}

void test_seq_hash_ops() {
    SECTION("5. Sequential Hash Operations");

    const int capacity = 10000;
    const int num_ops = capacity / 2;
    std::vector<int> keys(num_ops);
    std::vector<int> values(num_ops);

    LinearSequentialTable linear_table(capacity);
    generate_keys(keys.data(), values.data(), num_ops, capacity, KEYS_SEQUENTIAL);
    linear_table.clear();
    run_sequential_ops(linear_table, keys, values);

    int found = 0;
    for (int key : keys) {
        if (table_contains(linear_table, key)) {
            ++found;
        }
    }
    TEST("Sequential: all inserted keys found", found == num_ops);

    int correct_vals = 1;
    for (int i = 0; i < num_ops; ++i) {
        if (table_get(linear_table, keys[i]) != values[i]) {
            correct_vals = 0;
            break;
        }
    }
    TEST("Sequential: all values correct after insert", correct_vals);
    TEST("Sequential: occupied slots == unique inserts",
         count_occupied(linear_table) == num_ops);

    linear_table.clear();
    linear_table.insert(42, 420);
    linear_table.insert(42, 999);
    linear_table.insert(99, 990);
    TEST("Sequential: duplicate keys not double-inserted", count_occupied(linear_table) == 2);
    TEST("Sequential: duplicate insert keeps original value", table_get(linear_table, 42) == 420);

    QuadraticSequentialTable quadratic_table(capacity);
    quadratic_table.clear();
    generate_keys(keys.data(), values.data(), num_ops, capacity, KEYS_SEQUENTIAL);
    run_sequential_ops(quadratic_table, keys, values);
    found = 0;
    for (int key : keys) {
        if (table_contains(quadratic_table, key)) {
            ++found;
        }
    }
    TEST("Quadratic: all inserted keys found", found == num_ops);

    linear_table.clear();
    generate_keys(keys.data(), values.data(), num_ops, capacity, KEYS_RANDOM);
    run_sequential_ops(linear_table, keys, values);
    int any_found = 0;
    for (int key : keys) {
        if (table_contains(linear_table, key)) {
            any_found = 1;
            break;
        }
    }
    TEST("Sequential: random keys inserted and findable", any_found);
}

template <typename Table>
void run_parallel_correctness_case(const char* label,
                                   Table& table,
                                   int table_size,
                                   int num_ops,
                                   int key_dist) {
    std::vector<int> keys(num_ops);
    std::vector<int> values(num_ops);
    generate_keys(keys.data(), values.data(), num_ops, table_size, key_dist);
    table.clear();
    run_parallel_ops(table, keys, values);
    TEST(label, count_mismatches(table, keys) == 0);
}

void test_parallel_correctness() {
    SECTION("6. Parallel Correctness (all probing modes)");

    const int table_size = 1000000;
    const int num_ops = 500000;
    const int num_threads = 8;

    LinearParallelTable linear_table(table_size, num_threads, ParallelBackend::CAS);
    run_parallel_correctness_case("Linear + sequential keys (8 threads)",
                                  linear_table, table_size, num_ops, KEYS_SEQUENTIAL);
    run_parallel_correctness_case("Linear + random keys (8 threads)",
                                  linear_table, table_size, num_ops, KEYS_RANDOM);
    run_parallel_correctness_case("Linear + Zipf keys (8 threads)",
                                  linear_table, table_size, num_ops, KEYS_ZIPF);

    QuadraticParallelTable quadratic_table(table_size, num_threads, ParallelBackend::CAS);
    run_parallel_correctness_case("Quadratic + sequential keys (8 threads)",
                                  quadratic_table, table_size, num_ops, KEYS_SEQUENTIAL);
    run_parallel_correctness_case("Quadratic + random keys (8 threads)",
                                  quadratic_table, table_size, num_ops, KEYS_RANDOM);
    run_parallel_correctness_case("Quadratic + Zipf keys (8 threads)",
                                  quadratic_table, table_size, num_ops, KEYS_ZIPF);

    LinearParallelTable cas_table(table_size, num_threads, ParallelBackend::CAS);
    run_parallel_correctness_case("CAS + sequential keys (8 threads)",
                                  cas_table, table_size, num_ops, KEYS_SEQUENTIAL);
    run_parallel_correctness_case("CAS + random keys (8 threads)",
                                  cas_table, table_size, num_ops, KEYS_RANDOM);
    run_parallel_correctness_case("CAS + Zipf keys (8 threads)",
                                  cas_table, table_size, num_ops, KEYS_ZIPF);

    LinearParallelTable mutex_table(table_size, num_threads, ParallelBackend::MUTEX);
    run_parallel_correctness_case("Mutex + sequential keys (8 threads)",
                                  mutex_table, table_size, num_ops, KEYS_SEQUENTIAL);
    run_parallel_correctness_case("Mutex + random keys (8 threads)",
                                  mutex_table, table_size, num_ops, KEYS_RANDOM);
    run_parallel_correctness_case("Mutex + Zipf keys (8 threads)",
                                  mutex_table, table_size, num_ops, KEYS_ZIPF);
}

void test_thread_scaling_correctness() {
    SECTION("7. Thread Scaling Correctness (linear, sequential keys)");

    const int table_size = 1000000;
    const int num_ops = 500000;
    std::vector<int> keys(num_ops);
    std::vector<int> values(num_ops);
    generate_keys(keys.data(), values.data(), num_ops, table_size, KEYS_SEQUENTIAL);

    const int thread_counts[] = {1, 2, 4, 8, 16, 32, 64};
    char label[64];

    for (int thread_count : thread_counts) {
        LinearParallelTable table(table_size, thread_count, ParallelBackend::CAS);
        table.clear();
        const int found = run_parallel_ops(table, keys, values);
        std::snprintf(label, sizeof(label), "%d thread(s): all %d keys found",
                      thread_count, num_ops);
        TEST(label, found == num_ops);
    }
}

void test_edge_cases() {
    SECTION("8. Edge Cases");

    {
        const int table_size = 100;
        const int num_ops = 10;
        std::vector<int> keys(num_ops);
        std::vector<int> values(num_ops);
        generate_keys(keys.data(), values.data(), num_ops, table_size, KEYS_SEQUENTIAL);

        LinearParallelTable table(table_size, 4, ParallelBackend::CAS);
        table.clear();
        const int found = run_parallel_ops(table, keys, values);
        TEST("Tiny table (100 slots, 10 ops): all keys found", found == num_ops);
    }

    {
        const int table_size = 1000000;
        const int num_ops = 100000;
        std::vector<int> keys(num_ops);
        std::vector<int> values(num_ops);
        generate_keys(keys.data(), values.data(), num_ops, table_size, KEYS_SEQUENTIAL);

        LinearParallelTable table(table_size, 8, ParallelBackend::CAS);
        table.clear();
        const int found = run_parallel_ops(table, keys, values);
        TEST("Low load 10%: all keys found", found == num_ops);
    }

    {
        const int table_size = 1000000;
        const int num_ops = 900000;
        std::vector<int> keys(num_ops);
        std::vector<int> values(num_ops);
        generate_keys(keys.data(), values.data(), num_ops, table_size, KEYS_SEQUENTIAL);

        LinearParallelTable table(table_size, 8, ParallelBackend::CAS);
        table.clear();
        const int found = run_parallel_ops(table, keys, values);
        TEST("High load 90%: all keys found", found == num_ops);
    }

    {
        const int table_size = 100000;
        const int num_ops = 50000;
        std::vector<int> keys(num_ops);
        std::vector<int> values(num_ops);
        generate_keys(keys.data(), values.data(), num_ops, table_size, KEYS_SEQUENTIAL);

        LinearSequentialTable seq_table(table_size);
        LinearParallelTable par_table(table_size, 1, ParallelBackend::CAS);
        seq_table.clear();
        par_table.clear();
        run_sequential_ops(seq_table, keys, values);
        run_parallel_ops(par_table, keys, values);

        int match = 1;
        for (int key : keys) {
            if (!table_contains(seq_table, key) || !table_contains(par_table, key)) {
                match = 0;
                break;
            }
        }
        TEST("1-thread parallel: same result as sequential", match);
    }

    {
        const int table_size = 1000;
        const int num_ops = 100;
        std::vector<int> keys(num_ops, 42);
        std::vector<int> values(num_ops, 420);

        LinearParallelTable table(table_size, 4, ParallelBackend::CAS);
        table.clear();
        run_parallel_ops(table, keys, values);
        TEST("All-duplicate keys: table holds exactly 1 slot", count_occupied(table) == 1);
        TEST("All-duplicate keys: key 42 is findable", table_contains(table, 42));
    }
}

void test_statistics() {
    SECTION("9. Statistics Helpers");

    const double arr1[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    TEST("mean([1,2,3,4,5]) == 3.0", std::fabs(mean_fn(arr1, 5) - 3.0) < 1e-9);

    const double arr2[] = {10.0, 10.0};
    TEST("mean([10,10]) == 10.0", std::fabs(mean_fn(arr2, 2) - 10.0) < 1e-9);

    const double arr3[] = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    const double mean3 = mean_fn(arr3, 8);
    const double stddev = stddev_fn(arr3, 8, mean3);
    TEST("stddev of known dataset is ~2.0", std::fabs(stddev - 2.0) < 0.01);

    const double arr4[] = {5.0, 5.0, 5.0};
    TEST("stddev of constant array is 0.0",
         std::fabs(stddev_fn(arr4, 3, 5.0)) < 1e-9);

    const double arr5[] = {1.0, 2.0, 3.0, 4.0, 100.0};
    TEST("trimmed_mean drops min and max outlier",
         std::fabs(trimmed_mean(arr5, 5) - 3.0) < 1e-9);

    const double arr6[] = {5.0, 5.0};
    TEST("trimmed_mean with n=2 falls back to regular mean",
         std::fabs(trimmed_mean(arr6, 2) - 5.0) < 1e-9);

    const double arr7[] = {3.0, 1.0, 2.0};
    TEST("trimmed_mean with n=3 returns middle value",
         std::fabs(trimmed_mean(arr7, 3) - 2.0) < 1e-9);
}

void test_performance_sanity() {
    SECTION("10. Performance Sanity Checks");

    const int table_size = 1000000;
    const int num_ops = 500000;
    std::vector<int> keys(num_ops);
    std::vector<int> values(num_ops);
    generate_keys(keys.data(), values.data(), num_ops, table_size, KEYS_SEQUENTIAL);

    LinearSequentialTable seq_table(table_size);
    seq_table.clear();
    double start = omp_get_wtime();
    run_sequential_ops(seq_table, keys, values);
    const double seq_time = omp_get_wtime() - start;
    std::printf("  INFO : Sequential 1M/500K : %.4f s\n", seq_time);
    TEST("Sequential completes in < 5 seconds", seq_time < 5.0);

    LinearParallelTable par_table(table_size, 8, ParallelBackend::CAS);
    par_table.clear();
    start = omp_get_wtime();
    run_parallel_ops(par_table, keys, values);
    const double par_time = omp_get_wtime() - start;
    std::printf("  INFO : Parallel 8-thread 1M/500K : %.4f s\n", par_time);
    TEST("Parallel (8 threads) completes in < 5 seconds", par_time < 5.0);
    std::printf("  INFO : Speedup vs sequential  : %.2fx\n", seq_time / par_time);

    LinearParallelTable mutex_table(table_size, 8, ParallelBackend::MUTEX);
    mutex_table.clear();
    start = omp_get_wtime();
    run_parallel_ops(mutex_table, keys, values);
    const double mutex_time = omp_get_wtime() - start;
    std::printf("  INFO : Mutex 8-thread 1M/500K  : %.4f s\n", mutex_time);
    TEST("Mutex mode completes in < 30 seconds", mutex_time < 30.0);
}

}  // namespace

int main() {
    std::printf("\n");
    std::printf("════════════════════════════════════════\n");
    std::printf("  test_hashmap.cpp — C++ Unit Test Suite\n");
    std::printf("════════════════════════════════════════\n");

    test_hash_function();
    test_next_slot();
    test_init_table();
    test_generate_keys();
    test_seq_hash_ops();
    test_parallel_correctness();
    test_thread_scaling_correctness();
    test_edge_cases();
    test_statistics();
    test_performance_sanity();

    std::printf("\n════════════════════════════════════════\n");
    std::printf("  RESULTS\n");
    std::printf("════════════════════════════════════════\n");
    std::printf("  Passed : %d\n", pass_count);
    std::printf("  Failed : %d\n", fail_count);
    std::printf("  Total  : %d\n", pass_count + fail_count);
    if (fail_count == 0) {
        std::printf("  ALL TESTS PASSED\n");
    } else {
        std::printf("  %d TEST(S) FAILED\n", fail_count);
    }
    std::printf("════════════════════════════════════════\n\n");

    return fail_count > 0 ? 1 : 0;
}
