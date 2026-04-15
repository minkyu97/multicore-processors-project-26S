# OpenMP Hash Table

`OpenMP Hash Table` is a header-only C++20 library for fixed-capacity open-addressing hash tables designed for OpenMP workloads. The project includes:

- a sequential hash table,
- a parallel hash table with `CAS` and `MUTEX` backends,
- a small C++ test program,
- benchmark binaries and shell scripts for thread scaling, load factor, table size, key distribution, and baseline comparisons against standard library maps.

The library is intended for experiments and benchmarking on integer-key workloads rather than as a fully general-purpose replacement for `std::unordered_map`.

## Repository Layout

- `include/hashtable.hpp`: public library entrypoint
- `include/detail/hashtable.hpp`: internal header containing declarations and template definitions
- `include/SeparateChainTable.h`: separate-chaining prototype header kept in the repository, but not integrated into the default CMake targets
- `tests/`: C++ and optional legacy C tests
- `benchmarks/`: benchmark executables
- `scripts/`: benchmark driver scripts that generate TSV summaries and raw logs

## Requirements

### Core library and C++ test

- CMake 3.16 or newer
- A C++20 compiler
- OpenMP for C++

Tested configuration in this repository currently uses Clang/clangd with `libomp` on macOS, but the CMake project is not tied to a specific compiler.

### Benchmark and script requirements

- Everything required for the core library
- `bash`
- `awk`
- `tee`

The shell scripts use standard Unix command-line tools and expect the benchmark binaries to exist under `build/bin/` or another path passed explicitly.

### Optional dependencies

- Abseil, if you want the `absl::flat_hash_map` baseline in `bench_map_baselines`
- OpenMP for C, if you want the legacy C benchmark and test

Abseil is optional. If it is not found, the baseline benchmark still builds, but the Abseil row is unavailable.

## Build

### Configure and build with CMake

```bash
cmake -S . -B build
cmake --build build -j
```

This builds:

- `build/bin/test_hashmap`
- `build/bin/bench_hashmap`
- `build/bin/bench_map_baselines`

By default, the top-level build enables:

- `BUILD_TESTING=ON`
- `OPENMP_HASH_TABLE_BUILD_BENCHMARKS=ON`
- `OPENMP_HASH_TABLE_BUILD_LEGACY_C=OFF`
- `OPENMP_HASH_TABLE_ENABLE_ABSL=ON`

### Build with Make

The repository also provides a small `Makefile` wrapper:

```bash
make
```

Useful targets:

```bash
make test
make bench_hashmap
make bench_map_baselines
```

### Common configuration options

Disable benchmarks:

```bash
cmake -S . -B build -DOPENMP_HASH_TABLE_BUILD_BENCHMARKS=OFF
```

Disable tests:

```bash
cmake -S . -B build -DBUILD_TESTING=OFF
```

Enable legacy C targets:

```bash
cmake -S . -B build -DOPENMP_HASH_TABLE_BUILD_LEGACY_C=ON
```

Point CMake at an Abseil installation:

```bash
cmake -S . -B build -DOPENMP_HASH_TABLE_ABSL_ROOT=/path/to/abseil/install
```

## Install and Consume

Install the header-only package into a prefix:

```bash
cmake -S . -B build
cmake --build build -j
cmake --install build --prefix /path/to/install
```

The project exports the CMake package target:

- `OpenMPHashTable::hashtable`

Example consumer `CMakeLists.txt`:

```cmake
find_package(OpenMPHashTable REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE OpenMPHashTable::hashtable)
```

Example consumer source:

```cpp
#include <hashtable.hpp>

int main() {
    ParallelHashTable<int, int> table(1'000'000, 8, ParallelBackend::CAS);
    table.insert(1, 10);
    return 0;
}
```

## Implemented Functionality

### Data structures

- `SequentialHashTable<K, V, ProbingStrategy>`
- `ParallelHashTable<K, V, ProbingStrategy>`

### Probing modes

- `ProbingStrategy::LINEAR`
- `ProbingStrategy::QUADRATIC`

### Parallel backends

- `ParallelBackend::CAS`
- `ParallelBackend::MUTEX`

### Operations

- `insert`
- `get`
- `remove`
- `clear`
- `size`
- `hash`
- `insert_batch`
- `get_batch`
- `remove_batch`

### Workload support in the bundled benchmarks

- sequential keys
- random keys
- Zipf-like skewed keys
- single-thread sequential runs
- multi-thread CAS and mutex runs
- optional comparisons against `std::unordered_map` and `absl::flat_hash_map`

## Limitations

- The implementation is fixed-capacity. There is no automatic resize or rehash.
- Sentinel values are hard-coded as `EMPTY_KEY = -1` and `DELETED_KEY = -2`. This makes the current implementation best suited to integer-like key domains that avoid those reserved values.
- The benchmark and test suite are written around `int` keys and values. The templates are generic, but the project is primarily exercised with integer workloads.
- The CAS backend requires trivially copyable key types.
- The data structure is not a drop-in replacement for `std::unordered_map`. It does not provide iterators, heterogeneous lookup, allocators, or the broader STL container interface.
- The current benchmark comparison against `std::unordered_map` and `absl::flat_hash_map` is a workload comparison, not a claim of full feature equivalence.
- The legacy C code path is optional and not part of the default build.

## Running Tests

### C++ test

Build and run through CTest:

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Or run the executable directly:

```bash
./build/bin/test_hashmap
```

The C++ test exercises:

- hash function behavior
- insert/get/remove basics
- duplicate handling
- probing behavior
- batch APIs
- parallel correctness
- simple performance sanity checks

### Legacy C test

Enable the legacy targets first:

```bash
cmake -S . -B build -DOPENMP_HASH_TABLE_BUILD_LEGACY_C=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Or run it directly:

```bash
./build/bin/test_hashmap_c
```

## Running Benchmarks

### Benchmark executables

Build the benchmark targets:

```bash
cmake -S . -B build
cmake --build build -j --target bench_hashmap bench_map_baselines
```

Main binaries:

- `./build/bin/bench_hashmap`
- `./build/bin/bench_map_baselines`

### `bench_hashmap`

Usage:

```bash
./build/bin/bench_hashmap table_size num_ops who threads probing key_dist reps
```

Arguments:

- `who`: `0=sequential`, `1=parallel-cas`, `2=parallel-mutex`
- `threads`: thread count for the run
- `probing`: `0=linear`, `1=quadratic`
- `key_dist`: `0=sequential`, `1=random`, `2=zipf`

Example:

```bash
./build/bin/bench_hashmap 10000000 5000000 1 16 0 1 7
```

### `bench_map_baselines`

Usage:

```bash
./build/bin/bench_map_baselines impl table_size num_ops key_dist reps
```

Arguments:

- `impl`: `0=std::unordered_map`, `1=absl::flat_hash_map`
- `key_dist`: `0=sequential`, `1=random`, `2=zipf`

Example:

```bash
./build/bin/bench_map_baselines 0 10000000 5000000 1 7
```

Check whether the current build has Abseil support:

```bash
./build/bin/bench_map_baselines --has-absl
```

### Benchmark scripts

The scripts in `scripts/` generate a timestamped output directory under `benchmark_results/` with:

- `summary.tsv`
- `run_config.txt`
- raw per-run logs under `logs/`

Examples:

Thread scaling:

```bash
./scripts/bench_table_threads.sh
```

Table size sweep:

```bash
./scripts/bench_table_sizes.sh
```

Load factor sweep:

```bash
./scripts/bench_table_load_factors.sh
```

Key distribution sweep:

```bash
./scripts/bench_table_key_distribution.sh
```

Comparison against STL and Abseil baselines:

```bash
./scripts/compare_32t_baselines.sh
```

You can override defaults such as thread count, probing mode, key distribution, repetition count, output directory, or binary path. For example:

```bash
./scripts/bench_table_threads.sh --threads "1 2 4 8 16 32" --reps 7
./scripts/compare_32t_baselines.sh --threads 32 --load-factor 50 --key-dist 1
```

## Notes on Fairness and Interpretation

The custom hash table is specialized for:

- fixed capacity,
- open addressing,
- integer-oriented workloads,
- batch OpenMP execution.

The standard-library and Abseil baselines are more general-purpose containers. Benchmark results should therefore be interpreted as workload-specific comparisons, not as a claim that the custom implementation is a universal substitute for those libraries.
