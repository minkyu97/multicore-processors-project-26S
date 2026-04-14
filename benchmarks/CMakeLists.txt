if(MSVC)
    set(OPENMP_HASH_TABLE_BENCHMARK_OPT_FLAGS /O2)
else()
    set(OPENMP_HASH_TABLE_BENCHMARK_OPT_FLAGS -O3)
endif()

add_executable(bench_hashmap bench_hashmap.cpp)
target_compile_features(bench_hashmap PRIVATE cxx_std_20)
target_compile_options(bench_hashmap PRIVATE ${OPENMP_HASH_TABLE_BENCHMARK_OPT_FLAGS})
target_link_libraries(
    bench_hashmap
    PRIVATE
        OpenMPHashTable::hashtable
        openmp_hash_table_warnings
)

add_executable(bench_map_baselines bench_map_baselines.cpp)
target_compile_features(bench_map_baselines PRIVATE cxx_std_20)
target_compile_options(bench_map_baselines PRIVATE ${OPENMP_HASH_TABLE_BENCHMARK_OPT_FLAGS})
target_link_libraries(
    bench_map_baselines
    PRIVATE
        OpenMP::OpenMP_CXX
        openmp_hash_table_warnings
)
if(TARGET absl::flat_hash_map)
    target_link_libraries(bench_map_baselines PRIVATE absl::flat_hash_map)
endif()

if(OPENMP_HASH_TABLE_BUILD_LEGACY_C)
    add_executable(bench_hashmap_c bench_hashmap_c.c)
    target_compile_options(bench_hashmap_c PRIVATE ${OPENMP_HASH_TABLE_BENCHMARK_OPT_FLAGS})
    target_link_libraries(
        bench_hashmap_c
        PRIVATE
            OpenMP::OpenMP_C
            openmp_hash_table_warnings
    )
    if(NOT MSVC)
        target_link_libraries(bench_hashmap_c PRIVATE m)
    endif()
endif()
