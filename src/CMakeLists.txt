find_package(OpenMP REQUIRED COMPONENTS C CXX)

if(HASHMAP_ENABLE_ABSL)
    find_package(absl CONFIG QUIET)
endif()

add_library(hashmap_warnings INTERFACE)
target_compile_options(
    hashmap_warnings
    INTERFACE
        $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -Wpedantic -O2>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(bench_hashmap bench_hashmap.cpp hashmap.cpp)
target_link_libraries(bench_hashmap PRIVATE OpenMP::OpenMP_CXX hashmap_warnings)
target_include_directories(bench_hashmap PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(bench_map_baselines bench_map_baselines.cpp)
target_link_libraries(bench_map_baselines PRIVATE OpenMP::OpenMP_CXX hashmap_warnings)
if(TARGET absl::flat_hash_map)
    target_link_libraries(bench_map_baselines PRIVATE absl::flat_hash_map)
endif()

add_executable(test_hashmap test_hashmap.cpp hashmap.cpp)
target_link_libraries(test_hashmap PRIVATE OpenMP::OpenMP_CXX hashmap_warnings)
target_include_directories(test_hashmap PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(bench_hashmap_c hashmap.c)
target_link_libraries(bench_hashmap_c PRIVATE OpenMP::OpenMP_C hashmap_warnings m)

add_executable(test_hashmap_c test_hashmap.c)
target_link_libraries(test_hashmap_c PRIVATE OpenMP::OpenMP_C hashmap_warnings m)
