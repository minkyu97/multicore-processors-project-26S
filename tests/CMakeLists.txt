add_executable(test_hashmap test_hashmap.cpp)
target_compile_features(test_hashmap PRIVATE cxx_std_20)
target_link_libraries(
    test_hashmap
    PRIVATE
        OpenMPHashTable::hashtable
        openmp_hash_table_warnings
)

add_test(NAME openmp_hash_table_cpp COMMAND test_hashmap)

if(OPENMP_HASH_TABLE_BUILD_LEGACY_C)
    add_executable(test_hashmap_c test_hashmap_c.c)
    target_link_libraries(
        test_hashmap_c
        PRIVATE
            OpenMP::OpenMP_C
            openmp_hash_table_warnings
    )
    if(NOT MSVC)
        target_link_libraries(test_hashmap_c PRIVATE m)
    endif()

    add_test(NAME openmp_hash_table_c COMMAND test_hashmap_c)
endif()
