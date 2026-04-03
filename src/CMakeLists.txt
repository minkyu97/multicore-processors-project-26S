find_package(OpenMP REQUIRED COMPONENTS C CXX)
add_executable(test_hashmap test_hashmap.cpp hashmap.cpp)
target_link_libraries(test_hashmap PRIVATE OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
target_include_directories(test_hashmap PUBLIC ${CMAKE_SOURCE_DIR}/include)