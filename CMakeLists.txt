cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

project(
    hashmap
    VERSION 0.0.1
    DESCRIPTION "Concurrent hash map benchmarks and tests"
    LANGUAGES C CXX
)

option(HASHMAP_ENABLE_ABSL "Enable Abseil baseline benchmark integration when available" ON)
set(HASHMAP_ABSL_ROOT "" CACHE PATH "Prefix where Abseil is installed")

if(HASHMAP_ABSL_ROOT)
    list(PREPEND CMAKE_PREFIX_PATH "${HASHMAP_ABSL_ROOT}")
endif()

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src)
