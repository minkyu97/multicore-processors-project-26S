cmake_minimum_required(VERSION 3.16)

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

project(
    OpenMPHashTable
    VERSION 0.1.0
    DESCRIPTION "Concurrent open-addressing hash table for OpenMP workloads"
    LANGUAGES C CXX
)

set(OPENMP_HASH_TABLE_TOP_LEVEL OFF)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(OPENMP_HASH_TABLE_TOP_LEVEL ON)
endif()

option(OPENMP_HASH_TABLE_ENABLE_ABSL "Enable Abseil baseline benchmark integration when available" ON)
option(OPENMP_HASH_TABLE_BUILD_BENCHMARKS "Build benchmark executables" ${OPENMP_HASH_TABLE_TOP_LEVEL})
option(OPENMP_HASH_TABLE_BUILD_LEGACY_C "Build legacy C benchmarks and tests" OFF)
option(BUILD_TESTING "Build the test suite" ${OPENMP_HASH_TABLE_TOP_LEVEL})
set(OPENMP_HASH_TABLE_ABSL_ROOT "" CACHE PATH "Prefix where Abseil is installed")

if(OPENMP_HASH_TABLE_ABSL_ROOT)
    list(PREPEND CMAKE_PREFIX_PATH "${OPENMP_HASH_TABLE_ABSL_ROOT}")
endif()

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(OPENMP_HASH_TABLE_OPENMP_COMPONENTS CXX)
if(OPENMP_HASH_TABLE_BUILD_LEGACY_C)
    list(PREPEND OPENMP_HASH_TABLE_OPENMP_COMPONENTS C)
endif()
find_package(OpenMP REQUIRED COMPONENTS ${OPENMP_HASH_TABLE_OPENMP_COMPONENTS})

if(OPENMP_HASH_TABLE_ENABLE_ABSL AND OPENMP_HASH_TABLE_BUILD_BENCHMARKS)
    find_package(absl CONFIG QUIET)
endif()

add_library(openmp_hash_table_warnings INTERFACE)
target_compile_options(
    openmp_hash_table_warnings
    INTERFACE
        $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -Wpedantic -O2>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(hashtable INTERFACE)
add_library(OpenMPHashTable::hashtable ALIAS hashtable)

target_compile_features(hashtable INTERFACE cxx_std_20)
target_include_directories(
    hashtable
    INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(hashtable INTERFACE OpenMP::OpenMP_CXX)

if(OPENMP_HASH_TABLE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

set(OPENMP_HASH_TABLE_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/OpenMPHashTable")

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/OpenMPHashTableConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/OpenMPHashTableConfig.cmake"
    INSTALL_DESTINATION "${OPENMP_HASH_TABLE_CMAKE_INSTALL_DIR}"
)

write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/OpenMPHashTableConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

install(
    TARGETS hashtable
    EXPORT OpenMPHashTableTargets
)

install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(
    EXPORT OpenMPHashTableTargets
    NAMESPACE OpenMPHashTable::
    DESTINATION "${OPENMP_HASH_TABLE_CMAKE_INSTALL_DIR}"
)

install(
    FILES
        "${PROJECT_BINARY_DIR}/OpenMPHashTableConfig.cmake"
        "${PROJECT_BINARY_DIR}/OpenMPHashTableConfigVersion.cmake"
    DESTINATION "${OPENMP_HASH_TABLE_CMAKE_INSTALL_DIR}"
)
