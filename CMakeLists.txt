cmake_minimum_required(VERSION 3.10)

project(
    hashmap
    VERSION 0.0.1
    DESCRIPTION "A concurrent hash map implementation in C++"
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src)
