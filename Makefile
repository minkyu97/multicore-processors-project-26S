CXX ?= g++
CC ?= gcc

BUILD_DIR ?= build
CPP_BENCH_TARGET := $(BUILD_DIR)/bench_hashmap
CPP_TEST_TARGET := $(BUILD_DIR)/test_hashmap
C_TARGET := $(BUILD_DIR)/bench_hashmap_c
C_TEST_TARGET := $(BUILD_DIR)/test_hashmap_c
CPP_BENCH_SRCS := src/bench_hashmap.cpp src/hashmap.cpp
CPP_TEST_SRCS := src/test_hashmap.cpp src/hashmap.cpp
CPP_BENCH_OBJS := $(CPP_BENCH_SRCS:src/%.cpp=$(BUILD_DIR)/%.o)
CPP_TEST_OBJS := $(CPP_TEST_SRCS:src/%.cpp=$(BUILD_DIR)/%.o)
CPP_OBJS := $(sort $(CPP_BENCH_OBJS) $(CPP_TEST_OBJS))
CPP_DEPS := $(CPP_OBJS:.o=.d)

CPPFLAGS ?= -Iinclude
CXXFLAGS ?= -std=c++23 -Wall -Wextra -Wpedantic
CFLAGS ?= -std=c99 -Wall -Wextra -Wpedantic -O2
OPENMP_FLAGS ?= -fopenmp
LDLIBS ?= -lm

.PHONY: all clean bench_hashmap test_hashmap bench_hashmap_c

all: $(CPP_BENCH_TARGET) $(CPP_TEST_TARGET) $(C_TARGET) $(C_TEST_TARGET)

bench_hashmap: $(CPP_BENCH_TARGET)

test_hashmap: $(CPP_TEST_TARGET)

bench_hashmap_c: $(C_TARGET)

test_hashmap_c: $(C_TEST_TARGET)

$(CPP_BENCH_TARGET): $(CPP_BENCH_OBJS)
	$(CXX) $(OPENMP_FLAGS) $^ -o $@

$(CPP_TEST_TARGET): $(CPP_TEST_OBJS)
	$(CXX) $(OPENMP_FLAGS) $^ -o $@

$(BUILD_DIR)/%.o: src/%.cpp include/hashtable.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP_FLAGS) -MMD -MP -c $< -o $@

$(C_TARGET): src/hashmap.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) $< -o $@ $(LDLIBS)

$(C_TEST_TARGET): src/test_hashmap.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) $^ -o $@ $(LDLIBS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -f $(CPP_BENCH_TARGET) $(CPP_TEST_TARGET) $(C_TARGET) $(CPP_OBJS) $(CPP_DEPS)

-include $(CPP_DEPS)
