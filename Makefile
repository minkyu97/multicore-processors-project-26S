CXX ?= g++
CC ?= gcc

BUILD_DIR ?= build
CPP_TARGET := $(BUILD_DIR)/test_hashmap
C_TARGET := $(BUILD_DIR)/hashmap_c
CPP_SRCS := src/test_hashmap.cpp src/hashmap.cpp
CPP_OBJS := $(CPP_SRCS:src/%.cpp=$(BUILD_DIR)/%.o)
CPP_DEPS := $(CPP_OBJS:.o=.d)

CPPFLAGS ?= -Iinclude
CXXFLAGS ?= -std=c++23 -Wall -Wextra -Wpedantic
CFLAGS ?= -std=c99 -Wall -Wextra -Wpedantic -O2
OPENMP_FLAGS ?= -fopenmp
LDLIBS ?= -lm

.PHONY: all clean test_hashmap hashmap_c

all: $(CPP_TARGET) $(C_TARGET)

test_hashmap: $(CPP_TARGET)

hashmap_c: $(C_TARGET)

$(CPP_TARGET): $(CPP_OBJS)
	$(CXX) $(OPENMP_FLAGS) $^ -o $@

$(BUILD_DIR)/%.o: src/%.cpp include/hashtable.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP_FLAGS) -MMD -MP -c $< -o $@

$(C_TARGET): src/hashmap.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) $< -o $@ $(LDLIBS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -f $(CPP_TARGET) $(C_TARGET) $(CPP_OBJS) $(CPP_DEPS)

-include $(CPP_DEPS)
