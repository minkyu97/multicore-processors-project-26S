CMAKE ?= cmake
CTEST ?= ctest
BUILD_DIR ?= build
CONFIGURE_ARGS ?=

.PHONY: all configure clean test bench_hashmap bench_map_baselines test_hashmap bench_hashmap_c test_hashmap_c

all: configure
	$(CMAKE) --build $(BUILD_DIR)

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) $(CONFIGURE_ARGS)

bench_hashmap: configure
	$(CMAKE) --build $(BUILD_DIR) --target bench_hashmap

bench_map_baselines: configure
	$(CMAKE) --build $(BUILD_DIR) --target bench_map_baselines

test_hashmap: configure
	$(CMAKE) --build $(BUILD_DIR) --target test_hashmap

bench_hashmap_c: configure
	$(CMAKE) --build $(BUILD_DIR) --target bench_hashmap_c

test_hashmap_c: configure
	$(CMAKE) --build $(BUILD_DIR) --target test_hashmap_c

test: configure
	$(CMAKE) --build $(BUILD_DIR) --target test_hashmap
	$(CTEST) --test-dir $(BUILD_DIR) --output-on-failure

clean:
	@if [ -d "$(BUILD_DIR)" ]; then $(CMAKE) --build $(BUILD_DIR) --target clean; fi
