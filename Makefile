CMAKE ?= cmake
CTEST ?= ctest
ZIP ?= zip
BUILD_DIR ?= build
CONFIGURE_ARGS ?=
ZIP_NAME ?= group9_submission.zip

SUBMISSION_DIRS := benchmarks cmake include scripts tests
SUBMISSION_FILES := \
	.clang-format \
	.clangd \
	CMakeLists.txt \
	Makefile \
	README.md

.PHONY: all configure clean test bench_hashmap bench_map_baselines test_hashmap bench_hashmap_c test_hashmap_c zip

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

zip:
	@test -f out/report.pdf || { echo "Missing out/report.pdf. Build the report PDF first."; exit 1; }
	@rm -f "$(ZIP_NAME)"
	$(ZIP) -r "$(ZIP_NAME)" $(SUBMISSION_DIRS) $(wildcard $(SUBMISSION_FILES))
	$(ZIP) -j "$(ZIP_NAME)" out/report.pdf

clean:
	@if [ -d "$(BUILD_DIR)" ]; then $(CMAKE) --build $(BUILD_DIR) --target clean; fi
