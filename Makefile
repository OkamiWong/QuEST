all: build

.PHONY: config
config:
	cmake -S . -B ./build -DMULTITHREADED=0 \
	-DGPUACCELERATED=1 -DGPU_COMPUTE_CAPABILITY=80 \
	-DUSE_CUQUANTUM=0 -DUSE_MEMOPT=1 \
	-DUSER_SOURCE=examples/qft.c \
	-DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake \
	-DCMAKE_PREFIX_PATH=${ORTOOLS_ROOT} \

.PHONY: config-debug
config-debug:
	cmake -S . -B ./build -DMULTITHREADED=0 \
	-DGPUACCELERATED=1 -DGPU_COMPUTE_CAPABILITY=80 \
	-DUSE_CUQUANTUM=0 -DUSE_MEMOPT=1 \
	-DUSER_SOURCE=examples/qft.c \
	-DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake \
	-DCMAKE_PREFIX_PATH=${ORTOOLS_ROOT} \
	-DCMAKE_BUILD_TYPE=Debug

.PHONY: build
build:
	cmake --build ./build -j

.PHONY: build-sequential
build-sequential:
	cmake --build ./build

.PHONY: update-memopt
update-memopt:
	git submodule update --remote dependencies/optimize-cuda-memory-usage-v1/
