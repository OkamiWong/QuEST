all: build

.PHONY: config
config:
	cmake -S . -B ./build -DGPUACCELERATED=1 -DGPU_COMPUTE_CAPABILITY=80 -DUSE_CUQUANTUM=1 \
	-DUSER_SOURCE=examples/grovers_search.c \
	-DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake \
	-DCMAKE_PREFIX_PATH=${ORTOOLS_ROOT}

.PHONY: build
build:
	cmake --build ./build -j

.PHONY: build-sequential
build-sequential:
	cmake --build ./build
