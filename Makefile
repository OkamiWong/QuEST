all: build

.PHONY: config
config:
	cmake -S . -B ./build -DGPUACCELERATED=1 -DGPU_COMPUTE_CAPABILITY=80 -DUSE_CUQUANTUM=1 -DUSER_SOURCE=examples/grovers_search.c

.PHONY: build
build:
	cmake --build build
