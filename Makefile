HALIDE_ROOT ?= $(HOME)/_hl
CFLAGS += -DNDEBUG -I$(HALIDE_ROOT)/include -std=c++11 #-restrict
# LDFLAGS += -L$(HALIDE_ROOT)/lib -lHalideRuntime-host-cuda-debug
CLANG = $(HOME)/.linuxbrew/opt/llvm/bin/clang++
GPU_ARCH=sm_50
CLANG_CUDA_LDFLAGS=-L/usr/local/cuda-8.0/lib64 -lcudart_static -ldl -lrt -pthread

test: test.cu
	nvcc $(CFLAGS) $(LDFLAGS) -O3 -arch $(GPU_ARCH) -o $@ $<

test-llvm: test.cu
	$(CLANG) -o $@ $< $(CFLAGS) $(LDFLAGS) -O3 --cuda-gpu-arch=sm_50 $(CLANG_CUDA_LDFLAGS)

%.ptx: %
	cuobjdump -ptx $< > $@.tmp
	awk -vRS="\n\t" -vORS="" '1' $@.tmp > $@
	rm $@.tmp

run: test
	./test 20

dbg: test
	gdb ./test

kernel-ptx: test.ptx
	csplit --suffix-format='%02d.ptx' --prefix='test-' $< '/\.visible \.entry.*/' {*}

llvm-kernel-ptx: test-llvm.ptx
	csplit --suffix-format='%02d.ptx' --prefix='test-llvm-' $< '/\.visible \.entry.*/' {*}

.PHONY: run dbg

