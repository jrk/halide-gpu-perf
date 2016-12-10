HALIDE_ROOT ?= $(HOME)/hl-lin
CFLAGS += -I$(HALIDE_ROOT)/include
# LDFLAGS += -L$(HALIDE_ROOT)/lib -lHalideRuntime-host-cuda-debug

test: test.cu
	nvcc -DNDEBUG $(CFLAGS) $(LDFLAGS) -O3 -arch sm_50 -std=c++11 -o $@ $<

run: test
	./test

dbg: test
	gdb ./test

.PHONY: run dbg
