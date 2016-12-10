test: test.cu
	nvcc -g -std=c++11 -arch sm_50 -o $@ $<

run: test
	./test

dbg: test
	gdb ./test

.PHONY: run dbg
