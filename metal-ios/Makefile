HALIDE_DIR ?= $(HOME)/_hl
HALIDE_INC ?= $(HALIDE_DIR)/include
HALIDE_LIB ?= $(HALIDE_DIR)/bin

reaction_diffusion_2_generator: reaction_diffusion_2.cpp
	c++ -std=c++11 -I $(HALIDE_INC) -L $(HALIDE_LIB) -lHalide $< -o $@

Metal\ iOS\ Perf/reaction_diffusion_2_init.o: reaction_diffusion_2_generator
	DYLD_LIBRARY_PATH=$(HALIDE_LIB) HL_TARGET=arm-64-ios-metal-user_context ./reaction_diffusion_2_generator
	mv reaction_diffusion_2_*.h reaction_diffusion_2_*.o "./Metal iOS Perf/"

all: Metal\ iOS\ Perf/reaction_diffusion_2_init.o