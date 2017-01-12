HALIDE_DIR ?= $(HOME)/_hl
HALIDE_INC ?= $(HALIDE_DIR)/include
HALIDE_LIB ?= $(HALIDE_DIR)/bin
HALIDE_GENGEN ?= $(HALIDE_DIR)/tools/GenGen.cpp
GENERATED=Metal\ iOS\ Perf/reaction_diffusion_init_metal.o Metal\ iOS\ Perf/reaction_diffusion_init_arm.o Metal\ iOS\ Perf/reaction_diffusion_update_metal.o Metal\ iOS\ Perf/reaction_diffusion_update_arm.o Metal\ iOS\ Perf/reaction_diffusion_render_metal.o Metal\ iOS\ Perf/reaction_diffusion_render_arm.o

reaction_diffusion_2_generator: reaction_diffusion_2.cpp $(HALIDE_GENGEN)
	c++ -fno-rtti -std=c++11 -I $(HALIDE_INC) -L $(HALIDE_LIB) -lHalide $^ -o $@

Metal\ iOS\ Perf/reaction_diffusion_%_metal.o: reaction_diffusion_2_generator
	DYLD_LIBRARY_PATH=$(HALIDE_LIB) ./reaction_diffusion_2_generator -o "Metal iOS Perf" -f reaction_diffusion_$(*)_metal -n reaction_diffusion_$(*)_metal -e o,h -g reaction_diffusion_$(*) target=arm-64-ios-metal-user_context

Metal\ iOS\ Perf/reaction_diffusion_%_arm.o: reaction_diffusion_2_generator
	DYLD_LIBRARY_PATH=$(HALIDE_LIB) ./reaction_diffusion_2_generator -o "Metal iOS Perf" -f reaction_diffusion_$(*)_arm -n reaction_diffusion_$(*)_arm -e o,h -g reaction_diffusion_$(*) target=arm-64-ios-user_context

all: $(GENERATED)

clean:
	rm -f $(GENERATED)
