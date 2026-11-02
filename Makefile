CC := clang
AR := ar
CFLAGS := -std=c11 -O3 -I. -MMD -MP -Wall -Wextra -Wpedantic -Wshadow -DACCELERATE_NEW_LAPACK \
	-Wconversion -Wno-sign-conversion -D_DARWIN_C_SOURCE
OBJCFLAGS := $(CFLAGS) -fobjc-arc
FRAMEWORKS := -framework Foundation -framework Metal \
	-framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
	-framework Accelerate
LDLIBS := $(FRAMEWORKS) -licucore -lm -lz

# Vendored from h3-metal (MIT, Salvatore Sanfilippo) via qwen-metal: the Metal
# runtime, safetensors/weight plumbing, media pipeline, tokenizer, and the
# linenoise REPL editor. Kept under their original h3_ names.
VENDORED_C := h3_safetensors.c h3_weights.c h3_ffmpeg.c linenoise.c
VENDORED_M := h3_gpu.m h3_metal.m h3_tokenizer.m
YUE_C := yue_json.c yue_protocol.c yue_sampling.c yue_generate.c yue_vae.c
YUE_M := yue_ar.m yue_nar.m
LIB_OBJ := $(VENDORED_C:.c=.o) $(VENDORED_M:.m=.o) $(YUE_C:.c=.o) $(YUE_M:.m=.o)
CLI_OBJ := main.o

.PHONY: all m0-tests clean

all: yue

yue: $(CLI_OBJ) $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

tests/test_m0: tests/test_m0.o $(filter-out main.o,$(LIB_OBJ))
	$(CC) -o $@ $^ $(LDLIBS)

tests/test_tokenizer: tests/test_tokenizer.o $(filter-out main.o,$(LIB_OBJ))
	$(CC) -o $@ $^ $(LDLIBS)

m0-tests: tests/test_m0
	./tests/test_m0

tokenizer-fixture:
	/Volumes/Work/mlx-Yue/.venv/bin/python tests/gen_tokenizer_fixture.py

tests/test_ar: tests/test_ar.o $(filter-out main.o,$(LIB_OBJ))
	$(CC) -o $@ $^ $(LDLIBS)

tests/test_sampling: tests/test_sampling.o $(filter-out main.o,$(LIB_OBJ))
	$(CC) -o $@ $^ $(LDLIBS)

tests/test_nar: tests/test_nar.o $(filter-out main.o,$(LIB_OBJ))
	$(CC) -o $@ $^ $(LDLIBS)

nar-fixture:
	LYRA_FORCE_BF16_SDPA=1 /Volumes/Work/mlx-Yue/.venv/bin/python tests/gen_nar_fixture.py

nar-tests: tests/test_nar
	./tests/test_nar

tests/test_vae: tests/test_vae.o $(filter-out main.o,$(LIB_OBJ))
	$(CC) -o $@ $^ $(LDLIBS)

tests/bench_gemv: tests/bench_gemv.o $(filter-out main.o,$(LIB_OBJ))
	$(CC) -o $@ $^ $(LDLIBS)

bench-gemv: tests/bench_gemv
	./tests/bench_gemv

tests/test_gemv: tests/test_gemv.o $(filter-out main.o,$(LIB_OBJ))
	$(CC) -o $@ $^ $(LDLIBS)

gemv-tests: tests/test_gemv
	./tests/test_gemv

vae-fixture:
	/Volumes/Work/mlx-Yue/.venv/bin/python tests/gen_vae_fixture.py

vae-tests: tests/test_vae
	./tests/test_vae

tokenizer-tests: tests/test_tokenizer
	./tests/test_tokenizer

ar-fixture:
	/Volumes/Work/mlx-Yue/.venv/bin/python tests/gen_ar_fixture.py

ar-tests: tests/test_ar
	./tests/test_ar

sampling-fixture:
	/Volumes/Work/mlx-Yue/.venv/bin/python tests/gen_sampling_fixture.py

sampling-tests: tests/test_sampling
	./tests/test_sampling

clean:
	rm -f $(LIB_OBJ) $(CLI_OBJ) tests/*.o tests/test_m0 yue *.d tests/*.d

%.o: %.c
	$(CC) $(CFLAGS) -I. -c $< -o $@

%.o: %.m
	$(CC) $(OBJCFLAGS) -I. -c $< -o $@

tests/%.o: tests/%.c
	$(CC) $(CFLAGS) -I. -c $< -o $@

-include $(wildcard *.d tests/*.d)

tests/bench_dispatch: tests/bench_dispatch.o $(filter-out main.o,$(LIB_OBJ))
	$(CC) -o $@ $^ $(LDLIBS)

bench-dispatch: tests/bench_dispatch
	./tests/bench_dispatch
