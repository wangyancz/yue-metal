/* GEMV micro-benchmark: rows==1 bf16 linear through the dispatch paths on
 * the YuE decode shapes. The custom qw_gemv_bf16 variants (H3_GEMV_KERNEL)
 * race the MPSGraph path, and every variant is cross-checked bit-exact
 * against MPS — the custom kernel must reproduce the MPS bytes exactly or
 * the regression suites lose meaning.
 * Run from the repository root: make bench-gemv */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "h3_gpu.h"
#include "yue_gpu_ext.h"

typedef struct {
    uint32_t in_dim, out_dim;
    const char *label;
} shape;

static const shape shapes[] = {
    {2048, 1024, "k/v proj"},
    {2048, 2048, "q/o proj"},
    {6144, 2048, "down proj"},
    {2048, 12288, "fc1"},
    {2048, 32769, "semantic lm_head"},
    {2048, 151644, "abc lm_head"},
    {256, 2048, "nar time fc1"},
};

static const char *variants[] = {
    "qw_gemv_bf16",     "qw_gemv_bf16_u2",       "qw_gemv_bf16_u4",
    "qw_gemv_bf16_u8",  "qw_gemv_bf16_c2u4",     "qw_gemv_bf16_g128u4",
    "qw_gemv_bf16_g128c2u4",
};

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

/* deterministic bf16 fill in [-1, 1) */
static void fill_bf16(uint16_t *data, size_t count, uint32_t seed) {
    uint32_t state = seed;
    for (size_t i = 0; i < count; i++) {
        state = state * 1664525u + 1013904223u;
        float unit = (float)(state >> 8) / (float)(1 << 24) * 2.0f - 1.0f;
        data[i] = f32_to_bf16(unit);
    }
}

static int run_once(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                    const h3_gpu_tensor *bias, uint32_t in_dim,
                    uint32_t out_dim) {
    return h3_gpu_begin(gpu) &&
           h3_gpu_linear_bf16(gpu, output, input, weight, bias, 1, in_dim,
                              out_dim) &&
           h3_gpu_submit(gpu);
}

/* Batched timing, production-style: 28 back-to-back dispatches inside ONE
 * command buffer (like a decode layer chain), one submit per iteration.
 * Reports wall and stats.gpu_seconds per call. */
static int time_batch(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t in_dim,
                      uint32_t out_dim, double *wall_ms, double *gpu_ms) {
    for (int i = 0; i < 3; i++)
        if (!run_once(gpu, output, input, weight, bias, in_dim, out_dim))
            return 0;
    h3_gpu_stats start_stats, end_stats;
    if (!h3_gpu_get_stats(gpu, &start_stats)) return 0;
    double start = (double)clock() / CLOCKS_PER_SEC;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    start = (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
    const int iters = 10, batch = 28;
    for (int i = 0; i < iters; i++) {
        if (!h3_gpu_begin(gpu)) return 0;
        for (int b = 0; b < batch; b++)
            if (!h3_gpu_linear_bf16(gpu, output, input, weight, bias, 1,
                                    in_dim, out_dim))
                return 0;
        if (!h3_gpu_submit(gpu)) return 0;
    }
    clock_gettime(CLOCK_MONOTONIC, &ts);
    double wall = (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9 - start;
    if (!h3_gpu_get_stats(gpu, &end_stats)) return 0;
    *wall_ms = wall / (iters * batch) * 1e3;
    *gpu_ms = (double)(end_stats.gpu_seconds - start_stats.gpu_seconds) /
              (iters * batch) * 1e3;
    return 1;
}


int main(void) {
    char error[512] = "";
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "gpu init failed: %s\n", error);
        return 1;
    }
    int failures = 0;
    size_t variant_count = sizeof(variants) / sizeof(variants[0]);
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        uint32_t in_dim = shapes[s].in_dim, out_dim = shapes[s].out_dim;
        size_t in_count = in_dim, w_count = (size_t)out_dim * in_dim;
        uint16_t *host_input = malloc(in_count * 2);
        uint16_t *host_weight = malloc(w_count * 2);
        uint16_t *host_bias = malloc((size_t)out_dim * 2);
        if (!host_input || !host_weight || !host_bias) return 1;
        fill_bf16(host_input, in_count, 0x1234u + (uint32_t)s);
        fill_bf16(host_weight, w_count, 0xbeefu + (uint32_t)s);
        fill_bf16(host_bias, out_dim, 0x777u + (uint32_t)s);
        h3_gpu_tensor *input =
            h3_gpu_tensor_from_bf16(gpu, host_input, in_count);
        h3_gpu_tensor *weight =
            h3_gpu_tensor_from_bf16(gpu, host_weight, w_count);
        h3_gpu_tensor *bias = h3_gpu_tensor_from_bf16(gpu, host_bias, out_dim);
        h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, out_dim);
        uint16_t *ref = malloc((size_t)out_dim * 2);
        uint16_t *bits = malloc((size_t)out_dim * 2);
        if (!input || !weight || !bias || !output || !ref || !bits) return 1;

        unsetenv("H3_GEMV");
        setenv("H3_DISABLE_MPS_LINEAR", "0", 1);
        double mps_wall = 0, mps_gpu = 0;
        int ok = time_batch(gpu, output, input, weight, bias, in_dim, out_dim,
                            &mps_wall, &mps_gpu) &&
                 h3_gpu_tensor_read_bf16(output, ref, out_dim);
        double weight_gb = (double)w_count * 2 / 1e9;
        printf("%-18s mps wall %6.3f gpu %6.3f ms (%5.1f GB/s gpu)\n",
               shapes[s].label, mps_wall, mps_gpu,
               weight_gb / (mps_gpu * 1e-3));
        for (size_t v = 0; ok && v < variant_count; v++) {
            setenv("H3_GEMV", "1", 1);
            setenv("H3_GEMV_KERNEL", variants[v], 1);
            double wall = 0, gpu_ms = 0;
            if (!time_batch(gpu, output, input, weight, bias, in_dim, out_dim,
                            &wall, &gpu_ms) ||
                !h3_gpu_tensor_read_bf16(output, bits, out_dim)) {
                fprintf(stderr, "%s: %s dispatch failed: %s\n",
                        shapes[s].label, variants[v], h3_gpu_error(gpu));
                failures++;
                continue;
            }
            int same = memcmp(bits, ref, (size_t)out_dim * 2) == 0;
            printf("  %-22s wall %6.3f gpu %6.3f ms (%5.1f GB/s)  bit-eq "
                   "mps=%s\n",
                   variants[v], wall, gpu_ms, weight_gb / (gpu_ms * 1e-3),
                   same ? "yes" : "NO");
            if (!same) failures++;
        }
        printf("\n");
        h3_gpu_tensor_free(input);
        h3_gpu_tensor_free(weight);
        h3_gpu_tensor_free(bias);
        h3_gpu_tensor_free(output);
        free(host_input);
        free(host_weight);
        free(host_bias);
        free(ref);
        free(bits);
    }
    printf(failures ? "bench-gemv: %d failure(s)\n"
                    : "bench-gemv: all outputs bit-exact, %d failure(s)\n",
           failures);
    h3_gpu_free(gpu);
    return failures ? 1 : 0;
}
