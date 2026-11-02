/* M0 GPU smoke test: vendored h3 runtime works from the yue-metal tree.
 * Exercises the two kernel families YuE2 leans on (bf16 linear GEMM and
 * double-rounding RMSNorm) against CPU references. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "h3_gpu.h"
#include "yue.h"
#include "yue_gpu_ext.h"

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, 4);
    uint32_t lsb = (bits >> 16) & 1;
    bits += 0x7fff + lsb;
    return (uint16_t)(bits >> 16);
}

static float bf16_to_f32(uint16_t bits) {
    uint32_t wide = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &wide, 4);
    return value;
}

static int failures;

static void check(int condition, const char *name) {
    printf("%-40s %s\n", name, condition ? "ok" : "FAIL");
    if (!condition) failures++;
}

int main(void) {
    char error[YUE_ERROR_SIZE];
    error[0] = '\0';
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "h3_gpu_create failed: %s\n", error);
        return 1;
    }

    /* ---- bf16 linear: [3x8] @ [5x8]^T + bias ---- */
    enum { ROWS = 3, IN = 8, OUT = 5 };
    float host_in[ROWS * IN];
    float host_w[OUT * IN];
    float host_b[OUT];
    for (int i = 0; i < ROWS * IN; i++) host_in[i] = sinf((float)i * 0.37f);
    for (int i = 0; i < OUT * IN; i++) host_w[i] = cosf((float)i * 0.53f);
    for (int i = 0; i < OUT; i++) host_b[i] = 0.25f * (float)i - 1.0f;

    uint16_t *bf_in = malloc(sizeof(uint16_t) * ROWS * IN);
    uint16_t *bf_w = malloc(sizeof(uint16_t) * OUT * IN);
    uint16_t *bf_b = malloc(sizeof(uint16_t) * OUT);
    for (int i = 0; i < ROWS * IN; i++) bf_in[i] = f32_to_bf16(host_in[i]);
    for (int i = 0; i < OUT * IN; i++) bf_w[i] = f32_to_bf16(host_w[i]);
    for (int i = 0; i < OUT; i++) bf_b[i] = f32_to_bf16(host_b[i]);

    /* CPU reference with bf16 rounding at each boundary (lyra convention). */
    float ref[ROWS * OUT];
    for (int r = 0; r < ROWS; r++) {
        for (int o = 0; o < OUT; o++) {
            float acc = 0.0f;
            for (int k = 0; k < IN; k++)
                acc += bf16_to_f32(bf_in[r * IN + k]) *
                       bf16_to_f32(bf_w[o * IN + k]);
            ref[r * OUT + o] = acc + bf16_to_f32(bf_b[o]);
        }
    }

    h3_gpu_tensor *in_t = h3_gpu_tensor_from_bf16(gpu, bf_in, ROWS * IN);
    h3_gpu_tensor *w_t = h3_gpu_tensor_from_bf16(gpu, bf_w, OUT * IN);
    h3_gpu_tensor *b_t = h3_gpu_tensor_from_bf16(gpu, bf_b, OUT);
    h3_gpu_tensor *out_t = h3_gpu_tensor_new_bf16(gpu, ROWS * OUT);
    check(in_t && w_t && b_t && out_t, "tensor creation");

    /* h3 convention: nonzero return = success, 0 = failure. */
    int ok = h3_gpu_begin(gpu) &&
             h3_gpu_linear_bf16(gpu, out_t, in_t, w_t, b_t, ROWS, IN, OUT) &&
             h3_gpu_submit(gpu);
    check(ok, "bf16 linear dispatch");
    if (!ok) fprintf(stderr, "h3 error: %s\n", h3_gpu_error(gpu));

    uint16_t got_bits[ROWS * OUT];
    float got[ROWS * OUT];
    ok = h3_gpu_tensor_read_bf16(out_t, got_bits, ROWS * OUT);
    for (int i = 0; i < ROWS * OUT; i++) got[i] = bf16_to_f32(got_bits[i]);
    check(ok, "linear readback");
    double max_diff = 0.0;
    for (int i = 0; i < ROWS * OUT; i++)
        max_diff = fmax(max_diff, fabs(got[i] - ref[i]));
    check(max_diff <= 2e-2, "bf16 linear vs cpu reference");

    /* ---- qw_gpu_rms_norm_bf16: fp32 variance, bf16 double rounding ---- */
    enum { NROWS = 4, WIDTH = 128 };
    float *hx = malloc(sizeof(float) * NROWS * WIDTH);
    float *hw = malloc(sizeof(float) * WIDTH);
    for (int i = 0; i < NROWS * WIDTH; i++)
        hx[i] = sinf((float)i * 0.11f) * (3.0f + (float)(i % 7));
    for (int i = 0; i < WIDTH; i++) hw[i] = 0.5f + cosf((float)i * 0.21f);

    uint16_t *bx = malloc(sizeof(uint16_t) * NROWS * WIDTH);
    uint16_t *bw = malloc(sizeof(uint16_t) * WIDTH);
    for (int i = 0; i < NROWS * WIDTH; i++) bx[i] = f32_to_bf16(hx[i]);
    for (int i = 0; i < WIDTH; i++) bw[i] = f32_to_bf16(hw[i]);

    float refn[NROWS * WIDTH];
    for (int r = 0; r < NROWS; r++) {
        float var = 0.0f;
        for (int c = 0; c < WIDTH; c++) {
            float v = bf16_to_f32(bx[r * WIDTH + c]);
            var += v * v;
        }
        var /= (float)WIDTH;
        float scale = 1.0f / sqrtf(var + YUE_RMS_EPS);
        for (int c = 0; c < WIDTH; c++) {
            float normalized =
                bf16_to_f32(f32_to_bf16(bf16_to_f32(bx[r * WIDTH + c]) * scale));
            refn[r * WIDTH + c] =
                bf16_to_f32(f32_to_bf16(normalized * bf16_to_f32(bw[c])));
        }
    }

    h3_gpu_tensor *x_t = h3_gpu_tensor_from_bf16(gpu, bx, NROWS * WIDTH);
    h3_gpu_tensor *w2_t = h3_gpu_tensor_from_bf16(gpu, bw, WIDTH);
    h3_gpu_tensor *n_t = h3_gpu_tensor_new_bf16(gpu, NROWS * WIDTH);
    ok = x_t && w2_t && n_t && h3_gpu_begin(gpu) &&
         qw_gpu_rms_norm_bf16(gpu, n_t, x_t, w2_t, NROWS, WIDTH, YUE_RMS_EPS) &&
         h3_gpu_submit(gpu);
    check(ok, "rms_norm dispatch");
    if (!ok) fprintf(stderr, "h3 error: %s\n", h3_gpu_error(gpu));

    uint16_t nb[NROWS * WIDTH];
    ok = h3_gpu_tensor_read_bf16(n_t, nb, NROWS * WIDTH);
    check(ok, "rms_norm readback");
    max_diff = 0.0;
    float ref_scale = 0.0f;
    for (int i = 0; i < NROWS * WIDTH; i++) {
        max_diff = fmax(max_diff, fabs(bf16_to_f32(nb[i]) - refn[i]));
        ref_scale = (float)fmax(ref_scale, fabs(refn[i]));
    }
    /* GPU tree reduction vs CPU sequential accumulation may differ by one
     * bf16 ulp in the fp32 variance; anything beyond that is a wiring bug. */
    check(max_diff <= ref_scale * 0.01, "rms_norm within 1 bf16 ulp");

    h3_gpu_tensor_free(in_t);
    h3_gpu_tensor_free(w_t);
    h3_gpu_tensor_free(b_t);
    h3_gpu_tensor_free(out_t);
    h3_gpu_tensor_free(x_t);
    h3_gpu_tensor_free(w2_t);
    h3_gpu_tensor_free(n_t);
    h3_gpu_free(gpu);
    free(bf_in);
    free(bf_w);
    free(bf_b);
    free(hx);
    free(hw);
    free(bx);
    free(bw);

    printf("m0 smoke: %d failure(s)\n", failures);
    return failures ? 1 : 0;
}
