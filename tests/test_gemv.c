/* GEMV unit test (H3_GEMV=1 opt-in kernels).
 * A. weight row 0 = bf16(k*0.001+1) (distinct per k), input = unit vector
 *    e_k: out[0] must equal w[0][k]; whatever the kernel consumed reveals
 *    which k it actually read. Run for every per-column AND 2D variant.
 * B. 2D variants: every weight row carries a distinct base, so e_k checks
 *    the ROW mapping too (simd_gid slab + tail shift). A second shape
 *    (2056 -> 300) exercises the K-tail (2056 % 128 != 0) and the
 *    row-tail (300 % blockM != 0) paths.
 * C. 2D variants: random weights vs a host fp32 reference on the
 *    production decode shapes (2D fma/reduction order differs from MPS by
 *    ulps by design, so the gate is a tolerance, not bit-exactness).
 * D. fused q/k/v GEMV (per-column) bit-exact against three per-matrix
 *    per-column GEMV dispatches (env forced to the per-column family). */
#include <math.h>
#include <stdio.h>

#include <stdlib.h>
#include <string.h>

#include "h3_gpu.h"
#include "yue.h"
#include "yue_gpu_ext.h"

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float bf16_to_f32(uint16_t bits) {
    uint32_t wide = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &wide, sizeof(value));
    return value;
}

/* deterministic bf16 fill in [-1, 1) for the cross-check weights */
static void fill_rand_bf16(uint16_t *data, size_t count, uint32_t seed) {
    uint32_t state = seed;
    for (size_t i = 0; i < count; i++) {
        state = state * 1664525u + 1013904223u;
        float unit = (float)(state >> 8) / (float)(1 << 24) * 2.0f - 1.0f;
        data[i] = f32_to_bf16(unit);
    }
}

static const char *per_column_variants[] = {
    "qw_gemv_bf16",      "qw_gemv_bf16_u2",   "qw_gemv_bf16_u4",
    "qw_gemv_bf16_u8",   "qw_gemv_bf16_c2u4", "qw_gemv_bf16_g128u4",
    "qw_gemv_bf16_g128c2u4",
};
static const char *gemv2d_variants[] = {
    "qw_gemv2d_m4",      "qw_gemv2d_m8",     "qw_gemv2d_m16",
    "qw_gemv2d_m32",     "qw_gemv2d_m64",    "qw_gemv2d_m64x8",
    "qw_gemv2d_m32u8",   "qw_gemv2d_m64u8",  "qw_gemv2d_m64x8u8",
};

static int dispatch_gemv(h3_gpu *gpu, h3_gpu_tensor *out,
                         h3_gpu_tensor *in, h3_gpu_tensor *w,
                         uint32_t in_dim, uint32_t out_dim) {
    return h3_gpu_begin(gpu) &&
           h3_gpu_linear_bf16(gpu, out, in, w, NULL, 1, in_dim, out_dim) &&
           h3_gpu_submit(gpu);
}

int main(void) {
    setenv("H3_GEMV", "1", 1);
    char error[512] = "";
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "gpu: %s\n", error);
        return 1;
    }
    const uint32_t out_dim = 512, in_dim = YUE_HIDDEN;
    size_t w_count = (size_t)out_dim * in_dim;
    uint16_t *host_w = malloc(w_count * 2);
    uint16_t *host_x = calloc(in_dim, 2);
    if (!host_w || !host_x) return 1;
    for (uint32_t k = 0; k < in_dim; k++)
        host_w[k] = f32_to_bf16(1.0f + (float)k * 0.001f); /* row 0 */
    for (size_t i = in_dim; i < w_count; i++) host_w[i] = 0;
    h3_gpu_tensor *weight = h3_gpu_tensor_from_bf16(gpu, host_w, w_count);
    h3_gpu_tensor *input = h3_gpu_tensor_new_bf16(gpu, in_dim);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, out_dim);
    if (!weight || !input || !output) return 1;
    int failures = 0;
    uint32_t keys[] = {0,    1,    7,    8,    9,    15,   16,   17,
                       503,  504,  505,  511,  512,  513,  519,  520,
                       1015, 1016, 1017, 1527, 1528, 1529, 2039, 2040,
                       2041, 2046, 2047};

    /* ---- A. e_k -> out[0] == w[0][k], every variant ---- */
    for (size_t v = 0;
         v < sizeof(per_column_variants) / sizeof(per_column_variants[0]) +
                 sizeof(gemv2d_variants) / sizeof(gemv2d_variants[0]);
         v++) {
        const char *variant =
            v < sizeof(per_column_variants) / sizeof(per_column_variants[0])
                ? per_column_variants[v]
                : gemv2d_variants[v - sizeof(per_column_variants) /
                                          sizeof(per_column_variants[0])];
        setenv("H3_GEMV_KERNEL", variant, 1);
        int variant_failures = 0;
        for (size_t t = 0; t < sizeof(keys) / sizeof(keys[0]); t++) {
            uint32_t k = keys[t];
            memset(host_x, 0, in_dim * 2);
            host_x[k] = f32_to_bf16(1.0f);
            if (!h3_gpu_tensor_write_bf16(input, host_x, in_dim)) return 1;
            if (!dispatch_gemv(gpu, output, input, weight, in_dim,
                               out_dim)) {
                fprintf(stderr, "dispatch (%s): %s\n", variant,
                        h3_gpu_error(gpu));
                return 1;
            }
            uint16_t got[512];
            if (!h3_gpu_tensor_read_bf16(output, got, out_dim)) return 1;
            float expect = 1.0f + (float)k * 0.001f;
            if (got[0] != f32_to_bf16(expect)) {
                printf("A %s e_%-4zu -> out[0]=%.6g (expect %.6g) FAIL\n",
                       variant, (size_t)k, bf16_to_f32(got[0]), expect);
                variant_failures++;
            }
        }
        if (variant_failures) failures += variant_failures;
    }
    printf("A. k-mapping: %zu variants x %zu keys done, %d failure(s) so far\n",
           sizeof(per_column_variants) / sizeof(per_column_variants[0]) +
               sizeof(gemv2d_variants) / sizeof(gemv2d_variants[0]),
           sizeof(keys) / sizeof(keys[0]), failures);

    /* ---- B. 2D row mapping: row r base 1 + 0.25*(r%8), e_k -> out[r]
     * must be exactly bf16(base + k*0.001) (single-term fp32 fma, so the
     * output is the weight element's own bits). Shape 2 (2056 -> 300)
     * exercises K-tail (2056 % 128 != 0) and row-tail (300 % 32 != 0). */
    {
        const uint32_t b_in = 2056, b_out = 300;
        uint16_t *bw = malloc((size_t)b_in * b_out * 2);
        uint16_t *bx = calloc(b_in, 2);
        if (!bw || !bx) return 1;
        for (uint32_t r = 0; r < b_out; r++)
            for (uint32_t k = 0; k < b_in; k++)
                bw[(size_t)r * b_in + k] =
                    f32_to_bf16(1.0f + 0.25f * (float)(r % 8) +
                                (float)k * 0.001f);
        h3_gpu_tensor *bw_t = h3_gpu_tensor_from_bf16(gpu, bw,
                                                      (size_t)b_in * b_out);
        h3_gpu_tensor *bx_t = h3_gpu_tensor_new_bf16(gpu, b_in);
        h3_gpu_tensor *bo_t = h3_gpu_tensor_new_bf16(gpu, b_out);
        if (!bw_t || !bx_t || !bo_t) return 1;
        uint32_t rows[] = {0, 1, 3, 4, 7, 8, 15, 16, 31, 32, 296, 298, 299};
        uint32_t kks[] = {0, 1, 33, 127, 128, 129, 1023, 1024,
                          2047, 2048, 2055};
        for (size_t v = 0;
             v < sizeof(gemv2d_variants) / sizeof(gemv2d_variants[0]); v++) {
            setenv("H3_GEMV_KERNEL", gemv2d_variants[v], 1);
            int bad = 0;
            for (size_t ri = 0; ri < sizeof(rows) / sizeof(rows[0]); ri++) {
                for (size_t ki = 0; ki < sizeof(kks) / sizeof(kks[0]); ki++) {
                    uint32_t r = rows[ri], k = kks[ki];
                    if (r >= b_out || k >= b_in) continue;
                    memset(bx, 0, b_in * 2);
                    bx[k] = f32_to_bf16(1.0f);
                    if (!h3_gpu_tensor_write_bf16(bx_t, bx, b_in)) return 1;
                    if (!dispatch_gemv(gpu, bo_t, bx_t, bw_t, b_in,
                                       b_out)) {
                        fprintf(stderr, "dispatch (%s): %s\n",
                                gemv2d_variants[v], h3_gpu_error(gpu));
                        return 1;
                    }
                    uint16_t got[300];
                    if (!h3_gpu_tensor_read_bf16(bo_t, got, b_out)) return 1;
                    float expect = 1.0f + 0.25f * (float)(r % 8) +
                                   (float)k * 0.001f;
                    if (got[r] != f32_to_bf16(expect)) {
                        printf("B %s out[%u] (e_%u) = %.6g (expect %.6g)"
                               " FAIL\n", gemv2d_variants[v], r, k,
                               bf16_to_f32(got[r]), expect);
                        bad++;
                    }
                }
            }
            if (bad) failures += bad;
        }
        h3_gpu_tensor_free(bw_t);
        h3_gpu_tensor_free(bx_t);
        h3_gpu_tensor_free(bo_t);
        free(bw);
        free(bx);
        printf("B. 2D row/tail mapping: %zu variants x %zu checks done,"
               " %d failure(s) so far\n",
               sizeof(gemv2d_variants) / sizeof(gemv2d_variants[0]),
               sizeof(rows) / sizeof(rows[0]) *
                   sizeof(kks) / sizeof(kks[0]),
               failures);
    }

    /* ---- C. 2D random cross-check vs host fp32 on production shapes ---- */
    {
        struct {
            uint32_t in_dim, out_dim;
            const char *label;
        } shapes[] = {
            {2048, 1024, "k/v"},   {2048, 2048, "q/o"},
            {2048, 12288, "fc1"},  {6144, 2048, "down"},
            {2048, 32769, "lm"},
        };
        uint16_t *hx = NULL, *hw = NULL, *got = NULL;
        double max_scaled = 0.0;
        int bad = 0;
        for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
            uint32_t id = shapes[s].in_dim, od = shapes[s].out_dim;
            size_t wc = (size_t)id * od;
            hw = malloc(wc * 2);
            hx = malloc(id * 2);
            got = malloc((size_t)od * 2);
            float *ref = malloc(sizeof(float) * od);
            float *scale = malloc(sizeof(float) * od);
            if (!hw || !hx || !got || !ref || !scale) return 1;
            fill_rand_bf16(hx, id, 0x51a4u + (uint32_t)s);
            h3_gpu_tensor *w_t = h3_gpu_tensor_from_bf16(gpu, NULL, wc);
            h3_gpu_tensor *x_t = h3_gpu_tensor_new_bf16(gpu, id);
            h3_gpu_tensor *o_t = h3_gpu_tensor_new_bf16(gpu, od);
            if (!w_t || !x_t || !o_t) return 1;
            if (!h3_gpu_tensor_write_bf16(x_t, hx, id)) return 1;
            /* one weight fill, one upload; host reference + per-row
             * |w.x| scale read straight from the host buffer */
            fill_rand_bf16(hw, wc, 0x51b0u + (uint32_t)s);
            if (!h3_gpu_tensor_write_bf16(w_t, hw, wc)) return 1;
            for (uint32_t r = 0; r < od; r++) {
                const uint16_t *row = hw + (size_t)r * id;
                double sum = 0.0, abs_sum = 0.0;
                for (uint32_t k = 0; k < id; k++) {
                    float t = bf16_to_f32(row[k]) * bf16_to_f32(hx[k]);
                    sum += (double)t;
                    abs_sum += fabs((double)t);
                }
                ref[r] = (float)sum;
                scale[r] = (float)abs_sum;
            }
            for (size_t v = 0;
                 v < sizeof(gemv2d_variants) / sizeof(gemv2d_variants[0]);
                 v++) {
                setenv("H3_GEMV_KERNEL", gemv2d_variants[v], 1);
                if (!dispatch_gemv(gpu, o_t, x_t, w_t, id, od)) {
                    fprintf(stderr, "dispatch (%s): %s\n",
                            gemv2d_variants[v], h3_gpu_error(gpu));
                    return 1;
                }
                if (!h3_gpu_tensor_read_bf16(o_t, got, od)) return 1;
                for (uint32_t r = 0; r < od; r++) {
                    double err =
                        fabs((double)bf16_to_f32(got[r]) - (double)ref[r]);
                    /* fp32 order noise + bf16 output rounding, relative
                     * to the row's |sum w.x|; a mapping bug overshoots by
                     * orders of magnitude. */
                    double tol = 2e-3 * (double)scale[r] + 1e-3;
                    double scaled = err / tol;
                    if (scaled > max_scaled) max_scaled = scaled;
                    if (scaled > 1.0) {
                        if (bad < 8)
                            printf("C %s %s row %u err %.4g tol %.4g FAIL\n",
                                   gemv2d_variants[v], shapes[s].label, r,
                                   err, tol);
                        bad++;
                    }
                }
            }
            h3_gpu_tensor_free(w_t);
            h3_gpu_tensor_free(x_t);
            h3_gpu_tensor_free(o_t);
            free(hw);
            free(hx);
            free(got);
            free(ref);
            free(scale);
        }
        printf("C. 2D random vs host fp32: max err/tol %.3f, %d violation(s),"
               " %d failure(s) so far\n",
               max_scaled, bad, failures + bad);
        failures += bad;
    }

    /* ---- D. fused q/k/v GEMV (qw_gemv_qkv_bf16_*) ---- */
    setenv("H3_GEMV_KERNEL", "qw_gemv_bf16_u8", 1); /* per-column family */
    const uint32_t qkv_in = YUE_HIDDEN, qkv_out_q = 2048, qkv_out_kv = 1024;
    size_t wq_count = (size_t)qkv_out_q * qkv_in;
    size_t wkv_count = (size_t)qkv_out_kv * qkv_in;
    uint16_t *hwq = malloc(wq_count * 2), *hwk = malloc(wkv_count * 2),
             *hwv = malloc(wkv_count * 2), *hx2 = calloc(qkv_in, 2);
    if (!hwq || !hwk || !hwv || !hx2) return 1;
    /* identity weights for the k-mapping phase: row 0 of each matrix
     * carries distinct per-k values, all other rows zero; e_k must
     * surface exactly one of them per output. */
    for (uint32_t k = 0; k < qkv_in; k++) {
        hwq[k] = f32_to_bf16(1.0f + (float)k * 0.001f);
        hwk[k] = f32_to_bf16(2.0f + (float)k * 0.001f);
        hwv[k] = f32_to_bf16(3.0f + (float)k * 0.001f);
    }
    for (size_t i = qkv_in; i < wq_count; i++) hwq[i] = 0;
    for (size_t i = qkv_in; i < wkv_count; i++) {
        hwk[i] = 0;
        hwv[i] = 0;
    }
    h3_gpu_tensor *wqt = h3_gpu_tensor_from_bf16(gpu, hwq, wq_count);
    h3_gpu_tensor *wkt = h3_gpu_tensor_from_bf16(gpu, hwk, wkv_count);
    h3_gpu_tensor *wvt = h3_gpu_tensor_from_bf16(gpu, hwv, wkv_count);
    h3_gpu_tensor *x2 = h3_gpu_tensor_new_bf16(gpu, qkv_in);
    h3_gpu_tensor *oq = h3_gpu_tensor_new_bf16(gpu, qkv_out_q);
    h3_gpu_tensor *ok_ = h3_gpu_tensor_new_bf16(gpu, qkv_out_kv);
    h3_gpu_tensor *ov = h3_gpu_tensor_new_bf16(gpu, qkv_out_kv);
    h3_gpu_tensor *oq_ref = h3_gpu_tensor_new_bf16(gpu, qkv_out_q);
    h3_gpu_tensor *ok_ref = h3_gpu_tensor_new_bf16(gpu, qkv_out_kv);
    h3_gpu_tensor *ov_ref = h3_gpu_tensor_new_bf16(gpu, qkv_out_kv);
    if (!wqt || !wkt || !wvt || !x2 || !oq || !ok_ || !ov || !oq_ref ||
        !ok_ref || !ov_ref) return 1;
    uint16_t *qbits = malloc(qkv_out_q * 2), *kbits = malloc(qkv_out_kv * 2),
             *vbits = malloc(qkv_out_kv * 2), *rq = malloc(qkv_out_q * 2),
             *rk = malloc(qkv_out_kv * 2), *rv = malloc(qkv_out_kv * 2);
    if (!qbits || !kbits || !vbits || !rq || !rk || !rv) return 1;

    int qkv_failures = 0;
    uint32_t qkv_keys[] = {0,    1,    7,    8,    9,    15,   16,   17,
                           503,  504,  505,  511,  512,  513,  519,  520,
                           1015, 1016, 1017, 1527, 1528, 1529, 2039, 2040,
                           2041, 2046, 2047};
    for (size_t t = 0; t < sizeof(qkv_keys) / sizeof(qkv_keys[0]); t++) {
        uint32_t k = qkv_keys[t];
        memset(hx2, 0, qkv_in * 2);
        hx2[k] = f32_to_bf16(1.0f);
        if (!h3_gpu_tensor_write_bf16(x2, hx2, qkv_in)) return 1;
        if (!(h3_gpu_begin(gpu) &&
              qw_gpu_qkv_gemv_bf16(gpu, oq, ok_, ov, x2, wqt, wkt, wvt,
                                   qkv_in, qkv_out_q, qkv_out_kv) &&
              h3_gpu_submit(gpu))) {
            fprintf(stderr, "qkv fused dispatch: %s\n", h3_gpu_error(gpu));
            return 1;
        }
        if (!h3_gpu_tensor_read_bf16(oq, qbits, qkv_out_q) ||
            !h3_gpu_tensor_read_bf16(ok_, kbits, qkv_out_kv) ||
            !h3_gpu_tensor_read_bf16(ov, vbits, qkv_out_kv)) return 1;
        int good = qbits[0] == f32_to_bf16(1.0f + (float)k * 0.001f) &&
                   kbits[0] == f32_to_bf16(2.0f + (float)k * 0.001f) &&
                   vbits[0] == f32_to_bf16(3.0f + (float)k * 0.001f);
        if (!good) {
            printf("qkv e_%-4zu -> q=%.6g k=%.6g v=%.6g FAIL\n", (size_t)k,
                   bf16_to_f32(qbits[0]), bf16_to_f32(kbits[0]),
                   bf16_to_f32(vbits[0]));
            qkv_failures++;
        }
    }
    /* bit-exact cross-check against three per-matrix GEMV dispatches on
     * random weights (production shapes). Both sides forced onto the
     * per-column family so the bit-exact property is variant-independent. */
    fill_rand_bf16(hwq, wq_count, 0x51a1u);
    fill_rand_bf16(hwk, wkv_count, 0x51a2u);
    fill_rand_bf16(hwv, wkv_count, 0x51a3u);
    if (!h3_gpu_tensor_write_bf16(wqt, hwq, wq_count) ||
        !h3_gpu_tensor_write_bf16(wkt, hwk, wkv_count) ||
        !h3_gpu_tensor_write_bf16(wvt, hwv, wkv_count)) return 1;
    /* random x */
    fill_rand_bf16(hx2, qkv_in, 0x51a4u);
    if (!h3_gpu_tensor_write_bf16(x2, hx2, qkv_in)) return 1;
    if (!(h3_gpu_begin(gpu) &&
          qw_gpu_qkv_gemv_bf16(gpu, oq, ok_, ov, x2, wqt, wkt, wvt, qkv_in,
                               qkv_out_q, qkv_out_kv) &&
          h3_gpu_submit(gpu))) {
        fprintf(stderr, "qkv cross dispatch: %s\n", h3_gpu_error(gpu));
        return 1;
    }
    if (!(h3_gpu_begin(gpu) &&
          h3_gpu_linear_bf16(gpu, oq_ref, x2, wqt, NULL, 1, qkv_in,
                             qkv_out_q) &&
          h3_gpu_linear_bf16(gpu, ok_ref, x2, wkt, NULL, 1, qkv_in,
                             qkv_out_kv) &&
          h3_gpu_linear_bf16(gpu, ov_ref, x2, wvt, NULL, 1, qkv_in,
                             qkv_out_kv) &&
          h3_gpu_submit(gpu))) {
        fprintf(stderr, "qkv reference dispatch: %s\n", h3_gpu_error(gpu));
        return 1;
    }
    if (!h3_gpu_tensor_read_bf16(oq, qbits, qkv_out_q) ||
        !h3_gpu_tensor_read_bf16(ok_, kbits, qkv_out_kv) ||
        !h3_gpu_tensor_read_bf16(ov, vbits, qkv_out_kv) ||
        !h3_gpu_tensor_read_bf16(oq_ref, rq, qkv_out_q) ||
        !h3_gpu_tensor_read_bf16(ok_ref, rk, qkv_out_kv) ||
        !h3_gpu_tensor_read_bf16(ov_ref, rv, qkv_out_kv)) return 1;
    int bit_exact = memcmp(qbits, rq, qkv_out_q * 2) == 0 &&
                    memcmp(kbits, rk, qkv_out_kv * 2) == 0 &&
                    memcmp(vbits, rv, qkv_out_kv * 2) == 0;
    printf("qkv fused vs per-matrix gemv bit-exact: %s\n",
           bit_exact ? "yes" : "NO");
    if (!bit_exact) qkv_failures++;

    /* ---- E. fused 2D qkv: bit-exact vs three separate 2D GEMVs (same
     * row arithmetic by construction; guards the segment mapping). ---- */
    {
        const char *pairs[][2] = {
            {"qw_gemv2d_qkv_m32u8", "qw_gemv2d_m32u8"},
            {"qw_gemv2d_qkv_m64u8", "qw_gemv2d_m64u8"},
        };
        int bad2d = 0;
        for (size_t p = 0;
             p < sizeof(pairs) / sizeof(pairs[0]); p++) {
            setenv("H3_QKV_GEMV_KERNEL", pairs[p][0], 1);
            setenv("H3_GEMV_KERNEL", pairs[p][1], 1);
            if (!(h3_gpu_begin(gpu) &&
                  qw_gpu_qkv_gemv_bf16(gpu, oq, ok_, ov, x2, wqt, wkt, wvt,
                                       qkv_in, qkv_out_q, qkv_out_kv) &&
                  h3_gpu_submit(gpu))) {
                fprintf(stderr, "fused 2D qkv dispatch: %s\n",
                        h3_gpu_error(gpu));
                return 1;
            }
            if (!h3_gpu_tensor_read_bf16(oq, qbits, qkv_out_q) ||
                !h3_gpu_tensor_read_bf16(ok_, kbits, qkv_out_kv) ||
                !h3_gpu_tensor_read_bf16(ov, vbits, qkv_out_kv)) return 1;
            if (!(h3_gpu_begin(gpu) &&
                  h3_gpu_linear_bf16(gpu, oq_ref, x2, wqt, NULL, 1, qkv_in,
                                     qkv_out_q) &&
                  h3_gpu_linear_bf16(gpu, ok_ref, x2, wkt, NULL, 1, qkv_in,
                                     qkv_out_kv) &&
                  h3_gpu_linear_bf16(gpu, ov_ref, x2, wvt, NULL, 1, qkv_in,
                                     qkv_out_kv) &&
                  h3_gpu_submit(gpu))) {
                fprintf(stderr, "2D reference dispatch: %s\n",
                        h3_gpu_error(gpu));
                return 1;
            }
            if (!h3_gpu_tensor_read_bf16(oq_ref, rq, qkv_out_q) ||
                !h3_gpu_tensor_read_bf16(ok_ref, rk, qkv_out_kv) ||
                !h3_gpu_tensor_read_bf16(ov_ref, rv, qkv_out_kv)) return 1;
            int same = memcmp(qbits, rq, qkv_out_q * 2) == 0 &&
                       memcmp(kbits, rk, qkv_out_kv * 2) == 0 &&
                       memcmp(vbits, rv, qkv_out_kv * 2) == 0;
            printf("E. fused %s vs 3x %s bit-exact: %s\n", pairs[p][0],
                   pairs[p][1], same ? "yes" : "NO");
            if (!same) bad2d++;
        }
        unsetenv("H3_QKV_GEMV_KERNEL");
        qkv_failures += bad2d;
    }

    printf(qkv_failures ? "qkv gemv tests: %d failure(s)\n"
                        : "qkv gemv tests: 0 failure(s)\n",
           qkv_failures);
    failures += qkv_failures;
    free(hwq); free(hwk); free(hwv); free(hx2);
    free(qbits); free(kbits); free(vbits); free(rq); free(rk); free(rv);
    return failures ? 1 : 0;
}
