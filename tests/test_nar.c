/* M4 NAR parity against the lyra reference stack (gen_nar_fixture.py):
 * conditioning K/V, velocity evaluations on explicit noise, the midpoint
 * solver, and multi-chunk synthesis with a tiny context window. bf16
 * cross-implementation noise follows the M2 gates (relL2 bands, bit
 * mismatch counts reported). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "h3_gpu.h"
#include "yue.h"
#include "yue_ar.h"
#include "yue_json.h"
#include "yue_nar.h"
#include "yue_protocol.h"

#define MODEL_DIR "/Volumes/Work/models/YuE2.models/lyra-YuE2-3B"

static int failures;

static void check(int condition, const char *name) {
    printf("%-52s %s\n", name, condition ? "ok" : "FAIL");
    if (!condition) failures++;
}

static void *read_npy(const char *path, size_t *count, int *elem_size) {
    FILE *handle = fopen(path, "rb");
    if (!handle) return NULL;
    unsigned char magic[8];
    if (fread(magic, 1, 8, handle) != 8 || memcmp(magic, "\x93NUMPY", 6)) {
        fclose(handle);
        return NULL;
    }
    unsigned short header_len;
    if (fread(&header_len, 1, 2, handle) != 2) {
        fclose(handle);
        return NULL;
    }
    char header[4096];
    if (header_len >= sizeof(header) ||
        fread(header, 1, header_len, handle) != header_len) {
        fclose(handle);
        return NULL;
    }
    header[header_len] = '\0';
    *elem_size = strstr(header, "'<f4'") || strstr(header, "'<u4'") ||
                         strstr(header, "'<i4'")
                     ? 4
                     : 2;
    const char *shape = strstr(header, "'shape'");
    long long elems = -1;
    if (shape) {
        const char *cursor = strchr(shape, '(');
        if (cursor) {
            elems = 1;
            cursor++;
            for (;;) {
                char *end = NULL;
                long long dim = strtoll(cursor, &end, 10);
                if (end == cursor) break;
                elems *= dim;
                cursor = end;
                while (*cursor == ' ' || *cursor == ',') cursor++;
                if (*cursor == ')' || !*cursor) break;
            }
        }
    }
    if (elems <= 0) {
        fclose(handle);
        return NULL;
    }
    size_t bytes = (size_t)elems * (size_t)*elem_size;
    void *data = malloc(bytes);
    if (!data || fread(data, 1, bytes, handle) != bytes) {
        free(data);
        fclose(handle);
        return NULL;
    }
    fclose(handle);
    *count = (size_t)elems;
    return data;
}

static uint16_t *load_bits(const char *name, size_t *count) {
    char path[512];
    snprintf(path, sizeof(path), "tests/nar_fixture/%s.npy", name);
    int elem = 2;
    return read_npy(path, count, &elem);
}

static float *load_f32(const char *name, size_t *count) {
    char path[512];
    snprintf(path, sizeof(path), "tests/nar_fixture/%s.npy", name);
    int elem = 4;
    return read_npy(path, count, &elem);
}

static float bf16_to_f32(uint16_t bits) {
    uint32_t wide = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &wide, 4);
    return value;
}

static double compare_bits(const uint16_t *got, const uint16_t *ref,
                           size_t count, size_t *bit_mismatches) {
    double sum_sq = 0.0, diff_sq = 0.0;
    size_t mismatches = 0;
    for (size_t i = 0; i < count; i++) {
        float g = bf16_to_f32(got[i]);
        float r = bf16_to_f32(ref[i]);
        diff_sq += (double)(g - r) * (g - r);
        sum_sq += (double)r * r;
        if (got[i] != ref[i]) mismatches++;
    }
    if (bit_mismatches) *bit_mismatches = mismatches;
    return sqrt(diff_sq / (sum_sq > 0 ? sum_sq : 1.0));
}

static double compare_f32(const float *got, const float *ref, size_t count) {
    double sum_sq = 0.0, diff_sq = 0.0;
    for (size_t i = 0; i < count; i++) {
        double d = (double)got[i] - (double)ref[i];
        diff_sq += d * d;
        sum_sq += (double)ref[i] * ref[i];
    }
    return sqrt(diff_sq / (sum_sq > 0 ? sum_sq : 1.0));
}

static char *read_text(const char *path) {
    FILE *handle = fopen(path, "rb");
    if (!handle) return NULL;
    fseek(handle, 0, SEEK_END);
    long size = ftell(handle);
    fseek(handle, 0, SEEK_SET);
    char *text = malloc((size_t)size + 1);
    if (!text || fread(text, 1, (size_t)size, handle) != (size_t)size) {
        free(text);
        fclose(handle);
        return NULL;
    }
    text[size] = '\0';
    fclose(handle);
    return text;
}

static uint32_t *json_ids(const yw_json *value, size_t *count) {
    *count = yw_json_count(value);
    uint32_t *ids = malloc(sizeof(uint32_t) * (*count ? *count : 1));
    for (size_t i = 0; i < *count; i++)
        ids[i] = (uint32_t)yw_json_number(yw_json_at(value, i));
    return ids;
}

int main(void) {
    char error[256];
    char *text = read_text("tests/nar_fixture/manifest.json");
    yw_json *manifest = NULL;
    if (!text || yw_json_parse(text, &manifest, error, sizeof(error)) == 0) {
        fprintf(stderr, "manifest: %s\n", error);
        return 1;
    }

    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "gpu: %s\n", error);
        return 1;
    }
    yue_ar *ar = yue_ar_create(gpu, MODEL_DIR, error, sizeof(error));
    if (!ar) {
        fprintf(stderr, "ar: %s\n", error);
        return 1;
    }
    yue_nar *nar = yue_nar_create(gpu, ar, MODEL_DIR, error, sizeof(error));
    if (!nar) {
        fprintf(stderr, "nar: %s\n", error);
        return 1;
    }

    /* ---- conditioning K/V parity (layers 0 and 27) --------------------- */
    {
        const yw_json *vel = yw_json_at(yw_json_get(manifest, "velocity"), 0);
        size_t token_count = 0;
        uint32_t *tokens = json_ids(yw_json_get(vel, "ar_tokens"), &token_count);
        yue_nar_chunk *chunk = yue_nar_chunk_create(
            nar, tokens, token_count, (size_t)yw_json_number(yw_json_get(vel, "frames")),
            0, error, sizeof(error));
        check(chunk != NULL, "chunk create");
        if (chunk) {
            size_t kv_rows = token_count * YUE_KV_HEADS * YUE_HEAD_DIM;
            struct { const char *name; int layer; int value; } kvs[] = {
                {"cond_l0_key", 0, 0}, {"cond_l0_value", 0, 1},
                {"cond_l27_key", 27, 0}, {"cond_l27_value", 27, 1},
            };
            for (size_t c = 0; c < 4; c++) {
                size_t n_ref = 0;
                uint16_t *ref = load_bits(kvs[c].name, &n_ref);
                uint16_t *got = malloc(sizeof(uint16_t) * kv_rows);
                int ok = ref && n_ref == kv_rows && got &&
                         h3_gpu_tensor_read_bf16(
                             kvs[c].value ? yue_ar_cache_value(
                                                yue_nar_chunk_cond(chunk),
                                                kvs[c].layer)
                                          : yue_ar_cache_key(
                                                yue_nar_chunk_cond(chunk),
                                                kvs[c].layer),
                             got, kv_rows);
                if (ok) {
                    size_t bits = 0;
                    double rel = compare_bits(got, ref, kv_rows, &bits);
                    /* deep layers accumulate bf16 noise (layer 27 ~0.02,
                     * same band as the M2 per-layer gates) */
                    ok = rel < 0.05;
                    if (!ok)
                        printf("  %s relL2=%.4g bits=%zu/%zu\n", kvs[c].name,
                               rel, bits, kv_rows);
                    else
                        printf("%-52s relL2=%.2g bits=%zu\n", kvs[c].name, rel,
                               bits);
                } else {
                    printf("  %s load/read failed\n", kvs[c].name);
                }
                check(ok, kvs[c].name);
                free(ref);
                free(got);
            }
            /* ---- velocity unit cases ----------------------------------- */
            size_t cases = yw_json_count(yw_json_get(manifest, "velocity"));
            for (size_t v = 0; v < cases; v++) {
                const yw_json *item =
                    yw_json_at(yw_json_get(manifest, "velocity"), v);
                const char *name = yw_json_get_string(item, "name", "?");
                size_t frames = (size_t)yw_json_number(yw_json_get(item, "frames"));
                size_t n_state = 0;
                uint16_t *state = load_bits(yw_json_get_string(item, "state", ""),
                                            &n_state);
                size_t n_ref = 0;
                uint16_t *ref =
                    load_bits(yw_json_get_string(item, "out", ""), &n_ref);
                uint16_t *got =
                    malloc(sizeof(uint16_t) * frames * YUE_LATENT_DIM);
                int ok = state && n_state == frames * YUE_LATENT_DIM && ref &&
                         n_ref == frames * YUE_LATENT_DIM && got &&
                         yue_nar_velocity(chunk, state,
                                          yw_json_get_number(item, "raw_t", 0),
                                          got, error, sizeof(error));
                if (ok) {
                    size_t bits = 0;
                    double rel = compare_bits(got, ref, frames * YUE_LATENT_DIM,
                                              &bits);
                    ok = rel < 0.06;
                    if (!ok)
                        printf("  %s relL2=%.4g bits=%zu/%zu\n", name, rel,
                               bits, frames * YUE_LATENT_DIM);
                    else
                        printf("%-52s relL2=%.2g bits=%zu\n", name, rel, bits);
                } else {
                    printf("  %s failed: %s\n", name, error);
                }
                check(ok, name);
                free(state);
                free(ref);
                free(got);
            }

            /* ---- midpoint solve, per-step trajectory comparison -------- */
            const yw_json *solve = yw_json_get(manifest, "solve");
            size_t n_noise = 0;
            float *noise =
                load_f32(yw_json_get_string(solve, "noise", ""), &n_noise);
            size_t frames = (size_t)yw_json_number(yw_json_get(solve, "frames"));
            const int steps = (int)yw_json_number(yw_json_get(solve, "steps"));
            size_t n = frames * YUE_LATENT_DIM;
            uint16_t *state = malloc(sizeof(uint16_t) * n);
            uint16_t *mid = malloc(sizeof(uint16_t) * n);
            uint16_t *vel_buf = malloc(sizeof(uint16_t) * n);
            float *out = malloc(sizeof(float) * n);
            int ok = noise && n_noise == n && state && mid && vel_buf && out;
            /* host-side midpoint loop mirroring yue_nar_solve so each step
             * can be compared against the reference trajectory */
            const double dt = 1.0 / (double)steps;
            for (size_t i = 0; ok && i < n; i++) {
                uint32_t wide;
                float v = noise[i];
                memcpy(&wide, &v, 4);
                wide += 0x7fffu + ((wide >> 16) & 1u);
                state[i] = (uint16_t)(wide >> 16);
            }
            for (int step = 0; ok && step < steps; step++) {
                double t = 1.0 - (double)step * dt;
                double raw = t <= 0 ? -20.0 : t >= 1 ? 20.0
                             : fmax(-20.0, fmin(20.0, log(t / (1.0 - t))));
                ok = yue_nar_velocity(chunk, state, raw, vel_buf, error,
                                      sizeof(error));
                if (!ok) break;
                float half_dt = (float)(dt / 2.0);
                for (size_t i = 0; i < n; i++) {
                    float prod = bf16_to_f32(vel_buf[i]) * half_dt;
                    uint32_t b;
                    memcpy(&b, &prod, 4);
                    b += 0x7fffu + ((b >> 16) & 1u);
                    b &= 0xffff0000u;
                    float prod_r;
                    memcpy(&prod_r, &b, 4);
                    float diff = bf16_to_f32(state[i]) - prod_r;
                    memcpy(&b, &diff, 4);
                    b += 0x7fffu + ((b >> 16) & 1u);
                    b &= 0xffff0000u;
                    mid[i] = (uint16_t)(b >> 16);
                }
                double mid_t = t - dt / 2.0;
                double raw_mid = mid_t <= 0 ? -20.0
                                 : mid_t >= 1 ? 20.0
                                 : fmax(-20.0, fmin(20.0, log(mid_t / (1.0 - mid_t))));
                ok = yue_nar_velocity(chunk, mid, raw_mid, vel_buf, error,
                                      sizeof(error));
                if (!ok) break;
                float full_dt = (float)dt;
                for (size_t i = 0; i < n; i++) {
                    float prod = bf16_to_f32(vel_buf[i]) * full_dt;
                    uint32_t b;
                    memcpy(&b, &prod, 4);
                    b += 0x7fffu + ((b >> 16) & 1u);
                    b &= 0xffff0000u;
                    float prod_r;
                    memcpy(&prod_r, &b, 4);
                    float diff = bf16_to_f32(state[i]) - prod_r;
                    memcpy(&b, &diff, 4);
                    b += 0x7fffu + ((b >> 16) & 1u);
                    b &= 0xffff0000u;
                    state[i] = (uint16_t)(b >> 16);
                }
                /* compare against the reference per-step state */
                char step_name[64];
                snprintf(step_name, sizeof(step_name), "traj_state%d", step);
                size_t n_ref = 0;
                uint16_t *ref = load_bits(step_name, &n_ref);
                if (ref && n_ref == n) {
                    size_t bits = 0;
                    double rel = compare_bits(state, ref, n, &bits);
                    printf("  step %d: relL2=%.4g bits=%zu/%zu\n", step, rel,
                           bits, n);
                    if (rel > 0.12) ok = 0;
                }
                free(ref);
            }
            if (ok) {
                size_t n_ref = 0;
                float *ref =
                    load_f32(yw_json_get_string(solve, "final", ""), &n_ref);
                ok = ref && n_ref == n;
                if (ok) {
                    for (size_t i = 0; i < n; i++)
                        out[i] = bf16_to_f32(state[i]);
                    double rel = compare_f32(out, ref, n);
                    ok = rel < 0.12;
                    if (!ok)
                        printf("  solve relL2=%.4g\n", rel);
                    else
                        printf("%-52s relL2=%.2g\n", "midpoint solve", rel);
                }
                free(ref);
            } else {
                printf("  solve failed: %s\n", error);
            }
            check(ok, "midpoint solve");
            free(noise);
            free(state);
            free(mid);
            free(vel_buf);
            free(out);
            yue_nar_chunk_free(chunk);
        }
        free(tokens);
    }

    /* ---- multi-chunk with a tiny context -------------------------------- */
    {
        size_t prefix_len =
            (size_t)yw_json_number(yw_json_get(manifest, "multi_prefix_len"));
        size_t total_frames =
            (size_t)yw_json_number(yw_json_get(manifest, "multi_frames"));
        int context =
            (int)yw_json_number(yw_json_get(manifest, "multi_context"));
        yue_chunk_range ranges[64];
        size_t count = 0;
        int ranges_ok =
            yue_chunk_ranges(total_frames, prefix_len, (size_t)context,
                             ranges, 64, &count, error, sizeof(error));
        const yw_json *want = yw_json_get(manifest, "multi_ranges");
        ranges_ok = ranges_ok && count == yw_json_count(want);
        for (size_t i = 0; ranges_ok && i < count; i++) {
            const yw_json *r = yw_json_at(want, i);
            ranges_ok = (size_t)yw_json_number(yw_json_at(r, 0)) ==
                            ranges[i].start &&
                        (size_t)yw_json_number(yw_json_at(r, 1)) ==
                            ranges[i].end;
        }
        check(ranges_ok, "chunk_ranges parity");

        const yw_json *chunks = yw_json_get(manifest, "multi");
        size_t n_chunks = yw_json_count(chunks);
        for (size_t c = 0; c < n_chunks; c++) {
            const yw_json *item = yw_json_at(chunks, c);
            char name[64];
            snprintf(name, sizeof(name), "multi chunk %zu", c);
            size_t token_count = 0;
            uint32_t *tokens =
                json_ids(yw_json_get(item, "ar_tokens"), &token_count);
            size_t n_noise = 0;
            float *noise =
                load_f32(yw_json_get_string(item, "noise", ""), &n_noise);
            size_t frames = n_noise / YUE_LATENT_DIM;
            yue_nar_chunk *chunk = yue_nar_chunk_create(
                nar, tokens, token_count, frames, 0, error, sizeof(error));
            float *out = malloc(sizeof(float) * frames * YUE_LATENT_DIM);
            int ok = chunk && noise && out &&
                     yue_nar_solve(chunk, noise, 2, out, error, sizeof(error));
            if (ok) {
                size_t n_ref = 0;
                float *ref =
                    load_f32(yw_json_get_string(item, "final", ""), &n_ref);
                ok = ref && n_ref == frames * YUE_LATENT_DIM;
                if (ok) {
                    double rel =
                        compare_f32(out, ref, frames * YUE_LATENT_DIM);
                    ok = rel < 0.12;
                    if (!ok)
                        printf("  %s relL2=%.4g\n", name, rel);
                    else
                        printf("%-52s relL2=%.2g\n", name, rel);
                }
                free(ref);
            } else {
                printf("  %s failed: %s\n", name, error);
            }
            check(ok, name);
            yue_nar_chunk_free(chunk);
            free(tokens);
            free(noise);
            free(out);
        }
    }

    yw_json_free(manifest);
    free(text);
    yue_nar_free(nar);
    yue_ar_free(ar);
    h3_gpu_free(gpu);
    printf("m4 nar tests: %d failure(s)\n", failures);
    return failures != 0;
}
