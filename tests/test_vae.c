/* M5 VAE decoder parity against the lyra reference (gen_vae_fixture.py):
 * per-stage fp32 activations for a small full decode plus final audio for
 * full and chunked (core=16/halo=16) decodes. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "yue.h"
#include "yue_json.h"
#include "yue_vae.h"
#include "h3_gpu.h"

#define VAE_DIR "/Volumes/Work/models/YuE2.models/lyra-YuE2-3B-vae"
#define DUMP_DIR "/tmp/yue_vae_dump"

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
    *elem_size = 4;
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

static double rel_f32(const float *got, const float *ref, size_t count,
                      double *max_abs) {
    double sum_sq = 0.0, diff_sq = 0.0;
    *max_abs = 0.0;
    for (size_t i = 0; i < count; i++) {
        double d = (double)got[i] - (double)ref[i];
        diff_sq += d * d;
        sum_sq += (double)ref[i] * ref[i];
        double a = fabs(d);
        if (a > *max_abs) *max_abs = a;
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

int main(void) {
    char error[256];
    mkdir(DUMP_DIR, 0755);
    /* clear stale stage dumps */
    {
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "rm -f %s/stage*.bin", DUMP_DIR);
        if (system(cmd) != 0) {
        }
    }
    setenv("QWEN_VAE_DUMP", DUMP_DIR, 1);

    /* Exercise the GPU decoder when Metal is reachable (the test runs
     * from the repo root, where h3_shaders.metal lives); fall back to
     * the reference CPU path otherwise. */
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu)
        fprintf(stderr, "note: GPU decoder unavailable (%s); using CPU\n",
                error);

    yue_vae *vae = yue_vae_create(VAE_DIR, gpu, error, sizeof(error));
    if (!vae) {
        fprintf(stderr, "vae: %s\n", error);
        h3_gpu_free(gpu);
        return 1;
    }

    char *text = read_text("tests/vae_fixture/manifest.json");
    yw_json *manifest = NULL;
    if (!text || yw_json_parse(text, &manifest, error, sizeof(error)) == 0) {
        fprintf(stderr, "manifest: %s\n", error);
        return 1;
    }

    /* ---- small full decode with per-stage comparison ------------------- */
    {
        size_t n_lat = 0;
        float *latents = read_npy("tests/vae_fixture/small_latents.npy", &n_lat,
                                  &(int){4});
        if (!latents || n_lat != 8 * 64) {
            check(0, "small latents load");
            return 1;
        }
        float *audio =
            malloc(sizeof(float) * yue_vae_output_length(vae, 8) * 2);
        int ok = yue_vae_decode(vae, latents, 8, audio, error, sizeof(error));
        check(ok, "small decode");
        if (ok) {
            /* compare every dumped stage against the fixture */
            const yw_json *stages = yw_json_get(manifest, "stages");
            size_t count = yw_json_count(stages);
            int all_ok = 1;
            double worst = 0.0;
            for (size_t s = 0; s < count && all_ok; s++) {
                char name[32];
                snprintf(name, sizeof(name), "stage%02zu", s);
                const yw_json *shape = yw_json_get(stages, name);
                size_t t = (size_t)yw_json_number(yw_json_at(shape, 0));
                int c = (int)yw_json_number(yw_json_at(shape, 1));
                char dump_path[512];
                snprintf(dump_path, sizeof(dump_path),
                         DUMP_DIR "/stage%02zu_t%zu_c%d.bin", s, t, c);
                FILE *file = fopen(dump_path, "rb");
                if (!file) {
                    printf("  %s: C dump missing (%s)\n", name, dump_path);
                    all_ok = 0;
                    break;
                }
                size_t elems = t * (size_t)c;
                float *got = malloc(sizeof(float) * elems);
                size_t read = got ? fread(got, sizeof(float), elems, file) : 0;
                fclose(file);
                size_t n_ref = 0;
                float *ref = read_npy(
                    (snprintf(dump_path, sizeof(dump_path),
                              "tests/vae_fixture/%s.npy", name),
                     dump_path),
                    &n_ref, &(int){4});
                if (!got || read != elems || !ref || n_ref != elems) {
                    printf("  %s: load failed (read %zu want %zu)\n", name,
                           read, elems);
                    all_ok = 0;
                    free(got);
                    free(ref);
                    break;
                }
                double max_abs = 0.0;
                double rel = rel_f32(got, ref, elems, &max_abs);
                if (rel > 1e-3) {
                    printf("  %s: relL2=%.3g max=%.3g\n", name, rel, max_abs);
                    all_ok = 0;
                }
                if (rel > worst) worst = rel;
                free(got);
                free(ref);
            }
            if (all_ok)
                printf("%-52s worst relL2=%.2g\n", "small per-stage", worst);
            check(all_ok, "small per-stage parity");

            /* final audio */
            size_t n_ref = 0;
            float *ref =
                read_npy("tests/vae_fixture/small_audio.npy", &n_ref,
                         &(int){4});
            double max_abs = 0.0;
            double rel = rel_f32(audio, ref, n_ref, &max_abs);
            check(rel < 2e-4, "small audio parity");
            if (rel >= 2e-4)
                printf("  audio relL2=%.3g max=%.3g\n", rel, max_abs);
            free(ref);
        }
        free(latents);
        free(audio);
    }

    /* ---- full 40-frame per-stage comparison (long sequences) ------------ */
    {
        if (system("rm -f " DUMP_DIR "/stage*.bin") != 0) {
        }
        size_t n_lat = 0;
        float *latents = read_npy("tests/vae_fixture/chunk_latents.npy", &n_lat,
                                  &(int){4});
        float *audio =
            malloc(sizeof(float) * yue_vae_output_length(vae, 40) * 2);
        int ok = latents && n_lat == 40 * 64 && audio &&
                 yue_vae_decode(vae, latents, 40, audio, error, sizeof(error));
        check(ok, "full40 decode");
        if (ok) {
            const yw_json *stages = yw_json_get(manifest, "full40_stages");
            size_t count = yw_json_count(stages);
            int all_ok = 1;
            double worst = 0.0;
            for (size_t s = 0; s < count && all_ok; s++) {
                char name[32];
                snprintf(name, sizeof(name), "full%02zu", s);
                const yw_json *shape = yw_json_get(stages, name);
                size_t t = (size_t)yw_json_number(yw_json_at(shape, 0));
                int c = (int)yw_json_number(yw_json_at(shape, 1));
                char dump_path[512];
                snprintf(dump_path, sizeof(dump_path),
                         DUMP_DIR "/stage%02zu_t%zu_c%d.bin", s, t, c);
                FILE *file = fopen(dump_path, "rb");
                if (!file) {
                    printf("  %s: C dump missing (%s)\n", name, dump_path);
                    all_ok = 0;
                    break;
                }
                size_t elems = t * (size_t)c;
                float *got = malloc(sizeof(float) * elems);
                size_t read = got ? fread(got, sizeof(float), elems, file) : 0;
                fclose(file);
                size_t n_ref = 0;
                float *ref = read_npy(
                    (snprintf(dump_path, sizeof(dump_path),
                              "tests/vae_fixture/%s.npy", name),
                     dump_path),
                    &n_ref, &(int){4});
                if (!got || read != elems || !ref || n_ref != elems) {
                    printf("  %s: load failed\n", name);
                    all_ok = 0;
                    free(got);
                    free(ref);
                    break;
                }
                double max_abs = 0.0;
                double rel = rel_f32(got, ref, elems, &max_abs);
                if (rel > 1e-3) {
                    printf("  %s: relL2=%.3g max=%.3g\n", name, rel, max_abs);
                    all_ok = 0;
                }
                if (rel > worst) worst = rel;
                free(got);
                free(ref);
            }
            if (all_ok)
                printf("%-52s worst relL2=%.2g\n", "full40 per-stage", worst);
            check(all_ok, "full40 per-stage parity");

            size_t n_ref = 0;
            float *ref = read_npy("tests/vae_fixture/full40_audio.npy", &n_ref,
                                  &(int){4});
            double max_abs = 0.0;
            double rel = rel_f32(audio, ref, n_ref, &max_abs);
            check(rel < 2e-4, "full40 audio parity");
            if (rel >= 2e-4)
                printf("  full40 audio relL2=%.3g max=%.3g\n", rel, max_abs);
            free(ref);
        }
        free(latents);
        free(audio);
    }

    /* ---- chunked decode across tile boundaries -------------------------- */
    {
        size_t n_lat = 0;
        float *latents = read_npy("tests/vae_fixture/chunk_latents.npy", &n_lat,
                                  &(int){4});
        if (!latents || n_lat != 40 * 64) {
            check(0, "chunk latents load");
            return 1;
        }
        float *audio =
            malloc(sizeof(float) * yue_vae_output_length(vae, 40) * 2);
        int ok = yue_vae_decode(vae, latents, 40, audio, error, sizeof(error));
        if (ok) {
            size_t n_ref = 0;
            float *ref = read_npy("tests/vae_fixture/chunk_audio.npy", &n_ref,
                                  &(int){4});
            ok = ref && n_ref == yue_vae_output_length(vae, 40) * 2;
            if (ok) {
                double max_abs = 0.0;
                double rel = rel_f32(audio, ref, n_ref, &max_abs);
                check(rel < 2e-4, "chunked audio parity");
                if (rel >= 2e-4)
                    printf("  chunked relL2=%.3g max=%.3g\n", rel, max_abs);
            } else {
                check(0, "chunked audio parity");
                printf("  reference load failed\n");
            }
            free(ref);
        } else {
            check(0, "chunked decode");
            printf("  %s\n", error);
        }
        free(latents);
        free(audio);
    }

    yw_json_free(manifest);
    free(text);
    yue_vae_free(vae);
    h3_gpu_free(gpu);
    printf("m5 vae tests: %d failure(s)\n", failures);
    return failures != 0;
}
