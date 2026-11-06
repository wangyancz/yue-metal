/* YuE2 acoustic flow matching on Metal (lyra nar.py parity). The acoustic
 * experts attend non-causally over the AR backbone's causal conditioning
 * cache; the midpoint solver runs on the host so the bf16 rounding points
 * (velocity products, solver updates) match the reference exactly. */
#import <Foundation/Foundation.h>

#include "yue_nar.h"
#include "yue_gpu_ext.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* Weight file resolution, mirroring yue_ar.m: prefer the lyra split layout,
 * fall back to the official single-file checkpoint. */
static int resolve_weights_path(char *out, size_t size, const char *model_dir,
                                const char *split_name) {
    int written = snprintf(out, size, "%s/%s", model_dir, split_name);
    if (written < 0 || (size_t)written >= size) return 0;
    struct stat st;
    if (stat(out, &st) == 0 && S_ISREG(st.st_mode)) return 1;
    written = snprintf(out, size, "%s/model.safetensors", model_dir);
    if (written < 0 || (size_t)written >= size) return 0;
    return stat(out, &st) == 0 && S_ISREG(st.st_mode);
}

#define YUE_NAR_KV_HEADS YUE_KV_HEADS
#define YUE_NAR_PAIRS (YUE_HEAD_DIM / 2)
#define YUE_TIME_HALF (YUE_TIME_EMBED / 2)

/* RNE bf16 rounding for host-side derived values. */
static float bf16_round(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float bf16_to_f32(uint16_t bits) {
    uint32_t wide = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &wide, sizeof(value));
    return value;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static int set_error(char *error, size_t error_size, const char *fmt, ...) {
    if (error && error_size) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(error, error_size, fmt, args);
        va_end(args);
    }
    return 0;
}

typedef struct {
    h3_gpu_tensor *input_norm;
    h3_gpu_tensor *q, *k, *v, *o;
    h3_gpu_tensor *q_norm, *k_norm;
    h3_gpu_tensor *post_norm;
    h3_gpu_tensor *fc1; /* [2*mlp, hidden] gate|up concat */
    h3_gpu_tensor *down;
} yue_nar_layer;

struct yue_nar {
    h3_gpu *gpu;
    yue_ar *ar; /* shared backbone, not owned */
    int layers;
    char weights_path[4096];
    yue_nar_layer *layer;
    h3_gpu_tensor *llm2vae_w, *llm2vae_b; /* [64,2048], [64] */
    h3_gpu_tensor *vae2llm_w, *vae2llm_b; /* [2048,64], [2048] */
    h3_gpu_tensor *time1_w, *time1_b;     /* [2048,256], [2048] */
    h3_gpu_tensor *time2_w, *time2_b;     /* [2048,2048], [2048] */
    h3_gpu_tensor *pos_embed;             /* [max_frames, hidden] */
    size_t pos_rows;
    float timestep_shift;
    float time_freq[YUE_TIME_HALF]; /* fp32 sinusoid frequencies */

    /* grow-only workspaces (rows = nar_length) */
    size_t ws_rows;
    h3_gpu_tensor *x, *normed, *q, *k, *v, *attn, *branch, *fc1out, *mlp_mid;
    h3_gpu_tensor *time_emb, *boundary, *llm_out;
    h3_gpu_tensor *time_in_t; /* [1, 256] sinusoid embedding */
    h3_gpu_tensor *time_mid;  /* [1, hidden] timestep MLP staging */
    h3_gpu_tensor *time_vec;  /* [1, hidden] broadcast source */
};

struct yue_nar_chunk {
    yue_nar *nar;
    size_t ar_len;   /* conditioning tokens (prefix + codec + MUSIC_END) */
    size_t visible;  /* conditioning rows the acoustic layers attend to */
    size_t frames;   /* latent frames (state rows) */
    size_t nar_len;  /* frames + 2 boundary rows */
    yue_ar_cache *cond; /* AR K/V over all ar_len rows */
    /* combined [visible + nar_len, kv_heads, head_dim] per layer */
    h3_gpu_tensor **ck, **cv;
    h3_gpu_tensor *rope_cos, *rope_sin; /* [nar_len, pairs] fp32 */
    h3_gpu_tensor *pos_slice;           /* [nar_len, hidden] bf16 */
    h3_gpu_tensor *limit;               /* u32 per query row, all visible+nar */
};

/* ------------------------------------------------------------------ create */

static h3_gpu_tensor *load_offset(h3_gpu *gpu, const h3_st_header *header,
                                  const char *path, const char *name, int ndim,
                                  const uint64_t *shape, char *error,
                                  size_t error_size) {
    const h3_st_tensor *tensor = h3_st_find(header, name);
    if (!tensor) {
        set_error(error, error_size, "missing acoustic weight %s", name);
        return NULL;
    }
    if (tensor->dtype != H3_DTYPE_BF16) {
        set_error(error, error_size, "weight %s is not BF16", name);
        return NULL;
    }
    if (tensor->ndim != ndim) {
        set_error(error, error_size, "weight %s rank mismatch", name);
        return NULL;
    }
    for (int d = 0; d < ndim; d++) {
        if (tensor->shape[d] != shape[d]) {
            set_error(error, error_size, "weight %s shape mismatch", name);
            return NULL;
        }
    }
    return h3_gpu_tensor_load_bf16(gpu, path, tensor->file_offset,
                                   (size_t)h3_st_tensor_elements(tensor));
}

yue_nar *yue_nar_create(h3_gpu *gpu, yue_ar *ar, const char *model_dir,
                        char *error, size_t error_size) {
    if (!gpu || !ar || !model_dir) {
        set_error(error, error_size, "invalid nar arguments");
        return NULL;
    }
    yue_nar *nar = calloc(1, sizeof(yue_nar));
    if (!nar) {
        set_error(error, error_size, "out of memory");
        return NULL;
    }
    nar->gpu = gpu;
    nar->ar = ar;
    nar->layers = YUE_AR_LAYERS;
    if (!resolve_weights_path(nar->weights_path, sizeof(nar->weights_path),
                              model_dir, "nar-bf16.safetensors")) {
        yue_nar_free(nar);
        set_error(error, error_size, "model directory path is too long");
        return NULL;
    }

    h3_st_header header;
    if (!h3_st_read_header(nar->weights_path, &header, error, error_size)) {
        yue_nar_free(nar);
        return NULL;
    }

    const uint64_t hidden2[2] = {YUE_HIDDEN, YUE_HIDDEN};
    const uint64_t kv_hidden[2] = {YUE_NAR_KV_HEADS * YUE_HEAD_DIM, YUE_HIDDEN};
    const uint64_t hidden_mlp[2] = {YUE_HIDDEN, YUE_MLP};
    const uint64_t scalar[1] = {YUE_HIDDEN};
    const uint64_t scalar_dim[1] = {YUE_HEAD_DIM};
    const uint64_t vae_in[2] = {YUE_HIDDEN, YUE_LATENT_DIM};
    const uint64_t vae_out[2] = {YUE_LATENT_DIM, YUE_HIDDEN};
    const uint64_t vae_bias[1] = {YUE_LATENT_DIM};
    const uint64_t time_in[2] = {YUE_HIDDEN, YUE_TIME_EMBED};

    nar->layer = calloc((size_t)nar->layers, sizeof(yue_nar_layer));
    if (!nar->layer) {
        h3_st_free_header(&header);
        yue_nar_free(nar);
        set_error(error, error_size, "out of memory");
        return NULL;
    }

    char name[160];
    for (int i = 0; i < nar->layers; i++) {
        yue_nar_layer *layer = &nar->layer[i];
#define YUE_NAR_LOAD(field, suffix, nd, shp)                                \
        do {                                                                \
            snprintf(name, sizeof(name), "model.layers.%d." suffix, i);     \
            layer->field = load_offset(gpu, &header, nar->weights_path,     \
                                       name, nd, shp, error, error_size);   \
            if (!layer->field) {                                            \
                h3_st_free_header(&header);                                 \
                yue_nar_free(nar);                                          \
                return NULL;                                                \
            }                                                               \
        } while (0)
        YUE_NAR_LOAD(input_norm, "nar_input_layernorm.weight", 1, scalar);
        YUE_NAR_LOAD(q, "nar_self_attn.q_proj.weight", 2, hidden2);
        YUE_NAR_LOAD(k, "nar_self_attn.k_proj.weight", 2, kv_hidden);
        YUE_NAR_LOAD(v, "nar_self_attn.v_proj.weight", 2, kv_hidden);
        YUE_NAR_LOAD(o, "nar_self_attn.o_proj.weight", 2, hidden2);
        YUE_NAR_LOAD(q_norm, "nar_self_attn.q_norm.weight", 1, scalar_dim);
        YUE_NAR_LOAD(k_norm, "nar_self_attn.k_norm.weight", 1, scalar_dim);
        YUE_NAR_LOAD(post_norm, "nar_pre_mlp_layernorm.weight", 1, scalar);
        YUE_NAR_LOAD(down, "nar_mlp.down_proj.weight", 2, hidden_mlp);
#undef YUE_NAR_LOAD

        /* gate|up concat for the fused swiglu kernel */
        const h3_st_tensor *gate_t = NULL, *up_t = NULL;
        snprintf(name, sizeof(name), "model.layers.%d.nar_mlp.gate_proj.weight", i);
        gate_t = h3_st_find(&header, name);
        snprintf(name, sizeof(name), "model.layers.%d.nar_mlp.up_proj.weight", i);
        up_t = h3_st_find(&header, name);
        if (!gate_t || !up_t || gate_t->dtype != H3_DTYPE_BF16 ||
            up_t->dtype != H3_DTYPE_BF16) {
            h3_st_free_header(&header);
            yue_nar_free(nar);
            set_error(error, error_size, "nar mlp gate/up missing at layer %d", i);
            return NULL;
        }
        size_t half = (size_t)YUE_MLP * YUE_HIDDEN;
        uint16_t *host = malloc(sizeof(uint16_t) * half);
        layer->fc1 = h3_gpu_tensor_new_bf16(gpu, 2 * half);
        int ok = layer->fc1 && host &&
                 h3_st_read_data(&header, gate_t, host, half * 2, error,
                                 error_size) &&
                 h3_gpu_tensor_write_bf16_range(layer->fc1, 0, host, half) &&
                 h3_st_read_data(&header, up_t, host, half * 2, error,
                                 error_size) &&
                 h3_gpu_tensor_write_bf16_range(layer->fc1, half, host, half);
        free(host);
        if (!ok) {
            h3_st_free_header(&header);
            yue_nar_free(nar);
            set_error(error, error_size, "nar fc1 concat failed at layer %d", i);
            return NULL;
        }
    }

#define YUE_NAR_TOP(field, key, nd, shp)                                    \
    nar->field = load_offset(gpu, &header, nar->weights_path, key, nd, shp, \
                             error, error_size);                            \
    if (!nar->field) {                                                      \
        h3_st_free_header(&header);                                         \
        yue_nar_free(nar);                                                  \
        return NULL;                                                        \
    }
    YUE_NAR_TOP(llm2vae_w, "llm2vae.weight", 2, vae_out);
    YUE_NAR_TOP(llm2vae_b, "llm2vae.bias", 1, vae_bias);
    YUE_NAR_TOP(vae2llm_w, "vae2llm.weight", 2, vae_in);
    YUE_NAR_TOP(vae2llm_b, "vae2llm.bias", 1, scalar);
    YUE_NAR_TOP(time1_w, "time_embedder.mlp.0.weight", 2, time_in);
    YUE_NAR_TOP(time1_b, "time_embedder.mlp.0.bias", 1, scalar);
    YUE_NAR_TOP(time2_w, "time_embedder.mlp.2.weight", 2, hidden2);
    YUE_NAR_TOP(time2_b, "time_embedder.mlp.2.bias", 1, scalar);
#undef YUE_NAR_TOP

    /* latent_pos_embed.pe: [max_latent_frames, hidden] */
    const h3_st_tensor *pe = h3_st_find(&header, "latent_pos_embed.pe");
    if (!pe || pe->dtype != H3_DTYPE_BF16 || pe->ndim != 2 ||
        pe->shape[1] != YUE_HIDDEN || pe->shape[0] < 1) {
        h3_st_free_header(&header);
        yue_nar_free(nar);
        set_error(error, error_size, "latent_pos_embed.pe missing or invalid");
        return NULL;
    }
    nar->pos_embed = h3_gpu_tensor_load_bf16(
        gpu, nar->weights_path, pe->file_offset,
        (size_t)pe->shape[0] * pe->shape[1]);
    h3_st_free_header(&header);
    if (!nar->pos_embed) {
        yue_nar_free(nar);
        set_error(error, error_size, "pos embed load failed");
        return NULL;
    }
    nar->pos_rows = (size_t)pe->shape[0];

    /* Timestep shift from config.json (validated downstream users). */
    nar->timestep_shift = 1.0f;
    {
        char cfg_path[4096];
        snprintf(cfg_path, sizeof(cfg_path), "%s/config.json", model_dir);
        FILE *file = fopen(cfg_path, "rb");
        if (file) {
            char text[16384];
            size_t got = fread(text, 1, sizeof(text) - 1, file);
            fclose(file);
            text[got] = '\0';
            /* A config that fills the buffer was truncated mid-JSON: the
             * strstr/atof scan below could read a half-written number and
             * produce a wrong-but-positive shift. Keep the reference
             * default instead (same policy as the generation-config
             * short-read rejection). */
            if (got < sizeof(text) - 1) {
                const char *key = strstr(text, "\"timestep_shift\"");
                if (key) {
                    const char *colon = strchr(key, ':');
                    if (colon) {
                        /* A zero or negative shift makes the shifted
                         * timestep degenerate (0/(1-t)); fall back to the
                         * reference default instead of silently corrupting
                         * the schedule. */
                        float value = (float)atof(colon + 1);
                        nar->timestep_shift = value > 0.0f ? value : 1.0f;
                    }
                }
            }
        }
    }

    /* fp32 sinusoid frequencies: exp((-ln 10000) * i / half), matching the
     * MLX expression order ((scalar * arange) / half). */
    const float neg_log = (float)(-log(10000.0));
    for (int i = 0; i < YUE_TIME_HALF; i++)
        nar->time_freq[i] = expf((neg_log * (float)i) / (float)YUE_TIME_HALF);

    nar->time_in_t = h3_gpu_tensor_new_bf16(gpu, YUE_TIME_EMBED);
    nar->time_mid = h3_gpu_tensor_new_bf16(gpu, YUE_HIDDEN);
    nar->time_vec = h3_gpu_tensor_new_bf16(gpu, YUE_HIDDEN);
    if (!nar->time_in_t || !nar->time_mid || !nar->time_vec) {
        yue_nar_free(nar);
        set_error(error, error_size, "time staging allocation failed");
        return NULL;
    }

    return nar;
}

void yue_nar_free(yue_nar *nar) {
    if (!nar) return;
    if (nar->layer) {
        for (int i = 0; i < nar->layers; i++) {
            h3_gpu_tensor_free(nar->layer[i].input_norm);
            h3_gpu_tensor_free(nar->layer[i].q);
            h3_gpu_tensor_free(nar->layer[i].k);
            h3_gpu_tensor_free(nar->layer[i].v);
            h3_gpu_tensor_free(nar->layer[i].o);
            h3_gpu_tensor_free(nar->layer[i].q_norm);
            h3_gpu_tensor_free(nar->layer[i].k_norm);
            h3_gpu_tensor_free(nar->layer[i].post_norm);
            h3_gpu_tensor_free(nar->layer[i].fc1);
            h3_gpu_tensor_free(nar->layer[i].down);
        }
        free(nar->layer);
    }
    h3_gpu_tensor_free(nar->llm2vae_w);
    h3_gpu_tensor_free(nar->llm2vae_b);
    h3_gpu_tensor_free(nar->vae2llm_w);
    h3_gpu_tensor_free(nar->vae2llm_b);
    h3_gpu_tensor_free(nar->time1_w);
    h3_gpu_tensor_free(nar->time1_b);
    h3_gpu_tensor_free(nar->time2_w);
    h3_gpu_tensor_free(nar->time2_b);
    h3_gpu_tensor_free(nar->pos_embed);
    h3_gpu_tensor_free(nar->x);
    h3_gpu_tensor_free(nar->normed);
    h3_gpu_tensor_free(nar->q);
    h3_gpu_tensor_free(nar->k);
    h3_gpu_tensor_free(nar->v);
    h3_gpu_tensor_free(nar->attn);
    h3_gpu_tensor_free(nar->branch);
    h3_gpu_tensor_free(nar->fc1out);
    h3_gpu_tensor_free(nar->mlp_mid);
    h3_gpu_tensor_free(nar->time_emb);
    h3_gpu_tensor_free(nar->boundary);
    h3_gpu_tensor_free(nar->llm_out);
    h3_gpu_tensor_free(nar->time_in_t);
    h3_gpu_tensor_free(nar->time_mid);
    h3_gpu_tensor_free(nar->time_vec);
    free(nar);
}

/* -------------------------------------------------------------- workspaces */

static int workspace_reserve(yue_nar *nar, size_t rows, char *error,
                             size_t error_size) {
    if (nar->ws_rows >= rows) return 1;
    size_t cap = nar->ws_rows ? nar->ws_rows : 1024;
    /* The tensor sizes below multiply cap by up to 2*YUE_MLP elements;
     * refuse pathological row counts instead of wrapping size_t. */
    if (rows > ((size_t)-1) / (2 * YUE_MLP)) {
        set_error(error, error_size, "NAR workspace too large");
        return 0;
    }
    while (cap < rows) {
        if (cap > ((size_t)-1) / 2) {
            set_error(error, error_size, "NAR workspace too large");
            return 0;
        }
        cap *= 2;
    }
    h3_gpu_tensor_free(nar->x);
    h3_gpu_tensor_free(nar->normed);
    h3_gpu_tensor_free(nar->q);
    h3_gpu_tensor_free(nar->k);
    h3_gpu_tensor_free(nar->v);
    h3_gpu_tensor_free(nar->attn);
    h3_gpu_tensor_free(nar->branch);
    h3_gpu_tensor_free(nar->fc1out);
    h3_gpu_tensor_free(nar->mlp_mid);
    h3_gpu_tensor_free(nar->time_emb);
    h3_gpu_tensor_free(nar->boundary);
    h3_gpu_tensor_free(nar->llm_out);
    nar->x = h3_gpu_tensor_new_bf16(nar->gpu, cap * YUE_HIDDEN);
    nar->normed = h3_gpu_tensor_new_bf16(nar->gpu, cap * YUE_HIDDEN);
    nar->q = h3_gpu_tensor_new_bf16(nar->gpu, cap * YUE_HEADS * YUE_HEAD_DIM);
    nar->k = h3_gpu_tensor_new_bf16(nar->gpu,
                                    cap * YUE_NAR_KV_HEADS * YUE_HEAD_DIM);
    nar->v = nar->k ? h3_gpu_tensor_new_bf16(
                          nar->gpu, cap * YUE_NAR_KV_HEADS * YUE_HEAD_DIM)
                    : NULL;
    nar->attn = nar->x ? h3_gpu_tensor_new_bf16(nar->gpu, cap * YUE_HIDDEN)
                       : NULL;
    nar->branch = nar->x ? h3_gpu_tensor_new_bf16(nar->gpu, cap * YUE_HIDDEN)
                         : NULL;
    nar->fc1out = nar->x ? h3_gpu_tensor_new_bf16(nar->gpu, cap * 2 * YUE_MLP)
                         : NULL;
    nar->mlp_mid = nar->x ? h3_gpu_tensor_new_bf16(nar->gpu, cap * YUE_MLP)
                          : NULL;
    nar->time_emb = nar->x ? h3_gpu_tensor_new_bf16(nar->gpu, cap * YUE_HIDDEN)
                           : NULL;
    nar->boundary = h3_gpu_tensor_new_bf16(nar->gpu, cap * YUE_LATENT_DIM);
    nar->llm_out = h3_gpu_tensor_new_bf16(nar->gpu, cap * YUE_LATENT_DIM);
    if (!nar->x || !nar->normed || !nar->q || !nar->k || !nar->v ||
        !nar->attn || !nar->branch || !nar->fc1out || !nar->mlp_mid ||
        !nar->time_emb || !nar->boundary || !nar->llm_out) {
        /* Some slots are NULL now; ws_rows=0 forces the next call to
         * rebuild the whole set instead of fast-pathing on the stale
         * size and feeding NULL tensors to kernels (nar is engine-
         * resident, so the poisoning would otherwise persist across
         * requests until a larger one rebuilds by chance). */
        nar->ws_rows = 0;
        set_error(error, error_size, "nar workspace allocation failed");
        return 0;
    }
    nar->ws_rows = cap;
    return 1;
}

/* ------------------------------------------------------------------- chunk */

yue_nar_chunk *yue_nar_chunk_create(yue_nar *nar, const uint32_t *ar_tokens,
                                    size_t token_count, size_t noise_frames,
                                    size_t visible, char *error,
                                    size_t error_size) {
    if (!nar || !ar_tokens || !token_count || !noise_frames) {
        set_error(error, error_size, "invalid chunk arguments");
        return NULL;
    }
    if (token_count > YUE_CONTEXT ||
        token_count + noise_frames + 2 > YUE_CONTEXT) {
        set_error(error, error_size, "acoustic chunk exceeds the context");
        return NULL;
    }
    size_t ar_len = token_count;
    if (visible == 0 || visible > ar_len) visible = ar_len;
    size_t nar_len = noise_frames + 2;
    if (nar_len > nar->pos_rows) {
        set_error(error, error_size,
                  "chunk frames exceed the latent position table");
        return NULL;
    }

    yue_nar_chunk *chunk = calloc(1, sizeof(*chunk));
    if (!chunk) {
        set_error(error, error_size, "out of memory");
        return NULL;
    }
    chunk->nar = nar;
    chunk->ar_len = ar_len;
    chunk->visible = visible;
    chunk->frames = noise_frames;
    chunk->nar_len = nar_len;

    /* Capacity bucketed (see yue.h): ar_len drifts a few codec tokens per
     * request; the exact row counts below drive all write/read logic. */
    chunk->cond = yue_ar_cache_create(nar->ar, yue_align_rows(ar_len), error,
                                      error_size);
    if (!chunk->cond) {
        yue_nar_chunk_free(chunk);
        return NULL;
    }
    /* Conditioning prefill = the AR backbone over the chunk tokens (causal,
     * absolute positions 0..ar_len-1); its K/V becomes the static prefix. */
    h3_gpu_tensor *hidden = yue_ar_forward(nar->ar, ar_tokens, ar_len,
                                           chunk->cond, error, error_size);
    if (!hidden) {
        yue_nar_chunk_free(chunk);
        return NULL;
    }

    chunk->ck = calloc((size_t)nar->layers, sizeof(h3_gpu_tensor *));
    chunk->cv = calloc((size_t)nar->layers, sizeof(h3_gpu_tensor *));
    if (!chunk->ck || !chunk->cv) {
        yue_nar_chunk_free(chunk);
        set_error(error, error_size, "out of memory");
        return NULL;
    }
    size_t kv_rows = visible + nar_len;
    /* Allocation bucketed; kv_rows (exact) still drives the SDPA and the
     * acoustic-tail copy offsets. */
    size_t kv_elems = yue_align_rows(kv_rows) * YUE_NAR_KV_HEADS * YUE_HEAD_DIM;
    size_t cond_elems = visible * YUE_NAR_KV_HEADS * YUE_HEAD_DIM;
    /* One chain for all layer copies: the per-layer begin/copy/submit
     * cadence cost a wait round-trip per layer at chunk setup. */
    int copied = h3_gpu_begin(nar->gpu);
    for (int i = 0; copied && i < nar->layers; i++) {
        chunk->ck[i] = h3_gpu_tensor_new_bf16(nar->gpu, kv_elems);
        chunk->cv[i] = h3_gpu_tensor_new_bf16(nar->gpu, kv_elems);
        if (!chunk->ck[i] || !chunk->cv[i]) {
            h3_gpu_submit(nar->gpu); /* drain the partial chain */
            yue_nar_chunk_free(chunk);
            set_error(error, error_size, "combined kv allocation failed");
            return NULL;
        }
        copied = h3_gpu_copy_bf16(nar->gpu, chunk->ck[i], 0,
                                  yue_ar_cache_key(chunk->cond, i), 0,
                                  cond_elems) &&
                 h3_gpu_copy_bf16(nar->gpu, chunk->cv[i], 0,
                                  yue_ar_cache_value(chunk->cond, i), 0,
                                  cond_elems);
    }
    copied = copied && h3_gpu_submit(nar->gpu);
    if (!copied) {
        /* A failed copy short-circuits the submit above; drain the open
         * chain or every later h3_gpu_begin on this context fails. */
        (void)h3_gpu_submit(nar->gpu);
        yue_nar_chunk_free(chunk);
        set_error(error, error_size, "conditioning kv copy failed: %s",
                  h3_gpu_error(nar->gpu));
        return NULL;
    }

    int ok = yue_ar_rope_slice(nar->ar, ar_len, nar_len, &chunk->rope_cos,
                               &chunk->rope_sin, error, error_size);
    if (ok) {
        /* pos rows 0..nar_len-1 are a contiguous slice of the table. */
        chunk->pos_slice = h3_gpu_tensor_new_bf16(
            nar->gpu, yue_align_rows(nar_len) * YUE_HIDDEN);
        ok = chunk->pos_slice && h3_gpu_begin(nar->gpu) &&
             h3_gpu_copy_bf16(nar->gpu, chunk->pos_slice, 0, nar->pos_embed, 0,
                              nar_len * YUE_HIDDEN) &&
             h3_gpu_submit(nar->gpu);
    }
    if (ok) {
        size_t padded = (nar_len + 63u) & ~(size_t)63u;
        uint32_t *host = malloc(sizeof(uint32_t) * padded);
        ok = host != NULL;
        if (ok) {
            for (size_t i = 0; i < padded; i++) host[i] = (uint32_t)kv_rows;
            chunk->limit = h3_gpu_tensor_from_u32(nar->gpu, host, padded);
            free(host);
            ok = chunk->limit != NULL;
        }
    }
    if (!ok) {
        /* Covers a failed pos-slice chain (buffer possibly still open);
         * the drain is a no-op once the chain is closed. */
        (void)h3_gpu_submit(nar->gpu);
        yue_nar_chunk_free(chunk);
        set_error(error, error_size, "chunk setup failed: %s",
                  h3_gpu_error(nar->gpu));
        return NULL;
    }
    return chunk;
}

yue_ar_cache *yue_nar_chunk_cond(yue_nar_chunk *chunk) {
    return chunk ? chunk->cond : NULL;
}

void yue_nar_chunk_free(yue_nar_chunk *chunk) {
    if (!chunk) return;
    yue_ar_cache_free(chunk->cond);
    /* ck/cv are allocated independently: either calloc may fail on its own,
     * so each array must be released independently (a NULL cv[i] access
     * would crash, and a NULL ck would leak cv entirely). */
    if (chunk->ck) {
        for (int i = 0; i < YUE_AR_LAYERS; i++)
            h3_gpu_tensor_free(chunk->ck[i]);
        free(chunk->ck);
    }
    if (chunk->cv) {
        for (int i = 0; i < YUE_AR_LAYERS; i++)
            h3_gpu_tensor_free(chunk->cv[i]);
        free(chunk->cv);
    }
    h3_gpu_tensor_free(chunk->rope_cos);
    h3_gpu_tensor_free(chunk->rope_sin);
    h3_gpu_tensor_free(chunk->pos_slice);
    h3_gpu_tensor_free(chunk->limit);
    free(chunk);
}

/* ---------------------------------------------------------------- velocity */

/* shift_timestep: bf16 quantize the raw time, then fp32 sigmoid + wrapped
 * bf16-rounded coefficients (lyra AcousticModel.shift_timestep). */
static double shift_timestep(const yue_nar *nar, double raw_t) {
    float raw = bf16_round((float)raw_t);
    float t_sig = bf16_round(1.0f / (1.0f + expf(-raw)));
    float shift = nar->timestep_shift;
    float numerator = bf16_round(shift * t_sig);
    float delta = bf16_round((shift - 1.0f) * t_sig);
    return (double)bf16_round(numerator / (1.0f + delta));
}

static int run_linear_nar(yue_nar *nar, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t rows,
                          uint32_t in_dim, uint32_t out_dim, char *error,
                          size_t error_size, const char *what) {
    if (h3_gpu_linear_bf16(nar->gpu, output, input, weight, bias, rows,
                           in_dim, out_dim))
        return 1;
    set_error(error, error_size, "%s linear failed: %s", what,
              h3_gpu_error(nar->gpu));
    return 0;
}

int yue_nar_velocity(yue_nar_chunk *chunk, const uint16_t *state_bits,
                     double raw_t, uint16_t *out_bits, char *error,
                     size_t error_size) {
    yue_nar *nar = chunk ? chunk->nar : NULL;
    if (!nar || !state_bits || !out_bits) {
        set_error(error, error_size, "invalid velocity arguments");
        return 0;
    }
    const size_t frames = chunk->frames;
    const size_t nar_len = chunk->nar_len;
    if (!workspace_reserve(nar, nar_len, error, error_size)) return 0;
    uint32_t rows32 = (uint32_t)nar_len;

    /* Fused q/k head-norm + rope prologue (default on;
     * YUE_HN_ROPE_SEPARATE=1 keeps the four-kernel path) — the same
     * bit-identical single dispatch the AR forward uses (M11 probe T:
     * 1.39 ms/branch-step on the decode chain). */
    const char *hnr_env = getenv("YUE_HN_ROPE_SEPARATE");
    int hn_rope_separate = hnr_env && hnr_env[0] != '0';

    /* boundary-padded state: zero rows around the frame state */
    uint16_t *boundary = malloc(sizeof(uint16_t) * nar_len * YUE_LATENT_DIM);
    if (!boundary) {
        set_error(error, error_size, "out of memory");
        return 0;
    }
    memset(boundary, 0, sizeof(uint16_t) * YUE_LATENT_DIM);
    memcpy(boundary + YUE_LATENT_DIM, state_bits,
           sizeof(uint16_t) * frames * YUE_LATENT_DIM);
    memset(boundary + (frames + 1) * YUE_LATENT_DIM, 0,
           sizeof(uint16_t) * YUE_LATENT_DIM);
    int ok = h3_gpu_tensor_write_bf16_range(nar->boundary, 0, boundary,
                                            nar_len * YUE_LATENT_DIM);
    free(boundary);
    if (!ok) {
        set_error(error, error_size, "state upload failed");
        return 0;
    }

    /* Timestep embedding: fp32 sinusoid row -> bf16, then the checkpoint's
     * two-layer MLP (identical across positions; broadcast on the host). */
    const double shifted = shift_timestep(nar, raw_t);
    const float t_f = (float)shifted;
    uint16_t embed[YUE_TIME_EMBED];
    for (int i = 0; i < YUE_TIME_HALF; i++) {
        float arg = t_f * nar->time_freq[i];
        embed[i] = f32_to_bf16(cosf(arg));
        embed[YUE_TIME_HALF + i] = f32_to_bf16(sinf(arg));
    }
    ok = h3_gpu_tensor_write_bf16_range(nar->time_in_t, 0, embed,
                                        YUE_TIME_EMBED) &&
         h3_gpu_begin(nar->gpu) &&
         run_linear_nar(nar, nar->time_mid, nar->time_in_t, nar->time1_w,
                        nar->time1_b, 1, YUE_TIME_EMBED, YUE_HIDDEN, error,
                        error_size, "time fc1") &&
         h3_gpu_submit(nar->gpu);
    if (!ok) {
        (void)h3_gpu_submit(nar->gpu); /* drain the possibly-open chain */
        set_error(error, error_size, "time fc1 failed: %s",
                  h3_gpu_error(nar->gpu));
        return 0;
    }
    uint16_t stage[YUE_HIDDEN];
    if (!h3_gpu_tensor_read_bf16(nar->time_mid, stage, YUE_HIDDEN)) {
        set_error(error, error_size, "time fc1 readback failed");
        return 0;
    }
    for (int i = 0; i < YUE_HIDDEN; i++) {
        float value = bf16_to_f32(stage[i]);
        stage[i] = f32_to_bf16(value * (1.0f / (1.0f + expf(-value))));
    }
    ok = h3_gpu_tensor_write_bf16_range(nar->time_mid, 0, stage,
                                        YUE_HIDDEN) &&
         h3_gpu_begin(nar->gpu) &&
         run_linear_nar(nar, nar->time_vec, nar->time_mid, nar->time2_w,
                        nar->time2_b, 1, YUE_HIDDEN, YUE_HIDDEN, error,
                        error_size, "time fc2") &&
         h3_gpu_submit(nar->gpu);
    if (!ok) {
        (void)h3_gpu_submit(nar->gpu); /* drain the possibly-open chain */
        set_error(error, error_size, "time fc2 failed: %s",
                  h3_gpu_error(nar->gpu));
        return 0;
    }
    uint16_t *time_row = malloc(sizeof(uint16_t) * YUE_HIDDEN);
    uint16_t *time_all = malloc(sizeof(uint16_t) * nar_len * YUE_HIDDEN);
    ok = time_row && time_all &&
         h3_gpu_tensor_read_bf16(nar->time_vec, time_row, YUE_HIDDEN);
    if (ok) {
        for (size_t r = 0; r < nar_len; r++)
            memcpy(time_all + r * YUE_HIDDEN, time_row,
                   sizeof(uint16_t) * YUE_HIDDEN);
        ok = h3_gpu_tensor_write_bf16_range(nar->time_emb, 0, time_all,
                                            nar_len * YUE_HIDDEN);
    }
    free(time_row);
    free(time_all);
    if (!ok) {
        set_error(error, error_size, "time embedding broadcast failed");
        return 0;
    }

    /* x = vae2llm(boundary) + time_emb + pos_emb (bf16 adds, in order) */
    ok = h3_gpu_begin(nar->gpu) &&
         run_linear_nar(nar, nar->x, nar->boundary, nar->vae2llm_w,
                        nar->vae2llm_b, rows32, YUE_LATENT_DIM, YUE_HIDDEN,
                        error, error_size, "vae2llm") &&
         h3_gpu_add_bf16(nar->gpu, nar->x, nar->x, nar->time_emb,
                         (uint32_t)(nar_len * YUE_HIDDEN)) &&
         h3_gpu_add_bf16(nar->gpu, nar->x, nar->x, chunk->pos_slice,
                         (uint32_t)(nar_len * YUE_HIDDEN)) &&
         h3_gpu_continue(nar->gpu);
    if (!ok) {
        (void)h3_gpu_submit(nar->gpu); /* drain the possibly-open chain */
        set_error(error, error_size, "acoustic embed failed: %s",
                  h3_gpu_error(nar->gpu));
        return 0;
    }

    const size_t kv_rows = chunk->visible + nar_len;
    const size_t acoustic_kv = nar_len * YUE_NAR_KV_HEADS * YUE_HEAD_DIM;
    for (int i = 0; ok && i < nar->layers; i++) {
        const yue_nar_layer *layer = &nar->layer[i];
        ok = qw_gpu_rms_norm_bf16(nar->gpu, nar->normed, nar->x,
                                  layer->input_norm, rows32, YUE_HIDDEN,
                                  YUE_RMS_EPS) &&
             run_linear_nar(nar, nar->q, nar->normed, layer->q, NULL, rows32,
                            YUE_HIDDEN, YUE_HEADS * YUE_HEAD_DIM, error,
                            error_size, "nar q") &&
             run_linear_nar(nar, nar->k, nar->normed, layer->k, NULL, rows32,
                            YUE_HIDDEN, YUE_NAR_KV_HEADS * YUE_HEAD_DIM,
                            error, error_size, "nar k") &&
             run_linear_nar(nar, nar->v, nar->normed, layer->v, NULL, rows32,
                            YUE_HIDDEN, YUE_NAR_KV_HEADS * YUE_HEAD_DIM,
                            error, error_size, "nar v") &&
             /* Fused head-norm + rope (bit-identical, one dispatch for
              * q+k); wrapper gate failure or YUE_HN_ROPE_SEPARATE=1
              * falls back to the four-kernel prologue. Same-buffer
              * kernels run in order: one submit per layer tail (the
              * per-op round-trips dominated the flow stage). */
             ((hn_rope_separate
                   ? 0
                   : qw_gpu_head_rms_rope_bf16(
                         nar->gpu, nar->q, nar->k, layer->q_norm,
                         layer->k_norm, chunk->rope_cos, chunk->rope_sin,
                         rows32, YUE_HEADS, YUE_NAR_KV_HEADS, YUE_HEAD_DIM,
                         YUE_RMS_EPS)) ||
              (qw_gpu_head_rms_norm_bf16(nar->gpu, nar->q, layer->q_norm,
                                         rows32, YUE_HEADS, YUE_HEAD_DIM,
                                         YUE_RMS_EPS) &&
               qw_gpu_head_rms_norm_bf16(nar->gpu, nar->k, layer->k_norm,
                                         rows32, YUE_NAR_KV_HEADS,
                                         YUE_HEAD_DIM, YUE_RMS_EPS) &&
               qw_gpu_qk_rope_half_bf16(nar->gpu, nar->q, nar->q, nar->q,
                                        nar->q, chunk->rope_cos,
                                        chunk->rope_sin, rows32, YUE_HEADS,
                                        YUE_HEAD_DIM) &&
               qw_gpu_qk_rope_half_bf16(nar->gpu, nar->k, nar->k, nar->k,
                                        nar->k, chunk->rope_cos,
                                        chunk->rope_sin, rows32,
                                        YUE_NAR_KV_HEADS, YUE_HEAD_DIM))) &&
             /* refresh the acoustic tail of the combined K/V */
             h3_gpu_copy_bf16(nar->gpu, chunk->ck[i],
                              chunk->visible * YUE_NAR_KV_HEADS *
                                  YUE_HEAD_DIM,
                              nar->k, 0, acoustic_kv) &&
             h3_gpu_copy_bf16(nar->gpu, chunk->cv[i],
                              chunk->visible * YUE_NAR_KV_HEADS *
                                  YUE_HEAD_DIM,
                              nar->v, 0, acoustic_kv) &&
             qw_gpu_flash_sdpa_prefix_bf16(
                 nar->gpu, nar->attn, nar->q, chunk->ck[i], chunk->cv[i],
                 chunk->limit, rows32, (uint32_t)kv_rows, YUE_HEADS,
                 YUE_NAR_KV_HEADS, YUE_HEAD_DIM,
                 1.0f / sqrtf((float)YUE_HEAD_DIM));
        if (!ok) {
            set_error(error, error_size, "nar layer %d attention: %s", i,
                      h3_gpu_error(nar->gpu));
            break;
        }
        ok = run_linear_nar(nar, nar->branch, nar->attn, layer->o, NULL,
                            rows32, YUE_HEADS * YUE_HEAD_DIM, YUE_HIDDEN,
                            error, error_size, "nar o") &&
             h3_gpu_add_bf16(nar->gpu, nar->x, nar->x, nar->branch,
                             (uint32_t)(nar_len * YUE_HIDDEN)) &&
             qw_gpu_rms_norm_bf16(nar->gpu, nar->normed, nar->x,
                                  layer->post_norm, rows32, YUE_HIDDEN,
                                  YUE_RMS_EPS) &&
             run_linear_nar(nar, nar->fc1out, nar->normed, layer->fc1, NULL,
                            rows32, YUE_HIDDEN, 2 * YUE_MLP, error,
                            error_size, "nar fc1") &&
             qw_gpu_swiglu_bf16(nar->gpu, nar->mlp_mid, nar->fc1out,
                                (uint32_t)(nar_len * YUE_MLP), YUE_MLP) &&
             run_linear_nar(nar, nar->branch, nar->mlp_mid, layer->down, NULL,
                            rows32, YUE_MLP, YUE_HIDDEN, error, error_size,
                            "nar down") &&
             h3_gpu_add_bf16(nar->gpu, nar->x, nar->x, nar->branch,
                             (uint32_t)(nar_len * YUE_HIDDEN)) &&
             /* commit without waiting (pipelined layers, one wait at the
              * velocity tail) */
             h3_gpu_continue(nar->gpu);
        if (!ok) {
            /* Copy first: error is both the destination and an argument
             * (vsnprintf overlap is UB), and the inner op may not have
             * written it, in which case only the GPU error is relevant. */
            char prior[256] = "";
            if (error && error_size)
                snprintf(prior, sizeof(prior), "%.*s",
                         (int)sizeof(prior) - 1, error);
            if (prior[0])
                set_error(error, error_size, "nar layer %d mlp: %s (%s)", i,
                          prior, h3_gpu_error(nar->gpu));
            else
                set_error(error, error_size, "nar layer %d mlp: %s",
                          h3_gpu_error(nar->gpu));
            break;
        }
    }
    if (!ok) {
        (void)h3_gpu_submit(nar->gpu); /* drain the possibly-open chain */
        return 0;
    }

    /* shared final norm, then the llm2vae head; drop the boundary rows */
    ok = qw_gpu_rms_norm_bf16(nar->gpu, nar->normed, nar->x,
                              yue_ar_norm_weight(nar->ar), rows32, YUE_HIDDEN,
                              YUE_RMS_EPS) &&
         run_linear_nar(nar, nar->llm_out, nar->normed, nar->llm2vae_w,
                        nar->llm2vae_b, rows32, YUE_HIDDEN, YUE_LATENT_DIM,
                        error, error_size, "llm2vae") &&
         h3_gpu_submit(nar->gpu);
    if (!ok) {
        (void)h3_gpu_submit(nar->gpu); /* drain the possibly-open chain */
        set_error(error, error_size, "velocity head failed: %s",
                  h3_gpu_error(nar->gpu));
        return 0;
    }
    uint16_t *rows = malloc(sizeof(uint16_t) * nar_len * YUE_LATENT_DIM);
    if (!rows ||
        !h3_gpu_tensor_read_bf16(nar->llm_out, rows,
                                 nar_len * YUE_LATENT_DIM)) {
        free(rows);
        set_error(error, error_size, "velocity readback failed");
        return 0;
    }
    memcpy(out_bits, rows + YUE_LATENT_DIM,
           sizeof(uint16_t) * frames * YUE_LATENT_DIM);
    free(rows);
    return 1;
}

/* ------------------------------------------------------------------- solve */

/* logit flow time, clamped to +-20 (lyra _logit). */
static double logit_t(double t) {
    if (t <= 0.0) return -20.0;
    if (t >= 1.0) return 20.0;
    double value = log(t / (1.0 - t));
    if (value < -20.0) value = -20.0;
    if (value > 20.0) value = 20.0;
    return value;
}

int yue_nar_solve(yue_nar_chunk *chunk, const float *initial_noise, int steps,
                  float *out, char *error, size_t error_size) {
    yue_nar *nar = chunk ? chunk->nar : NULL;
    if (!nar || !initial_noise || steps < 1 || !out) {
        set_error(error, error_size, "invalid solve arguments");
        return 0;
    }
    const size_t n = chunk->frames * YUE_LATENT_DIM;
    uint16_t *state = malloc(sizeof(uint16_t) * n);
    uint16_t *mid = malloc(sizeof(uint16_t) * n);
    uint16_t *velocity = malloc(sizeof(uint16_t) * n);
    if (!state || !mid || !velocity) {
        free(state);
        free(mid);
        free(velocity);
        set_error(error, error_size, "out of memory");
        return 0;
    }
    for (size_t i = 0; i < n; i++)
        state[i] = f32_to_bf16(initial_noise[i]);

    const double dt = 1.0 / (double)steps;
    for (int step = 0; step < steps; step++) {
        const double t = 1.0 - (double)step * dt;
        if (!yue_nar_velocity(chunk, state, logit_t(t), velocity, error,
                              error_size)) {
            free(state);
            free(mid);
            free(velocity);
            return 0;
        }
        /* midpoint = state - v1 * dt/2 (separate buffer: the full step
         * updates from the ORIGINAL state, not the midpoint) */
        const float half_dt = (float)(dt / 2.0);
        for (size_t i = 0; i < n; i++) {
            float product = bf16_round(bf16_to_f32(velocity[i]) * half_dt);
            mid[i] = f32_to_bf16(bf16_to_f32(state[i]) - product);
        }

        const double mid_t = t - dt / 2.0;
        if (!yue_nar_velocity(chunk, mid, logit_t(mid_t), velocity, error,
                              error_size)) {
            free(state);
            free(mid);
            free(velocity);
            return 0;
        }
        const float full_dt = (float)dt;
        for (size_t i = 0; i < n; i++) {
            float product = bf16_round(bf16_to_f32(velocity[i]) * full_dt);
            state[i] = f32_to_bf16(bf16_to_f32(state[i]) - product);
        }
    }
    for (size_t i = 0; i < n; i++) out[i] = bf16_to_f32(state[i]);
    free(state);
    free(mid);
    free(velocity);
    return 1;
}
