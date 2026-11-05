/* YuE2 AR backbone (Qwen3 3B) on Metal. Mirrors lyra ar.py arithmetic:
 * SourceRMSNorm (fp32 variance, bf16 normalization/weight), fp32-SiLU SwiGLU,
 * half-split RoPE with fp32 angles, bf16 causal GQA, and phase-local output
 * projections ([0,EOD)+ABC_END for abc, [MUSIC_END,CODEC) for semantic). */
#import <Foundation/Foundation.h>

#include "yue_ar.h"
#include "yue_gpu_ext.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* Weight file resolution: prefer the lyra split layout (ar-bf16 /
 * nar-bf16.safetensors); fall back to the official single-file checkpoint
 * (m-a-p/YuE2-3B model.safetensors), which carries both partitions under the
 * same tensor key names (byte-identical to the lyra split, all 628 tensors
 * verified). If neither file exists the caller's header read reports the
 * model.safetensors path. */
static int resolve_weights_path(char *out, size_t size, const char *model_dir,
                                const char *split_name) {
    int written = snprintf(out, size, "%s/%s", model_dir, split_name);
    if (written < 0 || (size_t)written >= size) return 0;
    struct stat st;
    if (stat(out, &st) == 0) return 1;
    written = snprintf(out, size, "%s/model.safetensors", model_dir);
    return written >= 0 && (size_t)written < size;
}

#define YUE_AR_KV_HEADS YUE_KV_HEADS
#define YUE_AR_PAIRS (YUE_HEAD_DIM / 2)
/* Split-K decode SDPA: 256-row slabs, enough for the full context. */
#define YUE_AR_MAX_SPLITS (((uint32_t)YUE_CONTEXT + 255u) / 256u)

/* RNE bf16 rounding for host-side derived tables. */
static float bf16_round(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* One AR projection: the bf16 [out, in] weight (no nn bias in YuE2). */
typedef struct {
    h3_gpu_tensor *w;
} yue_ar_linear;

typedef struct {
    h3_gpu_tensor *input_norm;
    yue_ar_linear q_proj, k_proj, v_proj, o_proj;
    h3_gpu_tensor *q_norm, *k_norm;
    h3_gpu_tensor *post_norm;
    yue_ar_linear fc1; /* [2*mlp, hidden] gate|up concat */
    yue_ar_linear down;
} yue_ar_layer;

struct yue_ar {
    h3_gpu *gpu;
    int layers;
    char weights_path[4096];
    h3_gpu_tensor *embed;
    h3_gpu_tensor *norm;
    yue_ar_linear lm_abc_content; /* lm_head rows [0, EOD) */
    yue_ar_linear lm_abc_end;     /* lm_head row ABC_END */
    yue_ar_linear lm_semantic;    /* lm_head rows [MUSIC_END, 184621) */
    yue_ar_layer *layer;

    /* fp32 RoPE tables over absolute positions 0..context-1 (host copies
     * retained for per-step offset uploads). */
    float *rope_cos_host, *rope_sin_host;
    h3_gpu_tensor *rope_cos, *rope_sin;

    /* Persistent workspace, reallocated when row count grows. */
    size_t ws_rows;
    h3_gpu_tensor *x, *normed, *q, *k, *v, *attn, *branch, *fc1out, *mlp_mid;
    h3_gpu_tensor *causal_limit; /* u32 per query row (padded) */
    /* Split-K decode SDPA partials: [head][split][head_dim+32] fp32
     * (acc[128], m, l, 30 pad — stride 640 B = 5 cache lines, see the
     * merge kernel comment for the false-sharing bug this avoids).
     * Fixed size; decode steps only. */
    h3_gpu_tensor *sdpa_partial;
    uint32_t q_pad_rows;
};

struct yue_ar_cache {
    size_t capacity;
    size_t length;
    int layers;
    h3_gpu_tensor **keys;   /* [cap, kv_heads, head_dim] bf16 per layer */
    h3_gpu_tensor **values;
};

static int set_error(char *error, size_t error_size, const char *fmt, ...) {
    if (error && error_size) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(error, error_size, fmt, args);
        va_end(args);
    }
    return 0;
}

/* ------------------------------------------------------------------ create */

/* Load one BF16 tensor from the AR checkpoint, validating dtype/shape. */
static h3_gpu_tensor *load_weight(h3_gpu *gpu, const h3_st_header *header,
                                  const char *name, int ndim,
                                  const uint64_t *shape, char *error,
                                  size_t error_size) {
    const h3_st_tensor *tensor = h3_st_find(header, name);
    if (!tensor) {
        set_error(error, error_size, "missing weight %s", name);
        return NULL;
    }
    if (tensor->dtype != H3_DTYPE_BF16) {
        set_error(error, error_size, "weight %s has dtype %s, expected BF16",
                  name, h3_dtype_name(tensor->dtype));
        return NULL;
    }
    if (tensor->ndim != ndim) {
        set_error(error, error_size, "weight %s has rank %d, expected %d",
                  name, tensor->ndim, ndim);
        return NULL;
    }
    for (int d = 0; d < ndim; d++) {
        if (tensor->shape[d] != shape[d]) {
            set_error(error, error_size, "weight %s shape mismatch", name);
            return NULL;
        }
    }
    size_t elements = (size_t)h3_st_tensor_elements(tensor);
    void *host = malloc(elements * 2);
    if (!host) {
        set_error(error, error_size, "out of memory loading %s", name);
        return NULL;
    }
    int ok = h3_st_read_data(header, tensor, host, elements * 2, error,
                             error_size);
    h3_gpu_tensor *out = ok ? h3_gpu_tensor_from_bf16(gpu, host, elements)
                            : NULL;
    free(host);
    return out;
}

/* Load one AR linear (weights only; no nn bias in YuE2). */
static int load_linear(h3_gpu *gpu, const h3_st_header *header,
                       const char *name, uint64_t out, uint64_t in,
                       yue_ar_linear *lin, char *error,
                       size_t error_size) {
    char full[224];
    snprintf(full, sizeof(full), "%s.weight", name);
    lin->w = load_weight(gpu, header, full, 2,
                         (const uint64_t[]){out, in}, error, error_size);
    return lin->w != NULL;
}

yue_ar *yue_ar_create(h3_gpu *gpu, const char *model_dir, char *error,
                      size_t error_size) {
    if (!gpu || !model_dir) {
        set_error(error, error_size, "invalid arguments");
        return NULL;
    }
    yue_ar *ar = calloc(1, sizeof(yue_ar));
    if (!ar) {
        set_error(error, error_size, "out of memory");
        return NULL;
    }
    ar->gpu = gpu;
    ar->layers = YUE_AR_LAYERS;
    if (!resolve_weights_path(ar->weights_path, sizeof(ar->weights_path),
                              model_dir, "ar-bf16.safetensors")) {
        set_error(error, error_size, "model directory path is too long");
        free(ar);
        return NULL;
    }

    const uint64_t scalar[1] = {YUE_HIDDEN};
    const uint64_t scalar_dim[1] = {YUE_HEAD_DIM};
    const uint64_t mlp_hidden[2] = {YUE_MLP, YUE_HIDDEN};

    h3_st_header header;
    if (!h3_st_read_header(ar->weights_path, &header, error, error_size)) {
        yue_ar_free(ar);
        return NULL;
    }

    ar->embed = load_weight(gpu, &header, "model.embed_tokens.weight", 2,
                            (const uint64_t[]){YUE_VOCAB_SIZE, YUE_HIDDEN},
                            error, error_size);
    ar->norm = load_weight(gpu, &header, "model.norm.weight", 1, scalar,
                           error, error_size);
    if (!ar->embed || !ar->norm) {
        h3_st_free_header(&header);
        yue_ar_free(ar);
        return NULL;
    }

    ar->layer = calloc((size_t)ar->layers, sizeof(yue_ar_layer));
    if (!ar->layer) {
        h3_st_free_header(&header);
        yue_ar_free(ar);
        set_error(error, error_size, "out of memory");
        return NULL;
    }
    for (int i = 0; i < ar->layers; i++) {
        char name[192];
        yue_ar_layer *layer = &ar->layer[i];
#define YUE_LOAD_NORM(field, suffix, shape)                                    \
        do {                                                                   \
            snprintf(name, sizeof(name), "model.layers.%d." suffix, i);        \
            layer->field = load_weight(gpu, &header, name, 1, shape,           \
                                       error, error_size);                     \
            if (!layer->field) {                                               \
                h3_st_free_header(&header);                                    \
                yue_ar_free(ar);                                               \
                return NULL;                                                   \
            }                                                                  \
        } while (0)
        YUE_LOAD_NORM(input_norm, "input_layernorm.weight", scalar);
        YUE_LOAD_NORM(q_norm, "self_attn.q_norm.weight", scalar_dim);
        YUE_LOAD_NORM(k_norm, "self_attn.k_norm.weight", scalar_dim);
        YUE_LOAD_NORM(post_norm, "post_attention_layernorm.weight", scalar);
#undef YUE_LOAD_NORM
#define YUE_LOAD_LINEAR(field, suffix, out, in)                                \
        do {                                                                   \
            snprintf(name, sizeof(name), "model.layers.%d." suffix, i);        \
            if (!load_linear(gpu, &header, name, out, in,                      \
                             &layer->field, error, error_size)) {              \
                h3_st_free_header(&header);                                    \
                yue_ar_free(ar);                                               \
                return NULL;                                                   \
            }                                                                  \
        } while (0)
        YUE_LOAD_LINEAR(q_proj, "self_attn.q_proj", YUE_HIDDEN, YUE_HIDDEN);
        YUE_LOAD_LINEAR(k_proj, "self_attn.k_proj",
                        YUE_AR_KV_HEADS * YUE_HEAD_DIM, YUE_HIDDEN);
        YUE_LOAD_LINEAR(v_proj, "self_attn.v_proj",
                        YUE_AR_KV_HEADS * YUE_HEAD_DIM, YUE_HIDDEN);
        YUE_LOAD_LINEAR(o_proj, "self_attn.o_proj", YUE_HIDDEN, YUE_HIDDEN);
        YUE_LOAD_LINEAR(down, "mlp.down_proj", YUE_HIDDEN, YUE_MLP);
#undef YUE_LOAD_LINEAR

        /* gate|up concat: qw_swiglu_bf16 consumes [rows, 2*mlp] rows split
         * into left gate / right up blocks. */
        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", i);
        h3_gpu_tensor *gate = load_weight(gpu, &header, name, 2, mlp_hidden,
                                          error, error_size);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.up_proj.weight", i);
        h3_gpu_tensor *up = load_weight(gpu, &header, name, 2, mlp_hidden,
                                        error, error_size);
        if (!gate || !up) {
            h3_gpu_tensor_free(gate);
            h3_gpu_tensor_free(up);
            h3_st_free_header(&header);
            yue_ar_free(ar);
            return NULL;
        }
        int ok = 1;
        size_t half = (size_t)YUE_MLP * YUE_HIDDEN;
        layer->fc1.w = h3_gpu_tensor_new_bf16(
            gpu, (size_t)2 * YUE_MLP * YUE_HIDDEN);
        uint16_t *host = malloc(sizeof(uint16_t) * half);
        ok = layer->fc1.w && host &&
             h3_gpu_tensor_read_bf16(gate, host, half) &&
             h3_gpu_tensor_write_bf16_range(layer->fc1.w, 0, host, half) &&
             h3_gpu_tensor_read_bf16(up, host, half) &&
             h3_gpu_tensor_write_bf16_range(layer->fc1.w, half, host,
                                            half);
        free(host);
        h3_gpu_tensor_free(gate);
        h3_gpu_tensor_free(up);
        if (!ok) {
            h3_st_free_header(&header);
            yue_ar_free(ar);
            set_error(error, error_size, "fc1 concat failed at layer %d", i);
            return NULL;
        }
    }
    /* Phase-local lm_head slices, mapped straight from the checkpoint
     * (zero-copy). */
    const h3_st_tensor *lm = h3_st_find(&header, "lm_head.weight");
    if (!lm || lm->dtype != H3_DTYPE_BF16 || lm->ndim != 2 ||
        lm->shape[1] != YUE_HIDDEN || lm->shape[0] < YUE_LATENT_START) {
        h3_st_free_header(&header);
        yue_ar_free(ar);
        set_error(error, error_size,
                  "lm_head.weight missing or has an invalid shape");
        return NULL;
    }
    uint64_t row_bytes = (uint64_t)YUE_HIDDEN * sizeof(uint16_t);
    ar->lm_abc_content.w = h3_gpu_tensor_load_bf16(
        gpu, ar->weights_path, lm->file_offset,
        (size_t)YUE_EOD * YUE_HIDDEN);
    ar->lm_abc_end.w = h3_gpu_tensor_load_bf16(
        gpu, ar->weights_path, lm->file_offset + (uint64_t)YUE_ABC_END * row_bytes,
        YUE_HIDDEN);
    ar->lm_semantic.w = h3_gpu_tensor_load_bf16(
        gpu, ar->weights_path,
        lm->file_offset + (uint64_t)YUE_MUSIC_END * row_bytes,
        (size_t)(YUE_CODEC_OFFSET + YUE_CODEC_SIZE - YUE_MUSIC_END) * YUE_HIDDEN);
    h3_st_free_header(&header);
    if (!ar->lm_abc_content.w || !ar->lm_abc_end.w || !ar->lm_semantic.w) {
        yue_ar_free(ar);
        set_error(error, error_size, "lm_head slice load failed");
        return NULL;
    }

    /* RoPE: fp32 inv_freq (never quantized — lyra keeps it outside the
     * parameter tree), fp32 phases, cos/sin quantized to bf16 once. */
    size_t positions = YUE_CONTEXT;
    ar->rope_cos_host = malloc(sizeof(float) * positions * YUE_AR_PAIRS);
    ar->rope_sin_host = malloc(sizeof(float) * positions * YUE_AR_PAIRS);
    if (!ar->rope_cos_host || !ar->rope_sin_host) {
        yue_ar_free(ar);
        set_error(error, error_size, "out of memory for rope tables");
        return NULL;
    }
    for (size_t pos = 0; pos < positions; pos++) {
        for (int pair = 0; pair < YUE_AR_PAIRS; pair++) {
            float inv_freq = (float)pow(YUE_ROPE_THETA,
                                        -2.0 * pair / YUE_HEAD_DIM);
            double angle = (double)pos * (double)inv_freq;
            ar->rope_cos_host[pos * YUE_AR_PAIRS + pair] =
                bf16_round((float)cos(angle));
            ar->rope_sin_host[pos * YUE_AR_PAIRS + pair] =
                bf16_round((float)sin(angle));
        }
    }
    ar->rope_cos = h3_gpu_tensor_from_f32(gpu, ar->rope_cos_host,
                                          positions * YUE_AR_PAIRS);
    ar->rope_sin = h3_gpu_tensor_from_f32(gpu, ar->rope_sin_host,
                                          positions * YUE_AR_PAIRS);
    if (!ar->rope_cos || !ar->rope_sin) {
        yue_ar_free(ar);
        set_error(error, error_size, "rope table upload failed");
        return NULL;
    }
    return ar;
}

h3_gpu *yue_ar_gpu(yue_ar *ar) { return ar ? ar->gpu : NULL; }

h3_gpu_tensor *yue_ar_norm_weight(yue_ar *ar) {
    return ar ? ar->norm : NULL;
}

int yue_ar_rope_slice(yue_ar *ar, size_t offset, size_t rows,
                      h3_gpu_tensor **cos_out, h3_gpu_tensor **sin_out,
                      char *error, size_t error_size) {
    if (!ar || !cos_out || !sin_out || !rows ||
        offset > YUE_CONTEXT || rows > YUE_CONTEXT - offset) {
        set_error(error, error_size, "invalid rope slice request");
        return 0;
    }
    size_t elements = rows * YUE_AR_PAIRS;
    float *cos_host = malloc(sizeof(float) * elements);
    float *sin_host = malloc(sizeof(float) * elements);
    if (!cos_host || !sin_host) {
        free(cos_host);
        free(sin_host);
        set_error(error, error_size, "out of memory for rope slice");
        return 0;
    }
    memcpy(cos_host, ar->rope_cos_host + offset * YUE_AR_PAIRS,
           sizeof(float) * elements);
    memcpy(sin_host, ar->rope_sin_host + offset * YUE_AR_PAIRS,
           sizeof(float) * elements);
    *cos_out = h3_gpu_tensor_from_f32(ar->gpu, cos_host, elements);
    *sin_out = h3_gpu_tensor_from_f32(ar->gpu, sin_host, elements);
    free(cos_host);
    free(sin_host);
    if (!*cos_out || !*sin_out) {
        /* Half-uploaded slice: release whatever landed so the caller has
         * nothing to clean up on the failure path. */
        h3_gpu_tensor_free(*cos_out);
        h3_gpu_tensor_free(*sin_out);
        *cos_out = NULL;
        *sin_out = NULL;
        set_error(error, error_size, "rope slice upload failed");
        return 0;
    }
    return 1;
}

static void free_linear(yue_ar_linear *lin) {
    h3_gpu_tensor_free(lin->w);
    memset(lin, 0, sizeof(*lin));
}

void yue_ar_free(yue_ar *ar) {
    if (!ar) return;
    h3_gpu_tensor_free(ar->embed);
    h3_gpu_tensor_free(ar->norm);
    free_linear(&ar->lm_abc_content);
    free_linear(&ar->lm_abc_end);
    free_linear(&ar->lm_semantic);
    if (ar->layer) {
        for (int i = 0; i < ar->layers; i++) {
            h3_gpu_tensor_free(ar->layer[i].input_norm);
            free_linear(&ar->layer[i].q_proj);
            free_linear(&ar->layer[i].k_proj);
            free_linear(&ar->layer[i].v_proj);
            free_linear(&ar->layer[i].o_proj);
            h3_gpu_tensor_free(ar->layer[i].q_norm);
            h3_gpu_tensor_free(ar->layer[i].k_norm);
            h3_gpu_tensor_free(ar->layer[i].post_norm);
            free_linear(&ar->layer[i].fc1);
            free_linear(&ar->layer[i].down);
        }
        free(ar->layer);
    }
    h3_gpu_tensor_free(ar->rope_cos);
    h3_gpu_tensor_free(ar->rope_sin);
    free(ar->rope_cos_host);
    free(ar->rope_sin_host);
    h3_gpu_tensor_free(ar->x);
    h3_gpu_tensor_free(ar->normed);
    h3_gpu_tensor_free(ar->q);
    h3_gpu_tensor_free(ar->k);
    h3_gpu_tensor_free(ar->v);
    h3_gpu_tensor_free(ar->attn);
    h3_gpu_tensor_free(ar->branch);
    h3_gpu_tensor_free(ar->fc1out);
    h3_gpu_tensor_free(ar->mlp_mid);
    h3_gpu_tensor_free(ar->causal_limit);
    h3_gpu_tensor_free(ar->sdpa_partial);
    free(ar);
}

/* ------------------------------------------------------------------ cache */

yue_ar_cache *yue_ar_cache_create(yue_ar *ar, size_t capacity, char *error,
                                  size_t error_size) {
    if (!ar || capacity < 1 || capacity > YUE_CONTEXT) {
        set_error(error, error_size, "invalid cache capacity %zu", capacity);
        return NULL;
    }
    yue_ar_cache *cache = calloc(1, sizeof(yue_ar_cache));
    if (!cache) {
        set_error(error, error_size, "out of memory");
        return NULL;
    }
    cache->capacity = capacity;
    cache->layers = ar->layers;
    cache->keys = calloc((size_t)ar->layers, sizeof(h3_gpu_tensor *));
    cache->values = calloc((size_t)ar->layers, sizeof(h3_gpu_tensor *));
    if (!cache->keys || !cache->values) {
        yue_ar_cache_free(cache);
        set_error(error, error_size, "out of memory");
        return NULL;
    }
    if (getenv("QWEN_AR_TRACE")) fprintf(stderr, "[c] cache create cap=%zu\n", capacity);
    size_t per_layer = capacity * YUE_AR_KV_HEADS * YUE_HEAD_DIM;
    for (int i = 0; i < ar->layers; i++) {
        cache->keys[i] = h3_gpu_tensor_new_bf16(ar->gpu, per_layer);
        cache->values[i] = h3_gpu_tensor_new_bf16(ar->gpu, per_layer);
        if (!cache->keys[i] || !cache->values[i]) {
            yue_ar_cache_free(cache);
            set_error(error, error_size, "cache allocation failed");
            return NULL;
        }
    }
    return cache;
}

void yue_ar_cache_free(yue_ar_cache *cache) {
    if (!cache) return;
    if (cache->keys) {
        for (int i = 0; i < cache->layers; i++)
            h3_gpu_tensor_free(cache->keys[i]);
        free(cache->keys);
    }
    if (cache->values) {
        for (int i = 0; i < cache->layers; i++)
            h3_gpu_tensor_free(cache->values[i]);
        free(cache->values);
    }
    free(cache);
}

void yue_ar_cache_reset(yue_ar_cache *cache) {
    if (cache) cache->length = 0;
}

size_t yue_ar_cache_length(const yue_ar_cache *cache) {
    return cache ? cache->length : 0;
}

/* -------------------------------------------------------------- workspace */

static int workspace_reserve(yue_ar *ar, size_t rows, char *error,
                             size_t error_size) {
    if (ar->ws_rows >= rows) return 1;
    size_t cap = ar->ws_rows ? ar->ws_rows : 1024;
    while (cap < rows) cap *= 2;
    ar->q_pad_rows = ((uint32_t)cap + 63u) & ~63u;
    /* Free + NULL every slot before reallocating: a failed rebuild midway
     * must leave only NULLs (free(NULL) is safe on retry), and the limit
     * tensor is owned by causal_limit_reserve — leaving a freed pointer
     * here turned into a double free on the next reserve (the serve path
     * grew the workspace from ~100 decode rows to a 2000-row prefill and
     * crashed inside h3_gpu_tensor_free). */
    h3_gpu_tensor_free(ar->x);
    ar->x = NULL;
    h3_gpu_tensor_free(ar->normed);
    ar->normed = NULL;
    h3_gpu_tensor_free(ar->q);
    ar->q = NULL;
    h3_gpu_tensor_free(ar->k);
    ar->k = NULL;
    h3_gpu_tensor_free(ar->v);
    ar->v = NULL;
    h3_gpu_tensor_free(ar->attn);
    ar->attn = NULL;
    h3_gpu_tensor_free(ar->branch);
    ar->branch = NULL;
    h3_gpu_tensor_free(ar->fc1out);
    ar->fc1out = NULL;
    h3_gpu_tensor_free(ar->mlp_mid);
    ar->mlp_mid = NULL;
    h3_gpu_tensor_free(ar->causal_limit);
    ar->causal_limit = NULL;
    size_t hidden_elems = cap * YUE_HIDDEN;
    size_t partial_elems = (size_t)YUE_HEADS * YUE_AR_MAX_SPLITS *
                           (YUE_HEAD_DIM + 32);
    if (!ar->sdpa_partial)
        ar->sdpa_partial = h3_gpu_tensor_new_f32(ar->gpu, partial_elems);
    if (!ar->sdpa_partial) {
        /* Every tensor slot is NULL now (or was freed above); reset ws_rows
         * so the fast path cannot hand NULL tensors to the forward pass. */
        ar->ws_rows = 0;
        set_error(error, error_size, "sdpa partial allocation failed");
        return 0;
    }
    size_t kv_elems = cap * YUE_AR_KV_HEADS * YUE_HEAD_DIM;
    if (getenv("QWEN_AR_TRACE")) fprintf(stderr, "[ws] alloc cap=%zu pad=%u\n", cap, ar->q_pad_rows);
    ar->x = h3_gpu_tensor_new_bf16(ar->gpu, hidden_elems);
    ar->normed = h3_gpu_tensor_new_bf16(ar->gpu, hidden_elems);
    ar->q = h3_gpu_tensor_new_bf16(ar->gpu,
                                   (size_t)ar->q_pad_rows * YUE_HIDDEN);
    ar->k = h3_gpu_tensor_new_bf16(ar->gpu, kv_elems);
    ar->v = h3_gpu_tensor_new_bf16(ar->gpu, kv_elems);
    ar->attn = h3_gpu_tensor_new_bf16(ar->gpu, hidden_elems);
    ar->branch = h3_gpu_tensor_new_bf16(ar->gpu, hidden_elems);
    ar->fc1out = h3_gpu_tensor_new_bf16(ar->gpu, cap * 2 * YUE_MLP);
    ar->mlp_mid = h3_gpu_tensor_new_bf16(ar->gpu, cap * YUE_MLP);
    if (!ar->x || !ar->normed || !ar->q || !ar->k || !ar->v || !ar->attn ||
        !ar->branch || !ar->fc1out || !ar->mlp_mid) {
        /* ws_rows must follow the actual allocations: leaving the old value
         * would let the fast path skip the next rebuild while some slots
         * are still NULL, failing every forward until a larger prefill. */
        ar->ws_rows = 0;
        set_error(error, error_size, "workspace allocation failed");
        return 0;
    }
    ar->ws_rows = cap;
    return 1;
}

static int causal_limit_reserve(yue_ar *ar, size_t rows, size_t offset,
                                size_t real_rows, char *error,
                                size_t error_size) {
    uint32_t padded = ((uint32_t)rows + 63u) & ~63u;
    uint32_t *host = malloc(sizeof(uint32_t) * padded);
    if (!host) {
        set_error(error, error_size, "out of memory");
        return 0;
    }
    /* Real rows keep their causal bound; bucket-pad rows clamp to the real
     * kv length so attention never touches unwritten cache slots. */
    for (uint32_t r = 0; r < padded; r++) {
        uint32_t row = r < (uint32_t)real_rows ? r : (uint32_t)real_rows - 1;
        host[r] = (uint32_t)offset + row + 1;
    }
    h3_gpu_tensor_free(ar->causal_limit);
    ar->causal_limit = h3_gpu_tensor_from_u32(ar->gpu, host, padded);
    free(host);
    if (!ar->causal_limit) {
        set_error(error, error_size, "limit upload failed");
        return 0;
    }
    return 1;
}

/* ---------------------------------------------------------------- forward */

static void dump_tensor(h3_gpu *gpu, const h3_gpu_tensor *tensor,
                        const char *dir, const char *name) {
    if (!dir || !dir[0] || !tensor) return;
    size_t count = h3_gpu_tensor_elements(tensor);
    /* Copy through a staging tensor + submit so the readback is strictly
     * ordered after the in-flight work (CPU reads of a shared buffer racing
     * unsubmitted encodes observe stale bytes). */
    h3_gpu_tensor *staging = h3_gpu_tensor_new_bf16(gpu, count);
    if (!staging) return;
    /* A command buffer may already be open in the caller's encode chain;
     * continue on the ordered queue instead of failing. */
    if ((h3_gpu_begin(gpu) || h3_gpu_continue(gpu)) &&
        h3_gpu_copy_bf16(gpu, staging, 0, tensor, 0, count) &&
        h3_gpu_submit(gpu)) {
        uint16_t *bits = malloc(sizeof(uint16_t) * count);
        if (bits && h3_gpu_tensor_read_bf16(staging, bits, count)) {
            char path[4096];
            snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
            FILE *file = fopen(path, "wb");
            if (file) {
                fwrite(bits, sizeof(uint16_t), count, file);
                fclose(file);
            }
        }
        free(bits);
    }
    h3_gpu_tensor_free(staging);
    /* Leaves the command buffer CLOSED (submit semantics); call this only at
     * submit boundaries, never mid-chain. */
    (void)gpu;
}

/* One AR projection dispatch: the ordinary bf16 linear (MPS/GEMV). */
static int run_linear(yue_ar *ar, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const yue_ar_linear *lin,
                      uint32_t rows, uint32_t in_dim, uint32_t out_dim,
                      char *error, size_t error_size, const char *what) {
    if (h3_gpu_linear_bf16(ar->gpu, output, input, lin->w, NULL, rows,
                           in_dim, out_dim))
        return 1;
    set_error(error, error_size, "%s linear failed: %s", what,
              h3_gpu_error(ar->gpu));
    return 0;
}

/* Rope tables for a chunk at absolute position `offset`: the shared table
 * serves offset 0; later offsets upload a fresh [rows, pairs] slice. */
static int rope_for_offset(yue_ar *ar, size_t offset, size_t rows,
                           h3_gpu_tensor **cos_out, h3_gpu_tensor **sin_out,
                           char *error, size_t error_size) {
    if (offset == 0) {
        *cos_out = ar->rope_cos;
        *sin_out = ar->rope_sin;
        return 1;
    }
    size_t elements = rows * YUE_AR_PAIRS;
    float *cos_host = malloc(sizeof(float) * elements);
    float *sin_host = malloc(sizeof(float) * elements);
    if (!cos_host || !sin_host) {
        free(cos_host);
        free(sin_host);
        set_error(error, error_size, "out of memory for rope slice");
        return 0;
    }
    memcpy(cos_host, ar->rope_cos_host + offset * YUE_AR_PAIRS,
           sizeof(float) * elements);
    memcpy(sin_host, ar->rope_sin_host + offset * YUE_AR_PAIRS,
           sizeof(float) * elements);
    *cos_out = h3_gpu_tensor_from_f32(ar->gpu, cos_host, elements);
    *sin_out = h3_gpu_tensor_from_f32(ar->gpu, sin_host, elements);
    free(cos_host);
    free(sin_host);
    if (!*cos_out || !*sin_out) {
        /* Half-uploaded slice: release whatever landed so the caller has
         * nothing to clean up on the failure path. */
        h3_gpu_tensor_free(*cos_out);
        h3_gpu_tensor_free(*sin_out);
        *cos_out = NULL;
        *sin_out = NULL;
        set_error(error, error_size, "rope slice upload failed");
        return 0;
    }
    return 1;
}

/* Forward core. chain_open=0: self-contained — opens its own command
 * buffer, gives the embed gather its own submission, and submits at the
 * tail. chain_open=1: the caller owns an open command buffer; everything
 * encodes onto it and the chain stays open for the caller's submit, so a
 * decode step can chain both branch forwards and their logit projections
 * into one wait (the per-layer continue commits keep the GPU fed while
 * later layers encode on the CPU). */
static h3_gpu_tensor *forward_run(yue_ar *ar, const uint32_t *tokens,
                                  size_t count, yue_ar_cache *cache,
                                  int chain_open, char *error,
                                  size_t error_size) {
    if (!ar || !tokens || !count || !cache) {
        set_error(error, error_size, "invalid forward arguments");
        return NULL;
    }
    if (getenv("QWEN_AR_TRACE")) fprintf(stderr, "[f] forward enter count=%zu\n", count);
    size_t offset = cache->length;
    if (offset + count > cache->capacity) {
        set_error(error, error_size,
                  "cache capacity %zu exceeded by %zu+%zu", cache->capacity,
                  offset, count);
        return NULL;
    }
    if (offset + count > YUE_CONTEXT) {
        set_error(error, error_size, "context %u exceeded", YUE_CONTEXT);
        return NULL;
    }
    /* Guard the gather: an out-of-range id would read past the embedding
     * table on the device. */
    for (size_t r = 0; r < count; r++) {
        if (tokens[r] >= YUE_VOCAB_SIZE) {
            set_error(error, error_size, "token id %u out of range",
                      tokens[r]);
            return NULL;
        }
    }
    /* Prefill row bucketing (constant-water-level work): every distinct
     * prefix length compiles a fresh set of MPS linear graphs whose freed
     * pages the Metal driver pool never returns, so songs with unique
     * prompt+token counts ratcheted footprint ~2 GB per request. Pad the
     * prefill to a 512-row bucket with token id 0; the causal limit keeps
     * real rows from ever attending the pad rows, so the real rows are
     * bit-identical (per-row kernels and K-dim-unchanged GEMMs), and the
     * pad K/V slots beyond cache->length are never read again. Decode
     * steps (count == 1) keep their exact row count. */
    size_t rows = count > 1 ? yue_align_rows(count) : count;
    const char *pad_env = getenv("YUE_NO_PREFILL_PAD");
    if (pad_env && pad_env[0] != '0') rows = count;
    if (rows > count && offset + rows > YUE_CONTEXT) {
        /* The padded rows would need rope phases past the host table
         * (rope_cos_host holds YUE_CONTEXT * PAIRS). Pad rows are masked
         * out by the causal limit, so dropping the padding is
         * numerically transparent. */
        rows = count;
    }
    uint32_t *ids_host = NULL;
    const uint32_t *ids_in = tokens;
    if (rows > count) {
        ids_host = malloc(sizeof(uint32_t) * rows);
        if (!ids_host) {
            set_error(error, error_size, "out of memory");
            return NULL;
        }
        memcpy(ids_host, tokens, sizeof(uint32_t) * count);
        for (size_t r = count; r < rows; r++) ids_host[r] = 0;
        ids_in = ids_host;
    }
    if (!workspace_reserve(ar, rows, error, error_size) ||
        !causal_limit_reserve(ar, rows, offset, count, error, error_size)) {
        free(ids_host);
        return NULL;
    }

    h3_gpu_tensor *ids = h3_gpu_tensor_from_u32(ar->gpu, ids_in, rows);
    free(ids_host);
    if (!ids) {
        set_error(error, error_size, "token upload failed");
        return NULL;
    }
    h3_gpu_tensor *rope_cos = NULL, *rope_sin = NULL;
    if (!rope_for_offset(ar, offset, rows, &rope_cos, &rope_sin, error,
                         error_size)) {
        h3_gpu_tensor_free(ids);
        return NULL;
    }

    size_t hidden_elems = rows * YUE_HIDDEN;
    uint32_t rows32 = (uint32_t)rows;

    /* Layer-tail commit cadence (H3 pipeline: per-layer continue keeps the
     * GPU fed while later layers encode). YUE_AR_CONTINUE_EVERY=N batches N
     * layers per command buffer — same kernels in the same order, fewer
     * buffer-scheduling/completion round-trips. A/B measured on serve. */
    int continue_every = 1;
    const char *cadence = getenv("YUE_AR_CONTINUE_EVERY");
    if (cadence && *cadence) {
        long value = atol(cadence);
        if (value >= 1 && value <= 64) continue_every = (int)value;
    }

    /* Fused q/k/v decode GEMV (default on; H3_QKV_GEMV=0 disables): one
     * dispatch over all q+k+v columns instead of three serial dispatches
     * whose k/v halves (16 threadgroups each) starve occupancy in the
     * serial decode chain. Bit-exact with the per-matrix GEMV; serve A/B
     * wins ~4% per token (M7 round 4). Prefill (rows > 1) keeps the
     * batched MPS path, and the wrapper's shape/alignment gates fall
     * back to the three per-matrix projections on any mismatch. */
    const char *qkv_env = getenv("H3_QKV_GEMV");
    int fused_qkv = rows32 == 1 && (!qkv_env || qkv_env[0] != '0');

    /* Fused q/k head-norm + rope prologue (default on;
     * YUE_HN_ROPE_SEPARATE=1 keeps the four-kernel path): one dispatch
     * covers both tensors, the square-sum keeps the reference
     * single-thread ascending fma chain, so it is bit-identical to the
     * separate norm + rope pair. Probe T (M11): 1.39 ms/branch-step on
     * the dual-branch decode chain; the first cut that serialized norm
     * AND rope on one thread per head was 0.45 ms SLOWER than the four
     * separate kernels — the parallel phases are the whole point. */
    const char *hnr_env = getenv("YUE_HN_ROPE_SEPARATE");
    int hn_rope_separate = hnr_env && hnr_env[0] != '0';

    int ok;
    if (chain_open) {
        /* The caller owns the open command buffer; the gather rides it. */
        ok = qw_gpu_embed_gather_bf16(ar->gpu, ar->x, ar->embed, ids, rows32,
                                      YUE_HIDDEN);
    } else {
        int opened = h3_gpu_begin(ar->gpu);
        ok = opened &&
             qw_gpu_embed_gather_bf16(ar->gpu, ar->x, ar->embed, ids, rows32,
                                      YUE_HIDDEN) &&
             h3_gpu_submit(ar->gpu);
        /* A failed gather leaves the buffer open; drain it now or every
         * later h3_gpu_begin on this context is blocked forever. */
        if (opened && !ok) (void)h3_gpu_submit(ar->gpu);
    }
    h3_gpu_tensor_free(ids);
    const char *dump_dir = getenv("QWEN_AR_DUMP");
    /* Dumps are tagged by absolute position so decode-step captures do not
     * overwrite the prefill ones ("layer3" prefill vs "d54_layer3" decode). */
    char pfx[24] = "";
    if (offset) snprintf(pfx, sizeof(pfx), "d%zu_", offset);
    if (ok) {
        h3_gpu_begin(ar->gpu);
        char dname[64];
        snprintf(dname, sizeof(dname), "%sembed", pfx);
        dump_tensor(ar->gpu, ar->x, dump_dir, dname);
        if (getenv("QWEN_AR_TRACE_LAYER0")) {
            snprintf(dname, sizeof(dname), "%sc_x_early", pfx);
            dump_tensor(ar->gpu, ar->x, dump_dir, dname);
        }
    }
    if (getenv("QWEN_AR_TRACE")) fprintf(stderr, "[f] prefill gather ok=%d\n", ok);
    if (!ok) {
        if (rope_cos != ar->rope_cos) h3_gpu_tensor_free(rope_cos);
        if (rope_sin != ar->rope_sin) h3_gpu_tensor_free(rope_sin);
        set_error(error, error_size, "embed gather failed: %s",
                  h3_gpu_error(ar->gpu));
        return NULL;
    }
    h3_gpu_begin(ar->gpu);

    for (int i = 0; ok && i < ar->layers; i++) {
        const yue_ar_layer *layer = &ar->layer[i];
        const int trace_layer0 = i == 0 && getenv("QWEN_AR_TRACE_LAYER0") != NULL;
        if (getenv("QWEN_AR_TRACE")) fprintf(stderr, "[f] layer %d begin\n", i);
        /* attention block */
        ok = qw_gpu_rms_norm_bf16(ar->gpu, ar->normed, ar->x,
                                  layer->input_norm, rows32, YUE_HIDDEN,
                                  YUE_RMS_EPS) &&
             (fused_qkv
                  ? (qw_gpu_qkv_gemv_bf16(
                         ar->gpu, ar->q, ar->k, ar->v, ar->normed,
                         layer->q_proj.w, layer->k_proj.w,
                         layer->v_proj.w, YUE_HIDDEN, YUE_HIDDEN,
                         YUE_AR_KV_HEADS * YUE_HEAD_DIM) ||
                     (run_linear(ar, ar->q, ar->normed, &layer->q_proj,
                                 rows32, YUE_HIDDEN, YUE_HIDDEN, error,
                                 error_size, "q_proj") &&
                      run_linear(ar, ar->k, ar->normed, &layer->k_proj,
                                 rows32, YUE_HIDDEN,
                                 YUE_AR_KV_HEADS * YUE_HEAD_DIM, error,
                                 error_size, "k_proj") &&
                      run_linear(ar, ar->v, ar->normed, &layer->v_proj,
                                 rows32, YUE_HIDDEN,
                                 YUE_AR_KV_HEADS * YUE_HEAD_DIM, error,
                                 error_size, "v_proj")))
                  : run_linear(ar, ar->q, ar->normed, &layer->q_proj,
                               rows32, YUE_HIDDEN, YUE_HIDDEN, error,
                               error_size, "q_proj") &&
                        run_linear(ar, ar->k, ar->normed, &layer->k_proj,
                                   rows32, YUE_HIDDEN,
                                   YUE_AR_KV_HEADS * YUE_HEAD_DIM, error,
                                   error_size, "k_proj") &&
                        run_linear(ar, ar->v, ar->normed, &layer->v_proj,
                                   rows32, YUE_HIDDEN,
                                   YUE_AR_KV_HEADS * YUE_HEAD_DIM, error,
                                   error_size, "v_proj")) &&
             /* Fused head-norm + rope (bit-identical, one dispatch for
              * q+k); wrapper gate failure or YUE_HN_ROPE_SEPARATE=1
              * falls back to the four-kernel prologue below. Same-buffer
              * kernels run in order, so the whole layer encodes into ONE
              * command buffer and submits once at the tail. */
             ((hn_rope_separate
                   ? 0
                   : qw_gpu_head_rms_rope_bf16(
                         ar->gpu, ar->q, ar->k, layer->q_norm, layer->k_norm,
                         rope_cos, rope_sin, rows32, YUE_HEADS,
                         YUE_AR_KV_HEADS, YUE_HEAD_DIM, YUE_RMS_EPS)) ||
              (qw_gpu_head_rms_norm_bf16(ar->gpu, ar->q, layer->q_norm, rows32,
                                         YUE_HEADS, YUE_HEAD_DIM,
                                         YUE_RMS_EPS) &&
               qw_gpu_head_rms_norm_bf16(ar->gpu, ar->k, layer->k_norm, rows32,
                                         YUE_AR_KV_HEADS, YUE_HEAD_DIM,
                                         YUE_RMS_EPS) &&
               qw_gpu_qk_rope_half_bf16(ar->gpu, ar->q, ar->q, ar->q, ar->q,
                                        rope_cos, rope_sin, rows32, YUE_HEADS,
                                        YUE_HEAD_DIM) &&
               qw_gpu_qk_rope_half_bf16(ar->gpu, ar->k, ar->k, ar->k, ar->k,
                                        rope_cos, rope_sin, rows32,
                                        YUE_AR_KV_HEADS, YUE_HEAD_DIM)));
        if (getenv("QWEN_AR_TRACE")) fprintf(stderr, "[f] layer %d rope done\n", i);
        if (trace_layer0) {
            char dname[64];
            dump_tensor(ar->gpu, layer->input_norm, dump_dir, "c_weight_norm");
            snprintf(dname, sizeof(dname), "%sc_x_norm_input", pfx);
            dump_tensor(ar->gpu, ar->x, dump_dir, dname);
            snprintf(dname, sizeof(dname), "%sl0_normed", pfx);
            dump_tensor(ar->gpu, ar->normed, dump_dir, dname);
            snprintf(dname, sizeof(dname), "%sl0_q", pfx);
            dump_tensor(ar->gpu, ar->q, dump_dir, dname);
            snprintf(dname, sizeof(dname), "%sl0_k", pfx);
            dump_tensor(ar->gpu, ar->k, dump_dir, dname);
            snprintf(dname, sizeof(dname), "%sl0_v", pfx);
            dump_tensor(ar->gpu, ar->v, dump_dir, dname);
            h3_gpu_begin(ar->gpu); /* dump batch ends with a closed buffer */
        }
        if (!ok) {
            set_error(error, error_size, "layer %d attention prologue: %s", i,
                      h3_gpu_error(ar->gpu));
            break;
        }

        /* append K/V to cache, then attend over [0, offset+count). Only
         * the real rows are copied: the bucket-pad rows' cache slots stay
         * unwritten and unreachable (limits clamp to offset+count). */
        ok = h3_gpu_copy_bf16(ar->gpu, cache->keys[i],
                              offset * YUE_AR_KV_HEADS * YUE_HEAD_DIM, ar->k,
                              0, count * YUE_AR_KV_HEADS * YUE_HEAD_DIM) &&
             h3_gpu_copy_bf16(ar->gpu, cache->values[i],
                              offset * YUE_AR_KV_HEADS * YUE_HEAD_DIM, ar->v,
                              0, count * YUE_AR_KV_HEADS * YUE_HEAD_DIM);
        const char *flash2_env = getenv("YUE_SDPA_FLASH2");
        const int flash2_fallback =
            rows32 > 1 || (flash2_env && flash2_env[0] != '0');
        ok = ok && (flash2_fallback ?
                  qw_gpu_flash_sdpa_prefix_bf16(
                      ar->gpu, ar->attn, ar->q, cache->keys[i],
                      cache->values[i], ar->causal_limit, rows32,
                      (uint32_t)(offset + count), YUE_HEADS, YUE_AR_KV_HEADS,
                      YUE_HEAD_DIM, 1.0f / sqrtf((float)YUE_HEAD_DIM)) :
                  /* Decode-step SDPA (default; YUE_SDPA_FLASH2=1 returns to
                   * the flash2 tile kernel). The flash2 tile engine pays
                   * ~134 us at q_rows=1/kv=513 (probe U) vs ~28 us for the
                   * split-K pair. The first cut of the scalar kernels broke
                   * the score: a single per-thread q element was hoisted out
                   * of the row loop, so the score was q[tid]*Sigma_d k[r][d]
                   * instead of the full dot product — bit-exact vs flash2
                   * only for constant q (which is exactly what the probe's
                   * kv=1 mini-case wrote into the shared q tensor, masking
                   * the bug as "kv=1/64 three-way bit-exact"). Fixed by
                   * staging the full q row in threadgroup memory; ar gate
                   * went 1.25 -> 0.027 relL2, all suites green. */
                  qw_gpu_sdpa_decode_split_bf16(
                      ar->gpu, ar->attn, ar->q, cache->keys[i],
                      cache->values[i], (uint32_t)(offset + count),
                      YUE_HEADS, YUE_AR_KV_HEADS, YUE_HEAD_DIM,
                      1.0f / sqrtf((float)YUE_HEAD_DIM), ar->sdpa_partial,
                      YUE_AR_MAX_SPLITS));
        if (!ok) {
            set_error(error, error_size, "layer %d attention: %s", i,
                      h3_gpu_error(ar->gpu));
            break;
        }
        if (trace_layer0) {
            char dname[64];
            snprintf(dname, sizeof(dname), "%sl0_attn", pfx);
            dump_tensor(ar->gpu, ar->attn, dump_dir, dname);
            h3_gpu_begin(ar->gpu);
        }

        /* output projection, residual, MLP block */
        ok = run_linear(ar, ar->branch, ar->attn, &layer->o_proj, rows32,
                        YUE_HIDDEN, YUE_HIDDEN, error, error_size,
                        "o_proj") &&
             h3_gpu_add_bf16(ar->gpu, ar->x, ar->x, ar->branch,
                             (uint32_t)hidden_elems) &&
             qw_gpu_rms_norm_bf16(ar->gpu, ar->normed, ar->x,
                                  layer->post_norm, rows32, YUE_HIDDEN,
                                  YUE_RMS_EPS) &&
             run_linear(ar, ar->fc1out, ar->normed, &layer->fc1, rows32,
                        YUE_HIDDEN, 2 * YUE_MLP, error, error_size, "fc1") &&
             qw_gpu_swiglu_bf16(ar->gpu, ar->mlp_mid, ar->fc1out,
                                (uint32_t)(rows * YUE_MLP), YUE_MLP) &&
             run_linear(ar, ar->branch, ar->mlp_mid, &layer->down, rows32,
                        YUE_MLP, YUE_HIDDEN, error, error_size, "down") &&
             h3_gpu_add_bf16(ar->gpu, ar->x, ar->x, ar->branch,
                             (uint32_t)hidden_elems);
        /* commit without waiting: the next layer encodes while the GPU
         * runs this one (submit would serialize CPU/GPU per layer). The
         * last layer skips the continue — the final norm (and the caller's
         * submit or next chain op) ride the same buffer. */
        if (ok && i != ar->layers - 1 && (i % continue_every) == continue_every - 1)
            ok = h3_gpu_continue(ar->gpu);
        if (getenv("QWEN_AR_TRACE")) fprintf(stderr, "[f] layer %d done\n", i);
        if (trace_layer0) {
            /* fc1out/mlp_mid still hold this layer's values here. */
            char dname[64];
            snprintf(dname, sizeof(dname), "%sl0_fc1out", pfx);
            dump_tensor(ar->gpu, ar->fc1out, dump_dir, dname);
            snprintf(dname, sizeof(dname), "%sl0_mlp", pfx);
            dump_tensor(ar->gpu, ar->mlp_mid, dump_dir, dname);
            h3_gpu_begin(ar->gpu);
        }
        if (!ok) {
            /* Copy first: error is both the destination and an argument
             * (vsnprintf overlap is UB), and the inner op may not have
             * written it, in which case only the GPU error is relevant. */
            char prior[256] = "";
            if (error && error_size)
                snprintf(prior, sizeof(prior), "%.*s",
                         (int)sizeof(prior) - 1, error);
            if (prior[0])
                set_error(error, error_size, "layer %d mlp: %s (%s)", i,
                          prior, h3_gpu_error(ar->gpu));
            else
                set_error(error, error_size, "layer %d mlp: %s",
                          h3_gpu_error(ar->gpu));
            break;
        }
        if (dump_dir && dump_dir[0]) {
            char name[64];
            snprintf(name, sizeof(name), "%slayer%d", pfx, i);
            dump_tensor(ar->gpu, ar->x, dump_dir, name);
            /* dump_tensor leaves the command buffer closed; reopen so the
             * next layer's prologue (and the final norm) find it open. */
            h3_gpu_begin(ar->gpu);
        }
    }

    if (rope_cos != ar->rope_cos) h3_gpu_tensor_free(rope_cos);
    if (rope_sin != ar->rope_sin) h3_gpu_tensor_free(rope_sin);

    if (!ok) {
        /* Self-contained mode drains the half-encoded chain; chain-open
         * mode leaves that to the caller (documented in yue_ar.h). */
        if (!chain_open) h3_gpu_submit(ar->gpu);
        return NULL;
    }

    /* final norm -> hidden; result lives in ar->normed. */
    ok = qw_gpu_rms_norm_bf16(ar->gpu, ar->normed, ar->x, ar->norm, rows32,
                              YUE_HIDDEN, YUE_RMS_EPS);
    if (!chain_open) {
        ok = ok && h3_gpu_submit(ar->gpu);
        if (!ok) (void)h3_gpu_submit(ar->gpu); /* drain: no-op once closed */
    }
    {
        char dname[64];
        snprintf(dname, sizeof(dname), "%sfinal", pfx);
        dump_tensor(ar->gpu, ar->normed, dump_dir, dname);
        /* dump_tensor submits (closing the chain); chain-open mode must
         * hand the caller an open buffer, so reopen it after a dump. */
        if (chain_open && dump_dir && dump_dir[0]) h3_gpu_begin(ar->gpu);
    }
    if (!ok) {
        set_error(error, error_size, "final norm failed: %s",
                  h3_gpu_error(ar->gpu));
        return NULL;
    }
    cache->length = offset + count;
    return ar->normed;
}

h3_gpu_tensor *yue_ar_forward(yue_ar *ar, const uint32_t *tokens, size_t count,
                              yue_ar_cache *cache, char *error,
                              size_t error_size) {
    return forward_run(ar, tokens, count, cache, 0, error, error_size);
}

h3_gpu_tensor *yue_ar_forward_open(yue_ar *ar, const uint32_t *tokens,
                                   size_t count, yue_ar_cache *cache,
                                   char *error, size_t error_size) {
    return forward_run(ar, tokens, count, cache, 1, error, error_size);
}

/* Batched CFG decode forward (M15): both branch rows ride one [2, hidden]
 * stream so every linear reads the weight stream once (probe H: rows=2
 * gpu time = 0.98x rows=1 on fc1/down/lm; the serial dual-branch path
 * pays the full weight stream twice). Each row attends and writes its
 * own cache; the caches may sit at different absolute offsets (the
 * negative prefix is longer), so the rope table carries one phase row
 * per branch. count is fixed at 1 (one decode token per branch). The
 * fused qkv GEMV and the dump harness are single-branch features and
 * are not used here; the projection stays with the caller (chain-open
 * semantics, like yue_ar_forward_open). */
static h3_gpu_tensor *forward_run2(yue_ar *ar, uint32_t token_pos,
                                   uint32_t token_neg,
                                   yue_ar_cache *cache_pos,
                                   yue_ar_cache *cache_neg, char *error,
                                   size_t error_size) {
    if (!ar || !cache_pos || !cache_neg || cache_pos == cache_neg) {
        set_error(error, error_size, "invalid batch forward arguments");
        return NULL;
    }
    size_t off_p = cache_pos->length, off_n = cache_neg->length;
    if (off_p + 1 > cache_pos->capacity || off_n + 1 > cache_neg->capacity) {
        set_error(error, error_size, "batch decode cache capacity exceeded");
        return NULL;
    }
    if (off_p + 1 > YUE_CONTEXT || off_n + 1 > YUE_CONTEXT) {
        set_error(error, error_size, "batch decode context exceeded");
        return NULL;
    }
    if (token_pos >= YUE_VOCAB_SIZE || token_neg >= YUE_VOCAB_SIZE) {
        set_error(error, error_size, "batch decode token id out of range");
        return NULL;
    }
    size_t rows = 2;
    if (!workspace_reserve(ar, rows, error, error_size)) return NULL;

    uint32_t ids_host[2] = {token_pos, token_neg};
    h3_gpu_tensor *ids = h3_gpu_tensor_from_u32(ar->gpu, ids_host, rows);
    if (!ids) {
        set_error(error, error_size, "batch token upload failed");
        return NULL;
    }
    size_t pairs = YUE_AR_PAIRS;
    float *cos_host = malloc(sizeof(float) * rows * pairs);
    float *sin_host = malloc(sizeof(float) * rows * pairs);
    if (!cos_host || !sin_host) {
        free(cos_host);
        free(sin_host);
        h3_gpu_tensor_free(ids);
        set_error(error, error_size, "out of memory for batch rope");
        return NULL;
    }
    memcpy(cos_host, ar->rope_cos_host + off_p * pairs,
           sizeof(float) * pairs);
    memcpy(sin_host, ar->rope_sin_host + off_p * pairs,
           sizeof(float) * pairs);
    memcpy(cos_host + pairs, ar->rope_cos_host + off_n * pairs,
           sizeof(float) * pairs);
    memcpy(sin_host + pairs, ar->rope_sin_host + off_n * pairs,
           sizeof(float) * pairs);
    h3_gpu_tensor *rope_cos =
        h3_gpu_tensor_from_f32(ar->gpu, cos_host, rows * pairs);
    h3_gpu_tensor *rope_sin =
        h3_gpu_tensor_from_f32(ar->gpu, sin_host, rows * pairs);
    free(cos_host);
    free(sin_host);
    if (!rope_cos || !rope_sin) {
        h3_gpu_tensor_free(ids);
        h3_gpu_tensor_free(rope_cos);
        h3_gpu_tensor_free(rope_sin);
        set_error(error, error_size, "batch rope upload failed");
        return NULL;
    }

    uint32_t rows32 = (uint32_t)rows;
    size_t hidden_elems = rows * YUE_HIDDEN;
    size_t kv1 = YUE_AR_KV_HEADS * YUE_HEAD_DIM; /* one branch k/v row */
    int hn_rope_separate = getenv("YUE_HN_ROPE_SEPARATE") &&
                           getenv("YUE_HN_ROPE_SEPARATE")[0] != '0';

    /* Chain-open mode: the caller owns the open buffer (decode_step2
     * began it) — do NOT begin again; h3_gpu_begin returns 0 when a
     * command is already open, which would silently skip the gather. */
    int ok = qw_gpu_embed_gather_bf16(ar->gpu, ar->x, ar->embed, ids, rows32,
                                      YUE_HIDDEN);
    h3_gpu_tensor_free(ids);
    if (!ok) {
        h3_gpu_tensor_free(rope_cos);
        h3_gpu_tensor_free(rope_sin);
        set_error(error, error_size, "batch embed gather failed: %s",
                  h3_gpu_error(ar->gpu));
        return NULL;
    }
    /* The caller (decode_step2) owns the open command buffer. Calling
     * h3_gpu_begin() here is a no-op normally, but with a closed chain it
     * would open a fresh buffer and split the caller's work in two. */

    for (int i = 0; ok && i < ar->layers; i++) {
        const yue_ar_layer *layer = &ar->layer[i];
        ok = qw_gpu_rms_norm_bf16(ar->gpu, ar->normed, ar->x,
                                  layer->input_norm, rows32, YUE_HIDDEN,
                                  YUE_RMS_EPS) &&
             run_linear(ar, ar->q, ar->normed, &layer->q_proj, rows32,
                        YUE_HIDDEN, YUE_HIDDEN, error, error_size,
                        "q_proj") &&
             run_linear(ar, ar->k, ar->normed, &layer->k_proj, rows32,
                        YUE_HIDDEN, YUE_AR_KV_HEADS * YUE_HEAD_DIM, error,
                        error_size, "k_proj") &&
             run_linear(ar, ar->v, ar->normed, &layer->v_proj, rows32,
                        YUE_HIDDEN, YUE_AR_KV_HEADS * YUE_HEAD_DIM, error,
                        error_size, "v_proj") &&
             ((hn_rope_separate
                   ? 0
                   : qw_gpu_head_rms_rope_bf16(
                         ar->gpu, ar->q, ar->k, layer->q_norm,
                         layer->k_norm, rope_cos, rope_sin, rows32,
                         YUE_HEADS, YUE_AR_KV_HEADS, YUE_HEAD_DIM,
                         YUE_RMS_EPS)) ||
              (qw_gpu_head_rms_norm_bf16(ar->gpu, ar->q, layer->q_norm,
                                         rows32, YUE_HEADS, YUE_HEAD_DIM,
                                         YUE_RMS_EPS) &&
               qw_gpu_head_rms_norm_bf16(ar->gpu, ar->k, layer->k_norm,
                                         rows32, YUE_AR_KV_HEADS,
                                         YUE_HEAD_DIM, YUE_RMS_EPS) &&
               qw_gpu_qk_rope_half_bf16(ar->gpu, ar->q, ar->q, ar->q, ar->q,
                                        rope_cos, rope_sin, rows32,
                                        YUE_HEADS, YUE_HEAD_DIM) &&
               qw_gpu_qk_rope_half_bf16(ar->gpu, ar->k, ar->k, ar->k, ar->k,
                                        rope_cos, rope_sin, rows32,
                                        YUE_AR_KV_HEADS, YUE_HEAD_DIM)));
        if (!ok) {
            set_error(error, error_size, "batch layer %d prologue: %s", i,
                      h3_gpu_error(ar->gpu));
            break;
        }
        /* Split K/V into the two caches, then each row attends its own
         * cache (the scratch partial is reused serially — the command
         * buffer's hazard tracking orders split1/merge1/split2/merge2). */
        ok = h3_gpu_copy_bf16(ar->gpu, cache_pos->keys[i], off_p * kv1,
                              ar->k, 0, kv1) &&
             h3_gpu_copy_bf16(ar->gpu, cache_neg->keys[i], off_n * kv1,
                              ar->k, kv1, kv1) &&
             h3_gpu_copy_bf16(ar->gpu, cache_pos->values[i], off_p * kv1,
                              ar->v, 0, kv1) &&
             h3_gpu_copy_bf16(ar->gpu, cache_neg->values[i], off_n * kv1,
                              ar->v, kv1, kv1) &&
             qw_gpu_sdpa_decode_split_bf16_off(
                 ar->gpu, ar->attn, ar->q, cache_pos->keys[i],
                 cache_pos->values[i], 0, 0, (uint32_t)(off_p + 1),
                 YUE_HEADS, YUE_AR_KV_HEADS, YUE_HEAD_DIM,
                 1.0f / sqrtf((float)YUE_HEAD_DIM), ar->sdpa_partial,
                 YUE_AR_MAX_SPLITS) &&
             qw_gpu_sdpa_decode_split_bf16_off(
                 ar->gpu, ar->attn, ar->q, cache_neg->keys[i],
                 cache_neg->values[i],
                 (size_t)YUE_HEADS * YUE_HEAD_DIM,
                 (size_t)YUE_HEADS * YUE_HEAD_DIM, (uint32_t)(off_n + 1),
                 YUE_HEADS, YUE_AR_KV_HEADS, YUE_HEAD_DIM,
                 1.0f / sqrtf((float)YUE_HEAD_DIM), ar->sdpa_partial,
                 YUE_AR_MAX_SPLITS);
        if (!ok) {
            set_error(error, error_size, "batch layer %d attention: %s", i,
                      h3_gpu_error(ar->gpu));
            break;
        }
        ok = run_linear(ar, ar->branch, ar->attn, &layer->o_proj, rows32,
                        YUE_HIDDEN, YUE_HIDDEN, error, error_size,
                        "o_proj") &&
             h3_gpu_add_bf16(ar->gpu, ar->x, ar->x, ar->branch,
                             (uint32_t)hidden_elems) &&
             qw_gpu_rms_norm_bf16(ar->gpu, ar->normed, ar->x,
                                  layer->post_norm, rows32, YUE_HIDDEN,
                                  YUE_RMS_EPS) &&
             run_linear(ar, ar->fc1out, ar->normed, &layer->fc1, rows32,
                        YUE_HIDDEN, 2 * YUE_MLP, error, error_size, "fc1") &&
             qw_gpu_swiglu_bf16(ar->gpu, ar->mlp_mid, ar->fc1out,
                                (uint32_t)(rows * YUE_MLP), YUE_MLP) &&
             run_linear(ar, ar->branch, ar->mlp_mid, &layer->down,
                        rows32, YUE_MLP, YUE_HIDDEN, error, error_size,
                        "down") &&
             h3_gpu_add_bf16(ar->gpu, ar->x, ar->x, ar->branch,
                             (uint32_t)hidden_elems);
        if (ok && i != ar->layers - 1) ok = h3_gpu_continue(ar->gpu);
        if (!ok) {
            char prior[256] = "";
            if (error && error_size)
                snprintf(prior, sizeof(prior), "%.*s",
                         (int)sizeof(prior) - 1, error);
            if (prior[0])
                set_error(error, error_size, "batch layer %d mlp: %s (%s)",
                          i, prior, h3_gpu_error(ar->gpu));
            else
                set_error(error, error_size, "batch layer %d mlp: %s",
                          h3_gpu_error(ar->gpu));
            break;
        }
    }
    h3_gpu_tensor_free(rope_cos);
    h3_gpu_tensor_free(rope_sin);
    if (!ok) return NULL;
    ok = qw_gpu_rms_norm_bf16(ar->gpu, ar->normed, ar->x, ar->norm, rows32,
                              YUE_HIDDEN, YUE_RMS_EPS);
    if (!ok) {
        set_error(error, error_size, "batch final norm failed: %s",
                  h3_gpu_error(ar->gpu));
        return NULL;
    }
    cache_pos->length = off_p + 1;
    cache_neg->length = off_n + 1;
    return ar->normed;
}

h3_gpu_tensor *yue_ar_forward_open2(yue_ar *ar, uint32_t token_pos,
                                    uint32_t token_neg,
                                    yue_ar_cache *cache_pos,
                                    yue_ar_cache *cache_neg, char *error,
                                    size_t error_size) {
    return forward_run2(ar, token_pos, token_neg, cache_pos, cache_neg,
                        error, error_size);
}

/* ---------------------------------------------------------- projection */

/* Projection core; chain_open=1 encodes onto the caller's open command
 * buffer and leaves it open (see yue_ar_project_open). */
static int project_run(yue_ar *ar, const h3_gpu_tensor *hidden, size_t rows,
                       const char *phase, h3_gpu_tensor *logits,
                       int chain_open, char *error, size_t error_size) {
    if (!ar || !hidden || !logits || !phase || rows < 1 ||
        h3_gpu_tensor_elements(hidden) < rows * YUE_HIDDEN) {
        set_error(error, error_size, "invalid projection arguments");
        return 0;
    }
    int ok = 1;
    if (!strcmp(phase, "abc")) {
        if (rows != 1) {
            /* The content GEMM writes [rows, EOD] contiguously while the
             * scatter below assumes an [rows, EOD+1] row stride — the two
             * layouts only agree for a single row (abc is always sampled
             * without CFG today). Fail loudly instead of corrupting. */
            set_error(error, error_size,
                      "abc projection requires rows == 1, got %zu", rows);
            return 0;
        }
        /* The scatter writes the EOD-th column of a [1, EOD+1] row. */
        if (h3_gpu_tensor_elements(logits) < (size_t)YUE_EOD + 1) {
            set_error(error, error_size, "abc logits buffer too small");
            return 0;
        }
        /* [0, EOD) content GEMM, then the ABC_END row into the last
         * column of each row. The vendored fused patch kernel only
         * supports qwen's patch shapes, so project a [rows, 1] column
         * and scatter one element per row. */
        h3_gpu_tensor *end_col = h3_gpu_tensor_new_bf16(ar->gpu, rows);
        if (!chain_open) ok = end_col && h3_gpu_begin(ar->gpu);
        ok = ok && end_col &&
             run_linear(ar, logits, hidden, &ar->lm_abc_content,
                        (uint32_t)rows, YUE_HIDDEN, YUE_EOD, error,
                        error_size, "lm_abc") &&
             run_linear(ar, end_col, hidden, &ar->lm_abc_end,
                        (uint32_t)rows, YUE_HIDDEN, 1, error, error_size,
                        "lm_abc_end");
        for (size_t r = 0; ok && r < rows; r++)
            ok = h3_gpu_copy_bf16(ar->gpu, logits,
                                  r * (YUE_EOD + 1) + YUE_EOD, end_col, r, 1);
        if (ok && !chain_open) ok = h3_gpu_submit(ar->gpu);
        h3_gpu_tensor_free(end_col);
    } else if (!strcmp(phase, "semantic")) {
        if (!chain_open) ok = h3_gpu_begin(ar->gpu);
        ok = ok && run_linear(ar, logits, hidden, &ar->lm_semantic,
                              (uint32_t)rows, YUE_HIDDEN,
                              YUE_CODEC_SIZE + 1, error, error_size,
                              "lm_semantic");
        if (ok && !chain_open) ok = h3_gpu_submit(ar->gpu);
    } else {
        set_error(error, error_size, "phase must be abc or semantic");
        return 0;
    }
    if (!ok) {
        /* A failed op inside an opened self-contained chain leaves the
         * command buffer open; drain it or the context is stuck. Chain-open
         * mode leaves the drain to the caller (documented in yue_ar.h). */
        if (!chain_open) (void)h3_gpu_submit(ar->gpu);
        set_error(error, error_size, "logits projection failed: %s",
                  h3_gpu_error(ar->gpu));
        return 0;
    }
    return 1;
}

int yue_ar_project_logits(yue_ar *ar, const h3_gpu_tensor *hidden, size_t rows,
                          const char *phase, h3_gpu_tensor *logits,
                          char *error, size_t error_size) {
    return project_run(ar, hidden, rows, phase, logits, 0, error, error_size);
}

int yue_ar_project_open(yue_ar *ar, const h3_gpu_tensor *hidden, size_t rows,
                        const char *phase, h3_gpu_tensor *logits,
                        char *error, size_t error_size) {
    return project_run(ar, hidden, rows, phase, logits, 1, error, error_size);
}

h3_gpu_tensor *yue_ar_embed(yue_ar *ar, const uint32_t *tokens, size_t count,
                            char *error, size_t error_size) {
    if (!ar || !tokens || !count) {
        set_error(error, error_size, "invalid embed arguments");
        return NULL;
    }
    if (!workspace_reserve(ar, count, error, error_size)) return NULL;
    for (size_t r = 0; r < count; r++) {
        if (tokens[r] >= YUE_VOCAB_SIZE) {
            set_error(error, error_size, "token id %u out of range",
                      tokens[r]);
            return NULL;
        }
    }
    h3_gpu_tensor *ids = h3_gpu_tensor_from_u32(ar->gpu, tokens, count);
    int opened = 0;
    int ok = ids && (opened = h3_gpu_begin(ar->gpu)) &&
             qw_gpu_embed_gather_bf16(ar->gpu, ar->x, ar->embed, ids,
                                      (uint32_t)count, YUE_HIDDEN) &&
             h3_gpu_submit(ar->gpu);
    if (opened && !ok) (void)h3_gpu_submit(ar->gpu);
    h3_gpu_tensor_free(ids);
    if (!ok) {
        set_error(error, error_size, "embed gather failed: %s",
                  h3_gpu_error(ar->gpu));
        return NULL;
    }
    return ar->x;
}

h3_gpu_tensor *yue_ar_cache_key(yue_ar_cache *cache, int layer) {
    return cache && cache->keys && layer >= 0 && layer < cache->layers
               ? cache->keys[layer] : NULL;
}

h3_gpu_tensor *yue_ar_cache_value(yue_ar_cache *cache, int layer) {
    return cache && cache->values && layer >= 0 && layer < cache->layers
               ? cache->values[layer] : NULL;
}
