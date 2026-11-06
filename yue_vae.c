/* Oobleck VAE decoder, fp32 (lyra vae.py parity). Weights fold from
 * the official checkpoint (weight_norm v*g/||v|| + layout transpose) into
 * [out, tap, in] and every convolution executes as per-tap GEMMs;
 * ConvTranspose accumulates strided scatter rows, dilated Conv1d gathers
 * phase sub-lattices so all GEMM operands stay in-buffer.
 *
 * GPU path (default when a h3_gpu is supplied): the same tile structure
 * with per-tap accumulating TensorOps matmul2d GEMMs over zero-ringed
 * activations (qwen-metal qw_vae_conv_mm_f32 scheme adapted to 1D), a
 * strided-accumulate ConvT tap kernel, elementwise SnakeBeta/tanh, and
 * qw_vae_add_f32 for the residual. YUE_VAE_CPU=1 falls back to the
 * reference Accelerate implementation. */
#include <Accelerate/Accelerate.h>
#include <errno.h>
#include <math.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

#include "h3_safetensors.h"
#include "yue_gpu_ext.h"
#include "yue_json.h"
#include "yue_vae.h"

#define YUE_VAE_MAX_STRIDES 8
#define YUE_VAE_MAX_BLOCKS (YUE_VAE_MAX_STRIDES)
#define YUE_VAE_CORE_FRAMES 256
#define YUE_VAE_HALO_FRAMES 16

static int set_error(char *error, size_t error_size, const char *fmt, ...) {
    if (error && error_size) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(error, error_size, fmt, args);
        va_end(args);
    }
    return 0;
}

typedef enum { YUE_L_CONV = 0, YUE_L_CONVT, YUE_L_SNAKE } yue_vae_kind;

typedef struct {
    yue_vae_kind kind;
    int cin, cout;
    int k, stride, pad, dilation;
    float *w;    /* conv: [cout][k][cin]; snake: unused */
    float *bias; /* [cout] or NULL */
    float *ea;   /* snake exp(alpha) per channel */
    float *eb;   /* snake exp(beta) per channel */
    int in_place;    /* snake: operates on the same buffer */
    int out_of_place; /* snake: writes the next buffer (res-unit head keeps
                       * the input alive for the residual add) */
    int residual;     /* conv1: output += layer input (res-unit tail) */
    char key[128]; /* checkpoint key prefix, e.g. decoder.layers.1.layers.0 */
    /* GPU path residents (repacked weight, uploaded vectors). */
    h3_gpu_tensor *w_t;    /* [k][cout_pad][cin] GEMM layout */
    h3_gpu_tensor *bias_t; /* [cout] or NULL */
    h3_gpu_tensor *ea_t;   /* snake only */
    h3_gpu_tensor *eb_t;   /* snake only */
} yue_vae_layer;

struct yue_vae {
    int channels;
    int latent_dim;
    int out_channels;
    int final_tanh;
    int blocks;
    int strides[YUE_VAE_MAX_STRIDES]; /* decoder order (reversed encoder) */
    int c_mults[YUE_VAE_MAX_BLOCKS + 1];
    yue_vae_layer *layers;
    int layer_count;
    h3_gpu *gpu; /* borrowed (engine-owned); NULL keeps the CPU path */
};

/* ------------------------------------------------------------------ load */

typedef struct {
    float *data;
    size_t count;
} f32_buf;

static int read_tensor(h3_st_header *header, const char *name, f32_buf *out,
                       int *shape, int *ndim, char *error, size_t error_size) {
    const h3_st_tensor *t = h3_st_find(header, name);
    if (!t || t->dtype != H3_DTYPE_F32) {
        set_error(error, error_size, "missing/invalid fp32 tensor %s", name);
        return 0;
    }
    *ndim = t->ndim;
    /* Clamp before writing: callers pass small stack arrays sized for the
     * expected rank, and a corrupt checkpoint may declare a larger ndim
     * (h3_safetensors caps ndim at 8 but never less). The rank check at
     * the call site then rejects it cleanly. */
    int dims = t->ndim > 8 ? 8 : t->ndim;
    for (int d = 0; d < dims; d++) shape[d] = (int)t->shape[d];
    out->count = (size_t)h3_st_tensor_elements(t);
    out->data = malloc(out->count * sizeof(float));
    if (!out->data ||
        !h3_st_read_data(header, t, out->data, out->count * sizeof(float),
                         error, error_size)) {
        free(out->data);
        out->data = NULL;
        set_error(error, error_size, "read failed: %s", name);
        return 0;
    }
    return 1;
}

/* Fold weight_norm: w = v * (g / ||v||_F per first axis), then transpose to
 * [out][tap][in]. Conv1d checkpoints are [out][in][k]; ConvTranspose are
 * [in][out][k] (is_convt). */
static float *fold_weight(const float *v, const float *g, int a, int b, int k,
                          int is_convt, char *error, size_t error_size) {
    /* norm over the last two axes, per first axis */
    float *w = malloc(sizeof(float) * (size_t)a * b * k);
    if (!w) {
        set_error(error, error_size, "out of memory");
        return NULL;
    }
    for (int i = 0; i < a; i++) {
        double sum = 0.0;
        for (int j = 0; j < b * k; j++) {
            double value = (double)v[(size_t)i * b * k + j];
            sum += value * value;
        }
        float norm = (float)sqrt(sum);
        if (norm == 0.0f) {
            free(w);
            set_error(error, error_size, "zero weight-norm denominator");
            return NULL;
        }
        float scale = g[i] / norm;
        for (int j = 0; j < b * k; j++)
            w[(size_t)i * b * k + j] = v[(size_t)i * b * k + j] * scale;
    }
    /* transpose to [out][tap][in]: the checkpoint is [d0][d1][k] for both
     * conv ([out][in][k]) and convt ([in][out][k]) — only the destination
     * layout differs */
    float *out = malloc(sizeof(float) * (size_t)a * b * k);
    if (!out) {
        free(w);
        set_error(error, error_size, "out of memory");
        return NULL;
    }
    for (int i = 0; i < a; i++)
        for (int x = 0; x < b; x++)
            for (int y = 0; y < k; y++)
                out[is_convt ? ((size_t)x * k + y) * a + i
                             : ((size_t)i * k + y) * b + x] =
                    w[((size_t)i * b + x) * k + y];
    free(w);
    return out;
}

/* Upload resident weights and repack the conv weights into the GEMM
 * layout ([cout][k][cin] -> [k][cout_pad][cin]). Runs inside one
 * begin/submit window; the caller's chain must be drained. */
static int vae_upload_gpu(yue_vae *vae, char *error, size_t error_size) {
    h3_gpu *gpu = vae->gpu;
    /* If the caller already holds an open chain, our work merges onto it
     * and the tail submit below closes it — remember to re-open. */
    int was_open = !h3_gpu_begin(gpu);
    if (was_open && !h3_gpu_continue(gpu)) {
        set_error(error, error_size, "vae gpu begin: %s", h3_gpu_error(gpu));
        return 0;
    }
    int ok = 1;
    for (int i = 0; ok && i < vae->layer_count; i++) {
        yue_vae_layer *layer = &vae->layers[i];
        if (layer->kind == YUE_L_SNAKE) {
            layer->ea_t = h3_gpu_tensor_from_f32(gpu, layer->ea,
                                                 (size_t)layer->cin);
            layer->eb_t = layer->ea_t ?
                h3_gpu_tensor_from_f32(gpu, layer->eb, (size_t)layer->cin) :
                NULL;
            if (!layer->ea_t || !layer->eb_t) ok = 0;
            continue;
        }
        uint32_t cout_pad = ((uint32_t)layer->cout + 63u) & ~63u;
        layer->w_t = h3_gpu_tensor_new_f32(
            gpu, (size_t)layer->k * cout_pad * (size_t)layer->cin);
        h3_gpu_tensor *raw = layer->w_t ?
            h3_gpu_tensor_from_f32(gpu, layer->w,
                                   (size_t)layer->cout * layer->k *
                                       (size_t)layer->cin) : NULL;
        if (!layer->w_t || !raw ||
            !yv_gpu_conv_repack_f32(gpu, layer->w_t, raw,
                                    (uint32_t)layer->cin,
                                    (uint32_t)layer->cout, cout_pad,
                                    (uint32_t)layer->k))
            ok = 0;
        h3_gpu_tensor_free(raw);
        if (ok && layer->bias) {
            layer->bias_t = h3_gpu_tensor_from_f32(gpu, layer->bias,
                                                   (size_t)layer->cout);
            if (!layer->bias_t) ok = 0;
        }
    }
    if (ok && !h3_gpu_submit(gpu)) {
        set_error(error, error_size, "vae gpu upload: %s", h3_gpu_error(gpu));
        ok = 0;
    }
    if (!ok) {
        /* The chain is open with partially encoded uploads; drain it or
         * every later h3_gpu_begin on the shared context fails forever.
         * Also restore the caller's open chain on this path (the failed
         * submit above used to return before doing so). */
        (void)h3_gpu_submit(gpu);
        if (was_open) (void)h3_gpu_begin(gpu);
        if (error && error_size && !error[0])
            set_error(error, error_size, "vae gpu upload: %s",
                      h3_gpu_error(gpu));
        return 0;
    }
    if (was_open) (void)h3_gpu_begin(gpu);
    return 1;
}

yue_vae *yue_vae_create(const char *model_dir, h3_gpu *gpu, char *error,
                        size_t error_size) {
    if (!model_dir) {
        set_error(error, error_size, "invalid vae arguments");
        return NULL;
    }
    yue_vae *vae = calloc(1, sizeof(yue_vae));
    if (!vae) {
        set_error(error, error_size, "out of memory");
        return NULL;
    }
    vae->gpu = gpu;

    /* config.json decoder_config */
    char cfg_path[4096];
    snprintf(cfg_path, sizeof(cfg_path), "%s/config.json", model_dir);
    char *text = NULL;
    {
        FILE *file = fopen(cfg_path, "rb");
        if (!file) {
            yue_vae_free(vae);
            set_error(error, error_size, "missing %s", cfg_path);
            return NULL;
        }
        fseek(file, 0, SEEK_END);
        long size = ftell(file);
        fseek(file, 0, SEEK_SET);
        if (size < 0 || size > 1 << 20) {
            fclose(file);
            yue_vae_free(vae);
            set_error(error, error_size, "unreadable %s", cfg_path);
            return NULL;
        }
        text = malloc((size_t)size + 1);
        size_t got = text ? fread(text, 1, (size_t)size, file) : 0;
        fclose(file);
        if (!text || got != (size_t)size) {
            free(text);
            yue_vae_free(vae);
            set_error(error, error_size, "short read on %s", cfg_path);
            return NULL;
        }
        text[got] = '\0';
    }
    char jerr[256];
    yw_json *root = NULL;
    if (yw_json_parse(text, &root, jerr, sizeof(jerr)) == 0) {
        free(text);
        yue_vae_free(vae);
        set_error(error, error_size, "vae config parse: %s", jerr);
        return NULL;
    }
    free(text);
    const yw_json *cfg = yw_json_get(root, "decoder_config");
    if (!cfg) {
        yw_json_free(root);
        yue_vae_free(vae);
        set_error(error, error_size, "missing decoder_config");
        return NULL;
    }
    vae->channels = (int)yw_json_get_number(cfg, "channels", 64);
    vae->latent_dim = (int)yw_json_get_number(cfg, "latent_dim", 64);
    vae->out_channels = (int)yw_json_get_number(cfg, "out_channels", 2);
    vae->final_tanh = yw_json_get_bool(cfg, "final_tanh", 1);
    const yw_json *strides = yw_json_get(cfg, "strides");
    const yw_json *mults = yw_json_get(cfg, "c_mults");
    int n_strides = (int)yw_json_count(strides);
    int n_mults = (int)yw_json_count(mults);
    if (n_strides < 1 || n_strides > YUE_VAE_MAX_STRIDES ||
        n_mults != n_strides) {
        yw_json_free(root);
        yue_vae_free(vae);
        set_error(error, error_size, "bad decoder strides/c_mults");
        return NULL;
    }
    vae->blocks = n_strides;
    /* decoder walks the encoder strides in reverse */
    for (int i = 0; i < n_strides; i++)
        vae->strides[i] =
            (int)yw_json_number(yw_json_at(strides, n_strides - 1 - i));
    vae->c_mults[0] = 1;
    for (int i = 0; i < n_mults; i++)
        vae->c_mults[i + 1] = (int)yw_json_number(yw_json_at(mults, i));
    yw_json_free(root);

    /* Config values feed divisions, sizes and (size_t) casts. Reject
     * zero/negative/out-of-range values here instead of dividing by zero
     * or wrapping an allocation size further down the decoder. */
    if (vae->channels < 1 || vae->latent_dim < 1 ||
        vae->out_channels < 1) {
        yue_vae_free(vae);
        set_error(error, error_size, "bad decoder channel dimensions");
        return NULL;
    }
    for (int i = 0; i < n_strides; i++) {
        if (vae->strides[i] < 1 || vae->strides[i] > 64 ||
            vae->c_mults[i + 1] < 1) {
            yue_vae_free(vae);
            set_error(error, error_size, "bad decoder strides/c_mults values");
            return NULL;
        }
    }

    /* layer plan: conv_in, blocks (snake, convt, 3x[snake,conv,snake,conv]),
     * snake, conv_out */
    int max_layers = 1 + n_strides * (2 + 3 * 4) + 2;
    vae->layers = calloc((size_t)max_layers, sizeof(yue_vae_layer));
    if (!vae->layers) {
        yue_vae_free(vae);
        set_error(error, error_size, "out of memory");
        return NULL;
    }
    yue_vae_layer *L = vae->layers;
    int count = 0;
#define YUE_KEY(l, fmt, ...)                                                \
    do {                                                                    \
        snprintf(l.key, sizeof(l.key), fmt, __VA_ARGS__);                   \
    } while (0)
    L[count] = (yue_vae_layer){.kind = YUE_L_CONV,
                               .cin = vae->latent_dim,
                               .cout = vae->c_mults[n_strides] * vae->channels,
                               .k = 7,
                               .stride = 1,
                               .pad = 3,
                               .dilation = 1};
    YUE_KEY(L[count], "decoder.layers.%d", 0);
    count++;
    for (int i = n_strides; i >= 1; i--) {
        int cin = vae->c_mults[i] * vae->channels;
        int cout = vae->c_mults[i - 1] * vae->channels;
        int stride = vae->strides[n_strides - i];
        int block = n_strides - i + 1; /* checkpoint module index */
        L[count] = (yue_vae_layer){.kind = YUE_L_SNAKE, .cin = cin, .in_place = 1};
        YUE_KEY(L[count], "decoder.layers.%d.layers.0", block);
        count++;
        L[count] = (yue_vae_layer){.kind = YUE_L_CONVT,
                                   .cin = cin,
                                   .cout = cout,
                                   .k = 2 * stride,
                                   .stride = stride,
                                   .pad = (stride + 1) / 2,
                                   .dilation = 1};
        YUE_KEY(L[count], "decoder.layers.%d.layers.1", block);
        count++;
        static const int dilations[3] = {1, 3, 9};
        for (int di = 0; di < 3; di++) {
            int unit = 2 + di; /* res unit submodule index */
            int d = dilations[di];
            L[count] = (yue_vae_layer){.kind = YUE_L_SNAKE, .cin = cout,
                                       .out_of_place = 1};
            YUE_KEY(L[count], "decoder.layers.%d.layers.%d.layers.0", block,
                    unit);
            count++;
            L[count] = (yue_vae_layer){.kind = YUE_L_CONV,
                                       .cin = cout,
                                       .cout = cout,
                                       .k = 7,
                                       .stride = 1,
                                       .pad = 3 * d,
                                       .dilation = d};
            YUE_KEY(L[count], "decoder.layers.%d.layers.%d.layers.1", block,
                    unit);
            count++;
            L[count] = (yue_vae_layer){.kind = YUE_L_SNAKE, .cin = cout, .in_place = 1};
            YUE_KEY(L[count], "decoder.layers.%d.layers.%d.layers.2", block,
                    unit);
            count++;
            L[count] = (yue_vae_layer){.kind = YUE_L_CONV,
                                       .cin = cout,
                                       .cout = cout,
                                       .k = 1,
                                       .stride = 1,
                                       .pad = 0,
                                       .dilation = 1,
                                       .residual = 1};
            YUE_KEY(L[count], "decoder.layers.%d.layers.%d.layers.3", block,
                    unit);
            count++;
        }
    }
    L[count] = (yue_vae_layer){.kind = YUE_L_SNAKE,
                               .cin = vae->channels,
                               .in_place = 1};
    YUE_KEY(L[count], "decoder.layers.%d", n_strides + 1);
    count++;
    L[count] = (yue_vae_layer){.kind = YUE_L_CONV,
                               .cin = vae->channels,
                               .cout = vae->out_channels,
                               .k = 7,
                               .stride = 1,
                               .pad = 3,
                               .dilation = 1};
    YUE_KEY(L[count], "decoder.layers.%d", n_strides + 2);
    count++;
#undef YUE_KEY
    vae->layer_count = count;

    /* weights */
    char weights_path[4096];
    snprintf(weights_path, sizeof(weights_path), "%s/model.safetensors",
             model_dir);
    h3_st_header header;
    if (!h3_st_read_header(weights_path, &header, error, error_size)) {
        yue_vae_free(vae);
        return NULL;
    }
    char name[192];
    for (int i = 0; i < count; i++) {
        yue_vae_layer *layer = &L[i];
        if (layer->kind == YUE_L_SNAKE) {
            f32_buf alpha = {0}, beta = {0};
            /* read_tensor clamps to 8 dims before the rank check, so the
             * scratch must hold the worst case a corrupt header can
             * declare. */
            int shape[8], ndim;
            snprintf(name, sizeof(name), "%s.alpha", layer->key);
            if (!read_tensor(&header, name, &alpha, shape, &ndim, error,
                             error_size) ||
                ndim != 1 || shape[0] != layer->cin) {
                free(alpha.data); /* NULL when read_tensor already failed */
                alpha.data = NULL;
                h3_st_free_header(&header);
                yue_vae_free(vae);
                return NULL;
            }
            snprintf(name, sizeof(name), "%s.beta", layer->key);
            if (!read_tensor(&header, name, &beta, shape, &ndim, error,
                             error_size) ||
                ndim != 1 || shape[0] != layer->cin ||
                beta.count != (size_t)layer->cin ||
                alpha.count != (size_t)layer->cin) {
                free(alpha.data);
                free(beta.data);
                h3_st_free_header(&header);
                yue_vae_free(vae);
                return NULL;
            }
            layer->ea = alpha.data;
            layer->eb = beta.data;
            /* precompute exp(): the kernel consumes e^alpha / e^beta */
            for (int c = 0; c < layer->cin; c++) {
                alpha.data[c] = expf(alpha.data[c]);
                beta.data[c] = expf(beta.data[c]);
            }
            if (getenv("QWEN_VAE_DUMP") && i < 4) {
                char sp[4096];
                snprintf(sp, sizeof(sp), "%s/ce%02d.bin", getenv("QWEN_VAE_DUMP"), i);
                FILE *sf = fopen(sp, "wb");
                if (sf) {
                    fwrite(alpha.data, sizeof(float), (size_t)layer->cin, sf);
                    fwrite(beta.data, sizeof(float), (size_t)layer->cin, sf);
                    fclose(sf);
                }
            }
            continue;
        }
        f32_buf v = {0}, g = {0};
        int shape[8], ndim;
        int is_convt = layer->kind == YUE_L_CONVT;
        int a = is_convt ? layer->cin : layer->cout;
        int b = is_convt ? layer->cout : layer->cin;
        snprintf(name, sizeof(name), "%s.weight_v", layer->key);
        if (!read_tensor(&header, name, &v, shape, &ndim, error, error_size) ||
            ndim != 3 || shape[0] != a || shape[1] != b ||
            shape[2] != layer->k) {
            free(v.data);
            h3_st_free_header(&header);
            yue_vae_free(vae);
            set_error(error, error_size, "weight_v shape mismatch at %s", name);
            return NULL;
        }
        snprintf(name, sizeof(name), "%s.weight_g", layer->key);
        if (!read_tensor(&header, name, &g, shape, &ndim, error, error_size) ||
            ndim != 3 || shape[0] != a || shape[1] != 1 || shape[2] != 1) {
            free(g.data);
            free(v.data);
            h3_st_free_header(&header);
            yue_vae_free(vae);
            set_error(error, error_size, "weight_g shape mismatch at %s", name);
            return NULL;
        }
        layer->w = fold_weight(v.data, g.data, a, b, layer->k, is_convt, error,
                               error_size);
        free(v.data);
        free(g.data);
        if (!layer->w) {
            h3_st_free_header(&header);
            yue_vae_free(vae);
            return NULL;
        }
        if (getenv("QWEN_VAE_DUMP")) {
            char wpath[4096];
            snprintf(wpath, sizeof(wpath), "%s/cw%02d.bin", getenv("QWEN_VAE_DUMP"),
                     i);
            FILE *wf = fopen(wpath, "wb");
            if (wf) {
                fwrite(layer->w, sizeof(float),
                       (size_t)layer->cout * layer->k * layer->cin, wf);
                fclose(wf);
            }
        }
        /* bias (final conv has none) */
        snprintf(name, sizeof(name), "%s.bias", layer->key);
        const h3_st_tensor *bt = h3_st_find(&header, name);
        if (bt) {
            if (bt->dtype != H3_DTYPE_F32 ||
                (int)h3_st_tensor_elements(bt) != layer->cout) {
                h3_st_free_header(&header);
                yue_vae_free(vae);
                set_error(error, error_size, "bias mismatch at %s", name);
                return NULL;
            }
            layer->bias = malloc(sizeof(float) * (size_t)layer->cout);
            if (!layer->bias ||
                !h3_st_read_data(&header, bt, layer->bias,
                                 sizeof(float) * (size_t)layer->cout, error,
                                 error_size)) {
                h3_st_free_header(&header);
                yue_vae_free(vae);
                return NULL;
            }
        }
    }
    h3_st_free_header(&header);
    if (vae->gpu && !vae_upload_gpu(vae, error, error_size)) {
        yue_vae_free(vae);
        return NULL;
    }
    return vae;
}

void yue_vae_free(yue_vae *vae) {
    if (!vae) return;
    if (vae->layers) {
        for (int i = 0; i < vae->layer_count; i++) {
            free(vae->layers[i].w);
            free(vae->layers[i].bias);
            free(vae->layers[i].ea);
            free(vae->layers[i].eb);
            h3_gpu_tensor_free(vae->layers[i].w_t);
            h3_gpu_tensor_free(vae->layers[i].bias_t);
            h3_gpu_tensor_free(vae->layers[i].ea_t);
            h3_gpu_tensor_free(vae->layers[i].eb_t);
        }
        free(vae->layers);
    }
    free(vae);
}

/* ------------------------------------------------------------ activations */

static void snake_forward(float *x, size_t count, int channels,
                          const float *ea, const float *eb) {
    /* vForce variant: one scalar pass gathers z = x * ea[i % c], vvsinf
     * covers the whole buffer in a single vectorized call, and the final
     * combine keeps the scalar rounding points. Same fp32 ops as the
     * scalar loop except the sin itself (vvsinf vs sinf may differ by
     * 1 ulp on some inputs — inside the fixture noise band). Falls back
     * to the scalar loop if the scratch allocation fails. */
    /* vvsinf takes an int element count and scratch is 2*count floats;
     * fall back to the scalar loop if either would truncate or wrap. */
    float *scratch = (count > (size_t)2147483647 ||
                      count > ((size_t)-1) / (2 * sizeof(float)))
                         ? NULL : malloc(sizeof(float) * count * 2);
    if (!scratch) {
        for (size_t i = 0; i < count; i++) {
            int c = (int)(i % (size_t)channels);
            float value = x[i];
            float s = sinf(value * ea[c]);
            x[i] = value + s * s / (eb[c] + 1e-9f);
        }
        return;
    }
    float *z = scratch, *s = scratch + count;
    for (size_t i = 0; i < count; i++)
        z[i] = x[i] * ea[i % (size_t)channels];
    int n = (int)count;
    vvsinf(s, z, &n);
    for (size_t i = 0; i < count; i++) {
        float sv = s[i];
        x[i] += sv * sv / (eb[i % (size_t)channels] + 1e-9f);
    }
    free(scratch);
}

/* ------------------------------------------------------------ conv kernels */

/* out[t, oc] = bias + sum_j sum_ic x[t + j*dil - pad, ic] * w[oc][j][ic] */
static void conv1d_forward(const yue_vae_layer *layer, const float *x,
                           size_t t_in, float *out, size_t t_out) {
    const int cin = layer->cin, cout = layer->cout, k = layer->k;
    const int dil = layer->dilation, pad = layer->pad;
    /* init with bias */
    for (size_t t = 0; t < t_out; t++) {
        if (layer->bias)
            memcpy(out + t * (size_t)cout, layer->bias,
                   sizeof(float) * (size_t)cout);
        else
            memset(out + t * (size_t)cout, 0, sizeof(float) * (size_t)cout);
    }
    for (int j = 0; j < k; j++) {
        /* valid t: 0 <= t + j*dil - pad < t_in */
        long long lo = -(long long)(j * dil - pad);
        if (lo < 0) lo = 0;
        long long hi = (long long)t_in + (long long)(pad - j * dil);
        if (hi > (long long)t_out) hi = (long long)t_out;
        if (hi <= lo) continue;
        const float *a = x + (size_t)(j * dil - pad + lo) * cin;
        const float *wj = layer->w + (size_t)j * cin; /* [cout][cin] view */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)(hi - lo), cout, cin, 1.0f, a, dil * cin, wj,
                    k * cin, 1.0f, out + (size_t)lo * cout, cout);
    }
}

/* out[s*t + j - pad, oc] += x[t, ic] * w[oc][j][ic]; out pre-initialized
 * with bias. */
static void convt_forward(const yue_vae_layer *layer, const float *x,
                          size_t t_in, float *out, size_t t_out) {
    const int cin = layer->cin, cout = layer->cout;
    const int s = layer->stride, k = layer->k, pad = layer->pad;
    for (int j = 0; j < k; j++) {
        long long pos0 = (long long)j - pad; /* pos = s*t + pos0 */
        /* valid t: 0 <= s*t + pos0 < t_out */
        long long lo = 0;
        if (pos0 < 0) lo = (-pos0 + s - 1) / s;
        long long hi = t_in;
        if (pos0 < (long long)t_out)
            hi = (t_out - 1 - pos0) / s + 1;
        else
            hi = 0;
        if (hi > (long long)t_in) hi = t_in;
        if (hi <= lo) continue;
        const float *a = x + (size_t)lo * cin;
        const float *wj = layer->w + (size_t)j * cin;
        float *c = out + (size_t)(pos0 + (size_t)lo * s) * cout;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)(hi - lo), cout, cin, 1.0f, a, cin, wj, k * cin,
                    1.0f, c, s * cout);
    }
}

/* Dilated conv via phase sub-lattices: for each residue r mod dil, gather
 * the (contiguous) tap rows of that phase, sgemm, accumulate strided. */
static void conv1d_dilated(const yue_vae_layer *layer, const float *x,
                           size_t t_in, float *out, size_t t_out,
                           float *scratch) {
    const int cin = layer->cin, cout = layer->cout, k = layer->k;
    const int dil = layer->dilation, pad = layer->pad;
    for (size_t t = 0; t < t_out; t++) {
        if (layer->bias)
            memcpy(out + t * (size_t)cout, layer->bias,
                   sizeof(float) * (size_t)cout);
        else
            memset(out + t * (size_t)cout, 0, sizeof(float) * (size_t)cout);
    }
    for (int r = 0; r < dil; r++) {
        /* output rows t = r, r+dil, ... */
        size_t rows = (t_out - (size_t)r + (size_t)dil - 1) / (size_t)dil;
        for (int j = 0; j < k; j++) {
            /* gather A[i] = x[r + i*dil + j*dil - pad] into scratch */
            long long base = (long long)r + (long long)j * dil - pad;
            size_t lo = 0;
            if (base < 0) lo = (size_t)((-base + dil - 1) / dil);
            size_t hi = rows;
            /* last valid: r + i*dil + base_index < t_in */
            long long max_i = ((long long)t_in - 1 - base) / dil;
            if (max_i < 0) continue;
            if ((size_t)max_i + 1 < hi) hi = (size_t)max_i + 1;
            if (hi <= lo) continue;
            for (size_t i = lo; i < hi; i++)
                memcpy(scratch + (i - lo) * (size_t)cin,
                       x + (size_t)((long long)(r + i * dil) + (long long)j * dil - pad) *
                              cin,
                       sizeof(float) * (size_t)cin);
            const float *wj = layer->w + (size_t)j * cin;
            float *c = out + (size_t)(r + (size_t)lo * dil) * cout;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        (int)(hi - lo), cout, cin, 1.0f, scratch, cin, wj,
                        k * cin, 1.0f, c, dil * cout);
        }
    }
}

/* ------------------------------------------------------------------ decode */

size_t yue_vae_output_length(const yue_vae *vae, size_t frames) {
    size_t t = frames;
    for (int i = 0; i < vae->blocks; i++) {
        int s = vae->strides[i];
        t = t * (size_t)s - (size_t)(s % 2);
    }
    return t;
}

/* One full decoder pass over [t_in, latent_dim] latents -> [t_out, 2].
 * Buffer rotation: res units need the unit input alive for the residual add
 * while snake1/conv7/conv1 each write fresh buffers, so three slots rotate
 * (conv7 always takes cur+2). */
static int decoder_pass(const yue_vae *vae, const float *latents, size_t t_in,
                        float *audio, char *error, size_t error_size) {
    const float *x = latents;
    size_t t_cur = t_in;
    int c_cur = vae->latent_dim;
    float *bufs[3] = {NULL, NULL, NULL};
    size_t caps[3] = {0, 0, 0};
    int cur = -1; /* which buffer holds x (-1: external latents) */
    const float *res_in = NULL; /* res-unit input for the residual add */
    const char *dump_dir = getenv("QWEN_VAE_DUMP");
    int stage = 0;
    int ok = 1;

    for (int i = 0; ok && i < vae->layer_count; i++) {
        const yue_vae_layer *layer = &vae->layers[i];
        if (layer->kind == YUE_L_SNAKE && layer->in_place) {
            snake_forward((float *)x, t_cur * (size_t)c_cur, c_cur, layer->ea,
                          layer->eb);
        } else if (layer->kind == YUE_L_SNAKE) {
            /* out-of-place snake (res-unit head; input stays alive for the
             * residual add at the unit tail) */
            int dst = (cur + 1) % 3;
            size_t need = t_cur * (size_t)c_cur;
            if (caps[dst] < need) {
                free(bufs[dst]);
                bufs[dst] = malloc(sizeof(float) * need);
                if (!bufs[dst]) {
                    caps[dst] = 0;
                    set_error(error, error_size, "out of memory (vae buffers)");
                    ok = 0;
                    break;
                }
                caps[dst] = need;
            }
            memcpy(bufs[dst], x, sizeof(float) * need);
            snake_forward(bufs[dst], need, c_cur, layer->ea, layer->eb);
            res_in = x;
            x = bufs[dst];
            cur = dst;
        } else {
            size_t t_next = t_cur;
            if (layer->kind == YUE_L_CONVT)
                t_next = (t_cur - 1) * (size_t)layer->stride -
                         2 * (size_t)layer->pad + (size_t)layer->k;
            size_t need = t_next * (size_t)layer->cout;
            int dst = (cur + 1) % 3;
            if (layer->residual) dst = (cur + 2) % 3;
            if (caps[dst] < need) {
                free(bufs[dst]);
                bufs[dst] = malloc(sizeof(float) * need);
                if (!bufs[dst]) {
                    caps[dst] = 0;
                    set_error(error, error_size, "out of memory (vae buffers)");
                    ok = 0;
                    break;
                }
                caps[dst] = need;
            }
            float *outb = bufs[dst];
            if (layer->kind == YUE_L_CONVT) {
                memset(outb, 0, sizeof(float) * need);
                if (layer->bias)
                    for (size_t t = 0; t < t_next; t++)
                        memcpy(outb + t * (size_t)layer->cout, layer->bias,
                               sizeof(float) * (size_t)layer->cout);
                convt_forward(layer, x, t_cur, outb, t_next);
            } else if (layer->dilation > 1) {
                float *scratch =
                    malloc(sizeof(float) * t_next * (size_t)layer->cin);
                if (!scratch) {
                    set_error(error, error_size, "out of memory (vae scratch)");
                    ok = 0;
                    break;
                }
                conv1d_dilated(layer, x, t_cur, outb, t_next, scratch);
                free(scratch);
            } else {
                conv1d_forward(layer, x, t_cur, outb, t_next);
            }
            if (layer->residual) {
                /* res-unit tail: out += unit input */
                size_t elems = t_next * (size_t)layer->cout;
                for (size_t e = 0; e < elems; e++) outb[e] += res_in[e];
            }
            x = outb;
            cur = dst;
            t_cur = t_next;
            c_cur = layer->cout;
        }
        if (dump_dir && dump_dir[0] &&
            (layer->kind == YUE_L_SNAKE || layer->kind == YUE_L_CONVT ||
             layer->residual)) {
            char path[4096];
            snprintf(path, sizeof(path), "%s/stage%02d_t%zu_c%d.bin", dump_dir,
                     stage++, t_cur, c_cur);
            FILE *file = fopen(path, "wb");
            if (file) {
                fwrite(x, sizeof(float), t_cur * (size_t)c_cur, file);
                fclose(file);
            }
        }
    }
    if (!ok || c_cur != vae->out_channels) {
        if (ok && c_cur != vae->out_channels)
            set_error(error, error_size, "decoder channel mismatch");
        free(bufs[0]);
        free(bufs[1]);
        free(bufs[2]);
        return 0;
    }
    if (vae->final_tanh) {
        float *curb = (float *)x;
        for (size_t i = 0; i < t_cur * (size_t)c_cur; i++)
            curb[i] = tanhf(curb[i]);
    }
    memcpy(audio, x, sizeof(float) * t_cur * (size_t)c_cur);
    free(bufs[0]);
    free(bufs[1]);
    free(bufs[2]);
    return 1;
}

/* ---------------------------------------------------------- GPU decode */

static size_t vae_ceil64(size_t v) { return (v + 63) & ~(size_t)63; }

/* One full decoder pass on the GPU over [t_in, latent_dim] latents ->
 * [t_out, 2]. Same three-slot buffer rotation as decoder_pass; the
 * activations stay resident between layers and the whole pass encodes
 * into one command chain (submitted at the tail, or per dumped stage
 * when QWEN_VAE_DUMP is set). The zero ring is a shared scratch reused
 * by every convolution of the pass. */
/* Per-decode GPU scratch (three-slot rotation + conv ring), hoisted above
 * the tile loop. Tile inputs are a fixed set ({core+halo, core+2*halo}), so
 * after the first tile the caps never grow again and not a single Metal
 * buffer is allocated or freed for the rest of the song. Reallocating per
 * tile handed the driver pool a free-then-realloc cycle whose ~70 MB ring
 * sizes differ by a few KB per layer (t*cin), so every freed big block got
 * carved by a slightly-different-sized ring and each tile added ~135 MB of
 * never-reused pool regions — the resident-footprint ratchet
 * (~2 GB per song, measured). */
typedef struct {
    h3_gpu_tensor *bufs[3];
    size_t caps[3];
    h3_gpu_tensor *ring;
    size_t ring_cap;
} yv_gpu_scratch;

static void yv_gpu_scratch_free(yv_gpu_scratch *s) {
    for (int i = 0; i < 3; i++) {
        h3_gpu_tensor_free(s->bufs[i]);
        s->bufs[i] = NULL;
        s->caps[i] = 0;
    }
    h3_gpu_tensor_free(s->ring);
    s->ring = NULL;
    s->ring_cap = 0;
}

static int decoder_pass_gpu(const yue_vae *vae, const float *latents,
                            size_t t_in, float *audio, yv_gpu_scratch *sc,
                            char *error, size_t error_size) {
    h3_gpu *gpu = vae->gpu;
    const char *dump_dir = getenv("QWEN_VAE_DUMP");
    int stage = 0;
    int ok = 1;
    h3_gpu_tensor *lat = NULL;
    h3_gpu_tensor **bufs = sc->bufs;
    size_t *caps = sc->caps;
    h3_gpu_tensor *ring = sc->ring;
    size_t ring_cap = sc->ring_cap;
    h3_gpu_tensor *x = NULL;
    h3_gpu_tensor *res_in = NULL;
    size_t t_cur = t_in;
    int c_cur = vae->latent_dim;
    int cur = -1;

    int was_open = !h3_gpu_begin(gpu);
    if (was_open && !h3_gpu_continue(gpu)) {
        set_error(error, error_size, "vae decode begin: %s",
                  h3_gpu_error(gpu));
        return 0;
    }
    lat = h3_gpu_tensor_from_f32(gpu, latents,
                                 t_in * (size_t)vae->latent_dim);
    if (!lat) {
        set_error(error, error_size, "out of memory (vae latents upload)");
        ok = 0;
    }
    x = lat;

    for (int i = 0; ok && i < vae->layer_count; i++) {
        const yue_vae_layer *layer = &vae->layers[i];
        if (layer->kind == YUE_L_SNAKE && layer->in_place) {
            ok = yv_gpu_snake_f32(gpu, x, x, layer->ea_t, layer->eb_t,
                                  (uint32_t)(t_cur * (size_t)c_cur),
                                  (uint32_t)c_cur);
        } else if (layer->kind == YUE_L_SNAKE) {
            /* out-of-place snake (res-unit head; the input stays alive
             * for the residual add at the unit tail) */
            int dst = (cur + 1) % 3;
            size_t need = t_cur * (size_t)c_cur;
            if (caps[dst] < need) {
                h3_gpu_tensor_free(bufs[dst]);
                bufs[dst] = h3_gpu_tensor_new_f32(gpu, need);
                caps[dst] = bufs[dst] ? need : 0;
                if (!bufs[dst]) {
                    set_error(error, error_size,
                              "out of memory (vae gpu buffers)");
                    ok = 0;
                    break;
                }
            }
            ok = yv_gpu_snake_f32(gpu, bufs[dst], x, layer->ea_t,
                                  layer->eb_t, (uint32_t)need,
                                  (uint32_t)c_cur);
            if (ok) {
                res_in = x;
                x = bufs[dst];
                cur = dst;
            }
        } else {
            size_t t_next = t_cur;
            if (layer->kind == YUE_L_CONVT)
                t_next = (t_cur - 1) * (size_t)layer->stride -
                         2 * (size_t)layer->pad + (size_t)layer->k;
            size_t need = t_next * (size_t)layer->cout;
            int dst = layer->residual ? (cur + 2) % 3 : (cur + 1) % 3;
            if (caps[dst] < need) {
                h3_gpu_tensor_free(bufs[dst]);
                bufs[dst] = h3_gpu_tensor_new_f32(gpu, need);
                caps[dst] = bufs[dst] ? need : 0;
                if (!bufs[dst]) {
                    set_error(error, error_size,
                              "out of memory (vae gpu buffers)");
                    ok = 0;
                    break;
                }
            }
            h3_gpu_tensor *outb = bufs[dst];
            if (layer->kind == YUE_L_CONVT) {
                /* The kernel declares its A operand as (C, m_span) while the
                 * last M tile reads up to row lo+m_span-1; with pad <= stride
                 * lo can reach 1, so m_span must carry the stride slack or
                 * the tile slice exits the declared extent. The slack rows
                 * are allocated (and zeroed by pad_ring) and dropped by the
                 * kernel's write-back guard, so the result is unchanged. */
                size_t m_span = vae_ceil64(t_cur) + (size_t)layer->stride;
                size_t rneed = m_span * (size_t)layer->cin;
                if (getenv("YUE_ALLOC_TRACE"))
                    fprintf(stderr, "[ring-t] convt t=%zu cin=%d stride=%d"
                            " rneed=%zu\n", t_cur, layer->cin, layer->stride,
                            rneed);
                if (ring_cap < rneed) {
                    h3_gpu_tensor_free(ring);
                    ring = h3_gpu_tensor_new_f32(gpu, rneed);
                    ring_cap = ring ? rneed : 0;
                    sc->ring = ring;
                    sc->ring_cap = ring_cap;
                    if (!ring) {
                        set_error(error, error_size,
                                  "out of memory (vae gpu ring)");
                        ok = 0;
                        break;
                    }
                }
                ok = yv_gpu_pad_ring_f32(gpu, ring, x, (uint32_t)t_cur,
                                         (uint32_t)layer->cin, 0,
                                         (uint32_t)m_span) &&
                     yv_gpu_fill_bias_f32(gpu, outb, layer->bias_t,
                                          (uint32_t)t_next,
                                          (uint32_t)layer->cout);
                for (int j = 0; ok && j < layer->k; j++) {
                    long long pos0 = (long long)j - layer->pad;
                    long long lo = 0;
                    if (pos0 < 0)
                        lo = (-pos0 + layer->stride - 1) / layer->stride;
                    long long hi = (long long)t_cur;
                    if (pos0 < (long long)t_next)
                        hi = ((long long)t_next - 1 - pos0) / layer->stride + 1;
                    else
                        hi = 0;
                    if (hi > (long long)t_cur) hi = (long long)t_cur;
                    if (hi <= lo) continue;
                    ok = yv_gpu_convt_tap_f32(
                        gpu, outb, ring, layer->w_t, (uint32_t)j,
                        (uint32_t)layer->cin, (uint32_t)layer->cout,
                        (uint32_t)layer->stride, (uint32_t)(int32_t)pos0,
                        (uint32_t)lo, (uint32_t)(hi - lo),
                        (uint32_t)t_next, (uint32_t)m_span);
                }
            } else {
                size_t m_span = vae_ceil64(t_next);
                size_t rows = m_span + (size_t)(layer->k - 1) *
                                         (size_t)layer->dilation;
                size_t rneed = rows * (size_t)layer->cin;
                if (getenv("YUE_ALLOC_TRACE"))
                    fprintf(stderr, "[ring-c] conv t=%zu cin=%d k=%d dil=%d"
                            " rneed=%zu\n", t_next, layer->cin, layer->k,
                            layer->dilation, rneed);
                if (ring_cap < rneed) {
                    h3_gpu_tensor_free(ring);
                    ring = h3_gpu_tensor_new_f32(gpu, rneed);
                    ring_cap = ring ? rneed : 0;
                    sc->ring = ring;
                    sc->ring_cap = ring_cap;
                    if (!ring) {
                        set_error(error, error_size,
                                  "out of memory (vae gpu ring)");
                        ok = 0;
                        break;
                    }
                }
                ok = yv_gpu_pad_ring_f32(gpu, ring, x, (uint32_t)t_next,
                                         (uint32_t)layer->cin,
                                         (uint32_t)layer->pad,
                                         (uint32_t)rows) &&
                     yv_gpu_conv1d_mm_f32(
                         gpu, outb, ring, layer->w_t, layer->bias_t,
                         (uint32_t)t_next, (uint32_t)layer->cin,
                         (uint32_t)layer->cout, (uint32_t)layer->k,
                         (uint32_t)layer->dilation);
            }
            if (ok && layer->residual)
                ok = qw_gpu_vae_add_f32(gpu, outb, outb, res_in,
                                        (uint32_t)(t_next *
                                                   (size_t)layer->cout));
            if (ok) {
                x = outb;
                cur = dst;
                t_cur = t_next;
                c_cur = layer->cout;
            }
        }
        if (ok && dump_dir && dump_dir[0] &&
            (layer->kind == YUE_L_SNAKE || layer->kind == YUE_L_CONVT ||
             layer->residual)) {
            /* Dump stages settle the chain first, then re-open it. */
            if (!h3_gpu_submit(gpu)) {
                ok = 0;
            } else {
                size_t elems = t_cur * (size_t)c_cur;
                float *host = malloc(sizeof(float) * elems);
                if (host && h3_gpu_tensor_read_f32(x, host, elems)) {
                    char path[4096];
                    snprintf(path, sizeof(path), "%s/stage%02d_t%zu_c%d.bin",
                             dump_dir, stage++, t_cur, c_cur);
                    FILE *file = fopen(path, "wb");
                    if (file) {
                        fwrite(host, sizeof(float), elems, file);
                        fclose(file);
                    }
                }
                free(host);
                if (!h3_gpu_begin(gpu)) ok = 0;
            }
        }
    }

    if (ok) {
        if (vae->final_tanh)
            ok = yv_gpu_tanh_f32(gpu, x, x,
                                 (uint32_t)(t_cur * (size_t)c_cur));
        if (ok && c_cur != vae->out_channels) {
            set_error(error, error_size, "decoder channel mismatch");
            ok = 0;
        }
        if (ok && !h3_gpu_submit(gpu)) {
            set_error(error, error_size, "vae decode submit: %s",
                      h3_gpu_error(gpu));
            ok = 0;
        }
        if (ok &&
            !h3_gpu_tensor_read_f32(x, audio, t_cur * (size_t)c_cur)) {
            set_error(error, error_size, "vae audio readback failed");
            ok = 0;
        }
    }
    if (!ok) {
        /* Also covers failures that flipped ok inside the block above
         * (tanh / channel mismatch / submit): the chain is still open and
         * must be drained before the caller releases its scratch tensors. */
        (void)h3_gpu_submit(gpu);
    }
    if (was_open) {
        /* The submits above closed the caller's chain too (our work was
         * encoded onto it); restore the open state the caller expects. */
        (void)h3_gpu_begin(gpu);
    }
    /* bufs/ring live in the caller's scratch (hoisted across tiles). */
    h3_gpu_tensor_free(lat);
    return ok;
}

int yue_vae_decode(const yue_vae *vae, const float *latents, size_t frames,
                   float *audio, char *error, size_t error_size) {
    if (!vae || !latents || !frames || !audio) {
        set_error(error, error_size, "invalid decode arguments");
        return 0;
    }
    /* YUE_VAE_CPU forces the reference path; "=0" opts back into GPU. */
    const char *cpu_env = getenv("YUE_VAE_CPU");
    int use_gpu = vae->gpu && !(cpu_env && *cpu_env &&
                                strcmp(cpu_env, "0") != 0);
    size_t ratio = 1;
    for (int i = 0; i < vae->blocks; i++)
        ratio *= (size_t)vae->strides[i];
    const size_t total = yue_vae_output_length(vae, frames);
    const size_t core = YUE_VAE_CORE_FRAMES;
    const size_t halo = YUE_VAE_HALO_FRAMES;

    size_t tile_latents = core + 2 * halo;
    size_t tile_out = yue_vae_output_length(vae, tile_latents);
    float *tile = malloc(sizeof(float) * tile_out * (size_t)vae->out_channels);
    if (!tile) {
        set_error(error, error_size, "out of memory (vae tile)");
        return 0;
    }
    /* Tile walk with a fixed input-length set. A naive last tile carries a
     * per-request frame count (song length drifts a few frames with
     * temperature sampling), handing the Metal driver pool a fresh set of
     * mid-network buffer sizes on every song — the resident-footprint
     * ratchet. Widening the last tile back to exactly `core` frames (the
     * overlap decodes the same crop from a tile whose halo still covers
     * the receptive field) keeps every tile input at core+halo or
     * core+2*halo frames for any song longer than one wide tile. Songs up
     * to one wide tile decode a single exact-length tile: the two-tile
     * walk would decode the whole clip twice (the widened second tile's
     * left halo clamps to 0 and re-covers [0, frames)). Crops are
     * unchanged, so results are bit-identical. */
    const size_t step = frames <= core + 2 * halo ? frames : core;
    size_t start = 0;
    /* GPU scratch hoisted across tiles (see yv_gpu_scratch): with the fixed
     * tile-input set the caps saturate on the first tile and the pool sees
     * zero further churn for the rest of the song. */
    yv_gpu_scratch scratch = {{NULL, NULL, NULL}, {0, 0, 0}, NULL, 0};
    while (start < frames) {
        size_t end = start + step < frames ? start + step : frames;
        if (end == frames && end - start < core && start > 0)
            start = frames - core; /* widen the last tile */
        end = start + step < frames ? start + step : frames;
        size_t left = start > halo ? start - halo : 0;
        size_t right = end + halo < frames ? end + halo : frames;
        const float *tile_in = latents + left * (size_t)vae->latent_dim;
        if ((use_gpu ? decoder_pass_gpu(vae, tile_in, right - left, tile,
                                        &scratch, error, error_size)
                     : decoder_pass(vae, tile_in, right - left, tile, error,
                                    error_size)) == 0) {
            yv_gpu_scratch_free(&scratch);
            free(tile);
            return 0;
        }
        size_t offset = (start - left) * ratio;
        size_t length = (end * ratio < total ? end * ratio : total) -
                        start * ratio;
        memcpy(audio + start * ratio * (size_t)vae->out_channels,
               tile + offset * (size_t)vae->out_channels,
               sizeof(float) * length * (size_t)vae->out_channels);
        start = end;
    }
    yv_gpu_scratch_free(&scratch);
    free(tile);
    return 1;
}

/* ------------------------------------------------------------- wav / flac */

int yue_write_wav24(const char *path, const float *samples, size_t frames,
                    int channels, int rate, char *error, size_t error_size) {
    if (!path || !samples || frames < 1 || channels < 1) {
        set_error(error, error_size, "invalid wav arguments");
        return 0;
    }
    /* Compute the RIFF size before opening the file: frames*channels*3
     * must not wrap size_t (a wrapped value would slip past the 4 GiB
     * guard and produce a truncated file with a bogus header). */
    if (frames > ((size_t)-1) / 3 / (size_t)channels) {
        set_error(error, error_size, "wav too large for a RIFF header");
        return 0;
    }
    size_t data_bytes = frames * (size_t)channels * 3;
    if (data_bytes > 0xFFFFFFFFu - 44) {
        set_error(error, error_size, "wav too large for a RIFF header");
        return 0;
    }
    FILE *file = fopen(path, "wb");
    if (!file) {
        set_error(error, error_size, "cannot open %s", path);
        return 0;
    }
    unsigned char header[44];
    unsigned int byte_rate = (unsigned int)(rate * channels * 3);
    unsigned int chunk = (unsigned int)(36 + data_bytes);
    memcpy(header, "RIFF", 4);
    memcpy(header + 4, &chunk, 4);
    memcpy(header + 8, "WAVE", 4);
    memcpy(header + 12, "fmt ", 4);
    unsigned int fmt_size = 16;
    unsigned short fmt_tag = 1, chans = (unsigned short)channels;
    unsigned short block = (unsigned short)(channels * 3);
    unsigned short bits = 24;
    unsigned int rate32 = (unsigned int)rate;
    memcpy(header + 16, &fmt_size, 4);
    memcpy(header + 20, &fmt_tag, 2);
    memcpy(header + 22, &chans, 2);
    memcpy(header + 24, &rate32, 4);
    memcpy(header + 28, &byte_rate, 4);
    memcpy(header + 32, &block, 2);
    memcpy(header + 34, &bits, 2);
    memcpy(header + 36, "data", 4);
    unsigned int data32 = (unsigned int)data_bytes;
    memcpy(header + 40, &data32, 4);
    if (fwrite(header, 1, 44, file) != 44) {
        fclose(file);
        set_error(error, error_size, "wav header write failed");
        return 0;
    }
    /* clamp + 24-bit little-endian conversion */
    unsigned char row[4096];
    size_t pending = 0;
    for (size_t i = 0; i < frames * (size_t)channels; i++) {
        float value = samples[i];
        if (value > 1.0f) value = 1.0f;
        if (value < -1.0f) value = -1.0f;
        long long q = (long long)llrintf(value * 8388607.0f);
        unsigned int u = (unsigned int)(q & 0xFFFFFF);
        if (pending + 3 > sizeof(row)) {
            if (fwrite(row, 1, pending, file) != pending) {
                fclose(file);
                set_error(error, error_size, "wav write failed");
                return 0;
            }
            pending = 0;
        }
        row[pending++] = (unsigned char)(u & 0xFF);
        row[pending++] = (unsigned char)((u >> 8) & 0xFF);
        row[pending++] = (unsigned char)((u >> 16) & 0xFF);
    }
    if (pending && fwrite(row, 1, pending, file) != pending) {
        fclose(file);
        set_error(error, error_size, "wav write failed");
        return 0;
    }
    if (fclose(file) != 0) {
        set_error(error, error_size, "wav close failed");
        return 0;
    }
    return 1;
}

/* ffmpeg parses a leading '-' as an option (argv is not a shell): prefix
 * such relative paths with "./" (same file, same cwd). */
static char *dash_safe_path(const char *path) {
    if (!path || path[0] != '-') return NULL;
    size_t n = strlen(path);
    char *normalized = malloc(n + 3);
    if (!normalized) return NULL;
    normalized[0] = '.';
    normalized[1] = '/';
    memcpy(normalized + 2, path, n + 1);
    return normalized;
}

int yue_wav_to_flac(const char *wav_path, const char *flac_path, char *error,
                    size_t error_size) {
    if (!wav_path || !flac_path) {
        set_error(error, error_size, "invalid flac arguments");
        return 0;
    }
    /* posix_spawnp with an argument array (same discipline as h3_ffmpeg.c):
     * paths are passed verbatim, so quotes or shell metacharacters in the
     * output filename can neither break the command nor inject code. Both
     * arguments are dash-normalized: the scratch WAV derives from the
     * output path, so a leading '-' would otherwise be parsed as an option
     * on "-i" too (the output side alone was normalized before). */
    const char *override = getenv("H3_FFMPEG");
    const char *program = override && *override ? override : "ffmpeg";
    char *normalized_in = dash_safe_path(wav_path);
    char *normalized_out = dash_safe_path(flac_path);
    if ((wav_path[0] == '-' && !normalized_in) ||
        (flac_path[0] == '-' && !normalized_out)) {
        set_error(error, error_size, "out of memory (flac path)");
        free(normalized_in);
        free(normalized_out);
        return 0;
    }
    const char *in_arg = normalized_in ? normalized_in : wav_path;
    const char *out_arg = normalized_out ? normalized_out : flac_path;
    char *arguments[] = {(char *)program, "-y", "-loglevel", "error", "-i",
                         (char *)in_arg, "-c:a", "flac",
                         (char *)out_arg, NULL};
    pid_t child = -1;
    int code = posix_spawnp(&child, program, NULL, NULL, arguments, environ);
    if (code != 0) {
        set_error(error, error_size, "cannot start ffmpeg: %s",
                  strerror(code));
        free(normalized_in);
        free(normalized_out);
        return 0;
    }
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        set_error(error, error_size, "cannot wait for ffmpeg: %s",
                  strerror(errno));
        free(normalized_in);
        free(normalized_out);
        return 0;
    }
    free(normalized_in);
    free(normalized_out);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        set_error(error, error_size, "ffmpeg flac encode failed (%s)",
                  WIFEXITED(status)
                      ? "check input/output paths"
                      : "killed by signal");
        return 0;
    }
    return 1;
}
