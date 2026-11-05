/* lyra/yue2 protocol layer: generation config, prompt construction, and
 * acoustic chunking. Numeric/token semantics mirror
 * vendor/yue/src/yue2/{protocol,tokenization_yue2}.py exactly. */
#include "yue_protocol.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Instruction templates keyed by cot mode (protocol INSTRUCTIONS). */
const char *const yue_instructions[3] = {
    [YUE_COT_FULL] =
        "Generate a chord-annotated ABC transcription, then generate music "
        "with codec tokens from the given conditions.",
    [YUE_COT_MELODY] =
        "Generate a melody-only ABC transcription without chord symbols, then "
        "generate music with codec tokens from the given conditions.",
    [YUE_COT_OFF] =
        "Generate music with codec tokens from the given conditions.",
};

/* ------------------------------------------------------------ defaults */

static void sampling_defaults(yue_sampling *s, double temperature,
                              double top_p, int top_k,
                              double repetition_penalty, int penalty_window,
                              int min_tokens, int max_tokens) {
    s->temperature = temperature;
    s->top_p = top_p;
    s->top_k = top_k;
    s->repetition_penalty = repetition_penalty;
    s->penalty_window = penalty_window;
    s->min_tokens = min_tokens;
    s->max_tokens = max_tokens;
}

/* Applies the shipped defaults; private helper (no header declaration). */
static void yue_generation_config_defaults(yue_generation_config *cfg) {
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    sampling_defaults(&cfg->abc, 0.7, 0.9, 30, 1.005, 100, 32, 4096);
    sampling_defaults(&cfg->semantic, 1.0, 0.95, 100, 1.2, 50, 200, 9000);
    cfg->ode_steps = 32;
    cfg->context = (int)YUE_CONTEXT;
}

/* double -> int with an integer/range check first: casting an
 * out-of-int-range double is undefined behaviour and could wrap into the
 * valid band, sneaking past check_sampling. Out-of-range members are
 * ignored (defaults kept), matching the wrong-JSON-type policy; the
 * precise error still comes from check_sampling at generation time. */
static int json_int_in_range(double v, int lo, int hi, int *out) {
    if (v != floor(v) || v < (double)lo || v > (double)hi) return 0;
    *out = (int)v;
    return 1;
}

/* Overlay a config JSON object. Members of the wrong JSON type are ignored
 * (keeping the defaults) instead of coerced: yw_json_number() returns 0.0
 * for non-numbers, which would silently turn a mistyped "temperature":
 * "0.7" into greedy decoding. */
static void overlay_sampling(yue_sampling *target, const yw_json *value) {
    if (!value || yw_json_type(value) != yue_json_OBJECT) return;
    const yw_json *member;
    if ((member = yw_json_get(value, "temperature")) &&
        yw_json_type(member) == yue_json_NUMBER)
        target->temperature = yw_json_number(member);
    if ((member = yw_json_get(value, "top_p")) &&
        yw_json_type(member) == yue_json_NUMBER)
        target->top_p = yw_json_number(member);
    if ((member = yw_json_get(value, "top_k")) &&
        yw_json_type(member) == yue_json_NUMBER) {
        int v;
        if (json_int_in_range(yw_json_number(member), 1, 1 << 20, &v))
            target->top_k = v;
    }
    if ((member = yw_json_get(value, "repetition_penalty")) &&
        yw_json_type(member) == yue_json_NUMBER)
        target->repetition_penalty = yw_json_number(member);
    if ((member = yw_json_get(value, "penalty_window")) &&
        yw_json_type(member) == yue_json_NUMBER) {
        int v;
        /* Keep the accepted range aligned with check_sampling()'s
         * YUE_MAX_WINDOW gate, otherwise generation only reports a vague
         * "invalid sampling parameters" later. */
        if (json_int_in_range(yw_json_number(member), 1, YUE_MAX_WINDOW, &v))
            target->penalty_window = v;
    }
    if ((member = yw_json_get(value, "min_tokens")) &&
        yw_json_type(member) == yue_json_NUMBER) {
        int v;
        if (json_int_in_range(yw_json_number(member), 0, 1 << 20, &v))
            target->min_tokens = v;
    }
    if ((member = yw_json_get(value, "max_tokens")) &&
        yw_json_type(member) == yue_json_NUMBER) {
        int v;
        if (json_int_in_range(yw_json_number(member), 1, 1 << 20, &v))
            target->max_tokens = v;
    }
}

int yue_generation_config_load(const char *model_dir,
                               yue_generation_config *cfg,
                               char *error, size_t error_size) {
    yue_generation_config_defaults(cfg);
    if (!cfg) return 0;
    if (!model_dir) return 1;

    char path[1024];
    int written = snprintf(path, sizeof(path),
                           "%s/yue2_generation_config.json", model_dir);
    if (written < 0 || (size_t)written >= sizeof(path)) {
        if (error && error_size)
            snprintf(error, error_size, "generation config path too long");
        return 0;
    }
    FILE *handle = fopen(path, "rb");
    if (!handle) return 1; /* absent file keeps defaults */

    if (error && error_size) error[0] = '\0';
    fseek(handle, 0, SEEK_END);
    long size = ftell(handle);
    fseek(handle, 0, SEEK_SET);
    if (size < 0 || size > 1 << 20) {
        if (error && error_size)
            snprintf(error, error_size, "generation config size invalid");
        fclose(handle);
        return 0;
    }
    char *text = malloc((size_t)size + 1);
    if (!text) {
        fclose(handle);
        return 0;
    }
    size_t got = fread(text, 1, (size_t)size, handle);
    fclose(handle);
    if (got != (size_t)size) {
        free(text);
        if (error && error_size)
            snprintf(error, error_size, "generation config short read");
        return 0;
    }
    text[got] = '\0';

    yw_json *root = NULL;
    char parse_error[YUE_ERROR_SIZE];
    if (yw_json_parse(text, &root, parse_error, sizeof(parse_error)) == 0) {
        if (error && error_size)
            snprintf(error, error_size, "generation config parse: %s",
                     parse_error);
        free(text);
        return 0;
    }
    free(text);
    overlay_sampling(&cfg->abc, yw_json_get(root, "abc"));
    overlay_sampling(&cfg->semantic, yw_json_get(root, "semantic"));
    const yw_json *steps = yw_json_get(root, "ode_steps");
    if (steps && yw_json_type(steps) == yue_json_NUMBER) {
        int value;
        if (!json_int_in_range(yw_json_number(steps), 1, 1024, &value)) {
            if (error && error_size)
                snprintf(error, error_size,
                         "ode_steps must be in [1, 1024]");
            yw_json_free(root);
            return 0;
        }
        cfg->ode_steps = value;
    }
    yw_json_free(root);
    return 1;
}

/* ------------------------------------------------------------- requests */

int yue_cot_index(const char *cot) {
    if (!cot) return -1;
    if (!strcmp(cot, "full")) return YUE_COT_FULL;
    if (!strcmp(cot, "melody")) return YUE_COT_MELODY;
    if (!strcmp(cot, "off")) return YUE_COT_OFF;
    return -1;
}

double yue_request_guidance(int cot, double cfg_scale, int has_cfg_scale) {
    if (has_cfg_scale && isfinite(cfg_scale) && cfg_scale >= 0.0 &&
        cfg_scale <= 20.0)
        return cfg_scale;
    return cot == YUE_COT_OFF ? 1.01 : 1.0;
}

/* --------------------------------------------------- prompt construction */

static int append_token(uint32_t **buffer, size_t *count, size_t *capacity,
                        uint32_t token, char *error, size_t error_size) {
    if (*count == *capacity) {
        size_t grown = *capacity ? *capacity * 2 : 256;
        uint32_t *moved = realloc(*buffer, grown * sizeof(*moved));
        if (!moved) {
            if (error && error_size)
                snprintf(error, error_size, "out of memory building prefix");
            return 0;
        }
        *buffer = moved;
        *capacity = grown;
    }
    (*buffer)[(*count)++] = token;
    return 1;
}

/* Compose "{instructions}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n" and encode
 * it (lyra SongRequest.text). */
static int encode_song_text(const h3_tokenizer *tokenizer, const char *style,
                            const char *lyrics, int cot, uint32_t **out,
                            size_t *count, char *error, size_t error_size) {
    size_t length = strlen(yue_instructions[cot]) + strlen(style) +
                    strlen(lyrics) + 32;
    char *text = malloc(length);
    if (!text) {
        if (error && error_size)
            snprintf(error, error_size, "out of memory composing prompt");
        return 0;
    }
    snprintf(text, length, "%s\n[Tags]\n%s\n[Lyrics]\n%s\n",
             yue_instructions[cot], style, lyrics);
    int ok = h3_tokenizer_encode(tokenizer, text, 0, out, count, error,
                                 error_size) != 0;
    free(text);
    return ok;
}

static int prefixes_common(const h3_tokenizer *tokenizer, const char *style,
                           const char *lyrics, int cot,
                           const uint32_t *abc_ids, size_t abc_count,
                           int negative, uint32_t **out, size_t *count,
                           char *error, size_t error_size) {
    *out = NULL;
    *count = 0;
    if (cot < YUE_COT_FULL || cot > YUE_COT_OFF) {
        if (error && error_size)
            snprintf(error, error_size, "cot mode %d is invalid", cot);
        return 0;
    }
    if (!style || !lyrics) {
        if (error && error_size)
            snprintf(error, error_size, "style and lyrics are required");
        return 0;
    }
    size_t capacity = 0;
    uint32_t *buffer = NULL;
    if (!append_token(&buffer, count, &capacity, YUE_EOD, error, error_size))
        return 0;

    if (negative) {
        uint32_t *ids = NULL;
        size_t ids_count = 0;
        if (!h3_tokenizer_encode(tokenizer, yue_instructions[cot], 0, &ids,
                                 &ids_count, error, error_size)) {
            free(buffer);
            return 0;
        }
        for (size_t i = 0; i < ids_count; i++)
            if (!append_token(&buffer, count, &capacity, ids[i], error,
                              error_size)) {
                free(ids);
                free(buffer);
                return 0;
            }
        free(ids);
    } else {
        uint32_t *ids = NULL;
        size_t ids_count = 0;
        if (!encode_song_text(tokenizer, style, lyrics, cot, &ids, &ids_count,
                              error, error_size)) {
            free(buffer);
            return 0;
        }
        for (size_t i = 0; i < ids_count; i++)
            if (!append_token(&buffer, count, &capacity, ids[i], error,
                              error_size)) {
                free(ids);
                free(buffer);
                return 0;
            }
        free(ids);
    }

    int ok = 1;
    if (cot == YUE_COT_OFF) {
        if (negative) {
            ok = append_token(&buffer, count, &capacity, YUE_MUSIC_START,
                              error, error_size);
        } else {
            ok = append_token(&buffer, count, &capacity, YUE_ABC_START, error,
                              error_size) &&
                 append_token(&buffer, count, &capacity, YUE_ABC_END, error,
                              error_size) &&
                 append_token(&buffer, count, &capacity, YUE_MUSIC_START,
                              error, error_size);
        }
    } else if (abc_ids && abc_count) {
        ok = append_token(&buffer, count, &capacity, YUE_ABC_START, error,
                          error_size);
        for (size_t i = 0; ok && i < abc_count; i++) {
            if (abc_ids[i] >= YUE_EOD) {
                if (error && error_size)
                    snprintf(error, error_size, "abc token id out of range");
                ok = 0;
                break;
            }
            ok = append_token(&buffer, count, &capacity, abc_ids[i], error,
                              error_size);
        }
        if (ok)
            ok = append_token(&buffer, count, &capacity, YUE_ABC_END, error,
                               error_size) &&
                 append_token(&buffer, count, &capacity, YUE_MUSIC_START,
                              error, error_size);
    } else {
        /* planning prefix ends at ABC_START; negative without a plan is
         * rejected by the caller (symbolic CFG requires exact ABC ids). */
        ok = append_token(&buffer, count, &capacity, YUE_ABC_START, error,
                          error_size);
    }
    if (!ok) {
        free(buffer);
        return 0;
    }
    *out = buffer;
    return 1;
}

int yue_token_prefixes(const h3_tokenizer *tokenizer, const char *style,
                       const char *lyrics, int cot, const uint32_t *abc_ids,
                       size_t abc_count, uint32_t **out, size_t *count,
                       char *error, size_t error_size) {
    return prefixes_common(tokenizer, style, lyrics, cot, abc_ids, abc_count,
                           0, out, count, error, error_size);
}

int yue_negative_prefix(const h3_tokenizer *tokenizer, const char *style,
                        const char *lyrics, int cot, const uint32_t *abc_ids,
                        size_t abc_count, uint32_t **out, size_t *count,
                        char *error, size_t error_size) {
    if (cot != YUE_COT_OFF && !(abc_ids && abc_count)) {
        if (error && error_size)
            snprintf(error, error_size,
                     "symbolic CFG requires the exact positive ABC ids");
        return 0;
    }
    return prefixes_common(tokenizer, style, lyrics, cot, abc_ids, abc_count,
                           1, out, count, error, error_size);
}

/* ------------------------------------------------------ acoustic chunks */

int yue_chunk_ranges(size_t frames, size_t prefix_tokens, size_t context,
                     yue_chunk_range *out, size_t capacity, size_t *count,
                     char *error, size_t error_size) {
    *count = 0;
    if (frames < 1 || prefix_tokens >= context ||
        context - prefix_tokens <= 3) {
        if (error && error_size)
            snprintf(error, error_size,
                     "empty codec or prefix leaves no acoustic context");
        return 0;
    }
    size_t size = (context - prefix_tokens - 3) / 2;
    if (size > context) size = context;
    if (size < 1) {
        if (error && error_size)
            snprintf(error, error_size, "acoustic chunk size underflow");
        return 0;
    }
    /* Avoid frames + size - 1 (overflow) and bound the write loop by both
     * the computed count and the caller's capacity. */
    size_t total = frames / size + (frames % size != 0);
    if (total > capacity) {
        if (error && error_size)
            snprintf(error, error_size, "chunk range capacity too small");
        return 0;
    }
    for (size_t index = 0; index < total && index < capacity; index++) {
        size_t start = index * size;
        out[index].start = start;
        out[index].end = size > frames - start ? frames : start + size;
    }
    *count = total;
    return 1;
}
