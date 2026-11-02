#ifndef YUE_PROTOCOL_H
#define YUE_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

#include "h3_tokenizer.h"
#include "yue.h"
#include "yue_json.h"

/* ----------------------------------------------------- generation config */

typedef struct {
    double temperature;
    double top_p;
    int top_k;
    double repetition_penalty;
    int penalty_window;
    int min_tokens;
    int max_tokens;
} yue_sampling;

typedef struct {
    yue_sampling abc;     /* t.7 / p.9 / k30 / rep1.005 / win100 / 32..4096 */
    yue_sampling semantic; /* t1.0 / p.95 / k100 / rep1.2 / win50 / 200..9000 */
    int ode_steps;        /* 32 */
    int context;          /* fixed YUE_CONTEXT */
} yue_generation_config;

/* Fill cfg with the lyra defaults, then overlay
 * <model_dir>/yue2_generation_config.json when present. */
int yue_generation_config_load(const char *model_dir,
                               yue_generation_config *cfg,
                               char *error, size_t error_size);

/* ------------------------------------------------------------- requests */

int yue_cot_index(const char *cot); /* YUE_COT_* or -1 */

/* Effective CFG per lyra SongRequest.guidance: explicit scale wins, else
 * 1.01 for cot=off and 1.0 otherwise. */
double yue_request_guidance(int cot, double cfg_scale, int has_cfg_scale);

/* --------------------------------------------------- prompt construction */

/* token_prefixes(): [EOD] + encode(instructions[cot] + tags + lyrics)
 * then the ABC/codec framing. abc_ids is the plan's ABC token ids
 * (NULL/0 while planning: the prefix ends at ABC_START). Caller frees. */
int yue_token_prefixes(const h3_tokenizer *tokenizer, const char *style,
                       const char *lyrics, int cot, const uint32_t *abc_ids,
                       size_t abc_count, uint32_t **out, size_t *count,
                       char *error, size_t error_size);

/* negative_prefix(): instruction-only positive framing for CFG. For
 * cot=off: [EOD] + encode(instructions) + [MUSIC_START]; otherwise the
 * exact plan ABC ids are retained. Caller frees. */
int yue_negative_prefix(const h3_tokenizer *tokenizer, const char *style,
                        const char *lyrics, int cot, const uint32_t *abc_ids,
                        size_t abc_count, uint32_t **out, size_t *count,
                        char *error, size_t error_size);

/* ------------------------------------------------------ acoustic chunks */

typedef struct {
    size_t start;
    size_t end; /* exclusive */
} yue_chunk_range;

/* chunk_ranges(): size = min((context - prefix_tokens - 3) / 2, CONTEXT).
 * Returns the number of ranges written (<= *capacity); fails when empty. */
int yue_chunk_ranges(size_t frames, size_t prefix_tokens, size_t context,
                     yue_chunk_range *out, size_t capacity, size_t *count,
                     char *error, size_t error_size);

#endif
