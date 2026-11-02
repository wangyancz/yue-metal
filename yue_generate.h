#ifndef YUE_GENERATE_H
#define YUE_GENERATE_H

#include <stddef.h>
#include <stdint.h>

#include "yue_ar.h"
#include "yue_sampling.h"

/* AR token generation: lyra ar.py generate_tokens() as a C loop. Prefills
 * the phase prefix (and the negative branch when CFG is active), then
 * decodes with host-side sampling until the phase end token or the
 * max-token budget. The callback receives every sampled token INCLUDING
 * the end token, mirroring the production emit contract. */

typedef void (*yue_token_fn)(yue_phase phase, uint32_t token, void *ctx);

typedef struct {
    uint32_t *tokens; /* content ids in native space (end token excluded) */
    size_t count;
    size_t capacity; /* internal growth state */
    int eos;       /* 1 when the end token was sampled */
    int truncated; /* 1 when max_tokens ran out without eos */
    /* Greedy diagnostics: per-step margin between the top-2 scores. Two
     * bf16 implementations diverge on near-ties (margin within the bf16
     * intrinsic noise band), so parity tests accept token flips only at
     * small margins. Sampled steps (temperature > 0) record no margins. */
    double *margins;
    size_t margin_count;
} yue_generation;

void yue_generation_free(yue_generation *gen);

int yue_ar_generate(yue_ar *ar, const uint32_t *prefix, size_t prefix_len,
                    const uint32_t *negative, size_t negative_len,
                    double cfg_scale, yue_phase phase, const yue_sampling *sp,
                    uint64_t seed, int legacy, yue_token_fn on_token,
                    void *ctx, yue_generation **out, char *error,
                    size_t error_size);

#endif
