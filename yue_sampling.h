#ifndef YUE_SAMPLING_H
#define YUE_SAMPLING_H

#include <stddef.h>
#include <stdint.h>

#include "yue.h"
#include "yue_protocol.h"

/* Host-side YuE2 sampling, mirroring lyra/sampling.py operation for
 * operation. Two numeric paths exist in production:
 *   - fp32 ("legacy_off=0"): abc phase and cot=melody/full semantic.
 *   - bf16 ("legacy_off=1"): cot=off semantic keeps the historical BF16
 *     arithmetic; every op rounds to bf16 (round-to-nearest-even) and
 *     top-p always keeps the first three candidates.
 * The RNG is engine-local by design (M0 decision): a splitmix64 stream
 * seeded per generate_tokens call, replacing MLX's threefry categorical.
 * Only greedy (temperature=0) output is expected to match lyra exactly. */

typedef enum {
    YUE_PHASE_ABC = 0,
    YUE_PHASE_SEMANTIC = 1
} yue_phase;

/* Phase-local projection width: EOD+1 for abc, CODEC_SIZE+1 for semantic. */
size_t yue_phase_width(yue_phase phase);
/* Phase-local end index: EOD for abc, 0 for semantic. */
int yue_phase_end_local(yue_phase phase);

/* Local sampled index -> checkpoint-native id (abc: identity below EOD,
 * EOD row -> ABC_END; semantic: MUSIC_END + local). */
uint32_t yue_local_to_native(int local, yue_phase phase);
/* Native id -> local row, or -1 when the id is outside the phase. */
int yue_native_to_local(uint32_t token, yue_phase phase);

/* fp32 -> bf16 (round-to-nearest-even) -> fp32, the storage rounding lyra's
 * BF16 path applies after every op. NaN passes through. */
float yue_bf16_round(float value);

/* lyra guided_logits: historical CFG with explicit bf16 rounding after the
 * subtract, the scale multiply and the final add. cond/uncond/out are bf16
 * bit buffers of `width` rows; out may alias cond. */
void yue_guided_logits(const uint16_t *cond, const uint16_t *uncond,
                       size_t width, double cfg_scale, uint16_t *out);

/* Frequency-powered repetition penalty over the last `n_recent` local ids
 * (accumulated counts, not once-per-token). scores is fp32 holding bf16
 * values when legacy; alpha rounds to the score dtype like lyra. */
void yue_window_penalty(float *scores, size_t width, const int *recent_local,
                        size_t n_recent, double penalty, int legacy);

/* Full filter chain: min-tokens end mask, window penalty, temperature,
 * top-k (kth-value threshold, ties survive) and top-p (shifted cumulative
 * test). `history` holds native ids; `step` is the 0-based loop index.
 * Modifies scores in place (fp32; bf16-valued when legacy). Returns 0 only
 * on allocation failure (the caller must abort instead of sampling from an
 * unfiltered distribution). */
int yue_distribution(float *scores, size_t width, yue_phase phase,
                     const yue_sampling *sp, const uint32_t *history,
                     size_t history_len, int step, int legacy);

/* ------------------------------------------------------------ request RNG */

typedef struct {
    uint64_t state;
} yue_rng;

void yue_rng_init(yue_rng *rng, uint64_t seed);
uint64_t yue_rng_next(yue_rng *rng);
/* Uniform double in [0, 1). */
double yue_rng_uniform(yue_rng *rng);

/* Categorical draw over fp32 scores (post-distribution; -inf rows get zero
 * probability). Mirrors lyra's final softmax: bf16-rounded probabilities
 * when legacy. Returns the sampled index, or -1 when scores hold no finite
 * value. */
int yue_sample_categorical(const float *scores, size_t width, int legacy,
                           yue_rng *rng);

#endif
