/* Host-side YuE2 sampling (lyra/sampling.py parity). See yue_sampling.h. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "yue_sampling.h"

/* ------------------------------------------------------------ bf16 bits */

static uint16_t f32_to_bf16_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x7FFFFFu)) {
        /* NaN: the RNE carry can promote the payload into the exponent
         * field and flip the value to an infinity/zero (0x7FFF8001 ->
         * -0.0); the header contract says NaN passes through. */
        return (uint16_t)((bits >> 16) | 0x0040u); /* keep a quiet NaN */
    }
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float bf16_bits_to_f32(uint16_t bits) {
    uint32_t wide = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &wide, sizeof(value));
    return value;
}

float yue_bf16_round(float value) {
    return bf16_bits_to_f32(f32_to_bf16_bits(value));
}

/* --------------------------------------------------------------- phases */

size_t yue_phase_width(yue_phase phase) {
    return phase == YUE_PHASE_ABC ? (size_t)YUE_EOD + 1
                                  : (size_t)YUE_CODEC_SIZE + 1;
}

int yue_phase_end_local(yue_phase phase) {
    return phase == YUE_PHASE_ABC ? (int)YUE_EOD : 0;
}

uint32_t yue_local_to_native(int local, yue_phase phase) {
    if (phase == YUE_PHASE_ABC)
        return local < (int)YUE_EOD ? (uint32_t)local : YUE_ABC_END;
    return YUE_MUSIC_END + (uint32_t)local;
}

int yue_native_to_local(uint32_t token, yue_phase phase) {
    if (phase == YUE_PHASE_ABC) {
        if (token < YUE_EOD) return (int)token;
        return token == YUE_ABC_END ? (int)YUE_EOD : -1;
    }
    if (token == YUE_MUSIC_END) return 0;
    if (token >= YUE_CODEC_OFFSET &&
        token < YUE_CODEC_OFFSET + YUE_CODEC_SIZE)
        return (int)(token - YUE_MUSIC_END);
    return -1;
}

/* ------------------------------------------------------------------ CFG */

void yue_guided_logits(const uint16_t *cond, const uint16_t *uncond,
                       size_t width, double cfg_scale, uint16_t *out) {
    const float cfg = (float)cfg_scale;
    for (size_t i = 0; i < width; i++) {
        float c = bf16_bits_to_f32(cond[i]);
        float u = bf16_bits_to_f32(uncond[i]);
        float delta = yue_bf16_round(c - u);
        float scaled = yue_bf16_round(delta * cfg);
        out[i] = f32_to_bf16_bits(yue_bf16_round(u + scaled));
    }
}

/* --------------------------------------------------------- window penalty */

void yue_window_penalty(float *scores, size_t width, const int *recent_local,
                        size_t n_recent, double penalty, int legacy) {
    if (penalty == 1.0 || n_recent == 0 || width == 0) return;

    /* Accumulated per-id counts over the (<= 100-wide) window. */
    int ids[YUE_MAX_WINDOW];
    int counts[YUE_MAX_WINDOW];
    size_t unique = 0;
    for (size_t i = 0; i < n_recent; i++) {
        int id = recent_local[i];
        if (id < 0 || (size_t)id >= width) continue;
        size_t hit = unique;
        for (size_t u = 0; u < unique; u++) {
            if (ids[u] == id) {
                hit = u;
                break;
            }
        }
        if (hit == unique) {
            if (unique >= YUE_MAX_WINDOW) continue;
            ids[unique] = id;
            counts[unique] = 1;
            unique++;
        } else {
            counts[hit]++;
        }
    }

    const float base = (float)penalty;
    for (size_t u = 0; u < unique; u++) {
        float alpha = powf(base, (float)counts[u]);
        if (legacy) alpha = yue_bf16_round(alpha);
        /* A non-finite/zero factor would turn logits into NaN/Inf (and can
         * resurrect the EOS row that min_tokens just masked). Skip this
         * token's penalty instead. */
        if (!(alpha > 0.0f) || !isfinite(alpha)) continue;
        float s = scores[ids[u]];
        /* lyra multiplies negatives and divides non-negatives; -inf passes
         * through both. */
        scores[ids[u]] = legacy
                             ? yue_bf16_round(s < 0.0f ? s * alpha : s / alpha)
                             : (s < 0.0f ? s * alpha : s / alpha);
    }
}

/* -------------------------------------------------------------- top-k/p */

/* Min-heap of floats holding the k largest values seen so far. */
typedef struct {
    float *data;
    size_t size;
    size_t capacity;
} float_heap;

static void heap_push(float_heap *h, float value) {
    size_t i = h->size++;
    h->data[i] = value;
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (h->data[parent] <= h->data[i]) break;
        float tmp = h->data[parent];
        h->data[parent] = h->data[i];
        h->data[i] = tmp;
        i = parent;
    }
}

static float heap_pop(float_heap *h) {
    float top = h->data[0];
    h->data[0] = h->data[--h->size];
    size_t i = 0;
    for (;;) {
        size_t left = 2 * i + 1, right = left + 1, small = i;
        if (left < h->size && h->data[left] < h->data[small]) small = left;
        if (right < h->size && h->data[right] < h->data[small]) small = right;
        if (small == i) break;
        float tmp = h->data[small];
        h->data[small] = h->data[i];
        h->data[i] = tmp;
        i = small;
    }
    return top;
}

/* Descending sort by value with ascending index tiebreak (qsort). */
typedef struct {
    float value;
    size_t index;
} scored_cell;

static int cell_cmp(const void *a, const void *b) {
    const scored_cell *x = a, *y = b;
    if (x->value > y->value) return -1;
    if (x->value < y->value) return 1;
    if (x->index < y->index) return -1;
    if (x->index > y->index) return 1;
    return 0;
}

/* Nucleus filter over the surviving (finite) entries. `survivors` are the
 * finite scores in their original positions, already sorted descending. */
static void apply_top_p(float *scores, scored_cell *survivors,
                        size_t n_survivors, double top_p, int legacy,
                        size_t minimum) {
    if (n_survivors == 0) return;
    /* softmax over survivors == full softmax: masked rows are -inf. */
    float max_value = survivors[0].value;
    double total = 0.0;
    for (size_t i = 0; i < n_survivors; i++)
        total += exp((double)survivors[i].value - (double)max_value);
    if (!(total > 0.0)) return;

    /* lyra: preceding_mass = (cumsum - probs) with the source dtype applied
     * after each scan step; removed = fp32(preceding) > top_p, first
     * `minimum` sorted positions always kept. MLX's GPU scan may associate
     * the last bf16 ulp differently; only the boundary position can move. */
    double running = 0.0;
    for (size_t i = 0; i < n_survivors; i++) {
        float prob = (float)(exp((double)survivors[i].value -
                                 (double)max_value) / total);
        if (legacy) prob = yue_bf16_round(prob);
        float inclusive = legacy ? yue_bf16_round((float)running + prob)
                                 : (float)(running + (double)prob);
        float preceding = legacy ? yue_bf16_round(inclusive - prob)
                                 : inclusive - prob;
        if (i >= minimum && (double)preceding > top_p)
            scores[survivors[i].index] = -INFINITY;
        running = inclusive;
    }
}

/* ----------------------------------------------------------- distribution */

int yue_distribution(float *scores, size_t width, yue_phase phase,
                     const yue_sampling *sp, const uint32_t *history,
                     size_t history_len, int step, int legacy) {
    /* Public API: the phase end index and the fixed-size window scratch
     * assume the canonical projection width. Reject a mismatched buffer
     * instead of writing scores[end_local] out of bounds. */
    if (!scores || !sp || width != yue_phase_width(phase) ||
        (history_len && !history)) return 0;
    /* min-tokens: mask the phase end before anything else. */
    if (step < sp->min_tokens)
        scores[yue_phase_end_local(phase)] = -INFINITY;

    /* window penalty over the last penalty_window native ids */
    size_t take = history_len;
    if (take > (size_t)sp->penalty_window) take = (size_t)sp->penalty_window;
    int local[YUE_MAX_WINDOW];
    size_t n_local = 0;
    for (size_t i = history_len - take; i < history_len; i++) {
        int loc = yue_native_to_local(history[i], phase);
        if (loc >= 0 && n_local < YUE_MAX_WINDOW) local[n_local++] = loc;
    }
    yue_window_penalty(scores, width, local, n_local,
                       sp->repetition_penalty, legacy);

    if (sp->temperature == 0.0) return 1; /* greedy: caller takes argmax */

    if (sp->temperature != 1.0) {
        const float temp = (float)sp->temperature;
        for (size_t i = 0; i < width; i++)
            scores[i] = legacy ? yue_bf16_round(scores[i] / temp)
                               : scores[i] / temp;
    }

    /* top-k by the kth value so ties survive */
    size_t k = (size_t)sp->top_k;
    if (k > width) k = width;
    float threshold = -INFINITY;
    if (k > 0) {
        float_heap heap = {NULL, 0, k};
        heap.data = malloc(sizeof(float) * k);
        /* Fail closed: an unfiltered distribution would silently sample
         * outside the configured top-k. */
        if (!heap.data) return 0;
        for (size_t i = 0; i < width; i++) {
            if (heap.size < k) {
                heap_push(&heap, scores[i]);
            } else if (scores[i] > heap.data[0]) {
                heap_pop(&heap);
                heap_push(&heap, scores[i]);
            }
        }
        threshold = heap.data[0];
        free(heap.data);
        for (size_t i = 0; i < width; i++)
            if (scores[i] < threshold) scores[i] = -INFINITY;
    }

    if (sp->top_p < 1.0) {
        /* Collect survivors (finite scores; ties at the threshold survive
         * the kth-value mask by design) and sort descending. */
        scored_cell *cells = malloc(sizeof(scored_cell) * width);
        if (!cells) return 0;
        size_t n_cells = 0;
        for (size_t i = 0; i < width; i++) {
            if (scores[i] != -INFINITY && scores[i] == scores[i]) {
                cells[n_cells].value = scores[i];
                cells[n_cells].index = i;
                n_cells++;
            }
        }
        qsort(cells, n_cells, sizeof(scored_cell), cell_cmp);
        apply_top_p(scores, cells, n_cells, sp->top_p, legacy,
                    legacy ? 3 : 1);
        free(cells);
    }
    return 1;
}

/* ------------------------------------------------------------- request RNG */

void yue_rng_init(yue_rng *rng, uint64_t seed) {
    rng->state = seed ^ UINT64_C(0xA0761D6478BD642F);
    yue_rng_next(rng);
}

uint64_t yue_rng_next(yue_rng *rng) {
    /* splitmix64 */
    rng->state += UINT64_C(0x9E3779B97F4A7C15);
    uint64_t z = rng->state;
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

double yue_rng_uniform(yue_rng *rng) {
    return (double)(yue_rng_next(rng) >> 11) * 0x1.0p-53;
}

int yue_sample_categorical(const float *scores, size_t width, int legacy,
                           yue_rng *rng) {
    float max_value = -INFINITY;
    for (size_t i = 0; i < width; i++)
        if (scores[i] > max_value) max_value = scores[i];
    if (max_value == -INFINITY || max_value != max_value) return -1;

    double total = 0.0;
    for (size_t i = 0; i < width; i++)
        if (scores[i] != -INFINITY)
            total += exp((double)scores[i] - (double)max_value);
    if (!(total > 0.0)) return -1;

    const double u = yue_rng_uniform(rng);
    double acc = 0.0;
    int last = -1;
    for (size_t i = 0; i < width; i++) {
        if (scores[i] == -INFINITY) continue;
        last = (int)i;
        float prob = (float)(exp((double)scores[i] - (double)max_value) /
                             total);
        if (legacy) prob = yue_bf16_round(prob);
        acc += (double)prob;
        if (acc > u) return (int)i;
    }
    return last;
}
