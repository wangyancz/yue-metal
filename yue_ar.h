#ifndef YUE_AR_H
#define YUE_AR_H

#include <stddef.h>
#include <stdint.h>

#include "h3_gpu.h"
#include "yue.h"

/* YuE2 AR model: Qwen3 backbone loaded from a model directory — either the
 * lyra split (ar-bf16.safetensors) or the official m-a-p/YuE2-3B single-file
 * checkpoint (model.safetensors; same keys, byte-identical values).
 * Numeric conventions mirror lyra ar.py: fp32 RMSNorm variance with bf16
 * double rounding, fp32 SiLU opmath, half-split RoPE with fp32 angles,
 * phase-local output projection in bf16. */

typedef struct yue_ar yue_ar;

/* Per-layer fixed-capacity KV cache ([capacity, kv_heads, head_dim]
 * row-major bf16 per tensor; rows [0, length) are occupied). RoPE positions
 * equal cache offsets because YuE2 AR streams are unpadded. */
typedef struct yue_ar_cache yue_ar_cache;

yue_ar *yue_ar_create(h3_gpu *gpu, const char *model_dir, char *error,
                      size_t error_size);
void yue_ar_free(yue_ar *ar);
h3_gpu *yue_ar_gpu(yue_ar *ar);

yue_ar_cache *yue_ar_cache_create(yue_ar *ar, size_t capacity, char *error,
                                  size_t error_size);
void yue_ar_cache_free(yue_ar_cache *cache);
/* Drop occupied slots without freeing storage. */
void yue_ar_cache_reset(yue_ar_cache *cache);
size_t yue_ar_cache_length(const yue_ar_cache *cache);

/* Embed tokens and run all layers. Appends keys/values to the cache. The
 * returned tensor is the final-norm OUTPUT of shape [count, hidden] and is
 * owned by the model (valid until the next forward on the same instance).
 * Set QWEN_AR_DUMP=<dir> to dump embed/layer{i}/final bf16 files there. */
h3_gpu_tensor *yue_ar_forward(yue_ar *ar, const uint32_t *tokens, size_t count,
                              yue_ar_cache *cache, char *error,
                              size_t error_size);

/* Chain-open variant: identical arithmetic, but the caller owns an open
 * command buffer and the forward leaves it open (no begin/submit inside).
 * Decode steps merge pos/neg forwards and both logit projections into one
 * chain and wait once at the tail. On failure the chain may be mid-flight;
 * the caller must h3_gpu_submit() to drain before any further GPU call
 * (draining assumes an open buffer — the pathological case of a failed
 * h3_gpu_continue with no replacement buffer leaves only inflight work,
 * which only a successful submit on a new buffer can retire). */
h3_gpu_tensor *yue_ar_forward_open(yue_ar *ar, const uint32_t *tokens,
                                   size_t count, yue_ar_cache *cache,
                                   char *error, size_t error_size);

/* Batched CFG decode forward (M15): one [2, hidden] stream, both branch
 * rows riding every linear (one weight read; probe H: rows=2 = 0.98x
 * rows=1 gpu time on the big shapes). count must be 1 — one decode
 * token per branch. Each row attends and writes its own cache; the two
 * caches advance independently. Returns the [2, hidden] final-normed
 * hidden (row 0 = pos, row 1 = neg). */
h3_gpu_tensor *yue_ar_forward_open2(yue_ar *ar, uint32_t token_pos,
                                    uint32_t token_neg,
                                    yue_ar_cache *cache_pos,
                                    yue_ar_cache *cache_neg, char *error,
                                    size_t error_size);

/* Gather-only helper (parity/debug): [count, hidden] bf16. */
h3_gpu_tensor *yue_ar_embed(yue_ar *ar, const uint32_t *tokens, size_t count,
                            char *error, size_t error_size);

/* Raw cache accessors ([capacity, kv_heads, head_dim] row-major bf16;
 * rows [0, length) are occupied). */
h3_gpu_tensor *yue_ar_cache_key(yue_ar_cache *cache, int layer);
h3_gpu_tensor *yue_ar_cache_value(yue_ar_cache *cache, int layer);

/* Shared final RMSNorm weight (the NAR output head reuses it). */
h3_gpu_tensor *yue_ar_norm_weight(yue_ar *ar);

/* fp32 RoPE cos/sin tables (bf16-rounded values) for absolute positions
 * [offset, offset+rows). Caller frees both tensors. */
int yue_ar_rope_slice(yue_ar *ar, size_t offset, size_t rows,
                      h3_gpu_tensor **cos_out, h3_gpu_tensor **sin_out,
                      char *error, size_t error_size);

/* Phase-local output projection onto `logits` ([rows, width] bf16, allocated
 * by the caller): abc width = YUE_EOD + 1 rows ([0, EOD) then ABC_END),
 * semantic width = YUE_CODEC_SIZE + 1 rows (MUSIC_END first, then codec).
 * rows is the number of hidden rows to project (the hidden tensor may be a
 * larger workspace, only the first rows are read). */
int yue_ar_project_logits(yue_ar *ar, const h3_gpu_tensor *hidden, size_t rows,
                          const char *phase, h3_gpu_tensor *logits,
                          char *error, size_t error_size);

/* Chain-open projection: same math, encodes into the caller's open command
 * buffer and leaves it open. */
int yue_ar_project_open(yue_ar *ar, const h3_gpu_tensor *hidden, size_t rows,
                        const char *phase, h3_gpu_tensor *logits,
                        char *error, size_t error_size);

#endif
