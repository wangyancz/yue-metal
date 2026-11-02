#ifndef YUE_NAR_H
#define YUE_NAR_H

#include <stddef.h>
#include <stdint.h>

#include "h3_gpu.h"
#include "yue_ar.h"

/* YuE2 acoustic flow matching (lyra nar.py): non-causal acoustic expert
 * layers attending over the AR backbone's causal conditioning cache, with a
 * timestep-embedded input, latent position buffer and the shared final
 * RMSNorm + llm2vae head. The midpoint solver runs host-side so every bf16
 * rounding point matches the reference exactly. */

typedef struct yue_nar yue_nar;
typedef struct yue_nar_chunk yue_nar_chunk;

/* Loads nar-bf16.safetensors (acoustic experts + heads); `ar` supplies the
 * shared embedding, final norm and conditioning backbone and must outlive
 * the NAR instance. */
yue_nar *yue_nar_create(h3_gpu *gpu, yue_ar *ar, const char *model_dir,
                        char *error, size_t error_size);
void yue_nar_free(yue_nar *nar);

/* One acoustic chunk: prefill the AR conditioning over
 * [prefix | codec+OFFSET | MUSIC_END] and allocate the combined K/V
 * buffers. `noise_frames` is the chunk's latent frame count;
 * `visible` truncates the conditioning the acoustic layers see by keeping
 * the FIRST visible rows (the style/lyrics prefix) and dropping the rest
 * (0 = all tokens, the production path). */
yue_nar_chunk *yue_nar_chunk_create(yue_nar *nar, const uint32_t *ar_tokens,
                                    size_t token_count, size_t noise_frames,
                                    size_t visible, char *error,
                                    size_t error_size);
void yue_nar_chunk_free(yue_nar_chunk *chunk);

/* Conditioning cache access for parity tests ([ar_len, kv_heads, head_dim]
 * bf16 per layer). */
yue_ar_cache *yue_nar_chunk_cond(yue_nar_chunk *chunk);

/* One velocity evaluation on bf16 state bits [frames, 64]; writes
 * [frames, 64] bf16 bits. `raw_t` is the unshifted flow time. */
int yue_nar_velocity(yue_nar_chunk *chunk, const uint16_t *state_bits,
                     double raw_t, uint16_t *out_bits, char *error,
                     size_t error_size);

/* Reference midpoint solver (two velocities per step) from fp32 initial
 * noise [frames, 64]; `out` receives the final fp32 latents. */
int yue_nar_solve(yue_nar_chunk *chunk, const float *initial_noise, int steps,
                  float *out, char *error, size_t error_size);

#endif
