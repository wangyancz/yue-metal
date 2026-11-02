#ifndef YUE_VAE_H
#define YUE_VAE_H

#include <stddef.h>
#include <stdint.h>

#include "h3_gpu.h"
#include "yue.h"

/* YuE2 acoustic VAE decoder (Oobleck, fp32; lyra vae.py parity). Conv1d /
 * ConvTranspose1d weights are folded from weight_norm (v * g/||v||) into a
 * uniform [out_channel, kernel_tap, in_channel] layout. GPU path: per-tap
 * accumulating TensorOps matmul2d GEMMs over zero-ringed activations,
 * SnakeBeta elementwise; latents are decoded in halo tiles exactly like
 * the reference (core 256 / halo 16). Passing gpu == NULL (or setting
 * YUE_VAE_CPU to a non-zero value) keeps the reference CPU path
 * (Accelerate sgemm). */

typedef struct yue_vae yue_vae;

yue_vae *yue_vae_create(const char *model_dir, h3_gpu *gpu, char *error,
                        size_t error_size);
void yue_vae_free(yue_vae *vae);

/* Total output samples (per channel) for `frames` latent frames. */
size_t yue_vae_output_length(const yue_vae *vae, size_t frames);

/* fp32 latents [frames, 64] -> fp32 interleaved stereo [samples, 2]
 * (caller-allocated). Set QWEN_VAE_DUMP=<dir> to dump per-stage activations
 * for parity testing. */
int yue_vae_decode(const yue_vae *vae, const float *latents, size_t frames,
                   float *audio, char *error, size_t error_size);

/* 24-bit WAV writer + ffmpeg FLAC conversion (production output path). */
int yue_write_wav24(const char *path, const float *samples, size_t frames,
                    int channels, int rate, char *error, size_t error_size);
int yue_wav_to_flac(const char *wav_path, const char *flac_path, char *error,
                    size_t error_size);

#endif
