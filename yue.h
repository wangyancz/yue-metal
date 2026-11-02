#ifndef YUE_H
#define YUE_H

#include <stddef.h>
#include <stdint.h>

/* yue-metal: native YuE2-3B inference engine (C11 + ObjC/Metal).
 *
 * Reference stack: the lyra MLX port at mlx-Yue plus its vendored official
 * yue2 protocol. Numeric conventions mirror lyra exactly; see README.
 */

/* ---------------------------------------------------------------- protocol
 * Checkpoint-native token IDs (lyra vendor/yue2/protocol.py). */
#define YUE_EOD UINT32_C(151643)
#define YUE_ABC_START UINT32_C(151847)
#define YUE_ABC_END UINT32_C(151848)
#define YUE_MUSIC_START UINT32_C(151851)
#define YUE_MUSIC_END UINT32_C(151852)
#define YUE_CODEC_OFFSET UINT32_C(151853)
#define YUE_CODEC_SIZE UINT32_C(32768)
#define YUE_LATENT_START UINT32_C(184621)
#define YUE_LATENT_END UINT32_C(184622)
#define YUE_LATENT_PAD UINT32_C(184623)
#define YUE_VOCAB_SIZE UINT32_C(184704)
#define YUE_CONTEXT UINT32_C(24576)

/* Sampling repetition-window cap (protocol Sampling validation). */
#define YUE_MAX_WINDOW 100

/* Instruction templates, keyed by cot mode (protocol INSTRUCTIONS). */
extern const char *const yue_instructions[3];
#define YUE_COT_FULL 0
#define YUE_COT_MELODY 1
#define YUE_COT_OFF 2

/* ------------------------------------------------------- AR model geometry
 * lyra-YuE2-3B/config.json; validated at load time. */
#define YUE_AR_LAYERS 28
#define YUE_HIDDEN 2048
#define YUE_HEADS 16
#define YUE_KV_HEADS 8
#define YUE_HEAD_DIM 128
#define YUE_MLP 6144
#define YUE_RMS_EPS 1e-6f
#define YUE_ROPE_THETA 1000000.0

/* ------------------------------------------------------ NAR model geometry */
#define YUE_LATENT_DIM 64
#define YUE_TIME_EMBED 256

/* --------------------------------------------------------- audio constants */
#define YUE_SAMPLE_RATE 48000
#define YUE_DOWNSAMPLING 1920 /* product of decoder strides 2*2*4*4*5*6 */

#define YUE_ERROR_SIZE 512

/* Shape bucketing (resident-footprint control). The Metal driver pool
 * reuses freed buffers by exact size: AR generation drifts a few codec
 * tokens per request (bf16 GEMM run-to-run nondeterminism amplified by
 * temperature sampling), so every capacity derived from actual token
 * counts would be a new size each request and the resident footprint
 * would grow without bound. Bucketing capacities up to YUE_SHAPE_BUCKET
 * rows keeps the shape space finite (the pool converges after the bucket
 * set has been seen); all write/read logic keeps using the exact row
 * counts, so results are bit-identical. */
#define YUE_SHAPE_BUCKET 512

static inline size_t yue_align_rows(size_t rows) {
    return (rows + YUE_SHAPE_BUCKET - 1) / YUE_SHAPE_BUCKET *
           YUE_SHAPE_BUCKET;
}

#endif
