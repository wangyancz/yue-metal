#ifndef QWEN_GPU_EXT_H
#define QWEN_GPU_EXT_H

/* Dispatch wrappers for the qw_* kernels appended to h3_shaders.metal.
 * These are qwen-metal additions; the h3_* API in h3_gpu.h is
 * unchanged vendored code. */

#include "h3_gpu.h"

/* Embedding table gather: output rows come from table[ids[r]]. */
int qw_gpu_embed_gather_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *table,
                             const h3_gpu_tensor *ids, uint32_t rows,
                             uint32_t hidden);

/* Affine-free LayerNorm then per-row (1 + scale) modulation. */
int qw_gpu_layer_norm_mod_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                               const h3_gpu_tensor *input,
                               const h3_gpu_tensor *modulation,
                               const h3_gpu_tensor *row_map, uint32_t rows,
                               uint32_t width, uint32_t slots,
                               uint32_t scale_slot, float epsilon);

/* RMSNorm whose effective weight is (weight + 1), F32 math. */
int qw_gpu_rms_norm_one_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight, uint32_t rows,
                             uint32_t width, float epsilon);

/* Elementwise tanh over BF16. */
int qw_gpu_tanh_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements);

/* Interleaved complex RoPE on [row, heads, head_dim] BF16 Q/K with F32
 * cos/sin tables of [row, head_dim / 2]. In-place safe. */
int qw_gpu_qk_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query_output,
                        h3_gpu_tensor *key_output,
                        const h3_gpu_tensor *query_input,
                        const h3_gpu_tensor *key_input,
                        const h3_gpu_tensor *rope_cos,
                        const h3_gpu_tensor *rope_sin, uint32_t sequence,
                        uint32_t heads, uint32_t head_dim);

/* Half-split (rotate_half) RoPE: head element i pairs with element
 * i + head_dim / 2 (transformers text-tower convention), cos/sin tables
 * of [row, head_dim / 2]. In-place safe. */
int qw_gpu_qk_rope_half_bf16(h3_gpu *gpu, h3_gpu_tensor *query_output,
                             h3_gpu_tensor *key_output,
                             const h3_gpu_tensor *query_input,
                             const h3_gpu_tensor *key_input,
                             const h3_gpu_tensor *rope_cos,
                             const h3_gpu_tensor *rope_sin, uint32_t sequence,
                             uint32_t heads, uint32_t head_dim);

/* RMSNorm with PyTorch CPU bf16 rounding points: normalized fp32 value
 * quantized to bf16 BEFORE the weight multiply (double rounding). */int qw_gpu_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon);

/* Per-head RMSNorm with the same double rounding (Qwen3 q_norm/k_norm). */
int qw_gpu_head_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *tensor,
                              const h3_gpu_tensor *weight,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, float epsilon);

/* Fused per-head RMSNorm + half-split RoPE over the q tensor (q_heads
 * heads) and the k tensor (kv_heads heads) — bit-identical to the
 * separate norm + rope pair, one dispatch for both (pass a head count
 * as 0 to process only the other tensor). */
int qw_gpu_head_rms_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                              h3_gpu_tensor *key,
                              const h3_gpu_tensor *query_weight,
                              const h3_gpu_tensor *key_weight,
                              const h3_gpu_tensor *rope_cos,
                              const h3_gpu_tensor *rope_sin,
                              uint32_t sequence, uint32_t q_heads,
                              uint32_t kv_heads, uint32_t head_dim,
                              float epsilon);

/* Fused residual add + RMSNorm: residual = bf16(residual + addend) in
 * place, output = rms_norm(residual, weight). Bit-identical to
 * h3_gpu_add_bf16 followed by qw_gpu_rms_norm_bf16, one dispatch. */
int qw_gpu_add_rms_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                        h3_gpu_tensor *residual,
                        const h3_gpu_tensor *addend,
                        const h3_gpu_tensor *weight, uint32_t rows,
                        uint32_t width, float epsilon);

/* SwiGLU over the fused fc1 gemm output with PyTorch CPU bf16 rounding
 * points (silu quantized to bf16, product quantized again); width is
 * the per-half MLP width. */
int qw_gpu_swiglu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *fc1, uint32_t count,
                       uint32_t width);

/* Fused rows==1 q/k/v projection GEMV: one dispatch over all q+k+v
 * output columns reading the shared normalized input. Bit-exact with
 * three separate qw_gemv_bf16 dispatches (each output column keeps one
 * ascending-k fp32 fma chain). Returns 0 (caller falls back to the
 * three per-matrix projections) unless the shape/alignment gates pass:
 * in_dim % 8 == 0, out_q/out_kv % 64 == 0, 16-byte aligned. */
int qw_gpu_qkv_gemv_bf16(h3_gpu *gpu, h3_gpu_tensor *output_q,
                         h3_gpu_tensor *output_k, h3_gpu_tensor *output_v,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight_q,
                         const h3_gpu_tensor *weight_k,
                         const h3_gpu_tensor *weight_v, uint32_t in_dim,
                         uint32_t out_dim_q, uint32_t out_dim_kv);

/* TensorOps decode GEMV (rows == 1): matmul2d over one padded M tile with
 * a direct dispatch — no MPSGraph encode. Gate failure (no TensorOps,
 * output_dim % 32 != 0, input buffer < 16 padded rows) returns 0 without
 * set_error so callers can fall back to the MPS linear. */
int qw_gpu_tops_gemv_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight, uint32_t in_dim,
                          uint32_t out_dim);

/* Qwen VAE decoder (fp32, channels-last [H, W, C] activations). */

/* 3D convolution with explicit zero padding on all six NDHWC sides. */
int h3_gpu_conv3d_pad_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t batch,
                          uint32_t depth, uint32_t height, uint32_t width,
                          uint32_t input_channels, uint32_t output_channels,
                          uint32_t kernel_depth, uint32_t kernel_height,
                          uint32_t kernel_width, uint32_t stride_depth,
                          uint32_t stride_height, uint32_t stride_width,
                          uint32_t pad_front, uint32_t pad_back,
                          uint32_t pad_top, uint32_t pad_bottom,
                          uint32_t pad_left, uint32_t pad_right);

/* Channel RMSNorm: x / max(||x||2, eps) * sqrt(dim) * gamma, one
 * threadgroup per row. */
int qw_gpu_vae_rms_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *gamma, uint32_t rows,
                            uint32_t width, float scale, float epsilon);

/* Nearest-exact 2x spatial upsample. */
int qw_gpu_vae_nearest2x_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input, uint32_t height,
                             uint32_t width, uint32_t channels);

/* DupUp3D spatial shortcut for the image path (single frame,
 * first_chunk drops the first duplicated temporal frame). */
int qw_gpu_vae_dupup_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input, uint32_t height,
                         uint32_t width, uint32_t in_channels,
                         uint32_t out_channels, uint32_t factor_t,
                         uint32_t factor_s, uint32_t frame_index);

/* Split the fused [rows, 3C] qkv activation: parts 0/1 -> [rows, C],
 * part 2 -> transposed [C, rows]. */
int qw_gpu_vae_split3_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input, uint32_t rows,
                          uint32_t width, uint32_t part);

/* Row softmax over the full width; scale applied before max/sum (SDPA). */
int qw_gpu_vae_softmax_rows_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t width, float scale);

/* Flash-style online-softmax SDPA (bf16, fp32 accumulation): image
 * queries against every key, 64-query tiles. */
int qw_gpu_flash_sdpa_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query,
                           const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value, uint32_t q_rows,
                           uint32_t kv_rows, uint32_t heads,
                           uint32_t head_dim, float scale);

/* Prefix-visible variant: kv_limit[q_row] caps the key range per query
 * row (block-causal mask as a prefix; U32 tensor, q_rows elements,
 * values are non-negative key counts). kv_rows is the full key length
 * (buffer extent); limits <= kv_rows. Grouped-query attention:
 * heads % kv_heads == 0, q head h reads kv head h * kv_heads / heads
 * (pass heads for plain MHA). */
int qw_gpu_flash_sdpa_prefix_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *query,
                                  const h3_gpu_tensor *key,
                                  const h3_gpu_tensor *value,
                                  const h3_gpu_tensor *kv_limit,
                                  uint32_t q_rows, uint32_t kv_rows,
                                  uint32_t heads, uint32_t kv_heads,
                                  uint32_t head_dim, float scale);

/* Decode-step SDPA for q_rows == 1 (AR token generation): one
 * 128-thread group per head, flash2-remainder math (ascending-d fp32
 * fma dot, -FLT_MAX online softmax, bf16-rounded P for PV, fp32 p for
 * the denominator, bfloat(acc/l) output). Only the accumulation order
 * differs from flash2's TensorOps tiles (1-ulp bf16 differences
 * possible). */
int qw_gpu_sdpa_decode_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *query,
                            const h3_gpu_tensor *key,
                            const h3_gpu_tensor *value, uint32_t kv_rows,
                            uint32_t heads, uint32_t kv_heads,
                            uint32_t head_dim, float scale);

/* Split-K decode SDPA (q_rows == 1, flash-decoding): phase 1 over
 * (heads x splits) slabs into the caller-owned fp32 `partial` scratch
 * (>= heads*max_splits*(head_dim+2) elements), phase 2 merges. splits
 * is derived from kv_rows (256-row slabs) and clamped to max_splits.
 * Two dispatches inside one command encoder. */
int qw_gpu_sdpa_decode_split_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *query,
                                  const h3_gpu_tensor *key,
                                  const h3_gpu_tensor *value,
                                  uint32_t kv_rows, uint32_t heads,
                                  uint32_t kv_heads, uint32_t head_dim,
                                  float scale, h3_gpu_tensor *partial,
                                  uint32_t max_splits);

/* Offset variant for batched CFG decode: query/output at element offsets
 * inside shared [2, ...] branch buffers. */
int qw_gpu_sdpa_decode_split_bf16_off(h3_gpu *gpu, h3_gpu_tensor *output,
                                      const h3_gpu_tensor *query,
                                      const h3_gpu_tensor *key,
                                      const h3_gpu_tensor *value,
                                      size_t query_offset,
                                      size_t output_offset,
                                      uint32_t kv_rows, uint32_t heads,
                                      uint32_t kv_heads, uint32_t head_dim,
                                      float scale, h3_gpu_tensor *partial,
                                      uint32_t max_splits);

/* [C, H, W] <-> [H, W, C] layout transposes. */
int qw_gpu_vae_chw_to_hwc_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                              const h3_gpu_tensor *input, uint32_t channels,
                              uint32_t height, uint32_t width);
int qw_gpu_vae_hwc_to_chw_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                              const h3_gpu_tensor *input, uint32_t channels,
                              uint32_t height, uint32_t width);

/* Direct fp32 3x3 convolution, zero padding, stride 1, channels-last. */
int qw_gpu_vae_conv3x3_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t height,
                           uint32_t width, uint32_t in_channels,
                           uint32_t out_channels);

/* GEMM-based fp32 3x3 convolution: 9-tap accumulating matmul2d over a
 * ring-padded copy of the activation. `w9` is the repacked
 * [9, CO_pad, CI] weight from qw_gpu_vae_conv_repack_f32. `pad_arena`
 * is a caller-owned scratch of >= pad_rows*in_channels floats
 * (vae_pad_elems), reused across convs. */
int qw_gpu_vae_conv_mm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *w9,
                           const h3_gpu_tensor *bias, uint32_t height,
                           uint32_t width, uint32_t in_channels,
                           uint32_t out_channels, h3_gpu_tensor *pad_arena);

/* One-time conv weight repack [CO, CI, 9] -> [9, CO_pad, CI]
 * (CO_pad = CO rounded up to 64). */
int qw_gpu_vae_conv_repack_f32(h3_gpu *gpu, h3_gpu_tensor *w9,
                               const h3_gpu_tensor *w, uint32_t in_channels,
                               uint32_t out_channels, uint32_t out_pad);

/* Elementwise add. */
int qw_gpu_vae_add_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                       uint32_t count);

/* 3x3 conv stride 2 (encoder downsampler), channels-last activations,
 * weight [oc, ic, 9]. */
int qw_gpu_vae_downsample_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                              const h3_gpu_tensor *input,
                              const h3_gpu_tensor *weight,
                              const h3_gpu_tensor *bias, uint32_t height,
                              uint32_t width, uint32_t in_channels,
                              uint32_t out_channels);

/* AvgDown3D single-frame spatial mean with channel remap
 * (factor_t semantics identical to the CPU reference). */
int qw_gpu_vae_avgdown_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input, uint32_t height,
                           uint32_t width, uint32_t in_channels,
                           uint32_t out_channels, uint32_t factor_t);

/* Linear with the weight stored transposed [input_dim, output_dim]. */
int qw_gpu_vae_linear_t_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *bias, uint32_t rows,
                            uint32_t input_dim, uint32_t output_dim);

/* Fused q/k/v projection: one MPSGraph encode runs three matmuls that
 * share the input; weights keep their native [out, in] buffers. */
int qw_gpu_qkv_bf16(h3_gpu *gpu, h3_gpu_tensor *query_out,
                    h3_gpu_tensor *key_out, h3_gpu_tensor *value_out,
                    const h3_gpu_tensor *input,
                    const h3_gpu_tensor *query_weight,
                    const h3_gpu_tensor *key_weight,
                    const h3_gpu_tensor *value_weight, uint32_t rows,
                    uint32_t dim);

/* Causal GQA with the SDPA score-scaling convention (scale applied to
 * the fp32 scores after the dot product). */
int qw_gpu_gqa_causal_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query,
                           const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value, uint32_t sequence,
                           uint32_t query_heads, uint32_t kv_heads,
                           uint32_t head_dim, float scale);

/* Full-bidirectional SDPA whose query length may differ from the
 * key/value length (image-block rows against the full text+image prefix). */
int qw_gpu_sdpa_rect_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *query,
                          const h3_gpu_tensor *key,
                          const h3_gpu_tensor *value,
                          uint32_t query_sequence, uint32_t kv_sequence,
                          uint32_t heads, uint32_t head_dim, float scale);

/* ---- Qwen3-VL vision tower (bf16 activations, fp32 elementwise math) ---- */

/* Affine LayerNorm, one threadgroup per row (width <= 4608). */
int qw_gpu_vision_ln_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t rows,
                          uint32_t width, float epsilon);

/* GELU (tanh approximation). */
int qw_gpu_vision_gelu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements);

/* Elementwise add (residual). */
int qw_gpu_vision_add_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *left,
                           const h3_gpu_tensor *right, uint32_t elements);

/* Split the row-fused [rows, 3C] qkv into one [rows, C] part
 * (part_index: 0=q 1=k 2=v). */
int qw_gpu_vision_split3_bf16(h3_gpu *gpu, h3_gpu_tensor *part,
                              const h3_gpu_tensor *qkv, uint32_t rows,
                              uint32_t channels, uint32_t part_index);

/* Concatenate every 4 consecutive tokens: [rows, C] -> [rows/4, 4C]. */
int qw_gpu_vision_merge2x2_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t channels);

/* Axial half-split RoPE with F32 cos/sin tables ([rows, head_dim/2]),
 * fp32 math + single bf16 output rounding (vision semantics, unlike the
 * text tower's quantized-table double rounding). */
int qw_gpu_vision_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *rope_cos,
                            const h3_gpu_tensor *rope_sin, uint32_t rows,
                            uint32_t heads, uint32_t head_dim);

/* ---- yue-metal additions: 1D acoustic VAE decoder (Oobleck, fp32) ---- */

/* 1D zero ring: [t_in, C] rows copied at row `offset` into a
 * [ring_rows, C] buffer, the rest zero-filled. */
int yv_gpu_pad_ring_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input, uint32_t t_in,
                        uint32_t channels, uint32_t offset,
                        uint32_t ring_rows);

/* [cout, taps, cin] weight -> [taps, cout_pad, cin] GEMM layout. */
int yv_gpu_conv_repack_f32(h3_gpu *gpu, h3_gpu_tensor *w_r,
                           const h3_gpu_tensor *w, uint32_t in_channels,
                           uint32_t out_channels, uint32_t out_pad,
                           uint32_t taps);

/* Fill [rows, channels] with the bias row (or zeros when bias is NULL). */
int yv_gpu_fill_bias_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *bias, uint32_t rows,
                         uint32_t channels);

/* SnakeBeta, elementwise, in-place safe. */
int yv_gpu_snake_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, const h3_gpu_tensor *ea,
                     const h3_gpu_tensor *eb, uint32_t count,
                     uint32_t channels);

/* Elementwise tanh. */
int yv_gpu_tanh_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t count);

/* Conv1d as k accumulating matmul2d taps. `ring` is the zero-ringed
 * activation of m_span + (taps-1)*dilation rows (head `pad` rows baked
 * in by the caller's yv_gpu_pad_ring_f32 offset); `w_r` the repacked
 * [taps, cout_pad, cin] weight. */
int yv_gpu_conv1d_mm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *ring, const h3_gpu_tensor *w_r,
                         const h3_gpu_tensor *bias, uint32_t t_out,
                         uint32_t in_channels, uint32_t out_channels,
                         uint32_t taps, uint32_t dilation);

/* One ConvTranspose1d tap: dense GEMM + strided accumulating write-back
 * at pos = pos0 + t*stride (bias pre-filled separately). */
int yv_gpu_convt_tap_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *ring, const h3_gpu_tensor *w_r,
                         uint32_t tap, uint32_t in_channels,
                         uint32_t out_channels, uint32_t stride,
                         uint32_t pos0, uint32_t lo, uint32_t rows,
                         uint32_t t_out, uint32_t m_span);

#endif
