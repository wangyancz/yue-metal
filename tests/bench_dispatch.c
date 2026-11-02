/* Dispatch-cost & bandwidth probe (M7 round 4, step 1: measure before
 * writing kernels). Everything runs inside ONE command buffer per
 * iteration, mirroring the merged-chain decode pipeline.
 *
 *   A. empty-dispatch cost: 1-element add (compute encoder) and 1-element
 *      copy (blit encoder), N back-to-back in one command buffer.
 *   B. production-style linear chain: 28 layers x {q,k,v,o,fc1,down} =
 *      168 dispatches ~= one branch step, one command buffer. Every layer
 *      owns distinct weights (~2.55 GB total, far beyond any cache), so
 *      gpu time / branch-step is honest DRAM bandwidth. MPS path vs every
 *      qw_gemv variant, outputs cross-checked bit-exact.
 *   C. qkv merge simulation: one 2048->4096 dispatch over a contiguous
 *      [q;k;v] weight vs three separate dispatches, same bytes.
 *   D. per-shape serial chains. E. small-kernel interleave tax.
 *   F. PRODUCTION decode chain (the only regime that matters): real data
 *      dependencies, layer-tail h3_gpu_continue (production cadence),
 *      optional CFG dual-branch, 2048 hot steps.
 *
 * Probe F conclusions (M7 round 5, mystery closed): the earlier
 * "probes win / production loses" contradiction was three stacked
 * artifacts — (1) B/D chains are fake-dependency (same input every
 * dispatch), so neighbouring kernels overlap; (2) the F-v1 MPS variant
 * had no layer-tail continue, so the MPSGraph encode serialized and its
 * numbers were fake by 2x in BOTH directions; (3) every production
 * GEMV A/B ran the DEFAULT H3_GEMV_KERNEL (64x1 unroll-2, ~230 GB/s)
 * because H3_GEMV_KERNEL was never set — u8 is ~15% faster than that
 * but still LOSES to MPS in the true serial chain (probe: 13.3 vs
 * 12.2 ms/branch; serve A/B: 17.4 vs 14.3 ms/branch-token). CFG dual
 * branch costs exactly 2x (no thermal/weight-heat penalty). Storage
 * mode (mmap bytesNoCopy vs private shared) is irrelevant for GEMV
 * (serve A/B identical). The serial-chain ceiling for a per-column
 * GEMV is ~222 GB/s vs MPS ~246 GB/s — a rewrite (MLX-style 2D block
 * + simdgroup reduction) could at best reach parity with MPS, so the
 * GEMV track is CLOSED at a ~1.1x theoretical ceiling. lm (32769 out)
 * GEMV differs from MPS by ulps (chain amplifies to a few bf16 bits in
 * the final logits; expected, distribution-equivalent).
 *
 * 2D-block rewrite verdict (2026-11-01, SECOND AND FINAL GEMV VETO):
 * the MLX-style family (qw_gemv2d_*, SM=1/SN=32 whole-simd-per-row-slab,
 * TM rows per thread, TN=8 -> uint4 loads, BN=1 so the reduction is a
 * simd-local shuffle tree with ZERO barriers; plus the fused
 * qw_gemv2d_qkv_* one-dispatch q/k/v prologue) was built, unit-mapped
 * (test_gemv A/B/C/E: e_k mapping for all 16 variants, row/tail mapping
 * incl. K-tail 2056 & row-tail 300, host-fp32 tolerance cross-check,
 * fused-2D-qkv bit-exact vs three 2D dispatches) and swept (probe S:
 * fixes the per-column small-matrix disaster — k/v 63 -> 471 GB/s,
 * q/o 261 -> 735 GB/s fake-dep; fc1/down/lm were already DRAM-bound and
 * stay flat). Probe F (the arbiter, three consistent runs):
 * all-2D(m64u8) 12.03, qkv2D+2D(m64u8) 11.99 vs all-MPS 12.19 and
 * shipping 12.18 ms/branch — a real but marginal +1.6%, nowhere near
 * the 1.6x lyra bandwidth gap, and NOT clearly below the 12.2 bar, so
 * per discipline the serve-default flip was not attempted beyond one
 * confirmation run. Production serve A/B (seed 831001): gpu busy
 * 27.34 -> 25.94 s but WALL e2e 29.8 -> 29.9 s (trajectories diverge,
 * 746 vs 747 tokens, wall is the arbiter) — no production gain.
 * Conclusion: the decode chain is no longer GEMV-bound. 2.95 GB of
 * weight traffic per branch floors the GEMMs at ~5.9 ms (~520 GB/s
 * DRAM); probe F runs ~12 ms, so ~6 ms is dispatch overhead +
 * interleaved small kernels + dependency bubbles — unreachable from the
 * GEMV side. The GEMV track is closed at MPS parity; the shipping
 * default (fused per-column qkv GEMV + MPS) stays. The qw_gemv2d_* /
 * qw_gemv2d_qkv_* kernels remain as env-selected, unit-tested variants
 * (H3_GEMV_KERNEL / H3_QKV_GEMV_KERNEL). Remaining semantic levers live
 * outside GEMV: whole-step graph capture / MPS mixing (M7 known item).
 *
 * Probe T/U conclusions (M10, decode SDPA landed as default): probe T
 * [cfg2] wall 24.03 ms/branch with flash2, sdpa->split 18.45 (sdpa share
 * 8.85 -> 5.58); probe U per-call kv=513 134 -> 28.5 us (kv=2048 470 ->
 * 50). Production serve A/B: semantic 38.1 -> 19.9 ms/token, e2e RTF
 * 1.24 -> 0.80. METHODOLOGY TRAP (fixed in probe U): the kv=1 mini-case
 * refilled the SHARED q tensor with constants, and a constant q makes a
 * per-thread-element score bug coincide bit-exactly with the true score —
 * the first cut of qw_sdpa_decode_* hoisted q[head*D+tid] out of the row
 * loop (score = q[tid]*Sigma_d k[r][d]), which the A/B then blessed as
 * "kv=1/64 three-way bit-exact" while production attention collapsed to
 * a near-uniform V average (ar gate 1.25 relL2). The mini-case now uses
 * dedicated tensors and kv=55 (the AR fixture's first decode step) is in
 * the sweep. Comparison metric note: with random q, kernel-vs-host
 * "worst-rel" shows ~0.2 with ~100/2048 elements >2% even for the
 * production flash2 kernel — near-zero output elements against the 1e-3
 * denominator floor, NOT a math bug; the broken signature was ~57 with
 * ~2000/2048 >2%.
 *
 * Graph-capture verdict (M12 probe, 2026-11-01 — the decode chain is
 * CLOSED to scheduling work). The "whole-step MPSGraph capture" pool was
 * measured with slope stubs (AF_SLOPE*: K dependent 1-element adds per
 * layer tail, production shape = fused prologue + add/rms fusion +
 * split SDPA). The real-chain marginal cost of one dispatch is
 * 0.89-1.08 us — the same as the all-stub skeleton (1.15) — so there is
 * NO hidden bubble pool: prod 18.44 ms/branch-step decomposes to ~93%
 * real execution (GEMM ~12 ms at the ~246 GB/s kernel-efficiency floor
 * both MPS and the rewritten 2D GEMV share, split SDPA ~5.5 ms, small
 * kernels ~1.1 ms) and only ~0.6 ms of dispatch tax across ~570
 * dispatches (~0.3 ms/branch-token, ~1.6% of semantic). MPSGraph cannot
 * represent the custom kernels (qkv GEMV / fused hn+rope / split SDPA /
 * rms / swiglu), their bf16 rounding order is locked by the
 * melody_abc-vs-lyra bit-exact gate, and the o/fc1/down adjacency needed
 * for per-layer matmul graphs is broken by those same kernels — so even
 * that 0.6 ms is reachable only through further manual fusion rounds
 * with ~0.1 ms menu items left. VERDICT: graph capture STOPPED per the
 * probe-first discipline; the decode chain is execution-bound end to
 * end. A hot-machine run can also scramble the fused-variant deltas
 * (h1/h2 disagreeing by ~0.8 ms) — trust only back-to-back runs with
 * agreeing halves.
 *
 * GEMM-track final audit (M13, 2026-11-02, fans pinned): the track the
 * M12 verdict left open ("kernel efficiency is the only lever") is now
 * CLOSED with a per-shape bandwidth calibration (probe G) run at the
 * same thermal state as the production A/B.
 *   Probe F re-run (all-MPS [1x] 12.43 vs M9's 12.19, shipping 12.61,
 *   all-2D 12.16, all-GEMV(u8) 13.35): probe numbers reproduce at full
 *   fan — the "246 GB/s floor" was NOT thermal throttling. Probe T
 *   re-run: full 24.37 (M10 24.03) but prod (f+split) 17.12 vs M12's
 *   18.44 — that delta IS thermal noise (T runs after F's 15 minutes).
 *   CORRECTION of the M12 attribution: 18.44 was the CFG-dual total, so
 *   "GEMM ~12 ms @246 GB/s" double-counted; per-branch GEMM is ~6 ms at
 *   ~470 GB/s effective. The prod-minus-gemm-stub delta (5.34 ms/2
 *   branches) is NOT a reliable GEMM share either (stubbing changes the
 *   chain's overlap structure).
 *   Probe G (new, 'G'): each production shape as a REAL 28-deep
 *   dependent chain (bridge add reads the dispatch output into the next
 *   input; a bridge-only twin subtracts the bridge; per-shape sum =
 *   honest GEMM floor). Results at full fan: fc1 2048->12288 100-107 us
 *   (471-503 GB/s), down 6144->2048 46-52 us (485-546), lm 2048->32769
 *   ~259 us (514-519) — the three big shapes (75% of the 100.7 MB/layer
 *   weight traffic) already run at 90-100% of the ~520 GB/s stream
 *   ceiling. The small shapes (q/k/v/o/qkv) show 612-2375 GB/s in probe
 *   G but that is L2 reuse (4 rotating weights = 16-34 MB fit L2); in
 *   production those caches are flushed by fc1/down every layer, which
 *   is exactly why the M9 2D kernels won every probe and lost every
 *   serve A/B. Synthesis: production per-branch GEMM = 28x(qkv 24.5 +
 *   o 16 + fc1 100 + down 52) us + lm 261 us ~= 5.8-6.1 ms vs the
 *   all-shapes-at-500-GB/s floor of ~5.9 ms — the chain is ON the GEMM
 *   floor; serve busy 97.9 s / 4850 tok = 8.6 ms/branch-token reconciles
 *   as GEMM 6 + split SDPA 1.6 + small kernels 0.6 + dispatch 0.3.
 *   Remaining GEMM upside <= ~0.3 ms/branch (~3% semantic) and would
 *   require winning the small shapes under real L2 pressure — the same
 *   race the M9 production A/B already lost twice. VERDICT: the GEMM
 *   track is closed; the semantic gap to lyra is NOT GEMM efficiency
 *   (lyra at ~394 GB/s would need 7.2 ms/branch for the same weights —
 *   slower than our 6); it lives in split-SDPA growth with KV length
 *   and the per-token host tail (logits readback + sampling).
 *
 * Run from the repository root: make bench-dispatch (or ./tests/bench_dispatch F
 * for probe F only, S for the 2D sweep, T for the timeline ablation,
 * U for the SDPA microbenchmark, G for the per-shape true-dependency
 * bandwidth calibration) */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "h3_gpu.h"
#include "yue_gpu_ext.h"

#define LAYERS 28

typedef struct {
    uint32_t in_dim, out_dim;
    const char *label;
} shape;

static const shape layer_shapes[] = {
    {2048, 2048, "q"},   {2048, 1024, "k"},    {2048, 1024, "v"},
    {2048, 2048, "o"},   {2048, 12288, "fc1"}, {6144, 2048, "down"},
};
#define NSHAPES 6

static const char *variants[] = {
    "qw_gemv_bf16",    "qw_gemv_bf16_u4",       "qw_gemv_bf16_u8",
    "qw_gemv_bf16_c2u4", "qw_gemv_bf16_g128u4", "qw_gemv_bf16_g128c2u4",
};

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float bf16_host(uint16_t bits) {
    uint32_t wide = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &wide, sizeof(value));
    return value;
}

static void fill_bf16(uint16_t *data, size_t count, uint32_t seed) {
    uint32_t state = seed;
    for (size_t i = 0; i < count; i++) {
        state = state * 1664525u + 1013904223u;
        float unit = (float)(state >> 8) / (float)(1 << 24) * 2.0f - 1.0f;
        data[i] = f32_to_bf16(unit);
    }
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static h3_gpu_tensor *make_weight(h3_gpu *gpu, uint32_t in_dim,
                                  uint32_t out_dim, uint32_t seed) {
    size_t count = (size_t)in_dim * out_dim;
    uint16_t *host = malloc(count * 2);
    if (!host) return NULL;
    fill_bf16(host, count, seed);
    h3_gpu_tensor *t = h3_gpu_tensor_from_bf16(gpu, host, count);
    free(host);
    return t;
}

/* ---------------- probe A: per-dispatch constants ---------------- */

static void probe_empty(h3_gpu *gpu) {
    h3_gpu_tensor *a = h3_gpu_tensor_new_bf16(gpu, 1);
    h3_gpu_tensor *b = h3_gpu_tensor_new_bf16(gpu, 1);
    h3_gpu_tensor *o = h3_gpu_tensor_new_bf16(gpu, 1);
    if (!a || !b || !o) exit(1);
    const int N = 4096, ITERS = 4;
    for (int i = 0; i < 2; i++) {
        if (h3_gpu_begin(gpu) && h3_gpu_add_bf16(gpu, o, a, b, 1))
            h3_gpu_submit(gpu);
    }
    h3_gpu_stats s0, s1;
    /* compute encoder */
    if (!h3_gpu_get_stats(gpu, &s0)) return;
    double t0 = now();
    for (int it = 0; it < ITERS; it++) {
        if (!h3_gpu_begin(gpu)) return;
        for (int i = 0; i < N; i++)
            if (!h3_gpu_add_bf16(gpu, o, a, b, 1)) return;
        if (!h3_gpu_submit(gpu)) return;
    }
    double wall = now() - t0;
    if (!h3_gpu_get_stats(gpu, &s1)) return;
    double gpu_us = (s1.gpu_seconds - s0.gpu_seconds) / (ITERS * N) * 1e6;
    double wall_us = wall / (ITERS * N) * 1e6;
    printf("A. compute add(1elem)  %4d dispatches/chain: gpu %6.2f us,"
           " wall %6.2f us /dispatch\n", N, gpu_us, wall_us);
    /* blit encoder */
    if (!h3_gpu_get_stats(gpu, &s0)) return;
    t0 = now();
    for (int it = 0; it < ITERS; it++) {
        if (!h3_gpu_begin(gpu)) return;
        for (int i = 0; i < N; i++)
            if (!h3_gpu_copy_bf16(gpu, o, 0, a, 0, 1)) return;
        if (!h3_gpu_submit(gpu)) return;
    }
    wall = now() - t0;
    if (!h3_gpu_get_stats(gpu, &s1)) return;
    gpu_us = (s1.gpu_seconds - s0.gpu_seconds) / (ITERS * N) * 1e6;
    wall_us = wall / (ITERS * N) * 1e6;
    printf("A. blit   copy(1elem)  %4d dispatches/chain: gpu %6.2f us,"
           " wall %6.2f us /dispatch\n", N, gpu_us, wall_us);
    /* mixed: the production layer mixes compute ops and two blit copies */
    if (!h3_gpu_get_stats(gpu, &s0)) return;
    t0 = now();
    for (int it = 0; it < ITERS; it++) {
        if (!h3_gpu_begin(gpu)) return;
        for (int i = 0; i < N / 4; i++)
            if (h3_gpu_add_bf16(gpu, o, a, b, 1))
                h3_gpu_copy_bf16(gpu, o, 0, a, 0, 1);
            else
                return;
        if (!h3_gpu_submit(gpu)) return;
    }
    wall = now() - t0;
    if (!h3_gpu_get_stats(gpu, &s1)) return;
    gpu_us = (s1.gpu_seconds - s0.gpu_seconds) / (ITERS * N) * 1e6;
    wall_us = wall / (ITERS * N) * 1e6;
    printf("A. mixed  add+copy     %4d dispatches/chain: gpu %6.2f us,"
           " wall %6.2f us /dispatch\n", N, gpu_us, wall_us);
    h3_gpu_tensor_free(a);
    h3_gpu_tensor_free(b);
    h3_gpu_tensor_free(o);
}

/* ---------------- probe B: 28-layer linear chain ---------------- */

typedef struct {
    h3_gpu_tensor *w[LAYERS * NSHAPES];
    h3_gpu_tensor *out[LAYERS * NSHAPES];
    h3_gpu_tensor *x;
} chain;

static int chain_run(h3_gpu *gpu, chain *c) {
    if (!h3_gpu_begin(gpu)) return 0;
    for (int l = 0; l < LAYERS; l++)
        for (int s = 0; s < NSHAPES; s++)
            if (!h3_gpu_linear_bf16(gpu, c->out[l * NSHAPES + s], c->x,
                                    c->w[l * NSHAPES + s], NULL, 1,
                                    layer_shapes[s].in_dim,
                                    layer_shapes[s].out_dim))
                return 0;
    return h3_gpu_submit(gpu);
}

static void probe_chain(h3_gpu *gpu) {
    chain c;
    memset(&c, 0, sizeof(c));
    c.x = h3_gpu_tensor_new_bf16(gpu, 6144);
    if (!c.x) exit(1);
    uint16_t *hx = malloc(6144 * 2);
    fill_bf16(hx, 6144, 0xabcdu);
    if (!h3_gpu_tensor_write_bf16(c.x, hx, 6144)) exit(1);
    free(hx);
    size_t bytes = 0;
    for (int l = 0; l < LAYERS; l++)
        for (int s = 0; s < NSHAPES; s++) {
            c.w[l * NSHAPES + s] = make_weight(gpu, layer_shapes[s].in_dim,
                                               layer_shapes[s].out_dim,
                                               0x1000u + (uint32_t)(l * 8 + s));
            c.out[l * NSHAPES + s] =
                h3_gpu_tensor_new_bf16(gpu, layer_shapes[s].out_dim);
            if (!c.w[l * NSHAPES + s] || !c.out[l * NSHAPES + s]) exit(1);
            bytes += (size_t)layer_shapes[s].in_dim *
                     layer_shapes[s].out_dim * 2;
        }
    printf("B. linear chain: %d layers x 6 proj = %d dispatches,"
           " %.2f GB weights\n", LAYERS, LAYERS * NSHAPES,
           (double)bytes / 1e9);

    const int ITERS = 8;
    uint16_t *ref = malloc((size_t)LAYERS * NSHAPES * 12288 * 2);
    uint16_t *got = malloc((size_t)LAYERS * NSHAPES * 12288 * 2);
    if (!ref || !got) exit(1);

    /* reference: MPS path */
    unsetenv("H3_GEMV");
    setenv("H3_DISABLE_MPS_LINEAR", "0", 1);
    if (!chain_run(gpu, &c)) {
        fprintf(stderr, "chain (mps) failed: %s\n", h3_gpu_error(gpu));
        exit(1);
    }
    for (int i = 0; i < LAYERS * NSHAPES; i++)
        if (!h3_gpu_tensor_read_bf16(c.out[i],
                                     ref + (size_t)i * 12288,
                                     layer_shapes[i % NSHAPES].out_dim))
            exit(1);
    h3_gpu_stats s0, s1;
    if (!h3_gpu_get_stats(gpu, &s0)) return;
    double t0 = now();
    for (int it = 0; it < ITERS; it++)
        if (!chain_run(gpu, &c)) exit(1);
    double wall = now() - t0;
    if (!h3_gpu_get_stats(gpu, &s1)) return;
    double gpu_ms = (s1.gpu_seconds - s0.gpu_seconds) / ITERS * 1e3;
    double wall_ms = wall / ITERS * 1e3;
    printf("B. %-22s gpu %6.3f ms wall %6.3f ms /branch-step"
           " (%5.1f GB/s gpu, %5.2f us/dispatch)\n", "MPS (shipping)",
           gpu_ms, wall_ms, (double)bytes / (gpu_ms * 1e-3) / 1e9,
           gpu_ms * 1e3 / (LAYERS * NSHAPES));

    for (size_t v = 0; v < sizeof(variants) / sizeof(variants[0]); v++) {
        setenv("H3_GEMV", "1", 1);
        setenv("H3_GEMV_KERNEL", variants[v], 1);
        if (!chain_run(gpu, &c)) {
            fprintf(stderr, "chain (%s) failed: %s\n", variants[v],
                    h3_gpu_error(gpu));
            exit(1);
        }
        int mismatch = 0;
        for (int i = 0; i < LAYERS * NSHAPES; i++) {
            if (!h3_gpu_tensor_read_bf16(c.out[i],
                                         got + (size_t)i * 12288,
                                         layer_shapes[i % NSHAPES].out_dim))
                exit(1);
            if (memcmp(got + (size_t)i * 12288, ref + (size_t)i * 12288,
                       (size_t)layer_shapes[i % NSHAPES].out_dim * 2))
                mismatch++;
        }
        if (!h3_gpu_get_stats(gpu, &s0)) return;
        t0 = now();
        for (int it = 0; it < ITERS; it++)
            if (!chain_run(gpu, &c)) exit(1);
        wall = now() - t0;
        if (!h3_gpu_get_stats(gpu, &s1)) return;
        gpu_ms = (s1.gpu_seconds - s0.gpu_seconds) / ITERS * 1e3;
        wall_ms = wall / ITERS * 1e3;
        printf("B. %-22s gpu %6.3f ms wall %6.3f ms /branch-step"
               " (%5.1f GB/s gpu, %5.2f us/dispatch) layer-mismatch %d/168\n",
               variants[v], gpu_ms, wall_ms,
               (double)bytes / (gpu_ms * 1e-3) / 1e9,
               gpu_ms * 1e3 / (LAYERS * NSHAPES), mismatch);
    }
    unsetenv("H3_GEMV");
    free(ref);
    free(got);
    for (int i = 0; i < LAYERS * NSHAPES; i++) {
        h3_gpu_tensor_free(c.w[i]);
        h3_gpu_tensor_free(c.out[i]);
    }
    h3_gpu_tensor_free(c.x);
}

/* ---------------- probe C: qkv merge ---------------- */

static void probe_qkv_merge(h3_gpu *gpu) {
    /* Wq [2048, 2048], Wk/Wv [1024, 2048]; Wbig [4096, 2048] holds the
     * same rows contiguously. One dispatch vs three. */
    uint32_t in_dim = 2048;
    h3_gpu_tensor *wq = make_weight(gpu, in_dim, 2048, 0x77a);
    h3_gpu_tensor *wk = make_weight(gpu, in_dim, 1024, 0x77b);
    h3_gpu_tensor *wv = make_weight(gpu, in_dim, 1024, 0x77c);
    h3_gpu_tensor *wbig = h3_gpu_tensor_new_bf16(gpu, (size_t)4096 * in_dim);
    h3_gpu_tensor *x = h3_gpu_tensor_new_bf16(gpu, in_dim);
    h3_gpu_tensor *outq = h3_gpu_tensor_new_bf16(gpu, 2048);
    h3_gpu_tensor *outk = h3_gpu_tensor_new_bf16(gpu, 1024);
    h3_gpu_tensor *outv = h3_gpu_tensor_new_bf16(gpu, 1024);
    h3_gpu_tensor *outbig = h3_gpu_tensor_new_bf16(gpu, 4096);
    if (!wq || !wk || !wv || !wbig || !x || !outq || !outk || !outv ||
        !outbig) exit(1);
    uint16_t *qbits = malloc(2048 * 2), *kbits = malloc(1024 * 2),
             *vbits = malloc(1024 * 2),
             *big = malloc((size_t)4096 * in_dim * 2),
             *hx = malloc(in_dim * 2);
    if (!qbits || !kbits || !vbits || !big || !hx) exit(1);
    /* wbig rows are per-OUT-row [in_dim] wide: copy the full weights. */
    uint16_t *qfull = malloc((size_t)2048 * in_dim * 2);
    uint16_t *kfull = malloc((size_t)1024 * in_dim * 2);
    uint16_t *vfull = malloc((size_t)1024 * in_dim * 2);
    if (!qfull || !kfull || !vfull) exit(1);
    if (!h3_gpu_tensor_read_bf16(wq, qfull, (size_t)2048 * in_dim) ||
        !h3_gpu_tensor_read_bf16(wk, kfull, (size_t)1024 * in_dim) ||
        !h3_gpu_tensor_read_bf16(wv, vfull, (size_t)1024 * in_dim)) exit(1);
    memcpy(big, qfull, (size_t)2048 * in_dim * 2);
    memcpy(big + (size_t)2048 * in_dim, kfull, (size_t)1024 * in_dim * 2);
    memcpy(big + (size_t)3072 * in_dim, vfull, (size_t)1024 * in_dim * 2);
    if (!h3_gpu_tensor_write_bf16(wbig, big, (size_t)4096 * in_dim)) exit(1);
    fill_bf16(hx, in_dim, 0x5151u);
    if (!h3_gpu_tensor_write_bf16(x, hx, in_dim)) exit(1);
    free(qfull); free(kfull); free(vfull);

    const int ITERS = 12, REPS = 28;
    /* one dispatch per layer */
    for (int i = 0; i < 2; i++) {
        if (!(h3_gpu_begin(gpu) &&
              h3_gpu_linear_bf16(gpu, outbig, x, wbig, NULL, 1, in_dim,
                                 4096) &&
              h3_gpu_submit(gpu))) exit(1);
    }
    h3_gpu_stats s0, s1;
    if (!h3_gpu_get_stats(gpu, &s0)) return;
    double t0 = now();
    for (int it = 0; it < ITERS; it++) {
        if (!h3_gpu_begin(gpu)) return;
        for (int r = 0; r < REPS; r++)
            if (!h3_gpu_linear_bf16(gpu, outbig, x, wbig, NULL, 1, in_dim,
                                    4096)) exit(1);
        if (!h3_gpu_submit(gpu)) return;
    }
    double wall1 = now() - t0;
    if (!h3_gpu_get_stats(gpu, &s1)) return;
    double gpu1 = (s1.gpu_seconds - s0.gpu_seconds) / ITERS * 1e3;
    /* three dispatches per layer */
    if (!h3_gpu_get_stats(gpu, &s0)) return;
    t0 = now();
    for (int it = 0; it < ITERS; it++) {
        if (!h3_gpu_begin(gpu)) return;
        for (int r = 0; r < REPS; r++)
            if (!(h3_gpu_linear_bf16(gpu, outq, x, wq, NULL, 1, in_dim,
                                     2048) &&
                  h3_gpu_linear_bf16(gpu, outk, x, wk, NULL, 1, in_dim,
                                     1024) &&
                  h3_gpu_linear_bf16(gpu, outv, x, wv, NULL, 1, in_dim,
                                     1024))) exit(1);
        if (!h3_gpu_submit(gpu)) return;
    }
    double wall3 = now() - t0;
    if (!h3_gpu_get_stats(gpu, &s1)) return;
    double gpu3 = (s1.gpu_seconds - s0.gpu_seconds) / ITERS * 1e3;
    /* numerics: split outputs must equal the merged output rows */
    uint16_t merged[4096];
    if (!h3_gpu_tensor_read_bf16(outbig, merged, 4096) ||
        !h3_gpu_tensor_read_bf16(outq, qbits, 2048) ||
        !h3_gpu_tensor_read_bf16(outk, kbits, 1024) ||
        !h3_gpu_tensor_read_bf16(outv, vbits, 1024)) exit(1);
    int mismatch = 0;
    for (uint32_t i = 0; i < 2048; i++)
        if (merged[i] != qbits[i]) mismatch++;
    for (uint32_t i = 0; i < 1024; i++)
        if (merged[2048 + i] != kbits[i]) mismatch++;
    for (uint32_t i = 0; i < 1024; i++)
        if (merged[3072 + i] != vbits[i]) mismatch++;
    printf("C. qkv merge (28 reps): 1x dispatch gpu %6.3f ms wall %6.3f ms"
           " | 3x dispatch gpu %6.3f ms wall %6.3f ms /28 | merged-vs-split"
           " element mismatch %d/4096\n",
           gpu1, wall1 / ITERS * 1e3, gpu3, wall3 / ITERS * 1e3, mismatch);
    free(big); free(hx);
    h3_gpu_tensor_free(wq); h3_gpu_tensor_free(wk); h3_gpu_tensor_free(wv);
    h3_gpu_tensor_free(wbig); h3_gpu_tensor_free(x);
    h3_gpu_tensor_free(outq); h3_gpu_tensor_free(outk);
    h3_gpu_tensor_free(outv); h3_gpu_tensor_free(outbig);
}

/* ---------------- probe D: per-shape serial chains ----------------
 * Production decode is a strictly serial dependency chain; every kernel
 * occupies the GPU alone. Time each projection shape alone in a 28-deep
 * chain (4 weights rotating, all DRAM-resident) through the MPS path vs
 * the best GEMV variants, to see which shapes actually win when they
 * cannot overlap with themselves. */

static void probe_shape_chain(h3_gpu *gpu) {
    struct {
        uint32_t in_dim, out_dim;
        const char *label;
    } shapes[] = {
        {2048, 2048, "q/o 2048->2048"},
        {2048, 1024, "k/v 2048->1024"},
        {2048, 12288, "fc1 2048->12288"},
        {6144, 2048, "down 6144->2048"},
        {2048, 32769, "lm semantic 2048->32769"},
    };
    const int DEPTH = 28, ITERS = 10;
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        uint32_t in_dim = shapes[s].in_dim, out_dim = shapes[s].out_dim;
        h3_gpu_tensor *x = h3_gpu_tensor_new_bf16(gpu, in_dim);
        h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, out_dim);
        h3_gpu_tensor *weights[4];
        for (int w = 0; w < 4; w++)
            weights[w] = make_weight(gpu, in_dim, out_dim,
                                     0x9000u + (uint32_t)(s * 4 + w));
        if (!x || !out || !weights[0] || !weights[1] || !weights[2] ||
            !weights[3]) exit(1);
        size_t bytes = (size_t)out_dim * in_dim * 2;
        struct { const char *name; int gemv; const char *kernel; } paths[] = {
            {"MPS", 0, NULL},
            {"gemv_u8", 1, "qw_gemv_bf16_u8"},
            {"gemv_u4", 1, "qw_gemv_bf16_u4"},
        };
        printf("D. %s (%.1f MB):\n", shapes[s].label,
               (double)bytes / 1e6);
        for (size_t p = 0; p < sizeof(paths) / sizeof(paths[0]); p++) {
            if (paths[p].gemv) {
                setenv("H3_GEMV", "1", 1);
                setenv("H3_GEMV_KERNEL", paths[p].kernel, 1);
            } else {
                unsetenv("H3_GEMV");
            }
            for (int i = 0; i < 2; i++) {
                if (!(h3_gpu_begin(gpu) &&
                      h3_gpu_linear_bf16(gpu, out, x, weights[0], NULL, 1,
                                         in_dim, out_dim) &&
                      h3_gpu_submit(gpu))) exit(1);
            }
            h3_gpu_stats st0, st1;
            if (!h3_gpu_get_stats(gpu, &st0)) return;
            double t0 = now();
            for (int it = 0; it < ITERS; it++) {
                if (!h3_gpu_begin(gpu)) return;
                for (int d = 0; d < DEPTH; d++)
                    if (!h3_gpu_linear_bf16(gpu, out, x, weights[d % 4],
                                            NULL, 1, in_dim, out_dim))
                        exit(1);
                if (!h3_gpu_submit(gpu)) return;
            }
            double wall = now() - t0;
            if (!h3_gpu_get_stats(gpu, &st1)) return;
            double gpu_us = (st1.gpu_seconds - st0.gpu_seconds) /
                            (ITERS * DEPTH) * 1e6;
            double wall_us = wall / (ITERS * DEPTH) * 1e6;
            printf("   %-9s gpu %7.2f us wall %7.2f us /dispatch"
                   " (%5.1f / %5.1f GB/s)\n", paths[p].name, gpu_us,
                   wall_us, (double)bytes / (gpu_us * 1e-6) / 1e9,
                   (double)bytes / (wall_us * 1e-6) / 1e9);
        }
        unsetenv("H3_GEMV");
        for (int w = 0; w < 4; w++) h3_gpu_tensor_free(weights[w]);
        h3_gpu_tensor_free(x);
        h3_gpu_tensor_free(out);
    }
}

/* ---------------- probe G: per-shape TRUE-dependency bandwidth ----------------
 * M12 follow-up (GEMM track, step 1: aim the denominator). Probes D/E are
 * fake-dependency (same input every dispatch, neighbours overlap) and lie
 * in both directions. G runs each production shape as a REAL 28-deep
 * dependent chain: dispatch -> a bridge add that reads the dispatch output
 * into the next dispatch's input (a += out[:min(in,out)]), so every layer
 * consumes its predecessor's result; a bridge-only twin chain subtracts
 * the bridge tax. Layer-tail h3_gpu_continue = production cadence, one
 * tail submit. Reports per-dispatch wall/gpu time and achieved GB/s over
 * the weight bytes only. The per-shape sum is the honest GEMM floor of
 * the production chain; the shape with the largest (floor - bytes/stream)
 * gap is where kernel-efficiency work pays. */
static void probe_shape_true_chain(h3_gpu *gpu) {
    struct { uint32_t in_dim, out_dim; const char *label; int qkv; } shapes[] = {
        {2048, 2048,   "q    2048->2048 ", 0},
        {2048, 1024,   "k/v  2048->1024 ", 0},
        {2048, 2048,   "o    2048->2048 ", 0},
        {2048, 12288,  "fc1  2048->12288", 0},
        {6144, 2048,   "down 6144->2048 ", 0},
        {2048, 32769,  "lm   2048->32769", 0},
        {2048, 4096,   "qkv  fused->4096", 1},
    };
    const int DEPTH = 28, ITERS = 24;
    struct { const char *name; int mode; const char *kernel;
             const char *qkvk; } paths[] = {
        {"MPS",      0, NULL,               NULL},
        {"gemv_u8",  1, "qw_gemv_bf16_u8",  "qw_gemv_qkv_bf16_u8"},
        {"2D m64u8", 1, "qw_gemv2d_m64u8",  "qw_gemv2d_qkv_m64u8"},
        {"TOPS",     2, NULL,               NULL},
    };
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        uint32_t in_dim = shapes[s].in_dim;
        uint32_t out_dim = shapes[s].qkv ? 4096 : shapes[s].out_dim;
        uint32_t bridge = in_dim < out_dim ? in_dim : out_dim;
        /* a is padded to 16 rows: the TensorOps GEMV reads a padded M tile
         * (garbage rows are consumed but never written back). */
        h3_gpu_tensor *a = h3_gpu_tensor_new_bf16(gpu,
                                                  (size_t)16 * in_dim);
        h3_gpu_tensor *p = h3_gpu_tensor_new_bf16(gpu, out_dim);
        h3_gpu_tensor *ko = NULL, *vo = NULL;
        h3_gpu_tensor *wq[4], *wk[4], *wv[4];
        h3_gpu_tensor *w[4];
        int have = a && p;
        int tops_ok_shape = (out_dim % 64) == 0;
        double bytes = (double)out_dim * in_dim * 2.0;
        if (shapes[s].qkv) {
            for (int i = 0; i < 4 && have; i++) {
                wq[i] = make_weight(gpu, 2048, 2048, 0xb000u + (uint32_t)i);
                wk[i] = make_weight(gpu, 2048, 1024, 0xb010u + (uint32_t)i);
                wv[i] = make_weight(gpu, 2048, 1024, 0xb020u + (uint32_t)i);
                have = have && wq[i] && wk[i] && wv[i];
            }
            ko = h3_gpu_tensor_new_bf16(gpu, 1024);
            vo = h3_gpu_tensor_new_bf16(gpu, 1024);
            have = have && ko && vo;
        } else {
            for (int i = 0; i < 4 && have; i++) {
                w[i] = make_weight(gpu, in_dim, out_dim,
                                   0xb100u + (uint32_t)i);
                have = have && w[i];
            }
        }
        if (!have) exit(1);
        printf("G. %s (%.1f MB/dispatch, bridge %u):\n", shapes[s].label,
               bytes / 1e6, bridge);
        for (size_t pi = 0; pi < sizeof(paths) / sizeof(paths[0]); pi++) {
            double gpu_us[2] = {0, 0}, wall_us[2] = {0, 0};
            if (paths[pi].mode == 1) {
                setenv("H3_GEMV", "1", 1);
                setenv("H3_GEMV_KERNEL", paths[pi].kernel, 1);
                if (shapes[s].qkv) {
                    setenv("H3_QKV_GEMV", "1", 1);
                    setenv("H3_QKV_GEMV_KERNEL", paths[pi].qkvk, 1);
                } else {
                    setenv("H3_QKV_GEMV", "0", 1);
                }
            } else {
                unsetenv("H3_GEMV");
                setenv("H3_QKV_GEMV", "0", 1);
            }
            if (paths[pi].mode == 2 && !tops_ok_shape) {
                printf("   %-9s skipped (output_dim %% 64 != 0)\n",
                       paths[pi].name);
                continue;
            }
            if (paths[pi].mode == 2) {
                /* one-shot correctness: MPS reference vs TOPS, host bits */
                h3_gpu_tensor *ref = h3_gpu_tensor_new_bf16(gpu, out_dim);
                if (!ref) exit(1);
                if (shapes[s].qkv) {
                    if (!(h3_gpu_begin(gpu) &&
                          h3_gpu_linear_bf16(gpu, ref, a, wq[0], NULL, 1,
                                             2048, 2048) &&
                          h3_gpu_submit(gpu) &&
                          h3_gpu_begin(gpu) &&
                          qw_gpu_tops_gemv_bf16(gpu, p, a, wq[0], 2048,
                                                2048) &&
                          h3_gpu_submit(gpu))) exit(1);
                } else {
                    if (!(h3_gpu_begin(gpu) &&
                          h3_gpu_linear_bf16(gpu, ref, a, w[0], NULL, 1,
                                             in_dim, out_dim) &&
                          h3_gpu_submit(gpu) &&
                          h3_gpu_begin(gpu) &&
                          qw_gpu_tops_gemv_bf16(gpu, p, a, w[0], in_dim,
                                                out_dim) &&
                          h3_gpu_submit(gpu))) exit(1);
                }
                size_t n = out_dim;
                uint16_t *hr = malloc(n * 2), *hp = malloc(n * 2);
                if (!hr || !hp ||
                    !h3_gpu_tensor_read_bf16(ref, hr, n) ||
                    !h3_gpu_tensor_read_bf16(p, hp, n)) exit(1);
                size_t diff = 0;
                int maxulp = 0;
                for (size_t i = 0; i < n; i++) {
                    if (hr[i] == hp[i]) continue;
                    diff++;
                    int d = hr[i] > hp[i] ? (int)(hr[i] - hp[i])
                                          : (int)(hp[i] - hr[i]);
                    if (d > maxulp) maxulp = d;
                }
                printf("   %-9s correctness vs MPS: %zu/%zu differ"
                       " (maxbit %d)\n", paths[pi].name, diff, n, maxulp);
                free(hr); free(hp);
                h3_gpu_tensor_free(ref);
            }
            for (int round = 0; round < 2; round++) {
                /* round 0 = dispatch + bridge, round 1 = bridge only */
                for (int i = 0; i < 2; i++) {
                    if (!(h3_gpu_begin(gpu) &&
                          h3_gpu_linear_bf16(gpu, p, a,
                                             shapes[s].qkv ? wq[0] : w[0],
                                             NULL, 1,
                                             shapes[s].qkv ? 2048u : in_dim,
                                             shapes[s].qkv ? 2048u : out_dim) &&
                          h3_gpu_submit(gpu))) exit(1);
                }
                h3_gpu_stats st0, st1;
                if (!h3_gpu_get_stats(gpu, &st0)) return;
                double t0 = now();
                for (int it = 0; it < ITERS; it++) {
                    if (!h3_gpu_begin(gpu)) return;
                    for (int d = 0; d < DEPTH; d++) {
                        if (!round) {
                            if (paths[pi].mode == 2) {
                                if (shapes[s].qkv) {
                                    if (!qw_gpu_tops_gemv_bf16(
                                            gpu, p, a, wq[d % 4], 2048,
                                            2048) ||
                                        !qw_gpu_tops_gemv_bf16(
                                            gpu, ko, a, wk[d % 4], 2048,
                                            1024) ||
                                        !qw_gpu_tops_gemv_bf16(
                                            gpu, vo, a, wv[d % 4], 2048,
                                            1024))
                                        exit(1);
                                } else if (!qw_gpu_tops_gemv_bf16(
                                               gpu, p, a, w[d % 4], in_dim,
                                               out_dim)) exit(1);
                            } else if (shapes[s].qkv) {
                                if (!qw_gpu_qkv_gemv_bf16(
                                        gpu, p, ko, vo, a, wq[d % 4],
                                        wk[d % 4], wv[d % 4], in_dim, 2048,
                                        1024))
                                    exit(1);
                            } else if (!h3_gpu_linear_bf16(
                                           gpu, p, a, w[d % 4], NULL, 1,
                                           in_dim, out_dim)) exit(1);
                        }
                        /* bridge: a += p[:bridge] closes the dependency
                         * chain (next dispatch reads a; the add reads its
                         * own writes, so nothing overlaps). */
                        if (!h3_gpu_add_bf16(gpu, a, a, p, bridge)) return;
                        if (d + 1 < DEPTH && !h3_gpu_continue(gpu)) return;
                    }
                    if (!h3_gpu_submit(gpu)) return;
                }
                double wall = now() - t0;
                if (!h3_gpu_get_stats(gpu, &st1)) return;
                wall_us[round] = wall / (ITERS * DEPTH) * 1e6;
                gpu_us[round] = (st1.gpu_seconds - st0.gpu_seconds) /
                                (ITERS * DEPTH) * 1e6;
            }
            double pw = wall_us[0] - wall_us[1];
            double pg = gpu_us[0] - gpu_us[1];
            printf("   %-9s net %6.2f us wall (%5.0f GB/s)  %6.2f us gpu"
                   " (%5.0f GB/s)  [bridge %.2f/%.2f us]\n",
                   paths[pi].name, pw, pw > 0 ? bytes / (pw * 1e-6) / 1e9 : 0,
                   pg, pg > 0 ? bytes / (pg * 1e-6) / 1e9 : 0,
                   wall_us[1], gpu_us[1]);
        }
        unsetenv("H3_GEMV");
        unsetenv("H3_QKV_GEMV");
        unsetenv("H3_QKV_GEMV_KERNEL");
        if (shapes[s].qkv) {
            for (int i = 0; i < 4; i++) {
                h3_gpu_tensor_free(wq[i]);
                h3_gpu_tensor_free(wk[i]);
                h3_gpu_tensor_free(wv[i]);
            }
            h3_gpu_tensor_free(ko);
            h3_gpu_tensor_free(vo);
        } else {
            for (int i = 0; i < 4; i++) h3_gpu_tensor_free(w[i]);
        }
        h3_gpu_tensor_free(a);
        h3_gpu_tensor_free(p);
    }
}

/* ---------------- probe H: rows-batch efficiency ----------------
 * CFG batch fusion premise check (M15 candidate): the two CFG branches
 * currently run serially and each reads the full 100.7 MB/layer weight
 * stream. If the MPS linear costs ~= the same gpu time at rows=2 as at
 * rows=1 (one weight read serving both branch rows), merging the
 * branches into one [2, H] batch would halve the GEMM time (~5-6
 * ms/token). gpu-window metric only — dispatch overlap makes wall
 * dishonest (M8 lesson). */
static void probe_batch_rows(h3_gpu *gpu) {
    struct { uint32_t in_dim, out_dim; const char *label; } shapes[] = {
        {2048, 2048,   "o 2048->2048  "},
        {2048, 12288,  "fc1 2048->12288"},
        {6144, 2048,   "down 6144->2048"},
        {2048, 32769,  "lm 2048->32769 "},
    };
    const int DEPTH = 28, ITERS = 64;
    const uint32_t ROWS[] = {1, 2, 4};
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        uint32_t in_dim = shapes[s].in_dim, out_dim = shapes[s].out_dim;
        double bytes = (double)out_dim * in_dim * 2.0;
        h3_gpu_tensor *x = h3_gpu_tensor_new_bf16(gpu,
                                                  (size_t)4 * in_dim);
        h3_gpu_tensor *y = h3_gpu_tensor_new_bf16(gpu,
                                                  (size_t)4 * out_dim);
        h3_gpu_tensor *weights[4];
        for (int w = 0; w < 4; w++)
            weights[w] = make_weight(gpu, in_dim, out_dim,
                                     0xc000u + (uint32_t)s * 4 + (uint32_t)w);
        if (!x || !y || !weights[0] || !weights[1] || !weights[2] ||
            !weights[3]) exit(1);
        printf("H. %s (%.1f MB weight/dispatch):\n", shapes[s].label,
               bytes / 1e6);
        /* numeric gate: rows=2 output must match two rows=1 runs on the
         * same two input rows (M10 lesson: timing probes can bless a
         * broken path; anchor the numbers). */
        {
            uint16_t *xr = malloc((size_t)2 * in_dim * 2);
            uint16_t *y2 = malloc((size_t)2 * out_dim * 2);
            uint16_t *y1a = malloc((size_t)out_dim * 2);
            uint16_t *y1b = malloc((size_t)out_dim * 2);
            if (!xr || !y2 || !y1a || !y1b) exit(1);
            fill_bf16(xr, (size_t)2 * in_dim, 0x51515);
            h3_gpu_tensor *x2 = h3_gpu_tensor_from_bf16(
                gpu, xr, (size_t)2 * in_dim);
            h3_gpu_tensor *yy2 =
                h3_gpu_tensor_new_bf16(gpu, (size_t)2 * out_dim);
            h3_gpu_tensor *yy1 =
                h3_gpu_tensor_new_bf16(gpu, out_dim);
            if (!x2 || !yy2 || !yy1) exit(1);
            if (!(h3_gpu_begin(gpu) &&
                  h3_gpu_linear_bf16(gpu, yy2, x2, weights[0], NULL, 2,
                                     in_dim, out_dim) &&
                  h3_gpu_submit(gpu)))
                exit(1);
            /* row-0 reference: rows=1 on the first input row */
            h3_gpu_tensor *x1 = h3_gpu_tensor_from_bf16(gpu, xr, in_dim);
            if (!x1 ||
                !(h3_gpu_begin(gpu) &&
                  h3_gpu_linear_bf16(gpu, yy1, x1, weights[0], NULL, 1,
                                     in_dim, out_dim) &&
                  h3_gpu_submit(gpu)) ||
                !h3_gpu_tensor_read_bf16(yy1, y1b, out_dim))
                exit(1);
            /* row-1 reference: rows=1 on the second input row */
            h3_gpu_tensor *x2r = h3_gpu_tensor_from_bf16(gpu, xr + in_dim,
                                                         in_dim);
            if (!x2r ||
                !(h3_gpu_begin(gpu) &&
                  h3_gpu_linear_bf16(gpu, yy1, x2r, weights[0], NULL, 1,
                                     in_dim, out_dim) &&
                  h3_gpu_submit(gpu)))
                exit(1);
            if (!h3_gpu_tensor_read_bf16(yy2, y2, (size_t)2 * out_dim) ||
                !h3_gpu_tensor_read_bf16(yy1, y1a, out_dim)) exit(1);
            size_t d0 = 0, d1 = 0;
            int m0 = 0, m1 = 0;
            for (uint32_t i = 0; i < out_dim; i++) {
                if (y1b[i] != y2[i]) {
                    d0++;
                    int dv = y1b[i] > y2[i] ? (int)(y1b[i] - y2[i])
                                            : (int)(y2[i] - y1b[i]);
                    if (dv > m0) m0 = dv;
                }
                if (y1a[i] != y2[out_dim + i]) {
                    d1++;
                    int dv = y1a[i] > y2[out_dim + i]
                                 ? (int)(y1a[i] - y2[out_dim + i])
                                 : (int)(y2[out_dim + i] - y1a[i]);
                    if (dv > m1) m1 = dv;
                }
            }
            printf("   numeric: row0 %zu/%zu differ (maxbit %d),"
                   " row1 %zu/%zu (maxbit %d)\n", d0, (size_t)out_dim, m0,
                   d1, (size_t)out_dim, m1);
            h3_gpu_tensor_free(x2);
            h3_gpu_tensor_free(x1);
            h3_gpu_tensor_free(x2r);
            h3_gpu_tensor_free(yy2);
            h3_gpu_tensor_free(yy1);
            free(xr); free(y2); free(y1a); free(y1b);
        }
        double base = 0;
        for (size_t ri = 0; ri < sizeof(ROWS) / sizeof(ROWS[0]); ri++) {
            uint32_t rows = ROWS[ri];
            for (int i = 0; i < 2; i++) {
                if (!(h3_gpu_begin(gpu) &&
                      h3_gpu_linear_bf16(gpu, y, x, weights[0], NULL, rows,
                                         in_dim, out_dim) &&
                      h3_gpu_submit(gpu))) exit(1);
            }
            h3_gpu_stats st0, st1;
            if (!h3_gpu_get_stats(gpu, &st0)) return;
            for (int it = 0; it < ITERS; it++) {
                if (!h3_gpu_begin(gpu)) return;
                for (int d = 0; d < DEPTH; d++)
                    if (!h3_gpu_linear_bf16(gpu, y, x, weights[d % 4],
                                            NULL, rows, in_dim, out_dim))
                        exit(1);
                if (!h3_gpu_submit(gpu)) return;
            }
            if (!h3_gpu_get_stats(gpu, &st1)) return;
            double gpu_us = (st1.gpu_seconds - st0.gpu_seconds) /
                            (ITERS * DEPTH) * 1e6;
            if (rows == 1) base = gpu_us;
            printf("   rows=%u gpu %7.2f us (%5.0f GB/s weight)"
                   "  batch-eff %.2fx\n", rows, gpu_us,
                   bytes / (gpu_us * 1e-6) / 1e9,
                   base > 0 ? gpu_us / base : 0);
        }
        for (int w = 0; w < 4; w++) h3_gpu_tensor_free(weights[w]);
        h3_gpu_tensor_free(x);
        h3_gpu_tensor_free(y);
    }
}

/* ---------------- probe E: interleave tax ----------------
 * The decode chain alternates projections with ~13 small kernels per
 * layer (norms, rope, copies, sdpa, adds, swiglu). Measure what those
 * interleaved small dispatches cost the surrounding GEMV stream: a
 * plain 28-deep q-projection GEMV_u8 chain vs the same chain with one
 * 2048-element add after every projection (the smallest realistic
 * neighbor kernel). */

static void probe_interleave(h3_gpu *gpu) {
    const int DEPTH = 28, ITERS = 8;
    h3_gpu_tensor *x = h3_gpu_tensor_new_bf16(gpu, 2048);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, 2048);
    h3_gpu_tensor *add_a = h3_gpu_tensor_new_bf16(gpu, 2048);
    h3_gpu_tensor *add_b = h3_gpu_tensor_new_bf16(gpu, 2048);
    h3_gpu_tensor *weights[4];
    for (int w = 0; w < 4; w++)
        weights[w] = make_weight(gpu, 2048, 2048, 0xa000u + (uint32_t)w);
    if (!x || !out || !add_a || !add_b || !weights[0] || !weights[1] ||
        !weights[2] || !weights[3]) exit(1);
    setenv("H3_GEMV", "1", 1);
    setenv("H3_GEMV_KERNEL", "qw_gemv_bf16_u8", 1);
    for (int variant = 0; variant < 2; variant++) {
        for (int i = 0; i < 2; i++) {
            if (!(h3_gpu_begin(gpu) &&
                  h3_gpu_linear_bf16(gpu, out, x, weights[0], NULL, 1, 2048,
                                     2048) &&
                  h3_gpu_submit(gpu))) exit(1);
        }
        h3_gpu_stats st0, st1;
        if (!h3_gpu_get_stats(gpu, &st0)) return;
        double t0 = now();
        for (int it = 0; it < ITERS; it++) {
            if (!h3_gpu_begin(gpu)) return;
            for (int d = 0; d < DEPTH; d++) {
                if (!h3_gpu_linear_bf16(gpu, out, x, weights[d % 4], NULL,
                                        1, 2048, 2048)) exit(1);
                if (variant &&
                    !h3_gpu_add_bf16(gpu, add_a, add_a, add_b, 2048))
                    exit(1);
            }
            if (!h3_gpu_submit(gpu)) return;
        }
        double wall = now() - t0;
        if (!h3_gpu_get_stats(gpu, &st1)) return;
        double gpu_us = (st1.gpu_seconds - st0.gpu_seconds) /
                        (ITERS * DEPTH) * 1e6;
        printf("E. q-chain %s small kernel: gpu %7.2f us wall %7.2f us"
               " /projection\n",
               variant ? "+ 1x add(2048)" : "plain          ", gpu_us,
               wall / (ITERS * DEPTH) * 1e6);
    }
    unsetenv("H3_GEMV");
    for (int w = 0; w < 4; w++) h3_gpu_tensor_free(weights[w]);
    h3_gpu_tensor_free(x);
    h3_gpu_tensor_free(out);
    h3_gpu_tensor_free(add_a);
    h3_gpu_tensor_free(add_b);
}

/* ---------------- probe F: production decode chain ----------------
 * probe B/D chains feed every dispatch the same input (no true data
 * dependency), so the GPU may overlap neighbouring kernels and their
 * bandwidth numbers are not the production regime. This probe replays
 * the actual decode topology — per layer: rms, qkv, head norms, rope,
 * KV blits, flash SDPA, o, add, rms, fc1, swiglu, down, add — chained
 * through real data dependencies, closed by the lm projection and the
 * logits readback that drains the chain every branch-step.
 *
 * Variants: shipping (fused-qkv GEMV + MPS), all-MPS, all-GEMV(u8);
 * each also runs without the logits readback to isolate the per-step
 * drain. Half-split walls expose any clock-throttling drift. */

static uint16_t f_bits[32769];

typedef struct {
    h3_gpu_tensor *q, *k, *v, *o, *fc1, *down;
    h3_gpu_tensor *in_n, *post_n, *qn, *kn;
} flayer;

typedef struct {
    int layers;
    int dual; /* CFG mirror: two forwards per branch-step */
    uint32_t hidden, kv, mlp, vocab, heads, kvh, hdim, offset, kv_rows;
    h3_gpu_tensor *x, *x_neg, *normed, *q, *k, *v, *attn, *branch, *fc1out,
        *mlp_mid, *logits, *keys, *values, *limit, *rope_cos, *rope_sin, *lm,
        *zero, *sdpa_partial, *slope;
    flayer *ly;
} fctx;

/* Timeline-ablation flags (probe T): every variant keeps the true
 * dependency chain closed — a removed kernel's consumer is re-wired to
 * the producer's output, and replacements are in-place self-dependent
 * adds — because fake-dependency chains overlap and lie (M9 lesson). */
#define AF_COMPUTE_COPY  0x01u /* KV blits -> compute add (no blit switch) */
#define AF_NO_HNORM_ROPE 0x02u /* drop head_norm x2 + rope x2 (q/k straight) */
#define AF_NO_RMS        0x04u /* drop both rms; consumers read the add output */
#define AF_NO_ADD        0x08u /* drop both residual adds; rms reads branch */
#define AF_NO_SWIGLU     0x10u /* drop swiglu; down reads fc1out head */
#define AF_NO_SDPA       0x20u /* sdpa -> add(attn = q + zero) */
#define AF_SDPA_SPLIT    0x100u /* sdpa -> split-K decode pair (probe U) */
#define AF_GEMM_STUB     0x40u /* the 4 GEMMs + lm -> chained adds */
#define AF_ALL_STUB      0x80u /* every dispatch -> in-place add */
#define AF_HN_ROPE_FUSED 0x200u /* head norms + rope -> 1 combined dispatch */
#define AF_HN_ROPE_FUSED2 0x800u /* head norms + rope -> 2 per-tensor dispatches */
#define AF_ADD_RMS_FUSED 0x400u /* residual add + rms -> fused kernel */
#define AF_SLOPE4        0x1000u /* +4 dependent no-op adds per layer */
#define AF_SLOPE16       0x2000u /* +16 per layer */
#define AF_SLOPE64       0x4000u /* +64 per layer */
/* AF_HN_ROPE_FUSED / AF_ADD_RMS_FUSED compose with each other but not
 * with the AF_NO_* ablations of the kernels they replace. The slope
 * stubs measure the REAL-chain marginal cost of one dispatch (the
 * all-stub 1.15 us/dispatch is a skeleton-only figure): K extra in-place
 * 1-element adds that read the branch tail's xb, so they cannot overlap
 * with real work. Marginal slope x (~450 dispatches/branch-step) bounds
 * the pool any graph-capture/scheduling work could reclaim. */

static int run_step_f(h3_gpu *g, fctx *c, int drain, unsigned flags) {
    const uint32_t H = c->hidden, KV = c->kv, MLP = c->mlp;
    const uint32_t HEADS = c->heads, KVH = c->kvh, HDIM = c->hdim;
    const uint32_t OFF = c->offset;
    /* Two forwards per branch-step mirror the CFG decode chain (the
     * negative branch re-reads the same weights with its own hidden
     * stream); dual == 0 runs the plain single branch. */
    int branches = c->dual ? 2 : 1;
    int stub_gemm = (flags & (AF_GEMM_STUB | AF_ALL_STUB)) != 0;
    if (!h3_gpu_begin(g)) return 0;
    for (int b = 0; b < branches; b++) {
        h3_gpu_tensor *xb = b ? c->x_neg : c->x;
        for (int l = 0; l < c->layers; l++) {
            if (flags & AF_ALL_STUB) {
                /* The production 16-dispatch layer shape with every kernel
                 * replaced by an add; each dispatch consumes its producer's
                 * output (self-dependent in-place where sizes demand), so
                 * nothing overlaps: pure front-end tax + bubbles. The qkv
                 * slot spreads to 3 chained adds (the fused GEMV is one);
                 * left operands widen via keys/values to keep sizes legal. */
                if (!h3_gpu_add_bf16(g, c->normed, c->normed, c->zero, H) ||
                    !h3_gpu_add_bf16(g, c->q, c->normed, c->zero, H) ||
                    !h3_gpu_add_bf16(g, c->k, c->q, c->zero, KV) ||
                    !h3_gpu_add_bf16(g, c->v, c->k, c->zero, KV) ||
                    !h3_gpu_add_bf16(g, c->q, c->attn, c->zero, H) ||
                    !h3_gpu_add_bf16(g, c->k, c->q, c->zero, KV) ||
                    !h3_gpu_add_bf16(g, c->q, c->q, c->zero, H) ||
                    !h3_gpu_add_bf16(g, c->k, c->k, c->zero, KV) ||
                    !h3_gpu_add_bf16(g, c->keys, c->k, c->zero, KV) ||
                    !h3_gpu_add_bf16(g, c->values, c->keys, c->zero, KV) ||
                    !h3_gpu_add_bf16(g, c->attn, c->keys, c->zero, H) ||
                    !h3_gpu_add_bf16(g, xb, c->attn, c->zero, H) ||
                    !h3_gpu_add_bf16(g, xb, xb, c->zero, H) ||
                    !h3_gpu_add_bf16(g, c->normed, xb, c->zero, H) ||
                    !h3_gpu_add_bf16(g, c->fc1out, c->keys, c->zero,
                                     2 * MLP) ||
                    !h3_gpu_add_bf16(g, c->mlp_mid, c->fc1out, c->zero, MLP) ||
                    !h3_gpu_add_bf16(g, xb, c->mlp_mid, c->zero, H) ||
                    !h3_gpu_add_bf16(g, xb, xb, c->zero, H)) return 0;
            } else {
                h3_gpu_tensor *qkv_in = (flags & AF_NO_RMS) ? xb : c->normed;
                /* AF_ADD_RMS_FUSED folds the previous layer's tail add
                 * into this input rms, so only layer 0 still runs one. */
                if (!(flags & AF_NO_RMS) &&
                    (l == 0 || !(flags & AF_ADD_RMS_FUSED)) &&
                    !qw_gpu_rms_norm_bf16(g, c->normed, xb, c->ly[l].in_n, 1,
                                          H, 1e-6f)) return 0;
                if (stub_gemm) {
                    /* Keep the qkv slot one dispatch; chain q<-in, k<-q,
                     * v<-k so no stage runs without a real producer. */
                    if (!h3_gpu_add_bf16(g, c->q, qkv_in, c->zero, H) ||
                        !h3_gpu_add_bf16(g, c->k, c->q, c->zero, KV) ||
                        !h3_gpu_add_bf16(g, c->v, c->k, c->zero, KV))
                        return 0;
                } else if (!qw_gpu_qkv_gemv_bf16(g, c->q, c->k, c->v, qkv_in,
                                                 c->ly[l].q, c->ly[l].k,
                                                 c->ly[l].v, H, H, KV)) {
                    if (!(h3_gpu_linear_bf16(g, c->q, qkv_in, c->ly[l].q,
                                             NULL, 1, H, H) &&
                          h3_gpu_linear_bf16(g, c->k, qkv_in, c->ly[l].k,
                                             NULL, 1, H, KV) &&
                          h3_gpu_linear_bf16(g, c->v, qkv_in, c->ly[l].v,
                                             NULL, 1, H, KV))) return 0;
                }
                if (!(flags & AF_NO_HNORM_ROPE)) {
                    if (flags & AF_HN_ROPE_FUSED) {
                        if (!qw_gpu_head_rms_rope_bf16(
                                g, c->q, c->k, c->ly[l].qn, c->ly[l].kn,
                                c->rope_cos, c->rope_sin, 1, HEADS, KVH,
                                HDIM, 1e-6f)) return 0;
                    } else if (flags & AF_HN_ROPE_FUSED2) {
                        if (!qw_gpu_head_rms_rope_bf16(
                                g, c->q, c->q, c->ly[l].qn, c->ly[l].qn,
                                c->rope_cos, c->rope_sin, 1, HEADS, 0,
                                HDIM, 1e-6f) ||
                            !qw_gpu_head_rms_rope_bf16(
                                g, c->k, c->k, c->ly[l].kn, c->ly[l].kn,
                                c->rope_cos, c->rope_sin, 1, 0, KVH,
                                HDIM, 1e-6f)) return 0;
                    } else if (!qw_gpu_head_rms_norm_bf16(g, c->q, c->ly[l].qn,
                                                          1, HEADS, HDIM,
                                                          1e-6f) ||
                               !qw_gpu_head_rms_norm_bf16(g, c->k, c->ly[l].kn,
                                                          1, KVH, HDIM,
                                                          1e-6f) ||
                               !qw_gpu_qk_rope_half_bf16(g, c->q, c->q, c->q,
                                                         c->q, c->rope_cos,
                                                         c->rope_sin, 1,
                                                         HEADS, HDIM) ||
                               !qw_gpu_qk_rope_half_bf16(g, c->k, c->k, c->k,
                                                         c->k, c->rope_cos,
                                                         c->rope_sin, 1,
                                                         KVH, HDIM)) return 0;
                }
                if (flags & AF_COMPUTE_COPY) {
                    /* compute-path copy lands at the cache head; sdpa
                     * still reads the same kv_rows extent, so timing is
                     * honest and the blit/encoder switch is gone. */
                    if (!h3_gpu_add_bf16(g, c->keys, c->k, c->zero, KV) ||
                        !h3_gpu_add_bf16(g, c->values, c->keys, c->zero, KV))
                        return 0;
                } else if (!h3_gpu_copy_bf16(g, c->keys, OFF * KVH * HDIM,
                                             c->k, 0, KVH * HDIM) ||
                           !h3_gpu_copy_bf16(g, c->values,
                                             OFF * KVH * HDIM, c->v, 0,
                                             KVH * HDIM)) return 0;
                if (flags & AF_NO_SDPA) {
                    if (!h3_gpu_add_bf16(g, c->attn, c->q, c->zero, H))
                        return 0;
                } else if (flags & AF_SDPA_SPLIT) {
                    if (!qw_gpu_sdpa_decode_split_bf16(
                            g, c->attn, c->q, c->keys, c->values, c->kv_rows,
                            HEADS, KVH, HDIM, 1.0f / sqrtf((float)HDIM),
                            c->sdpa_partial, 96))
                        return 0;
                } else if (!qw_gpu_flash_sdpa_prefix_bf16(
                               g, c->attn, c->q, c->keys, c->values, c->limit,
                               1, c->kv_rows, HEADS, KVH, HDIM,
                               1.0f / sqrtf((float)HDIM))) return 0;
                if (stub_gemm) {
                    if (!h3_gpu_add_bf16(g, xb, c->attn, c->zero, H))
                        return 0;
                } else if (!h3_gpu_linear_bf16(g, c->branch, c->attn,
                                               c->ly[l].o, NULL, 1, H, H))
                    return 0;
                if (flags & AF_ADD_RMS_FUSED) {
                    if (!qw_gpu_add_rms_bf16(g, c->normed, xb, c->branch,
                                             c->ly[l].post_n, 1, H,
                                             1e-6f)) return 0;
                } else {
                    if (!(flags & AF_NO_ADD)) {
                        if (!h3_gpu_add_bf16(g, xb, xb, c->branch, H))
                            return 0;
                    } else if (!stub_gemm) {
                        xb = c->branch; /* rms/down now read the branch */
                    }
                    if (!(flags & AF_NO_RMS) &&
                        !qw_gpu_rms_norm_bf16(g, c->normed, xb,
                                              c->ly[l].post_n, 1, H,
                                              1e-6f)) return 0;
                }
                if (stub_gemm) {
                    /* keys is the widest producer already in this layer's
                     * dependency chain (the KV append), keeping the stub
                     * serially dependent without widening normed. */
                    if (!h3_gpu_add_bf16(g, c->fc1out, c->keys, c->zero,
                                         2 * MLP) ||
                        !h3_gpu_add_bf16(g, c->mlp_mid, c->fc1out, c->zero,
                                         MLP) ||
                        !h3_gpu_add_bf16(g, xb, c->mlp_mid, c->zero, H))
                        return 0;
                } else {
                    h3_gpu_tensor *mlp_in = (flags & AF_NO_RMS) ? xb
                                                                : c->normed;
                    if (!h3_gpu_linear_bf16(g, c->fc1out, mlp_in,
                                            c->ly[l].fc1, NULL, 1, H,
                                            2 * MLP)) return 0;
                    if (!(flags & AF_NO_SWIGLU) &&
                        !qw_gpu_swiglu_bf16(g, c->mlp_mid, c->fc1out, MLP,
                                            MLP)) return 0;
                    h3_gpu_tensor *down_in = (flags & AF_NO_SWIGLU)
                                                 ? c->fc1out : c->mlp_mid;
                    if (!h3_gpu_linear_bf16(g, c->branch, down_in,
                                            c->ly[l].down, NULL, 1, MLP, H))
                        return 0;
                }
                if (flags & AF_ADD_RMS_FUSED) {
                    /* Tail add folded with the next layer's input rms;
                     * the last layer keeps a plain add (lm reads xb). */
                    if (l + 1 < c->layers
                            ? !qw_gpu_add_rms_bf16(g, c->normed, xb,
                                                   c->branch,
                                                   c->ly[l + 1].in_n, 1, H,
                                                   1e-6f)
                            : !h3_gpu_add_bf16(g, xb, xb, c->branch, H))
                        return 0;
                } else if (!(flags & AF_NO_ADD)) {
                    if (!h3_gpu_add_bf16(g, xb, xb, c->branch, H)) return 0;
                } else if (!stub_gemm) {
                    xb = c->branch; /* next layer's rms reads branch */
                }
                if (flags & (AF_SLOPE4 | AF_SLOPE16 | AF_SLOPE64)) {
                    int slope = (flags & AF_SLOPE64) ? 64
                              : (flags & AF_SLOPE16) ? 16 : 4;
                    for (int s = 0; s < slope; s++)
                        if (!h3_gpu_add_bf16(g, c->slope, c->slope, xb, 1))
                            return 0;
                }
            }
            /* Production layers continue (commit-without-wait) at the
             * layer tail so GPU execution overlaps host encoding of the
             * next layer; without this the MPS graph encode serializes
             * and its probe numbers go fake. */
            if (l + 1 < c->layers && !h3_gpu_continue(g)) return 0;
        }
        if (stub_gemm) {
            if (!h3_gpu_add_bf16(g, c->logits, c->keys, c->zero, c->vocab))
                return 0;
        } else if (!h3_gpu_linear_bf16(g, c->logits, xb, c->lm, NULL, 1, H,
                                       c->vocab)) return 0;
    }
    if (!h3_gpu_submit(g)) return 0;
    if (drain &&
        !h3_gpu_tensor_read_bf16(c->logits, f_bits, c->vocab)) return 0;
    return 1;
}

static int run_step(h3_gpu *g, fctx *c, int drain) {
    return run_step_f(g, c, drain, 0);
}

/* Shared production-chain context (probe F and probe T). */
static int fctx_build(h3_gpu *gpu, fctx *c, int layers) {
    const uint32_t H = 2048, KV = 1024, MLP = 6144, VOCAB = 32769;
    const uint32_t HEADS = 16, KVH = 8, HDIM = 128;
    const uint32_t OFFSET = 512, KV_ROWS = OFFSET + 1;
    c->layers = layers;
    c->hidden = H;
    c->kv = KV;
    c->mlp = MLP;
    c->vocab = VOCAB;
    c->heads = HEADS;
    c->kvh = KVH;
    c->hdim = HDIM;
    c->offset = OFFSET;
    c->kv_rows = KV_ROWS;
    c->ly = calloc((size_t)layers, sizeof(flayer));
    if (!c->ly) return 0;
    for (int l = 0; l < layers; l++) {
        c->ly[l].q = make_weight(gpu, H, H, 0x100u + (uint32_t)l * 9u);
        c->ly[l].k = make_weight(gpu, H, KV, 0x101u + (uint32_t)l * 9u);
        c->ly[l].v = make_weight(gpu, H, KV, 0x102u + (uint32_t)l * 9u);
        c->ly[l].o = make_weight(gpu, H, H, 0x103u + (uint32_t)l * 9u);
        c->ly[l].fc1 = make_weight(gpu, H, 2 * MLP, 0x104u + (uint32_t)l * 9u);
        c->ly[l].down = make_weight(gpu, MLP, H, 0x105u + (uint32_t)l * 9u);
        c->ly[l].in_n = h3_gpu_tensor_new_bf16(gpu, H);
        c->ly[l].post_n = h3_gpu_tensor_new_bf16(gpu, H);
        c->ly[l].qn = h3_gpu_tensor_new_bf16(gpu, HDIM);
        c->ly[l].kn = h3_gpu_tensor_new_bf16(gpu, HDIM);
        if (!c->ly[l].q || !c->ly[l].k || !c->ly[l].v || !c->ly[l].o ||
            !c->ly[l].fc1 || !c->ly[l].down || !c->ly[l].in_n ||
            !c->ly[l].post_n || !c->ly[l].qn || !c->ly[l].kn) return 0;
    }
    c->lm = make_weight(gpu, H, VOCAB, 0x3111u);
    c->x = h3_gpu_tensor_new_bf16(gpu, H);
    c->x_neg = h3_gpu_tensor_new_bf16(gpu, H);
    c->normed = h3_gpu_tensor_new_bf16(gpu, H);
    c->q = h3_gpu_tensor_new_bf16(gpu, HEADS * HDIM);
    c->k = h3_gpu_tensor_new_bf16(gpu, KVH * HDIM);
    c->v = h3_gpu_tensor_new_bf16(gpu, KVH * HDIM);
    c->attn = h3_gpu_tensor_new_bf16(gpu, HEADS * HDIM);
    c->branch = h3_gpu_tensor_new_bf16(gpu, H);
    c->fc1out = h3_gpu_tensor_new_bf16(gpu, 2 * MLP);
    c->mlp_mid = h3_gpu_tensor_new_bf16(gpu, MLP);
    c->logits = h3_gpu_tensor_new_bf16(gpu, VOCAB);
    c->keys = h3_gpu_tensor_new_bf16(gpu,
                                     (size_t)layers * KV_ROWS * KVH * HDIM);
    c->values = h3_gpu_tensor_new_bf16(gpu,
                                       (size_t)layers * KV_ROWS * KVH * HDIM);
    c->rope_cos = h3_gpu_tensor_new_f32(gpu, HDIM / 2);
    c->rope_sin = h3_gpu_tensor_new_f32(gpu, HDIM / 2);
    c->zero = h3_gpu_tensor_new_bf16(gpu, VOCAB);
    uint32_t one = KV_ROWS;
    c->limit = h3_gpu_tensor_from_u32(gpu, &one, 1);
    c->sdpa_partial = h3_gpu_tensor_new_f32(
        gpu, (size_t)c->heads * 96 * (HDIM + 2));
    c->slope = h3_gpu_tensor_new_bf16(gpu, 1);
    if (!c->lm || !c->x || !c->normed || !c->q || !c->k || !c->v ||
        !c->attn || !c->branch || !c->fc1out || !c->mlp_mid || !c->logits ||
        !c->keys || !c->values || !c->limit || !c->rope_cos || !c->rope_sin ||
        !c->zero || !c->sdpa_partial || !c->slope) return 0;
    float *cosv = malloc(sizeof(float) * (HDIM / 2));
    float *sinv = malloc(sizeof(float) * (HDIM / 2));
    uint16_t *hx = malloc(H * 2);
    uint16_t *hz = calloc(VOCAB, 2);
    if (!cosv || !sinv || !hx || !hz) return 0;
    for (uint32_t i = 0; i < HDIM / 2; i++) {
        cosv[i] = 0.9f;
        sinv[i] = 0.1f;
    }
    int ok = h3_gpu_tensor_write_f32(c->rope_cos, cosv, HDIM / 2) &&
             h3_gpu_tensor_write_f32(c->rope_sin, sinv, HDIM / 2);
    fill_bf16(hx, H, 0xbeadu);
    ok = ok && h3_gpu_tensor_write_bf16(c->x, hx, H);
    fill_bf16(hx, H, 0xf00du);
    ok = ok && h3_gpu_tensor_write_bf16(c->x_neg, hx, H);
    ok = ok && h3_gpu_tensor_write_bf16(c->zero, hz, VOCAB);
    free(cosv);
    free(sinv);
    free(hx);
    free(hz);
    return ok;
}

static void fctx_release(h3_gpu *gpu, fctx *c) {
    (void)gpu;
    for (int l = 0; l < c->layers; l++) {
        h3_gpu_tensor_free(c->ly[l].q);
        h3_gpu_tensor_free(c->ly[l].k);
        h3_gpu_tensor_free(c->ly[l].v);
        h3_gpu_tensor_free(c->ly[l].o);
        h3_gpu_tensor_free(c->ly[l].fc1);
        h3_gpu_tensor_free(c->ly[l].down);
        h3_gpu_tensor_free(c->ly[l].in_n);
        h3_gpu_tensor_free(c->ly[l].post_n);
        h3_gpu_tensor_free(c->ly[l].qn);
        h3_gpu_tensor_free(c->ly[l].kn);
    }
    free(c->ly);
    h3_gpu_tensor_free(c->lm);
    h3_gpu_tensor_free(c->x);
    h3_gpu_tensor_free(c->x_neg);
    h3_gpu_tensor_free(c->normed);
    h3_gpu_tensor_free(c->q);
    h3_gpu_tensor_free(c->k);
    h3_gpu_tensor_free(c->v);
    h3_gpu_tensor_free(c->attn);
    h3_gpu_tensor_free(c->branch);
    h3_gpu_tensor_free(c->fc1out);
    h3_gpu_tensor_free(c->mlp_mid);
    h3_gpu_tensor_free(c->logits);
    h3_gpu_tensor_free(c->keys);
    h3_gpu_tensor_free(c->values);
    h3_gpu_tensor_free(c->limit);
    h3_gpu_tensor_free(c->rope_cos);
    h3_gpu_tensor_free(c->rope_sin);
    h3_gpu_tensor_free(c->zero);
    h3_gpu_tensor_free(c->sdpa_partial);
    h3_gpu_tensor_free(c->slope);
}

/* ---------------- M11: fused prologue bit-exactness ----------------
 * Direct A/B of the fused twins against their separate kernels on
 * IDENTICAL dedicated tensors (no GEMM in the loop, nothing shared —
 * the M10 constant-q pollution lesson). Checks decode (rows=1) and a
 * prefill-like (rows=4) shape for both fusions:
 *   head_rms+rope: q/k post norm+rope bits must match exactly;
 *   add+rms:       the updated residual AND the normed output must
 *                  match exactly (the norm reads the just-rounded sum).
 * Any mismatch aborts the probe run. */

static int fused_bitexact_check(h3_gpu *g) {
    const uint32_t HEADS = 16, KVH = 8, HDIM = 128, H = 2048;
    int all_ok = 1;
    for (int shape_i = 0; shape_i < 2; shape_i++) {
        uint32_t rows = shape_i == 0 ? 1 : 4;
        size_t qn = (size_t)rows * HEADS * HDIM;
        size_t kn = (size_t)rows * KVH * HDIM;
        size_t hn = (size_t)rows * H;
        h3_gpu_tensor *q = h3_gpu_tensor_new_bf16(g, qn);
        h3_gpu_tensor *k = h3_gpu_tensor_new_bf16(g, kn);
        h3_gpu_tensor *q2 = h3_gpu_tensor_new_bf16(g, qn);
        h3_gpu_tensor *k2 = h3_gpu_tensor_new_bf16(g, kn);
        h3_gpu_tensor *qnw = h3_gpu_tensor_new_bf16(g, HDIM);
        h3_gpu_tensor *knw = h3_gpu_tensor_new_bf16(g, HDIM);
        h3_gpu_tensor *cos_t = h3_gpu_tensor_new_f32(g,
                                                     (size_t)rows * (HDIM / 2));
        h3_gpu_tensor *sin_t = h3_gpu_tensor_new_f32(g,
                                                     (size_t)rows * (HDIM / 2));
        h3_gpu_tensor *x = h3_gpu_tensor_new_bf16(g, hn);
        h3_gpu_tensor *x2 = h3_gpu_tensor_new_bf16(g, hn);
        h3_gpu_tensor *branch = h3_gpu_tensor_new_bf16(g, hn);
        h3_gpu_tensor *weight = h3_gpu_tensor_new_bf16(g, H);
        h3_gpu_tensor *out1 = h3_gpu_tensor_new_bf16(g, hn);
        h3_gpu_tensor *out2 = h3_gpu_tensor_new_bf16(g, hn);
        uint16_t *hq = malloc(qn * 2), *hk = malloc(kn * 2);
        uint16_t *hx = malloc(hn * 2), *hb = malloc(hn * 2);
        float *hc = malloc(sizeof(float) * rows * (HDIM / 2));
        float *hs = malloc(sizeof(float) * rows * (HDIM / 2));
        uint16_t *hw = malloc(H * 2);
        uint16_t *hqw = malloc(HDIM * 2), *hkw = malloc(HDIM * 2);
        uint16_t *rq = malloc(qn * 2), *rk = malloc(kn * 2);
        uint16_t *rx = malloc(hn * 2), *ro = malloc(hn * 2);
        if (!q || !k || !q2 || !k2 || !qnw || !knw || !cos_t || !sin_t ||
            !x || !x2 || !branch || !weight || !out1 || !out2 || !hq || !hk ||
            !hx || !hb || !hc || !hs || !hw || !hqw || !hkw || !rq || !rk ||
            !rx || !ro) {
            fprintf(stderr, "T. bitexact alloc failed\n");
            exit(1);
        }
        fill_bf16(hq, qn, 0x11u);
        fill_bf16(hk, kn, 0x22u);
        fill_bf16(hx, hn, 0x33u);
        fill_bf16(hb, hn, 0x44u);
        for (uint32_t r = 0; r < rows; r++)
            for (uint32_t p = 0; p < HDIM / 2; p++) {
                hc[r * (HDIM / 2) + p] = 0.9f - 0.01f * (float)r;
                hs[r * (HDIM / 2) + p] = 0.1f + 0.01f * (float)r;
            }
        fill_bf16(hw, H, 0x55u);
        fill_bf16(hqw, HDIM, 0x66u);
        fill_bf16(hkw, HDIM, 0x77u);
        int ok = h3_gpu_tensor_write_bf16(q, hq, qn) &&
                 h3_gpu_tensor_write_bf16(k, hk, kn) &&
                 h3_gpu_tensor_write_bf16(q2, hq, qn) &&
                 h3_gpu_tensor_write_bf16(k2, hk, kn) &&
                 h3_gpu_tensor_write_bf16(qnw, hqw, HDIM) &&
                 h3_gpu_tensor_write_bf16(knw, hkw, HDIM) &&
                 h3_gpu_tensor_write_f32(cos_t, hc, rows * (HDIM / 2)) &&
                 h3_gpu_tensor_write_f32(sin_t, hs, rows * (HDIM / 2)) &&
                 h3_gpu_tensor_write_bf16(x, hx, hn) &&
                 h3_gpu_tensor_write_bf16(x2, hx, hn) &&
                 h3_gpu_tensor_write_bf16(branch, hb, hn) &&
                 h3_gpu_tensor_write_bf16(weight, hw, H);
        /* separate path: head norms + rope per tensor, then add + rms */
        ok = ok && h3_gpu_begin(g) &&
             qw_gpu_head_rms_norm_bf16(g, q, qnw, rows, HEADS, HDIM,
                                       1e-6f) &&
             qw_gpu_head_rms_norm_bf16(g, k, knw, rows, KVH, HDIM, 1e-6f) &&
             qw_gpu_qk_rope_half_bf16(g, q, q, q, q, cos_t, sin_t, rows,
                                      HEADS, HDIM) &&
             qw_gpu_qk_rope_half_bf16(g, k, k, k, k, cos_t, sin_t, rows,
                                      KVH, HDIM) &&
             h3_gpu_add_bf16(g, x, x, branch, (uint32_t)hn) &&
             qw_gpu_rms_norm_bf16(g, out1, x, weight, rows, H, 1e-6f) &&
             h3_gpu_submit(g);
        /* round 2: fused twins, combined q+k dispatch */
        ok = ok && h3_gpu_begin(g) &&
             qw_gpu_head_rms_rope_bf16(g, q2, k2, qnw, knw, cos_t, sin_t,
                                       rows, HEADS, KVH, HDIM, 1e-6f) &&
             qw_gpu_add_rms_bf16(g, out2, x2, branch, weight, rows, H,
                                 1e-6f) &&
             h3_gpu_submit(g);
        ok = ok && h3_gpu_tensor_read_bf16(q, rq, qn) &&
             h3_gpu_tensor_read_bf16(k, rk, kn) &&
             h3_gpu_tensor_read_bf16(q2, hq, qn) &&
             h3_gpu_tensor_read_bf16(k2, hk, kn) &&
             h3_gpu_tensor_read_bf16(x, rx, hn) &&
             h3_gpu_tensor_read_bf16(x2, hx, hn) &&
             h3_gpu_tensor_read_bf16(out1, ro, hn) &&
             h3_gpu_tensor_read_bf16(out2, hb, hn);
        size_t bad = 0;
        if (ok) {
            if (memcmp(rq, hq, qn * 2) != 0) bad++;
            if (memcmp(rk, hk, kn * 2) != 0) bad++;
            if (memcmp(rx, hx, hn * 2) != 0) bad++;
            if (memcmp(ro, hb, hn * 2) != 0) bad++;
        }
        /* round 3: fused twins, one dispatch per tensor */
        fill_bf16(hq, qn, 0x11u);
        fill_bf16(hk, kn, 0x22u);
        fill_bf16(hx, hn, 0x33u);
        ok = ok && h3_gpu_tensor_write_bf16(q2, hq, qn) &&
             h3_gpu_tensor_write_bf16(k2, hk, kn) &&
             h3_gpu_tensor_write_bf16(x2, hx, hn) &&
             h3_gpu_begin(g) &&
             qw_gpu_head_rms_rope_bf16(g, q2, q2, qnw, qnw, cos_t, sin_t,
                                       rows, HEADS, 0, HDIM, 1e-6f) &&
             qw_gpu_head_rms_rope_bf16(g, k2, k2, knw, knw, cos_t, sin_t,
                                       rows, 0, KVH, HDIM, 1e-6f) &&
             qw_gpu_add_rms_bf16(g, out2, x2, branch, weight, rows, H,
                                 1e-6f) &&
             h3_gpu_submit(g);
        ok = ok && h3_gpu_tensor_read_bf16(q2, hq, qn) &&
             h3_gpu_tensor_read_bf16(k2, hk, kn) &&
             h3_gpu_tensor_read_bf16(x2, hx, hn) &&
             h3_gpu_tensor_read_bf16(out2, hb, hn);
        if (ok) {
            if (memcmp(rq, hq, qn * 2) != 0) bad++;
            if (memcmp(rk, hk, kn * 2) != 0) bad++;
            if (memcmp(rx, hx, hn * 2) != 0) bad++;
            if (memcmp(ro, hb, hn * 2) != 0) bad++;
        }
        printf("T. fused bit-exact rows=%u: %s (%zu/8 buffers differ)\n",
               rows, ok && bad == 0 ? "OK" : "FAIL", bad);
        all_ok = all_ok && ok && bad == 0;
        free(hq); free(hk); free(hx); free(hb); free(hc); free(hs); free(hw);
        free(hqw); free(hkw);
        free(rq); free(rk); free(rx); free(ro);
        h3_gpu_tensor_free(q); h3_gpu_tensor_free(k);
        h3_gpu_tensor_free(q2); h3_gpu_tensor_free(k2);
        h3_gpu_tensor_free(qnw); h3_gpu_tensor_free(knw);
        h3_gpu_tensor_free(cos_t); h3_gpu_tensor_free(sin_t);
        h3_gpu_tensor_free(x); h3_gpu_tensor_free(x2);
        h3_gpu_tensor_free(branch); h3_gpu_tensor_free(weight);
        h3_gpu_tensor_free(out1); h3_gpu_tensor_free(out2);
    }
    return all_ok;
}

/* ---------------- probe T: timeline ablation ----------------
 * Step 0 of the semantic optimization round: split the ~6 ms/branch of
 * non-GEMM wall time into per-kernel-class contributions WITHOUT GPU
 * timestamp counters. Each variant of run_step_f removes (or replaces
 * with an in-place self-dependent add) exactly one kernel class while
 * keeping the data-dependency chain closed — a removed kernel's
 * consumer is re-wired to its producer, so nothing can overlap. The
 * delta vs "full" attributes wall time to that class (execution +
 * its share of dispatch tax/bubbles). gemm-stub isolates GEMM
 * execution (plus the MPS dispatch tax), all-stub bounds the pure
 * front-end tax + dependency bubbles of the whole 449-dispatch chain.
 * The shipping path (fused qkv GEMV + MPS) is the baseline; CFG dual
 * branch matches the production regime. */

static void probe_timeline(h3_gpu *gpu) {
    const int L = 28;
    const int STEPS = 1024; /* halves must agree; deltas are the payload */
    fctx c;
    memset(&c, 0, sizeof(c));
    if (!fctx_build(gpu, &c, L)) {
        fprintf(stderr, "T. context build failed\n");
        exit(1);
    }
    if (!fused_bitexact_check(gpu)) {
        fprintf(stderr, "T. fused kernels are NOT bit-exact; aborting\n");
        exit(1);
    }
    c.dual = 1;
    struct { const char *name; unsigned flags; } vars[] = {
        {"full          ", 0},
        {"qk fused1     ", AF_HN_ROPE_FUSED},
        {"hn+rope fused ", AF_HN_ROPE_FUSED2},
        {"add+rms fused ", AF_ADD_RMS_FUSED},
        {"fused both    ", AF_HN_ROPE_FUSED | AF_ADD_RMS_FUSED},
        {"prod (f+split)", AF_HN_ROPE_FUSED | AF_ADD_RMS_FUSED |
                          AF_SDPA_SPLIT},
        {"prod slope4   ", AF_HN_ROPE_FUSED | AF_ADD_RMS_FUSED |
                          AF_SDPA_SPLIT | AF_SLOPE4},
        {"prod slope16  ", AF_HN_ROPE_FUSED | AF_ADD_RMS_FUSED |
                          AF_SDPA_SPLIT | AF_SLOPE16},
        {"prod slope64  ", AF_HN_ROPE_FUSED | AF_ADD_RMS_FUSED |
                          AF_SDPA_SPLIT | AF_SLOPE64},
        {"blit->compute ", AF_COMPUTE_COPY},
        {"no hn+rope    ", AF_NO_HNORM_ROPE},
        {"no rms        ", AF_NO_RMS},
        {"no add        ", AF_NO_ADD},
        {"no swiglu     ", AF_NO_SWIGLU},
        {"sdpa->add     ", AF_NO_SDPA},
        {"sdpa->split   ", AF_SDPA_SPLIT},
        {"gemm-stub     ", AF_GEMM_STUB},
        {"all-stub      ", AF_ALL_STUB},
    };
    double full = 0;
    for (size_t vi = 0; vi < sizeof(vars) / sizeof(vars[0]); vi++) {
        for (int i = 0; i < 3; i++)
            if (!run_step_f(gpu, &c, 1, vars[vi].flags)) {
                fprintf(stderr, "T. %s warmup failed: %s\n", vars[vi].name,
                        h3_gpu_error(gpu));
                exit(1);
            }
        h3_gpu_stats s0, s1;
        if (!h3_gpu_get_stats(gpu, &s0)) return;
        double t0 = now();
        for (int i = 0; i < STEPS / 2; i++)
            if (!run_step_f(gpu, &c, 1, vars[vi].flags)) exit(1);
        double half1 = now() - t0;
        t0 = now();
        for (int i = STEPS / 2; i < STEPS; i++)
            if (!run_step_f(gpu, &c, 1, vars[vi].flags)) exit(1);
        double half2 = now() - t0;
        if (!h3_gpu_get_stats(gpu, &s1)) return;
        double per = (half1 + half2) / STEPS * 1e3;
        double gpu_ms = (s1.gpu_seconds - s0.gpu_seconds) / STEPS * 1e3;
        if (vi == 0) full = per;
        printf("T. %s wall %6.3f ms/branch (h1 %6.3f h2 %6.3f)"
               " gpu-busy %6.3f ms  delta %6.3f\n", vars[vi].name, per,
               half1 / (STEPS / 2) * 1e3, half2 / (STEPS / 2) * 1e3,
               gpu_ms, full - per);
    }
    fctx_release(gpu, &c);
}

/* ---------------- probe U: SDPA microbenchmark ----------------
 * Probe T found the flash2 prefix SDPA costs ~157 us per decode call
 * (q_rows=1, kv_rows=513, GQA 16:8) — 4.4 ms/branch, the largest
 * non-GEMM item. Sweep shape and kernel variant to find where the time
 * goes: fixed setup (flat across kv_rows), tile padding (q_rows=1 pays
 * the full 64-row QT), or real attention work. Output rotates over 8
 * buffers so consecutive dispatches cannot coalesce on one target. */

static void probe_sdpa(h3_gpu *gpu) {
    const uint32_t HEADS = 16, KVH = 8, HDIM = 128;
    const float SCALE = 1.0f / sqrtf((float)HDIM);
    /* mini-case FIRST, before any timing loop: is the missing write a
     * basic dispatch problem or timing-loop pollution? */
    {
        uint32_t kv_rows = 1;
        h3_gpu_tensor *q = h3_gpu_tensor_new_bf16(gpu,
                                                  (size_t)HEADS * HDIM);
        h3_gpu_tensor *k = h3_gpu_tensor_new_bf16(
            gpu, (size_t)kv_rows * HEADS * HDIM);
        h3_gpu_tensor *v = h3_gpu_tensor_new_bf16(
            gpu, (size_t)kv_rows * HEADS * HDIM);
        h3_gpu_tensor *o1 = h3_gpu_tensor_new_bf16(gpu,
                                                   (size_t)HEADS * HDIM);
        h3_gpu_tensor *o3 = h3_gpu_tensor_new_bf16(gpu,
                                                   (size_t)HEADS * HDIM);
        uint32_t one = kv_rows;
        h3_gpu_tensor *limit = h3_gpu_tensor_from_u32(gpu, &one, 1);
        if (!q || !k || !v || !o1 || !o3 || !limit) exit(1);
        size_t kc1 = (size_t)kv_rows * HEADS * HDIM;
        uint16_t *fill = malloc(kc1 * 2);
        if (!fill) exit(1);
        for (size_t i = 0; i < kc1; i++) fill[i] = f32_to_bf16(1.0f);
        if (!h3_gpu_tensor_write_bf16(k, fill, kc1)) exit(1);
        for (size_t i = 0; i < kc1; i++) fill[i] = f32_to_bf16(0.5f);
        if (!h3_gpu_tensor_write_bf16(v, fill, kc1)) exit(1);
        for (size_t i = 0; i < (size_t)HEADS * HDIM; i++)
            fill[i] = f32_to_bf16(0.25f);
        if (!h3_gpu_tensor_write_bf16(q, fill, (size_t)HEADS * HDIM)) exit(1);
        for (size_t i = 0; i < (size_t)HEADS * HDIM; i++)
            fill[i] = f32_to_bf16(0.125f); /* sentinel that must vanish */
        if (!h3_gpu_tensor_write_bf16(o1, fill, (size_t)HEADS * HDIM)) exit(1);
        for (size_t i = 0; i < (size_t)HEADS * HDIM; i++)
            fill[i] = f32_to_bf16(0.125f);
        if (!h3_gpu_tensor_write_bf16(o3, fill, (size_t)HEADS * HDIM)) exit(1);
        free(fill);
        if (!(h3_gpu_begin(gpu) &&
              qw_gpu_sdpa_decode_bf16(gpu, o1, q, k, v, kv_rows, HEADS, KVH,
                                      HDIM, SCALE) &&
              h3_gpu_submit(gpu))) exit(1);
        if (!(h3_gpu_begin(gpu) &&
              qw_gpu_flash_sdpa_prefix_bf16(gpu, o3, q, k, v, limit, 1,
                                            kv_rows, HEADS, KVH, HDIM,
                                            SCALE) &&
              h3_gpu_submit(gpu))) exit(1);
        static uint16_t r1[16 * 128], r3[16 * 128];
        if (!h3_gpu_tensor_read_bf16(o1, r1, (size_t)HEADS * HDIM) ||
            !h3_gpu_tensor_read_bf16(o3, r3, (size_t)HEADS * HDIM)) exit(1);
        printf("U. FIRST mini-case: scalar %.4f %.4f | flash2 %.4f %.4f"
               " (sentinel 0.125 must vanish)\n", bf16_host(r1[0]),
               bf16_host(r1[1]), bf16_host(r3[0]), bf16_host(r3[1]));
        h3_gpu_tensor_free(q);
        h3_gpu_tensor_free(k);
        h3_gpu_tensor_free(v);
        h3_gpu_tensor_free(o1);
        h3_gpu_tensor_free(o3);
        h3_gpu_tensor_free(limit);
    }
    const int N = 256;
    uint32_t q_shapes[] = {1, 8, 64};
    /* kv=55 = the AR fixture's first decode step (prefill 54 + 1);
     * 4096/9500 = long-song decode territory (semantic cache caps near
     * prefix + 9000, M14 SDPA-per-KV-length calibration). */
    uint32_t kv_shapes[] = {1, 55, 64, 513, 1024, 2048, 4096, 9500};
    h3_gpu_tensor *out_prefix = NULL, *out_plain = NULL;
    for (size_t qi = 0; qi < sizeof(q_shapes) / sizeof(q_shapes[0]); qi++) {
        uint32_t q_rows = q_shapes[qi];
        h3_gpu_tensor *q = h3_gpu_tensor_new_bf16(
            gpu, (size_t)q_rows * HEADS * HDIM);
        if (!q) exit(1);
        uint16_t *hq = malloc((size_t)q_rows * HEADS * HDIM * 2);
        if (!hq) exit(1);
        fill_bf16(hq, (size_t)q_rows * HEADS * HDIM, 0x51u);
        if (!h3_gpu_tensor_write_bf16(q, hq, (size_t)q_rows * HEADS * HDIM))
            exit(1);
        free(hq);
        for (size_t ki = 0; ki < sizeof(kv_shapes) / sizeof(kv_shapes[0]);
             ki++) {
            uint32_t kv_rows = kv_shapes[ki];
            /* allocate for heads (the plain-flash twin requires the full
             * MHA extent even though prefix only touches KVH) */
            h3_gpu_tensor *k = h3_gpu_tensor_new_bf16(
                gpu, (size_t)kv_rows * HEADS * HDIM);
            h3_gpu_tensor *v = h3_gpu_tensor_new_bf16(
                gpu, (size_t)kv_rows * HEADS * HDIM);
            uint32_t one = kv_rows;
            h3_gpu_tensor *limit = h3_gpu_tensor_from_u32(gpu, &one, 1);
            if (!k || !v || !limit) exit(1);
            size_t kvc = (size_t)kv_rows * HEADS * HDIM;
            uint16_t *hk = malloc(kvc * 2);
            if (!hk) exit(1);
            /* random K and V (different seeds): a constant V makes the
             * cross-kernel comparison blind to accumulation order */
            fill_bf16(hk, kvc, 0x7700u + (uint32_t)kv_rows);
            if (!h3_gpu_tensor_write_bf16(k, hk, kvc)) exit(1);
            fill_bf16(hk, kvc, 0x9900u + (uint32_t)kv_rows);
            if (!h3_gpu_tensor_write_bf16(v, hk, kvc)) exit(1);
            free(hk);
            if (!out_prefix)
                out_prefix = h3_gpu_tensor_new_bf16(
                    gpu, (size_t)64 * HEADS * HDIM);
            if (!out_plain)
                out_plain = h3_gpu_tensor_new_bf16(
                    gpu, (size_t)64 * HEADS * HDIM);
            if (!out_prefix || !out_plain) exit(1);
            /* warmup */
            for (int i = 0; i < 4; i++)
                if (!(h3_gpu_begin(gpu) &&
                      qw_gpu_flash_sdpa_prefix_bf16(
                          gpu, out_prefix, q, k, v, limit, q_rows, kv_rows,
                          HEADS, KVH, HDIM, SCALE) &&
                      h3_gpu_submit(gpu))) exit(1);
            h3_gpu_stats s0, s1;
            if (!h3_gpu_get_stats(gpu, &s0)) return;
            double t0 = now();
            if (!h3_gpu_begin(gpu)) return;
            for (int i = 0; i < N; i++)
                if (!qw_gpu_flash_sdpa_prefix_bf16(
                        gpu, out_prefix, q, k, v, limit, q_rows, kv_rows,
                        HEADS, KVH, HDIM, SCALE)) exit(1);
            if (!h3_gpu_submit(gpu)) return;
            double wall = now() - t0;
            if (!h3_gpu_get_stats(gpu, &s1)) return;
            printf("U. flash2-prefix q=%-3u kv=%-4u gpu %7.2f us wall %7.2f us"
                   " /dispatch\n", q_rows, kv_rows,
                   (s1.gpu_seconds - s0.gpu_seconds) / N * 1e6,
                   wall / N * 1e6);
            /* non-prefix twin for comparison (no limit read/barrier) */
            if (!h3_gpu_get_stats(gpu, &s0)) return;
            t0 = now();
            if (!h3_gpu_begin(gpu)) return;
            for (int i = 0; i < N; i++)
                if (!qw_gpu_flash_sdpa_bf16(gpu, out_plain, q, k, v,
                                            q_rows, kv_rows, HEADS, HDIM,
                                            SCALE)) exit(1);
            if (!h3_gpu_submit(gpu)) return;
            wall = now() - t0;
            if (!h3_gpu_get_stats(gpu, &s1)) return;
            printf("U. flash2-plain  q=%-3u kv=%-4u gpu %7.2f us wall %7.2f us"
                   " /dispatch\n", q_rows, kv_rows,
                   (s1.gpu_seconds - s0.gpu_seconds) / N * 1e6,
                   wall / N * 1e6);
            if (q_rows == 1) {
                h3_gpu_tensor *out_dec = h3_gpu_tensor_new_bf16(
                    gpu, (size_t)HEADS * HDIM);
                h3_gpu_tensor *out_split = h3_gpu_tensor_new_bf16(
                    gpu, (size_t)HEADS * HDIM);
                h3_gpu_tensor *partial = h3_gpu_tensor_new_f32(
                    gpu, (size_t)HEADS * 256 * (HDIM + 2));
                if (!out_dec || !out_split || !partial) exit(1);
                /* timing */
                for (int i = 0; i < 4; i++)
                    if (!(h3_gpu_begin(gpu) &&
                          qw_gpu_sdpa_decode_bf16(
                              gpu, out_dec, q, k, v, kv_rows, HEADS, KVH,
                              HDIM, SCALE) &&
                          qw_gpu_sdpa_decode_split_bf16(
                              gpu, out_split, q, k, v, kv_rows, HEADS, KVH,
                              HDIM, SCALE, partial, 256) &&
                          h3_gpu_submit(gpu))) exit(1);
                if (!h3_gpu_get_stats(gpu, &s0)) return;
                t0 = now();
                if (!h3_gpu_begin(gpu)) return;
                for (int i = 0; i < N; i++)
                    if (!qw_gpu_sdpa_decode_bf16(gpu, out_dec, q, k, v,
                                                 kv_rows, HEADS, KVH, HDIM,
                                                 SCALE)) exit(1);
                if (!h3_gpu_submit(gpu)) return;
                wall = now() - t0;
                if (!h3_gpu_get_stats(gpu, &s1)) return;
                printf("U. decode-scalar q=%-3u kv=%-4u gpu %7.2f us"
                       " wall %7.2f us /dispatch\n", q_rows, kv_rows,
                       (s1.gpu_seconds - s0.gpu_seconds) / N * 1e6,
                       wall / N * 1e6);
                if (!h3_gpu_get_stats(gpu, &s0)) return;
                t0 = now();
                if (!h3_gpu_begin(gpu)) return;
                for (int i = 0; i < N; i++)
                    if (!qw_gpu_sdpa_decode_split_bf16(
                            gpu, out_split, q, k, v, kv_rows, HEADS, KVH,
                            HDIM, SCALE, partial, 256)) exit(1);
                if (!h3_gpu_submit(gpu)) return;
                wall = now() - t0;
                if (!h3_gpu_get_stats(gpu, &s1)) return;
                printf("U. decode-split  q=%-3u kv=%-4u gpu %7.2f us"
                       " wall %7.2f us /dispatch (2 dispatches)\n", q_rows,
                       kv_rows, (s1.gpu_seconds - s0.gpu_seconds) / N * 1e6,
                       wall / N * 1e6);
                /* bit-level A/B: flash2 prefix vs scalar vs split, all in
                 * the GQA layout, plus the host fp32 reference over the
                 * same [row][kv_head][d] data the kernels read. */
                static uint16_t ref_attn[16 * 128], dec_attn[16 * 128],
                    split_attn[16 * 128];
                static float host_ref[16 * 128];
                if (!h3_gpu_tensor_read_bf16(out_prefix, ref_attn,
                                             (size_t)HEADS * HDIM) ||
                    !h3_gpu_tensor_read_bf16(out_dec, dec_attn,
                                             (size_t)HEADS * HDIM) ||
                    !h3_gpu_tensor_read_bf16(out_split, split_attn,
                                             (size_t)HEADS * HDIM)) exit(1);
                {
                    uint16_t *qbits = malloc((size_t)HEADS * HDIM * 2);
                    uint16_t *kbits = malloc((size_t)kv_rows * KVH * HDIM * 2);
                    uint16_t *vbits = malloc((size_t)kv_rows * KVH * HDIM * 2);
                    if (!qbits || !kbits || !vbits) exit(1);
                    if (!h3_gpu_tensor_read_bf16(q, qbits,
                                                 (size_t)HEADS * HDIM) ||
                        !h3_gpu_tensor_read_bf16(k, kbits,
                                                 (size_t)kv_rows * KVH *
                                                     HDIM) ||
                        !h3_gpu_tensor_read_bf16(v, vbits,
                                                 (size_t)kv_rows * KVH *
                                                     HDIM)) exit(1);
                    for (uint32_t h = 0; h < HEADS; h++) {
                        uint32_t kvh = h * KVH / HEADS;
                        float m = -3.402823466e+38f, l = 0.0f;
                        float acc[128];
                        memset(acc, 0, sizeof(acc));
                        for (uint32_t r = 0; r < kv_rows; r++) {
                            float dot = 0.0f;
                            for (uint32_t d = 0; d < HDIM; d++)
                                dot += bf16_host(qbits[h * HDIM + d]) *
                                       bf16_host(kbits
                                                 [((size_t)r * KVH +
                                                   kvh) * HDIM + d]);
                            dot *= SCALE;
                            if (dot > m) m = dot;
                        }
                        for (uint32_t r = 0; r < kv_rows; r++) {
                            float dot = 0.0f;
                            for (uint32_t d = 0; d < HDIM; d++)
                                dot += bf16_host(qbits[h * HDIM + d]) *
                                       bf16_host(kbits
                                                 [((size_t)r * KVH +
                                                   kvh) * HDIM + d]);
                            float p = expf(dot * SCALE - m);
                            l += p;
                            for (uint32_t d = 0; d < HDIM; d++)
                                acc[d] += p * bf16_host(
                                              vbits[((size_t)r * KVH +
                                                     kvh) * HDIM + d]);
                        }
                        for (uint32_t d = 0; d < HDIM; d++)
                            host_ref[h * HDIM + d] = acc[d] / l;
                    }
                    free(qbits);
                    free(kbits);
                    free(vbits);
                }
                int mismatch = 0, max_diff = 0, mismatch2 = 0, max_diff2 = 0;
                int host_bad = 0, host_bad2 = 0, host_badp = 0;
                float host_rel = 0.0f, host_rel2 = 0.0f, host_relp = 0.0f;
                for (size_t i = 0; i < (size_t)HEADS * HDIM; i++) {
                    float want = host_ref[i];
                    float den = fabsf(want) > 1e-3f ? fabsf(want) : 1e-3f;
                    float rels = fabsf(bf16_host(dec_attn[i]) - want) / den;
                    float rel2 = fabsf(bf16_host(split_attn[i]) - want) / den;
                    float relp = fabsf(bf16_host(ref_attn[i]) - want) / den;
                    if (rels > host_rel) host_rel = rels;
                    if (rel2 > host_rel2) host_rel2 = rel2;
                    if (relp > host_relp) host_relp = relp;
                    if (rels > 0.02f) host_bad++;
                    if (rel2 > 0.02f) host_bad2++;
                    if (relp > 0.02f) host_badp++;
                    int diff = (int)ref_attn[i] - (int)dec_attn[i];
                    if (diff < 0) diff = -diff;
                    if (ref_attn[i] != dec_attn[i]) {
                        mismatch++;
                        if (diff > max_diff) max_diff = diff;
                    }
                    diff = (int)ref_attn[i] - (int)split_attn[i];
                    if (diff < 0) diff = -diff;
                    if (ref_attn[i] != split_attn[i]) {
                        mismatch2++;
                        if (diff > max_diff2) max_diff2 = diff;
                    }
                }
                printf("U. A/B q=1 kv=%-4u: split vs flash2 %d/2048"
                       " (maxbit %d), scalar vs flash2 %d/2048"
                       " (maxbit %d); host worst-rel split %.2e scalar"
                       " %.2e flash2 %.2e (>2%%: %d/%d/%d)\n", kv_rows,
                       mismatch2, max_diff2, mismatch, max_diff, host_rel2,
                       host_rel, host_relp, host_bad2, host_bad, host_badp);
                h3_gpu_tensor_free(out_dec);
                h3_gpu_tensor_free(out_split);
                h3_gpu_tensor_free(partial);
                if (kv_rows == 1) {
                    /* deterministic mini-case: K=1, V=0.5, Q=0.25 ->
                     * p=1, l=1, output must be 0.5 for every element.
                     * Dedicated tensors: refilling the SWEEP's q/k/v here
                     * poisoned every later A/B — with constant q the
                     * per-thread-element score bug coincides bit-exactly
                     * with the true score, and the sweep looked clean. */
                    uint16_t one_bf = f32_to_bf16(1.0f);
                    uint16_t half_bf = f32_to_bf16(0.5f);
                    uint16_t quart_bf = f32_to_bf16(0.25f);
                    size_t kc1 = (size_t)kv_rows * HEADS * HDIM;
                    uint16_t *fill = malloc(kc1 * 2);
                    if (!fill) exit(1);
                    h3_gpu_tensor *k2 = h3_gpu_tensor_new_bf16(
                        gpu, (size_t)kv_rows * HEADS * HDIM);
                    h3_gpu_tensor *v2 = h3_gpu_tensor_new_bf16(
                        gpu, (size_t)kv_rows * HEADS * HDIM);
                    h3_gpu_tensor *q2 = h3_gpu_tensor_new_bf16(
                        gpu, (size_t)HEADS * HDIM);
                    if (!k2 || !v2 || !q2) exit(1);
                    for (size_t i = 0; i < kc1; i++) fill[i] = one_bf;
                    if (!h3_gpu_tensor_write_bf16(k2, fill, kc1)) exit(1);
                    for (size_t i = 0; i < kc1; i++) fill[i] = half_bf;
                    if (!h3_gpu_tensor_write_bf16(v2, fill, kc1)) exit(1);
                    for (size_t i = 0; i < (size_t)HEADS * HDIM; i++)
                        fill[i] = quart_bf;
                    if (!h3_gpu_tensor_write_bf16(q2, fill,
                                                  (size_t)HEADS * HDIM))
                        exit(1);
                    free(fill);
                    h3_gpu_tensor *partial2 = h3_gpu_tensor_new_f32(
                        gpu, (size_t)HEADS * 256 * (HDIM + 2));
                    h3_gpu_tensor *o1 = h3_gpu_tensor_new_bf16(
                        gpu, (size_t)HEADS * HDIM);
                    h3_gpu_tensor *o2 = h3_gpu_tensor_new_bf16(
                        gpu, (size_t)HEADS * HDIM);
                    h3_gpu_tensor *o3 = h3_gpu_tensor_new_bf16(
                        gpu, (size_t)HEADS * HDIM);
                    float *pd = malloc((size_t)HEADS * 130 * 4);
                    if (!partial2 || !o1 || !o2 || !o3 || !pd) exit(1);
                    if (!(h3_gpu_begin(gpu) &&
                          h3_gpu_add_bf16(gpu, o1, o1, o1,
                                          (uint32_t)HEADS * HDIM) &&
                          h3_gpu_submit(gpu))) exit(1);
                    static uint16_t pre[16 * 128];
                    if (!h3_gpu_tensor_read_bf16(o1, pre,
                                                 (size_t)HEADS * HDIM)) exit(1);
                    printf("U. mini-case pre-scalar o1[0]=%.4f o1[1]=%.4f"
                           " (add sanity: 0.5+0.5=1.0? garbage?)\n",
                           bf16_host(pre[0]), bf16_host(pre[1]));
                    if (!(h3_gpu_begin(gpu) &&
                          qw_gpu_sdpa_decode_bf16(gpu, o1, q2, k2, v2,
                                                  kv_rows, HEADS, KVH, HDIM,
                                                  SCALE) &&
                          h3_gpu_submit(gpu))) exit(1);
                    if (!(h3_gpu_begin(gpu) &&
                          qw_gpu_sdpa_decode_split_bf16(gpu, o2, q2, k2, v2,
                                                        kv_rows, HEADS, KVH,
                                                        HDIM, SCALE, partial2,
                                                        256) &&
                          h3_gpu_submit(gpu))) exit(1);
                    if (!(h3_gpu_begin(gpu) &&
                          qw_gpu_flash_sdpa_prefix_bf16(
                              gpu, o3, q2, k2, v2, limit, 1, kv_rows, HEADS,
                              KVH, HDIM, SCALE) &&
                          h3_gpu_submit(gpu))) exit(1);
                    static uint16_t r1[16 * 128], r2[16 * 128], r3[16 * 128];
                    if (!h3_gpu_tensor_read_bf16(o1, r1,
                                                 (size_t)HEADS * HDIM) ||
                        !h3_gpu_tensor_read_bf16(o2, r2,
                                                 (size_t)HEADS * HDIM) ||
                        !h3_gpu_tensor_read_bf16(o3, r3,
                                                 (size_t)HEADS * HDIM) ||
                        !h3_gpu_tensor_read_f32(partial2, pd,
                                                (size_t)HEADS * 130)) exit(1);
                    printf("U. mini-case (expect 0.5): scalar %.4f %.4f"
                           " | split %.4f %.4f | flash2 %.4f %.4f"
                           " | split partial h0 m %.3f l %.3f acc0 %.3f"
                           " acc1 %.3f\n", bf16_host(r1[0]),
                           bf16_host(r1[1]), bf16_host(r2[0]),
                           bf16_host(r2[1]), bf16_host(r3[0]),
                           bf16_host(r3[1]), pd[128], pd[129], pd[0],
                           pd[1]);
                    h3_gpu_tensor_free(partial2);
                    h3_gpu_tensor_free(o1);
                    h3_gpu_tensor_free(o2);
                    h3_gpu_tensor_free(o3);
                    h3_gpu_tensor_free(k2);
                    h3_gpu_tensor_free(v2);
                    h3_gpu_tensor_free(q2);
                    free(pd);
                }
            }
            h3_gpu_tensor_free(k);
            h3_gpu_tensor_free(v);
            h3_gpu_tensor_free(limit);
        }
        h3_gpu_tensor_free(q);
    }
}

static void probe_production(h3_gpu *gpu) {
    const int L = 28;
    const int STEPS = 2048;

    fctx c;
    memset(&c, 0, sizeof(c));
    if (!fctx_build(gpu, &c, L)) {
        fprintf(stderr, "F. context build failed\n");
        exit(1);
    }
    const uint32_t H = c.hidden, KV = c.kv, MLP = c.mlp;
    const uint32_t OFFSET = c.offset;
    size_t weight_bytes = 0;
    for (int l = 0; l < L; l++)
        weight_bytes += ((size_t)H * (H + 2 * KV + H + 2 * MLP) +
                         (size_t)MLP * H) * 2;
    printf("F. production decode chain: %d layers, %.2f GB weights,"
           " offset %u\n", L, (double)weight_bytes / 1e9, OFFSET);

    struct {
        const char *name;
        const char *gemv;      /* H3_GEMV value or NULL */
        const char *kernel;    /* H3_GEMV_KERNEL */
        const char *qkv;       /* H3_QKV_GEMV value or NULL */
        const char *qkvkernel; /* H3_QKV_GEMV_KERNEL or NULL */
    } fvariants[] = {
        {"all-MPS          ", NULL, NULL, "0", NULL},
        {"shipping(qkv+MPS)", NULL, NULL, NULL, NULL},
        {"all-GEMV(u8)     ", "1", "qw_gemv_bf16_u8", NULL, NULL},
        /* MLX-style 2D block everywhere (qkv as three 2D dispatches). */
        {"all-2D(m32u8)    ", "1", "qw_gemv2d_m32u8", "0", NULL},
        {"all-2D(m64u8)    ", "1", "qw_gemv2d_m64u8", "0", NULL},
        /* 2D block + fused 2D qkv (one dispatch per attention prologue). */
        {"qkv2D+2D(m32u8)  ", "1", "qw_gemv2d_m32u8", NULL,
         "qw_gemv2d_qkv_m32u8"},
        {"qkv2D+2D(m64u8)  ", "1", "qw_gemv2d_m64u8", NULL,
         "qw_gemv2d_qkv_m64u8"},
    };
    for (size_t vi = 0; vi < sizeof(fvariants) / sizeof(fvariants[0]); vi++) {
        for (int dual = 1; dual >= 0; dual--) {
            c.dual = dual;
            if (fvariants[vi].gemv) {
                setenv("H3_GEMV", fvariants[vi].gemv, 1);
                if (fvariants[vi].kernel)
                    setenv("H3_GEMV_KERNEL", fvariants[vi].kernel, 1);
                else
                    unsetenv("H3_GEMV_KERNEL");
            } else {
                unsetenv("H3_GEMV");
                unsetenv("H3_GEMV_KERNEL");
            }
            if (fvariants[vi].qkvkernel)
                setenv("H3_QKV_GEMV_KERNEL", fvariants[vi].qkvkernel, 1);
            else
                unsetenv("H3_QKV_GEMV_KERNEL");
            if (fvariants[vi].qkv)
                setenv("H3_QKV_GEMV", fvariants[vi].qkv, 1);
            else
                unsetenv("H3_QKV_GEMV");
            for (int i = 0; i < 3; i++)
                if (!run_step(gpu, &c, 1)) {
                    fprintf(stderr, "F. %s warmup failed: %s\n",
                            fvariants[vi].name, h3_gpu_error(gpu));
                    exit(1);
                }
            h3_gpu_stats s0, s1;
            if (!h3_gpu_get_stats(gpu, &s0)) return;
            double t0 = now();
            for (int i = 0; i < STEPS / 2; i++)
                if (!run_step(gpu, &c, 1)) exit(1);
            double half1 = now() - t0;
            t0 = now();
            for (int i = STEPS / 2; i < STEPS; i++)
                if (!run_step(gpu, &c, 1)) exit(1);
            double half2 = now() - t0;
            if (!h3_gpu_get_stats(gpu, &s1)) return;
            double wall = half1 + half2;
            double gpu_ms = (s1.gpu_seconds - s0.gpu_seconds) / STEPS * 1e3;
            double per = wall / STEPS * 1e3;
            printf("F. %s %-6s wall %6.3f ms/step (h1 %6.3f h2 %6.3f)"
                   " gpu-busy %6.3f ms\n", fvariants[vi].name,
                   dual ? "[cfg2]" : "[1x]  ", per,
                   half1 / (STEPS / 2) * 1e3, half2 / (STEPS / 2) * 1e3,
                   gpu_ms);
        }
    }
    unsetenv("H3_GEMV");
    unsetenv("H3_GEMV_KERNEL");
    unsetenv("H3_QKV_GEMV");
    unsetenv("H3_QKV_GEMV_KERNEL");

    /* bit check: all-GEMV vs all-MPS logits must agree exactly */
    if (!run_step(gpu, &c, 1)) exit(1);
    uint16_t ref[8192];
    memcpy(ref, f_bits, sizeof(ref));
    setenv("H3_GEMV", "1", 1);
    setenv("H3_GEMV_KERNEL", "qw_gemv_bf16_u8", 1);
    if (!run_step(gpu, &c, 1)) exit(1);
    int mismatch = 0;
    for (size_t i = 0; i < sizeof(ref) / sizeof(ref[0]); i++)
        if (ref[i] != f_bits[i]) mismatch++;
    printf("F. bit check all-GEMV vs all-MPS logits: %d/8192 mismatch\n",
           mismatch);
    unsetenv("H3_GEMV");
    unsetenv("H3_GEMV_KERNEL");

    fctx_release(gpu, &c);
}

/* ---------------- probe S: 2D-block GEMV per-shape sweep ----------------
 * Coarse filter for the MLX-style 2D-block kernel family (qw_gemv2d_*):
 * same 28-deep serial-chain harness as probe D (fake-dependency, so the
 * absolute numbers are optimistic — the arbiter is probe F), sweeping
 * blockM (output rows per threadgroup = BM*TM) and TN at fixed
 * SM=1/SN=32. Purpose: drop hopeless configs before the probe F run. */

static void probe_2d_sweep(h3_gpu *gpu) {
    struct {
        uint32_t in_dim, out_dim;
        const char *label;
    } shapes[] = {
        {2048, 1024, "k/v 2048->1024"},
        {2048, 2048, "q/o 2048->2048"},
        {2048, 12288, "fc1 2048->12288"},
        {6144, 2048, "down 6144->2048"},
        {2048, 32769, "lm semantic 2048->32769"},
    };
    const char *g2d_variants[] = {
        "qw_gemv2d_m4", "qw_gemv2d_m8",    "qw_gemv2d_m16",
        "qw_gemv2d_m32", "qw_gemv2d_m64",  "qw_gemv2d_m64x8",
        "qw_gemv2d_m32u8", "qw_gemv2d_m64u8", "qw_gemv2d_m64x8u8",
    };
    const int DEPTH = 28, ITERS = 10;
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        uint32_t in_dim = shapes[s].in_dim, out_dim = shapes[s].out_dim;
        h3_gpu_tensor *x = h3_gpu_tensor_new_bf16(gpu, in_dim);
        h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, out_dim);
        h3_gpu_tensor *weights[4];
        for (int w = 0; w < 4; w++)
            weights[w] = make_weight(gpu, in_dim, out_dim,
                                     0x9000u + (uint32_t)(s * 4 + w));
        if (!x || !out || !weights[0] || !weights[1] || !weights[2] ||
            !weights[3]) exit(1);
        size_t bytes = (size_t)out_dim * in_dim * 2;
        printf("S. %s (%.1f MB):\n", shapes[s].label,
               (double)bytes / 1e6);
        struct path_entry {
            const char *name;
            int gemv;
            const char *kernel;
        } paths[11];
        paths[0] = (struct path_entry){"MPS", 0, NULL};
        paths[1] = (struct path_entry){"gemv_u8", 1, "qw_gemv_bf16_u8"};
        for (size_t v = 0; v < sizeof(g2d_variants) / sizeof(g2d_variants[0]); v++)
            paths[2 + v] = (struct path_entry){g2d_variants[v], 1, g2d_variants[v]};
        for (size_t p = 0;
             p < sizeof(paths) / sizeof(paths[0]); p++) {
            if (paths[p].gemv) {
                setenv("H3_GEMV", "1", 1);
                setenv("H3_GEMV_KERNEL", paths[p].kernel, 1);
            } else {
                unsetenv("H3_GEMV");
            }
            for (int i = 0; i < 2; i++) {
                if (!(h3_gpu_begin(gpu) &&
                      h3_gpu_linear_bf16(gpu, out, x, weights[0], NULL, 1,
                                         in_dim, out_dim) &&
                      h3_gpu_submit(gpu))) exit(1);
            }
            h3_gpu_stats st0, st1;
            if (!h3_gpu_get_stats(gpu, &st0)) return;
            double t0 = now();
            for (int it = 0; it < ITERS; it++) {
                if (!h3_gpu_begin(gpu)) return;
                for (int d = 0; d < DEPTH; d++)
                    if (!h3_gpu_linear_bf16(gpu, out, x, weights[d % 4],
                                            NULL, 1, in_dim, out_dim))
                        exit(1);
                if (!h3_gpu_submit(gpu)) return;
            }
            double wall = now() - t0;
            if (!h3_gpu_get_stats(gpu, &st1)) return;
            double gpu_us = (st1.gpu_seconds - st0.gpu_seconds) /
                            (ITERS * DEPTH) * 1e6;
            double wall_us = wall / (ITERS * DEPTH) * 1e6;
            printf("   %-18s gpu %7.2f us wall %7.2f us /dispatch"
                   " (%5.1f / %5.1f GB/s)\n", paths[p].name, gpu_us,
                   wall_us, (double)bytes / (gpu_us * 1e-6) / 1e9,
                   (double)bytes / (wall_us * 1e-6) / 1e9);
        }
        unsetenv("H3_GEMV");
        for (int w = 0; w < 4; w++) h3_gpu_tensor_free(weights[w]);
        h3_gpu_tensor_free(x);
        h3_gpu_tensor_free(out);
    }
}

int main(int argc, char **argv) {
    char error[512] = "";
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "gpu init failed: %s\n", error);
        return 1;
    }
    if (argc > 1 && argv[1][0] == 'F') {
        probe_production(gpu);
        printf("bench-dispatch: done (F only)\n");
        return 0;
    }
    if (argc > 1 && argv[1][0] == 'T') {
        probe_timeline(gpu);
        printf("bench-dispatch: done (T only)\n");
        return 0;
    }
    if (argc > 1 && argv[1][0] == 'U') {
        probe_sdpa(gpu);
        printf("bench-dispatch: done (U only)\n");
        return 0;
    }
    if (argc > 1 && argv[1][0] == 'S') {
        probe_2d_sweep(gpu);
        printf("bench-dispatch: done (S only)\n");
        return 0;
    }
    if (argc > 1 && argv[1][0] == 'G') {
        probe_shape_true_chain(gpu);
        printf("bench-dispatch: done (G only)\n");
        return 0;
    }
    if (argc > 1 && argv[1][0] == 'H') {
        probe_batch_rows(gpu);
        printf("bench-dispatch: done (H only)\n");
        return 0;
    }
    probe_empty(gpu);
    probe_chain(gpu);
    probe_qkv_merge(gpu);
    probe_shape_chain(gpu);
    probe_interleave(gpu);
    probe_production(gpu);
    printf("bench-dispatch: done\n");
    return 0;
}
