# yue-metal — Native Metal Inference Engine for YuE2-3B

English | [简体中文](README-ZH.md)

A native inference engine for the YuE2-3B song-generation model (AR semantic
tokens + NAR acoustic flow matching + Oobleck VAE), written in plain C11 +
Objective-C/Metal for Apple Silicon (unified memory, M-series). Numerical
conventions reproduce the reference stack lyra (MLX port, including its vendored
including its vendored official yue2 protocol layer) point by point; AR greedy
output is bit-identical to lyra (see "Numerical Conventions").

Same-machine, same-request comparison (2026-11-02; 442-char lyrics / cot=off /
cfgScale=1.2 / seed=831001, bf16 vs bf16, both measured; different sampling
trajectories are by design — initial noise comes from the engine's own stream):

| Engine | semantic | flow | save | e2e | RTF |
|---|---|---|---|---|---|
| lyra (MLX bf16) | 75.5s / 4995 tok (15.1 ms/token) | 106.5s | ≈2s | ≈184s | 0.92 |
| yue-metal (M15, CFG batch fusion) | 55.6s / 4730 tok (11.7 ms/token) | 84.8s | 2.6s | **143.5s** | **0.758** |

**~1.2× faster overall** (e2e −22%, RTF 0.92 → 0.758): semantic **−22%**,
flow **−20%**. M15's batch fusion merges the two CFG branches' GEMM weight
streams into one pass (see the performance section); on the flow side yue's
NAR/VAE has been ahead since M7. Same request, two runs of yue: RTF
0.725/0.763 (sampling trajectories differ).

Performance history (semantic ms/token is not directly comparable across
request profiles — treat as trend only; precise comparisons live in the
per-milestone same-request A/Bs; same for RTF):

| Milestone | semantic (different profiles, trend only) | e2e RTF | Main techniques |
|---|---|---|---|
| M6 first full pipeline | — | 2.2 | correctness first |
| M7 (four rounds) | 74 → 28.6 | 1.15 | per-layer submit merging, MPS linear, whole-step command chaining, snake vForce, fused qkv GEMV |
| M8 | 28.6 | 0.97 | VAE decode on GPU (save 6.2s → 0.6s) |
| M10 | 19.9 | 0.80 | decode split-K SDPA (1.91×, removes flash2's 64-row padding tax) |
| M11 | ≈18.7 | — | attention prologue fusion (hn+rope in one dispatch, −7%) |
| M14 | 17.2-20.2 | ≈0.95 | decode GEMV switched to 2D-block kernel (−4.3%) + host-tail attribution fix |
| **M15** | **10.3-12.6** | **0.758** | **CFG batch fusion (both branches in one stream, −40%)** |

Cumulative: semantic **≈6.3×**, e2e RTF **≈2.9×**. The scheduling track (M12),
the GEMM kernel track (M13), and the encoding/host track (M14) are all closed —
verdicts and data for each step are in the performance section and in the
header comment of `tests/bench_dispatch.c`. M18 (2026-11-05) is a correctness
fix for the long-KV split-K SDPA path (full cacheline writes + 16-split cap);
it leaves the table above unchanged (only the kv>4096 attention range costs
10-20% more) — see the performance section.

Historical: M10 (321-char request profile) semantic 19.9 ms/token, RTF 0.80
(lyra same profile ≈17.1/0.73); M7 four rounds 34.2s (RTF 1.15) → M8 VAE GPU
28.8s (RTF 0.97).

## Model & Dependencies

- Weights: a directory containing the official m-a-p single-file
  `model.safetensors` (the `YuE2-3B` checkpoint), plus the `YuE2-Vae`
  directory (when `--vae` is omitted it is derived as `<model>-vae`; the
  official naming requires an explicit `--vae`).
- The lyra conversion layout is also supported directly: point `-d` at a
  directory containing `ar-bf16`/`nar-bf16.safetensors` (byte-identical to the
  official single file, verified by sha256 across all 628 tensors; the engine
  falls back to the single file automatically when they are absent). The vae
  keys/shapes/dtypes are fully isomorphic between both layouts. The official
  directory ships a `yue2_generation_config.json` that matches the built-in
  defaults field by field.
- Reference stack: the lyra MLX package (mlx-Yue) + its `.venv`; only used to
  generate test fixtures.
- Runtime dependencies: Metal, MetalPerformanceShaders(+Graph), Accelerate,
  ffmpeg (see the ffmpeg section below).

### ffmpeg (the only non-framework runtime dependency)

The engine's save stage first writes an intermediate 24-bit WAV
(`yue_write_wav24`), then transcodes it to FLAC in-process via
`ffmpeg -y -i tmp.wav -c:a flac out.flac` (spawned with `posix_spawnp`, no
shell involved). This is the **only** use of ffmpeg in this engine — no
ffprobe, no image/video decoding (the `h3_ffmpeg.c` helpers for those are
vendored from the qwen side and unused here). FLAC is lossless; output is
24-bit / 48 kHz / stereo.

Install (any one):

```sh
brew install ffmpeg                 # Homebrew (recommended)
sudo port install ffmpeg            # MacPorts
# or grab a static build for macOS: https://evermeet.cx/ffmpeg/
#   and point H3_FFMPEG at the binary (see Environment Variables)
```

Verify:

```sh
ffmpeg -version    # any reasonably recent 4.x/5.x/6.x/7.x build works
```

If ffmpeg is missing, generation itself completes but the save stage fails
with `cannot start ffmpeg: ...` (or `ffmpeg flac encode failed`); the
`H3_FFMPEG=/path/to/ffmpeg` environment variable points the engine at a
binary that is not on PATH.

## Build

```sh
make yue            # builds ./yue (CLI, with serve + repl modes)
make m0-tests tokenizer-tests gemv-tests ar-tests sampling-tests nar-tests vae-tests
```

## Usage

```sh
# JSONL service mode
printf '%s\n' '{"id":"t","style":"upbeat pop","lyrics":"[verse]\nla la la\n[chorus]\nna na na","cot":"off","cfgScale":1.2,"seed":831001,"output":"/tmp/t.flac"}' \
  | ./yue serve -d /path/to/YuE2-3B --vae /path/to/YuE2-Vae

# Persistent REPL (linenoise)
./yue repl -d MODEL_DIR [--vae VAE_DIR]
```

REPL commands:

```
/style TEXT        style tags (required)
/lyrics TEXT       lyrics; \n escapes a newline (required)
/lyrics-file PATH  read lyrics from a UTF-8 file
/abc TEXT          ABC score; \n makes a newline; bare /abc clears it
/abc-file PATH     read an ABC score verbatim from a UTF-8 file
                   external scores need cot=full/melody (not off)
/cot full|melody|off  decoding mode (default full)
/cfg X             explicit CFG; 0 = auto
/seed N            base seed (default 831001, lyra's default; auto-increments after each /gen)
/out PREFIX        output path prefix (default out, produces PREFIX-<n>.flac)
/show /gen /help /quit
```

## JSONL Protocol (serve)

- Request: `{"id","style","lyrics","cot","cfgScale","seed","output","abc"}` (512KB
  line limit; validation mirrors worker.py: cfgScale only takes effect when it
  is a NUMBER > 0, range [0,20]; seed ∈ [0,2^63); `output` must end in `.flac`).
- `"abc"` is an optional STRING carrying a user-supplied ABC score (lyra's
  `SongRequest.abc`). Its text ids become the ABC prefix for both the positive
  and the symbolic-CFG negative branch, so the planning AR pass is skipped
  (the `abc` progress stage closes immediately). Requires `cot=full` or
  `cot=melody`; `cot=off` has no ABC section and rejects it. `abc: null` or a
  missing key means "no score" (`SongRequest.abc=None`); an empty or
  whitespace-only string is rejected. The text is encoded **verbatim** (never
  trimmed) so its BPE ids match `tokenizer.encode(request.abc)`. Limits: 20000
  bytes / 4096 tokens, all ids < `YUE_EOD`. Omitting it keeps the original
  planning path bit-identical.
- Events: `state loading/ready(loadSeconds)` → `progress(id/stage/percent/
  elapsed[, heartbeat])` → `done(path/seconds/tokens/sampleRate)` or
  `error(id/message)`.
- Progress stages: `abc 0.02-0.38` (idles when cot=off) → `semantic 0.40-0.78`
  → `flow 0.80-0.94` (heartbeat-thread extrapolation) → `save 0.96-1.00`;
  the 4s heartbeat does not refresh the host idle timer.
- Cancellation = the host kills the process group; stdin EOF exits.
- Single-GPU-instance lock: `flock /tmp/yue-metal-engine-<uid>.lock`. yue2-web
  now also runs the native engine (it launches this binary's serve mode), so
  both share the same lock and are mutually exclusive by construction.
- NAR / VAE are lazily created: `ready` loads only AR + tokenizer + generation
  config (~0.7s).

## Performance (post-M11)

- **M15: CFG batch fusion — both branches merged into one stream, semantic
  −40%, overtakes lyra** (2026-11-02). Probe H
  (`./tests/bench_dispatch H`, includes a 0-bit-diff gate for rows=2 vs
  2×rows=1) proved MPS linear rows=2 costs nearly the same as rows=1
  (fc1/down/lm batch-eff 0.94-1.00×, pure bandwidth-bound) — reading the
  weight stream twice per CFG pair (~12.6ms/token of GEMM) was pure waste.
  Implementation: `yue_ar_forward_open2` ([2,hidden] single-stream batch
  forward; each branch's rows attend independently and write their own cache;
  rope tables carry each row's own phase — the negative prefix is longer, so
  the two caches have different offsets) + `qw_gpu_sdpa_decode_split_bf16_off`
  (query/output element-offset binding variant, same kernel) +
  `decode_step2` ([2,width] one-shot logits projection + readback). **Key
  semantics: step 0 samples from the prefill logits (pos.host/neg.host);
  host2 is written by each batched step for the *next* token** — the first
  version pointed gen_branch.host into host2, so step 0 read uninitialized
  memory (margin 1e36) and cleanup double-freed; fixed (branch.host keeps its
  own malloc, guided dispatches on batch_on && step>0). serve (442-char
  lyrics/cot=off/cfg1.2): semantic **17.2-20.2 → 10.3-12.6 ms/token**
  (−40% vs the M14 default; breakdown enc 3.1 [MPS encode 84 calls] +
  wait 7.1→9.5 + host 0.12, matching prediction), e2e RTF **0.725-0.763
  vs lyra 0.92 — ahead**; semantic ~11.7 vs lyra ≈15.1 (−22%, same-request
  profile). **Numerical gates**: melody_abc (abc phase, no CFG, single branch)
  stays bit-identical to lyra; the semantic trajectory is globally reordered
  (rows=2 GEMM has a different reduction order, distribution-equivalent),
  flips within gate (off step12 margin 0.0625, melody step8); all seven
  regression suites green (single-stream path unaffected by fixtures).
  `YUE_CFG_BATCH=0` falls back to serial dual-branch; debug:
  `YUE_BATCH_DBG=1` scans logits for non-finite values each step and dumps
  /tmp/batch_nan_s*.bin on hit. Post-fix, 27 consecutive serve generations
  (multi-seed / multi-request same process) with zero failures.
- **M14: host-tail attribution fix + decode GEMV switched to the 2D-block
  kernel by default** (2026-11-02). Added `YUE_HOST_TRACE=1` (prints the
  per-token semantic breakdown wall/enc/wait/read/guided/unpack/dist/sample
  every 512 steps). **Attribution fix**: each semantic token's ~19ms =
  CPU encoding 4.4 (of which MPSGraph linear encode ~3.1 — an MPSGraph
  inherent tax: command reuse / TensorData cache / graph cache all already on
  by default) + GPU-chain wait ~14.5 (grows with KV) + **host compute only
  0.11** — the earlier "2.7ms host tail" was the wall−busy delta method
  misattributing flow's GPU time to semantic; retracted. **All three encode
  -side paths scouted and abandoned**: per-column GEMV (enc −3.1 but wait +6,
  dead), TensorOps GEMV `qw_tops_gemv_bf16` (matmul2d direct dispatch, 0-bit
  diff vs MPS, but M=1 tiles execute at −20~25% bandwidth; TM=4 fails to
  compile on this MPP and blows up the whole metallib, TN sweep found no
  solution, dead; kernel and probe G TOPS path kept), MPS encode itself (no
  lever). **Landed: decode GEMV defaults to `qw_gemv2d_m64u8`** (main.c
  `default_gemv_env`, setenv overwrite=0, explicit env still wins; the shared
  vendored h3 defaults used by qwen are unaffected) — the 2D kernel executes
  at the same bandwidth as MPS but encodes cheaper (1.3 vs 4.4ms/token); the
  serve same-window comparison (same window = same KV length; KV depends only
  on step) gives **wall −0.84ms/token (−4.3%)**; all seven regressions green,
  melody_abc stays bit-identical to lyra, semantic flips within gate (off
  step24 margin 0, melody step43 margin 0). M9's "2D no production e2e gain"
  is now explained: flow noise masked a ~0.45s semantic win. semantic
  ~18.6ms/token (long-request profile; superseded by batch in M15, see the
  M15 entry).
- **M13: GEMM track final review — full-fan retest + per-shape bandwidth
  calibration; track closed** (2026-11-02). Per M12 discipline: fans at full
  blast, probe and production at the same thermal state. ① Probe F rerun
  matches M9 exactly (all-MPS [1x] 12.43 vs 12.19; shipping 12.61; all-2D
  12.16; all-GEMV(u8) 13.35) — **the probe-side "~246GB/s floor" is not a
  thermal-throttling artifact**; probe T's prod variant at 17.12 vs M12's
  recorded 18.44 was the thermal noise (F runs faster when thermally soaked
  15 minutes in). ② **M12 attribution correction**: 18.44 was the CFG
  dual-branch total — "GEMM ~12ms @246GB/s" treated both branches as one; a
  single branch's GEMM is really ~6ms @ ~470GB/s effective. ③ Added
  `./tests/bench_dispatch G` (true-dependency single-shape chain + bridge
  self-loop elimination): **large shapes fc1 (100-107µs @471-503GB/s), down
  (46-52µs @485-546), lm (~259µs @514-519) already run at 90-100% of the
  streaming limit (~520GB/s)** (75% of per-layer weight traffic); the small
  shapes' (q/k/v/o/qkv) probe-side 612-2375GB/s is an L2-reuse illusion (4
  weights rotating, 16-34MB resident in L2; the production chain flushes L2
  every layer with fc1/down — this is the second half of M9's "probe wins
  everything, production flat" mystery). ④ Synthesis closes the loop:
  production per-branch GEMM 5.8-6.1ms vs an all-shapes 500GB/s theoretical
  floor of ~5.9ms — **GEMM is at the floor**; serve full-fan gpu busy
  97.9s/4850tok = 8.6ms/branch-token, decomposition GEMM 6 + split SDPA 1.6 +
  small kernels 0.6 + dispatch 0.3 fully consistent. ⑤ **The semantic gap vs
  lyra is not GEMM efficiency** (lyra's implied ~394GB/s on the same weights
  would need ~7.2ms/branch, slower than us); the gap is split SDPA's
  KV-growth cost plus the per-token host tail (logits readback + sampling).
  Remaining GEMM headroom ≤0.3ms/branch (~3% of semantic) and would require
  winning the small shapes under real L2 pressure — the same race M9 lost
  twice in production. **Verdict: GEMM track closed, no fourth attempt.**
- **M11: attention prologue fusion landed as default** (2026-11-01).
  `qw_head_rms_rope_bf16` covers q+k's per-head RMSNorm + rotate_half RoPE in
  a single dispatch (4 dispatches per layer → 1; one head_dim-thread
  threadgroup per (row,head); the sum of squares keeps the reference's
  single-thread ascending fma chain → bit-identical to the four-kernel path,
  bit-level A/B rows=1/4 dual shapes 0/8 buffer diffs; norm/rope phases fully
  threadgroup-parallel). **Probe T [cfg2]: full 24.15 → fused 22.76
  ms/branch (+1.39, eating 80% of the 1.74 no-hn+rope ablation ceiling)**.
  The first version ("per-head single-threaded serial norm+rope") was 0.45ms
  slower — the parallel phases are the actual win. serve A/B (same request,
  two runs, frame-normalized): semantic 20.30 → **18.9 ms/frame** (−6.9%,
  reruns 18.96/18.86; GPU busy 21.08 → 19.74 ms/frame — the win is purely
  GPU-side; per-frame submissions unchanged). `YUE_HN_ROPE_SEPARATE=1` falls
  back to the four-kernel path. **Vertical rms fusion (add+rms in one) was
  judged structurally zero-gain by probe and did not ship**: the add dispatch
  itself costs ~1.1µs, and folding the residual add into the 256-thread rms
  critical path costs exactly that (probe −0.05 ms/branch, within noise);
  the rms reduction order is bit-locked and cannot use a faster reduction
  tree — that item is bound to the melody_abc bit-exactness gate and cannot
  be bypassed.
- **M10: decode-step split-K SDPA landed as default** (2026-11-01). The
  flash2 TensorOps tile kernel pays a full 64-row QT padding at q_rows=1
  (probe U: kv=513 134µs, kv=2048 470µs); `qw_sdpa_decode_split_bf16` +
  `qw_sdpa_decode_merge_bf16` scalar pairs read each row's K/V once (28.5µs
  @513, 50µs @2048, q-row threadgroup staging). The first version hoisted a
  single thread's q element out of the row loop, turning the score into
  q[tid]·Σk instead of the full dot product — bit-identical to truth only
  when q is constant, and probe U's kv=1 mini-case happened to overwrite the
  shared q tensor with a constant 0.25, disguising the bug as "kv=1/64
  three-way bit-identical"; the production ar gate 1.25 relL2 / melody flip
  came from this. After the fix, the ar gate is 0.027; probe T [cfg2]: full
  24.03 → sdpa→split 18.45 ms/branch (sdpa share 8.85→5.58). serve A/B (same
  request, two runs each, measured warm): semantic 38.1 → **19.9 ms/token**
  (1.91×), e2e 99.8 → 56.1s, RTF 1.24 → 0.80; per-token GPU busy 48.5 →
  31.2ms, per-token submissions 56.1/56.2 unchanged (command-chain rhythm
  unchanged; purely a GPU-side win). `YUE_SDPA_FLASH2=1` falls back to flash2.
- **M18: split-K SDPA correctness fix + vectorized-rewrite verdict**
  (2026-11-05). The vectorization attempt (uint4 K loads + TILE templates,
  `qw_sdpa_dec_v_impl`, probe U v128/256/512) showed **no meaningful
  headroom**: v128 is bit-identical to the scalar kernel and equally fast
  (load-instruction count is not the bottleneck), v256/512 win only 2-3% at
  medium kv and lose 5-20% at large kv — the projected 263→500GB/s never
  materialized. The kernel is kept as probe infrastructure and is not used in
  production. **The real result came from probe V
  (`./tests/bench_dispatch V`), which exposed a latent correctness bug**:
  `ceil(kv/256)` lets the split count exceed ~400 groups at kv>6144 rows, and
  the partial's single 4-byte store (m/l) can be probabilistically lost by the
  memory system — the readback is **the previous dispatch's value** (probe V
  R11: with K=0, among mathematically constant quantities, l occasionally
  shows a historical value from the V=0 experiment; atomic stores,
  whole-group same-value writes, and writing a full cacheline alone did not
  fix it, while the 128-thread contiguous acc write never failed). This
  corrupted decode attention in the second half of long songs (the semantic
  cache tops out near prefix+9000), latent since M10 because fixtures only
  covered kv≤55 plus a few steps. **Two-layer fix**: ① the slot layout stays
  `[acc 128][m][l]` but the stride goes 130→**160 floats** (640 B,
  cacheline-aligned) and the m/l cacheline is written full (tid<32: tid0/1
  write m/l, tid2-31 write zero padding) — the lonely 4-byte store becomes
  32 same-line stores; the `[m][l][acc]` layout was **not adopted** (an 8 B
  acc start offset would straddle 5 lines). ② The splits cap went 384→**256
  groups** (16 splits, hard cap in the wrapper; the loss rate rises steeply
  with group count, and ≤256 groups showed zero losses across all probe V
  runs). Cost accounting (probe U control: cap38 + new layout 146µs@9500 is
  in the same band as before → the layout is free; the earlier 46.3µs@2048
  was a cold-boost outlier, warm steady state 49.5-50.5 in band): the only
  confirmed cost is cap16's large-kv range — kv=9500 about 146→160-180µs
  (+10-20%, ≈0.9 ms/token over 28 layers × 2 branches, only for kv>4096;
  kv≤4096 is unaffected). The probe V matrix (R1-R12) preserves the
  diagnosis; a residual observation (probe-only CPU `write_bf16` / GPU
  dispatch interleaving — production KV caches are written entirely in the
  GPU domain via blit/kernel, never by the CPU) was not pursued. End-to-end
  serve: 4373 tok and **7052 tok (kv~7100, fixed path)** long songs done ✓,
  FLAC 24bit/48k/stereo ✓; all seven regression suites green (melody_abc
  gate intact).
- semantic dual-branch 28.6ms/token (14.3ms/branch-token, M8 profile, old
  request); [gpu] encode 3.1s / wait 24.4s / busy 27.2s / submissions 43K;
  memory live ~7.7GB, footprint ~9.6GB.
- **M8: VAE decode on GPU** (save 6.2s → 0.6s, ~10×; e2e 34.2 → 28.8s, RTF
  1.15 → 0.97). The Oobleck decoder moved to TensorOps matmul2d: Conv1d =
  zero-ring + per-tap pure-pointer-offset GEMM (dilation acts between taps,
  so a fixed tap's input window is a dense view — no gather needed); ConvT =
  per-tap GEMM + strided `+=` write-back; weights repacked on GPU at load:
  `[cout][k][cin] → [k][cout_pad][cin]`. The tile structure (core 256/halo
  16) and three-buffer rotation are unchanged; `YUE_VAE_CPU=1` falls back to
  the Accelerate reference path. GPU fixture comparison: full40 worst 5.4e-06,
  small 3.7e-4 (8-frame short-sequence intrinsic band), audio gate 2e-4, all
  pass.
- M7 optimization history (same request): semantic 74 → 32.6 → 29.0 → 28.6
  ms/token (cumulative 2.6×). Main techniques: ① 4-5 submits per layer merged
  into 1 (continue pipeline); ② MPS linear threshold lowered to enable MPS
  GEMV; ③ 8 submit-waits per decode token → 1 (whole-step command chain: both
  branch forwards + both logits projections encoded into one chain, via the
  `yue_ar_forward_open`/`yue_ar_project_open` chained APIs); ④ snake vForce;
  ⑤ fused q/k/v GEMV enabled by default (3 dispatches → 1, 4096 columns in a
  single dispatch, bit-identical).
- **GEMV track: finally rejected (M8 mystery closed; do not retry)**. The old
  "probe wins 3-4× / production loses 16%" contradiction = three overlapping
  illusions: ① probe B/D chains were fake dependencies (same input every
  dispatch, adjacent kernels overlapped), bandwidths all inflated; ② without
  per-layer-tail `continue`, MPSGraph encode runs fully serial and numbers
  are bogus in both directions; ③ production GEMV A/Bs never set
  `H3_GEMV_KERNEL`, so they all ate the default slow variant (unroll-2,
  ~230GB/s). Corrected (probe F in `tests/bench_dispatch.c`: true dependency
  chain + layer-tail continue + CFG dual branches + 2048-step thermal
  soak), probe and production agree in direction: under a serial
  true-dependency chain, MPS 12.2 vs GEMV-u8 13.3 ms/branch; serve A/B 14.3
  vs 17.4 ms/branch-token — **GEMV at full open still loses to MPS**.
  Ruled out: mmap zero-copy (irrelevant), CFG dual-branch penalty (exactly
  2×, none), logits readback drain (irrelevant). The per-column GEMV serial
  chain ceiling is ~222GB/s vs MPS ~246GB/s — a theoretical ceiling of only
  ~1.1×; lyra's ~394GB/s proves an MLX-style 2D-block + simdgroup-reduction
  kernel has headroom, but that is future kernel engineering (pass probe F
  first, then production A/B).
- **The MLX-style 2D-block GEMV rewrite is complete and rejected twice
  (2026-11-01; GEMV track finally closed)**. `qw_gemv2d_*` (SM=1/SN=32: a
  full simd walks one row's K slider, each thread owns TM independent fma
  chains, TN=8 uint4 payloads, BN=1 pure simd tree reduction with zero
  barriers) + fused `qw_gemv2d_qkv_*` (single dispatch for q/k/v, 4096-row
  unified grid, bit-identical to three chained 2D dispatches). test_gemv
  A-E all green (e_k mapping ×16 variants, row/tail mappings covering K-tail
  2056 and row-tail 300, host fp32 tolerance comparison, fused 2D qkv
  bit-identical). Probe S: the small-matrix disaster is indeed fixed (k/v
  63→471GB/s, q/o 261→735GB/s, fake-dependency metric; fc1/down/lm were
  already DRAM-bound and unmoved). **Probe F, three reruns (the only arbiter):
  all-2D(m64u8) 12.03, qkv2D+2D 11.99 vs all-MPS 12.19 / shipping 12.18
  ms/branch — real but only +1.6%, below the "clearly below 12.2" bar**.
  Production serve A/B (seed 831001): gpu busy 27.34→25.94s but wall-clock
  e2e 29.8→29.9s (sampling trajectories diverge 746/747 tokens; wall clock
  rules) — **zero production gain**. Conclusion: the decode chain is no
  longer determined by GEMV bandwidth — probe F's ~12ms *is* the GEMM kernel
  execution itself (~246GB/s kernel-efficiency floor, M9 profile; the
  "520GB/s streaming floor" is a physical limit, not a kernel-attainable
  value), and the dispatch tax is a separate ~0.6ms/branch-step (M12 slope
  -stub measurement). Default path unchanged (fused qkv per-column + MPS);
  the 2D kernels are kept as env-selectable variants; the remaining semantic
  headroom is in kernel execution efficiency (see known limitations: the
  scheduling track is closed).

## Numerical Conventions

- **melody_abc greedy output is bit-identical to lyra** (tests/test_sampling
  e2e comparison gate).
- semantic (cot=off legacy bf16 path) has a bf16-intrinsic argmax-flip band:
  a divergence step from lyra with margin ≤ 1.0 is accepted (split-K SDPA:
  off_semantic first flip at step40 margin=0.0625, melody_semantic step8
  margin=0.0625; the flash2 fallback path flips at step24 margin=0 — the flip
  step drifting with the rounding path is expected; the gate is unchanged).
- Initial noise uses the engine's own splitmix64 stream (M0 decision): the
  **same seed does not bit-reproduce lyra's PCG64 sampling**; song
  length/melody distributions are equivalent; listening comparisons are by ear.
- bf16 weights, fp32 accumulation, RNE at op boundaries; each stage's
  comparison thresholds are calibrated to the "dual-implementation intrinsic
  noise band" (fixture gates in tests/; wiring bugs measure ≥0.3, giving
  7×+ discrimination).

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `H3_QKV_GEMV` | on | =0 disables the fused q/k/v GEMV (falls back to three MPS linears) |
| `H3_GEMV` | on for yue | rows==1 single-matrix GEMV fast path; yue processes default `1` + `qw_gemv2d_m64u8` (M14; explicit env wins; `H3_GEMV=0` falls back to all-MPS); qwen keeps h3's own default (off) |
| `H3_GEMV_KERNEL` | 2D for yue | GEMV variant (u2/u4/u8/c2u4/g128…, 2D-block family `qw_gemv2d_m4…m64u8`; since M14 yue defaults to `qw_gemv2d_m64u8`) |
| `H3_GEMV_MAX_OUT` / `H3_GEMV_MIN_IN` | none | per-shape knobs: use MPS when output_dim ≥ / input_dim < the threshold |
| `H3_QKV_GEMV_KERNEL` | qw_gemv_qkv_bf16_u8 | fused GEMV variant (u4/u8 or 2D `qw_gemv2d_qkv_m32u8`/`m64u8`; the latter requires out_q/out_kv divisible by row_tile) |
| `H3_ZERO_COPY_WEIGHTS` | 1 | zero-copy mmap weights (not counted in footprint, re-readable on demand); =0 private copies. M8 measured no performance impact on GEMV or MPS |
| `YUE_VAE_CPU` | off | =1 runs VAE decode on the Accelerate CPU reference path (GPU-path comparison / fallback) |
| `H3_FFMPEG` / `H3_FFPROBE` | none | override ffmpeg/ffprobe executables; yue itself only calls ffmpeg (WAV24→FLAC, see the ffmpeg section) — ffprobe is used only by the vendored qwen-side helpers. PATH by default |
| `YUE_SDPA_FLASH2` | off | =1 falls decode-step attention back to the flash2 tile kernel (default split-K scalar pair: probe U kv=513 134→28.5µs, see M10) |
| `YUE_HN_ROPE_SEPARATE` | off | =1 falls q/k head-norm+rope back to the four-kernel path (default fused single dispatch: bit-identical, probe T +1.39 ms/branch, see M11) |
| `YUE_AR_CONTINUE_EVERY` | 1 | layer-tail commit granularity (continue every N layers; measured GPU busy unchanged) |
| `YUE_SERVE_MEM` | off | =1 prints `[mem]` (live/peak/allocated/device-alloc/footprint) at request boundaries plus a `[gpu]` stats line (the standard A/B and memory-debugging tool) |
| `YUE_HOST_TRACE` | off | =1 prints the per-token semantic breakdown `[host]` (wall/enc/wait/read/guided/unpack/dist/sample) every 512 steps (M14 diagnostics) |
| `YUE_CFG_BATCH` | on | =0 falls back to serial dual-branch decode (M15 batch fusion escape hatch) |
| `YUE_BATCH_DBG` | off | =1 scans batched-decode logits for non-finite values each step; on hit prints the step/row distribution and dumps /tmp/batch_nan_s*.bin |
| `YUE_NO_PREFILL_PAD` | off | =1 disables prefill row-count bucketing (`yue_ar_forward` pads rows to multiples of 512 with token 0 — real rows are bit-identical under the causal limit; memory-debugging comparison) |
| `YUE_RESTART_AFTER` | 10 | in serve mode, execve self-restarts at request boundaries every N requests (same PID/fds, invisible to the host) to return driver-held Metal pages; =0 disables |
| `YUE_ALLOC_TRACE` | off | =1 prints Metal allocations ≥8MB with call stacks (memory attribution) |
| `H3_KEEP_GRAPHS` | off | =1 skips clearing the MPSGraph graph cache at request boundaries (attribution experiments) |
| `H3_DISABLE_MPS_LINEAR` | off | falls back to the scalar 16×16-tile linear kernels |
| `QWEN_AR_TRACE` / `QWEN_AR_TRACE_LAYER0` / `QWEN_AR_DUMP` | off | per-layer tracing / layer0 dump / full dump (debugging) |
| `QWEN_VAE_DUMP` | off | per-stage VAE activation dumps (fixture comparison; the GPU path auto submit/read) |

## Tests & Fixtures

Seven regression suites (all pass with `0 failure(s)`): `m0` (smoke),
`tokenizer` (tiktoken bit-exact), `gemv` (k-mapping + bit-level comparison +
fused qkv), `ar` (prefill/step parity vs lyra fixtures), `sampling` (11 unit
cases + melody_abc/semantic greedy e2e), `nar` (conditioning cache / velocity
/ solver / multi-chunk), `vae` (67 stages + audio comparison; decode uses the
GPU path by default and silently falls back to CPU when Metal is unavailable;
`YUE_VAE_CPU=1` forces the CPU path).

Fixture generation (needs the reference-stack venv): `make tokenizer-fixture
ar-fixture sampling-fixture nar-fixture vae-fixture` (nar-fixture needs
`LYRA_FORCE_BF16_SDPA=1`). The `tests/gen_*_fixture.py` scripts still point at
the retired lyra conversion directory; the shipped `.npy/.json` fixtures are
self-contained, and regenerating them requires re-running the lyra conversion
from the official weights first. Micro-benchmarks:
`make bench-gemv bench-dispatch`; `./tests/bench_dispatch F` runs probe F
only (the production-rhythm decode chain: true dependencies + layer-tail
continue + CFG dual branches + 2048-step thermal soak — the main probe
arbiter for kernel quality, conclusions in the file header comment; `G` =
per-shape true-dependency bandwidth calibration; `T`/`U`/`S` = timeline
ablation / SDPA microbenchmark / 2D sweep; `V` = the M18 split-K partial
write-loss matrix (R1-R12); `H` = the rows=2 bit-exactness / efficiency gate
for CFG batch fusion).

## Architecture Map

| File | Responsibility |
|---|---|
| `yue_ar.{h,m}` | 28-layer GQA AR backbone: KV cache, RMS/RoPE/SDPA, fused qkv GEMV, abc/semantic phase projections |
| `yue_sampling.{c,h}` | fp32 (abc/CFG) and legacy-bf16 (cot=off) dual sampling paths, window frequency power penalty, splitmix64 RNG |
| `yue_generate.{c,h}` | AR generation loop: CFG dual branch (single-stream batch `decode_step2` since M15, `YUE_CFG_BATCH=0` falls back to serial), guided_logits |
| `yue_nar.{h,m}` | NAR acoustic flow matching: conditioning prefill reuses the AR cache, non-causal prefix SDPA, midpoint solver |
| `yue_vae.{c,h}` | Oobleck VAE decode: GPU path (TensorOps matmul2d per-tap GEMM + zero ring, default) and Accelerate CPU reference path (`YUE_VAE_CPU=1`), halo tiling, WAV24 + ffmpeg FLAC |
| `yue_gpu_ext.h` | GPU wrapper declarations for the yue/qwen side (qwen VAE family `qw_*` + yue acoustic VAE family `yv_*`) |
| `yue_protocol.c` / `yue_json.c` | protocol constants / instruction templates / token prefixes / JSON parsing |
| `main.c` | serve (JSONL) + REPL + GPU lock + engine lifecycle |
| `h3_*` (vendored) | Metal runtime / weights / safetensors / tokenizer (MIT, from h3-metal via qwen-metal; includes qwen's added `qw_*` kernels and yue's added `yv_*` acoustic VAE kernels, all registered in h3_shaders.metal + h3_gpu.m + the pipeline list) |

Milestones: M0 skeleton → M1 tokenizer → M2 AR parity → M3 sampling +
generation loop → M4 NAR → M5 VAE+FLAC → M6 serve/REPL full pipeline → M7
performance rounds (four) → M8 VAE GPU + GEMV mystery closed → M9 MLX-style
2D-block GEMV (rejected twice) → M10 decode split-K SDPA landed → M11
attention prologue fusion → M12 graph-capture verdict (scheduling closed) →
M13 GEMM track final review (full-fan + per-shape calibration, kernels
closed) → M14 host-tail attribution fix + decode GEMV switched to the 2D
block kernel → M15 CFG batch fusion (semantic −40%, overtakes lyra) → M18
split-K SDPA long-KV correctness fix (full cacheline writes + 16-split cap;
the vectorized rewrite was judged to bring no gain).
M16/M17 once implemented `--precision 8bit` (lyra ar-8bit weights + int8
GEMV); measured to save memory only, not speed (slightly slower, ~4%); it
has been removed entirely.

## Known Limitations

- **Semantic breakdown (M14 instrumentation profile, before batch fusion,
  ~19ms/token)**: GEMM ~10.5-12 (streaming floor, M13) + MPSGraph linear
  encode ~3.1 (inherent tax; all three alternative paths abandoned) +
  own-dispatch encoding ~1.3 + split SDPA ~2-3 (grows with KV; the ~0.5-1ms
  constant term is the last movable item) + small kernels ~1 (fused to the
  end) + host 0.11. Slope-stub measurement: the real chain costs 0.89-1.08µs
  per dispatch marginally — there is no hidden bubble pool; whole-step
  MPSGraph graph capture is structurally infeasible (custom kernels cannot
  enter MPSGraph, and the bf16 rounding order is bit-locked by the
  melody_abc gate). **The decode chain is fully closed** (scheduling M12,
  kernels M13, encoding/host M14). M15's batch fusion merged the duplicated
  dual-branch GEMM weight streams into one pass; semantic 11.7ms/token is
  now **ahead** of lyra's same-profile ≈15.1 (−22%), with only the SDPA
  constant term left as an option; flow/save are at parity or ahead.
- Semantic generation does one logits readback + host sampling per token,
  costing one GPU drain per step plus a ~1.2ms host tail (per-step top-2
  margins are recorded).
- VAE GPU weights are resident (~150MB fp32 after repacking); VAE decode
  runs on 64-aligned tiles and its output sits in the same intrinsic band as
  the CPU reference path (not bit-identical).
- **Continuous-generation memory profile (2026-11-02 measurement + two
  rounds of fixes)**: h3 tensor live is byte-for-byte flat and `leaks` under
  `MallocStackLogging` reports 0 — no leaks; the footprint ratchet comes
  entirely from "shape drift × driver pool" (Metal freed pages are never
  returned to the OS; the driver pool reuses by exact size). Five layers of
  mitigation are in place: ① KV cache / NAR chunk capacities bucketed to
  512-row multiples (`YUE_SHAPE_BUCKET` is a compile-time constant in
  `yue.h`, not an environment variable); ② VAE halo-tile last-block start
  pull-back (each tile's input converges to {core+halo, core+2*halo};
  overlap regions decode twice to the same value thanks to the halo —
  bit-identical); **③ prefill row-count bucketing** (`yue_ar_forward` pads
  count>1 forwards to 512-row multiples with token 0 — real rows are
  bit-identical under the causal limit; count==1 decode is not padded;
  `YUE_NO_PREFILL_PAD=1` falls back; covers the three song-length-dependent
  shape sources semantic/abc/NAR-cond); **④ VAE GPU decode scratch hoisted
  across tiles** (bufs[3]+ring allocated once per song — previously rebuilt
  per tile, and the ring size family differed by only KBs (t×cin varies by
  layer), shredding freed large blocks into non-reusable small chunks;
  measured ~135MB-class pool regions per tile, ~2GB/song — the main ratchet
  for long cot=full songs; after hoisting, 6 songs' footprint went 9.7→11.2GB
  flat vs 12→24.4GB before); ⑤ **serve-mode `YUE_RESTART_AFTER` (default 10
  songs) execve self-restarts at request boundaries** — same PID, same
  stdin/stdout fds, fully invisible to the host; weight mmap lazily reloads
  in ~1s and the footprint periodically returns to zero. Current
  measurement (6 cot=full songs with mixed lengths back-to-back): live flat
  at 9.08GB, peak 11.8GB, footprint 9.7→11.2GB flat (the residual
  ~0.25GB/song is NAR velocity row-count MPS shape drift, backstopped by
  self-restart). `allocated` only ever grows — it is a cumulative-throughput
  metric; the leak criterion is the `[mem] live` line from
  `YUE_SERVE_MEM=1` (plus `device-alloc` = currentAllocatedSize, which
  separates "tensor bookkeeping" from "resources still held by the driver").
  For comparison, the MLX stack (lyra) is naturally flat — same/different
  seeds and different song lengths back-to-back hold the footprint at
  ~7.55GB (`mx.clear_cache()` explicitly returns the user-space pool +
  per-kernel deterministic execution keeps the shape space stable; native's
  MPS rows=1 GEMM has 1-ulp run-to-run nondeterminism that chaotic decoding
  amplifies into shape drift, hence the five-layer mitigation + self-restart
  backstop).
- Apple Silicon only, and effectively M5-series: TensorOps kernels are enabled
  by device name; on non-M5 chips or with `H3_NAX=0`, the pipeline list and the
  shader's conditional compilation disagree and engine creation fails (known
  platform limitation). The preferred weights layout is the official single-file
  directory (`YuE2-3B/model.safetensors` + `YuE2-Vae`), which has no lyra
  conversion constraints; only the legacy lyra layout (`ar-bf16`/`nar-bf16`)
  requires its directory rules (no symlinks, manifest tolerates zero extra
  files, both weight files required).
- Not a git-tracked development tree here; vendored `linenoise.c` carries 30
  legacy compile warnings (own code compiles warning-free).
