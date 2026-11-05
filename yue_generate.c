/* AR generation loop (lyra ar.py generate_tokens parity). See yue_generate.h. */
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "yue_generate.h"

/* Temporary host-tail instrumentation (YUE_HOST_TRACE=1): per-token cost
 * of the semantic decode loop's non-GPU segments. M14 diagnostic, to be
 * removed once the host tail is optimized. */
static double ht_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static struct {
    double wait, read, guided, unpack, dist, sample, enc, wall;
    long n;
    int on;
} ht;

static void set_error(char *error, size_t error_size, const char *fmt, ...) {
    if (!error || error_size == 0) return;
    va_list args;
    va_start(args, fmt);
    vsnprintf(error, error_size, fmt, args);
    va_end(args);
}

static const char *phase_name(yue_phase phase) {
    return phase == YUE_PHASE_ABC ? "abc" : "semantic";
}

void yue_generation_free(yue_generation *gen) {
    if (!gen) return;
    free(gen->tokens);
    free(gen->margins);
    free(gen);
}

/* Validation mirrors lyra generate_tokens + protocol.Sampling. NaN passes
 * every range comparison, so the finite checks are explicit. */
static int check_sampling(const yue_sampling *sp, char *error,
                          size_t error_size) {
    const float temp_f = (float)sp->temperature;
    const float penalty_f = (float)sp->repetition_penalty;
    if (!isfinite(sp->temperature) || !isfinite(sp->top_p) ||
        !isfinite(sp->repetition_penalty) ||
        sp->temperature < 0.0 || sp->temperature > 5.0 ||
        sp->top_p <= 0.0 || sp->top_p > 1.0 || sp->top_k < 1 ||
        sp->repetition_penalty <= 0.0 || sp->penalty_window < 1 ||
        sp->penalty_window > YUE_MAX_WINDOW || sp->min_tokens < 0 ||
        sp->min_tokens > sp->max_tokens || sp->max_tokens < 1 ||
        /* double values can underflow/overflow when cast to float: a
         * 1e-300 temperature would divide by zero, a 1e308 penalty
         * would become inf. Reject those at the boundary. */
        !isfinite(temp_f) || temp_f < 0.0f ||
        !isfinite(penalty_f) || penalty_f <= 0.0f) {
        set_error(error, error_size, "invalid sampling parameters");
        return 0;
    }
    return 1;
}

/* One decode branch: cache + a [1, width] bf16 logits row + its host bits. */
typedef struct {
    yue_ar_cache *cache;
    h3_gpu_tensor *logits;
    uint16_t *host;
    size_t width;
    yue_phase phase;
    size_t prefix_len;
} gen_branch;

static void branch_free(gen_branch *b) {
    if (!b) return;
    yue_ar_cache_free(b->cache);
    h3_gpu_tensor_free(b->logits);
    free(b->host);
    memset(b, 0, sizeof(*b));
}

/* Project the LAST row of a forward result and read it to host bits. */
static int project_last_row(yue_ar *ar, const h3_gpu_tensor *hidden,
                            size_t rows, gen_branch *b, char *error,
                            size_t error_size) {
    size_t offset = (rows - 1) * YUE_HIDDEN;
    h3_gpu_tensor *last = h3_gpu_tensor_new_bf16(yue_ar_gpu(ar), YUE_HIDDEN);
    int opened = 0;
    int ok = last && (opened = h3_gpu_begin(yue_ar_gpu(ar))) &&
             h3_gpu_copy_bf16(yue_ar_gpu(ar), last, 0, hidden, offset,
                              YUE_HIDDEN) &&
             h3_gpu_submit(yue_ar_gpu(ar)) &&
             yue_ar_project_logits(ar, last, 1, phase_name(b->phase),
                                   b->logits, error, error_size);
    if (opened && !ok) h3_gpu_submit(yue_ar_gpu(ar)); /* drain: copy failed */
    h3_gpu_tensor_free(last);
    if (!ok) {
        /* The GPU helpers above do not all write an error (a failed
         * h3_gpu_begin/copy/submit returns 0 silently). Keep the caller's
         * more specific message if one was already set. */
        if (error && error_size && !error[0])
            set_error(error, error_size,
                      "prefill logits projection failed: %s",
                      h3_gpu_error(yue_ar_gpu(ar)));
        return 0;
    }
    return h3_gpu_tensor_read_bf16(b->logits, b->host, b->width);
}

/* One merged decode step. Both branch forwards and their logit projections
 * encode into a single command-buffer chain (each layer tail still commits
 * without waiting, so the GPU stays busy while later layers encode) and ONE
 * submit waits for the whole step. The former per-branch
 * forward/submit/copy/submit/project/submit cadence put four wait
 * round-trips between every token and dominated the semantic wall once the
 * encode overlapped with the GPU. */
static int decode_step(yue_ar *ar, const uint32_t *tokens, size_t count,
                       gen_branch *pos, gen_branch *neg, char *error,
                       size_t error_size) {
    h3_gpu *gpu = yue_ar_gpu(ar);
    double he0 = ht.on ? ht_now() : 0;
    if (!h3_gpu_begin(gpu)) {
        set_error(error, error_size, "decode step begin failed: %s",
                  h3_gpu_error(gpu));
        return 0;
    }
    h3_gpu_tensor *hidden = yue_ar_forward_open(ar, tokens, count, pos->cache,
                                                error, error_size);
    int ok = hidden &&
             yue_ar_project_open(ar, hidden, count, phase_name(pos->phase),
                                 pos->logits, error, error_size);
    if (ok && neg) {
        hidden = yue_ar_forward_open(ar, tokens, count, neg->cache, error,
                                     error_size);
        ok = hidden &&
             yue_ar_project_open(ar, hidden, count, phase_name(neg->phase),
                                 neg->logits, error, error_size);
    }
    if (!ok) {
        h3_gpu_submit(gpu); /* drain the half-encoded chain */
        return 0;
    }
    double ht0 = ht.on ? ht_now() : 0;
    if (!h3_gpu_submit(gpu)) {
        set_error(error, error_size, "decode step submit failed: %s",
                  h3_gpu_error(gpu));
        return 0;
    }
    double ht1 = ht.on ? ht_now() : 0;
    if (!h3_gpu_tensor_read_bf16(pos->logits, pos->host, pos->width) ||
        (neg && !h3_gpu_tensor_read_bf16(neg->logits, neg->host,
                                         neg->width))) {
        set_error(error, error_size, "logits readback failed");
        return 0;
    }
    double ht2 = ht_now();
    if (ht.on) {
        ht.enc += ht0 - he0;
        ht.wait += ht1 - ht0;
        ht.read += ht2 - ht1;
    }
    return 1;
}

/* Batched CFG decode step (M15): both branches ride one [2, hidden]
 * forward (probe H: rows=2 = 0.98x rows=1 on the big shapes, so the
 * weight stream is read once instead of twice) and one [2, width]
 * logits projection + readback. logits2/host2 are caller-owned
 * [2*width] buffers reused across steps; pos->host / neg->host are
 * pointed at the two halves. The sampling path is unchanged —
 * guided_logits sees the same two host rows. */
static int decode_step2(yue_ar *ar, const uint32_t *token, gen_branch *pos,
                        gen_branch *neg, h3_gpu_tensor *logits2,
                        uint16_t *host2, long step, char *error,
                        size_t error_size) {
    h3_gpu *gpu = yue_ar_gpu(ar);
    double he0 = ht.on ? ht_now() : 0;
    if (!h3_gpu_begin(gpu)) {
        set_error(error, error_size, "batch decode begin failed: %s",
                  h3_gpu_error(gpu));
        return 0;
    }
    h3_gpu_tensor *hidden = yue_ar_forward_open2(ar, token[0], token[0],
                                                 pos->cache, neg->cache,
                                                 error, error_size);
    int ok = hidden &&
             yue_ar_project_open(ar, hidden, 2, phase_name(pos->phase),
                                 logits2, error, error_size);
    if (!ok) {
        h3_gpu_submit(gpu); /* drain the half-encoded chain */
        return 0;
    }
    double ht0 = ht.on ? ht_now() : 0;
    if (!h3_gpu_submit(gpu)) {
        set_error(error, error_size, "batch decode submit failed: %s",
                  h3_gpu_error(gpu));
        return 0;
    }
    double ht1 = ht.on ? ht_now() : 0;
    if (!h3_gpu_tensor_read_bf16(logits2, host2, 2 * pos->width)) {
        set_error(error, error_size, "batch logits readback failed");
        return 0;
    }
    double ht2 = ht_now();
    if (ht.on) {
        ht.enc += ht0 - he0;
        ht.wait += ht1 - ht0;
        ht.read += ht2 - ht1;
    }
    if (getenv("YUE_BATCH_DBG")) {
        size_t np = 0, nn = 0, ip = 0, in = 0;
        for (size_t i = 0; i < pos->width; i++) {
            uint32_t a = (uint32_t)host2[i] << 16;
            uint32_t b = (uint32_t)host2[pos->width + i] << 16;
            float fa, fb;
            memcpy(&fa, &a, 4);
            memcpy(&fb, &b, 4);
            if (!isfinite(fa)) {
                if (!np) ip = i;
                np++;
            }
            if (!isfinite(fb)) {
                if (!nn) in = i;
                nn++;
            }
        }
        if (np || nn) {
            fprintf(stderr,
                    "[dbg] step %ld logits nonfinite: pos %zu (first %zu)"
                    " neg %zu (first %zu)\n", step, np, ip, nn, in);
            char path[64];
            snprintf(path, sizeof(path), "/tmp/batch_nan_s%ld.bin", step);
            FILE *f = fopen(path, "wb");
            if (f) {
                fwrite(host2, 2, 2 * pos->width, f);
                fclose(f);
                fprintf(stderr, "[dbg] dumped %s (token %u)\n", path,
                        token[0]);
            }
        }
    }
    return 1;
}

int yue_ar_generate(yue_ar *ar, const uint32_t *prefix, size_t prefix_len,
                    const uint32_t *negative, size_t negative_len,
                    double cfg_scale, yue_phase phase, const yue_sampling *sp,
                    uint64_t seed, int legacy, yue_token_fn on_token,
                    void *ctx, yue_generation **out, char *error,
                    size_t error_size) {
    if (!ar || !prefix || !prefix_len || !sp || !out) {
        set_error(error, error_size, "invalid generate arguments");
        return 0;
    }
    if (phase != YUE_PHASE_ABC && phase != YUE_PHASE_SEMANTIC) {
        set_error(error, error_size, "invalid generation phase");
        return 0;
    }
    if (!isfinite(cfg_scale) || cfg_scale < 0.0 || cfg_scale > 20.0 ||
        (phase == YUE_PHASE_ABC && cfg_scale != 1.0)) {
        set_error(error, error_size,
                  "invalid CFG scale (ABC phase requires no guidance)");
        return 0;
    }
    /* Validate before the length checks: a negative max_tokens wraps its
     * size_t casts, so the context arithmetic below would misreport. */
    if (!check_sampling(sp, error, error_size)) return 0;
    if (prefix_len + (size_t)sp->max_tokens > (size_t)YUE_CONTEXT) {
        set_error(error, error_size,
                   "prefix + max_tokens exceeds the 24576 context");
        return 0;
    }
    const int use_cfg = cfg_scale != 1.0;
    /* Batched-CFG step buffers (M15), declared before the first fail
     * jump so the cleanup path always sees initialized values. */
    h3_gpu_tensor *logits2 = NULL;
    uint16_t *host2 = NULL;
    if (use_cfg && (!negative || !negative_len)) {
        set_error(error, error_size, "CFG requires a negative prefix");
        return 0;
    }
    if (use_cfg && negative_len + (size_t)sp->max_tokens > YUE_CONTEXT) {
        set_error(error, error_size,
                   "negative prefix + max_tokens exceeds the context");
        return 0;
    }

    gen_branch pos, neg;
    memset(&pos, 0, sizeof(pos));
    memset(&neg, 0, sizeof(neg));
    pos.phase = phase;
    neg.phase = phase;
    pos.width = yue_phase_width(phase);
    neg.width = pos.width;
    const int end_local = yue_phase_end_local(phase);

    yue_generation *gen = calloc(1, sizeof(*gen));
    float *scores = malloc(sizeof(float) * pos.width);
    uint16_t *guided = use_cfg ? malloc(pos.width * 2) : NULL;
    yue_rng rng;
    int ok = gen && scores && (!use_cfg || guided);
    if (!ok) {
        set_error(error, error_size, "out of memory");
        goto fail;
    }
    gen->capacity = 64;
    gen->tokens = malloc(sizeof(uint32_t) * gen->capacity);
    if (!gen->tokens) {
        set_error(error, error_size, "out of memory");
        ok = 0;
        goto fail;
    }

    pos.prefix_len = prefix_len;
    /* Capacity bucketed (see yue.h): prefix length drifts a few tokens per
     * request; the exact row counts below drive all write/read logic. */
    pos.cache = yue_ar_cache_create(
        ar, yue_align_rows(prefix_len + (size_t)sp->max_tokens),
        error, error_size);
    pos.logits = h3_gpu_tensor_new_bf16(yue_ar_gpu(ar), pos.width);
    pos.host = malloc(pos.width * 2);
    ok = pos.cache && pos.logits && pos.host;
    if (!ok) {
        set_error(error, error_size, "out of memory");
        goto fail;
    }
    if (use_cfg) {
        neg.prefix_len = negative_len;
        neg.cache = yue_ar_cache_create(
            ar, yue_align_rows(negative_len + (size_t)sp->max_tokens),
            error, error_size);
        neg.logits = h3_gpu_tensor_new_bf16(yue_ar_gpu(ar), neg.width);
        neg.host = malloc(neg.width * 2);
        ok = neg.cache && neg.logits && neg.host;
        if (!ok) {
            set_error(error, error_size, "out of memory");
            goto fail;
        }
    }
    /* Batched-CFG step buffers (M15): one [2, width] logits tensor and
     * its host twin, reused across every decode step. */
    if (use_cfg) {
        logits2 = h3_gpu_tensor_new_bf16(yue_ar_gpu(ar), 2 * pos.width);
        host2 = malloc(2 * pos.width * 2);
        if (!logits2 || !host2) {
            set_error(error, error_size, "out of memory");
            ok = 0;
            goto fail;
        }
    }

    /* Prefill: run the backbone, project the last hidden row per branch. */
    {
        h3_gpu_tensor *hidden =
            yue_ar_forward(ar, prefix, prefix_len, pos.cache, error,
                           error_size);
        ok = hidden && project_last_row(ar, hidden, prefix_len, &pos, error,
                                        error_size);
        if (!ok) goto fail;
        if (use_cfg) {
            hidden = yue_ar_forward(ar, negative, negative_len, neg.cache,
                                    error, error_size);
            ok = hidden && project_last_row(ar, hidden, negative_len, &neg,
                                            error, error_size);
            if (!ok) goto fail;
        }
    }

    if (getenv("QWEN_AR_TRACE"))
        fprintf(stderr, "[pre] pos %04x %04x %04x neg %04x %04x %04x\n",
                pos.host[0], pos.host[1], pos.host[2],
                use_cfg ? neg.host[0] : 0, use_cfg ? neg.host[1] : 0,
                use_cfg ? neg.host[2] : 0);
    /* The two stages reset the request-local seed (lyra convention). */
    const char *batch_env = getenv("YUE_CFG_BATCH");
    const int batch_on = !batch_env || batch_env[0] != '0';
    yue_rng_init(&rng, seed);
    size_t margin_count = 0, margin_cap = 0;
    const char *trace_env = getenv("QWEN_AR_TRACE");
    ht.on = getenv("YUE_HOST_TRACE") != NULL;
    ht.wait = ht.read = ht.guided = ht.unpack = ht.dist = ht.sample = 0;
    ht.enc = ht.wall = 0;
    ht.n = 0;
    double ht_win0 = ht.on ? ht_now() : 0;

    for (int step = 0; step < sp->max_tokens; step++) {
        const uint16_t *cond = pos.host;
        double h0 = ht.on ? ht_now() : 0;
        if (use_cfg) {
            /* M15: with the batched step both rows live in host2 (the
             * gen_branch host pointers stay owned by their own mallocs —
             * pointing them into host2 would double-free at cleanup). */
            /* step 0 samples the PREFILL logits (pos.host/neg.host);
             * host2 is written by each batched step for the NEXT token. */
            if (batch_on && step > 0)
                yue_guided_logits(host2, host2 + pos.width, pos.width,
                                  cfg_scale, guided);
            else
                yue_guided_logits(pos.host, neg.host, pos.width, cfg_scale,
                                  guided);
            cond = guided;
        }
        double h1 = ht.on ? ht_now() : 0;
        for (size_t i = 0; i < pos.width; i++) {
            uint32_t wide = (uint32_t)cond[i] << 16;
            memcpy(&scores[i], &wide, sizeof(float));
        }
        double h2 = ht.on ? ht_now() : 0;
        if (!yue_distribution(scores, pos.width, phase, sp, gen->tokens,
                              gen->count, step, legacy)) {
            set_error(error, error_size,
                      "invalid distribution parameters or out of memory");
            ok = 0;
            goto fail;
        }

        double h3 = ht.on ? ht_now() : 0;
        int local;
        if (sp->temperature == 0.0) {
            /* All-NaN scores (broken logits) must fail like the sampled
             * path does, not silently pick token 0. */
            int any_finite = 0;
            for (size_t i = 0; i < pos.width; i++)
                if (scores[i] == scores[i]) {
                    any_finite = 1;
                    break;
                }
            if (!any_finite) {
                set_error(error, error_size,
                          "sampling produced no finite score");
                ok = 0;
                goto fail;
            }
            local = -1;
            for (size_t i = 0; i < pos.width; i++)
                if (isfinite(scores[i]) &&
                    (local < 0 || scores[i] > scores[local])) local = (int)i;
            /* Greedy diagnostic: margin between the top-2 candidates. Two
             * bf16 implementations flip argmax on near-ties (within the
             * intrinsic noise band), so parity tests accept token flips
             * only at small margins. */
            int second = -1;
            for (size_t i = 0; i < pos.width; i++)
                if ((int)i != local && isfinite(scores[i]) &&
                    (second < 0 || scores[i] > scores[second]))
                    second = (int)i;
            double margin = second >= 0
                                ? (double)scores[local] - (double)scores[second]
                                : 0.0;
            if (trace_env)
                fprintf(stderr,
                        "[g] step %d local %d (%.6g) runner-up %d (%.6g)\n",
                        step, local, (double)scores[local], second,
                        second >= 0 ? (double)scores[second] : 0.0);
            if (margin_count == margin_cap) {
                size_t grown = margin_cap ? margin_cap * 2 : 64;
                double *moved = realloc(gen->margins, sizeof(double) * grown);
                if (!moved) {
                    set_error(error, error_size, "out of memory");
                    ok = 0;
                    goto fail;
                }
                gen->margins = moved;
                margin_cap = grown;
            }
            gen->margins[margin_count++] = margin;
        } else {
            local = yue_sample_categorical(scores, pos.width, legacy, &rng);
            if (local < 0) {
                set_error(error, error_size, "sampling produced no finite score");
                ok = 0;
                goto fail;
            }
        }

        double h4 = ht.on ? ht_now() : 0;
        if (ht.on) {
            ht.guided += h1 - h0;
            ht.unpack += h2 - h1;
            ht.dist += h3 - h2;
            ht.sample += h4 - h3;
            ht.n++;
            if (ht.n % 512 == 0) {
                double htn = (double)ht.n;
                double htw = ht_now() - ht_win0;
                fprintf(stderr,
                        "[host] %s n=%ld wall %.2f enc %.2f wait %.2f "
                        "read %.2f guided %.3f unpack %.3f dist %.3f "
                        "sample %.3f ms/tok\n",
                        phase_name(phase), ht.n, htw / htn * 1e3,
                        ht.enc / htn * 1e3, ht.wait / htn * 1e3,
                        ht.read / htn * 1e3, ht.guided / htn * 1e3,
                        ht.unpack / htn * 1e3, ht.dist / htn * 1e3,
                        ht.sample / htn * 1e3);
                ht.wait = ht.read = ht.guided = ht.unpack = ht.dist = 0;
                ht.sample = ht.enc = 0;
                ht.n = 0;
                ht_win0 = ht_now();
            }
        }
        const uint32_t token = yue_local_to_native(local, phase);
        if (on_token) on_token(phase, token, ctx);
        if (local == end_local) {
            gen->eos = 1;
            break;
        }
        if (gen->count == gen->capacity) {
            size_t grown = gen->capacity * 2;
            uint32_t *moved = realloc(gen->tokens, sizeof(uint32_t) * grown);
            if (!moved) {
                set_error(error, error_size, "out of memory");
                ok = 0;
                goto fail;
            }
            gen->tokens = moved;
            gen->capacity = grown;
        }
        gen->tokens[gen->count++] = token;

        if (step + 1 < sp->max_tokens) {
            if (use_cfg && batch_on) {
                if (!decode_step2(ar, &token, &pos, &neg, logits2, host2,
                                  step, error, error_size))
                    goto fail;
            } else if (!decode_step(ar, &token, 1, &pos,
                                    (use_cfg && !batch_on) ? &neg : NULL,
                                    error, error_size)) {
                goto fail;
            }
        }
    }

    gen->truncated = !gen->eos;
    gen->margin_count = margin_count;
    free(scores);
    free(guided);
    free(host2);
    h3_gpu_tensor_free(logits2);
    branch_free(&pos);
    branch_free(&neg);
    *out = gen;
    return 1;

fail:
    free(scores);
    free(guided);
    free(host2);
    h3_gpu_tensor_free(logits2);
    branch_free(&pos);
    branch_free(&neg);
    if (gen) yue_generation_free(gen);
    return 0;
}
