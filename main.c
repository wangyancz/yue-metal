/* yue-metal CLI: `yue serve` (JSONL worker compatible with the yue2-web
 * python worker protocol) and `yue repl` (interactive session).
 *
 * M6 milestone: the full generation stack is wired in both modes:
 *   style/lyrics -> token prefixes -> AR abc planning (cot != off) ->
 *   AR semantic sampling (CFG per lyra SongRequest.guidance) ->
 *   acoustic flow matching (NAR chunks over the conditioning cache) ->
 *   VAE decode (halo tiles) -> 24-bit WAV -> ffmpeg FLAC.
 *
 * Serve protocol (mirrors tools/yue2-web/worker.py): requests
 * {"id","style","lyrics","cot","cfgScale","seed","output","abc"} produce
 * state loading/ready, progress (stage abc|semantic|flow|save with
 * percent spans 0.02-0.38 / 0.40-0.78 / 0.80-0.94 / 0.96-1.00, 4 s
 * heartbeats that never refresh the host idle timer), done(path,
 * seconds, tokens, sampleRate) and error(id, message). Cancellation is
 * the host killing the process group; stdin EOF exits.
 *
 * The optional "abc" string carries a user-supplied ABC score (lyra's
 * SongRequest.abc): its text tokens become the ABC prefix and the planning
 * AR pass is skipped. Omitting it keeps the original planning behaviour.
 *
 * The initial acoustic noise comes from the engine-local splitmix64
 * stream (M0 decision, documented): same seed does NOT replay lyra's
 * PCG64 draw bit-exactly. Greedy AR output matches lyra (see tests).
 */
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "linenoise.h"
#include "h3_gpu.h"
#include "h3_tokenizer.h"
#include "yue.h"
#include "yue_ar.h"
#include "yue_generate.h"
#include "yue_json.h"
#include "yue_nar.h"
#include "yue_protocol.h"
#include "yue_sampling.h"
#include "yue_vae.h"

#define YUE_DIR_MAX 1024
/* Request output paths (and the scratch WAV next to them) must fit. */
#define YUE_OUTPUT_MAX 3072
/* Request lines carry full lyrics as JSON; lyrics are capped by the
 * semantic budget anyway (max_tokens <= 9000), 512 KB is generous. */
#define SERVE_LINE_MAX (512u * 1024u)
/* User-supplied ABC scores: mirror the yue2-web lyrics cap and the ABC
 * planning stage's own max_tokens, so an external score cannot make the
 * prefill longer than a model-planned one. */
#define YUE_ABC_TEXT_MAX 20000u
#define YUE_ABC_TOKENS_MAX 4096u

typedef struct {
    char model_dir[YUE_DIR_MAX];
    char vae_dir[YUE_DIR_MAX];
    h3_gpu *gpu;
    h3_tokenizer *tokenizer;
    yue_ar *ar;
    yue_nar *nar; /* lazy: created by the first request that needs it */
    yue_vae *vae; /* lazy */
    yue_generation_config cfg;
    int serve; /* 1 = JSONL worker mode (progress events); 0 = REPL */
} yue_engine;

/* ----------------------------------------------------------------- misc */

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void log_stderr(const char *fmt, ...) {
    va_list args;
    fputs("[yue] ", stderr);
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    fflush(stderr);
}

static char *trim(char *s) {
    /* Trim in place: the return value must stay the original allocation
     * start, callers free() it (serve_handle / /lyrics-file). */
    char *start = s;
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') s++;
    size_t len = strlen(s);
    memmove(start, s, len + 1);
    char *end = start + len;
    while (end > start && (end[-1] == ' ' || end[-1] == '\t' ||
                           end[-1] == '\r' || end[-1] == '\n'))
        *--end = '\0';
    return start;
}

static const char *cot_name(int cot) {
    return cot == YUE_COT_FULL    ? "full"
           : cot == YUE_COT_MELODY ? "melody"
                                   : "off";
}

/* strtod-style tail check: only blanks may follow the number. */
static int blanks_only(const char *s) {
    while (*s == ' ' || *s == '\t') s++;
    return *s == '\0';
}

/* mkdir -p for the output path's parent; returns 1 when the directory
 * exists afterwards. */
static int ensure_parent_dir(const char *path) {
    /* REPL paths carry a "-<index>.flac" suffix (~21 bytes) on top of the
     * prefix, so size for the longest legal output, not just YUE_OUTPUT_MAX. */
    char buf[YUE_OUTPUT_MAX + 40];
    size_t len = strlen(path);
    if (len == 0 || len >= sizeof(buf)) return 0;
    memcpy(buf, path, len + 1);
    char *slash = strrchr(buf, '/');
    if (!slash || slash == buf) return 1; /* cwd-relative or root */
    *slash = '\0';
    for (char *p = buf + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(buf, 0755);
            *p = '/';
        }
    }
    struct stat st;
    if (stat(buf, &st) == 0 && S_ISDIR(st.st_mode)) return 1;
    if (mkdir(buf, 0755) != 0) {
        /* EEXIST also covers "a regular file with that name exists" —
         * re-check the type instead of treating it as success. */
        return stat(buf, &st) == 0 && S_ISDIR(st.st_mode);
    }
    return 1;
}

static char *read_text_file(const char *path) {
    FILE *handle = fopen(path, "rb");
    if (!handle) return NULL;
    if (fseek(handle, 0, SEEK_END) != 0) {
        fclose(handle);
        return NULL;
    }
    long size = ftell(handle);
    if (size < 0 || size > 16 << 20) {
        fclose(handle);
        return NULL;
    }
    if (fseek(handle, 0, SEEK_SET) != 0) {
        fclose(handle);
        return NULL;
    }
    char *text = malloc((size_t)size + 1);
    if (!text) {
        fclose(handle);
        return NULL;
    }
    size_t got = fread(text, 1, (size_t)size, handle);
    fclose(handle);
    /* A short read or an embedded NUL would silently truncate lyrics. */
    if (got != (size_t)size || memchr(text, '\0', got)) {
        free(text);
        return NULL;
    }
    text[got] = '\0';
    return text;
}

/* The h3 Metal runtime resolves "h3_shaders.metal" against the cwd. When
 * the binary is launched from elsewhere (e.g. a web tool spawning it),
 * fall back to the executable's directory instead of failing. */
static void ensure_shader_dir(void) {
    struct stat st;
    if (stat("h3_shaders.metal", &st) == 0) return;
    char exe[4096];
    uint32_t size = sizeof(exe);
    if (_NSGetExecutablePath(exe, &size) != 0) return;
    char *copy = strdup(exe);
    if (!copy) return;
    char *dir = dirname(copy);
    if (chdir(dir) != 0)
        log_stderr("note: cannot chdir to %s (h3_shaders.metal must be "
                   "reachable)",
                   dir);
    else
        log_stderr("note: resolved runtime directory %s", dir);
    free(copy);
}

/* Single-instance guard so two engines never share the GPU. The fd stays
 * open for the process lifetime; the kernel releases the lock on exit. */
static int acquire_gpu_lock(void) {
    char path[128];
    snprintf(path, sizeof(path), "/tmp/yue-metal-engine-%d.lock",
             (int)getuid());
    int fd = open(path, O_CREAT | O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0666);
    if (fd < 0) return -1;
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        close(fd);
        return 0;
    }
    return fd;
}

static double phys_footprint_gb(void) {
    struct task_vm_info info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info,
                  &count) != KERN_SUCCESS)
        return -1.0;
    return (double)info.phys_footprint / (1024.0 * 1024.0 * 1024.0);
}

/* ------------------------------------------------------- engine loading */

static int engine_load(yue_engine *eng, char *error, size_t error_size) {
    ensure_shader_dir();
    double t0 = now_seconds();
    eng->gpu = h3_gpu_create("h3_shaders.metal", error, error_size);
    if (!eng->gpu) return 0;

    char path[2048];
    snprintf(path, sizeof(path), "%s/qwen.tiktoken", eng->model_dir);
    eng->tokenizer = h3_tokenizer_load_tiktoken(path, error, error_size);
    if (!eng->tokenizer) return 0;

    if (!yue_generation_config_load(eng->model_dir, &eng->cfg, error,
                                    error_size))
        return 0;
    log_stderr("generation config: ode_steps=%d abc max_tokens=%d "
               "semantic max_tokens=%d",
               eng->cfg.ode_steps, eng->cfg.abc.max_tokens,
               eng->cfg.semantic.max_tokens);

    log_stderr("loading AR backbone to Metal ...");
    eng->ar = yue_ar_create(eng->gpu, eng->model_dir, error, error_size);
    if (!eng->ar) return 0;
    log_stderr("models resident in %.1fs", now_seconds() - t0);
    return 1;
}

/* Request boundary: the AR prefix length differs per song, so every
 * request would otherwise pin a new ~95 MB MPSGraph compile for the
 * process lifetime. */
static void request_boundary(yue_engine *eng) {
    if (eng->gpu) h3_gpu_clear_graph_cache(eng->gpu);
    if (!getenv("YUE_SERVE_MEM")) return;
    h3_gpu_stats st;
    if (h3_gpu_get_stats(eng->gpu, &st))
        /* Counters are cumulative since process start (load included);
         * single-request runs read them as per-request numbers. The
         * device-allocated line separates live tensor bookkeeping from
         * what the driver still holds as named resources. */
        log_stderr("[mem] live %.2f GB, peak live %.2f GB, allocated %.2f GB,"
                   " device-alloc %.2f GB, footprint %.2f GB",
                   (double)st.live_bytes / 1e9,
                   (double)st.peak_live_bytes / 1e9,
                   (double)st.allocated_bytes / 1e9,
                   (double)h3_gpu_device_allocated(eng->gpu) / 1e9,
                   phys_footprint_gb());
    if (h3_gpu_get_stats(eng->gpu, &st))
        log_stderr("[gpu] encode %.2fs, wait %.2fs, gpu busy %.2fs,"
                   " submissions %llu",
                   st.command_encode_seconds, st.command_wait_seconds,
                   st.gpu_seconds, (unsigned long long)st.submissions);
}

/* --------------------------------------------------- serve wire format */

static pthread_mutex_t g_emit_mu = PTHREAD_MUTEX_INITIALIZER;

static void serve_json_escape(const char *in, char *out, size_t out_size) {
    size_t o = 0;
    for (const unsigned char *p = (const unsigned char *)in;
         *p && o + 7 < out_size; p++) {
        unsigned char c = *p;
        if (c == '"' || c == '\\') {
            out[o++] = '\\';
            out[o++] = (char)c;
        } else if (c == '\n') {
            out[o++] = '\\';
            out[o++] = 'n';
        } else if (c == '\r') {
            out[o++] = '\\';
            out[o++] = 'r';
        } else if (c == '\t') {
            out[o++] = '\\';
            out[o++] = 't';
        } else if (c < 0x20) {
            o += (size_t)snprintf(out + o, out_size - o, "\\u%04x", c);
        } else {
            out[o++] = (char)c; /* UTF-8 passes through */
        }
    }
    out[o] = '\0';
}

/* Set right before the serve self-restart execve(). exec() destroys every
 * other thread (including the heartbeat), so an emit must not be in flight
 * and none may start afterwards. Guarded by g_emit_mu. */
static int g_restarting = 0;

/* One protocol line to stdout. The heartbeat thread writes concurrently,
 * so the write is serialized. The frame (JSON + newline) is written in a
 * single fwrite: otherwise execve() could kill a thread between the body
 * and its newline, splicing two JSON objects into one protocol line.
 * 32 KB covers the longest done event (a fully escaped 3 KB path, where
 * each control character expands to six bytes). */
static void serve_emit(const char *fmt, ...) {
    char line[32768];
    va_list args;
    va_start(args, fmt);
    int n = vsnprintf(line, sizeof(line), fmt, args);
    va_end(args);
    if (n <= 0) return;
    pthread_mutex_lock(&g_emit_mu);
    if (!g_restarting) {
        if ((size_t)n >= sizeof(line) - 1) {
            /* Never emit a truncated (invalid) JSON frame. */
            static const char too_long[] =
                "{\"type\":\"error\",\"id\":\"\","
                "\"message\":\"protocol event too large\"}\n";
            fwrite(too_long, 1, sizeof(too_long) - 1, stdout);
        } else {
            line[n++] = '\n';
            if (fwrite(line, 1, (size_t)n, stdout) != (size_t)n)
                log_stderr("protocol stdout write failed");
        }
        if (fflush(stdout) != 0) log_stderr("protocol stdout flush failed");
    }
    pthread_mutex_unlock(&g_emit_mu);
}

static void serve_error(const char *id, const char *fmt, ...) {
    char message[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(message, sizeof(message), fmt, args);
    va_end(args);
    /* Worst case: each byte expands to a six-byte control escape. */
    char message_esc[1024 * 6 + 8];
    serve_json_escape(message, message_esc, sizeof(message_esc));
    char id_esc[176];
    serve_json_escape(id ? id : "", id_esc, sizeof(id_esc));
    serve_emit("{\"type\":\"error\",\"id\":\"%s\",\"message\":\"%s\"}",
               id_esc, message_esc);
}

/* ------------------------------------------------------------ progress */

typedef struct {
    pthread_mutex_t mu;
    char rid[80];
    char stage[16];
    double percent;
    double started;  /* job start (elapsed is measured from here) */
    double stage_t0; /* current stage start (flow extrapolation) */
    double last_emit;
    int active; /* 0 in REPL mode: stage/heartbeat emitters become no-ops */
} serve_progress;

static serve_progress g_prog = {.mu = PTHREAD_MUTEX_INITIALIZER};

/* Total-progress spans per stage, mirroring worker.py exactly. */
static int serve_span(const char *stage, double *lo, double *hi) {
    if (strcmp(stage, "abc") == 0) { *lo = 0.02; *hi = 0.38; return 1; }
    if (strcmp(stage, "semantic") == 0) { *lo = 0.40; *hi = 0.78; return 1; }
    if (strcmp(stage, "flow") == 0) { *lo = 0.80; *hi = 0.94; return 1; }
    if (strcmp(stage, "save") == 0) { *lo = 0.96; *hi = 1.00; return 1; }
    return 0;
}

static void serve_progress_raw(const char *rid, const char *stage,
                               double percent, double elapsed,
                               int heartbeat) {
    char rid_esc[176];
    serve_json_escape(rid, rid_esc, sizeof(rid_esc));
    serve_emit("{\"type\":\"progress\",\"id\":\"%s\",\"stage\":\"%s\","
               "\"percent\":%.4f,\"elapsed\":%.1f%s}",
               rid_esc, stage, percent, elapsed,
               heartbeat ? ",\"heartbeat\":true" : "");
}

/* Stage transition / ratio update, rate-limited like the reference. */
static void serve_stage(const char *stage, double ratio, int force) {
    double lo, hi;
    if (!serve_span(stage, &lo, &hi)) return;
    if (ratio < 0.0) ratio = 0.0;
    if (ratio > 1.0) ratio = 1.0;
    double percent = lo + ratio * (hi - lo);
    double now = now_seconds();
    pthread_mutex_lock(&g_prog.mu);
    if (!g_prog.active) {
        pthread_mutex_unlock(&g_prog.mu);
        return;
    }
    if (!force && now - g_prog.last_emit < 0.25) {
        pthread_mutex_unlock(&g_prog.mu);
        return;
    }
    if (strcmp(g_prog.stage, stage) != 0) {
        snprintf(g_prog.stage, sizeof(g_prog.stage), "%s", stage);
        g_prog.stage_t0 = now;
    }
    g_prog.percent = percent;
    g_prog.last_emit = now;
    double started = g_prog.started;
    char rid[80];
    snprintf(rid, sizeof(rid), "%s", g_prog.rid);
    pthread_mutex_unlock(&g_prog.mu);
    serve_progress_raw(rid, stage, percent, now - started, 0);
}

/* Every 4 s re-send the current progress while a request is active. In
 * the flow stage the percentage extrapolates with wall time (the NAR has
 * no per-step callback), capped at 0.94 exactly like worker.py. */
static void *serve_heartbeat(void *arg) {
    (void)arg;
    for (;;) {
        usleep(4 * 1000 * 1000);
        pthread_mutex_lock(&g_prog.mu);
        int active = g_prog.active;
        char rid[80], stage[16];
        double percent = 0.0, started = 0.0, stage_t0 = 0.0;
        if (active) {
            snprintf(rid, sizeof(rid), "%s", g_prog.rid);
            snprintf(stage, sizeof(stage), "%s", g_prog.stage);
            percent = g_prog.percent;
            started = g_prog.started;
            stage_t0 = g_prog.stage_t0;
        }
        pthread_mutex_unlock(&g_prog.mu);
        if (!active) continue;
        double now = now_seconds();
        if (strcmp(stage, "flow") == 0) {
            double extra = (now - stage_t0) / 150.0 * 0.14;
            if (extra > 0.14) extra = 0.14;
            percent = 0.80 + extra;
        }
        serve_progress_raw(rid, stage, percent, now - started, 1);
    }
    return NULL;
}

/* AR on_token callback: token-count progress inside the abc/semantic
 * spans, throttled to 0.5 s like the reference on_token. */
typedef struct {
    const char *stage;
    size_t max_tokens;
    size_t count;
    double last;
} token_progress;

static void token_progress_cb(yue_phase phase, uint32_t token, void *ctx) {
    (void)phase;
    (void)token;
    token_progress *tp = ctx;
    tp->count++;
    double now = now_seconds();
    if (now - tp->last < 0.5) return;
    tp->last = now;
    double frac = tp->max_tokens
                      ? (double)tp->count / (double)tp->max_tokens
                      : 1.0;
    if (frac > 1.0) frac = 1.0;
    serve_stage(tp->stage, frac, 0);
}

/* ------------------------------------------------- initial acoustic noise */

/* Engine-local RNG (M0 decision): splitmix64 uniform pairs through a
 * Box-Muller transform produce the [frames, 64] fp32 initial state. The
 * stream is request-local and seeded with the request seed. */
static float rng_normal(yue_rng *rng) {
    double u1 = yue_rng_uniform(rng);
    if (u1 <= 0.0) u1 = 1e-300;
    double u2 = yue_rng_uniform(rng);
    return (float)(sqrt(-2.0 * log(u1)) *
                   cos(2.0 * 3.14159265358979323846 * u2));
}

/* -------------------------------------------------------- the pipeline */

/* Encode a user-supplied ABC score into ordinary text ids that
 * yue_token_prefixes() wraps in ABC_START..ABC_END. This mirrors the
 * official SymbolicPlan path (tokenizer.encode(request.abc)) including its
 * "ids stay inside the ordinary text vocabulary" check. The text is encoded
 * verbatim: SongRequest only strips whitespace to test emptiness, it never
 * strips the bytes fed to the tokenizer. */
static int abc_text_blank(const char *s) {
    for (; *s; s++)
        if (*s != ' ' && *s != '\t' && *s != '\r' && *s != '\n' &&
            *s != '\v' && *s != '\f')
            return 0;
    return 1;
}

static int encode_external_abc(const h3_tokenizer *tok, const char *text,
                               uint32_t **out, size_t *count,
                               char *error, size_t error_size) {
    *out = NULL;
    *count = 0;
    if (abc_text_blank(text)) {
        snprintf(error, error_size, "abc score is empty or whitespace-only");
        return 0;
    }
    if (strlen(text) > YUE_ABC_TEXT_MAX) {
        snprintf(error, error_size, "abc text exceeds %u bytes",
                 YUE_ABC_TEXT_MAX);
        return 0;
    }
    uint32_t *ids = NULL;
    size_t n = 0;
    if (!h3_tokenizer_encode(tok, text, 0, &ids, &n, error, error_size))
        return 0;
    if (n == 0 || n > YUE_ABC_TOKENS_MAX) {
        h3_tokenizer_ids_free(ids);
        snprintf(error, error_size,
                 "abc must encode to 1..%u tokens (got %zu)",
                 YUE_ABC_TOKENS_MAX, n);
        return 0;
    }
    for (size_t i = 0; i < n; i++) {
        if (ids[i] >= YUE_EOD) {
            h3_tokenizer_ids_free(ids);
            snprintf(error, error_size,
                     "abc contains a special token id at position %zu", i);
            return 0;
        }
    }
    *out = ids;
    *count = n;
    return 1;
}

/* One full song: plan -> semantic -> acoustic flow -> decode -> FLAC.
 * Returns 1 on success. Progress events fire in serve mode; both modes
 * log the stage timings to stderr like the reference worker. */
static int run_song(yue_engine *eng, const char *style, const char *lyrics,
                    int cot, double guidance, uint64_t seed,
                    const uint32_t *ext_abc, size_t ext_abc_count,
                    const char *output, size_t *out_frames,
                    double *out_seconds, char *error, size_t error_size) {
    yue_generation *abc_gen = NULL;
    yue_generation *sem_gen = NULL;
    uint32_t *sem_prefix = NULL;
    uint32_t *neg_prefix = NULL;
    float *noise = NULL;
    float *latents = NULL;
    float *audio = NULL;
    uint32_t *chunk_tokens = NULL; /* per-chunk scratch, freed inline */
    yue_chunk_range *ranges = NULL;
    char *wav_tmp = NULL;
    /* An external score replaces the planning pass; these ids stay borrowed
     * from the caller (serve_handle / repl_generate owns them). */
    const uint32_t *abc_ids = ext_abc;
    size_t abc_count = ext_abc_count;
    size_t sem_len = 0;
    size_t neg_len = 0;
    size_t frames = 0;
    size_t samples = 0;
    int ok = 0;
    double t0 = now_seconds();
    double t_stage = t0;
    double t_abc = 0.0, t_sem = 0.0, t_flow = 0.0, t_save = 0.0;
    token_progress tp_abc, tp_sem;
    yue_token_fn cb = eng->serve ? token_progress_cb : NULL;
    memset(&tp_abc, 0, sizeof(tp_abc));
    memset(&tp_sem, 0, sizeof(tp_sem));

    if (ext_abc_count && cot == YUE_COT_OFF) {
        snprintf(error, error_size,
                 "an external abc requires cot=full or cot=melody");
        goto cleanup;
    }

    /* ---- stage 1: ABC planning (cot=off or an external score skips it) ---- */
    if (ext_abc_count)
        serve_stage("abc", 1.0, 1); /* no AR pass: close the stage at once */
    else
        serve_stage("abc", 0.0, 1);
    if (cot != YUE_COT_OFF && !ext_abc_count) {
        uint32_t *plan_prefix = NULL;
        size_t plan_len = 0;
        if (!yue_token_prefixes(eng->tokenizer, style, lyrics, cot, NULL, 0,
                                &plan_prefix, &plan_len, error, error_size))
            goto cleanup;
        tp_abc.stage = "abc";
        tp_abc.max_tokens = (size_t)eng->cfg.abc.max_tokens;
        tp_abc.last = now_seconds();
        int planned =
            yue_ar_generate(eng->ar, plan_prefix, plan_len, NULL, 0, 1.0,
                            YUE_PHASE_ABC, &eng->cfg.abc, seed, 0, cb,
                            &tp_abc, &abc_gen, error, error_size);
        free(plan_prefix);
        if (!planned) goto cleanup;
        abc_ids = abc_gen->tokens; /* borrowed; freed with abc_gen */
        abc_count = abc_gen->count;
    }
    t_abc = now_seconds() - t_stage;
    t_stage = now_seconds();

    /* ---- stage 2: semantic codec tokens ---- */
    serve_stage("semantic", 0.0, 1);
    if (!yue_token_prefixes(eng->tokenizer, style, lyrics, cot, abc_ids,
                            abc_count, &sem_prefix, &sem_len, error,
                            error_size))
        goto cleanup;
    if (guidance != 1.0 &&
        !yue_negative_prefix(eng->tokenizer, style, lyrics, cot, abc_ids,
                             abc_count, &neg_prefix, &neg_len, error,
                             error_size))
        goto cleanup;
    tp_sem.stage = "semantic";
    tp_sem.max_tokens = (size_t)eng->cfg.semantic.max_tokens;
    tp_sem.last = now_seconds();
    if (!yue_ar_generate(eng->ar, sem_prefix, sem_len, neg_prefix, neg_len,
                         guidance, YUE_PHASE_SEMANTIC, &eng->cfg.semantic,
                         seed, cot == YUE_COT_OFF, cb, &tp_sem, &sem_gen,
                         error, error_size))
        goto cleanup;
    frames = sem_gen->count;
    if (frames == 0) {
        snprintf(error, error_size,
                 "semantic generation produced no codec frames");
        goto cleanup;
    }
    for (size_t i = 0; i < frames; i++) {
        if (sem_gen->tokens[i] < YUE_CODEC_OFFSET ||
            sem_gen->tokens[i] - YUE_CODEC_OFFSET >= YUE_CODEC_SIZE) {
            snprintf(error, error_size,
                     "semantic token %zu is outside the codec vocabulary",
                     i);
            goto cleanup;
        }
    }
    t_sem = now_seconds() - t_stage;
    t_stage = now_seconds();

    /* ---- stage 3: acoustic flow matching ---- */
    serve_stage("flow", 0.0, 1);
    if (!eng->nar) {
        eng->nar = yue_nar_create(eng->gpu, eng->ar, eng->model_dir, error,
                                  error_size);
        if (!eng->nar) goto cleanup;
        log_stderr("acoustic model resident (%.1fs into the request)",
                   now_seconds() - t0);
    }
    noise = malloc(sizeof(float) * frames * YUE_LATENT_DIM);
    latents = malloc(sizeof(float) * frames * YUE_LATENT_DIM);
    ranges = malloc(sizeof(yue_chunk_range) * frames);
    if (!noise || !latents || !ranges) {
        snprintf(error, error_size, "out of memory");
        goto cleanup;
    }
    {
        yue_rng rng;
        yue_rng_init(&rng, seed);
        for (size_t i = 0; i < frames * YUE_LATENT_DIM; i++)
            noise[i] = rng_normal(&rng);
    }
    {
        size_t n_ranges = 0;
        if (!yue_chunk_ranges(frames, sem_len, (size_t)eng->cfg.context,
                              ranges, frames, &n_ranges, error, error_size))
            goto cleanup;
        for (size_t r = 0; r < n_ranges; r++) {
            const size_t start = ranges[r].start;
            const size_t end = ranges[r].end;
            const size_t chunk_frames = end - start;
            const size_t tokens = sem_len + chunk_frames + 1;
            chunk_tokens = malloc(sizeof(uint32_t) * tokens);
            if (!chunk_tokens) {
                snprintf(error, error_size, "out of memory");
                goto cleanup;
            }
            memcpy(chunk_tokens, sem_prefix, sem_len * sizeof(uint32_t));
            for (size_t j = 0; j < chunk_frames; j++)
                /* native codec id == CODEC_OFFSET + local (validated) */
                chunk_tokens[sem_len + j] = sem_gen->tokens[start + j];
            chunk_tokens[sem_len + chunk_frames] = YUE_MUSIC_END;
            yue_nar_chunk *chunk =
                yue_nar_chunk_create(eng->nar, chunk_tokens, tokens,
                                     chunk_frames, 0, error, error_size);
            free(chunk_tokens);
            chunk_tokens = NULL; /* cleanup must not double-free */
            if (!chunk) goto cleanup; /* keep chunk_create's error message */
            int solved =
                yue_nar_solve(chunk, noise + start * YUE_LATENT_DIM,
                              eng->cfg.ode_steps,
                              latents + start * YUE_LATENT_DIM, error,
                              error_size);
            yue_nar_chunk_free(chunk);
            if (!solved) goto cleanup;
        }
    }
    t_flow = now_seconds() - t_stage;
    t_stage = now_seconds();

    /* ---- stage 4: VAE decode + FLAC ---- */
    serve_stage("save", 0.0, 1);
    if (!eng->vae) {
        eng->vae = yue_vae_create(eng->vae_dir, eng->gpu, error, error_size);
        if (!eng->vae) goto cleanup;
        log_stderr("audio decoder resident (%.1fs into the request)",
                   now_seconds() - t0);
    }
    samples = yue_vae_output_length(eng->vae, frames);
    audio = malloc(sizeof(float) * samples * 2);
    if (!audio) {
        snprintf(error, error_size, "out of memory");
        goto cleanup;
    }
    if (!yue_vae_decode(eng->vae, latents, frames, audio, error, error_size))
        goto cleanup;
    if (!ensure_parent_dir(output)) {
        snprintf(error, error_size, "cannot create the output directory "
                                     "for %s",
                 output);
        goto cleanup;
    }
    wav_tmp = malloc(strlen(output) + 16);
    if (!wav_tmp) {
        snprintf(error, error_size, "out of memory");
        goto cleanup;
    }
    snprintf(wav_tmp, strlen(output) + 16, "%s.tmp.wav", output);
    if (!yue_write_wav24(wav_tmp, audio, samples, 2, YUE_SAMPLE_RATE, error,
                         error_size)) {
        unlink(wav_tmp); /* drop the half-written temp file */
        goto cleanup;
    }
    if (!yue_wav_to_flac(wav_tmp, output, error, error_size)) {
        unlink(wav_tmp);
        goto cleanup;
    }
    unlink(wav_tmp);
    t_save = now_seconds() - t_stage;

    *out_frames = frames;
    *out_seconds = (double)samples / (double)YUE_SAMPLE_RATE;
    log_stderr("timings: abc %.1fs, semantic %.1fs, flow %.1fs, "
               "save %.1fs, e2e %.1fs",
               t_abc, t_sem, t_flow, t_save, now_seconds() - t0);
    ok = 1;

cleanup:
    free(chunk_tokens);
    free(wav_tmp);
    free(audio);
    free(ranges);
    free(latents);
    free(noise);
    free(neg_prefix);
    free(sem_prefix);
    yue_generation_free(sem_gen);
    yue_generation_free(abc_gen);
    return ok;
}

/* --------------------------------------------------------------- serve */

static char *dup_trim(const char *src) {
    if (!src) src = "";
    size_t len = strlen(src);
    char *copy = malloc(len + 1);
    if (!copy) return NULL;
    memcpy(copy, src, len + 1);
    return trim(copy);
}

static void serve_handle(yue_engine *eng, const yw_json *req) {
    char error[YUE_ERROR_SIZE] = {0};
    const char *rid = yw_json_get_string(req, "id", "");

    char *style = dup_trim(yw_json_get_string(req, "style", ""));
    char *lyrics = dup_trim(yw_json_get_string(req, "lyrics", ""));
    char *cot_s = dup_trim(yw_json_get_string(req, "cot", "full"));
    char *output = dup_trim(yw_json_get_string(req, "output", ""));
    char *abc_text = NULL;
    uint32_t *abc_ids = NULL;
    size_t abc_count = 0;
    /* "abc" is optional; JSON null means "no score" (SongRequest.abc=None),
     * other non-string values are a protocol error. The text is kept verbatim
     * -- trimming it would change the BPE ids compared with the official
     * tokenizer.encode(request.abc). */
    const yw_json *abc_j = yw_json_get(req, "abc");
    int has_abc = abc_j && yw_json_type(abc_j) != yue_json_NULL;
    int failed = !style || !lyrics || !cot_s || !output;
    if (failed) {
        serve_error(rid, "out of memory");
        goto done;
    }
    if (has_abc && yw_json_type(abc_j) != yue_json_STRING) {
        serve_error(rid, "abc must be a string or null");
        goto done;
    }
    if (has_abc) {
        abc_text = strdup(yw_json_get_string(req, "abc", ""));
        if (!abc_text) {
            serve_error(rid, "out of memory");
            goto done;
        }
    }
    if (!*style) {
        serve_error(rid, "style cannot be empty");
        goto done;
    }
    if (!*lyrics) {
        serve_error(rid,
                    "lyrics cannot be empty (use [instrumental] for "
                    "instrumental)");
        goto done;
    }
    if (!*cot_s) {
        free(cot_s);
        cot_s = strdup("full"); /* worker.py: "" or "full" */
        if (!cot_s) {
            serve_error(rid, "out of memory");
            goto done;
        }
    }
    int cot = yue_cot_index(cot_s);
    if (cot < 0) {
        serve_error(rid, "cot must be full/melody/off, received %s", cot_s);
        goto done;
    }
    /* cot=off has no ABC section in the prefix, so the official protocol
     * rejects an external score there (SongRequest.__post_init__). */
    if (has_abc && cot == YUE_COT_OFF) {
        serve_error(rid, "abc requires cot=full or cot=melody "
                         "(cot=off has no ABC section)");
        goto done;
    }
    if (!*output) {
        serve_error(rid, "output path missing");
        goto done;
    }
    if (strlen(output) > YUE_OUTPUT_MAX) {
        serve_error(rid, "output path too long");
        goto done;
    }
    for (const unsigned char *p = (const unsigned char *)output; *p; p++) {
        if (*p < 0x20) {
            serve_error(rid, "output path contains control characters");
            goto done;
        }
    }
    /* worker.py contract (documented in README): output is the FLAC file
     * itself; ffmpeg is invoked with "-c:a flac", so a different container
     * name would only produce a confusing encoder failure later. */
    size_t output_len = strlen(output);
    if (output_len < 5 || strcmp(output + output_len - 5, ".flac") != 0) {
        serve_error(rid, "output path must end with .flac");
        goto done;
    }

    {
        int has_cfg = 0;
        double cfg_value = 0.0;
        const yw_json *cfg_j = yw_json_get(req, "cfgScale");
        if (cfg_j && yw_json_type(cfg_j) == yue_json_NUMBER) {
            double v = yw_json_number(cfg_j);
            if (v > 0.0) { /* <= 0 means "unspecified" like worker.py */
                if (v > 20.0) {
                    serve_error(rid, "cfgScale must be in [0, 20]");
                    goto done;
                }
                has_cfg = 1;
                cfg_value = v;
            }
        }
        uint64_t seed = 0;
        const yw_json *seed_j = yw_json_get(req, "seed");
        if (seed_j && yw_json_type(seed_j) == yue_json_NUMBER) {
            double v = yw_json_number(seed_j);
            if (!(v >= 0.0) || v >= 9223372036854775808.0) {
                serve_error(rid, "seed must be in [0, 2^63)");
                goto done;
            }
            if (v > 9007199254740992.0) {
                /* doubles cannot represent integers past 2^53 exactly;
                 * a silently rounded seed would break reproducibility. */
                serve_error(rid, "seed must be at most 2^53 (integer "
                                 "precision limit)");
                goto done;
            }
            seed = (uint64_t)v;
        }
        double guidance = yue_request_guidance(cot, cfg_value, has_cfg);

        /* An external score is encoded here (before any GPU work) and its
         * ids are used for both the positive and the symbolic CFG negative
         * prefix; the ABC planning AR pass is then skipped in run_song. */
        if (has_abc &&
            !encode_external_abc(eng->tokenizer, abc_text, &abc_ids,
                                 &abc_count, error, sizeof(error))) {
            serve_error(rid, "%s", error);
            goto done;
        }

        double t0 = now_seconds();
        log_stderr("job %s: cot=%s cfg=%s seed=%llu lyrics=%zu ch abc=%s",
                   rid, cot_s, has_cfg ? "explicit" : "auto",
                   (unsigned long long)seed, strlen(lyrics),
                   abc_count ? "external"
                             : (cot == YUE_COT_OFF ? "none" : "planned"));
        if (abc_count)
            log_stderr("job %s: external abc encoded to %zu tokens", rid,
                       abc_count);

        pthread_mutex_lock(&g_prog.mu);
        snprintf(g_prog.rid, sizeof(g_prog.rid), "%s", rid);
        g_prog.started = t0;
        g_prog.stage_t0 = t0;
        g_prog.last_emit = 0.0;
        snprintf(g_prog.stage, sizeof(g_prog.stage), "abc");
        g_prog.percent = 0.0;
        g_prog.active = 1;
        pthread_mutex_unlock(&g_prog.mu);

        size_t frames = 0;
        double seconds = 0.0;
        int ok = run_song(eng, style, lyrics, cot, guidance, seed,
                          abc_ids, abc_count, output,
                          &frames, &seconds, error, sizeof(error));

        pthread_mutex_lock(&g_prog.mu);
        g_prog.active = 0; /* heartbeat stops before done/error lands */
        pthread_mutex_unlock(&g_prog.mu);

        if (ok) {
            /* Worst case: each path byte expands to a six-byte escape. */
            char rid_esc[176], path_esc[YUE_OUTPUT_MAX * 6 + 8];
            serve_json_escape(rid, rid_esc, sizeof(rid_esc));
            serve_json_escape(output, path_esc, sizeof(path_esc));
            serve_emit("{\"type\":\"done\",\"id\":\"%s\",\"path\":\"%s\","
                       "\"seconds\":%.1f,\"tokens\":%zu,"
                       "\"sampleRate\":48000}",
                       rid_esc, path_esc, seconds, frames);
            log_stderr("job %s: done %.1fs audio, %zu semantic tokens, "
                       "e2e %.1fs",
                       rid, seconds, frames, now_seconds() - t0);
        } else {
            serve_error(rid, "%s", error);
            log_stderr("job %s: failed: %s", rid, error);
        }
        request_boundary(eng);
    }

done:
    h3_tokenizer_ids_free(abc_ids);
    free(abc_text);
    free(output);
    free(cot_s);
    free(lyrics);
    free(style);
}

/* Reads one request line (arbitrary length up to SERVE_LINE_MAX).
 * Returns 1 with *out malloc'd (no trailing newline); 0 on EOF; -1 on an
 * allocation failure. An over-long line sets *overflow, one containing a
 * raw NUL sets *malformed; both drain the rest of the physical line so the
 * stream stays aligned and leave *out NULL. A final line without a
 * trailing newline is accepted (EOF closes it). */
static int serve_read_line(char **out, int *overflow, int *malformed) {
    *out = NULL;
    *overflow = 0;
    *malformed = 0;
    size_t cap = 0;
    char *line = NULL;
    for (;;) {
        char chunk[4096];
        memset(chunk, 0, sizeof(chunk));
        if (!fgets(chunk, sizeof(chunk), stdin)) {
            if (cap) { /* final request without a trailing newline */
                *out = line;
                return 1;
            }
            free(line);
            return 0;
        }
        /* fgets() stops at the newline (or after 4095 bytes / EOF) and
         * terminates with NUL. Measure the physical bytes with the newline
         * position, not strlen(): a raw NUL inside the line would hide the
         * newline and make us splice the NEXT request onto this one. */
        char *nl = memchr(chunk, '\n', sizeof(chunk));
        size_t len = nl ? (size_t)(nl - chunk) + 1 : strlen(chunk);
        if (!nl) {
            /* With the chunk zeroed, any non-zero byte past the first NUL
             * was written by fgets() and reveals an embedded NUL. */
            for (size_t i = sizeof(chunk) - 1; i > len; i--) {
                if (chunk[i]) { len = i + 1; break; }
            }
        }
        int complete = nl != NULL;
        if (memchr(chunk, '\0', len)) {
            *malformed = 1;
            while (!complete) {
                int c = fgetc(stdin);
                if (c == EOF || c == '\n') break;
            }
            free(line);
            return 1;
        }
        if (cap + len > SERVE_LINE_MAX) {
            *overflow = 1;
            while (!complete) {
                int c = fgetc(stdin);
                if (c == EOF || c == '\n') break;
            }
            free(line);
            return 1;
        }
        size_t want = cap + len + 1;
        char *grown = realloc(line, want);
        if (!grown) {
            free(line);
            /* OOM is not EOF: drain the remainder of the line and report
             * -1 so the worker keeps serving instead of exiting silently. */
            while (!complete) {
                int c = fgetc(stdin);
                if (c == EOF || c == '\n') break;
            }
            return -1;
        }
        line = grown;
        memcpy(line + cap, chunk, len);
        cap += len;
        line[cap] = '\0';
        if (complete) {
            if (cap && line[cap - 1] == '\n') line[--cap] = '\0';
            if (cap && line[cap - 1] == '\r') line[--cap] = '\0';
            *out = line;
            return 1;
        }
    }
}

static int cmd_serve(const char *model_dir, const char *vae_dir) {
    int lock = acquire_gpu_lock();
    if (lock < 0) {
        log_stderr("cannot create the GPU lock file "
                   "(/tmp/yue-metal-engine-<uid>.lock)");
        return 1;
    }
    if (lock == 0) {
        log_stderr("another engine instance holds the GPU lock "
                   "(/tmp/yue-metal-engine-<uid>.lock)");
        return 1;
    }

    char pid[32];
    snprintf(pid, sizeof(pid), "%d", (int)getpid());
    serve_emit("{\"type\":\"state\",\"state\":\"loading\",\"pid\":%s}", pid);

    yue_engine eng;
    memset(&eng, 0, sizeof(eng));
    snprintf(eng.model_dir, sizeof(eng.model_dir), "%s", model_dir);
    snprintf(eng.vae_dir, sizeof(eng.vae_dir), "%s", vae_dir);
    eng.serve = 1;

    char error[YUE_ERROR_SIZE] = {0};
    double t0 = now_seconds();
    if (!engine_load(&eng, error, sizeof(error))) {
        log_stderr("load failed: %s", error);
        char esc[2048];
        serve_json_escape(error, esc, sizeof(esc));
        serve_emit("{\"type\":\"error\",\"id\":\"\",\"message\":\"init: %s\"}",
                   esc);
        return 1;
    }
    pthread_t beat;
    if (pthread_create(&beat, NULL, serve_heartbeat, NULL) != 0) {
        log_stderr("cannot start the heartbeat thread");
        serve_error("", "init: heartbeat");
        return 1;
    }
    /* Do not advertise "ready" before the heartbeat exists: on creation
     * failure the host would otherwise queue a request into a dead pipe. */
    serve_emit("{\"type\":\"state\",\"state\":\"ready\",\"loadSeconds\":%.1f}",
               now_seconds() - t0);
    log_stderr("serve ready (%.1fs)", now_seconds() - t0);

    /* Soft self-restart (YUE_RESTART_AFTER, default 10 requests, 0 =
     * disabled). Metal is the one allocator whose freed pages are not
     * guaranteed to return: after many requests the driver's retained
     * resources ratchet the resident footprint up (~1 GB per song,
     * measured). execve() swaps the whole process image on the same
     * stdin/stdout fds, so the host sees one continuous worker that
     * periodically returns to the load-time footprint; weights re-pager in
     * lazily (<1 s). The lock fd is O_CLOEXEC, so the new image re-acquires
     * it cleanly. */
    unsigned long long served = 0;
    unsigned long long restart_after = 10;
    {
        const char *env = getenv("YUE_RESTART_AFTER");
        if (env && *env) {
            char *end = NULL;
            errno = 0;
            long value = strtol(env, &end, 10);
            if (errno == 0 && end != env && *end == '\0' && value >= 0)
                restart_after = (unsigned long long)value;
            else
                log_stderr("ignoring invalid YUE_RESTART_AFTER=%s", env);
        }
    }

    /* Unbuffered stdin: the self-restart below execve()s this process on
     * the same fd, and stdio would otherwise discard a request line it had
     * pre-read into its user-space buffer (kernel pipe data survives exec,
     * FILE buffers do not). fgets/fgetc then fall back to 1-byte reads,
     * which is negligible next to multi-second GPU requests. */
    setvbuf(stdin, NULL, _IONBF, 0);

    for (;;) {
        char *line;
        int overflow = 0, malformed = 0;
        int got = serve_read_line(&line, &overflow, &malformed);
        if (got < 0) {
            log_stderr("out of memory reading the request line");
            serve_error("", "out of memory");
            continue;
        }
        if (!got) break;
        if (overflow) {
            log_stderr("bad request line: exceeds %u bytes", SERVE_LINE_MAX);
            serve_error("", "request line too long");
            continue;
        }
        if (malformed) {
            log_stderr("bad request line: raw NUL byte");
            serve_error("", "malformed request line");
            continue;
        }
        const char *trimmed = line;
        while (*trimmed == ' ' || *trimmed == '\t') trimmed++;
        if (*trimmed) {
            yw_json *req = NULL;
            char perr[YUE_ERROR_SIZE];
            if (yw_json_parse(trimmed, &req, perr, sizeof(perr)) == 0) {
                log_stderr("bad request line: %s", perr);
                serve_error("", "malformed request line");
            } else if (!strcmp(yw_json_get_string(req, "cmd", ""), "ping")) {
                serve_emit("{\"type\":\"state\",\"state\":\"ready\"}");
            } else {
                serve_handle(&eng, req);
                served++;
            }
            yw_json_free(req);
        }
        free(line);
        if (restart_after && served >= restart_after) {
            char exe[4096];
            uint32_t size = sizeof(exe);
            log_stderr("served %llu requests: exec-restarting to release "
                       "driver-held Metal pages",
                       served);
            if (_NSGetExecutablePath(exe, &size) == 0) {
                /* Rebuild the restart argv from the RESOLVED absolute
                 * paths instead of replaying the original argv:
                 * ensure_shader_dir() may have chdir'd to the executable
                 * directory, so a relative -d/--vae (or a relative
                 * argv[0]) would resolve against the new cwd and the
                 * restarted image would fail to find the model. The
                 * serve mode only takes -d/--vae, so this reconstruction
                 * is lossless. */
                char *restart_argv[] = {(char *)exe,  (char *)"serve",
                                        (char *)"-d", (char *)model_dir,
                                        (char *)"--vae", (char *)vae_dir,
                                        NULL};
                /* Quiesce the heartbeat under the emitter's lock: execve()
                 * destroys that thread instantly, so no emit may be in
                 * flight and none may start before the new image runs. */
                pthread_mutex_lock(&g_emit_mu);
                g_restarting = 1;
                fflush(NULL);
                pthread_mutex_unlock(&g_emit_mu);
                execv(exe, restart_argv);
                int saved_errno = errno;
                pthread_mutex_lock(&g_emit_mu);
                g_restarting = 0;
                pthread_mutex_unlock(&g_emit_mu);
                log_stderr("execv failed: %s (continuing)",
                           strerror(saved_errno));
            }
            served = 0; /* keep serving if the restart failed */
        }
    }
    log_stderr("stdin closed, exiting");
    return 0;
}

/* ----------------------------------------------------------------- repl */

typedef struct {
    char *style;
    char *lyrics;
    char *abc; /* optional user-supplied ABC score; NULL = plan it */
    int cot;
    double cfg_scale; /* 0 = auto (1.01 when cot=off, else 1.0) */
    uint64_t seed;
    char out_prefix[YUE_OUTPUT_MAX];
    unsigned long long out_index;
} repl_state;

static const char *arg_after(const char *line, const char *cmd) {
    size_t n = strlen(cmd);
    if (strncmp(line, cmd, n) != 0) return NULL;
    const char *rest = line + n;
    if (*rest != '\0' && *rest != ' ' && *rest != '\t') return NULL;
    while (*rest == ' ' || *rest == '\t') rest++;
    return rest;
}

/* Interprets the C escapes that matter for lyrics input. */
static void unescape_inplace(char *s) {
    if (!s) return;
    char *w = s;
    for (char *r = s; *r; r++) {
        if (*r == '\\' && r[1]) {
            r++;
            if (*r == 'n') {
                *w++ = '\n';
            } else if (*r == 't') {
                *w++ = '\t';
            } else if (*r == '\\') {
                *w++ = '\\';
            } else {
                *w++ = '\\';
                *w++ = *r;
            }
        } else {
            *w++ = *r;
        }
    }
    *w = '\0';
}

/* ABC notation has no backslash escapes of its own. Only the explicit \n
 * separator for one-line /abc input is expanded; every other byte (including
 * backslashes) stays verbatim, matching /abc-file and the serve "abc" field. */
static void unescape_abc_newlines(char *s) {
    char *w = s;
    for (char *r = s; *r; r++) {
        if (r[0] == '\\' && r[1] == 'n') {
            *w++ = '\n';
            r++;
        } else {
            *w++ = *r;
        }
    }
    *w = '\0';
}

static int set_field(char **field, const char *value) {
    char *copy = strdup(value);
    if (!copy) {
        printf("out of memory\n");
        return 0;
    }
    free(*field);
    *field = copy;
    return 1;
}

static void repl_show(const repl_state *st) {
    double guidance =
        yue_request_guidance(st->cot, st->cfg_scale, st->cfg_scale > 0.0);
    printf("cot: %s\ncfg: %s (effective guidance %.3f)\n"
           "seed: %llu (auto-increments after each /gen)\n"
           "out: %s-<n>.flac\n",
           cot_name(st->cot),
           st->cfg_scale > 0.0 ? "explicit" : "auto", guidance,
           (unsigned long long)st->seed, st->out_prefix);
    printf("style: %s\n", st->style ? st->style : "");
    const char *lyrics = st->lyrics ? st->lyrics : "";
    size_t len = strlen(lyrics);
    printf("lyrics (%zu bytes): ", len);
    for (size_t i = 0; i < len && i < 400; i++)
        putchar(lyrics[i] == '\n' ? ' ' : lyrics[i]);
    if (len > 400) printf(" ...");
    putchar('\n');
    printf("abc: %s\n", st->abc && *st->abc
                            ? "user-supplied score (planning skipped)"
                            : "none (model plans its own)");
}

static void repl_generate(yue_engine *eng, repl_state *st) {
    if (!st->style || !*st->style) {
        printf("style is empty: set it with /style TEXT\n");
        return;
    }
    if (!st->lyrics || !*st->lyrics) {
        printf("lyrics are empty: set them with /lyrics TEXT or "
               "/lyrics-file PATH\n");
        return;
    }
    double guidance =
        yue_request_guidance(st->cot, st->cfg_scale, st->cfg_scale > 0.0);
    char out[YUE_OUTPUT_MAX + 32];
    snprintf(out, sizeof(out), "%s-%llu.flac", st->out_prefix, st->out_index);
    log_stderr("generating: cot=%s cfg=%s seed=%llu -> %s", cot_name(st->cot),
               st->cfg_scale > 0.0 ? "explicit" : "auto",
               (unsigned long long)st->seed, out);
    char error[YUE_ERROR_SIZE] = {0};
    size_t frames = 0;
    double seconds = 0.0;
    uint32_t *abc_ids = NULL;
    size_t abc_count = 0;
    if (st->abc && *st->abc) {
        if (st->cot == YUE_COT_OFF) {
            printf("abc requires cot=full or cot=melody (use /cot full)\n");
            return;
        }
        if (!encode_external_abc(eng->tokenizer, st->abc, &abc_ids,
                                 &abc_count, error, sizeof(error))) {
            printf("error: %s\n", error);
            return;
        }
    }
    int done = run_song(eng, st->style, st->lyrics, st->cot, guidance,
                        st->seed, abc_ids, abc_count, out, &frames,
                        &seconds, error, sizeof(error));
    h3_tokenizer_ids_free(abc_ids);
    /* Same request boundary as serve: each song has its own prefix length,
     * and without this every new length pins a ~95 MB MPSGraph compile for
     * the whole REPL session. */
    request_boundary(eng);
    if (!done) {
        printf("error: %s\n", error);
        return;
    }
    printf("wrote %s  (%.1fs audio, %zu semantic tokens, seed %llu)\n", out,
           seconds, frames, (unsigned long long)st->seed);
    st->out_index++;
    st->seed++;
}

static void print_help(void) {
    printf("/style TEXT        style tags (required)\n"
           "/lyrics TEXT       lyrics; \\n escapes a newline (required)\n"
           "/lyrics-file PATH  load lyrics from a UTF-8 file\n"
           "/abc TEXT          ABC score; \\n makes a newline; bare /abc clears\n"
           "/abc-file PATH     load an ABC score verbatim from a UTF-8 file\n"
           "                   external scores need cot=full/melody (not off)\n"
           "/cot full|melody|off  decoding mode (default full)\n"
           "/cfg X             explicit CFG scale; 0 = auto\n"
           "/seed N            base seed (default 831001, lyra default)\n"
           "/out PREFIX        output path prefix (default out)\n"
           "/show              show current settings\n"
           "/gen               generate one song\n"
           "/help              this text\n"
           "/quit              exit\n");
}

static void repl_command(yue_engine *eng, repl_state *st, const char *line) {
    const char *arg;
    if (!strcmp(line, "/help")) {
        print_help();
        return;
    }
    if (!strcmp(line, "/show")) {
        repl_show(st);
        return;
    }
    if (!strcmp(line, "/gen")) {
        repl_generate(eng, st);
        return;
    }
    if ((arg = arg_after(line, "/abc-file")) != NULL) {
        if (!*arg) {
            printf("usage: /abc-file PATH\n");
            return;
        }
        char *text = read_text_file(arg);
        if (!text) {
            printf("cannot read %s\n", arg);
            return;
        }
        free(st->abc);
        st->abc = text; /* verbatim, like the official abc_path read */
        printf("abc: %zu bytes (cot=%s)\n", strlen(st->abc),
               cot_name(st->cot));
        return;
    }
    if ((arg = arg_after(line, "/abc")) != NULL) {
        if (!*arg) { /* bare /abc clears the score */
            free(st->abc);
            st->abc = NULL;
            printf("abc: cleared (model will plan its own score)\n");
            return;
        }
        if (!set_field(&st->abc, arg)) return;
        unescape_abc_newlines(st->abc);
        return;
    }
    if ((arg = arg_after(line, "/style")) != NULL) {
        if (!*arg) {
            printf("usage: /style TEXT\n");
            return;
        }
        if (!set_field(&st->style, arg)) return;
        unescape_inplace(st->style);
        return;
    }
    if ((arg = arg_after(line, "/lyrics-file")) != NULL) {
        if (!*arg) {
            printf("usage: /lyrics-file PATH\n");
            return;
        }
        char *text = read_text_file(arg);
        if (!text) {
            printf("cannot read %s\n", arg);
            return;
        }
        free(st->lyrics);
        st->lyrics = trim(text);
        printf("lyrics: %zu bytes\n", strlen(st->lyrics));
        return;
    }
    if ((arg = arg_after(line, "/lyrics")) != NULL) {
        if (!*arg) {
            printf("usage: /lyrics TEXT\n");
            return;
        }
        if (!set_field(&st->lyrics, arg)) return;
        unescape_inplace(st->lyrics);
        return;
    }
    if ((arg = arg_after(line, "/cot")) != NULL) {
        char buf[16];
        snprintf(buf, sizeof(buf), "%s", arg);
        int cot = yue_cot_index(trim(buf));
        if (cot < 0) {
            printf("cot must be full/melody/off\n");
            return;
        }
        st->cot = cot;
        return;
    }
    if ((arg = arg_after(line, "/cfg")) != NULL) {
        if (!*arg) {
            printf("usage: /cfg X (0 = auto)\n");
            return;
        }
        char *end = NULL;
        double v = strtod(arg, &end);
        if (end == arg || !blanks_only(end) || !(v >= 0.0 && v <= 20.0)) {
            printf("cfg must be a number in [0, 20] (0 = auto)\n");
            return;
        }
        st->cfg_scale = v;
        return;
    }
    if ((arg = arg_after(line, "/seed")) != NULL) {
        if (!*arg) {
            printf("usage: /seed N\n");
            return;
        }
        char *end = NULL;
        unsigned long long v = strtoull(arg, &end, 10);
        if (arg[0] == '-' || end == arg || !blanks_only(end) ||
            v > 9223372036854775807ULL) {
            printf("seed must be an integer in [0, 2^63)\n");
            return;
        }
        st->seed = (uint64_t)v;
        return;
    }
    if ((arg = arg_after(line, "/out")) != NULL) {
        if (!*arg) {
            printf("usage: /out PREFIX\n");
            return;
        }
        if (strlen(arg) >= sizeof(st->out_prefix)) {
            printf("output prefix too long\n");
            return;
        }
        strcpy(st->out_prefix, arg);
        trim(st->out_prefix);
        st->out_index = 0;
        return;
    }
    printf("unknown command: %s (try /help)\n", line);
}

static int cmd_repl(const char *model_dir, const char *vae_dir) {
    int lock = acquire_gpu_lock();
    if (lock < 0) {
        log_stderr("cannot create the GPU lock file "
                   "(/tmp/yue-metal-engine-<uid>.lock)");
        return 1;
    }
    if (lock == 0) {
        log_stderr("another engine instance holds the GPU lock "
                   "(/tmp/yue-metal-engine-<uid>.lock)");
        return 1;
    }

    yue_engine eng;
    memset(&eng, 0, sizeof(eng));
    snprintf(eng.model_dir, sizeof(eng.model_dir), "%s", model_dir);
    snprintf(eng.vae_dir, sizeof(eng.vae_dir), "%s", vae_dir);

    char error[YUE_ERROR_SIZE] = {0};
    double t0 = now_seconds();
    if (!engine_load(&eng, error, sizeof(error))) {
        log_stderr("load failed: %s", error);
        return 1;
    }
    log_stderr("repl ready in %.1fs (model=%s vae=%s)",
               now_seconds() - t0, model_dir, vae_dir);
    printf("yue repl: /help for commands\n");

    repl_state st = {NULL, NULL, NULL, YUE_COT_FULL, 0.0, 831001, "out", 0};
    set_field(&st.style, "");
    set_field(&st.lyrics, "");

    char *line;
    while ((line = linenoise("yue> ")) != NULL) {
        if (line[0]) {
            linenoiseHistoryAdd(line);
            if (!strcmp(line, "/quit")) {
                free(line);
                break;
            }
            repl_command(&eng, &st, line);
        }
        free(line);
    }
    free(st.style);
    free(st.lyrics);
    free(st.abc);
    /* The lock fd and the engine are released by process exit (see
     * engine_free): shutdown must not unmap ~20 GB of lazily-paged
     * weights page by page. */
    return 0;
}

/* ----------------------------------------------------------------- main */

static int copy_dir(char *out, size_t size, const char *value) {
    if (!value || !*value) {
        fprintf(stderr, "yue: -d model directory is required\n");
        return -1;
    }
    int written = snprintf(out, size, "%s", value);
    if (written < 0 || (size_t)written >= size) {
        fprintf(stderr, "yue: model directory path is too long\n");
        return -1;
    }
    return 0;
}

/* ensure_shader_dir() can chdir() to the executable directory when the
 * binary is launched from elsewhere; every argv-derived path must be made
 * absolute before engine_load() resolves tokenizer/weights/output against
 * it. Also gives --vae the same length check as -d. */
static int absolutize_dir(char *out, size_t size, const char *value) {
    if (!value || !*value) return copy_dir(out, size, value);
    if (value[0] == '/') return copy_dir(out, size, value);
    char cwd[YUE_DIR_MAX];
    if (!getcwd(cwd, sizeof(cwd))) {
        fprintf(stderr, "yue: cannot resolve the working directory\n");
        return -1;
    }
    int written = snprintf(out, size, "%s/%s", cwd, value);
    if (written < 0 || (size_t)written >= size) {
        fprintf(stderr, "yue: model directory path is too long\n");
        return -1;
    }
    return 0;
}

/* M14: default the decode GEMV path to the MLX-style 2D block kernel.
 * The per-column GEMV and the TensorOps GEMV both lose the execution
 * race, but qw_gemv2d_m64u8 matches MPS execution while saving ~3
 * ms/token of MPSGraph encode (serve A/B, same-window: wall -0.63
 * ms/token, seven suites green, melody_abc bit-exact kept). setenv with
 * overwrite=0: an explicit env still wins, and the shared vendored h3
 * keeps its own defaults for qwen. */
static void default_gemv_env(void) {
    setenv("H3_GEMV", "1", 0);
    setenv("H3_GEMV_KERNEL", "qw_gemv2d_m64u8", 0);
}

int main(int argc, char **argv) {
    default_gemv_env();
    if (argc < 2) {
        fprintf(stderr,
                "usage: yue serve -d MODEL_DIR [--vae VAE_DIR]\n"
                "       yue repl  -d MODEL_DIR [--vae VAE_DIR]\n");
        return 2;
    }
    const char *mode = argv[1];
    const char *model_dir = NULL;
    const char *vae_dir = NULL;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-d") && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (!strcmp(argv[i], "--vae") && i + 1 < argc) {
            vae_dir = argv[++i];
        } else {
            fprintf(stderr, "yue: unknown argument %s\n", argv[i]);
            return 2;
        }
    }

    char resolved_model[YUE_DIR_MAX];
    char resolved_vae[YUE_DIR_MAX];
    memset(resolved_vae, 0, sizeof(resolved_vae));
    if (absolutize_dir(resolved_model, sizeof(resolved_model),
                       model_dir) != 0)
        return 2;

    /* Default VAE: sibling <model>-vae directory (lyra layout). */
    if (!vae_dir || !*vae_dir) {
        size_t len = strlen(resolved_model);
        int written = snprintf(resolved_vae, sizeof(resolved_vae), "%.*s-vae",
                               (int)len, resolved_model);
        if (written < 0 || (size_t)written >= sizeof(resolved_vae)) return 2;
        vae_dir = resolved_vae;
    } else if (absolutize_dir(resolved_vae, sizeof(resolved_vae),
                              vae_dir) != 0) {
        return 2;
    } else {
        vae_dir = resolved_vae;
    }

    if (!strcmp(mode, "serve")) return cmd_serve(resolved_model, vae_dir);
    if (!strcmp(mode, "repl")) return cmd_repl(resolved_model, vae_dir);
    fprintf(stderr, "yue: unknown mode %s\n", mode);
    return 2;
}
