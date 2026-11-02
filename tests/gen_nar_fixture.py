#!/usr/bin/env python3
"""NAR fixtures from the lyra reference stack.

Run: LYRA_FORCE_BF16_SDPA=1 /Volumes/Work/mlx-Yue/.venv/bin/python tests/gen_nar_fixture.py

Dumps conditioning K/V, velocity evaluations (explicit noise, several raw
times), a hand-rolled midpoint trajectory (per-step states), and a
multi-chunk run with a tiny context window so chunk_ranges produces
multiple chunks. The C engine replays everything against the same inputs.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("LYRA_FORCE_BF16_SDPA", "1")

import mlx.core as mx
import numpy as np

from lyra.ar import load_ar
from lyra.nar import Chunk, CachedNAR, load_nar
from yue2.protocol import (
    CODEC_OFFSET,
    MUSIC_END,
    chunk_ranges,
    negative_prefix,
    token_prefixes,
)
from yue2.tokenization_yue2 import YuE2TextTokenizer

MODEL = "/Volumes/Work/models/YuE2.models/lyra-YuE2-3B"
OUT = Path(__file__).parent / "nar_fixture"

STYLE = "upbeat synth-pop, female vocals, 128 BPM"
LYRICS = "[Verse]\nCity lights are burning slow\n[Chorus]\nWe run, we run"


def save(name, array):
    """bf16 arrays as uint16 bits; fp32 arrays as float32."""
    path = OUT / f"{name}.npy"
    if array.dtype == mx.bfloat16:
        np.save(path, np.array(mx.view(array, mx.uint16)))
    else:
        np.save(path, np.array(array))
    return path.name


def save_kv(name, key):
    """[1, kv_heads, len, head_dim] -> [len, kv_heads, head_dim] bits."""
    moved = key.transpose(0, 2, 1, 3)[0]
    np.save(OUT / f"{name}.npy", np.array(mx.view(moved, mx.uint16)))


def midpoint_trajectory(engine, steps):
    """Reference midpoint solver with per-step dumps (lyra CachedNAR.solve)."""
    state = engine._initial_state
    dt = 1.0 / steps
    half_dt = mx.array(dt / 2.0, dtype=mx.float32)
    full_dt = mx.array(dt, dtype=mx.float32)
    frames = []

    def logit(t):
        if t <= 0:
            return -20.0
        if t >= 1:
            return 20.0
        return max(-20.0, min(20.0, np.log(t / (1.0 - t))))

    for step in range(steps):
        t = 1.0 - step * dt
        first = engine.velocity(state, logit(t))
        midpoint = state - (first.astype(mx.float32) * half_dt).astype(mx.bfloat16)
        mid_t = t - dt / 2.0
        velocity = engine.velocity(midpoint, logit(mid_t))
        state = state - (velocity.astype(mx.float32) * full_dt).astype(mx.bfloat16)
        frames.append(np.array(mx.view(state, mx.uint16)).copy())
    return frames, np.array(state.astype(mx.float32))


def main():
    OUT.mkdir(exist_ok=True)
    tokenizer = YuE2TextTokenizer(f"{MODEL}/qwen.tiktoken")
    request_style = STYLE
    request_lyrics = LYRICS

    ar_model = load_ar(MODEL, precision="bf16", verify=False)
    model = load_nar(MODEL, ar_model, verify=False)

    rng = np.random.default_rng(99)
    codec = rng.integers(0, 32768, size=100).astype(np.int64).tolist()
    noise_all = rng.standard_normal((100, 64), dtype=np.float32)

    manifest = {"velocity": [], "solve": None, "multi": []}

    # cot=off prefix, exactly protocol.token_prefixes(request, tokenizer).
    text = f"Generate music with codec tokens from the given conditions.\n[Tags]\n{request_style}\n[Lyrics]\n{request_lyrics}\n"
    prefix = [151643] + tokenizer.encode(text) + [151847, 151848, 151851]

    frames = 12
    noise = noise_all[:frames]
    ar_tokens = prefix + [int(t) + CODEC_OFFSET for t in codec[:frames]] + [MUSIC_END]
    chunk = Chunk(ar_tokens=ar_tokens, noise=noise, nar_cond_end=0)
    engine = CachedNAR(model, chunk)

    # conditioning K/V for layers 0 and 27
    save_kv("cond_l0_key", engine.cache[0][0])
    save_kv("cond_l0_value", engine.cache[0][1])
    save_kv("cond_l27_key", engine.cache[27][0])
    save_kv("cond_l27_value", engine.cache[27][1])

    state = mx.array(noise, dtype=mx.bfloat16)
    for idx, raw in enumerate([1.0, 20.0, -20.0, 0.914]):
        out = engine.velocity(state, raw)
        case = {
            "name": f"vel{idx}",
            "raw_t": raw,
            "frames": frames,
            "ar_tokens": ar_tokens,
            "state": f"vel{idx}_state",
            "out": f"vel{idx}_out",
        }
        save(f"vel{idx}_state", state)
        save(f"vel{idx}_out", out)
        manifest["velocity"].append(case)

    # ---- midpoint trajectory, 4 steps ------------------------------------
    states, final = midpoint_trajectory(engine, 4)
    for i, s in enumerate(states):
        np.save(OUT / f"traj_state{i}.npy", s)
    np.save(OUT / "traj_final.npy", final)
    manifest["solve"] = {
        "steps": 4,
        "frames": frames,
        "ar_tokens": ar_tokens,
        "noise": "solve_noise",
        "states": [f"traj_state{i}" for i in range(len(states))],
        "final": "traj_final",
    }
    np.save(OUT / "solve_noise.npy", np.array(noise))
    engine.close()

    # ---- multi-chunk with a tiny context ---------------------------------
    small_context = 160
    ranges = chunk_ranges(len(codec), len(prefix), small_context)
    manifest["multi_context"] = small_context
    manifest["multi_prefix_len"] = len(prefix)
    manifest["multi_frames"] = len(codec)
    for ci, (start, end) in enumerate(ranges):
        c_tokens = prefix + [int(t) + CODEC_OFFSET for t in codec[start:end]] + [MUSIC_END]
        c_noise = noise_all[start:end]
        cchunk = Chunk(ar_tokens=c_tokens, noise=c_noise, nar_cond_end=0)
        cengine = CachedNAR(model, cchunk)
        cstates, cfinal = midpoint_trajectory(cengine, 2)
        np.save(OUT / f"multi{ci}_final.npy", cfinal)
        manifest["multi"].append({
            "chunk": ci,
            "start": start,
            "end": end,
            "ar_tokens": c_tokens,
            "noise": f"multi{ci}_noise",
            "final": f"multi{ci}_final",
        })
        np.save(OUT / f"multi{ci}_noise.npy", np.array(c_noise))
        cengine.close()
    manifest["multi_ranges"] = [list(r) for r in ranges]

    (OUT / "manifest.json").write_text(json.dumps(manifest))
    print(f"nar fixtures written to {OUT}")


if __name__ == "__main__":
    main()
