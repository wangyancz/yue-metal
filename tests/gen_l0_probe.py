#!/usr/bin/env python3
"""Dump layer-0 internals from lyra for comparison with QWEN_AR_TRACE_LAYER0.

Run: /Volumes/Work/mlx-Yue/.venv/bin/python tests/gen_l0_probe.py
Writes tests/ar_fixture/l0_*.npy (shapes match the C dumps).
"""
import os
from pathlib import Path

import mlx.core as mx
import numpy as np

import lyra.ar as lyra_ar
if os.environ.get("LYRA_FORCE_BF16_SDPA") == "1":
    lyra_ar._full_attention_requires_promotion = lambda: False

from lyra.ar import load_ar  # noqa: E402

MODEL = "/Volumes/Work/models/YuE2.models/lyra-YuE2-3B"
OUT = Path(__file__).parent / "ar_fixture"

STYLE = "upbeat synth-pop, female vocals, 128 BPM"
LYRICS = "[Verse]\nCity lights are burning slow\n[Chorus]\nWe run, we run"


def dump(name, array):
    """Store bf16 bits as uint16, shapes aligned with the C dumps."""
    array = mx.contiguous(array)
    bits = mx.view(array, mx.uint16)
    np.save(OUT / f"{name}.npy", np.array(bits))


def main() -> None:
    from yue2.protocol import SongRequest, token_prefixes
    from yue2.tokenization_yue2 import YuE2TextTokenizer

    tok = YuE2TextTokenizer(f"{MODEL}/qwen.tiktoken")
    req = SongRequest(style=STYLE, lyrics=LYRICS, cot="off", seed=7)
    tokens = token_prefixes(req, tok)
    t = len(tokens)

    model = load_ar(MODEL, precision="bf16", verify=False)
    x = model.model.embed_tokens(mx.array([tokens], dtype=mx.int32))
    layer = model.model.layers[0]
    sa = layer.self_attn
    mlp = layer.mlp

    normed = layer.input_layernorm(x)
    dump("l0_normed", normed[0])

    q = sa.q_proj(normed).reshape(1, t, sa.n_heads, -1)
    q = sa.q_norm(q)
    k = sa.k_proj(normed).reshape(1, t, sa.n_kv_heads, -1)
    k = sa.k_norm(k)
    v = sa.v_proj(normed).reshape(1, t, sa.n_kv_heads, -1)

    q_r = sa.rope(q.transpose(0, 2, 1, 3), offset=0)
    k_r = sa.rope(k.transpose(0, 2, 1, 3), offset=0)
    dump("l0_q", q_r.transpose(0, 2, 1, 3)[0])      # [t, heads, dim]
    dump("l0_k", k_r.transpose(0, 2, 1, 3)[0])      # [t, kv_heads, dim]
    dump("l0_v", v[0])

    out = lyra_ar._source_sdpa(q_r, k_r, v.transpose(0, 2, 1, 3),
                               scale=sa.scale, mask="causal")
    attn_flat = out.transpose(0, 2, 1, 3).reshape(1, t, -1)
    dump("l0_attn", attn_flat[0])
    branch = sa.o_proj(attn_flat)
    dump("l0_branch", branch[0])

    h = x + branch
    normed2 = layer.post_attention_layernorm(h)
    gate = mlp.gate_proj(normed2)
    up = mlp.up_proj(normed2)
    dump("l0_fc1out", mx.concatenate([gate, up], axis=-1)[0])
    mid = lyra_ar._source_silu(gate) * up
    dump("l0_mlp", mid[0])
    print("l0 probe written")


if __name__ == "__main__":
    main()
