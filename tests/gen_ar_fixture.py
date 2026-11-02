#!/usr/bin/env python3
"""AR fixture from the lyra reference stack (layer-by-layer).

Run: /Volumes/Work/mlx-Yue/.venv/bin/python tests/gen_ar_fixture.py [outdir]

Dumps, for a fixed off-mode prefix and two decode steps:
  embed bf16 [T,2048], layer0..27 hidden bf16, final bf16, semantic logits,
  layer 0/27 prefill KV, and per-step hidden/logits for the decode steps.
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

import lyra.ar as lyra_ar

# A/B switch: force lyra's bf16 SDPA path (no fp32 promotion) to separate
# intrinsic bf16-attention differences from real wiring bugs.
import os
FORCE_BF16_SDPA = os.environ.get("LYRA_FORCE_BF16_SDPA") == "1"
if FORCE_BF16_SDPA:
    lyra_ar._full_attention_requires_promotion = lambda: False

from lyra.ar import load_ar, make_cache, project_logits

MODEL = "/Volumes/Work/models/YuE2.models/lyra-YuE2-3B"

STYLE = "upbeat synth-pop, female vocals, 128 BPM"
LYRICS = "[Verse]\nCity lights are burning slow\n[Chorus]\nWe run, we run"
DECODE_TOKENS = [1234, 5678]


def dump(out, name, array):
    # MLX bf16 has no PEP 3118 descriptor; view the raw bits as uint16.
    np.save(out / f"{name}.npy", np.array(mx.view(array, mx.uint16)))


def main() -> None:
    out_dir = (Path(sys.argv[1]) if len(sys.argv) > 1
               else Path(__file__).parent / "ar_fixture")
    out_dir.mkdir(exist_ok=True)
    OUT = out_dir
    from yue2.protocol import SongRequest, token_prefixes
    from yue2.tokenization_yue2 import YuE2TextTokenizer

    tok = YuE2TextTokenizer(f"{MODEL}/qwen.tiktoken")
    req = SongRequest(style=STYLE, lyrics=LYRICS, cot="off", seed=7)
    tokens = token_prefixes(req, tok)
    print(f"prefix tokens: {len(tokens)}")

    model = load_ar(MODEL, precision="bf16", verify=False)
    capacity = len(tokens) + len(DECODE_TOKENS)
    caches = make_cache(model, capacity)

    # ---- manual layer-by-layer prefill (matches TextModel.forward) ----
    hidden = model.model.embed_tokens(mx.array([tokens], dtype=mx.int32))
    dump(OUT, "embed", hidden[0])
    layer_hidden = []
    for i, layer in enumerate(model.model.layers):
        hidden = layer(hidden, mask="causal", cache=caches[i])
        mx.eval(hidden)
        layer_hidden.append(hidden[0])
    final = model.model.norm(hidden)
    mx.eval(final)
    for i, state in enumerate(layer_hidden):
        dump(OUT, f"layer{i}", state)
    dump(OUT, "final", final[0])

    logits = project_logits(model, final[:, -1, :], "semantic")
    mx.eval(logits)
    dump(OUT, "logits_semantic", logits[0])

    for idx in (0, len(layer_hidden) - 1):
        keys, values = caches[idx].state
        dump(OUT, f"kvK{idx}", keys[0])   # [kv_heads, T, head_dim]
        dump(OUT, f"kvv{idx}", values[0])

    # ---- two decode steps ----
    # Manual layer-by-layer replication of model.model's decode loop:
    # create_attention_mask(N=1) returns None, so every layer sees mask=None.
    steps = []
    for step, token in enumerate(DECODE_TOKENS):
        hidden = model.model.embed_tokens(mx.array([[token]], dtype=mx.int32))
        for i, layer in enumerate(model.model.layers):
            hidden = layer(hidden, mask=None, cache=caches[i])
            mx.eval(hidden)
            dump(OUT, f"step{step}_layer{i}", hidden[0])
        final = model.model.norm(hidden)
        mx.eval(final)
        dump(OUT, f"step{step}_hidden", final[0, -1])
        logits = project_logits(model, final[:, -1, :], "semantic")
        mx.eval(logits)
        dump(OUT, f"step{step}_logits", logits[0])
        steps.append(len(tokens) + step + 1)
    keys, values = caches[0].state
    dump(OUT, "kv0_after", keys[0])

    meta = {
        "tokens": tokens,
        "decode_tokens": DECODE_TOKENS,
        "hidden_size": 2048,
        "layers": len(layer_hidden),
        "cache_capacity": capacity,
    }
    (OUT / "meta.json").write_text(json.dumps(meta))
    print(f"fixtures written to {OUT}")


if __name__ == "__main__":
    main()
