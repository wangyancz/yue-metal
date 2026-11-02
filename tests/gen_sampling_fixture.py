#!/usr/bin/env python3
"""Sampling fixtures from the lyra reference stack.

Run: /Volumes/Work/mlx-Yue/.venv/bin/python tests/gen_sampling_fixture.py

Part A (unit cases): random bf16 logits through lyra guided_logits() and
distribution() across a matrix of phases / legacy paths / penalties /
temperatures / top-k / top-p, dumping inputs and outputs bit-exactly.

Part B (greedy end-to-end): generate_tokens() with temperature=0 for the two
production flows — cot=off (CFG 1.01, legacy BF16 sampling) and cot=melody
(abc planning then semantic, fp32 sampling). Greedy runs are deterministic:
the C engine must reproduce the exact token sequences.
"""
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

import lyra.ar as lyra_ar

from lyra.ar import generate_tokens, load_ar
from yue2.protocol import (
    ABC_END,
    CODEC_OFFSET,
    CODEC_SIZE,
    EOD,
    MUSIC_END,
    GenerationConfig,
    SongRequest,
    negative_prefix,
    token_prefixes,
)
from yue2.tokenization_yue2 import YuE2TextTokenizer

MODEL = "/Volumes/Work/models/YuE2.models/lyra-YuE2-3B"
OUT = Path(__file__).parent / "sampling_fixture"

STYLE = "upbeat synth-pop, female vocals, 128 BPM"
LYRICS = "[Verse]\nCity lights are burning slow\n[Chorus]\nWe run, we run"
ABC_TEXT = "X:1\nM:4/4\nK:C\nCDEF GABc|"


def dump_bits(name, array):
    """Raw little-endian bits of an MLX array (bf16 -> uint16, f32 -> float32)."""
    if array.dtype == mx.bfloat16:
        np.save(OUT / f"{name}.bin", np.array(mx.view(array, mx.uint16)))
        return "bf16"
    np.save(OUT / f"{name}.bin", np.array(array.astype(mx.float32)))
    return "f32"


def random_bf16(seed, width, scale=4.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, scale, size=width).astype(np.float32)
    return mx.array(x).astype(mx.bfloat16)


def mixed_history(seed, n):
    """Native ids mixing in-phase tokens with out-of-phase noise."""
    rng = np.random.default_rng(seed)
    pools = [
        lambda: int(rng.integers(0, EOD)),                     # text ids
        lambda: ABC_END,
        lambda: MUSIC_END,
        lambda: int(CODEC_OFFSET + rng.integers(0, CODEC_SIZE)),
        lambda: 184621,                                        # latent junk
    ]
    return [pools[int(rng.integers(0, len(pools)))]() for _ in range(n)]


def sampling_params(temp, **overrides):
    base = GenerationConfig().semantic
    values = dict(
        temperature=temp,
        top_p=base.top_p,
        top_k=base.top_k,
        repetition_penalty=base.repetition_penalty,
        penalty_window=base.penalty_window,
        min_tokens=base.min_tokens,
        max_tokens=base.max_tokens,
    )
    values.update(overrides)
    return values


class _SamplingNS:
    """Attribute-view of a params dict for lyra's distribution()."""

    def __init__(self, values):
        self.__dict__.update(values)


def lyra_ar_distribution_params(params):
    return _SamplingNS(params)


def unit_cases():
    cases = []

    def add(name, phase, legacy, params, step, history, cfg=None):
        width = (EOD + 1) if phase == "abc" else (CODEC_SIZE + 1)
        logits = random_bf16(hash(name) & 0xFFFF, width)
        case = {
            "name": name,
            "phase": phase,
            "legacy": legacy,
            "step": step,
            "history": history,
            "params": params,
            "logits": name,
            "logits_dtype": dump_bits(name, logits),
        }
        scores_input = logits
        if cfg is not None:
            uncond = random_bf16((hash(name) >> 3) & 0xFFFF, width)
            guided = lyra_ar.guided_logits(logits, uncond, cfg)
            case["cfg"] = cfg
            case["uncond"] = name + "_uncond"
            case["uncond_dtype"] = dump_bits(name + "_uncond", uncond)
            case["guided"] = name + "_guided"
            case["guided_dtype"] = dump_bits(name + "_guided", guided)
            scores_input = guided
        scores = lyra_ar.distribution(scores_input,
                                      lyra_ar_distribution_params(params),
                                      history, step, phase, legacy_off=legacy)
        case["scores"] = name + "_scores"
        case["scores_dtype"] = dump_bits(name + "_scores", scores)
        case["argmax"] = int(mx.argmax(scores).item())
        cases.append(case)

    # Production abc (fp32): greedy + sampled parameter sets.
    hist_abc = mixed_history(11, 100)
    add("abc_greedy_mask", "abc", 0,
        sampling_params(0.0, top_p=0.9, top_k=30, repetition_penalty=1.005,
                        penalty_window=100, min_tokens=32, max_tokens=4096),
        0, hist_abc)
    add("abc_greedy_free", "abc", 0,
        sampling_params(0.0, top_p=0.9, top_k=30, repetition_penalty=1.005,
                        penalty_window=100, min_tokens=32, max_tokens=4096),
        40, hist_abc)
    add("abc_sampled", "abc", 0,
        sampling_params(0.7, top_p=0.9, top_k=30, repetition_penalty=1.005,
                        penalty_window=100, min_tokens=32, max_tokens=4096),
        40, hist_abc)
    # Production semantic (fp32).
    hist_sem = mixed_history(12, 50)
    add("sem_greedy_mask", "semantic", 0, sampling_params(0.0), 0, hist_sem)
    add("sem_greedy_free", "semantic", 0, sampling_params(0.0), 300, hist_sem)
    add("sem_sampled", "semantic", 0, sampling_params(1.0), 300, hist_sem)
    # Legacy semantic (cot=off): BF16 arithmetic throughout.
    add("sem_legacy_sampled", "semantic", 1, sampling_params(1.0), 300, hist_sem)
    add("sem_legacy_greedy", "semantic", 1, sampling_params(0.0), 300, hist_sem)
    add("sem_legacy_cfg", "semantic", 1, sampling_params(1.0), 300, hist_sem,
        cfg=1.01)
    # CFG unit case on the production path as well.
    add("sem_cfg_fp32", "semantic", 0, sampling_params(1.0), 300, hist_sem,
        cfg=1.01)
    return cases


def e2e_cases(model, tokenizer):
    cases = {}
    greedy_sem = sampling_params(0.0, min_tokens=2, max_tokens=64)
    greedy_abc = sampling_params(0.0, top_p=0.9, top_k=30,
                                 repetition_penalty=1.005, penalty_window=100,
                                 min_tokens=2, max_tokens=64)

    # cot=off: CFG 1.01 with legacy BF16 sampling.
    req_off = SongRequest(style=STYLE, lyrics=LYRICS, cot="off", seed=7)
    prefix = token_prefixes(req_off, tokenizer)
    negative = negative_prefix(req_off, tokenizer)
    history, timing, truncated = generate_tokens(
        model, prefix, lyra_ar_distribution_params(greedy_sem), req_off.seed,
        "semantic", negative=negative, cfg_scale=req_off.guidance,
        legacy_off=True)
    cases["off_semantic"] = {
        "name": "off_semantic", "phase": "semantic", "prefix": prefix,
        "negative": negative, "cfg_scale": req_off.guidance, "legacy": True,
        "params": greedy_sem, "seed": req_off.seed,
        "history": history, "eos": not truncated, "truncated": truncated,
    }

    # cot=melody: greedy abc plan feeds the semantic stage (fp32 sampling).
    req_mel = SongRequest(style=STYLE, lyrics=LYRICS, cot="melody", seed=7)
    abc_prefix = token_prefixes(req_mel, tokenizer)
    abc_ids, _, abc_trunc = generate_tokens(
        model, abc_prefix, lyra_ar_distribution_params(greedy_abc),
        req_mel.seed, "abc")
    sem_prefix = token_prefixes(req_mel, tokenizer, abc_ids)
    sem_history, _, sem_trunc = generate_tokens(
        model, sem_prefix, lyra_ar_distribution_params(greedy_sem),
        req_mel.seed, "semantic")
    cases["melody_abc"] = {
        "name": "melody_abc", "phase": "abc", "prefix": abc_prefix,
        "negative": None, "cfg_scale": 1.0, "legacy": False,
        "params": greedy_abc, "seed": req_mel.seed, "history": abc_ids,
        "eos": not abc_trunc, "truncated": abc_trunc,
    }
    cases["melody_semantic"] = {
        "name": "melody_semantic", "phase": "semantic", "prefix": sem_prefix,
        "negative": None, "cfg_scale": 1.0, "legacy": False,
        "params": greedy_sem, "seed": req_mel.seed, "history": sem_history,
        "eos": not sem_trunc, "truncated": sem_trunc,
    }
    return cases


def main():
    OUT.mkdir(exist_ok=True)
    manifest = {"unit": unit_cases(), "e2e": None}
    (OUT / "manifest.json").write_text(json.dumps(manifest))
    print("unit cases written; loading model for greedy runs...")

    tokenizer = YuE2TextTokenizer(f"{MODEL}/qwen.tiktoken")
    model = load_ar(MODEL, precision="bf16", verify=False)
    manifest["e2e"] = e2e_cases(model, tokenizer)
    (OUT / "manifest.json").write_text(json.dumps(manifest))
    print(f"fixtures written to {OUT}")


if __name__ == "__main__":
    main()
