#!/usr/bin/env python3
"""Generate tokenizer/protocol fixtures from the reference lyra stack.

Run with the mlx-Yue venv (tiktoken + vendored yue2 protocol):
  /Volumes/Work/mlx-Yue/.venv/bin/python tests/gen_tokenizer_fixture.py

Outputs tests/tokenizer_fixture/{encode,prefix,decode,chunk}.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/Volumes/Work/mlx-Yue/vendor/yue/src")
sys.path.insert(0, "/Volumes/Work/mlx-Yue/src")

from yue2.protocol import (  # noqa: E402
    GenerationConfig,
    SongRequest,
    negative_prefix,
    token_prefixes,
)
from yue2.tokenization_yue2 import YuE2TextTokenizer  # noqa: E402

TOK_PATH = "/Volumes/Work/models/YuE2.models/lyra-YuE2-3B/qwen.tiktoken"
OUT = Path(__file__).parent / "tokenizer_fixture"

TEXT_CASES = [
    "",
    "hello world",
    "Hello, World! 123 numbers 456",
    "pop, female vocals, upbeat synth-pop, 128 BPM",
    "upbeat synth-pop with punchy bass and bright arpeggios",
    "[Verse]\nCity lights are burning slow\n[Chorus]\nWe run, we run",
    "  leading spaces\tand\ttabs\n\nnewlines\r\ncrlf",
    "中文歌词测试：夜色温柔，星光满天。",
    "mixed 中english 文 with 数字42 and emoji 🎵🎶",
    "apostrophes don't can't won't it's I'll we've they're",
    "punct...!!!???;;;,,,---===",
    "   ",
    "\n\n\n",
    "X:1\nT:Test\nM:4/4\nL:1/8\nK:C\n|:CDEF GABc|cBAG FEDC:|",
    "<abc> literal markers </abc> stay ordinary <|endoftext|>",
    "repeat repeat repeat repeat repeat repeat repeat",
    "a" * 300,
    "Tab\there\nNew\n\r\nWindows\rline",
    "unicode accents: café naïve résumé Zürich 東京都",
    "URL https://example.com/path?q=1&x=2#frag end",
]

ABC_TEXT = "X:1\nT:Fixture\nM:4/4\nL:1/8\nK:C\n|:C2DE G2AB|c2BA GED2:|"

STYLE = "upbeat synth-pop, female vocals, 128 BPM"
LYRICS = "[Verse]\nCity lights are burning slow\n[Chorus]\nWe run, we run"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    tok = YuE2TextTokenizer(TOK_PATH)

    encode_cases = [{"text": text, "ids": tok.encode(text)}
                    for text in TEXT_CASES]
    (OUT / "encode.json").write_text(json.dumps(encode_cases))

    # Prefix construction over all cot modes with a fixed external ABC.
    prefix_cases = []
    for cot in ("full", "melody", "off"):
        abc_ids = None if cot == "off" else tok.encode(ABC_TEXT)
        req = SongRequest(style=STYLE, lyrics=LYRICS, cot=cot, seed=7,
                          abc=ABC_TEXT if cot != "off" else None)
        prefix_cases.append({
            "cot": cot,
            "prefix": token_prefixes(req, tok, abc_ids),
            "negative": negative_prefix(req, tok, abc_ids),
        })
    # Planning prefix (no ABC yet): ends at ABC_START.
    req = SongRequest(style=STYLE, lyrics=LYRICS, cot="full", seed=7)
    prefix_cases.append({"cot": "full+plan", "prefix": token_prefixes(req, tok),
                         "negative": None})
    (OUT / "prefix.json").write_text(json.dumps(prefix_cases))

    # Decode: ordinary text, specials, and codec-token skipping (lyra drops
    # ids >= 151851 and renders 151643..151850 as their literal strings).
    codec = 151853 + 123
    decode_cases = [
        {"ids": tok.encode("decoded text with spaces"), "text": "decoded text with spaces"},
        {"ids": [151643], "text": "<|endoftext|>"},
        {"ids": [151847, *tok.encode("score"), 151848], "text": "<abc>score</abc>"},
        {"ids": [151851, *tok.encode("a b"), 151852, codec, 184621], "text": "a b"},
        {"ids": [184703, 151850], "text": "<extra_199>"},
    ]
    (OUT / "decode.json").write_text(json.dumps(decode_cases))

    # chunk_ranges reference over representative sizes.
    from yue2.protocol import chunk_ranges
    chunk_cases = []
    for frames, prefix in ((1, 10), (9000, 500), (24575, 10), (100, 24572),
                           (24573, 0), (5, 24571)):
        try:
            ranges = chunk_ranges(frames, prefix)
        except ValueError as error:
            chunk_cases.append({"frames": frames, "prefix": prefix,
                                "error": str(error)})
            continue
        chunk_cases.append({"frames": frames, "prefix": prefix,
                            "ranges": [list(r) for r in ranges]})
    (OUT / "chunk.json").write_text(json.dumps(chunk_cases))

    config = GenerationConfig()
    (OUT / "generation_config.json").write_text(json.dumps(config.to_dict()))
    print(f"fixtures written to {OUT}")


if __name__ == "__main__":
    main()
