#!/usr/bin/env python3
"""Compare QWEN_AR_DUMP layer dumps against the lyra fixture, layer by layer.

Usage: QWEN_AR_DUMP=/tmp/ar_dump ./tests/test_ar   (fails fast not required)
       python3 tests/cmp_ar_layers.py /tmp/ar_dump
"""
import sys
from pathlib import Path

import numpy as np

DUMP = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ar_dump")
FIX = Path(__file__).parent / "ar_fixture"


def load_bf16(path):
    raw = np.fromfile(path, dtype="<u2")
    return raw.view(np.uint16).astype(np.uint32) << 16


def as_f32(bits):
    return bits.view(np.float32) if bits.dtype == np.float32 else bits


def main() -> None:
    prev = None
    for i in range(28):
        ref_bits = (np.load(FIX / f"layer{i}.npy").astype(np.uint32) << 16)
        ref = ref_bits.view(np.float32).astype(np.float64).ravel()
        dump = DUMP / f"layer{i}.bin"
        got_bits = np.fromfile(dump, dtype="<u2").astype(np.uint32) << 16
        got = got_bits.view(np.float32).astype(np.float64)
        n = min(ref.size, got.size)
        r, g = ref[:n], got[:n]
        rel = np.sqrt(((g - r) ** 2).sum() / max((r**2).sum(), 1e-9))
        mismatch = (got_bits.ravel()[: ref_bits.size] !=
                    ref_bits.ravel()).mean()
        print(f"layer{i:2d} relL2={rel:.5f} mismatch={mismatch:.3f} "
              f"rms_ref={np.sqrt((r**2).mean()):.3f}")


if __name__ == "__main__":
    main()
