#!/usr/bin/env python3
"""VAE decoder fixtures from the lyra reference stack.

Run: /Volumes/Work/mlx-Yue/.venv/bin/python tests/gen_vae_fixture.py

Dumps the folded decoder weights (weight_norm + layout transposes applied,
exactly what lyra's OobleckDecoder consumes), per-stage activations for a
small full decode, and final audio for a chunked (core=16/halo=16) decode.
The C engine replays the same latents and compares stage by stage.
"""
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from lyra.vae import load_decoder

VAE_DIR = "/Volumes/Work/models/YuE2.models/lyra-YuE2-3B-vae"
OUT = Path(__file__).parent / "vae_fixture"


def save(name, array):
    np.save(OUT / f"{name}.npy", np.array(array))


def flat_layers(model):
    """Flatten OobleckDecoder.layers the same way the C plan does (the
    trailing Identity for final_tanh=false is skipped)."""
    for layer in model.layers:
        if hasattr(layer, "layers") and not hasattr(layer, "weight"):
            yield from flat_layers(layer)
        elif type(layer).__name__ == "Identity":
            continue
        else:
            yield layer


def decode_stages(model, latents, save, manifest, prefix, key):
    """Reference decode with per-stage dumps at the C plan's boundaries:
    after every snake, every ConvTranspose, and every completed ResidualUnit
    (whose tail adds the unit input back). Returns the final [T, 2] audio."""
    x = mx.array(latents[None], dtype=mx.float32)
    index = 0

    def dump():
        nonlocal index
        save(f"{prefix}{index:02d}", np.array(x[0]))
        manifest.setdefault(key, {})[f"{prefix}{index:02d}"] = list(x[0].shape)
        index += 1

    modules = [l for l in model.layers if type(l).__name__ != "Identity"]
    for li, layer in enumerate(modules):
        if hasattr(layer, "layers") and not hasattr(layer, "weight"):
            # DecoderBlock: snake, ConvT, 3 ResidualUnits
            subs = list(layer.layers)
            x = subs[0](x)
            dump()
            x = subs[1](x)
            dump()
            for ru in subs[2:]:
                x_in = x
                x = ru.layers[0](x)
                dump()
                x = ru.layers[1](x)
                x = ru.layers[2](x)
                dump()
                x = ru.layers[3](x)
                x = x_in + x
                dump()
        else:
            # conv_in / conv_out: applied but not dumped (C dump policy)
            x = layer(x)
            if li != 0 and li != len(modules) - 1:
                dump()
    return np.array(x[0])


def main():
    OUT.mkdir(exist_ok=True)
    model = load_decoder(VAE_DIR)
    manifest = {"stages": {}, "weights": {}, "full": {}, "chunked": {}}

    # folded weights: [out, k, in] float32 exactly as lyra consumes them
    def walk(prefix, tree):
        if isinstance(tree, dict):
            for key, value in tree.items():
                walk(f"{prefix}.{key}" if prefix else str(key), value)
        elif isinstance(tree, (list, tuple)):
            for i, value in enumerate(tree):
                walk(f"{prefix}.{i}", value)
        else:
            save(f"w_{prefix}", np.array(tree))
            manifest["weights"][prefix] = list(np.array(tree).shape)

    walk("", model.parameters())

    # small full decode with per-stage activations
    rng = np.random.default_rng(7)
    latents = (rng.standard_normal((8, 64)) * 0.5).astype(np.float32)
    np.save(OUT / "small_latents.npy", latents)
    audio = decode_stages(model, latents, save, manifest, "stage", "stages")
    save("small_audio", audio)
    manifest["full"] = {"frames": 8, "audio_shape": list(audio.shape)}

    # 40-frame decode: per-stage activations + the production chunked path
    latents40 = (rng.standard_normal((40, 64)) * 0.5).astype(np.float32)
    np.save(OUT / "chunk_latents.npy", latents40)
    audio40 = decode_stages(model, latents40, save, manifest, "full", "full40_stages")
    save("full40_audio", audio40)
    manifest["full40"] = {"frames": 40, "audio_shape": list(audio40.shape)}
    audio40c = model.decode(
        mx.array(latents40), core_frames=16, halo_frames=16, full=False)
    audio40c = np.array(audio40c)
    save("chunk_audio", audio40c)
    manifest["chunked"] = {"frames": 40, "core": 16, "halo": 16,
                           "audio_shape": list(audio40c.shape)}

    (OUT / "manifest.json").write_text(json.dumps(manifest))
    print(f"vae fixtures written to {OUT} "
          f"({len(manifest['stages'])}+{len(manifest['full40_stages'])} stages)")


if __name__ == "__main__":
    main()
