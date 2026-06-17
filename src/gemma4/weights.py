"""
gemma4/weights.py — the weight-name seam + load (checkpoint → model), text only.

`google/gemma-4-e2b-it` is a multimodal `Gemma4ForConditionalGeneration`, so the text
decoder lives under `model.language_model.*` (vision = `model.vision_tower.*`, audio =
`model.audio_tower.*` — both ignored here). Our module names mirror the clean `model.*`
layout, and `to_raw` rewrites the prefix `model. → model.language_model.` (the family
owns its name map). `lm_head.weight` stays top-level and is tied to `embed_tokens`.

Shared-KV layers carry no k/v projections (our model doesn't create them), so those
tensors are simply never requested. Per-layer Q/K/V and MLP shapes differ by layer
(global head_dim 512 vs 256; double-wide MLP on shared layers) — the strict per-tensor
shape check still applies. GGUF is unsupported (config.from_gguf hard-fails).
"""

from __future__ import annotations

import torch

from .config import build_config
from .modeling_gemma4 import Gemma4Model


def _set_param(model, dotted, tensor):
    *path, last = dotted.split(".")
    m = model
    for name in path:
        m = getattr(m, name)
    m._parameters[last] = torch.nn.Parameter(tensor, requires_grad=False)


def to_raw(canonical: str, fmt: str):
    if fmt != "hf":
        raise NotImplementedError("gemma4 GGUF is not supported (see gemma4/config.py::from_gguf).")
    if canonical.startswith("lm_head."):
        return canonical
    if canonical.startswith("model."):
        return "model.language_model." + canonical[len("model."):]
    return canonical


def _resolve(raw_name, weights):
    """Return the checkpoint key present in `weights`: prefer the nested name, else fall
    back to the un-nested name (a text-only Gemma4ForCausalLM checkpoint)."""
    if raw_name in weights:
        return raw_name
    flat = raw_name.replace("model.language_model.", "model.", 1)
    return flat if flat in weights else None


def load(raw_config: dict, weights: dict, fmt: str, device="cpu", dtype=None,
         progress=None) -> Gemma4Model:
    cfg = build_config(raw_config, fmt)
    with torch.device("meta"):
        model = Gemma4Model(cfg)

    tied = _resolve(to_raw("lm_head.weight", fmt), weights) is None
    params = list(model.named_parameters())
    for i, (name, mp) in enumerate(params):
        if progress is not None:
            progress(i, len(params))
        if name == "lm_head.weight" and tied:
            continue
        raw = _resolve(to_raw(name, fmt), weights)
        if raw is None:
            raise RuntimeError(f"gemma4: missing tensor for {name} (looked for {to_raw(name, fmt)!r})")
        t = weights.pop(raw)
        if tuple(t.shape) != tuple(mp.shape):
            raise RuntimeError(f"gemma4: shape mismatch {name}: {tuple(t.shape)} vs {tuple(mp.shape)}")
        if dtype is not None:
            t = t.to(dtype)
        _set_param(model, name, t.to(device))
    if progress is not None:
        progress(len(params), len(params))
    if tied:
        model.lm_head.weight = model.model.embed_tokens.weight
    return model.eval()
