"""
qwen3_5/weights.py — the weight-name seam + streaming load (checkpoint → model).

Our module names mirror HF, so the safetensors map is identity. STEP 1: build the model
on `meta`, stream each weight onto the device (freeing the CPU source as we go), assert
shapes, and re-tie embeddings when `lm_head` is absent. No GGUF path — Gated DeltaNet
GGUF isn't standardized, so we hard-fail rather than guess.
"""

from __future__ import annotations

import torch

from .config import build_config
from .modeling_qwen3_5 import Qwen3_5Model


def _set_param(model, dotted, tensor):
    """Assign a tensor to model.<dotted> as a no-grad Parameter (streaming load)."""
    *path, last = dotted.split(".")
    m = model
    for name in path:
        m = getattr(m, name)
    m._parameters[last] = torch.nn.Parameter(tensor, requires_grad=False)


def to_raw(canonical: str, fmt: str):
    # The checkpoint is the multimodal VL model, so the text LM lives under
    # `model.language_model.*` (vision is `model.visual.*`, ignored). Our modeling keeps
    # the clean `model.*` names; the family rewrites the prefix here. `lm_head.weight`
    # stays top-level (present only when untied; 4B ties it → absent).
    if fmt == "hf":
        if canonical.startswith("model."):
            return "model.language_model." + canonical[len("model."):]
        return canonical
    raise NotImplementedError(
        "qwen3_5: only hf (safetensors) supported; GGUF not implemented")


def load(raw_config: dict, weights: dict, fmt: str, device="cpu", dtype=None,
         progress=None) -> Qwen3_5Model:
    cfg = build_config(raw_config, fmt)
    # Build on meta → no memory allocated for params yet.
    with torch.device("meta"):
        model = Qwen3_5Model(cfg)

    tied = to_raw("lm_head.weight", fmt) not in weights
    # STREAM: move each weight to the device and free the CPU source (pop) as we go.
    # `progress(done, total)` is an optional caller-supplied signal — no UI code here.
    params = list(model.named_parameters())
    for i, (name, mp) in enumerate(params):
        if progress is not None:
            progress(i, len(params))
        if name == "lm_head.weight" and tied:
            continue
        raw = to_raw(name, fmt)
        if raw is None or raw not in weights:
            raise RuntimeError(f"qwen3_5: missing tensor for {name}")
        t = weights.pop(raw)
        if tuple(t.shape) != tuple(mp.shape):
            raise RuntimeError(
                f"qwen3_5: shape mismatch {name}: {tuple(t.shape)} vs {tuple(mp.shape)}")
        if dtype is not None:
            t = t.to(dtype)
        _set_param(model, name, t.to(device))
    if progress is not None:
        progress(len(params), len(params))     # final tick → closes the bar
    if tied:                                   # share the embedding on the device
        model.lm_head.weight = model.model.embed_tokens.weight
    return model.eval()
