"""
gemma3/weights.py — the weight-name seam + load (checkpoint → model).

Our module names mirror HF Gemma3 (text-only Gemma3ForCausalLM), so the HF map is
identity — including the NEW q_norm/k_norm tensors. GGUF is not supported yet
(config.from_gguf hard-fails), so `to_raw` only handles the HF path.

Note for the 1B target: it's a text-only Gemma3ForCausalLM, so weights are keyed
top-level (`model.layers.N...`, `model.embed_tokens`, `model.norm`). The multimodal
4B+ checkpoints nest these under `model.language_model.*` and `language_model.*` —
that map would be added alongside the vision tower when we extend past text.
"""

from __future__ import annotations

import torch

from .config import build_config
from .modeling_gemma3 import Gemma3Model


def _set_param(model, dotted, tensor):
    """Assign a tensor to model.<dotted> as a no-grad Parameter (used for streaming)."""
    *path, last = dotted.split(".")
    m = model
    for name in path:
        m = getattr(m, name)
    m._parameters[last] = torch.nn.Parameter(tensor, requires_grad=False)


def to_raw(canonical: str, fmt: str):
    if fmt == "hf":
        return canonical                      # module names mirror HF → identity map
    raise NotImplementedError(
        "gemma3 GGUF is not supported yet (see gemma3/config.py::from_gguf).")


def load(raw_config: dict, weights: dict, fmt: str, device="cpu", dtype=None,
         progress=None) -> Gemma3Model:
    cfg = build_config(raw_config, fmt)
    # Build on the meta device → no memory allocated for the params yet.
    with torch.device("meta"):
        model = Gemma3Model(cfg)

    tied = to_raw("lm_head.weight", fmt) not in weights
    # STREAM: move each weight to the device and free the CPU source (pop) as we go,
    # so we never hold the full CPU copy and the full device copy at the same time.
    params = list(model.named_parameters())
    for i, (name, mp) in enumerate(params):
        if progress is not None:
            progress(i, len(params))
        if name == "lm_head.weight" and tied:
            continue
        raw = to_raw(name, fmt)
        if raw is None or raw not in weights:
            raise RuntimeError(f"gemma3: missing tensor for {name}")
        t = weights.pop(raw)
        if tuple(t.shape) != tuple(mp.shape):
            raise RuntimeError(f"gemma3: shape mismatch {name}: {tuple(t.shape)} vs {tuple(mp.shape)}")
        if dtype is not None:
            t = t.to(dtype)
        _set_param(model, name, t.to(device))
    if progress is not None:
        progress(len(params), len(params))         # final tick → closes the bar
    if tied:
        model.lm_head.weight = model.model.embed_tokens.weight
    return model.eval()
