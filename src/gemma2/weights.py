"""
gemma2/weights.py — the weight-name seam + load (checkpoint → model).

Our module names mirror HF Gemma2, so the HF map is identity. The GGUF map is
best-effort (Gemma2's extra norms have llama.cpp-specific names) — strict load will
fail loudly if a name is wrong, so validate GGUF against llama.cpp before trusting.
"""

from __future__ import annotations

import re

import torch

from .config import build_config
from .modeling_gemma2 import Gemma2Model


def _set_param(model, dotted, tensor):
    """Assign a tensor to model.<dotted> as a no-grad Parameter (used for streaming)."""
    *path, last = dotted.split(".")
    m = model
    for name in path:
        m = getattr(m, name)
    m._parameters[last] = torch.nn.Parameter(tensor, requires_grad=False)

_GGUF_TOP = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}
# NOTE: GGUF gemma2 norm names are best-effort — confirm against llama.cpp.
_GGUF_BLK = {
    "input_layernorm.weight": "attn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "post_attention_layernorm.weight": "post_attention_norm.weight",
    "pre_feedforward_layernorm.weight": "ffn_norm.weight",
    "post_feedforward_layernorm.weight": "post_ffw_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}


def to_raw(canonical: str, fmt: str):
    if fmt == "hf":
        return canonical
    if canonical in _GGUF_TOP:
        return _GGUF_TOP[canonical]
    m = re.match(r"model\.layers\.(\d+)\.(.+)", canonical)
    if m and m.group(2) in _GGUF_BLK:
        return f"blk.{m.group(1)}.{_GGUF_BLK[m.group(2)]}"
    return None


def load(raw_config: dict, weights: dict, fmt: str, device="cpu", dtype=None,
         progress=None) -> Gemma2Model:
    cfg = build_config(raw_config, fmt)
    # Build on the meta device → no memory allocated for the params yet.
    with torch.device("meta"):
        model = Gemma2Model(cfg)

    tied = to_raw("lm_head.weight", fmt) not in weights
    # STREAM: move each weight to the device and free the CPU source (pop) as we go,
    # so we never hold the full CPU copy and the full device copy at the same time.
    # `progress(done, total)` is an optional caller signal — no UI code here.
    params = list(model.named_parameters())
    for i, (name, mp) in enumerate(params):
        if progress is not None:
            progress(i, len(params))
        if name == "lm_head.weight" and tied:
            continue
        raw = to_raw(name, fmt)
        if raw is None or raw not in weights:
            raise RuntimeError(f"gemma2: missing tensor for {name}")
        t = weights.pop(raw)
        # GGUF quirk: llama.cpp bakes Gemma's "+1" into the RMSNorm weights (for a
        # plain w·x norm). Our GemmaRMSNorm re-adds 1, so undo the bake here.
        if fmt == "gguf" and name.endswith("norm.weight"):
            t = t - 1.0
        if tuple(t.shape) != tuple(mp.shape):
            raise RuntimeError(f"gemma2: shape mismatch {name}: {tuple(t.shape)} vs {tuple(mp.shape)}")
        if dtype is not None:
            t = t.to(dtype)
        _set_param(model, name, t.to(device))
    if progress is not None:
        progress(len(params), len(params))         # final tick → closes the bar
    if tied:
        model.lm_head.weight = model.model.embed_tokens.weight
    return model.eval()
