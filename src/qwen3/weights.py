"""
qwen3/weights.py — the weight-name seam + streaming load.

Like Qwen2, plus the QK-norm tensors (`q_norm`/`k_norm`) and no qkv-bias entries.
Our names mirror HF (identity map); the GGUF map renames to llama.cpp's scheme.
"""

from __future__ import annotations

import re

import torch

from .config import build_config
from .modeling_qwen3 import Qwen3Model


def _set_param(model, dotted, tensor):
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
_GGUF_BLK = {
    "input_layernorm.weight": "attn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",   # QK-norm (Qwen3)
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
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


def load(raw_config: dict, weights: dict, fmt: str, device="cpu", dtype=None) -> Qwen3Model:
    cfg = build_config(raw_config, fmt)
    with torch.device("meta"):
        model = Qwen3Model(cfg)

    tied = to_raw("lm_head.weight", fmt) not in weights
    for name, mp in list(model.named_parameters()):
        if name == "lm_head.weight" and tied:
            continue
        raw = to_raw(name, fmt)
        if raw is None or raw not in weights:
            raise RuntimeError(f"qwen3: missing tensor for {name}")
        t = weights.pop(raw)
        if tuple(t.shape) != tuple(mp.shape):
            raise RuntimeError(f"qwen3: shape mismatch {name}: {tuple(t.shape)} vs {tuple(mp.shape)}")
        if dtype is not None:
            t = t.to(dtype)
        _set_param(model, name, t.to(device))
    if tied:
        model.lm_head.weight = model.model.embed_tokens.weight
    return model.eval()
