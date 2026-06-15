"""
qwen2/weights.py — the weight-name seam + load (checkpoint → model).

This is the *loading* concern, kept out of modeling: how a Qwen2 checkpoint's
tensors map onto the model's parameters, per format, and how to build + fill the
model. Every format quirk for this family lives here.

  to_raw(canonical, fmt) : our param name → the file's raw tensor name
  load(raw_config, w, fmt): build the config + model, map names, strict-load

It imports the model (modeling_qwen2) and config — the connection is explicit:
weights.py loads *into* Qwen2Model.
"""

from __future__ import annotations

import re

import torch

from .config import build_config
from .modeling_qwen2 import Qwen2Model


def _set_param(model, dotted, tensor):
    """Assign a tensor to model.<dotted> as a no-grad Parameter (used for streaming)."""
    *path, last = dotted.split(".")
    m = model
    for name in path:
        m = getattr(m, name)
    m._parameters[last] = torch.nn.Parameter(tensor, requires_grad=False)

# Our module names mirror HF, so HF is identity. GGUF map handles the rename
# (Qwen2 GGUF Q/K are not permuted → no transform needed). Quirks go here.
_GGUF_TOP = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}
_GGUF_BLK = {
    "input_layernorm.weight": "attn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight", "self_attn.q_proj.bias": "attn_q.bias",
    "self_attn.k_proj.weight": "attn_k.weight", "self_attn.k_proj.bias": "attn_k.bias",
    "self_attn.v_proj.weight": "attn_v.weight", "self_attn.v_proj.bias": "attn_v.bias",
    "self_attn.o_proj.weight": "attn_output.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight",
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
         progress=None) -> Qwen2Model:
    cfg = build_config(raw_config, fmt)
    # Build on the meta device → no memory allocated for the params yet.
    with torch.device("meta"):
        model = Qwen2Model(cfg)

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
            raise RuntimeError(f"qwen2: missing tensor for {name}")
        t = weights.pop(raw)                       # take + free from the source dict
        if tuple(t.shape) != tuple(mp.shape):      # safeguard we'd otherwise lose vs load_state_dict
            raise RuntimeError(f"qwen2: shape mismatch {name}: {tuple(t.shape)} vs {tuple(mp.shape)}")
        if dtype is not None:
            t = t.to(dtype)
        _set_param(model, name, t.to(device))
    if progress is not None:
        progress(len(params), len(params))         # final tick → closes the bar
    if tied:                                       # share the embedding on the device
        model.lm_head.weight = model.model.embed_tokens.weight
    return model.eval()
