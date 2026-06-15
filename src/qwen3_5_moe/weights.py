"""
qwen3_5_moe/weights.py — the weight-name seam + streaming load (checkpoint → model).

Our module names mirror HF, so the safetensors map is identity. The MoE experts are stored
FUSED — `...mlp.experts.gate_up_proj` (num_experts, 2·moe_inter, hidden) and
`...mlp.experts.down_proj` (num_experts, hidden, moe_inter) — and `FusedExperts` holds them as
those exact stacked Parameters, so no un-stacking is needed: the names and shapes line up
1:1. The router (`...mlp.gate.weight`), shared expert (`...mlp.shared_expert.*`) and its gate
(`...mlp.shared_expert_gate.weight`) are per-tensor and also map by name. Build on `meta`,
stream each weight onto the device (freeing the CPU source as we go), assert shapes, and re-tie
embeddings when `lm_head` is absent. No GGUF path — Gated DeltaNet + MoE GGUF isn't standardized.

The strict shape check below is what verifies the fused layout: a mismatch (e.g. a checkpoint
that stores experts transposed, or per-expert) fails loud here rather than silently mis-loading.
"""

from __future__ import annotations

import torch

from .config import build_config
from .modeling_qwen3_5_moe import Qwen3_5MoeModel


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
    # stays top-level (present only when untied).
    if fmt == "hf":
        if canonical.startswith("model."):
            return "model.language_model." + canonical[len("model."):]
        return canonical
    raise NotImplementedError(
        "qwen3_5_moe: only hf (safetensors) supported; GGUF not implemented")


def load(raw_config: dict, weights: dict, fmt: str, device="cpu", dtype=None,
         progress=None) -> Qwen3_5MoeModel:
    cfg = build_config(raw_config, fmt)
    # Build on meta → no memory allocated for params yet.
    with torch.device("meta"):
        model = Qwen3_5MoeModel(cfg)

    tied = to_raw("lm_head.weight", fmt) not in weights
    # STREAM: move each weight to the device and free the CPU source (pop) as we go.
    params = list(model.named_parameters())
    for i, (name, mp) in enumerate(params):
        if progress is not None:
            progress(i, len(params))
        if name == "lm_head.weight" and tied:
            continue
        raw = to_raw(name, fmt)
        if raw is None or raw not in weights:
            raise RuntimeError(f"qwen3_5_moe: missing tensor for {name} (raw {raw})")
        t = weights.pop(raw)
        if tuple(t.shape) != tuple(mp.shape):
            raise RuntimeError(
                f"qwen3_5_moe: shape mismatch {name}: {tuple(t.shape)} vs {tuple(mp.shape)}")
        if dtype is not None:
            t = t.to(dtype)
        _set_param(model, name, t.to(device))
    if progress is not None:
        progress(len(params), len(params))     # final tick → closes the bar
    if tied:                                   # share the embedding on the device
        model.lm_head.weight = model.model.embed_tokens.weight
    return model.eval()
