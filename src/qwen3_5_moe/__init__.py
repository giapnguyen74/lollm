"""
qwen3_5_moe family — manifest + registration (Qwen3.5-MoE, model_type "qwen3_5_moe").

Shares the dense qwen3_5 hybrid backbone (gated full-attention + Gated DeltaNet + partial
RoPE + opt-in MTP) and swaps the per-layer FFN for a sparse Mixture-of-Experts block
(router + routed SwiGLU experts + always-on shared expert, Qwen3-Next layout).

NOTE: `MODEL_TYPES` must match the checkpoint's `config.json["model_type"]`. Adjust the list
if your checkpoint reports a different string (e.g. "qwen3_next").

Package roles:
  config.py                 parse text_config → Qwen3_5MoeConfig (+ MoE fields, is_moe_layer)
  blocks.py                 primitives (copied from qwen3_5) + SparseMoeBlock
  modeling_qwen3_5_moe.py   Qwen3_5MoeCache + DecoderLayer (MoE/dense FFN dispatch) + model.forward
  weights.py                identity name map + streaming load (shape-checked)
  mtp.py                    opt-in Eagle/MTP speculative head
"""

from models import Family, register
from .weights import load

MODEL_TYPES = ["qwen3_5_moe"]
DEFAULTS = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05}

register(Family(MODEL_TYPES, load, DEFAULTS))
