"""
qwen3_5 family — manifest + registration (Qwen3.5 / Qwen3.6, model_type "qwen3_5").

Implemented through step 6: config parsing + streaming load (1), gated full-attention (2),
Gated DeltaNet (3), the family cache + wired forward (4), compare_logits parity on
Qwen3.5-4B (5) — PASS (cosine ≈ 1, argmax match, fp32 CPU) — and the opt-in MTP head (6).
See docs/qwen3_5-architecture.md for the build order.

Package roles:
  config.py             parse text_config → Qwen3_5Config (+ derived GDN / rotary dims)
  blocks.py             primitives (RMSNorm, RMSNormGated, GatedAttention, GatedDeltaNet, MLP)
  modeling_qwen3_5.py   Qwen3_5Cache + DecoderLayer (linear/full dispatch) + Qwen3_5Model.forward
  weights.py            identity name map + streaming load (shape-checked)
  mtp.py                opt-in Eagle/MTP speculative head (loads mtp.*; base LM ignores it)
"""

from models import Family, register
from .weights import load

MODEL_TYPES = ["qwen3_5"]
DEFAULTS = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05}

register(Family(MODEL_TYPES, load, DEFAULTS))
