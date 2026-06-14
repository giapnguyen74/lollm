"""
qwen3_5 family — manifest + registration (Qwen3.5 / Qwen3.6, model_type "qwen3_5").

Implemented through step 4: config parsing + streaming load (1), gated full-attention (2),
Gated DeltaNet (3), and the family cache + wired forward (4). The model now runs
end-to-end (prefill + cached decode); compare_logits parity (5) and the MTP head (6)
remain. See docs/qwen3_5-architecture.md for the build order.

Package roles:
  config.py             parse text_config → Qwen3_5Config (+ derived GDN / rotary dims)
  blocks.py             primitives (RMSNorm, RMSNormGated, GatedAttention, GatedDeltaNet, MLP)
  modeling_qwen3_5.py   Qwen3_5Cache + DecoderLayer (linear/full dispatch) + Qwen3_5Model.forward
  weights.py            identity name map + streaming load (shape-checked)
"""

from models import Family, register
from .weights import load

MODEL_TYPES = ["qwen3_5"]
DEFAULTS = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05}

register(Family(MODEL_TYPES, load, DEFAULTS))
