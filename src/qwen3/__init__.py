"""
qwen3 family — manifest + registration.

  config.py            parse config.json / GGUF metadata → Qwen3Config
  blocks.py            RMSNorm · RoPE · attention (QK-norm, no bias) · SwiGLU MLP
  modeling_qwen3.py    Qwen3Model — composition + forward
  weights.py           weight-name seam (+ q_norm/k_norm) + streaming load
"""

from models import Family, register
from .weights import load

MODEL_TYPES = ["qwen3"]
DEFAULTS = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.1}

register(Family(MODEL_TYPES, load, DEFAULTS))
