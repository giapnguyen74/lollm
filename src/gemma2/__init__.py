"""
gemma2 family — manifest + registration.

Package roles (same shape as qwen2/):
  config.py            parse config.json / GGUF metadata → Gemma2Config
  blocks.py            (1+w) RMSNorm · RoPE · soft-cap + sliding-window attention · GeGLU
  modeling_gemma2.py   DecoderLayer (sandwich norm) + Gemma2Model (embed scale, final softcap)
  weights.py           the weight-name seam + load (checkpoint → model)
"""

from models import Family, register
from .weights import load

MODEL_TYPES = ["gemma2"]
# Gemma's recommended sampling.
DEFAULTS = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "repetition_penalty": 1.0}

register(Family(MODEL_TYPES, load, DEFAULTS))
