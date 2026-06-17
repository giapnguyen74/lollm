"""
gemma3 family — manifest + registration.

Package roles (same shape as qwen2/ / gemma2/):
  config.py            parse config.json (text_config) → Gemma3Config
  blocks.py            (1+w) RMSNorm · RoPE · QK-norm attention (no soft-cap) · GeGLU
  modeling_gemma3.py   DecoderLayer (sandwich norm) + Gemma3Model (embed scale, dual RoPE)
  weights.py           the weight-name seam + load (checkpoint → model)

Diff from gemma2: QK-norm replaces soft-capping; 5:1 local/global with dual RoPE.
Target: google/gemma-3-1b-it (text-only). model_type "gemma3" (multimodal top-level)
and "gemma3_text" (the 1B/text decoder) both route here.
"""

from models import Family, register
from .weights import load

MODEL_TYPES = ["gemma3", "gemma3_text"]
# Gemma's recommended sampling.
DEFAULTS = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "repetition_penalty": 1.0}

register(Family(MODEL_TYPES, load, DEFAULTS))
