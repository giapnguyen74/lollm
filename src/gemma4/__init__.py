"""
gemma4 family — manifest + registration (text decoder only).

Package roles (same shape as qwen2/ / gemma3/):
  config.py            parse config.json (text_config) → Gemma4Config
  blocks.py            plain RMSNorm · default/proportional RoPE · QK+V-norm attention
                       (scale=1.0, shared-KV, per-layer head_dim) · GeGLU MLP
  modeling_gemma4.py   DecoderLayer (sandwich norm + PLE inject) + Gemma4Model
                       (PLE pipeline, dual RoPE, final logit soft-cap)
  weights.py           nested `model.language_model.*` name map + streaming load

Scope: TEXT decoder only — vision/audio towers are intentionally not loaded
(see docs/gemma4-architecture.md and docs/multimodal-processors.md).
Target: google/gemma-4-e2b-it. model_type "gemma4" / "gemma4_text" route here.
"""

from models import Family, register
from .weights import load

MODEL_TYPES = ["gemma4", "gemma4_text"]
DEFAULTS = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "repetition_penalty": 1.0}

register(Family(MODEL_TYPES, load, DEFAULTS))
