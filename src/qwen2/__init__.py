"""
qwen2 family — manifest + registration.

Ties the package together: declares the model types it handles and the curated
sampling defaults, wires `load` (from weights.py, the loading concern) to the
registry. Importing this package registers the family.

Package roles:
  config.py          parse config.json / GGUF metadata → Qwen2Config
  blocks.py          primitives (RMSNorm, RoPE, attention, MLP, decoder layer)
  modeling_qwen2.py  Qwen2Model — composition + forward (pure architecture)
  weights.py         the weight-name seam + load (checkpoint → model)
"""

from models import Family, register
from .weights import load

MODEL_TYPES = ["qwen2"]
DEFAULTS = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.1}

register(Family(MODEL_TYPES, load, DEFAULTS))
