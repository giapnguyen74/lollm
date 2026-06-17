"""
models.py — the model registry (was the `families/` package, now one file).

Holds the `Family` record + register/get, and imports each `modeling_<family>`
module at the bottom so it self-registers. Adding a model = drop a
`modeling_<name>.py` and add one import line here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_REG: dict[str, "Family"] = {}


@dataclass
class Family:
    model_types: list
    load: object                       # (raw_config, weights, fmt) -> model
    defaults: dict = field(default_factory=dict)


def register(fam: "Family"):
    for mt in fam.model_types:
        _REG[mt] = fam


def get(model_type: str) -> "Family":
    if model_type not in _REG:
        raise ValueError(
            f"unsupported model_type '{model_type}'. Registered: {sorted(_REG)}")
    return _REG[model_type]


# ── self-registering models (import = register) ──
import qwen2     # noqa: E402,F401
import qwen3     # noqa: E402,F401
import gemma2    # noqa: E402,F401
import gemma3    # noqa: E402,F401
import gemma4    # noqa: E402,F401
import qwen3_5   # noqa: E402,F401
import qwen3_5_moe  # noqa: E402,F401
