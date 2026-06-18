"""
config.py — diffusion_gemma config (text backbone only, rung 1 scope).

Parses the HF `config.json`'s nested `text_config` into a flat dataclass. Vision /
diffusion-sampler fields are ignored for now (rungs 2/6). Mirrors what the transformers
`DiffusionGemmaTextConfig` exposes, named the same where it helps the weight map stay identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DiffusionGemmaConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_global_key_value_heads: int
    head_dim: int                       # local (sliding) head dim
    global_head_dim: int                # global (full) head dim
    intermediate_size: int              # dense MLP width
    moe_intermediate_size: int          # per-expert width
    num_experts: int
    top_k_experts: int
    layer_types: list                   # per-layer "sliding_attention" | "full_attention"
    sliding_window: int
    rms_norm_eps: float
    max_position_embeddings: int
    final_logit_softcapping: float
    hidden_act: str
    tie_word_embeddings: bool
    use_bidirectional_attention: str    # "vision" on this checkpoint → encoder stays causal
    attention_bias: bool
    # RoPE, per layer type
    rope_theta_local: float
    rope_theta_global: float
    partial_rotary_factor_global: float

    def is_sliding(self, i: int) -> bool:
        return self.layer_types[i] == "sliding_attention"

    @property
    def first_global_layer(self) -> int:
        """Index of the first full-attention layer (its cache is never clipped, so it tracks the
        true sequence length even when sliding layers are capped to the window)."""
        for i in range(self.num_hidden_layers):
            if not self.is_sliding(i):
                return i
        return 0

    @classmethod
    def from_hf(cls, raw: dict) -> "DiffusionGemmaConfig":
        t = raw.get("text_config", raw)
        rp = t["rope_parameters"]
        sl, fu = rp["sliding_attention"], rp["full_attention"]
        return cls(
            vocab_size=t["vocab_size"],
            hidden_size=t["hidden_size"],
            num_hidden_layers=t["num_hidden_layers"],
            num_attention_heads=t["num_attention_heads"],
            num_key_value_heads=t["num_key_value_heads"],
            num_global_key_value_heads=t["num_global_key_value_heads"],
            head_dim=t["head_dim"],
            global_head_dim=t["global_head_dim"],
            intermediate_size=t["intermediate_size"],
            moe_intermediate_size=t["moe_intermediate_size"],
            num_experts=t["num_experts"],
            top_k_experts=t["top_k_experts"],
            layer_types=list(t["layer_types"]),
            sliding_window=t["sliding_window"],
            rms_norm_eps=t["rms_norm_eps"],
            max_position_embeddings=t["max_position_embeddings"],
            final_logit_softcapping=t.get("final_logit_softcapping", 0.0),
            hidden_act=t["hidden_activation"],
            tie_word_embeddings=t.get("tie_word_embeddings", True),
            use_bidirectional_attention=t.get("use_bidirectional_attention", "vision"),
            attention_bias=t.get("attention_bias", False),
            rope_theta_local=sl["rope_theta"],
            rope_theta_global=fu["rope_theta"],
            partial_rotary_factor_global=fu.get("partial_rotary_factor", 1.0),
        )
