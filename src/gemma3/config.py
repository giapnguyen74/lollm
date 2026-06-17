"""
gemma3/config.py — Gemma3's config parsing.

Gemma3 is a diff from Gemma2: it DROPS both logit soft-caps and instead adds
QK-norm (RMSNorm over head_dim on Q and K). It keeps the per-layer attention
scale (`query_pre_attn_scalar`) and the alternating local/global attention, but
the pattern is now **5 local : 1 global** (`sliding_window_pattern`, default 6)
with a **dual RoPE**: local layers use a small theta (10k), global layers a large
theta (1e6) plus an optional long-context scaling factor.

Target checkpoint: google/gemma-3-1b-it (model_type "gemma3_text", text-only
Gemma3ForCausalLM). The multimodal 4B+ checkpoints nest the decoder under a
`text_config` — we read that too if present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Gemma3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    query_pre_attn_scalar: float          # attention scale = this ** -0.5
    sliding_window: int                   # window size for local layers
    sliding_window_pattern: int           # every Nth layer is global (5 local : 1 global → 6)
    rope_theta_global: float              # full-attention layers
    rope_theta_local: float               # sliding-attention layers
    rope_scaling_factor: float            # linear long-context scale on GLOBAL layers (1.0 = none)
    tie_word_embeddings: bool

    @property
    def n_rep(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def is_global(self, layer_idx: int) -> bool:
        """5 local : 1 global — layer i is global when (i+1) % pattern == 0."""
        return (layer_idx + 1) % self.sliding_window_pattern == 0

    @classmethod
    def from_hf(cls, raw: dict) -> "Gemma3Config":
        # Multimodal checkpoints (4B+) wrap the decoder config under "text_config".
        if "text_config" in raw:
            raw = raw["text_config"]

        hidden = raw["hidden_size"]
        n_head = raw["num_attention_heads"]
        head_dim = raw.get("head_dim", hidden // n_head)

        # Dual RoPE thetas: prefer the nested rope_parameters block (new format),
        # fall back to the flat keys the original Google configs ship.
        rp = raw.get("rope_parameters") or {}
        full = rp.get("full_attention") or {}
        local = rp.get("sliding_attention") or {}
        theta_global = full.get("rope_theta", raw.get("rope_theta", 1_000_000.0))
        theta_local = local.get("rope_theta", raw.get("rope_local_base_freq", 10_000.0))

        # Optional long-context scaling on the GLOBAL layers (linear). Absent on the
        # 1B (32K context); present (factor 8) on the long-context 4B+ models. May be
        # carried either inside rope_parameters.full_attention or as a top-level
        # rope_scaling dict.
        scale = 1.0
        rs = full if "factor" in full else raw.get("rope_scaling") or {}
        if rs and rs.get("rope_type") in ("linear", "default", None) and "factor" in rs:
            scale = float(rs["factor"])

        # Gemma3 dropped soft-capping; assert we aren't silently ignoring it.
        if raw.get("attn_logit_softcapping") is not None or raw.get("final_logit_softcapping") is not None:
            raise NotImplementedError(
                "gemma3: this checkpoint sets a logit soft-cap, but Gemma3 is "
                "expected to use QK-norm instead (no soft-caps). Refusing to guess.")

        return cls(
            vocab_size=raw["vocab_size"], hidden_size=hidden,
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=n_head,
            num_key_value_heads=raw.get("num_key_value_heads", n_head),
            head_dim=head_dim,
            rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
            query_pre_attn_scalar=raw.get("query_pre_attn_scalar", head_dim),
            sliding_window=raw.get("sliding_window", 512),
            sliding_window_pattern=raw.get("sliding_window_pattern",
                                           raw.get("_sliding_window_pattern", 6)),
            rope_theta_global=theta_global,
            rope_theta_local=theta_local,
            rope_scaling_factor=scale,
            tie_word_embeddings=raw.get("tie_word_embeddings", True))

    @classmethod
    def from_gguf(cls, meta: dict) -> "Gemma3Config":
        # Per the engine's vision: hard-fail rather than guess unverified GGUF keys.
        # Gemma3's QK-norm + dual-RoPE metadata names aren't validated against
        # llama.cpp yet, so refuse the GGUF path until a parity run confirms them.
        raise NotImplementedError(
            "gemma3 GGUF is not supported yet: the QK-norm / dual-RoPE / "
            "sliding-window-pattern metadata keys are unverified against llama.cpp. "
            "Per the project's 'hard-fail, never guess' rule we raise instead of "
            "defaulting. Validate the safetensors path first, then wire GGUF.")


def build_config(raw: dict, fmt: str) -> Gemma3Config:
    return Gemma3Config.from_hf(raw) if fmt == "hf" else Gemma3Config.from_gguf(raw)
