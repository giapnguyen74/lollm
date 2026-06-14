"""
gemma2/config.py — Gemma2's config parsing.

Gemma2 adds knobs Qwen2 doesn't have: a per-layer attention scale
(`query_pre_attn_scalar`), logit soft-capping (attention + final), and a sliding
window (used on alternating layers). Read into one `Gemma2Config`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Gemma2Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    query_pre_attn_scalar: float          # attention scale = this ** -0.5
    attn_logit_softcapping: Optional[float]
    final_logit_softcapping: Optional[float]
    sliding_window: int                   # window size for the local (even) layers
    tie_word_embeddings: bool

    @property
    def n_rep(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_hf(cls, raw: dict) -> "Gemma2Config":
        hidden = raw["hidden_size"]
        n_head = raw["num_attention_heads"]
        head_dim = raw.get("head_dim", hidden // n_head)
        return cls(
            vocab_size=raw["vocab_size"], hidden_size=hidden,
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=n_head,
            num_key_value_heads=raw.get("num_key_value_heads", n_head),
            head_dim=head_dim,
            rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
            rope_theta=raw.get("rope_theta", 10000.0),
            query_pre_attn_scalar=raw.get("query_pre_attn_scalar", head_dim),
            attn_logit_softcapping=raw.get("attn_logit_softcapping", 50.0),
            final_logit_softcapping=raw.get("final_logit_softcapping", 30.0),
            sliding_window=raw.get("sliding_window", 4096),
            tie_word_embeddings=raw.get("tie_word_embeddings", True))

    @classmethod
    def from_gguf(cls, meta: dict) -> "Gemma2Config":
        arch = meta["general.architecture"]
        g = lambda k, d=None: meta.get(f"{arch}.{k}", d)

        # The Gemma2-specific knobs (attn scale, both soft-caps, sliding window) are
        # the bits whose GGUF key names we have NOT validated against llama.cpp.
        # Per the vision, fail loud rather than silently default to a plausible-but-
        # wrong value (which would emit confident garbage). Confirm the key names,
        # then promote them to `g(...)` once a parity run passes.
        def greq(k):
            v = g(k)
            if v is None:
                raise NotImplementedError(
                    f"gemma2 GGUF: required key '{arch}.{k}' is missing/unverified. "
                    f"This family's GGUF metadata mapping has not been validated "
                    f"against llama.cpp — refusing to guess (fail loud). Confirm the "
                    f"key name and value, then wire it in.")
            return v

        hidden = g("embedding_length")
        n_head = g("attention.head_count")
        head_dim = g("attention.key_length", hidden // n_head)
        tokens = meta.get("tokenizer.ggml.tokens")
        return cls(
            vocab_size=len(tokens) if tokens is not None else g("vocab_size"),
            hidden_size=hidden,
            intermediate_size=g("feed_forward_length"),
            num_hidden_layers=g("block_count"),
            num_attention_heads=n_head,
            num_key_value_heads=g("attention.head_count_kv", n_head),
            head_dim=head_dim,
            rms_norm_eps=g("attention.layer_norm_rms_epsilon", 1e-6),
            rope_theta=g("rope.freq_base", 10000.0),
            query_pre_attn_scalar=greq("attention.scale_factor"),
            attn_logit_softcapping=greq("attn_logit_softcapping"),
            final_logit_softcapping=greq("final_logit_softcapping"),
            sliding_window=greq("attention.sliding_window"),
            tie_word_embeddings=True)


def build_config(raw: dict, fmt: str) -> Gemma2Config:
    return Gemma2Config.from_hf(raw) if fmt == "hf" else Gemma2Config.from_gguf(raw)
