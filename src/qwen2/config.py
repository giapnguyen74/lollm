"""
qwen2/config.py — Qwen2's own config parsing (the family owns this).

Reads a Qwen2 config from either format into one `Qwen2Config`. `build_config`
dispatches on `fmt`. This is sibling code inside the qwen2 package — imported by
modeling_qwen2.py and blocks.py, never by another family.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Qwen2Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @property
    def n_rep(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_hf(cls, raw: dict) -> "Qwen2Config":
        hidden = raw["hidden_size"]
        n_head = raw["num_attention_heads"]
        return cls(
            vocab_size=raw["vocab_size"], hidden_size=hidden,
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=n_head,
            num_key_value_heads=raw.get("num_key_value_heads", n_head),
            head_dim=raw.get("head_dim", hidden // n_head),
            rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
            rope_theta=raw.get("rope_theta", 1000000.0),
            tie_word_embeddings=raw.get("tie_word_embeddings", False))

    @classmethod
    def from_gguf(cls, meta: dict) -> "Qwen2Config":
        arch = meta["general.architecture"]
        g = lambda k, d=None: meta.get(f"{arch}.{k}", d)
        hidden = g("embedding_length")
        n_head = g("attention.head_count")
        tokens = meta.get("tokenizer.ggml.tokens")
        return cls(
            vocab_size=len(tokens) if tokens is not None else g("vocab_size"),
            hidden_size=hidden,
            intermediate_size=g("feed_forward_length"),
            num_hidden_layers=g("block_count"),
            num_attention_heads=n_head,
            num_key_value_heads=g("attention.head_count_kv", n_head),
            head_dim=g("attention.key_length", hidden // n_head),
            rms_norm_eps=g("attention.layer_norm_rms_epsilon", 1e-6),
            rope_theta=g("rope.freq_base", 1000000.0),
            tie_word_embeddings=False)


def build_config(raw: dict, fmt: str) -> Qwen2Config:
    return Qwen2Config.from_hf(raw) if fmt == "hf" else Qwen2Config.from_gguf(raw)
