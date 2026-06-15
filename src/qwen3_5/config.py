"""
qwen3_5/config.py — Qwen3.5 / Qwen3.6 config parsing (the family owns this).

One `Qwen3_5Config` for both 3.5 and 3.6: they share `model_type: "qwen3_5"` and only
differ in field values (sizes, `output_gate_type`, tying). The checkpoint is multimodal,
so the language-model fields live under `text_config` — we parse that sub-config and
ignore the vision tower. See docs/qwen3_5-architecture.md.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Qwen3_5Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    tie_word_embeddings: bool
    hidden_act: str
    # hybrid schedule
    layer_types: list
    full_attention_interval: int
    # full-attention extras
    attn_output_gate: bool
    output_gate_type: str
    # linear-attention (Gated DeltaNet)
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int

    # ── derived ──
    @property
    def n_rep(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def key_dim(self) -> int:
        return self.linear_key_head_dim * self.linear_num_key_heads

    @property
    def value_dim(self) -> int:
        return self.linear_value_head_dim * self.linear_num_value_heads

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim          # q + k + v → in_proj_qkv + conv1d

    @property
    def v_per_k(self) -> int:
        return self.linear_num_value_heads // self.linear_num_key_heads

    def layer_type(self, i: int) -> str:
        return self.layer_types[i]

    @classmethod
    def from_hf(cls, raw: dict) -> "Qwen3_5Config":
        t = raw.get("text_config", raw)                   # multimodal: text sub-config
        hidden = t["hidden_size"]
        n_head = t["num_attention_heads"]
        rp = t.get("rope_parameters", {})
        partial = rp.get("partial_rotary_factor", t.get("partial_rotary_factor", 1.0))
        return cls(
            vocab_size=t["vocab_size"],
            hidden_size=hidden,
            intermediate_size=t["intermediate_size"],
            num_hidden_layers=t["num_hidden_layers"],
            num_attention_heads=n_head,
            num_key_value_heads=t.get("num_key_value_heads", n_head),
            head_dim=t.get("head_dim", hidden // n_head),
            rms_norm_eps=t.get("rms_norm_eps", 1e-6),
            rope_theta=rp.get("rope_theta", t.get("rope_theta", 1e7)),
            partial_rotary_factor=partial,
            tie_word_embeddings=t.get(
                "tie_word_embeddings", raw.get("tie_word_embeddings", False)),
            hidden_act=t.get("hidden_act", "silu"),
            layer_types=t["layer_types"],
            full_attention_interval=t.get("full_attention_interval", 4),
            attn_output_gate=t.get("attn_output_gate", True),
            output_gate_type=t.get("output_gate_type", "sigmoid"),   # parsed but unused: the
            #   reference hardcodes sigmoid and ignores this field (27B sets it "swish")
            linear_num_key_heads=t["linear_num_key_heads"],
            linear_num_value_heads=t["linear_num_value_heads"],
            linear_key_head_dim=t["linear_key_head_dim"],
            linear_value_head_dim=t["linear_value_head_dim"],
            linear_conv_kernel_dim=t["linear_conv_kernel_dim"],
        )


def build_config(raw: dict, fmt: str) -> Qwen3_5Config:
    if fmt != "hf":
        raise NotImplementedError(
            "qwen3_5: only safetensors (hf) supported; GGUF for Gated DeltaNet "
            "is not standardized (fail loud rather than guess)")
    return Qwen3_5Config.from_hf(raw)
