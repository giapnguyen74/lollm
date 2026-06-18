"""
qwen3_5_moe/config.py — Qwen3.5-MoE config parsing (the family owns this).

Same hybrid backbone as the dense `qwen3_5` family (Gated DeltaNet + gated full-attention,
partial RoPE, optional MTP) — this config adds ONLY the sparse-FFN fields. Every transformer
layer's MLP becomes a Mixture-of-Experts block (router + N routed experts + a shared expert),
EXCEPT layers forced dense by `mlp_only_layers` or by the `decoder_sparse_step` stride. The
checkpoint is multimodal, so the language-model fields live under `text_config`. See
docs/qwen3_5-architecture.md for the shared backbone; the MoE block is documented in blocks.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Qwen3_5MoeConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int           # dense-MLP width (for any mlp_only / non-sparse layer)
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
    # ── MoE (the only thing this family adds over qwen3_5) ──
    num_experts: int                 # total routed experts
    num_experts_per_tok: int         # top-k
    moe_intermediate_size: int       # per-routed-expert SwiGLU width
    shared_expert_intermediate_size: int  # shared-expert SwiGLU width (0 → no shared expert)
    norm_topk_prob: bool             # renormalize the top-k routing weights to sum to 1
    decoder_sparse_step: int = 1     # layer is MoE when (idx+1) % step == 0 (1 → every layer)
    mlp_only_layers: list = field(default_factory=list)  # layer indices forced dense

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

    def is_moe_layer(self, i: int) -> bool:
        """MoE when the layer isn't forced dense and falls on the sparse stride (HF rule)."""
        if i in self.mlp_only_layers:
            return False
        if self.num_experts <= 0:
            return False
        return (i + 1) % self.decoder_sparse_step == 0

    @classmethod
    def from_hf(cls, raw: dict) -> "Qwen3_5MoeConfig":
        t = raw.get("text_config", raw)                   # multimodal: text sub-config
        hidden = t["hidden_size"]
        n_head = t["num_attention_heads"]
        rp = t.get("rope_parameters", {})
        partial = rp.get("partial_rotary_factor", t.get("partial_rotary_factor", 1.0))
        # A fully-sparse MoE checkpoint has no dense `intermediate_size` (every layer is MoE);
        # it's only needed for `mlp_only_layers` / skipped strides. Fall back to the expert
        # width so the dense MLP is still constructible if a config does force a dense layer.
        moe_inter = t.get("moe_intermediate_size", t.get("intermediate_size", 0))
        dense_inter = t.get("intermediate_size", moe_inter)
        # Routing width is load-bearing and must never be guessed: a missing key defaulting
        # to 0 makes `torch.topk(probs, 0)` select NO experts, so only the shared expert
        # survives and the model emits fluent-but-wrong text with no error. Per "hard-fail,
        # never guess" we raise instead of defaulting. (num_experts itself is allowed to be
        # absent → is_moe_layer treats <=0 as "no MoE layers".)
        if "num_experts_per_tok" in t:
            n_experts_per_tok = t["num_experts_per_tok"]
        elif "num_experts_per_token" in t:
            n_experts_per_tok = t["num_experts_per_token"]
        else:
            raise KeyError(
                "qwen3_5_moe config has no 'num_experts_per_tok' (router top-k width). "
                "Refusing to default to 0 — that would route to zero experts and emit "
                "confident garbage. Add the key to config.json.")
        return cls(
            vocab_size=t["vocab_size"],
            hidden_size=hidden,
            intermediate_size=dense_inter,
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
            output_gate_type=t.get("output_gate_type", "sigmoid"),
            linear_num_key_heads=t["linear_num_key_heads"],
            linear_num_value_heads=t["linear_num_value_heads"],
            linear_key_head_dim=t["linear_key_head_dim"],
            linear_value_head_dim=t["linear_value_head_dim"],
            linear_conv_kernel_dim=t["linear_conv_kernel_dim"],
            # ── MoE ──
            num_experts=t.get("num_experts", t.get("n_routed_experts", 0)),
            num_experts_per_tok=n_experts_per_tok,   # resolved above (raises if absent)
            moe_intermediate_size=moe_inter,
            shared_expert_intermediate_size=t.get("shared_expert_intermediate_size", 0),
            norm_topk_prob=t.get("norm_topk_prob", True),
            decoder_sparse_step=t.get("decoder_sparse_step", 1),
            mlp_only_layers=list(t.get("mlp_only_layers", [])),
        )


def build_config(raw: dict, fmt: str) -> Qwen3_5MoeConfig:
    if fmt != "hf":
        raise NotImplementedError(
            "qwen3_5_moe: only safetensors (hf) supported; GGUF for Gated DeltaNet + MoE "
            "is not standardized (fail loud rather than guess)")
    return Qwen3_5MoeConfig.from_hf(raw)
