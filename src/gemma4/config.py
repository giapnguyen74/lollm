"""
gemma4/config.py — Gemma4's config parsing (text decoder, E2B).

Gemma4 is a Gemma3-shaped decoder with several changes that all matter for parity
(verified against transformers `Gemma4TextConfig` / `modeling_gemma4.py` and the real
`google/gemma-4-e2b-it` config). Read into one `Gemma4Config`:

  - **Plain RMSNorm** `w·x̂` with ONES init — NOT Gemma2/3's `(1+w)`.
  - **Attention scale = 1.0** (no `query_pre_attn_scalar`); QK-norm absorbs scaling.
  - **Per-Layer Embeddings (PLE)** — a second embedding table + per-layer projection.
  - **Shared KV cache** — the last `num_kv_shared_layers` layers reuse K/V.
  - **Asymmetric heads** — global (full) layers use `global_head_dim` (512) with
    *proportional* RoPE; local (sliding) layers use `head_dim` (256) with default RoPE.
  - **Double-wide MLP** on the shared-KV layers (`use_double_wide_mlp`).
  - **Final logit soft-cap** (30.0) is back (Gemma3 had none); no attention soft-cap.

Target: `google/gemma-4-e2b-it` (model_type "gemma4" top-level / "gemma4_text").
Text-only scope: vision/audio sub-configs are ignored here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Gemma4Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int                          # local (sliding) layers
    global_head_dim: int                   # global (full) layers
    rms_norm_eps: float
    sliding_window: int
    layer_types: list                      # per-layer "sliding_attention" / "full_attention"
    rope_theta_global: float
    rope_theta_local: float
    partial_rotary_factor_global: float    # proportional RoPE on global layers (e.g. 0.25)
    final_logit_softcapping: Optional[float]
    hidden_size_per_layer_input: int       # PLE per-layer dim (256)
    vocab_size_per_layer_input: int
    num_kv_shared_layers: int
    use_double_wide_mlp: bool
    tie_word_embeddings: bool

    # derived (filled in __post_init__)
    first_kv_shared_idx: int = field(init=False)
    _donor: dict = field(init=False)       # layer_type -> donor layer index (last non-shared of type)

    def __post_init__(self):
        self.first_kv_shared_idx = self.num_hidden_layers - self.num_kv_shared_layers
        # donor = last NON-shared layer of each type; shared layers reuse its K/V.
        self._donor = {}
        for i in range(self.first_kv_shared_idx):
            self._donor[self.layer_types[i]] = i

    # ── per-layer helpers ──
    def is_global(self, i: int) -> bool:
        return self.layer_types[i] == "full_attention"

    def is_kv_shared(self, i: int) -> bool:
        return self.num_kv_shared_layers > 0 and i >= self.first_kv_shared_idx

    def is_donor(self, i: int) -> bool:
        """True if this non-shared layer is the one whose K/V the shared layers reuse."""
        return (not self.is_kv_shared(i)) and self._donor.get(self.layer_types[i]) == i

    def layer_head_dim(self, i: int) -> int:
        return self.global_head_dim if self.is_global(i) else self.head_dim

    def layer_intermediate(self, i: int) -> int:
        return self.intermediate_size * (2 if self.use_double_wide_mlp and self.is_kv_shared(i)
                                         else 1)

    @property
    def n_rep(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_hf(cls, raw: dict) -> "Gemma4Config":
        if "text_config" in raw:                      # multimodal checkpoint → text branch
            raw = raw["text_config"]

        n_layers = raw["num_hidden_layers"]
        layer_types = raw.get("layer_types")
        if layer_types is None:                        # fall back to the 5:1 generator (pattern 6)
            pat = raw.get("sliding_window_pattern", 6)
            layer_types = ["sliding_attention" if (i + 1) % pat else "full_attention"
                           for i in range(n_layers)]

        rp = raw.get("rope_parameters") or {}
        full = rp.get("full_attention") or {}
        local = rp.get("sliding_attention") or {}
        theta_global = full.get("rope_theta", raw.get("rope_theta", 1_000_000.0))
        theta_local = local.get("rope_theta", raw.get("rope_local_base_freq", 10_000.0))
        partial = full.get("partial_rotary_factor", 0.25)
        if full and full.get("rope_type") not in ("proportional", "default", None):
            raise NotImplementedError(
                f"gemma4: unsupported global rope_type {full.get('rope_type')!r} "
                f"(only proportional/default implemented).")

        # Gemma4 reintroduced ATTENTION soft-cap? No — only the final logit cap. Guard it.
        if raw.get("attn_logit_softcapping") is not None:
            raise NotImplementedError(
                "gemma4: attn_logit_softcapping set, but the reference uses no attention "
                "soft-cap (scale=1.0 + QK-norm). Refusing to guess.")
        if raw.get("enable_moe_block"):
            raise NotImplementedError(
                "gemma4: enable_moe_block=True (MoE) is not implemented in this text-only "
                "family yet — E2B is dense. Refusing to guess.")
        if raw.get("attention_k_eq_v"):
            raise NotImplementedError(
                "gemma4: attention_k_eq_v=True (V reuses K projection) not implemented "
                "(E2B is False). Refusing to guess.")

        head_dim = raw.get("head_dim", 256)
        return cls(
            vocab_size=raw["vocab_size"], hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"], num_hidden_layers=n_layers,
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw.get("num_key_value_heads", raw["num_attention_heads"]),
            head_dim=head_dim,
            global_head_dim=raw.get("global_head_dim", head_dim),
            rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
            sliding_window=raw.get("sliding_window", 512),
            layer_types=layer_types,
            rope_theta_global=theta_global, rope_theta_local=theta_local,
            partial_rotary_factor_global=partial,
            final_logit_softcapping=raw.get("final_logit_softcapping"),
            hidden_size_per_layer_input=raw.get("hidden_size_per_layer_input", 256),
            vocab_size_per_layer_input=raw.get("vocab_size_per_layer_input", raw["vocab_size"]),
            num_kv_shared_layers=raw.get("num_kv_shared_layers", 0),
            use_double_wide_mlp=raw.get("use_double_wide_mlp", False),
            tie_word_embeddings=raw.get("tie_word_embeddings", True))

    @classmethod
    def from_gguf(cls, meta: dict) -> "Gemma4Config":
        raise NotImplementedError(
            "gemma4 GGUF is not supported: PLE / shared-KV / proportional-RoPE metadata "
            "keys are unverified against llama.cpp. Per 'hard-fail, never guess' we raise.")


def build_config(raw: dict, fmt: str) -> Gemma4Config:
    return Gemma4Config.from_hf(raw) if fmt == "hf" else Gemma4Config.from_gguf(raw)
