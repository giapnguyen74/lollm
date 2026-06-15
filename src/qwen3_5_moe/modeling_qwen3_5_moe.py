"""
qwen3_5_moe/modeling_qwen3_5_moe.py — the architecture: decoder layer + model + forward.

Same hybrid backbone as dense qwen3_5 (pre-norm residual; every 4th layer full-attention,
the rest Gated DeltaNet; family-owned cache because the two mixers keep different per-step
state). The ONLY architectural change is the FFN sub-block: on MoE layers `self.mlp` is a
`SparseMoeBlock` (router + experts + shared expert); on `mlp_only_layers` / skipped strides
it's the dense `MLP`. Both share the `mlp(x) -> x` signature, so the residual update reads
identically either way. See docs/qwen3_5-architecture.md for the backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Qwen3_5MoeConfig
from .blocks import RMSNorm, RoPE, GatedAttention, GatedDeltaNet, MLP, SparseMoeBlock
from .kv import Qwen3_5MoeCache       # cache lives in kv.py (model talks to it via methods)

__all__ = ["DecoderLayer", "Qwen3_5MoeModel", "Qwen3_5MoeCache"]


class DecoderLayer(nn.Module):
    """Pre-norm residual block. The token mixer is full-attention every 4th layer (GQA,
    gated, partial RoPE) and Gated DeltaNet otherwise. The FFN is a `SparseMoeBlock` on MoE
    layers and a dense `MLP` on the rest. `cache=None` runs an uncached one-shot pass (used
    by the MTP head)."""

    def __init__(self, cfg: Qwen3_5MoeConfig, layer_idx: int,
                 layer_type: str | None = None, is_moe: bool | None = None):
        super().__init__()
        # `layer_type` / `is_moe` overrides let the MTP head reuse this block with an explicit
        # mixer + FFN kind, independent of its position in the main schedule.
        self.layer_type = layer_type or cfg.layer_type(layer_idx)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if self.layer_type == "full_attention":
            self.self_attn = GatedAttention(cfg)          # HF name: self_attn
        else:
            self.linear_attn = GatedDeltaNet(cfg)         # HF name: linear_attn
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        use_moe = cfg.is_moe_layer(layer_idx) if is_moe is None else is_moe
        self.mlp = SparseMoeBlock(cfg) if use_moe else MLP(cfg)

    def forward(self, x, cos, sin, cache=None, layer_idx=0, use_cache=True):
        # 1. TOKEN MIXER sub-block (mixes ACROSS tokens):  x = x + mix(norm(x))
        h_in = self.input_layernorm(x)
        if self.layer_type == "full_attention":
            h = self.self_attn(h_in, cos, sin, cache, layer_idx)   # RoPE + growing KV cache
        else:
            do_cache = use_cache and cache is not None
            conv_s, rec_s = cache.linear_state(layer_idx) if cache is not None else (None, None)
            h, new_conv, new_rec = self.linear_attn(h_in, conv_s, rec_s, use_cache=do_cache)
            if do_cache:
                cache.set_linear_state(layer_idx, new_conv, new_rec)
        x = x + h
        # 2. FFN sub-block (transforms EACH token): x = x + ffn(norm(x))  [MoE or dense]
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3_5MoeModel(nn.Module):
    def __init__(self, cfg: Qwen3_5MoeConfig):
        super().__init__()
        self.cfg = cfg
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.model.layers = nn.ModuleList(
            DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.model.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self._rope = None

    def _rope_for(self, device):
        # partial RoPE: only cfg.rotary_dim of head_dim is rotated (text mRoPE reduction)
        if self._rope is None:
            self._rope = RoPE(self.cfg.rotary_dim, self.cfg.rope_theta, device)
        return self._rope

    @torch.no_grad()
    def forward(self, input_ids, past=None, return_hidden=False):
        b, t = input_ids.shape
        # 1. EMBED.
        x = self.model.embed_tokens(input_ids)
        # 2. CACHE + POSITIONS — offset from the family counter, not a KV probe.
        cache = past if past is not None else Qwen3_5MoeCache(self.cfg.num_hidden_layers)
        past_len = cache.seen_tokens
        positions = torch.arange(past_len, past_len + t, device=input_ids.device)
        cos, sin = self._rope_for(input_ids.device).cos_sin(positions, x.dtype)
        # 3. DECODER STACK — each layer mixes tokens then runs its FFN (MoE or dense MLP).
        for i, layer in enumerate(self.model.layers):
            x = layer(x, cos, sin, cache, i, use_cache=True)
        # 4. FINAL NORM + 5. LM HEAD. (the pre-norm hidden is what the MTP head consumes)
        hidden = x
        cache.advance(t)
        normed = self.model.norm(hidden)
        logits = self.lm_head(normed)
        if return_hidden:
            return logits, cache, hidden
        return logits, cache
