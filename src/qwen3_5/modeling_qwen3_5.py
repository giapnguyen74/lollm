"""
qwen3_5/modeling_qwen3_5.py — the architecture: decoder layer + model + forward.

Pre-norm residual decoder (attention/mixer then MLP), but HYBRID: each layer is either a
full-attention block (`self_attn`, every 4th) or a Gated DeltaNet linear block
(`linear_attn`). The two mixers keep different per-step state — a growing KV cache vs a
fixed-size (conv_state, recurrent_state) — so the family carries its own cache object
(`Qwen3_5Cache`, in kv.py) and reads positions from its `seen_tokens` counter instead of
`past[0][0].shape[2]` (invalid here, since layer 0 has no KV). The model talks to the cache
only through methods (`append_kv` / `linear_state` / `advance`), which leaves room to swap
the KV storage for something fancier (paged/quantized) later without touching this file.
See docs/qwen3_5-architecture.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Qwen3_5Config
from .blocks import RMSNorm, RoPE, GatedAttention, GatedDeltaNet, MLP
from .kv import Qwen3_5Cache          # cache lives in kv.py (model talks to it via methods)

__all__ = ["DecoderLayer", "Qwen3_5Model", "Qwen3_5Cache"]


class DecoderLayer(nn.Module):
    """Pre-norm residual block. Every 4th layer is full-attention (GQA, gated, partial
    RoPE); the rest are Gated DeltaNet (linear attention). The layer reads/writes its slot
    through the cache's methods (`append_kv` / `linear_state`) — never the cache internals —
    so the KV storage policy is swappable without touching this code. `cache=None` runs an
    uncached one-shot pass (used by the MTP head)."""

    def __init__(self, cfg: Qwen3_5Config, layer_idx: int, layer_type: str | None = None):
        super().__init__()
        # `layer_type` override lets the MTP head reuse this block as a forced
        # full-attention layer without depending on its position in the schedule.
        self.layer_type = layer_type or cfg.layer_type(layer_idx)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if self.layer_type == "full_attention":
            self.self_attn = GatedAttention(cfg)          # HF name: self_attn
        else:
            self.linear_attn = GatedDeltaNet(cfg)         # HF name: linear_attn
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cache=None, layer_idx=0, use_cache=True):
        h_in = self.input_layernorm(x)
        if self.layer_type == "full_attention":
            # full layers use RoPE + a growing KV cache (owned by the cache object)
            h = self.self_attn(h_in, cos, sin, cache, layer_idx)
        else:
            # linear layers ignore RoPE; read/update the fixed (conv, recurrent) slot
            do_cache = use_cache and cache is not None
            conv_s, rec_s = cache.linear_state(layer_idx) if cache is not None else (None, None)
            h, new_conv, new_rec = self.linear_attn(h_in, conv_s, rec_s, use_cache=do_cache)
            if do_cache:
                cache.set_linear_state(layer_idx, new_conv, new_rec)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3_5Model(nn.Module):
    def __init__(self, cfg: Qwen3_5Config):
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
        cache = past if past is not None else Qwen3_5Cache(self.cfg.num_hidden_layers)
        past_len = cache.seen_tokens
        positions = torch.arange(past_len, past_len + t, device=input_ids.device)
        cos, sin = self._rope_for(input_ids.device).cos_sin(positions, x.dtype)
        # 3. DECODER STACK — each layer reads/writes its slot through the cache's methods.
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
