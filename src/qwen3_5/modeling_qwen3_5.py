"""
qwen3_5/modeling_qwen3_5.py — the architecture: decoder layer + model + forward.

Pre-norm residual decoder (attention/mixer then MLP), but HYBRID: each layer is either a
full-attention block (`self_attn`, every 4th) or a Gated DeltaNet linear block
(`linear_attn`). The two mixers keep different per-step state — a growing KV cache vs a
fixed-size (conv_state, recurrent_state) — so the family carries its own small cache
object and reads positions from a scalar token counter instead of `past[0][0].shape[2]`
(invalid here, since layer 0 has no KV). See docs/qwen3_5-architecture.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Qwen3_5Config
from .blocks import RMSNorm, RoPE, GatedAttention, GatedDeltaNet, MLP


class Qwen3_5Cache:
    """
    Family-local cache. `layers[i]` holds that layer's per-step state — a `(k, v)` KV tuple
    for full-attention layers, a `(conv_state, recurrent_state)` pair for linear layers —
    and `seen_tokens` counts processed positions (the position offset for RoPE). `generate`
    threads this object opaquely; only this family inspects it.
    """

    def __init__(self, n_layers: int):
        self.layers = [None] * n_layers
        self.seen_tokens = 0


class DecoderLayer(nn.Module):
    """Pre-norm residual block. Every 4th layer is full-attention (GQA, gated, partial
    RoPE); the rest are Gated DeltaNet (linear attention). `state` is this layer's slot in
    the family cache — KV tuple or (conv, recurrent) pair — and is threaded back out."""

    def __init__(self, cfg: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.layer_type = cfg.layer_type(layer_idx)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if self.layer_type == "full_attention":
            self.self_attn = GatedAttention(cfg)          # HF name: self_attn
        else:
            self.linear_attn = GatedDeltaNet(cfg)         # HF name: linear_attn
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, state, use_cache):
        h_in = self.input_layernorm(x)
        if self.layer_type == "full_attention":
            # full layers use RoPE + a growing KV cache (state = (k, v) or None)
            h, new_state = self.self_attn(h_in, cos, sin, state)
        else:
            # linear layers ignore RoPE; state = (conv_state, recurrent_state) or None
            conv_s, rec_s = (None, None) if state is None else state
            h, new_conv, new_rec = self.linear_attn(h_in, conv_s, rec_s, use_cache=use_cache)
            new_state = (new_conv, new_rec) if use_cache else None
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_state


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
    def forward(self, input_ids, past=None):
        b, t = input_ids.shape
        # 1. EMBED.
        x = self.model.embed_tokens(input_ids)
        # 2. CACHE + POSITIONS — offset from the family counter, not a KV probe.
        cache = past if past is not None else Qwen3_5Cache(self.cfg.num_hidden_layers)
        past_len = cache.seen_tokens
        positions = torch.arange(past_len, past_len + t, device=input_ids.device)
        cos, sin = self._rope_for(input_ids.device).cos_sin(positions, x.dtype)
        # 3. DECODER STACK — each layer reads/writes its own cache slot.
        for i, layer in enumerate(self.model.layers):
            x, cache.layers[i] = layer(x, cos, sin, cache.layers[i], use_cache=True)
        cache.seen_tokens = past_len + t
        # 4. FINAL NORM + 5. LM HEAD.
        x = self.model.norm(x)
        return self.lm_head(x), cache
