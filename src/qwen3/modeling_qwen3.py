"""
qwen3/modeling_qwen3.py — the architecture: decoder layer + model + forward.

Same shape as Qwen2 (pre-norm residual: attention then MLP, stacked, final norm,
LM head). The Qwen3-specific bits (QK-norm, no bias) live in blocks.Attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Qwen3Config
from .blocks import RMSNorm, RoPE, Attention, MLP
from .kv import Qwen3Cache


class DecoderLayer(nn.Module):
    """One transformer block: pre-norm residual attention, then pre-norm MLP."""

    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cache=None, layer_idx=0):
        h = self.self_attn(self.input_layernorm(x), cos, sin, cache, layer_idx)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.model.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.model.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self._rope = None

    def _rope_for(self, device):
        if self._rope is None:
            self._rope = RoPE(self.cfg.head_dim, self.cfg.rope_theta, device)
        return self._rope

    @torch.no_grad()
    def forward(self, input_ids, past=None):
        b, t = input_ids.shape
        # 1. EMBED — token ids → vectors.
        x = self.model.embed_tokens(input_ids)
        # 2. POSITIONS (offset by the cache's own counter) + RoPE cos/sin.
        cache = past if past is not None else Qwen3Cache(self.cfg.num_hidden_layers)
        past_len = cache.seen_tokens
        positions = torch.arange(past_len, past_len + t, device=input_ids.device)
        cos, sin = self._rope_for(input_ids.device).cos_sin(positions, x.dtype)
        # 3. DECODER STACK — each layer reads/grows its slot through the cache's methods.
        for i, layer in enumerate(self.model.layers):
            x = layer(x, cos, sin, cache, i)
        cache.advance(t)
        # 4. FINAL NORM.
        x = self.model.norm(x)
        # 5. LM HEAD.
        return self.lm_head(x), cache
