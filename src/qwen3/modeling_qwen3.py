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


class DecoderLayer(nn.Module):
    """One transformer block: pre-norm residual attention, then pre-norm MLP."""

    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, past_kv):
        h, new_kv = self.self_attn(self.input_layernorm(x), cos, sin, past_kv)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_kv


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
        # 2. POSITIONS (offset by KV cache) + RoPE cos/sin.
        past_len = 0 if past is None else past[0][0].shape[2]
        positions = torch.arange(past_len, past_len + t, device=input_ids.device)
        cos, sin = self._rope_for(input_ids.device).cos_sin(positions, x.dtype)
        # 3. DECODER STACK.
        new_past = []
        for i, layer in enumerate(self.model.layers):
            pkv = None if past is None else past[i]
            x, kv = layer(x, cos, sin, pkv)
            new_past.append(kv)
        # 4. FINAL NORM.
        x = self.model.norm(x)
        # 5. LM HEAD.
        return self.lm_head(x), new_past
