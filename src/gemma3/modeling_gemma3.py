"""
gemma3/modeling_gemma3.py — the architecture: decoder layer + model + forward.

Assembles gemma3/blocks.py using gemma3/config.py. The Gemma3-specific shape lives
here: the SANDWICH norm layer (norm before AND after each sublayer), the embedding
scaling (×√hidden), and the alternating local/global attention with a DUAL RoPE
(local theta vs global theta). Unlike Gemma2 there is no final-logit soft-cap.
No weight loading here — that's gemma3/weights.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Gemma3Config
from .blocks import GemmaRMSNorm, RoPE, GemmaAttention, GemmaMLP
from .kv import Gemma3Cache


class DecoderLayer(nn.Module):
    """
    Gemma3 block: sandwich norm — a norm BEFORE and AFTER each sublayer (4 total).
    5 local : 1 global — most layers use sliding-window (local) attention; every
    `sliding_window_pattern`-th layer is global.
    """

    def __init__(self, cfg: Gemma3Config, layer_idx: int):
        super().__init__()
        self.is_global = cfg.is_global(layer_idx)
        window = None if self.is_global else cfg.sliding_window
        self.input_layernorm = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = GemmaAttention(cfg, sliding_window=window)
        self.post_attention_layernorm = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = GemmaMLP(cfg)
        self.post_feedforward_layernorm = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, cache=None, layer_idx=0):
        # 1. ATTENTION sub-block (sandwich): x = x + post_norm(attn(pre_norm(x))).
        h = self.self_attn(self.input_layernorm(x), cos, sin, cache, layer_idx)
        x = x + self.post_attention_layernorm(h)
        # 2. MLP sub-block (sandwich):        x = x + post_norm(mlp(pre_norm(x))).
        h = self.mlp(self.pre_feedforward_layernorm(x))
        x = x + self.post_feedforward_layernorm(h)
        return x


class Gemma3Model(nn.Module):
    def __init__(self, cfg: Gemma3Config):
        super().__init__()
        self.cfg = cfg
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.model.layers = nn.ModuleList(
            DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.model.norm = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self._rope_local = None
        self._rope_global = None

    def _ropes_for(self, device):
        if self._rope_local is None:
            # Two tables, built once: local (small theta, no scale) and global
            # (large theta, optional long-context linear scale).
            self._rope_local = RoPE(self.cfg.head_dim, self.cfg.rope_theta_local, device)
            self._rope_global = RoPE(self.cfg.head_dim, self.cfg.rope_theta_global, device,
                                     scaling_factor=self.cfg.rope_scaling_factor)
        return self._rope_local, self._rope_global

    @torch.no_grad()
    def forward(self, input_ids, past=None):
        # input_ids: (B, T) token ids — tokenized upstream.
        b, t = input_ids.shape

        # 1. EMBED, then SCALE by √hidden_size (Gemma normalizer).   (B,T) -> (B,T,H)
        x = self.model.embed_tokens(input_ids)
        x = x * torch.tensor(self.cfg.hidden_size ** 0.5, dtype=x.dtype, device=x.device)

        # 2. POSITIONS (offset by the cache's own counter) + RoPE cos/sin for BOTH
        #    the local and global tables (each layer picks the one it needs).
        cache = past if past is not None else Gemma3Cache(self.cfg.num_hidden_layers)
        past_len = cache.seen_tokens
        positions = torch.arange(past_len, past_len + t, device=input_ids.device)
        rl, rg = self._ropes_for(input_ids.device)
        cos_l, sin_l = rl.cos_sin(positions, x.dtype)
        cos_g, sin_g = rg.cos_sin(positions, x.dtype)

        # 3. DECODER STACK — local (sliding) layers use the local RoPE, global layers
        #    the global RoPE; each reads/grows its slot through the cache.
        for i, layer in enumerate(self.model.layers):
            if layer.is_global:
                x = layer(x, cos_g, sin_g, cache, i)
            else:
                x = layer(x, cos_l, sin_l, cache, i)
        cache.advance(t)

        # 4. FINAL NORM.
        x = self.model.norm(x)

        # 5. LM HEAD — no final-logit soft-cap in Gemma3.           (B, T, vocab)
        logits = self.lm_head(x)
        return logits, cache
