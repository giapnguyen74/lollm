"""
gemma2/modeling_gemma2.py — the architecture: decoder layer + model + forward.

Assembles gemma2/blocks.py using gemma2/config.py. The Gemma2-specific shape lives
here: the SANDWICH norm layer (norm before AND after each sublayer), the embedding
scaling (×√hidden), the alternating local/global attention, and the final-logit
soft-cap. No weight loading here — that's gemma2/weights.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Gemma2Config
from .blocks import GemmaRMSNorm, RoPE, GemmaAttention, GemmaMLP
from .kv import Gemma2Cache


class DecoderLayer(nn.Module):
    """
    Gemma2 block: sandwich norm — a norm BEFORE and AFTER each sublayer (4 total).
    Even layers use sliding-window (local) attention, odd layers use global.
    """

    def __init__(self, cfg: Gemma2Config, layer_idx: int):
        super().__init__()
        is_sliding = (layer_idx % 2 == 0)
        window = cfg.sliding_window if is_sliding else None
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


class Gemma2Model(nn.Module):
    def __init__(self, cfg: Gemma2Config):
        super().__init__()
        self.cfg = cfg
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.model.layers = nn.ModuleList(
            DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.model.norm = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self._rope = None

    def _rope_for(self, device):
        if self._rope is None:
            self._rope = RoPE(self.cfg.head_dim, self.cfg.rope_theta, device)
        return self._rope

    @torch.no_grad()
    def forward(self, input_ids, past=None):
        # input_ids: (B, T) token ids — tokenized upstream.
        b, t = input_ids.shape

        # 1. EMBED, then SCALE by √hidden_size (Gemma normalizer).   (B,T) -> (B,T,H)
        x = self.model.embed_tokens(input_ids)
        x = x * torch.tensor(self.cfg.hidden_size ** 0.5, dtype=x.dtype, device=x.device)

        # 2. POSITIONS (offset by the cache's own counter) + RoPE cos/sin.
        cache = past if past is not None else Gemma2Cache(self.cfg.num_hidden_layers)
        past_len = cache.seen_tokens
        positions = torch.arange(past_len, past_len + t, device=input_ids.device)
        cos, sin = self._rope_for(input_ids.device).cos_sin(positions, x.dtype)

        # 3. DECODER STACK — alternating local (sliding) / global layers; each reads/grows
        #    its slot through the cache's methods.
        for i, layer in enumerate(self.model.layers):
            x = layer(x, cos, sin, cache, i)
        cache.advance(t)

        # 4. FINAL NORM.
        x = self.model.norm(x)

        # 5. LM HEAD, then FINAL-LOGIT soft-cap.                     (B, T, vocab)
        logits = self.lm_head(x)
        if self.cfg.final_logit_softcapping is not None:
            sc = self.cfg.final_logit_softcapping
            logits = torch.tanh(logits / sc) * sc
        return logits, cache
