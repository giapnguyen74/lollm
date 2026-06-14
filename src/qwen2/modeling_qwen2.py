"""
qwen2/modeling_qwen2.py — the architecture: decoder layer + model + forward.

Assembles the small components (qwen2/blocks.py) using the config (qwen2/config.py).
`DecoderLayer` shows the per-layer architecture (pre-norm residual: attention, then
MLP); `Qwen2Model` stacks them. No weight-loading / name-mapping here — that's the
loading concern in qwen2/weights.py. This file is "what the model is and how it runs."
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Qwen2Config
from .blocks import RMSNorm, RoPE, Attention, MLP


class DecoderLayer(nn.Module):
    """One transformer block: pre-norm residual attention, then pre-norm MLP."""

    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, past_kv):
        # 1. ATTENTION sub-block (mixes info ACROSS tokens): x = x + attn(norm(x)).
        h, new_kv = self.self_attn(self.input_layernorm(x), cos, sin, past_kv)
        x = x + h
        # 2. MLP sub-block (transforms EACH token on its own): x = x + mlp(norm(x)).
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_kv


class Qwen2Model(nn.Module):
    def __init__(self, cfg: Qwen2Config):
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
        # input_ids: (B, T) token ids — already tokenized upstream by the tokenizer.
        b, t = input_ids.shape

        # 1. EMBED — look up each token id's vector.            (B, T) -> (B, T, H)
        x = self.model.embed_tokens(input_ids)

        # 2. POSITIONS — where these tokens sit in the sequence. With a KV cache we
        #    feed only the *new* tokens, so positions start after the cached length.
        past_len = 0 if past is None else past[0][0].shape[2]
        positions = torch.arange(past_len, past_len + t, device=input_ids.device)
        #    Build the RoPE cos/sin tables for those positions (used in attention).
        cos, sin = self._rope_for(input_ids.device).cos_sin(positions, x.dtype)

        # 3. DECODER STACK — N layers of (attention + MLP); each reads & grows its
        #    own KV cache. This is the bulk of the model.
        new_past = []
        for i, layer in enumerate(self.model.layers):
            pkv = None if past is None else past[i]
            x, kv = layer(x, cos, sin, pkv)
            new_past.append(kv)

        # 4. FINAL NORM.
        x = self.model.norm(x)

        # 5. LM HEAD — project hidden states to one logit per vocab token. (B, T, V)
        return self.lm_head(x), new_past
