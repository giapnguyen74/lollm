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
from .kv import Qwen2Cache


# Residual write-backs this family taps, in order — the site names DecoderLayer.forward
# passes to self.hook_fn (see the shared seam in src/hook.py). Consumed by
# modification/capture.py to enumerate site names for extraction.
SITES = ("post_attn", "out")


class DecoderLayer(nn.Module):
    """One transformer block: pre-norm residual attention, then pre-norm MLP."""

    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)
        # Hook function 
        self.hook_fn = None


    def forward(self, x, cos, sin, cache=None, layer_idx=0):
        # 1. ATTENTION sub-block (mixes info ACROSS tokens): x = x + attn(norm(x)).
        h = self.self_attn(self.input_layernorm(x), cos, sin, cache, layer_idx)
        x = x + h
        if self.hook_fn:
            x = self.hook_fn(x, "post_attn")
        # 2. MLP sub-block (transforms EACH token on its own): x = x + mlp(norm(x)).
        x = x + self.mlp(self.post_attention_layernorm(x))
        if self.hook_fn:
            x = self.hook_fn(x, "out")
        return x


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
        #    feed only the *new* tokens, so positions start after the cached length
        #    (read from the cache's own counter, not a tensor-shape probe).
        cache = past if past is not None else Qwen2Cache(self.cfg.num_hidden_layers)
        past_len = cache.seen_tokens
        positions = torch.arange(past_len, past_len + t, device=input_ids.device)
        #    Build the RoPE cos/sin tables for those positions (used in attention).
        cos, sin = self._rope_for(input_ids.device).cos_sin(positions, x.dtype)

        # 3. DECODER STACK — N layers of (attention + MLP); each reads & grows its slot
        #    through the cache's methods. This is the bulk of the model.
        for i, layer in enumerate(self.model.layers):
            x = layer(x, cos, sin, cache, i)
        cache.advance(t)

        # 4. FINAL NORM.
        x = self.model.norm(x)

        # 5. LM HEAD — project hidden states to one logit per vocab token. (B, T, V)
        return self.lm_head(x), cache
