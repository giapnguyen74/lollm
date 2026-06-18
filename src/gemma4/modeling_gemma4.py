"""
gemma4/modeling_gemma4.py — architecture: decoder layer + model + forward (text only).

Gemma4-specific shape (vs gemma3): the **PLE** pipeline (a 2nd embedding table + a
context projection, combined and injected once per layer after the FFN), **shared-KV**
threading (a per-forward dict carries the donor layer's K/V to the shared layers),
**plain RMSNorm**, **dual RoPE** with a *proportional* table on global layers, and a
**final logit soft-cap**. Sandwich norm and ×√hidden embedding carry over from gemma3.
No weight loading here — that's gemma4/weights.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Gemma4Config
from .blocks import (Gemma4RMSNorm, Gemma4Attention, Gemma4MLP, RoPE,
                     build_inv_freq_default, build_inv_freq_proportional)
from .kv import Gemma4Cache


class DecoderLayer(nn.Module):
    """Sandwich norm (4 norms) + a Per-Layer-Embedding residual block after the FFN."""

    def __init__(self, cfg: Gemma4Config, layer_idx: int):
        super().__init__()
        self.input_layernorm = Gemma4RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Gemma4Attention(cfg, layer_idx)
        self.post_attention_layernorm = Gemma4RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = Gemma4MLP(cfg.hidden_size, cfg.layer_intermediate(layer_idx))
        self.post_feedforward_layernorm = Gemma4RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        # PLE injection (one per layer, after the FFN residual).
        ple = cfg.hidden_size_per_layer_input
        self.per_layer_input_gate = nn.Linear(cfg.hidden_size, ple, bias=False)
        self.per_layer_projection = nn.Linear(ple, cfg.hidden_size, bias=False)
        self.post_per_layer_input_norm = Gemma4RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        # Per-layer residual scale (LayerScale-style): a saved buffer, NOT always 1.0.
        # Omitting it leaves each layer's output at the wrong magnitude, which corrupts
        # the residual-stream proportions of every later layer (parity dies by layer 1).
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, x, per_layer_input, cos, sin, cache, shared_kv):
        # 1. ATTENTION sub-block (sandwich): x = x + post_norm(attn(pre_norm(x))).
        h = self.self_attn(self.input_layernorm(x), cos, sin, cache, shared_kv)
        x = x + self.post_attention_layernorm(h)
        # 2. MLP sub-block (sandwich):        x = x + post_norm(mlp(pre_norm(x))).
        h = self.mlp(self.pre_feedforward_layernorm(x))
        x = x + self.post_feedforward_layernorm(h)
        # 3. PLE inject: x = x + post_norm( proj( gelu(gate(x)) * per_layer_input ) ).
        h = self.per_layer_input_gate(x)
        h = F.gelu(h, approximate="tanh") * per_layer_input
        h = self.per_layer_projection(h)
        x = x + self.post_per_layer_input_norm(h)
        # 4. Per-layer residual scale (applied to the whole layer output).
        return x * self.layer_scalar


class Gemma4Model(nn.Module):
    def __init__(self, cfg: Gemma4Config):
        super().__init__()
        self.cfg = cfg
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        # PLE tables (model level): a packed per-layer embedding + a context projection.
        ple, L = cfg.hidden_size_per_layer_input, cfg.num_hidden_layers
        self.model.embed_tokens_per_layer = nn.Embedding(cfg.vocab_size_per_layer_input, L * ple)
        self.model.per_layer_model_projection = nn.Linear(cfg.hidden_size, L * ple, bias=False)
        self.model.per_layer_projection_norm = Gemma4RMSNorm(ple, cfg.rms_norm_eps)
        self.model.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(L))
        self.model.norm = Gemma4RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self._rope = None
        self._embed_scale = cfg.hidden_size ** 0.5
        self._ple_scale = ple ** 0.5
        self._ple_combine = 2.0 ** -0.5
        self._ple_proj_scale = cfg.hidden_size ** -0.5

    def _ropes(self, device):
        if self._rope is None:
            local = RoPE(build_inv_freq_default(self.cfg.head_dim, self.cfg.rope_theta_local, device))
            glob = RoPE(build_inv_freq_proportional(
                self.cfg.global_head_dim, self.cfg.rope_theta_global,
                self.cfg.partial_rotary_factor_global, device))
            self._rope = {"sliding_attention": local, "full_attention": glob}
        return self._rope

    def _per_layer_inputs(self, input_ids, embed):
        """PLE: combine the token-identity lookup with the context projection of `embed`.
        Returns (B, T, num_layers, ple_dim)."""
        cfg = self.cfg
        ple, L = cfg.hidden_size_per_layer_input, cfg.num_hidden_layers
        identity = self.model.embed_tokens_per_layer(input_ids) * self._ple_scale
        identity = identity.view(*input_ids.shape, L, ple)
        proj = self.model.per_layer_model_projection(embed) * self._ple_proj_scale
        proj = proj.view(*input_ids.shape, L, ple)
        proj = self.model.per_layer_projection_norm(proj)
        return (proj + identity) * self._ple_combine

    @torch.no_grad()
    def forward(self, input_ids, past=None):
        b, t = input_ids.shape

        # 1. EMBED ×√hidden, then compute PLE (from token IDs, before any merge).
        x = self.model.embed_tokens(input_ids) * torch.tensor(
            self._embed_scale, dtype=self.model.embed_tokens.weight.dtype, device=input_ids.device)
        per_layer = self._per_layer_inputs(input_ids, x)        # (B,T,L,ple)

        # 2. POSITIONS + per-type RoPE (local default / global proportional).
        cache = past if past is not None else Gemma4Cache(self.cfg.num_hidden_layers)
        past_len = cache.seen_tokens
        positions = torch.arange(past_len, past_len + t, device=input_ids.device)
        ropes = self._ropes(input_ids.device)
        cossin = {lt: r.cos_sin(positions, x.dtype) for lt, r in ropes.items()}

        # 3. DECODER STACK — shared_kv carries the donor layer's K/V to the shared layers.
        shared_kv = {}
        for i, layer in enumerate(self.model.layers):
            cos, sin = cossin[self.cfg.layer_types[i]]
            x = layer(x, per_layer[:, :, i, :], cos, sin, cache, shared_kv)
        cache.advance(t)

        # 4. FINAL NORM.
        x = self.model.norm(x)

        # 5. LM HEAD + final-logit soft-cap (Gemma4 reintroduced it).
        logits = self.lm_head(x)
        sc = self.cfg.final_logit_softcapping
        if sc is not None:
            logits = torch.tanh(logits / sc) * sc
        return logits, cache
