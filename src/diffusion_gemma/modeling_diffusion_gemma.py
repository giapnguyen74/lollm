"""
modeling_diffusion_gemma.py — the diffusion_gemma backbone.

Rung 1 (encoder / causal): `EncoderTextModel` = embed×√hidden → 30× causal layer → norm.
Rung 2 (decoder / denoise): `DecoderTextModel` = same tied backbone, but bidirectional canvas
attention that READS the encoder KV cache (cross-attention by concat) + a front-loaded
self-conditioning step. One denoise *pass* (no iteration/sampler — that's rungs 3–4).

The encoder and decoder share all common weights (HF ties them); only `self_conditioning` and the
lm-head are decoder-unique. Module/param names mirror transformers so the weight load is identity.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DiffusionGemmaConfig
from blocks import (Attention, DecoderAttention, DenseMLP, Experts, RMSNorm, RotaryEmbedding,
                    Router, SelfConditioning, default_inv_freq, proportional_inv_freq)


class _BackboneLayer(nn.Module):
    """Shared layer body: sandwich-normed attention slot + parallel dense-MLP/MoE FFN.
    Subclasses supply the attention module + how it's called (causal vs decoder/bidirectional)."""

    def __init__(self, cfg: DiffusionGemmaConfig, layer_idx: int, attn_cls):
        super().__init__()
        self.self_attn = attn_cls(cfg, layer_idx)
        self.mlp = DenseMLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.router = Router(cfg)
        self.experts = Experts(cfg)
        self.post_feedforward_layernorm_1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_feedforward_layernorm_2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.pre_feedforward_layernorm_2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

    def _ffn(self, x):
        # dense MLP and routed MoE run in PARALLEL off the same residual, then summed
        residual = x
        h = self.mlp(self.pre_feedforward_layernorm(x))
        hidden_1 = self.post_feedforward_layernorm_1(h)
        flat = residual.reshape(-1, residual.shape[-1])
        w, idx = self.router(flat)                               # routes on the raw residual
        h2 = self.experts(self.pre_feedforward_layernorm_2(flat), idx, w)
        h2 = self.post_feedforward_layernorm_2(h2.reshape(residual.shape))
        x = residual + self.post_feedforward_layernorm(hidden_1 + h2)
        return x * self.layer_scalar


class EncoderLayer(_BackboneLayer):
    def __init__(self, cfg, layer_idx):
        super().__init__(cfg, layer_idx, Attention)

    def forward(self, x, cos, sin, past_kv=None, return_kv=False):
        residual = x
        a = self.self_attn(self.input_layernorm(x), cos, sin, past_kv=past_kv, return_kv=return_kv)
        a, kv = a if return_kv else (a, None)
        x = residual + self.post_attention_layernorm(a)
        x = self._ffn(x)
        return (x, kv) if return_kv else x


class DecoderLayer(_BackboneLayer):
    def __init__(self, cfg, layer_idx):
        super().__init__(cfg, layer_idx, DecoderAttention)

    def forward(self, x, cos, sin, enc_k, enc_v):
        residual = x
        a = self.self_attn(self.input_layernorm(x), cos, sin, enc_k, enc_v)
        x = residual + self.post_attention_layernorm(a)
        return self._ffn(x)


class _RopePair(nn.Module):
    """Holds the two RoPE tables (sliding default / full proportional)."""

    def __init__(self, cfg: DiffusionGemmaConfig):
        super().__init__()
        self.cfg = cfg
        self.rope_sliding = RotaryEmbedding(default_inv_freq(cfg.rope_theta_local, cfg.head_dim))
        self.rope_full = RotaryEmbedding(
            proportional_inv_freq(cfg.rope_theta_global, cfg.global_head_dim,
                                  cfg.partial_rotary_factor_global))

    def cossin(self, x, pos):
        return self.rope_sliding(x, pos), self.rope_full(x, pos)


class EncoderTextModel(_RopePair):
    def __init__(self, cfg: DiffusionGemmaConfig):
        super().__init__(cfg)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=0)
        self.embed_scale = math.sqrt(cfg.hidden_size)
        self.layers = nn.ModuleList(EncoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, input_ids, past_cache=None, position_ids=None, return_cache=False):
        """Causal encode. `past_cache` (per-layer (k,v)) + `position_ids` give INCREMENTAL encoding:
        encode new tokens at offset positions, appending to the cache (re-encoding committed blocks)."""
        x = self.embed_tokens(input_ids) * self.embed_scale
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)[None]
        cs_s, cs_f = self.cossin(x, position_ids)
        cache = []
        for i, layer in enumerate(self.layers):
            cos, sin = cs_s if self.cfg.is_sliding(i) else cs_f
            past_kv = past_cache[i] if past_cache is not None else None
            if return_cache:
                x, kv = layer(x, cos, sin, past_kv=past_kv, return_kv=True)
                cache.append(kv)                                # accumulated (k,v) pre-repeat, per layer
            else:
                x = layer(x, cos, sin, past_kv=past_kv)
        x = self.norm(x)
        return (x, cache) if return_cache else x


class DecoderTextModel(_RopePair):
    def __init__(self, cfg: DiffusionGemmaConfig):
        super().__init__(cfg)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=0)
        self.embed_scale = math.sqrt(cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_conditioning = SelfConditioning(cfg)
        self.final_logit_softcapping = cfg.final_logit_softcapping

    def to_logits(self, hidden):
        """lm_head (tied to embed_tokens) + Gemma final-logit soft-cap, in fp32."""
        logits = F.linear(hidden, self.embed_tokens.weight).float()
        sc = self.final_logit_softcapping
        return torch.tanh(logits / sc) * sc

    def forward(self, canvas_ids, encoder_cache, self_conditioning_logits=None):
        emb = self.embed_tokens(canvas_ids) * self.embed_scale
        # self-conditioning: fold in last step's prediction (zeros on the first step)
        if self_conditioning_logits is not None:
            probs = self_conditioning_logits.softmax(dim=-1, dtype=torch.float32).to(emb.dtype)
            soft = torch.matmul(probs, self.embed_tokens.weight) * self.embed_scale
        else:
            soft = torch.zeros_like(emb)
        x = self.self_conditioning(emb, soft)                   # always applied (post_norm)

        # canvas positions continue AFTER the encoder sequence
        s_enc = encoder_cache[0][0].shape[2]
        pos = torch.arange(s_enc, s_enc + canvas_ids.shape[1], device=canvas_ids.device)[None]
        cs_s, cs_f = self.cossin(x, pos)
        for i, layer in enumerate(self.layers):
            cos, sin = cs_s if self.cfg.is_sliding(i) else cs_f
            enc_k, enc_v = encoder_cache[i]
            x = layer(x, cos, sin, enc_k, enc_v)
        return self.norm(x)
