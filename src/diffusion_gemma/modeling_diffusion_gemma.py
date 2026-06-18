"""
modeling_diffusion_gemma.py — the diffusion_gemma backbone.

There is **ONE** Gemma-lineage backbone (one set of weights — the checkpoint does NOT ship separate
encoder/decoder tensors; the reference ties them). It is run in two ROLES over the same weights:

  • `prefill(...)` — causal encode of the prompt (and each committed block) → grows the KV cache.
                     This is the "encoder" role: a normal Gemma causal forward.
  • `denoise(...)` — bidirectional pass over a canvas, reading the read-only encoder cache by concat,
                     with a front-loaded self-conditioning step. This is the "decoder" role.

The only weights used by `denoise` but not `prefill` are the `self_conditioning` block (and the tied
lm-head, via `to_logits`). Param names mirror transformers so the checkpoint loads identity.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DiffusionGemmaConfig
from blocks import (Attention, DenseMLP, Experts, RMSNorm, RotaryEmbedding, Router, SelfConditioning,
                    default_inv_freq, proportional_inv_freq)


class Layer(nn.Module):
    """One backbone layer: sandwich-normed attention + parallel dense-MLP / routed-MoE FFN. The SAME
    weights run in two modes — `encode` (causal, write-cache) and `denoise` (bidirectional, read-cache)."""

    def __init__(self, cfg: DiffusionGemmaConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(cfg, layer_idx)
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

    def encode(self, x, cos, sin, past_kv=None, return_kv=False):
        residual = x
        a = self.self_attn.causal(self.input_layernorm(x), cos, sin, past_kv=past_kv, return_kv=return_kv)
        a, kv = a if return_kv else (a, None)
        x = residual + self.post_attention_layernorm(a)
        x = self._ffn(x)
        return (x, kv) if return_kv else x

    def denoise(self, x, cos, sin, enc_k, enc_v):
        residual = x
        a = self.self_attn.cross(self.input_layernorm(x), cos, sin, enc_k, enc_v)
        x = residual + self.post_attention_layernorm(a)
        return self._ffn(x)


class DiffusionGemmaModel(nn.Module):
    """The whole text backbone. Build once, run in two roles: `prefill` (causal) and `denoise`
    (bidirectional). Holds the two RoPE tables, the embedding (tied to the lm-head), the layer stack,
    the final norm, and the decoder-only self-conditioning block."""

    def __init__(self, cfg: DiffusionGemmaConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=0)
        self.embed_scale = math.sqrt(cfg.hidden_size)
        self.layers = nn.ModuleList(Layer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_conditioning = SelfConditioning(cfg)
        self.final_logit_softcapping = cfg.final_logit_softcapping
        # two RoPE tables (non-persistent buffers), picked per layer type
        self.rope_sliding = RotaryEmbedding(default_inv_freq(cfg.rope_theta_local, cfg.head_dim))
        self.rope_full = RotaryEmbedding(
            proportional_inv_freq(cfg.rope_theta_global, cfg.global_head_dim, cfg.partial_rotary_factor_global))

    def _cossin(self, x, pos):
        return self.rope_sliding(x, pos), self.rope_full(x, pos)

    def to_logits(self, hidden):
        """lm_head (tied to embed_tokens) + Gemma final-logit soft-cap, in fp32."""
        logits = F.linear(hidden, self.embed_tokens.weight).float()
        sc = self.final_logit_softcapping
        return torch.tanh(logits / sc) * sc

    def prefill(self, input_ids, past_cache=None, position_ids=None, return_cache=True):
        """Causal encode (the 'encoder' role). `past_cache` + `position_ids` give INCREMENTAL encoding:
        encode new tokens at offset positions, appending to the cache (re-encoding committed blocks).
        Sliding layers clip their cache to the last `window-1` K/V (matches the reference; bounded memory)."""
        x = self.embed_tokens(input_ids) * self.embed_scale
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)[None]
        cs_s, cs_f = self._cossin(x, position_ids)
        cache = []
        w = self.cfg.sliding_window - 1
        for i, layer in enumerate(self.layers):
            cos, sin = cs_s if self.cfg.is_sliding(i) else cs_f
            past_kv = past_cache[i] if past_cache is not None else None
            if return_cache:
                x, kv = layer.encode(x, cos, sin, past_kv=past_kv, return_kv=True)
                if self.cfg.is_sliding(i):                       # clip sliding cache to the window
                    kv = (kv[0][..., -w:, :].contiguous(), kv[1][..., -w:, :].contiguous())
                cache.append(kv)
            else:
                x = layer.encode(x, cos, sin, past_kv=past_kv)
        x = self.norm(x)
        return (x, cache) if return_cache else x

    def denoise(self, canvas_ids, encoder_cache, self_conditioning_logits=None):
        """Bidirectional canvas pass (the 'decoder' role), reading the read-only encoder cache."""
        emb = self.embed_tokens(canvas_ids) * self.embed_scale
        # self-conditioning: fold in last step's prediction (zeros on the first step)
        if self_conditioning_logits is not None:
            probs = self_conditioning_logits.softmax(dim=-1, dtype=torch.float32).to(emb.dtype)
            soft = torch.matmul(probs, self.embed_tokens.weight) * self.embed_scale
        else:
            soft = torch.zeros_like(emb)
        x = self.self_conditioning(emb, soft)                   # always applied (post_norm)

        # canvas positions continue AFTER the encoder sequence. Use a GLOBAL layer's cache for the true
        # length — sliding layers are clipped to the window, so cache[0] would under-count.
        s_enc = encoder_cache[self.cfg.first_global_layer][0].shape[2]
        pos = torch.arange(s_enc, s_enc + canvas_ids.shape[1], device=canvas_ids.device)[None]
        cs_s, cs_f = self._cossin(x, pos)
        for i, layer in enumerate(self.layers):
            cos, sin = cs_s if self.cfg.is_sliding(i) else cs_f
            enc_k, enc_v = encoder_cache[i]
            x = layer.denoise(x, cos, sin, enc_k, enc_v)
        return self.norm(x)
