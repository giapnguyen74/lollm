"""
gemma2/blocks.py — Gemma2's primitive components (small, detail implementations).

Differs from Qwen2 in the details: RMSNorm scales by (1 + weight); the MLP is GeGLU
(approximate gelu gate); attention uses a custom scale + logit soft-capping and (on
local layers) a sliding-window mask — so it can't use SDPA's `is_causal`, it does
attention manually. modeling_gemma2.py assembles these into the layer + model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Gemma2Config


class GemmaRMSNorm(nn.Module):
    """Normalize, then scale by (1 + weight), in fp32. (weight is zero-initialized)"""

    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        d = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * (1.0 + self.weight.float())).to(d)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RoPE:
    def __init__(self, head_dim, theta, device):
        idx = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        self.inv_freq = 1.0 / (theta ** (idx / head_dim))

    def cos_sin(self, positions, dtype):
        freqs = positions[:, None].float() * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    @staticmethod
    def apply(q, k, cos, sin):
        cos, sin = cos[None, None], sin[None, None]
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, t, d).reshape(b, n_kv * n_rep, t, d)


class GemmaAttention(nn.Module):
    """
    GQA + RoPE, with Gemma2's extras: scale = query_pre_attn_scalar**-0.5, attention
    logit soft-capping, and (on local layers) a sliding-window mask. Done manually
    because SDPA can't soft-cap. `sliding_window=None` → global (full causal) layer.
    """

    def __init__(self, cfg: Gemma2Config, sliding_window):
        super().__init__()
        self.n_head, self.n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.n_rep, self.head_dim = cfg.n_rep, cfg.head_dim
        self.scaling = cfg.query_pre_attn_scalar ** -0.5
        self.attn_softcap = cfg.attn_logit_softcapping
        self.sliding_window = sliding_window
        h, hd = cfg.hidden_size, cfg.head_dim
        self.q_proj = nn.Linear(h, self.n_head * hd, bias=False)   # Gemma: no bias
        self.k_proj = nn.Linear(h, self.n_kv * hd, bias=False)
        self.v_proj = nn.Linear(h, self.n_kv * hd, bias=False)
        self.o_proj = nn.Linear(self.n_head * hd, h, bias=False)

    def forward(self, x, cos, sin, past_kv):
        b, t, _ = x.shape
        # 1. PROJECT to Q/K/V, split into heads (GQA: fewer KV heads).
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        # 2. ROPE.
        q, k = RoPE.apply(q, k, cos, sin)
        # 3. KV CACHE — append to the past.
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v)
        # 4. GQA EXPAND.
        k, v = _repeat_kv(k, self.n_rep), _repeat_kv(v, self.n_rep)
        # 5. SCORES — Gemma scale, then attention logit soft-cap.
        scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        if self.attn_softcap is not None:
            scores = torch.tanh(scores / self.attn_softcap) * self.attn_softcap
        # 6. MASK — causal, plus a sliding window on local layers.
        total_k = k.shape[2]
        past_len = total_k - t
        qpos = torch.arange(past_len, total_k, device=x.device)
        kpos = torch.arange(0, total_k, device=x.device)
        allowed = kpos[None, :] <= qpos[:, None]                       # causal
        if self.sliding_window is not None:
            allowed = allowed & ((qpos[:, None] - kpos[None, :]) < self.sliding_window)
        scores = scores.masked_fill(~allowed[None, None], float("-inf"))
        # 7. SOFTMAX (in fp32) + weighted sum.
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        o = torch.matmul(attn, v)
        # 8. MERGE heads + output projection.
        o = o.transpose(1, 2).reshape(b, t, self.n_head * self.head_dim)
        return self.o_proj(o), new_kv


class GemmaMLP(nn.Module):
    """GeGLU: down( gelu_tanh(gate(x)) * up(x) ) — approximate-gelu gate, not SiLU."""

    def __init__(self, cfg: Gemma2Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
