"""
qwen3/blocks.py — Qwen3's primitive components.

Almost identical to Qwen2, with two attention changes:
  • QK-norm — RMSNorm applied to each head's Q and K (over head_dim) before RoPE,
    Qwen3's training stabilizer (replaces Qwen2's reliance on the QKV bias).
  • no bias on q/k/v/o projections (Qwen2 had bias on q/k/v).
RMSNorm / RoPE / SwiGLU MLP are the same.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3Config


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        d = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(d)


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


class Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.n_head, self.n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.n_rep, self.head_dim = cfg.n_rep, cfg.head_dim
        h, hd = cfg.hidden_size, cfg.head_dim
        self.q_proj = nn.Linear(h, self.n_head * hd, bias=False)   # Qwen3: NO bias
        self.k_proj = nn.Linear(h, self.n_kv * hd, bias=False)
        self.v_proj = nn.Linear(h, self.n_kv * hd, bias=False)
        self.o_proj = nn.Linear(self.n_head * hd, h, bias=False)
        self.q_norm = RMSNorm(hd, cfg.rms_norm_eps)                # QK-norm (NEW)
        self.k_norm = RMSNorm(hd, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, past_kv):
        b, t, _ = x.shape
        # 1. PROJECT to Q/K/V, split into heads (no bias).
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim)
        k = self.k_proj(x).view(b, t, self.n_kv, self.head_dim)
        v = self.v_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        # 2. QK-NORM — RMSNorm each head's Q,K over head_dim, then put heads first.
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        # 3. ROPE.
        q, k = RoPE.apply(q, k, cos, sin)
        # 4. KV CACHE.
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v)
        # 5. GQA EXPAND.
        k, v = _repeat_kv(k, self.n_rep), _repeat_kv(v, self.n_rep)
        # 6. ATTENTION — scaled dot-product, causal.
        #    Three cases: plain prefill (no cache) → is_causal; decode (one query,
        #    attends to all cached keys) → no mask; chunked prefill (q_len>1 WITH a
        #    cache) → SDPA's is_causal misaligns (it assumes q,k share a start), so
        #    build an explicit position mask offset by the cached length.
        q_len, total_k = q.shape[2], k.shape[2]
        if q_len > 1 and q_len != total_k:
            qpos = torch.arange(total_k - q_len, total_k, device=q.device)
            kpos = torch.arange(total_k, device=q.device)
            mask = (kpos[None, :] <= qpos[:, None])[None, None]
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=q_len > 1)
        # 7. MERGE heads + output projection.
        o = o.transpose(1, 2).reshape(b, t, self.n_head * self.head_dim)
        return self.o_proj(o), new_kv


class MLP(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
