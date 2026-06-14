"""
qwen2/blocks.py — Qwen2's primitive components (small, detail implementations).

RMSNorm, RoPE, GQA attention (with a KV cache), and the SwiGLU MLP. These are the
small building blocks; modeling_qwen2.py assembles them into the decoder layer and
the model. Deliberately not shared with other families — duplication is intentional.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen2Config


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
    """Precomputes inv_freq; produces cos/sin and applies the rotation (HF layout)."""

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
    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.n_head, self.n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.n_rep, self.head_dim = cfg.n_rep, cfg.head_dim
        h, hd = cfg.hidden_size, cfg.head_dim
        self.q_proj = nn.Linear(h, self.n_head * hd, bias=True)   # Qwen2: q/k/v have bias
        self.k_proj = nn.Linear(h, self.n_kv * hd, bias=True)
        self.v_proj = nn.Linear(h, self.n_kv * hd, bias=True)
        self.o_proj = nn.Linear(self.n_head * hd, h, bias=False)

    def forward(self, x, cos, sin, past_kv):
        b, t, _ = x.shape
        # 1. PROJECT to queries/keys/values, split into heads.
        #    GQA: fewer KV heads (n_kv) than query heads (n_head).
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        # 2. ROPE — rotate Q and K by their positions.
        q, k = RoPE.apply(q, k, cos, sin)
        # 3. KV CACHE — append this step's K,V to the cached past (then keep it).
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v)
        # 4. GQA EXPAND — repeat KV heads to match the query head count.
        k, v = _repeat_kv(k, self.n_rep), _repeat_kv(v, self.n_rep)
        # 5. ATTENTION — scaled dot-product, causal.
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
        # 6. MERGE heads and project back to hidden size.
        o = o.transpose(1, 2).reshape(b, t, self.n_head * self.head_dim)
        return self.o_proj(o), new_kv


class MLP(nn.Module):
    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
