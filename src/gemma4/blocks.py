"""
gemma4/blocks.py — Gemma4's primitive components.

Diffs from gemma3/blocks.py (all parity-critical, from transformers modeling_gemma4):
  - **Gemma4RMSNorm** is plain `w·x̂` with ONES init (not `(1+w)`); a `with_scale=False`
    variant (used for v_norm) just normalizes.
  - **Attention scale = 1.0** (no query_pre_attn_scalar); q_norm/k_norm absorb scaling.
  - **v_norm** (no weight) normalizes V before attention; QK-norm on Q and K as before.
  - **Per-layer head_dim**: global (full) layers use `global_head_dim` (512), local
    (sliding) use `head_dim` (256).
  - **Proportional RoPE** on global layers: only the first `partial·head_dim/2` frequency
    pairs rotate; the rest are zero-padded (identity), keeping full head_dim width.
  - **Shared KV**: shared layers have no k/v projections — they reuse a donor layer's K/V.
The masked-softmax core is the shared `attention.torch_attention_with_scale` (scale=1.0).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import torch_attention_with_scale
from .config import Gemma4Config


class Gemma4RMSNorm(nn.Module):
    """Plain RMSNorm: x̂ · weight (ONES init), fp32. `with_scale=False` → x̂ only (v_norm)."""

    def __init__(self, dim, eps, with_scale=True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        d = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.with_scale:
            x = x * self.weight.float()
        return x.to(d)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def build_inv_freq_default(head_dim, theta, device):
    """Standard RoPE inv_freq over the full head_dim (length head_dim/2)."""
    idx = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    return 1.0 / (theta ** (idx / head_dim))


def build_inv_freq_proportional(head_dim, theta, partial, device):
    """Proportional (p-)RoPE: rotate only the first `partial·head_dim/2` frequency pairs,
    zero-pad the rest so the encoding stays full head_dim width (matches transformers
    `_compute_proportional_rope_parameters`)."""
    rope_angles = int(partial * head_dim // 2)
    idx = torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32, device=device)
    rotated = 1.0 / (theta ** (idx / head_dim))
    nope = head_dim // 2 - rope_angles
    if nope > 0:
        return torch.cat((rotated, torch.zeros(nope, dtype=torch.float32, device=device)))
    return rotated


class RoPE:
    """Holds one inv_freq table; `cos_sin` → (T, head_dim) cos/sin (emb = cat(freqs,freqs))."""

    def __init__(self, inv_freq):
        self.inv_freq = inv_freq

    def cos_sin(self, positions, dtype):
        freqs = positions[:, None].float() * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    @staticmethod
    def apply(x, cos, sin):                          # x:(B,H,T,D)  cos/sin:(T,D)
        cos, sin = cos[None, None], sin[None, None]
        return (x * cos) + (_rotate_half(x) * sin)


def _repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, t, d).reshape(b, n_kv * n_rep, t, d)


class Gemma4Attention(nn.Module):
    """
    GQA + QK-norm + V-norm, scale=1.0, per-layer head_dim, sliding/full mask, shared-KV.
    Shared layers (top `num_kv_shared_layers`) have NO k/v projections; they reuse the
    donor layer's K/V via the per-forward `shared_kv` dict (keyed by layer type).
    """

    def __init__(self, cfg: Gemma4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = cfg.layer_types[layer_idx]
        self.is_global = cfg.is_global(layer_idx)
        self.sliding_window = None if self.is_global else cfg.sliding_window
        self.head_dim = cfg.layer_head_dim(layer_idx)
        self.n_head, self.n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.n_rep = self.n_head // self.n_kv
        self.is_kv_shared = cfg.is_kv_shared(layer_idx)
        self.is_donor = cfg.is_donor(layer_idx)
        h, hd = cfg.hidden_size, self.head_dim
        self.q_proj = nn.Linear(h, self.n_head * hd, bias=False)
        self.q_norm = Gemma4RMSNorm(hd, cfg.rms_norm_eps)
        if not self.is_kv_shared:                     # shared layers store no k/v weights
            self.k_proj = nn.Linear(h, self.n_kv * hd, bias=False)
            self.v_proj = nn.Linear(h, self.n_kv * hd, bias=False)
            self.k_norm = Gemma4RMSNorm(hd, cfg.rms_norm_eps)
            self.v_norm = Gemma4RMSNorm(hd, cfg.rms_norm_eps, with_scale=False)
        self.o_proj = nn.Linear(self.n_head * hd, h, bias=False)

    def forward(self, x, cos, sin, cache, shared_kv):
        b, t, _ = x.shape
        # 1. Q — project, QK-norm, RoPE.
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        q = RoPE.apply(q, cos, sin)
        # 2. K/V — reuse donor's (shared layers) or compute + cache (non-shared).
        if self.is_kv_shared:
            k, v = shared_kv[self.layer_type]         # donor ran earlier this forward
        else:
            k = self.k_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
            k = self.k_norm(k)
            k = RoPE.apply(k, cos, sin)
            v = self.v_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
            v = self.v_norm(v)                         # V is normed but NOT roped
            if cache is not None:
                k, v = cache.append_kv(self.layer_idx, k, v)
            if self.is_donor:
                shared_kv[self.layer_type] = (k, v)   # hand full K/V to the shared layers
        # 3. GQA expand + masked attention (scale=1.0, no soft-cap; sliding band on local).
        kx, vx = _repeat_kv(k, self.n_rep), _repeat_kv(v, self.n_rep)
        o = torch_attention_with_scale(q, kx, vx, 1.0, self.sliding_window)
        # 4. Merge heads + output projection.
        o = o.transpose(1, 2).reshape(b, t, self.n_head * self.head_dim)
        return self.o_proj(o)


class Gemma4MLP(nn.Module):
    """GeGLU; intermediate size is double-wide on shared-KV layers (set by the caller)."""

    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
