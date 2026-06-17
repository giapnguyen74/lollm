"""
gemma3/blocks.py — Gemma3's primitive components (small, detail implementations).

Diff from gemma2/blocks.py:
  - QK-norm: a GemmaRMSNorm over head_dim applied to Q and K *after* the head
    reshape and *before* RoPE. This is Gemma3's headline change — it replaced
    Gemma2's attention logit soft-capping.
  - No soft-capping anywhere (attention or final logits).
Unchanged from Gemma2: (1+w) RMSNorm, GeGLU MLP, custom attention scale
(query_pre_attn_scalar**-0.5), and the manual sliding-window mask on local layers.
modeling_gemma3.py assembles these into the layer + model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import torch_attention_with_scale
from .config import Gemma3Config


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
    """A rotary table for one theta. `scaling_factor` linearly stretches positions
    (long-context scaling on global layers); 1.0 = no scaling."""

    def __init__(self, head_dim, theta, device, scaling_factor=1.0):
        idx = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        self.inv_freq = 1.0 / (theta ** (idx / head_dim))
        self.scaling_factor = scaling_factor

    def cos_sin(self, positions, dtype):
        pos = positions.float() / self.scaling_factor
        freqs = pos[:, None] * self.inv_freq[None, :]
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
    GQA + QK-norm + RoPE. Gemma3 specifics: scale = query_pre_attn_scalar**-0.5,
    QK-norm (RMSNorm over head_dim on Q,K before RoPE), and a sliding-window mask on
    local layers. Manual attention (no SDPA) so the window mask is explicit and the
    path reads top to bottom. `sliding_window=None` → global (full causal) layer.
    """

    def __init__(self, cfg: Gemma3Config, sliding_window):
        super().__init__()
        self.n_head, self.n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.n_rep, self.head_dim = cfg.n_rep, cfg.head_dim
        self.scaling = cfg.query_pre_attn_scalar ** -0.5
        self.sliding_window = sliding_window
        h, hd = cfg.hidden_size, cfg.head_dim
        self.q_proj = nn.Linear(h, self.n_head * hd, bias=False)   # Gemma: no bias
        self.k_proj = nn.Linear(h, self.n_kv * hd, bias=False)
        self.v_proj = nn.Linear(h, self.n_kv * hd, bias=False)
        self.o_proj = nn.Linear(self.n_head * hd, h, bias=False)
        self.q_norm = GemmaRMSNorm(hd, cfg.rms_norm_eps)           # NEW (QK-norm)
        self.k_norm = GemmaRMSNorm(hd, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, cache=None, layer_idx=0):
        b, t, _ = x.shape
        # 1. PROJECT to Q/K/V, split into heads (GQA: fewer KV heads).
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        # 2. QK-NORM — RMSNorm over head_dim on Q and K (Gemma3), before RoPE.
        q, k = self.q_norm(q), self.k_norm(k)
        # 3. ROPE.
        q, k = RoPE.apply(q, k, cos, sin)
        # 4. KV CACHE — append this step's K,V, read back the full K,V.
        #    (Local layers cache the FULL K,V too; the window is applied in the mask.)
        if cache is not None:
            k, v = cache.append_kv(layer_idx, k, v)
        # 5. GQA EXPAND.
        k, v = _repeat_kv(k, self.n_rep), _repeat_kv(v, self.n_rep)
        # 6. ATTENTION — Gemma scale + (local) sliding-window band mask, no soft-cap.
        #    Shared helper (fp32 softmax); see attention.torch_attention_with_scale.
        o = torch_attention_with_scale(q, k, v, self.scaling, self.sliding_window)
        # 7. MERGE heads + output projection.
        o = o.transpose(1, 2).reshape(b, t, self.n_head * self.head_dim)
        return self.o_proj(o)


class GemmaMLP(nn.Module):
    """GeGLU: down( gelu_tanh(gate(x)) * up(x) ) — approximate-gelu gate, not SiLU."""

    def __init__(self, cfg: Gemma3Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
