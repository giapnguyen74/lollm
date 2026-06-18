"""
blocks.py — diffusion_gemma small components (rung 1: the causal/encoder path).

Gemma4-lineage primitives, reimplemented to match `transformers` DiffusionGemma exactly
(verified by module-parity at tiny config — see _parity_backbone.py). Names mirror the HF
module names so the weight load is identity.

Confirmed-from-source quirks baked in here:
  • RMSNorm = plain w·x̂, ones init, fp32 reduction; `with_scale=False` → no weight (v_norm, router).
  • Attention scale = 1.0 (QK-norm absorbs it). Tensor layout (B,S,H,D) then transpose to (B,H,S,D).
  • Global (full_attention) layers have NO v_proj → V = v_norm(k_proj(x)) (pre-norm, pre-RoPE).
  • Dual RoPE: sliding = default formula; full = "proportional" (rope_angles rotated + NoPE zeros).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DiffusionGemmaConfig

ACT = {"gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh")}


# ───────────────────────── norms ─────────────────────────

class RMSNorm(nn.Module):
    """Plain w·x̂ (ones init), fp32 reduction. with_scale=False → no weight (returns x̂)."""

    def __init__(self, dim, eps=1e-6, with_scale=True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        out = x.float() * torch.pow(x.float().pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.with_scale:
            out = out * self.weight.float()
        return out.type_as(x)


# ───────────────────────── RoPE (dual: default + proportional) ─────────────────────────

def default_inv_freq(theta: float, dim: int) -> torch.Tensor:
    """Standard RoPE: 1/θ^(arange(0,dim,2)/dim) → dim/2 freqs, all rotated."""
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))


def proportional_inv_freq(theta: float, head_dim: int, partial: float) -> torch.Tensor:
    """Proportional RoPE (transformers `_compute_proportional_rope_parameters`):
    rotate the first `rope_angles` freqs, zero-pad the rest (NoPE) → length head_dim/2."""
    rope_angles = int(partial * head_dim // 2)
    rotated = 1.0 / (theta ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).float() / head_dim))
    nope = head_dim // 2 - rope_angles
    if nope > 0:
        return torch.cat([rotated, torch.zeros(nope)], dim=0)
    return rotated


class RotaryEmbedding(nn.Module):
    """Builds cos/sin for one layer type from its inv_freq. Matches HF forward exactly."""

    def __init__(self, inv_freq: torch.Tensor):
        super().__init__()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos = position_ids[:, None, :].float()
        freqs = (inv @ pos).transpose(1, 2)            # (B, S, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)        # (B, S, dim)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin, unsqueeze_dim=2):
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


# ───────────────────────── attention (encoder / causal) ─────────────────────────

def repeat_kv(x, n):
    if n == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n, s, d).reshape(b, h * n, s, d)


class Attention(nn.Module):
    """Encoder (causal) attention. scale=1.0; QK-norm + V-norm; global layers have no v_proj."""

    def __init__(self, cfg: DiffusionGemmaConfig, layer_idx: int):
        super().__init__()
        self.is_sliding = cfg.is_sliding(layer_idx)
        self.head_dim = cfg.head_dim if self.is_sliding else cfg.global_head_dim
        kv_heads = cfg.num_key_value_heads if self.is_sliding else cfg.num_global_key_value_heads
        self.n_heads = cfg.num_attention_heads
        self.kv_heads = kv_heads
        self.groups = self.n_heads // kv_heads
        self.sliding_window = cfg.sliding_window if self.is_sliding else None

        H = cfg.hidden_size
        self.q_proj = nn.Linear(H, self.n_heads * self.head_dim, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(H, kv_heads * self.head_dim, bias=cfg.attention_bias)
        self.v_proj = (nn.Linear(H, kv_heads * self.head_dim, bias=cfg.attention_bias)
                       if self.is_sliding else None)             # global: V reuses K projection
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, H, bias=cfg.attention_bias)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps, with_scale=False)

    def _qkv(self, x, cos, sin):
        """Project + norm + RoPE. Returns q,(B,H,S,D) and the PRE-repeat k,v (B,kv,S,D)."""
        B, S, _ = x.shape
        shape = (B, S, -1, self.head_dim)
        q = apply_rope(self.q_norm(self.q_proj(x).view(shape)), cos, sin).transpose(1, 2)
        k_raw = self.k_proj(x).view(shape)
        v_raw = self.v_proj(x).view(shape) if self.v_proj is not None else k_raw  # k_eq_v on global
        k = apply_rope(self.k_norm(k_raw), cos, sin).transpose(1, 2)
        v = self.v_norm(v_raw).transpose(1, 2)                   # NOTE: V gets no RoPE
        return q, k, v

    def forward(self, x, cos, sin, past_kv=None, return_kv=False):
        B, S, _ = x.shape
        q, k_new, v_new = self._qkv(x, cos, sin)                 # new tokens' q,k,v (B,*,S,D)
        if past_kv is not None:                                  # incremental: prepend cached K/V
            pk, pv = past_kv
            k_kv, v_kv = torch.cat([pk, k_new], dim=2), torch.cat([pv, v_new], dim=2)
        else:
            k_kv, v_kv = k_new, v_new
        k, v = repeat_kv(k_kv, self.groups), repeat_kv(v_kv, self.groups)

        # manual OFFSET-causal (+ sliding band) attention, scale 1.0, fp32 softmax — matches HF eager.
        # past_len=0 → plain prefill; past_len>0 → new tokens attend to [cache ; themselves] causally.
        total = k_kv.shape[2]
        past_len = total - S
        scores = torch.matmul(q, k.transpose(2, 3)) * 1.0
        qpos = torch.arange(past_len, total, device=x.device)
        kpos = torch.arange(total, device=x.device)
        allowed = kpos[None, :] <= qpos[:, None]
        if self.sliding_window is not None:
            allowed = allowed & ((qpos[:, None] - kpos[None, :]) < self.sliding_window)
        scores = scores.masked_fill(~allowed[None, None], float("-inf"))
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, -1)
        out = self.o_proj(out)
        return (out, (k_kv, v_kv)) if return_kv else out         # cache = accumulated pre-repeat K/V         # cache stores PRE-repeat k,v


class DecoderAttention(Attention):
    """Bidirectional canvas attention that READS a (read-only) encoder KV cache by concatenation.
    Same projections/norms/k_eq_v as the encoder attention; differs only in mask + cache concat."""

    def forward(self, x, cos, sin, enc_k, enc_v):
        # enc_k, enc_v: the encoder's cached (B, kv_heads, S_enc, D) for THIS layer (pre-repeat).
        B, S, _ = x.shape
        q, k_can, v_can = self._qkv(x, cos, sin)                 # canvas q,k,v
        k = torch.cat([enc_k, k_can], dim=2)                    # [encoder ; canvas] keys
        v = torch.cat([enc_v, v_can], dim=2)
        k, v = repeat_kv(k, self.groups), repeat_kv(v, self.groups)
        # no-padding dynamic-cache case → HF returns mask=None → FULL bidirectional, no window
        scores = torch.matmul(q, k.transpose(2, 3)) * 1.0
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(out)


class SelfConditioning(nn.Module):
    """GeGLU block: out = post_norm(inputs_embeds + geglu(pre_norm(signal))). post_norm is unscaled.
    Applied EVERY decoder step — when there's no prior step, signal=zeros (→ just post_norm(embeds))."""

    def __init__(self, cfg: DiffusionGemmaConfig):
        super().__init__()
        H, I = cfg.hidden_size, cfg.intermediate_size
        self.pre_norm = RMSNorm(H, cfg.rms_norm_eps)
        self.post_norm = RMSNorm(H, cfg.rms_norm_eps, with_scale=False)
        self.gate_proj = nn.Linear(H, I, bias=False)
        self.up_proj = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False)
        self.act = ACT[cfg.hidden_act]

    def forward(self, inputs_embeds, signal):
        n = self.pre_norm(signal)
        sc = self.down_proj(self.act(self.gate_proj(n)) * self.up_proj(n))
        return self.post_norm(inputs_embeds + sc)


# ───────────────────────── FFN: dense MLP + routed MoE ─────────────────────────

class DenseMLP(nn.Module):
    """The always-on dense GeGLU MLP (the card's '1 shared')."""

    def __init__(self, cfg: DiffusionGemmaConfig):
        super().__init__()
        H, I = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(H, I, bias=False)
        self.up_proj = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False)
        self.act = ACT[cfg.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class Router(nn.Module):
    def __init__(self, cfg: DiffusionGemmaConfig):
        super().__init__()
        self.top_k = cfg.top_k_experts
        self.root = cfg.hidden_size ** -0.5
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, with_scale=False)
        self.proj = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(cfg.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(cfg.num_experts))

    def forward(self, x):
        x = self.norm(x) * self.scale * self.root
        probs = F.softmax(self.proj(x), dim=-1, dtype=torch.float32)
        w, idx = torch.topk(probs, self.top_k, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        w = w * self.per_expert_scale[idx]
        return w, idx


class Experts(nn.Module):
    """Fused 3D expert weights, looped per hit expert (parity-faithful, not fast)."""

    def __init__(self, cfg: DiffusionGemmaConfig):
        super().__init__()
        E, H, I = cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size
        self.num_experts = E
        self.gate_up_proj = nn.Parameter(torch.empty(E, 2 * I, H))
        self.down_proj = nn.Parameter(torch.empty(E, H, I))
        self.act = ACT[cfg.hidden_act]

    def forward(self, x, idx, w):
        out = torch.zeros_like(x)
        mask = F.one_hot(idx, self.num_experts).permute(2, 1, 0)   # (E, K, T)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for e in hit:
            e = e[0]
            k_pos, tok = torch.where(mask[e])
            gate, up = F.linear(x[tok], self.gate_up_proj[e]).chunk(2, dim=-1)
            h = self.act(gate) * up
            h = F.linear(h, self.down_proj[e]) * w[tok, k_pos, None]
            out.index_add_(0, tok, h.to(out.dtype))
        return out
