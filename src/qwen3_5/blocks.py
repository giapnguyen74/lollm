"""
qwen3_5/blocks.py — Qwen3.5 primitive components.

STEP 1 declared the EXACT parameter shapes so the loader can build on `meta` and assert.
STEP 2 implemented GatedAttention (full-attention, gated, partial RoPE).
STEP 3 (this file) implements GatedDeltaNet — the causal conv, the chunked and recurrent
gated-delta-rule scans, and their pure-torch helpers — ported from the qwen3_next
reference (see docs/qwen3_5-architecture.md). The family cache that threads the GDN's
(conv_state, recurrent_state) through `forward` is step 4.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3_5Config


class RMSNorm(nn.Module):
    """(1 + weight)·x̂ RMSNorm in fp32 (zero-init weight) — Qwen3.5 / Gemma convention."""

    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        d = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (h * (1.0 + self.weight.float())).to(d)


class RMSNormGated(nn.Module):
    """Normalize (ones-init weight), then gate by silu(z). Used inside the GDN output."""

    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, gate):
        d = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        h = self.weight * h.to(d)
        return (h * F.silu(gate.float())).to(d)


def l2norm(x, eps=1e-6):
    """L2-normalize the last dim (FLA convention: 1/sqrt(Σx² + eps), no mean)."""
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def torch_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64,
                                 initial_state=None, use_qk_l2norm_in_kernel=False):
    """
    Chunked gated delta rule — the parallel-prefill form of the recurrence (port of the
    reference's pure-torch fallback). Inputs are (B, T, H, D); returns
    (core_attn_out (B,T,H,head_v), final_state (B,H,head_k,head_v)). The state write per
    chunk is `state·exp(g) + Σ kᵀ(v − ⟨state,k⟩)β`, reorganized into chunks of `chunk_size`
    so the intra-chunk part is one matmul and only the chunk boundary is sequential.
    """
    in_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query, key = l2norm(query), l2norm(key)
    # (B,T,H,D) → (B,H,T,D), fp32 for the scan
    query, key, value, beta, g = (
        t.transpose(1, 2).contiguous().float() for t in (query, key, value, beta, g))
    B, H, T, k_head = key.shape
    v_head = value.shape[-1]
    pad = (chunk_size - T % chunk_size) % chunk_size
    query, key, value = (F.pad(t, (0, 0, 0, pad)) for t in (query, key, value))
    beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
    Tp = T + pad
    query = query * (k_head ** -0.5)                       # scale q by 1/sqrt(head_k)

    v_beta, k_beta = value * beta.unsqueeze(-1), key * beta.unsqueeze(-1)
    # → (B,H,n_chunk,chunk,D)
    query, key, value, k_beta, v_beta = (
        t.reshape(B, H, -1, chunk_size, t.shape[-1]) for t in (query, key, value, k_beta, v_beta))
    g = g.reshape(B, H, -1, chunk_size)

    tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), 0)
    g = g.cumsum(-1)
    decay = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay).masked_fill(tri, 0)
    for i in range(1, chunk_size):                          # invert (I − strictly-lower)
        row = attn[..., i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    state = (torch.zeros(B, H, k_head, v_head, dtype=value.dtype, device=value.device)
             if initial_state is None else initial_state.to(value))
    out = torch.zeros_like(value)
    strict = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), 1)
    for i in range(Tp // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        intra = (q_i @ k_i.transpose(-1, -2) * decay[:, :, i]).masked_fill(strict, 0)
        v_new = v_i - k_cumdecay[:, :, i] @ state
        inter = (q_i * g[:, :, i, :, None].exp()) @ state
        out[:, :, i] = inter + intra @ v_new
        state = (state * g[:, :, i, -1, None, None].exp()
                 + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new)

    out = out.reshape(B, H, -1, v_head)[:, :, :T]
    return out.transpose(1, 2).contiguous().to(in_dtype), state


def torch_recurrent_gated_delta_rule(query, key, value, g, beta, initial_state=None,
                                     use_qk_l2norm_in_kernel=False):
    """
    Single-step (or short) recurrent gated delta rule — the cleanest reading of the math:
    per step `state ← state·exp(g); state ← state + kᵀ·(v − ⟨state,k⟩)·β; out = ⟨state,q⟩`.
    Inputs (B,T,H,D); returns (out (B,T,H,head_v), final_state (B,H,head_k,head_v)).
    """
    in_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query, key = l2norm(query), l2norm(key)
    query, key, value, beta, g = (
        t.transpose(1, 2).contiguous().float() for t in (query, key, value, beta, g))
    B, H, T, k_head = key.shape
    v_head = value.shape[-1]
    query = query * (k_head ** -0.5)

    out = torch.zeros(B, H, T, v_head, dtype=value.dtype, device=value.device)
    state = (torch.zeros(B, H, k_head, v_head, dtype=value.dtype, device=value.device)
             if initial_state is None else initial_state.to(value))
    for i in range(T):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(-2)        # ⟨state, k⟩
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (state * q_t.unsqueeze(-1)).sum(-2)  # ⟨state, q⟩
    return out.transpose(1, 2).contiguous().to(in_dtype), state


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, t, d).reshape(b, n_kv * n_rep, t, d)


class RoPE:
    """
    Partial RoPE: only the first `rotary_dim = head_dim · partial_rotary_factor` dims are
    rotated; the rest pass through. For text-only input the model's mRoPE reduces to this
    (all position axes share one position), so we build standard cos/sin tables.
    """

    def __init__(self, rotary_dim, theta, device):
        idx = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
        self.inv_freq = 1.0 / (theta ** (idx / rotary_dim))

    def cos_sin(self, positions, dtype):
        freqs = positions[:, None].float() * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)          # (T, rotary_dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    @staticmethod
    def apply(q, k, cos, sin):
        rd = cos.shape[-1]                               # rotary_dim (< head_dim)
        cos, sin = cos[None, None], sin[None, None]
        q_rot, q_pass = q[..., :rd], q[..., rd:]
        k_rot, k_pass = k[..., :rd], k[..., rd:]
        q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
        k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)
        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


class GatedAttention(nn.Module):
    """
    Full-attention layer (Qwen3.5): GQA with QK-norm, partial RoPE, and an output gate.
    `q_proj` emits query AND gate (hence ×2); after attention `o = o * gate_act(gate)`.
    No biases. The reference (`Qwen3_5Attention(Qwen3NextAttention)`) uses sigmoid; we
    keep it config-driven (`output_gate_type`) but default to sigmoid for 4B parity.
    """

    def __init__(self, cfg: Qwen3_5Config):
        super().__init__()
        self.n_head, self.n_kv, self.n_rep = (
            cfg.num_attention_heads, cfg.num_key_value_heads, cfg.n_rep)
        self.head_dim = cfg.head_dim
        h, hd = cfg.hidden_size, cfg.head_dim
        self.q_proj = nn.Linear(h, self.n_head * hd * 2, bias=False)   # query + output gate
        self.k_proj = nn.Linear(h, self.n_kv * hd, bias=False)
        self.v_proj = nn.Linear(h, self.n_kv * hd, bias=False)
        self.o_proj = nn.Linear(self.n_head * hd, h, bias=False)
        self.q_norm = RMSNorm(hd, cfg.rms_norm_eps)                    # QK-norm (per head_dim)
        self.k_norm = RMSNorm(hd, cfg.rms_norm_eps)
        self.gate_act = F.silu if cfg.output_gate_type in ("swish", "silu") else torch.sigmoid

    def forward(self, x, cos, sin, cache=None, layer_idx=0):
        b, t, _ = x.shape
        # 1. PROJECT q (+gate), k, v. q_proj packs query and gate per head (2·head_dim).
        qg = self.q_proj(x).view(b, t, self.n_head, self.head_dim * 2)
        q, gate = qg.chunk(2, dim=-1)                                  # each (b,t,n_head,hd)
        gate = gate.reshape(b, t, self.n_head * self.head_dim)
        # 2. QK-NORM (per head_dim) then heads-first.
        q = self.q_norm(q).transpose(1, 2)                             # (b,n_head,t,hd)
        k = self.k_norm(self.k_proj(x).view(b, t, self.n_kv, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        # 3. PARTIAL ROPE on q,k.
        q, k = RoPE.apply(q, k, cos, sin)
        # 4. KV CACHE — the cache owns storage: append this step's (k,v) post-RoPE /
        #    pre-GQA-expansion, and read back the full K,V to attend over. No cache (e.g. the
        #    MTP block's one-shot pass) → attend over just this step's k,v.
        if cache is not None:
            k, v = cache.append_kv(layer_idx, k, v)
        # 5. GQA EXPAND.
        k, v = _repeat_kv(k, self.n_rep), _repeat_kv(v, self.n_rep)
        # 6. ATTENTION — causal (scale = head_dim**-0.5 = SDPA default).
        q_len, total_k = q.shape[2], k.shape[2]
        if q_len > 1 and q_len != total_k:                            # chunked prefill w/ cache
            qpos = torch.arange(total_k - q_len, total_k, device=q.device)
            kpos = torch.arange(total_k, device=q.device)
            mask = (kpos[None, :] <= qpos[:, None])[None, None]
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=q_len > 1)
        # 7. MERGE heads, OUTPUT GATE, project.
        o = o.transpose(1, 2).reshape(b, t, self.n_head * self.head_dim)
        o = o * self.gate_act(gate)
        return self.o_proj(o)


class GatedDeltaNet(nn.Module):
    """
    Linear-attention mixer (Gated DeltaNet, Qwen3.5 layout): four input projections
    (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`), a causal depthwise Conv1d over
    (q,k,v), a gated delta-rule recurrence, and a gated output norm. Carries a
    (conv_state, recurrent_state) cache instead of KV.
    """

    def __init__(self, cfg: Qwen3_5Config):
        super().__init__()
        self.cfg = cfg
        k = cfg.linear_conv_kernel_dim
        self.in_proj_qkv = nn.Linear(cfg.hidden_size, cfg.conv_dim, bias=False)   # q,k,v
        self.in_proj_z = nn.Linear(cfg.hidden_size, cfg.value_dim, bias=False)    # output gate
        self.in_proj_b = nn.Linear(cfg.hidden_size, cfg.linear_num_value_heads, bias=False)
        self.in_proj_a = nn.Linear(cfg.hidden_size, cfg.linear_num_value_heads, bias=False)
        self.conv1d = nn.Conv1d(cfg.conv_dim, cfg.conv_dim, kernel_size=k,
                                groups=cfg.conv_dim, bias=False, padding=k - 1)
        self.dt_bias = nn.Parameter(torch.zeros(cfg.linear_num_value_heads))
        self.A_log = nn.Parameter(torch.zeros(cfg.linear_num_value_heads))
        self.norm = RMSNormGated(cfg.linear_value_head_dim, cfg.rms_norm_eps)
        self.out_proj = nn.Linear(cfg.value_dim, cfg.hidden_size, bias=False)
        # dims used by forward
        self.kernel = k
        self.key_dim, self.value_dim = cfg.key_dim, cfg.value_dim
        self.head_k, self.head_v = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        self.n_k, self.n_v, self.v_per_k = (
            cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.v_per_k)

    def _conv(self, qkv, conv_state, seq_len, use_cache):
        """
        Causal depthwise conv over cat(q,k,v) + SiLU. `qkv` is (B, conv_dim, L).
        Returns (activated (B, conv_dim, seq_len), new_conv_state (B, conv_dim, kernel-1)).
        conv_state is the last kernel-1 raw input columns (the per-step positional memory).
        """
        K = self.kernel
        if conv_state is not None and seq_len == 1:
            # cached single-token decode: window = [conv_state | new token] (width K)
            window = torch.cat([conv_state, qkv], dim=-1)
            out = F.conv1d(window, self.conv1d.weight, None, padding=0, groups=qkv.shape[1])
            out = F.silu(out[:, :, -seq_len:])
            return out, window[:, :, -(K - 1):]
        x = qkv if conv_state is None else torch.cat([conv_state, qkv], dim=-1)
        out = F.silu(self.conv1d(x)[:, :, : x.shape[-1]])          # left-padded → causal
        if conv_state is not None:
            out = out[:, :, -seq_len:]                              # drop the prepended context
        new_state = F.pad(x, (K - 1 - x.shape[-1], 0)) if use_cache else None
        return out, new_state

    def forward(self, x, conv_state=None, recurrent_state=None, use_cache=False):
        """
        GDN token mixer. State-explicit interface (step 4 threads these through the family
        cache): pass prior (conv_state, recurrent_state) for cached decode, get the updated
        pair back when `use_cache`. Returns (out (B,T,hidden), new_conv_state, new_rec_state).
        """
        B, T, _ = x.shape
        cached = conv_state is not None
        # 1. PROJECT — Qwen3.5's four separate linears (q,k,v already concatenated).
        qkv = self.in_proj_qkv(x).transpose(1, 2)              # (B, conv_dim, T)
        z = self.in_proj_z(x).reshape(B, T, self.n_v, self.head_v)
        b = self.in_proj_b(x)                                  # (B, T, n_v)
        a = self.in_proj_a(x)
        # 2. CAUSAL CONV (+ state).
        qkv, new_conv = self._conv(qkv, conv_state, T, use_cache)
        qkv = qkv.transpose(1, 2)                              # (B, T, conv_dim)
        # 3. SPLIT + HEADS (repeat q,k to v-head count when num_v > num_k).
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, T, self.n_k, self.head_k)
        k = k.reshape(B, T, self.n_k, self.head_k)
        v = v.reshape(B, T, self.n_v, self.head_v)
        if self.v_per_k > 1:
            q = q.repeat_interleave(self.v_per_k, dim=2)
            k = k.repeat_interleave(self.v_per_k, dim=2)
        # 4. GATING TERMS — beta gate, per-head decay g (fp32; A might be -inf in fp16).
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        # 5. DELTA-RULE SCAN — recurrent for cached single-token decode, chunked otherwise.
        if cached and T == 1:
            core, new_rec = torch_recurrent_gated_delta_rule(
                q, k, v, g, beta, initial_state=recurrent_state, use_qk_l2norm_in_kernel=True)
        else:
            core, new_rec = torch_chunk_gated_delta_rule(
                q, k, v, g, beta, initial_state=recurrent_state if cached else None,
                use_qk_l2norm_in_kernel=True)
        # 6. GATED OUTPUT NORM (gate=z) + project. Norm is per head_v dim.
        core = self.norm(core.reshape(-1, self.head_v), z.reshape(-1, self.head_v))
        core = core.reshape(B, T, self.value_dim)
        out = self.out_proj(core)
        if not use_cache:
            return out, None, None
        return out, new_conv, new_rec


class MLP(nn.Module):
    """SwiGLU: down(silu(gate(x)) · up(x)). Dense (no MoE in 4B/27B)."""

    def __init__(self, cfg: Qwen3_5Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
