"""Triton flash-attention kernel (offset-causal, KV-cache aware), forward only.

Productionized from the kernel studied in `triton/05_flash_attention_kvcache.py`.
Semantics match the torch SDPA reference exactly: a query at absolute position
`(total_k - q_len) + i` attends to keys `j <= that position`, which covers plain
causal (q_len == total_k), chunked/ragged prefill (cached K/V), and decode (q_len == 1).

Two changes vs the study version, for real autoregressive inference:
  • autotune keys on HEAD_DIM only — NOT on the sequence lengths. total_k grows by
    one every decode step, so keying on it re-ran the whole config sweep per token.
    HEAD_DIM is constant across a run, so we tune once and reuse.
  • the kernel is stride-aware (takes per-dim strides), so it reads Q/K/V directly
    from their (B,H,S,D) views. No `.reshape(...).contiguous()` — which previously
    copied the entire growing KV cache on every step.

This module does `import triton` at top level (the @triton.jit decorator needs it),
so it can only be imported where Triton is installed — i.e. on CUDA/Linux. The
dispatcher in `attention.py` imports it lazily and falls back to torch elsewhere.
Forward only; this is an inference engine.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def _configs():
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
        for bm in (64, 128) for bn in (32, 64) for w in (4, 8) for s in (2, 3)
    ]


# Key on HEAD_DIM only: it's constant across a run, so the 16-config sweep runs once
# (per head dim) instead of every decode step (when total_k changes each token).
@triton.autotune(configs=_configs(), key=["HEAD_DIM"])
@triton.jit
def kv_fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, scale, H, n_ctx_q, n_ctx_k,
                  q_sb, q_sh, q_sm, q_sd,
                  k_sb, k_sh, k_sn, k_sd,
                  v_sb, v_sh, v_sn, v_sd,
                  o_sb, o_sh, o_sm, o_sd,
                  HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_z = tl.program_id(1)                     # flattened batch*head
    b = pid_z // H
    h = pid_z % H
    offset = n_ctx_k - n_ctx_q                    # the KV-cache offset

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_pos = offset + offs_m                       # absolute position of each query
    q_mask = offs_m < n_ctx_q                     # decode: q_len may be < BLOCK_M

    # Strided gather straight from the (B,H,S,D) view — no contiguous copy needed.
    q_base = b * q_sb + h * q_sh
    q = tl.load(q_ptr + q_base + offs_m[:, None] * q_sm + offs_d[None, :] * q_sd,
                mask=q_mask[:, None], other=0.0)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    k_base = b * k_sb + h * k_sh
    v_base = b * v_sb + h * v_sh
    # A query at absolute pos q_pos can see keys up to q_pos; the block's max
    # query is offset + (pid_m+1)*BLOCK_M - 1, so stop the loop there.
    hi = tl.minimum(offset + (pid_m + 1) * BLOCK_M, n_ctx_k)
    for start_n in range(0, hi, BLOCK_N):
        cur_n = start_n + tl.arange(0, BLOCK_N)
        k_mask = cur_n < n_ctx_k
        k = tl.load(k_ptr + k_base + cur_n[:, None] * k_sn + offs_d[None, :] * k_sd,
                    mask=k_mask[:, None], other=0.0)
        v = tl.load(v_ptr + v_base + cur_n[:, None] * v_sn + offs_d[None, :] * v_sd,
                    mask=k_mask[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        keep = (q_pos[:, None] >= cur_n[None, :]) & k_mask[None, :]   # offset causal
        s = tl.where(keep, s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_base = b * o_sb + h * o_sh
    tl.store(o_ptr + o_base + offs_m[:, None] * o_sm + offs_d[None, :] * o_sd,
             acc.to(o_ptr.dtype.element_ty), mask=q_mask[:, None])


def flash_attention_kv(q, k, v):
    # q: (B,H,q_len,D)   k,v: (B,H,total_k,D)   total_k >= q_len
    # No reshape/contiguous: pass the views directly and hand the kernel their strides.
    B, H, Sq, D = q.shape
    Sk = k.shape[-2]
    scale = 1.0 / (D ** 0.5)
    o = torch.empty((B, H, Sq, D), device=q.device, dtype=q.dtype)
    grid = lambda meta: (triton.cdiv(Sq, meta["BLOCK_M"]), B * H)
    kv_fwd_kernel[grid](
        q, k, v, o, scale, H, Sq, Sk,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        HEAD_DIM=D,
    )
    return o
