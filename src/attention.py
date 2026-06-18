"""Attention dispatch: Triton flash kernel on CUDA, torch SDPA on MPS/CPU.

`attention(q, k, v)` is the single entry point used by every model's Attention
block. It is offset-causal (KV-cache aware) and covers all three inference
regimes — plain prefill, decode (q_len == 1), and chunked prefill with a cache.

Path selection:
Torch is the default. The Triton kernel is opt-in via the LOLLM_ATTN env var:

  • LOLLM_ATTN unset / "torch"  -> always torch_attention (F.scaled_dot_product_attention)
  • LOLLM_ATTN="triton"         -> Triton kernel; raises if unavailable (CUDA tensor + triton)
  • LOLLM_ATTN="auto"           -> Triton when available (CUDA + installed), else torch

The Triton import is lazy and guarded: on macOS/CPU-only installs Triton isn't
present, so the import fails once, the result is cached, and we stay on torch.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

_kernel = None
_triton_ok: bool | None = None   # None = not yet probed


def _have_triton() -> bool:
    global _kernel, _triton_ok
    if _triton_ok is None:
        try:
            import triton  # noqa: F401  — absent on macOS (Linux wheels only)

            from _triton_attn import flash_attention_kv

            _kernel, _triton_ok = flash_attention_kv, True
        except Exception:
            _kernel, _triton_ok = None, False
    return _triton_ok


def torch_attention(q, k, v):
    """Reference SDPA path (prefill / decode / chunked-prefill offset-causal)."""
    q_len, total_k = q.shape[-2], k.shape[-2]
    if q_len > 1 and q_len != total_k:
        qpos = torch.arange(total_k - q_len, total_k, device=q.device)
        kpos = torch.arange(total_k, device=q.device)
        mask = (kpos[None, :] <= qpos[:, None])[None, None]
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return F.scaled_dot_product_attention(q, k, v, is_causal=q_len > 1)


def torch_attention_with_scale(q, k, v, scale, sliding_window=None, softcap=None):
    """`torch_attention`, but with an explicit softmax `scale` and the Gemma extras —
    laid right below it so the two read line-by-line.

    Differences vs `torch_attention`:
      • explicit `scale` (Gemma: `query_pre_attn_scalar**-0.5`, not SDPA's `1/sqrt(D)`);
      • optional attn-logit `softcap` (Gemma2) — can't be expressed via SDPA;
      • optional `sliding_window` band mask (local layers);
      • manual scores + fp32 softmax (so the soft-cap fits and large Gemma logits stay
        safe), instead of `F.scaled_dot_product_attention`.
    The caller has already projected, applied QK-norm/RoPE, appended to the cache, and
    GQA-expanded K/V. A Gemma-specific kernel may replace this later (see docs/ROADMAP.md R-6).

        q: (B, H, q_len, D)   k, v: (B, H, total_k, D)   total_k >= q_len
    """
    q_len, total_k = q.shape[-2], k.shape[-2]
    scores = torch.matmul(q, k.transpose(2, 3)) * scale
    if softcap is not None:                                   # Gemma2 attn-logit soft-cap
        scores = torch.tanh(scores / softcap) * softcap
    past_len = total_k - q_len
    qpos = torch.arange(past_len, total_k, device=q.device)
    kpos = torch.arange(total_k, device=q.device)
    allowed = kpos[None, :] <= qpos[:, None]                  # causal
    if sliding_window is not None:                            # local layers: band mask
        allowed = allowed & ((qpos[:, None] - kpos[None, :]) < sliding_window)
    scores = scores.masked_fill(~allowed[None, None], float("-inf"))
    attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(attn, v)


def attention(q, k, v):
    """Offset-causal attention. Triton flash kernel on CUDA; torch SDPA elsewhere."""
    mode = os.environ.get("LOLLM_ATTN", "torch")  # torch (default) | triton | auto
    if mode == "triton" and not (q.is_cuda and _have_triton()):
        raise RuntimeError("LOLLM_ATTN=triton but Triton is unavailable (need a CUDA tensor + triton installed)")
    if mode != "torch" and q.is_cuda and _have_triton():
        return _kernel(q, k, v)
    return torch_attention(q, k, v)


def warmup(*, head_dim, n_heads=8, seq=1024, device, dtype=torch.bfloat16):
    """Compile + autotune the Triton kernel ahead of time (call once after model load).

    The autotune sweep (16 configs) and JIT compile otherwise land on the FIRST
    attention call — i.e. layer 0's prefill — inflating time-to-first-token. Running
    one throwaway call here moves that cost off the critical path. Tuning keys only on
    head_dim, so the config chosen here is reused for every later prefill and decode.

    No-op unless LOLLM_ATTN enables Triton and a CUDA device with Triton is present.
    """
    mode = os.environ.get("LOLLM_ATTN", "torch")
    if mode == "torch" or not (str(device).startswith("cuda") and _have_triton()):
        return
    print(f"[triton attention: warming up (head_dim={head_dim})]", file=sys.stderr, flush=True)
    q = torch.randn(1, n_heads, seq, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, n_heads, seq, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, n_heads, seq, head_dim, device=device, dtype=dtype)
    _kernel(q, k, v)
    torch.cuda.synchronize()
