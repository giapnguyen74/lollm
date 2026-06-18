"""
generate.py — diffusion_gemma block-autoregressive generation (the outer loop).

Wraps the inner denoise loop (`sampler.denoise_block`): prefill the prompt → denoise a canvas →
commit the finished block → **re-encode it into the KV cache** (incremental causal encode) → denoise
the next canvas, until eos or the canvas budget. Mirrors transformers `generate`'s outer loop.

Note: this is the family's own loop (the shared `src/generate.py` is autoregressive and can't drive
diffusion — CONVENTIONS §4). Integration into `run.py` is deferred (a separate, explicit step).
"""
from __future__ import annotations

import torch

from sampler import denoise_block


@torch.no_grad()
def generate_diffusion(enc, dec, sampler, stop, prompt, *, max_new_canvases, max_denoising_steps,
                       t_min, t_max, eos_ids=None, sample=True):
    """`enc`/`dec` = our Encoder/DecoderTextModel; returns the generated token ids (canvases concatenated)."""
    device = prompt.device
    batch = prompt.shape[0]
    eos = torch.tensor(list(eos_ids), device=device) if eos_ids else None

    cache = enc(prompt, return_cache=True)[1]                    # 1. prefill the prompt
    blocks = []
    for _ in range(max_new_canvases):
        def forward_logits(canvas, self_cond):                  # decoder reads the (read-only) cache
            return dec.to_logits(dec(canvas, cache, self_conditioning_logits=self_cond))

        block = denoise_block(forward_logits, sampler, stop, max_denoising_steps=max_denoising_steps,
                              t_min=t_min, t_max=t_max, batch_size=batch, device=device, sample=sample)
        blocks.append(block)
        if eos is not None and bool(torch.isin(block, eos).any()):
            break                                               # eos in the block → stop
        # 2. commit: re-encode the finished block into the cache (causal), growing it by one canvas
        clen = cache[0][0].shape[2]
        pos = torch.arange(clen, clen + block.shape[1], device=device)[None]
        cache = enc(block, past_cache=cache, position_ids=pos, return_cache=True)[1]
    return torch.cat(blocks, dim=1)
