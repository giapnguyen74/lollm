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


# ──────────────────────────────────────────────────────────────────────────────
# Real-prompt generation test (no reference comparison — that's compare_logits.py).
# Loads the real checkpoint ONCE, builds our modules on `meta` and ASSIGNs the reference's
# weights (so there's a single ~52GB copy, shared), then generates with `generate_diffusion`.
#   python src/diffusion_gemma/generate.py --prompt "Why is the sky blue?"
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from config import DiffusionGemmaConfig
    from modeling_diffusion_gemma import DecoderTextModel, EncoderTextModel
    from blocks import default_inv_freq, proportional_inv_freq
    from sampler import EntropyBoundSampler, StableAndConfidentStopping

    def _device(req):
        if req != "auto":
            return req
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _dtype(dev):
        return torch.bfloat16 if dev.startswith("cuda") else torch.float16 if dev == "mps" else torch.float32

    def _g(o, n, d):                                          # field from gen-config obj or dict
        v = getattr(o, n, None)
        if v is None and isinstance(o, dict):
            v = o.get(n)
        return d if v is None else v

    ap = argparse.ArgumentParser(description="diffusion_gemma real-prompt generation test")
    ap.add_argument("--model", default="google/diffusiongemma-26B-A4B-it")
    ap.add_argument("--prompt", default="Why is the sky blue?")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-new-canvases", type=int, default=4)
    args = ap.parse_args()

    device = _device(args.device)
    dtype = getattr(torch, args.dtype) if args.dtype else _dtype(device)

    from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion
    print(f"[loading {args.model} ({str(dtype).split('.')[-1]}) on {device}]", file=sys.stderr)
    ref = DiffusionGemmaForBlockDiffusion.from_pretrained(args.model, dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.model)

    cfg = DiffusionGemmaConfig.from_hf(ref.config.to_dict())
    canvas_length = ref.config.canvas_length
    with torch.device("meta"):                               # no weight allocation (L-2)
        enc, dec = EncoderTextModel(cfg), DecoderTextModel(cfg)
    enc.load_state_dict(ref.model.encoder.language_model.state_dict(), strict=True, assign=True)
    dec.load_state_dict(ref.model.decoder.state_dict(), strict=True, assign=True)
    for m in (enc, dec):                                      # rope inv_freq are computed, not loaded
        m.rope_sliding.inv_freq = default_inv_freq(cfg.rope_theta_local, cfg.head_dim).to(device)
        m.rope_full.inv_freq = proportional_inv_freq(
            cfg.rope_theta_global, cfg.global_head_dim, cfg.partial_rotary_factor_global).to(device)
    enc.eval()
    dec.eval()

    ids = tok.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=True,
                                  add_generation_prompt=True, return_dict=True,
                                  return_tensors="pt")["input_ids"].to(device)

    gc = ref.generation_config
    sampler = EntropyBoundSampler(_g(getattr(gc, "sampler_config", None), "entropy_bound", 0.1),
                                  canvas_length, cfg.vocab_size)
    stop = StableAndConfidentStopping(_g(gc, "stability_threshold", 1), _g(gc, "confidence_threshold", 0.005))
    print(f"[generating up to {args.max_new_canvases} × {canvas_length}-token canvases]", file=sys.stderr)
    out = generate_diffusion(enc, dec, sampler, stop, ids,
                             max_new_canvases=args.max_new_canvases,
                             max_denoising_steps=_g(gc, "max_denoising_steps", 48),
                             t_min=_g(gc, "t_min", 0.4), t_max=_g(gc, "t_max", 0.8),
                             eos_ids=_g(gc, "eos_token_id", None))
    print("\n[prompt] " + args.prompt)
    print("[output] " + tok.decode(out[0], skip_special_tokens=True))
