"""
run.py — diffusion_gemma CLI (standalone; the family is NOT wired into the shared engine yet).

Two runtime features:
  • lean weight load — `device_map` streams the checkpoint shards straight onto the accelerator
    (transformers' own progress bar) without a full CPU copy; our encoder + decoder then SHARE that
    single set of weights (they're tied), so there's exactly one copy in memory;
  • streaming generation — print each committed canvas (block) as it's produced.

NB: a hand-rolled per-module CPU→GPU stream does NOT work here — the encoder and decoder share tied
weights, so streaming one module can't free the CPU tensors the other still holds (CPU OOM) and the
two modules end up with separate GPU copies (2× memory). Sharing a single device-resident copy is
the correct lean load.

    python src/diffusion_gemma/run.py --prompt "Why is the sky blue?"
    python src/diffusion_gemma/run.py --prompt "..." --max-new-canvases 6

Weights/tokenizer are read via `transformers` for now (the standalone safetensors loader + the
hand-written tokenizer are part of the deferred integration — see README). Device/dtype auto-detect
like the engine: cuda→bf16, mps→fp16, cpu→fp32.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import DiffusionGemmaConfig                                   # noqa: E402
from modeling_diffusion_gemma import DecoderTextModel, EncoderTextModel   # noqa: E402
from blocks import default_inv_freq, proportional_inv_freq               # noqa: E402
from sampler import EntropyBoundSampler, StableAndConfidentStopping       # noqa: E402
from generate import generate_diffusion                                  # noqa: E402


def pick_device(req):
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device):
    if device.startswith("cuda"):
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def _g(o, n, d):                                              # gen-config field (obj or dict) + default
    v = getattr(o, n, None)
    if v is None and isinstance(o, dict):
        v = o.get(n)
    return d if v is None else v


def main():
    ap = argparse.ArgumentParser(description="diffusion_gemma standalone CLI (streaming)")
    ap.add_argument("--model", default="google/diffusiongemma-26B-A4B-it")
    ap.add_argument("--prompt", default="Why is the sky blue?")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-new-canvases", type=int, default=4)
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype) if args.dtype else pick_dtype(device)

    from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion
    t0 = time.time()
    # Lean GPU load: `device_map` makes transformers stream shards straight onto the accelerator
    # (its own "Loading weights" bar) WITHOUT materialising a full CPU copy. Our encoder + decoder then
    # SHARE that single set of weights via meta+assign — crucially they're tied (same tensors), so a
    # hand-rolled per-module CPU→GPU stream would (a) double GPU memory and (b) keep CPU pinned because
    # the other module still references the tied tensors → OOM. Sharing avoids both.
    dmap = device if device.startswith("cuda") else None
    print(f"[loading {args.model} ({str(dtype).split('.')[-1]}) → {device}]", file=sys.stderr)
    ref = DiffusionGemmaForBlockDiffusion.from_pretrained(args.model, dtype=dtype, device_map=dmap)
    if dmap is None:
        ref = ref.to(device)
    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = DiffusionGemmaConfig.from_hf(ref.config.to_dict())
    canvas_length = ref.config.canvas_length
    gen = ref.generation_config

    with torch.device("meta"):
        enc, dec = EncoderTextModel(cfg), DecoderTextModel(cfg)
    enc.load_state_dict(ref.model.encoder.language_model.state_dict(), strict=True, assign=True)
    dec.load_state_dict(ref.model.decoder.state_dict(), strict=True, assign=True)  # shares enc's tied weights
    for m in (enc, dec):                                      # RoPE inv_freq are computed, not loaded
        m.rope_sliding.inv_freq = default_inv_freq(cfg.rope_theta_local, cfg.head_dim).to(device)
        m.rope_full.inv_freq = proportional_inv_freq(
            cfg.rope_theta_global, cfg.global_head_dim, cfg.partial_rotary_factor_global).to(device)
    enc.eval()
    dec.eval()
    print(f"[loaded in {time.time() - t0:.1f}s — weights shared with the reference (single copy)]",
          file=sys.stderr)

    ids = tok.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=True,
                                  add_generation_prompt=True, return_dict=True,
                                  return_tensors="pt")["input_ids"].to(device)

    sampler = EntropyBoundSampler(_g(getattr(gen, "sampler_config", None), "entropy_bound", 0.1),
                                  canvas_length, cfg.vocab_size)
    stop = StableAndConfidentStopping(_g(gen, "stability_threshold", 1), _g(gen, "confidence_threshold", 0.005))

    print(f"\n>>> {args.prompt}\n", file=sys.stderr)
    n_tok = [0]
    g0 = time.time()

    def on_block(block):                                      # stream each finished canvas
        print(tok.decode(block[0], skip_special_tokens=True), end="", flush=True)
        n_tok[0] += block.shape[1]

    generate_diffusion(enc, dec, sampler, stop, ids,
                       max_new_canvases=args.max_new_canvases,
                       max_denoising_steps=_g(gen, "max_denoising_steps", 48),
                       t_min=_g(gen, "t_min", 0.4), t_max=_g(gen, "t_max", 0.8),
                       eos_ids=_g(gen, "eos_token_id", None), on_block=on_block)

    dt = time.time() - g0
    print(f"\n\n[{n_tok[0]} tokens in {dt:.1f}s = {n_tok[0] / max(dt, 1e-9):.1f} tok/s]", file=sys.stderr)


if __name__ == "__main__":
    main()
