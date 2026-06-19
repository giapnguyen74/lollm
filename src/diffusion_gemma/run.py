"""
run.py — diffusion_gemma CLI (standalone; the family is NOT wired into the shared engine yet).

Two runtime features:
  • lean weight load — load the checkpoint to CPU, then stream each tensor onto the GPU freeing the CPU
    source as we go (peak ≈ one copy, progress bar; no `accelerate`/`device_map` needed). This is safe
    because the model is a SINGLE backbone — with two tied modules it would pin CPU and double GPU (L-10);
  • streaming generation — print each committed canvas (block) as it's produced.

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
from modeling_diffusion_gemma import DiffusionGemmaModel                  # noqa: E402
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


def stream_to_gpu(model, device, label="weights"):
    """Move each loaded param/buffer CPU → `device` in place, freeing each CPU source as we go (peak ≈
    one copy), with a progress bar (LESSONS L-2). This is OUR streaming load — no `accelerate`/device_map.
    Safe here because the model is a SINGLE backbone: with two tied modules it would pin CPU + double GPU
    (L-10), but there's only one set of weights now. Meta tensors (RoPE, materialised next) are skipped."""
    tensors = [t for _, t in model.named_parameters()]
    tensors += [b for _, b in model.named_buffers() if b is not None]
    movable = [t for t in tensors if not t.is_meta]
    n, t0 = len(movable), time.time()
    for i, t in enumerate(movable, 1):
        t.data = t.data.to(device)                            # CPU tensor freed (model is the only ref)
        if i % 25 == 0 or i == n:
            print(f"\r[{label}] streaming → {device}: {i}/{n}", end="", file=sys.stderr, flush=True)
    print(f"   [{time.time() - t0:.1f}s, CPU freed]", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="diffusion_gemma standalone CLI (streaming)")
    ap.add_argument("--model", default="google/diffusiongemma-26B-A4B-it")
    ap.add_argument("--prompt", default="Why is the sky blue?")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-new-canvases", type=int, default=4)
    ap.add_argument("--demo", action="store_true",
                    help="visualise diffusion: redraw the canvas each denoise step (mask ▒ → confident "
                         "tokens). Slower (decodes + redraws every step); best with --max-new-canvases 1")
    ap.add_argument("--demo-delay", type=float, default=0.04, help="seconds between demo frames")
    ap.add_argument("--max-denoising-steps", type=int, default=None,
                    help="cap denoise steps per canvas (default: from generation_config, 48). Lower = "
                         "snappier demo, slightly rougher text")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype) if args.dtype else pick_dtype(device)

    from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion
    t0 = time.time()
    print(f"[loading {args.model} ({str(dtype).split('.')[-1]}) → CPU]", file=sys.stderr)
    ref = DiffusionGemmaForBlockDiffusion.from_pretrained(args.model, dtype=dtype)   # CPU (no accelerate)
    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = DiffusionGemmaConfig.from_hf(ref.config.to_dict())
    canvas_length = ref.config.canvas_length
    gen = ref.generation_config

    # ONE backbone, ONE load. The decoder state_dict carries the full text weights + the decoder-only
    # self_conditioning, so it populates the whole model; assign the reference's CPU tensors (share), then
    # DROP the reference so the model holds the only refs — and stream those tensors onto the GPU, freeing
    # CPU as we go (one copy; works precisely because there's a single backbone, not two tied modules).
    with torch.device("meta"):
        model = DiffusionGemmaModel(cfg)
    model.load_state_dict(ref.model.decoder.state_dict(), strict=True, assign=True)
    del ref                                                   # free the wrapper + vision tower (CPU)
    stream_to_gpu(model, device)
    model.rope_sliding.inv_freq = default_inv_freq(cfg.rope_theta_local, cfg.head_dim).to(device)
    model.rope_full.inv_freq = proportional_inv_freq(
        cfg.rope_theta_global, cfg.global_head_dim, cfg.partial_rotary_factor_global).to(device)
    model.eval()
    print(f"[loaded in {time.time() - t0:.1f}s]", file=sys.stderr)

    ids = tok.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=True,
                                  add_generation_prompt=True, return_dict=True,
                                  return_tensors="pt")["input_ids"].to(device)

    sampler = EntropyBoundSampler(_g(getattr(gen, "sampler_config", None), "entropy_bound", 0.1),
                                  canvas_length, cfg.vocab_size)
    stop = StableAndConfidentStopping(_g(gen, "stability_threshold", 1), _g(gen, "confidence_threshold", 0.005))

    print(f"\n>>> {args.prompt}\n", file=sys.stderr)
    n_tok = [0]
    g0 = time.time()
    max_steps = args.max_denoising_steps or _g(gen, "max_denoising_steps", 48)
    gkw = dict(max_new_canvases=args.max_new_canvases, max_denoising_steps=max_steps,
               t_min=_g(gen, "t_min", 0.4), t_max=_g(gen, "t_max", 0.8), eos_ids=_g(gen, "eos_token_id", None))

    if args.demo:
        # Visualise the reverse-diffusion: each step, accepted (confident) positions show their token,
        # the rest stay masked ▒. Watch the canvas converge from noise → text. (Reads back to CPU and
        # redraws every step → slow; that's the cost of the visualization.)
        step_ctr = [0]

        def on_step(cur_step, argmax, accepted):
            step_ctr[0] += 1
            am, acc = argmax[0].tolist(), accepted[0].tolist()
            cells = ["".join(tok.decode([tid]).replace("\n", " ") or "·") if a else "▒"
                     for tid, a in zip(am, acc)]
            filled = sum(acc)
            sys.stdout.write("\033[H\033[J")                  # cursor home + clear screen
            sys.stdout.write(f">>> {args.prompt}\n[denoise step {step_ctr[0]}/{max_steps} — "
                             f"{filled}/{len(acc)} tokens confident]\n\n")
            sys.stdout.write("".join(cells) + "\n")
            sys.stdout.flush()
            time.sleep(args.demo_delay)

        out = generate_diffusion(model, sampler, stop, ids, on_step=on_step, **gkw)
        sys.stdout.write("\033[H\033[J")
        print(f">>> {args.prompt}\n\n[final]\n" + tok.decode(out[0], skip_special_tokens=True))
        n_tok[0] = out.shape[1]
    else:
        def on_block(block):                                  # stream each finished canvas
            print(tok.decode(block[0], skip_special_tokens=True), end="", flush=True)
            n_tok[0] += block.shape[1]
        generate_diffusion(model, sampler, stop, ids, on_block=on_block, **gkw)

    dt = time.time() - g0
    print(f"\n\n[{n_tok[0]} tokens in {dt:.1f}s = {n_tok[0] / max(dt, 1e-9):.1f} tok/s]", file=sys.stderr)


if __name__ == "__main__":
    main()
