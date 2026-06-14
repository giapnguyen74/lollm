"""
run.py — CLI. loader → router → family.load → shared generate loop.

    python run.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Explain RoPE in one line."
    python run.py --model ./local/qwen2/dir --prompt "Hi" --temperature 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

import loader
import router
from generate import generate


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


def resolve_sampling(gen_sampling, fam_defaults, user):
    out = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}
    out.update(fam_defaults)                                   # curated per-family
    out.update(gen_sampling)                                   # generation_config (HF)
    out.update({k: v for k, v in user.items() if v is not None})  # user override
    return out


def encode_prompt(tok, prompt, no_chat):
    if not no_chat and tok.chat_template:
        return tok.apply_chat(prompt)
    return tok.encode(prompt)


def main():
    p = argparse.ArgumentParser(description="Study-first inference (qwen2).")
    p.add_argument("--model", required=True, help="HF repo id or local dir")
    p.add_argument("--prompt", default="Hello, who are you?")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--repetition-penalty", type=float, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-chat", action="store_true")
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"[loading {args.model} on {device}]", file=sys.stderr)
    t0 = time.time()

    L = loader.load(args.model)               # raw config + weights + tokenizer
    fam = router.route(L.model_type)          # probe + route (fail loud if unknown)
    # streams weights onto `device` and frees each CPU source as it goes (peak ≈ one copy)
    model = fam.load(L.raw_config, L.weights, L.fmt, device, pick_dtype(device))
    print(f"[{L.fmt} | {L.model_type} | loaded in {time.time()-t0:.1f}s]", file=sys.stderr)

    sampling = resolve_sampling(
        L.gen_meta["sampling"], fam.defaults,
        {"temperature": args.temperature, "top_k": args.top_k,
         "top_p": args.top_p, "repetition_penalty": args.repetition_penalty})
    stop_ids = L.gen_meta["stop_ids"]            # loader populates for both formats

    ids = encode_prompt(L.tokenizer, args.prompt, args.no_chat)
    decode = L.tokenizer.decode                  # uniform: decode(ids) -> str

    print(f"\n>>> {args.prompt}\n", file=sys.stderr)
    n, t0 = 0, time.time()
    for piece in generate(model, ids, decode, stop_ids, device,
                          max_new_tokens=args.max_new_tokens, **sampling):
        print(piece, end="", flush=True)
        n += 1
    dt = time.time() - t0
    print(f"\n\n[{n} tokens in {dt:.1f}s = {n/max(dt,1e-9):.1f} tok/s]", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Ctrl+C during a torch op can segfault on normal teardown (MPS/CUDA).
        # Flush what we streamed, then exit hard (130 = SIGINT) to skip the
        # teardown that crashes.
        sys.stdout.flush()
        print("\n[interrupted]", file=sys.stderr, flush=True)
        os._exit(130)
