"""
hook_test.py — real-run smoke test for the modification hook seam (qwen2).

Loads Qwen/Qwen2.5-0.5B-Instruct the normal way, injects a probe via the shared
`hook.attach` (src/hook.py), runs ONE prefill forward on a prompt, and prints what it saw at each
residual write-back. It demonstrates two things at once:
  1. the seam actually fires (every layer × {post_attn, out}), and
  2. the residual norm grows with depth (note §3) — visible in the printed table.

This is a RUN-time check: it needs torch + model access, and (unlike the fp32/CPU parity
gate) it exercises the real device/dtype path. Nothing here modifies the stream — the
probe returns None (observe only).

    python src/hook_test.py
    python src/hook_test.py --prompt "What is coffee?" --site out
    python src/hook_test.py --model Qwen/Qwen2.5-0.5B-Instruct --device cpu
"""

from __future__ import annotations

import argparse
import sys

import torch

import loader
import router
from hook import attach


def pick_device(req: str) -> str:
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str):
    if device.startswith("cuda"):
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def main() -> None:
    p = argparse.ArgumentParser(description="Inject a probe hook and inspect one prefill.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--prompt", default="Explain RoPE in one line.")
    p.add_argument("--site", default="both", choices=["post_attn", "out", "both"],
                   help="which residual write-back(s) to report")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    dtype = pick_dtype(device)

    # 1. LOAD the model the normal way (loader → router → family.load).
    print(f"[loading {args.model} on {device} ({dtype})]", file=sys.stderr)
    L = loader.load(args.model)
    if L.model_type != "qwen2":
        sys.exit(f"[hook_test targets qwen2; got model_type={L.model_type!r}]")
    fam = router.route(L.model_type)
    model = fam.load(L.raw_config, L.weights, L.fmt, device, dtype)
    model.eval()

    # 2. DEFINE the probe: record last-token stats of the residual at each write-back.
    #    `act` is (B, T, d); the last token (act[:, -1, :]) is the generation point.
    records = []

    def probe(act, ctx):
        if args.site != "both" and ctx.site != args.site:
            return None                                   # self-filter by site
        last = act[:, -1, :].float()                      # (B, d)
        records.append((ctx.layer_idx, ctx.n_layers, ctx.site, ctx.seqlen,
                        tuple(act.shape), float(last.norm(dim=-1).mean()),
                        float(last.abs().max())))
        return None                                       # observe only — never modify

    # 3. ENCODE the prompt (chat template — this is an instruct model) and run ONE prefill
    #    forward with the probe attached.
    ids = L.tokenizer.apply_chat(args.prompt)
    input_ids = torch.tensor([ids], device=device)

    with attach(model, probe):
        with torch.no_grad():
            logits, _ = model(input_ids)

    # 4. REPORT.
    top = int(logits[0, -1].argmax())
    n_sites = len({r[2] for r in records})
    print(f"\nprompt : {args.prompt!r}")
    print(f"tokens : {len(ids)}   next-token: {top} -> {L.tokenizer.decode([top])!r}")
    print(f"hook fired {len(records)} times "
          f"({records[0][1]} layers × {n_sites} site(s), reporting site={args.site!r})\n")

    print(f"{'layer':>5} {'site':>9} {'shape':>18} {'‖resid‖_last':>13} {'absmax':>9}")
    print("-" * 58)
    for layer, _n, site, _sl, shape, norm, absmax in records:
        print(f"{layer:>5} {site:>9} {str(shape):>18} {norm:>13.2f} {absmax:>9.2f}")

    outs = [(r[0], r[5]) for r in records if r[2] == "out"]
    if len(outs) >= 2:
        print(f"\n‖resid‖ at 'out': layer {outs[0][0]}={outs[0][1]:.1f}  →  "
              f"layer {outs[-1][0]}={outs[-1][1]:.1f}   (grows with depth — note §3)")


if __name__ == "__main__":
    main()
