"""
compare_logits.py — the parity gate (the one correctness proof).

Runs our family and `transformers` on the same prompt and compares next-token logits.
PASS = our top token equals the reference's and the directions agree (cosine ≈ 1).

    python compare_logits.py --model Qwen/Qwen2.5-0.5B-Instruct
    python compare_logits.py --model Qwen/Qwen3.5-4B            # hybrid GDN, fp32 CPU

The reference class is taken from the checkpoint's own `architectures` field rather than
hardcoding `AutoModelForCausalLM`. That matters for multimodal checkpoints like Qwen3.5:
its text LM lives under `model.language_model.*` and the correct reference is
`Qwen3_5ForConditionalGeneration` (text branch) — `AutoModelForCausalLM` would map to the
text-only `Qwen3_5ForCausalLM`, silently mis-load every text weight, and fail spuriously.

Memory note: a 4B model in fp32 is ~16 GB. We extract our logits and free our model
*before* building the reference so peak RSS stays near one model, not two.
"""

from __future__ import annotations

import argparse
import gc
import json
import os

import torch

import loader
import router


def _reference_class(path: str):
    """Resolve the transformers reference class from the checkpoint's `architectures`.

    Falls back to AutoModelForCausalLM when the named class isn't importable. Using the
    checkpoint's own architecture is what lets a multimodal wrapper (e.g.
    Qwen3_5ForConditionalGeneration) load its `model.language_model.*` text weights.
    """
    import transformers

    cfg = json.load(open(os.path.join(path, "config.json")))
    for arch in cfg.get("architectures", []) or []:
        cls = getattr(transformers, arch, None)
        if cls is not None:
            return cls, arch
    return transformers.AutoModelForCausalLM, "AutoModelForCausalLM"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="The capital of France is")
    args = p.parse_args()

    path = loader.resolve(args.model)

    # ── ours (fp32 on CPU) ──
    L = loader.load(args.model)
    model = router.route(L.model_type).load(L.raw_config, L.weights, L.fmt, "cpu", torch.float32)
    ids = L.tokenizer.encode(args.prompt)
    ours, _ = model(torch.tensor([ids]))
    ours = ours[0, -1].float().clone()
    del model, L                     # free ~16 GB before the reference loads
    gc.collect()

    # ── reference (same prompt, text branch for multimodal checkpoints) ──
    ref_cls, ref_name = _reference_class(path)
    print(f"[reference: {ref_name}]", flush=True)
    ref = ref_cls.from_pretrained(path, dtype=torch.float32).eval()
    with torch.no_grad():
        r = ref(torch.tensor([ids])).logits[0, -1].float()

    ot, rt = int(ours.argmax()), int(r.argmax())
    max_abs = (ours - r).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(ours, r, dim=0).item()
    top5_match = ours.topk(5).indices.tolist() == r.topk(5).indices.tolist()

    from tokenization import HFTokenizer
    decode = HFTokenizer(path).decode      # fresh tokenizer (we freed L above)
    print(f"our top : {ot} -> {decode([ot])!r}")
    print(f"ref top : {rt} -> {decode([rt])!r}")
    print(f"max |Δ| : {max_abs:.4f} | cosine : {cos:.6f} | top-5 match: {top5_match}")

    # Pass on the *meaningful* signals: same prediction + near-identical direction.
    # Raw max|Δ| on logits is noisy across kernels (fp32 summation order), so it's
    # reported for info, not gated. A real bug (wrong RoPE/norm) would drop cosine
    # and change argmax.
    ok = ot == rt and cos > 0.9999
    print("RESULT:", "PASS ✅" if ok else "CHECK ❌")


if __name__ == "__main__":
    main()
