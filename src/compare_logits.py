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


def compare(model_spec: str, prompt: str = "The capital of France is") -> dict:
    """Run our family + the transformers reference on `prompt`; return a result dict.

    Loads one model at a time (ours, freed, then the reference) so peak memory stays near a
    single model. PASS = same argmax token and cosine > 0.9999.
    """
    path = loader.resolve(model_spec)

    # ── ours (fp32 on CPU) ──
    L = loader.load(model_spec)
    model = router.route(L.model_type).load(L.raw_config, L.weights, L.fmt, "cpu", torch.float32)
    ids = L.tokenizer.encode(prompt)
    ours, _ = model(torch.tensor([ids]))
    ours = ours[0, -1].float().clone()
    del model, L                     # free before the reference loads
    gc.collect()

    # ── reference (same prompt, text branch for multimodal checkpoints) ──
    ref_cls, ref_name = _reference_class(path)
    ref = ref_cls.from_pretrained(path, dtype=torch.float32).eval()
    with torch.no_grad():
        r = ref(torch.tensor([ids])).logits[0, -1].float()
    del ref                          # free the reference before returning
    gc.collect()

    from tokenization import HFTokenizer
    decode = HFTokenizer(path).decode      # fresh tokenizer (we freed L above)
    ot, rt = int(ours.argmax()), int(r.argmax())
    # Gate on the *meaningful* signals: same prediction + near-identical direction. Raw
    # max|Δ| is noisy across kernels (fp32 summation order) — reported, not gated.
    return {
        "model": model_spec, "reference": ref_name,
        "our_top": ot, "our_text": decode([ot]),
        "ref_top": rt, "ref_text": decode([rt]),
        "max_abs": (ours - r).abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(ours, r, dim=0).item(),
        "top5_match": ours.topk(5).indices.tolist() == r.topk(5).indices.tolist(),
        "ok": ot == rt and torch.nn.functional.cosine_similarity(ours, r, dim=0).item() > 0.9999,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="The capital of France is")
    args = p.parse_args()

    res = compare(args.model, args.prompt)
    print(f"[reference: {res['reference']}]")
    print(f"our top : {res['our_top']} -> {res['our_text']!r}")
    print(f"ref top : {res['ref_top']} -> {res['ref_text']!r}")
    print(f"max |Δ| : {res['max_abs']:.4f} | cosine : {res['cosine']:.6f} | "
          f"top-5 match: {res['top5_match']}")
    print("RESULT:", "PASS ✅" if res["ok"] else "CHECK ❌")


if __name__ == "__main__":
    main()
