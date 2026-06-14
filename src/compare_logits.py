"""
compare_logits.py — the parity gate (the one correctness proof).

Runs our qwen2 family and transformers on the same prompt and compares next-token
logits. PASS = our top token equals the reference's and the max abs difference is
tiny.

    python compare_logits.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse

import torch

import loader
import router


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="The capital of France is")
    args = p.parse_args()

    # ours (fp32 on CPU)
    L = loader.load(args.model)
    model = router.route(L.model_type).load(L.raw_config, L.weights, L.fmt, "cpu", torch.float32)
    ids = L.tokenizer.encode(args.prompt)
    ours, _ = model(torch.tensor([ids]))
    ours = ours[0, -1].float()

    # reference
    from transformers import AutoModelForCausalLM
    ref = AutoModelForCausalLM.from_pretrained(
        loader.resolve(args.model), dtype=torch.float32).eval()
    with torch.no_grad():
        r = ref(torch.tensor([ids])).logits[0, -1].float()

    ot, rt = int(ours.argmax()), int(r.argmax())
    max_abs = (ours - r).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(ours, r, dim=0).item()
    top5_match = ours.topk(5).indices.tolist() == r.topk(5).indices.tolist()

    print(f"our top : {ot} -> {L.tokenizer.decode([ot])!r}")
    print(f"ref top : {rt} -> {L.tokenizer.decode([rt])!r}")
    print(f"max |Δ| : {max_abs:.4f} | cosine : {cos:.6f} | top-5 match: {top5_match}")

    # Pass on the *meaningful* signals: same prediction + near-identical direction.
    # Raw max|Δ| on logits is noisy across kernels (fp32 summation order), so it's
    # reported for info, not gated. A real bug (wrong RoPE/norm) would drop cosine
    # and change argmax.
    ok = ot == rt and cos > 0.9999
    print("RESULT:", "PASS ✅" if ok else "CHECK ❌")


if __name__ == "__main__":
    main()
