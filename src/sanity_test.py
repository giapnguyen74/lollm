"""
sanity_test.py — logit-parity check across families, in one run.

For each model below: load our engine + the transformers reference (fp32 on CPU), feed the
same prompt, and check the next-token argmax matches and cosine ≈ 1. Models are run one at
a time and freed between runs, so peak memory stays near the largest single model (still
heavy — these are full checkpoints in fp32; swap in smaller ids like
`Qwen/Qwen2.5-0.5B-Instruct` for a faster smoke).

    python sanity_test.py
    python sanity_test.py --prompt "Once upon a time"
    python sanity_test.py --only qwen3_5
"""

from __future__ import annotations

import argparse
import gc
import sys
import traceback

from compare_logits import compare

# (label, HF model id) — one representative checkpoint per family
MODELS = [
    ("qwen2",   "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen3",   "Qwen/Qwen3-0.5B"),
    ("qwen3_5", "Qwen/Qwen3.5-0.5B"),
    ("gemma2", "google/gemma-2-1b-it"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--only", default=None, help="run just one family label (e.g. qwen3_5)")
    args = p.parse_args()

    models = [m for m in MODELS if args.only is None or m[0] == args.only]
    if not models:
        sys.exit(f"no family matches --only {args.only!r} (have: {[m[0] for m in MODELS]})")

    rows = []
    for label, model in models:
        print(f"\n===== {label}: {model} =====", flush=True)
        try:
            r = compare(model, args.prompt)
            print(f"  reference: {r['reference']}")
            print(f"  our: {r['our_top']} {r['our_text']!r}  |  ref: {r['ref_top']} {r['ref_text']!r}")
            print(f"  cosine {r['cosine']:.6f} | max|Δ| {r['max_abs']:.4f} | top-5 {r['top5_match']}"
                  f"  ->  {'PASS ✅' if r['ok'] else 'CHECK ❌'}", flush=True)
            rows.append((label, model, r["ok"], r["cosine"]))
        except Exception as e:                       # keep going so one failure doesn't hide the rest
            traceback.print_exc()
            print(f"  ERROR: {e}", flush=True)
            rows.append((label, model, False, float("nan")))
        gc.collect()

    print("\n===== SUMMARY =====")
    for label, model, ok, cos in rows:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:8s} cos={cos:.6f}  {model}")
    all_ok = all(ok for _, _, ok, _ in rows)
    print("\nALL PASS ✅" if all_ok else "\nSOME FAILED ❌")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
