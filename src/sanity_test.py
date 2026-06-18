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

from compare_logits import compare, CompareError

# (label, HF model id) — one representative checkpoint per family
MODELS = [
    ("qwen2",   "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen3",   "Qwen/Qwen3-0.6B"),
    ("gemma2",  "google/gemma-2-2b-it"),
    ("gemma3",  "google/gemma-3-1b-it"),
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
            # ran end-to-end: PASS if logits agree, else CHECK (a *logits* divergence,
            # not an error — the model loaded and ran fine).
            status, cos = ("PASS" if r["ok"] else "CHECK"), r["cosine"]
            print(f"  reference: {r['reference']}")
            print(f"  our: {r['our_top']} {r['our_text']!r}  |  ref: {r['ref_top']} {r['ref_text']!r}")
            tok = {True: "tok✅", False: "tok❌", None: "tok–"}[r.get("tok_ok")]
            print(f"  cosine {cos:.6f} | max|Δ| {r['max_abs']:.4f} | top-5 {r['top5_match']} | {tok}"
                  f"  ->  {'PASS ✅' if r['ok'] else 'CHECK ❌ (logits diverged)'}", flush=True)
        except CompareError as e:
            # tag it: a download/access/load problem vs a forward/compare (logits) problem
            kind = "DOWNLOAD/LOAD" if e.is_load else "LOGITS/RUN"
            status, cos = f"ERROR[{e.phase}]", float("nan")
            print(f"  {kind} ERROR in '{e.phase}': {type(e.cause).__name__}: {e.cause}", flush=True)
        except Exception as e:                       # anything unexpected — keep going
            status, cos = "ERROR[unexpected]", float("nan")
            traceback.print_exc()
            print(f"  UNEXPECTED ERROR: {type(e).__name__}: {e}", flush=True)
        rows.append((label, model, status, cos))
        gc.collect()

    print("\n===== SUMMARY =====")
    for label, model, status, cos in rows:
        cs = f"cos={cos:.6f}" if cos == cos else "cos=   n/a  "   # cos != cos → NaN (errored)
        print(f"  {status:18s} {label:8s} {cs}  {model}")
    all_ok = all(s == "PASS" for _, _, s, _ in rows)
    print("\nALL PASS ✅" if all_ok else "\nSOME FAILED ❌  (CHECK = logits diverged, ERROR = could not run)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
