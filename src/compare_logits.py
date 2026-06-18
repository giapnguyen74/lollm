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
import contextlib
import gc
import json
import os

import torch
import transformers

import loader
import router
from tokenization import HFTokenizer


class CompareError(Exception):
    """A parity-check failure tagged with the phase it happened in, so callers can tell a
    download/access problem from our forward, the reference forward, or the comparison."""

    # phases that are about *getting the model* vs *running/comparing* it
    LOAD_PHASES = {"download", "load-ours", "load-reference"}

    def __init__(self, phase: str, cause: Exception):
        self.phase = phase
        self.cause = cause
        self.is_load = phase in self.LOAD_PHASES
        super().__init__(f"{phase}: {type(cause).__name__}: {cause}")


@contextlib.contextmanager
def _phase(name: str):
    try:
        yield
    except Exception as e:
        raise CompareError(name, e) from e


def _reference_class(path: str):
    """Resolve the transformers reference class from the checkpoint's `architectures`.

    Falls back to AutoModelForCausalLM when the named class isn't importable. Using the
    checkpoint's own architecture is what lets a multimodal wrapper (e.g.
    Qwen3_5ForConditionalGeneration) load its `model.language_model.*` text weights.
    """
    cfg = json.load(open(os.path.join(path, "config.json")))
    for arch in cfg.get("architectures", []) or []:
        cls = getattr(transformers, arch, None)
        if cls is not None:
            return cls, arch
    return transformers.AutoModelForCausalLM, "AutoModelForCausalLM"


def compare(model_spec: str, prompt: str = "The capital of France is") -> dict:
    """Run our family + the transformers reference on `prompt`; return a result dict.

    Loads one model at a time (ours, freed, then the reference) so peak memory stays near a
    single model. PASS = same argmax token and cosine > 0.9999. Raises `CompareError`
    (tagged with the failing phase) on any download / load / forward problem.
    """
    with _phase("download"):                         # snapshot_download (network/404/gated)
        path = loader.resolve(model_spec)

    # ── ours (fp32 on CPU) ──
    with _phase("load-ours"):
        L = loader.load(model_spec)
        model = router.route(L.model_type).load(
            L.raw_config, L.weights, L.fmt, "cpu", torch.float32)
    with _phase("forward-ours"):
        ids = L.tokenizer.encode(prompt)
        ours, _ = model(torch.tensor([ids]))
        ours = ours[0, -1].float().clone()
    del model, L                     # free before the reference loads
    gc.collect()

    # ── reference (same prompt, text branch for multimodal checkpoints) ──
    with _phase("load-reference"):
        ref_cls, ref_name = _reference_class(path)
        ref = ref_cls.from_pretrained(path, dtype=torch.float32).eval()
    with _phase("forward-reference"):
        with torch.no_grad():
            r = ref(torch.tensor([ids])).logits[0, -1].float()
    del ref                          # free the reference before returning
    gc.collect()

    with _phase("compare"):
        decode = HFTokenizer(path).decode      # fresh tokenizer (we freed L above)
        ot, rt = int(ours.argmax()), int(r.argmax())
        cos = torch.nn.functional.cosine_similarity(ours, r, dim=0).item()
        # Tiny tokenizer guard: the gate feeds OUR `ids` to both models, so it can't see a
        # tokenizer bug (ours could mis-encode and logits would still match). Cross-check
        # our encoding against the transformers tokenizer as the oracle. Skips on GGUF
        # (no HF tokenizer dir). tok_ok is reported separately — it doesn't gate logits.
        tok_ok = None
        try:
            import transformers
            ref_ids = transformers.AutoTokenizer.from_pretrained(path)(prompt)["input_ids"]
            tok_ok = ref_ids == ids
        except Exception:
            pass
        # Gate on the *meaningful* signals: same prediction + near-identical direction. Raw
        # max|Δ| is noisy across kernels (fp32 summation order) — reported, not gated.
        return {
            "model": model_spec, "reference": ref_name,
            "our_top": ot, "our_text": decode([ot]),
            "ref_top": rt, "ref_text": decode([rt]),
            "max_abs": (ours - r).abs().max().item(),
            "cosine": cos,
            "top5_match": ours.topk(5).indices.tolist() == r.topk(5).indices.tolist(),
            "tok_ok": tok_ok,
            "ok": ot == rt and cos > 0.9999,
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
    tok = res["tok_ok"]
    print("tokenizer:", "OK ✅" if tok else ("MISMATCH ❌ (our encoding ≠ transformers)"
                                            if tok is False else "skipped (no HF tokenizer)"))
    print("RESULT:", "PASS ✅" if res["ok"] else "CHECK ❌")


if __name__ == "__main__":
    main()
