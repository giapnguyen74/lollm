"""
compare_logits.py — parity gate for diffusion_gemma against the REAL checkpoint.

Diffusion has no single "next token", so the gate compares the DENOISER LOGITS: prefill a real
prompt → one decoder pass over a fixed canvas → OUR logits vs the transformers reference `.logits`
(per-position argmax agreement + cosine ≈ 1). `--generate` additionally streams a real generation
via `generate_diffusion` so you can eyeball coherent text.

    python src/diffusion_gemma/compare_logits.py \
        --model google/diffusiongemma-26B-A4B-it --prompt "Why is the sky blue?"
    python src/diffusion_gemma/compare_logits.py --generate            # + real output

Device/dtype auto-detect like run.py: cuda→bfloat16, mps→float16, cpu→float32 (override with
`--device` / `--dtype`). Memory: the reference is ~26B params (bf16 ≈ 52GB, ~18GB quantized; fp32
doubles it — cpu/fp32 ≈ 100GB). `--dtype float32` is the strict gate. Our modules SHARE the
reference's weight tensors (`assign=True`), so they add ~no extra memory. Keep the prompt within the
sliding window (1024) — the long-context sliding-cache clip is a tracked WIP item
(debug/diffusion_gemma/PLANNING.md).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import DiffusionGemmaConfig                                   # noqa: E402
from modeling_diffusion_gemma import DecoderTextModel, EncoderTextModel   # noqa: E402


def _cfg_get(obj, name, default):
    """Pull a field from a generation-config object OR a nested dict, with a fallback."""
    v = getattr(obj, name, None)
    if v is None and isinstance(obj, dict):
        v = obj.get(name)
    return default if v is None else v


def pick_device(req):
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device):
    """Same policy as run.py: cuda→bf16, mps→fp16, cpu→fp32."""
    if device.startswith("cuda"):
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def main():
    ap = argparse.ArgumentParser(description="diffusion_gemma parity vs transformers (real checkpoint)")
    ap.add_argument("--model", default="google/diffusiongemma-26B-A4B-it")
    ap.add_argument("--prompt", default="Why is the sky blue?")
    ap.add_argument("--device", default="auto", help="auto → cuda / mps / cpu")
    ap.add_argument("--generate", action="store_true", help="also run generate_diffusion and print text")
    ap.add_argument("--max-new-canvases", type=int, default=2)
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = pick_dtype(device)

    # text-only parity → use the tokenizer (AutoProcessor would pull in PIL for the image path)
    from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion
    print(f"[loading {args.model} ({str(dtype).split('.')[-1]}) on {device}]", file=sys.stderr)
    ref = DiffusionGemmaForBlockDiffusion.from_pretrained(args.model, dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.model)

    # build OUR modules on `meta` (NO weight allocation) then ASSIGN the reference's tensors, so we
    # genuinely SHARE the reference weights. Building with real weights first would allocate a full
    # ~24B-param copy per module on the GPU (the 128 MoE experts dominate) → OOM (L-2).
    from blocks import default_inv_freq, proportional_inv_freq
    cfg = DiffusionGemmaConfig.from_hf(ref.config.to_dict())
    canvas_length = ref.config.canvas_length
    with torch.device("meta"):
        enc, dec = EncoderTextModel(cfg), DecoderTextModel(cfg)
    enc.load_state_dict(ref.model.encoder.language_model.state_dict(), strict=True, assign=True)
    dec.load_state_dict(ref.model.decoder.state_dict(), strict=True, assign=True)
    # RoPE `inv_freq` are computed (non-persistent) buffers → not in the state_dict, so materialize
    # them on the real device (otherwise they stay meta tensors and the forward fails).
    for m in (enc, dec):
        m.rope_sliding.inv_freq = default_inv_freq(cfg.rope_theta_local, cfg.head_dim).to(device)
        m.rope_full.inv_freq = proportional_inv_freq(
            cfg.rope_theta_global, cfg.global_head_dim, cfg.partial_rotary_factor_global).to(device)
    enc.eval()
    dec.eval()

    # encode a real prompt via the model's own chat template
    enc_in = tok.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=True,
                                     add_generation_prompt=True, return_dict=True, return_tensors="pt")
    ids = enc_in["input_ids"].to(device)
    print(f"[prompt → {ids.shape[1]} tokens]", file=sys.stderr)

    # ── parity: decoder logits on a fixed canvas ──
    torch.manual_seed(0)
    canvas = torch.randint(0, cfg.vocab_size, (1, canvas_length), device=device)
    with torch.no_grad():
        cache = enc(ids, return_cache=True)[1]
        ours = dec.to_logits(dec(canvas, cache)).float()
        refl = ref(input_ids=ids, decoder_input_ids=canvas).logits.float()

    cos = torch.nn.functional.cosine_similarity(ours.flatten(), refl.flatten(), dim=0).item()
    mad = (ours - refl).abs().max().item()
    agree = (ours.argmax(-1) == refl.argmax(-1)).float().mean().item()
    print(f"\nparity — decoder logits on a {canvas_length}-token canvas:")
    print(f"  cosine            = {cos:.6f}")
    print(f"  max|Δ|            = {mad:.3e}")
    print(f"  argmax agreement  = {agree * 100:.2f}%")
    ok = cos > 0.999 and agree > 0.99
    print("  PASS ✓" if ok else "  FAIL ✗  (try --dtype float32 for a stricter check)")

    # ── optional: a real generation, to eyeball coherent output ──
    if args.generate:
        from sampler import EntropyBoundSampler, StableAndConfidentStopping
        from generate import generate_diffusion
        gc = ref.generation_config
        sc = getattr(gc, "sampler_config", None)
        sampler = EntropyBoundSampler(_cfg_get(sc, "entropy_bound", 0.1), canvas_length, cfg.vocab_size)
        stop = StableAndConfidentStopping(_cfg_get(gc, "stability_threshold", 1),
                                          _cfg_get(gc, "confidence_threshold", 0.005))
        with torch.no_grad():
            out = generate_diffusion(enc, dec, sampler, stop, ids,
                                     max_new_canvases=args.max_new_canvases,
                                     max_denoising_steps=_cfg_get(gc, "max_denoising_steps", 48),
                                     t_min=_cfg_get(gc, "t_min", 0.4), t_max=_cfg_get(gc, "t_max", 0.8),
                                     eos_ids=_cfg_get(gc, "eos_token_id", None))
        print("\n[generated]\n" + tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
