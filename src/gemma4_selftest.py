"""
gemma4_selftest.py — fast, offline correctness checks for the gemma4 text decoder.

No download (random weights, tiny config, CPU fp32). Targets the Gemma4-specific paths
the short-prompt parity gate wouldn't exercise:

  1. PREFILL == INCREMENTAL DECODE over a >sliding_window sequence — proves the KV cache
     AND the shared-KV reuse (donor → shared layers) across steps.
  2. SHARED-KV STRUCTURE — shared layers have no k/v projections; donors are the last
     non-shared layer of each type.
  3. DOUBLE-WIDE MLP — only shared-KV layers get the 2× intermediate size.
  4. PROPORTIONAL RoPE width — global rope table is full global_head_dim wide.
  5. FINAL SOFT-CAP — |logits| < final_logit_softcapping.
  6. forward shape.

    python src/gemma4_selftest.py
"""

from __future__ import annotations

import sys

import torch

from gemma4.config import Gemma4Config
from gemma4.modeling_gemma4 import Gemma4Model
from gemma4.blocks import build_inv_freq_proportional


LAYER_TYPES = ["sliding_attention", "sliding_attention", "full_attention",
               "sliding_attention", "full_attention", "sliding_attention",
               "sliding_attention", "full_attention"]   # 8 layers; last = full


def _cfg():
    return Gemma4Config(
        vocab_size=40, hidden_size=32, intermediate_size=16, num_hidden_layers=8,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, global_head_dim=16,
        rms_norm_eps=1e-6, sliding_window=4, layer_types=list(LAYER_TYPES),
        rope_theta_global=1_000_000.0, rope_theta_local=10_000.0,
        partial_rotary_factor_global=0.25, final_logit_softcapping=30.0,
        hidden_size_per_layer_input=4, vocab_size_per_layer_input=40,
        num_kv_shared_layers=3, use_double_wide_mlp=True, tie_word_embeddings=True)


def _build(seed=0):
    torch.manual_seed(seed)
    cfg = _cfg()
    model = Gemma4Model(cfg)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("norm.weight"):
                p.copy_(1.0 + torch.randn_like(p) * 0.05)   # near 1, but non-trivial
    return model.eval(), cfg


def check_prefill_equals_decode():
    model, cfg = _build()
    T = 10  # > sliding_window (4)
    ids = torch.randint(0, cfg.vocab_size, (1, T))
    with torch.no_grad():
        full, _ = model(ids)
        outs, cache = [], None
        for i in range(T):
            lg, cache = model(ids[:, i:i + 1], cache)
            outs.append(lg[:, -1])
        inc = torch.stack(outs, dim=1)
    ok = torch.allclose(full, inc, atol=1e-4, rtol=1e-4)
    return ok, f"prefill==decode: max|Δ|={(full - inc).abs().max().item():.2e} (T={T}, shared-KV active)"


def check_shared_kv_structure():
    model, cfg = _build()
    first_shared = cfg.num_hidden_layers - cfg.num_kv_shared_layers  # 5
    bad = []
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        shared = i >= first_shared
        has_kv = hasattr(attn, "k_proj")
        if shared and has_kv:
            bad.append(f"layer{i} shared but has k_proj")
        if not shared and not has_kv:
            bad.append(f"layer{i} non-shared but missing k_proj")
    donors = [i for i in range(cfg.num_hidden_layers) if cfg.is_donor(i)]
    ok = not bad and donors == [3, 4]   # last sliding (3) + last full (4) before idx 5
    return ok, f"shared-KV: first_shared={first_shared}, donors={donors}, issues={bad or 'none'}"


def check_double_wide_mlp():
    model, cfg = _build()
    base = cfg.intermediate_size
    l0 = model.model.layers[0].mlp.gate_proj.out_features      # non-shared → base
    l5 = model.model.layers[5].mlp.gate_proj.out_features      # shared → 2×
    ok = l0 == base and l5 == 2 * base
    return ok, f"double-wide MLP: layer0={l0} (={base}), layer5={l5} (=2×{base})"


def check_proportional_rope_width():
    cfg = _cfg()
    inv = build_inv_freq_proportional(cfg.global_head_dim, cfg.rope_theta_global,
                                      cfg.partial_rotary_factor_global, torch.device("cpu"))
    rope_angles = int(cfg.partial_rotary_factor_global * cfg.global_head_dim // 2)
    width = 2 * inv.shape[0]
    nonzero = int((inv != 0).sum())
    ok = width == cfg.global_head_dim and nonzero == rope_angles
    return ok, f"proportional RoPE: emb width={width} (=global_head_dim {cfg.global_head_dim}), rotated pairs={nonzero} (={rope_angles})"


def check_final_softcap():
    model, cfg = _build()
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        logits, _ = model(ids)
    cap = cfg.final_logit_softcapping
    ok = bool((logits.abs() < cap).all())
    return ok, f"final soft-cap: max|logit|={logits.abs().max().item():.3f} < {cap}"


def check_forward_shape():
    model, cfg = _build()
    ids = torch.randint(0, cfg.vocab_size, (1, 5))
    with torch.no_grad():
        logits, _ = model(ids)
    ok = tuple(logits.shape) == (1, 5, cfg.vocab_size)
    return ok, f"forward shape: {tuple(logits.shape)}"


CHECKS = [
    ("prefill==decode", check_prefill_equals_decode),
    ("shared-KV",       check_shared_kv_structure),
    ("double-wide-mlp", check_double_wide_mlp),
    ("proportional-rope", check_proportional_rope_width),
    ("final-softcap",   check_final_softcap),
    ("forward-shape",   check_forward_shape),
]


def main():
    all_ok = True
    for name, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"ERROR: {type(e).__name__}: {e}"
        all_ok &= ok
        print(f"  {'PASS ✅' if ok else 'FAIL ❌'}  {name:18s} {detail}", flush=True)
    print("\nALL PASS ✅" if all_ok else "\nSOME FAILED ❌")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
