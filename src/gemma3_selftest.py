"""
gemma3_selftest.py — fast, offline correctness checks for the gemma3 family.

The `compare_logits` parity gate proves the math against `transformers`, but it needs
a gated download + a long prompt to exercise everything. These checks need **no
download** (random weights, tiny config, CPU fp32) and target the two paths the parity
gate's short prompt does NOT cover:

  1. PREFILL == INCREMENTAL DECODE — a full forward over T tokens must equal feeding
     the same tokens one-at-a-time through the KV cache. (Proves the cache + per-step
     masking, the path `generate` actually uses.)
  2. SLIDING-WINDOW MASK — a local layer must ignore keys older than `sliding_window`;
     a global layer must not. (The parity prompt is ~6 tokens « 512, so the window is
     never crossed there — this is the gap that let a broken window "pass".)

Plus a couple of cheap sanity checks (layer typing, forward shape).

    python src/gemma3_selftest.py
"""

from __future__ import annotations

import sys

import torch

from attention import torch_attention_with_scale
from gemma3.config import Gemma3Config
from gemma3.modeling_gemma3 import Gemma3Model


def _tiny_cfg(**over):
    base = dict(
        vocab_size=64, hidden_size=64, intermediate_size=128,
        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, rms_norm_eps=1e-6, query_pre_attn_scalar=16,
        sliding_window=4, sliding_window_pattern=4,        # globals at layers 3, 7
        rope_theta_global=1_000_000.0, rope_theta_local=10_000.0,
        rope_scaling_factor=1.0, tie_word_embeddings=True)
    base.update(over)
    return Gemma3Config(**base)


def _build(seed=0):
    torch.manual_seed(seed)
    cfg = _tiny_cfg()
    model = Gemma3Model(cfg)
    # Random (not zero) norm weights so RMSNorm actually scales — a stronger test than
    # the default zero-init (which makes every (1+w) == 1).
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("norm.weight"):
                p.copy_(torch.randn_like(p) * 0.1)
    return model.eval(), cfg


def check_prefill_equals_decode():
    model, cfg = _build()
    T = 16  # > sliding_window (4): forces the band mask to actually bite mid-sequence
    ids = torch.randint(0, cfg.vocab_size, (1, T))

    with torch.no_grad():
        full, _ = model(ids)                       # (1, T, V) one-shot prefill
        outs, cache = [], None
        for i in range(T):
            lg, cache = model(ids[:, i:i + 1], cache)   # one token at a time, cached
            outs.append(lg[:, -1])
        inc = torch.stack(outs, dim=1)             # (1, T, V)

    max_abs = (full - inc).abs().max().item()
    ok = torch.allclose(full, inc, atol=1e-4, rtol=1e-4)
    return ok, f"prefill==decode: max|Δ|={max_abs:.2e}  (T={T}, window={cfg.sliding_window})"


def check_sliding_window_mask():
    torch.manual_seed(1)
    B, H, D, W = 1, 2, 16, 4
    P = 12                                          # query at last pos; total_k = P+1
    scale = D ** -0.5
    q = torch.randn(B, H, 1, D)
    k = torch.randn(B, H, P + 1, D)
    v = torch.randn(B, H, P + 1, D)

    # Allowed window for the query at pos P is (P-W, P] → positions 9,10,11,12.
    # Position 0 is well outside it. Perturb position 0's K/V dramatically.
    k_dirty, v_dirty = k.clone(), v.clone()
    k_dirty[:, :, 0] += 50.0
    v_dirty[:, :, 0] += 50.0

    o_win_clean = torch_attention_with_scale(q, k, v, scale, sliding_window=W)
    o_win_dirty = torch_attention_with_scale(q, k_dirty, v_dirty, scale, sliding_window=W)
    win_unaffected = torch.allclose(o_win_clean, o_win_dirty, atol=1e-6)

    # Global (no window) MUST see the change — otherwise the test is vacuous.
    o_glob_clean = torch_attention_with_scale(q, k, v, scale, sliding_window=None)
    o_glob_dirty = torch_attention_with_scale(q, k_dirty, v_dirty, scale, sliding_window=None)
    glob_affected = not torch.allclose(o_glob_clean, o_glob_dirty, atol=1e-6)

    ok = win_unaffected and glob_affected
    return ok, (f"sliding mask: local-ignores-old-key={win_unaffected}, "
                f"global-sees-it={glob_affected}")


def check_causal_no_future():
    """A query at position i must not depend on keys at positions > i (full/global layer)."""
    torch.manual_seed(2)
    B, H, D, T = 1, 2, 16, 6
    scale = D ** -0.5
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    o1 = torch_attention_with_scale(q, k, v, scale)          # full causal
    k2, v2 = k.clone(), v.clone()
    k2[:, :, -1] += 50.0; v2[:, :, -1] += 50.0         # perturb the LAST key/value
    o2 = torch_attention_with_scale(q, k2, v2, scale)
    # Only the last query may change; all earlier queries must be identical.
    ok = torch.allclose(o1[:, :, :-1], o2[:, :, :-1], atol=1e-6)
    return ok, f"causal: future key leaks into past queries = {not ok}"


def check_layer_typing():
    _, cfg = _build()
    types = [cfg.is_global(i) for i in range(cfg.num_hidden_layers)]
    expect = [(i + 1) % cfg.sliding_window_pattern == 0 for i in range(cfg.num_hidden_layers)]
    ok = types == expect
    globals_at = [i for i, g in enumerate(types) if g]
    return ok, f"layer typing: globals at {globals_at} (pattern {cfg.sliding_window_pattern})"


def check_forward_shape():
    model, cfg = _build()
    ids = torch.randint(0, cfg.vocab_size, (1, 5))
    with torch.no_grad():
        logits, _ = model(ids)
    ok = tuple(logits.shape) == (1, 5, cfg.vocab_size)
    return ok, f"forward shape: {tuple(logits.shape)}"


CHECKS = [
    ("prefill==decode", check_prefill_equals_decode),
    ("sliding-window",  check_sliding_window_mask),
    ("causal",          check_causal_no_future),
    ("layer-typing",    check_layer_typing),
    ("forward-shape",   check_forward_shape),
]


def main():
    rows, all_ok = [], True
    for name, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as e:  # a crash is a failure
            ok, detail = False, f"ERROR: {type(e).__name__}: {e}"
        all_ok &= ok
        rows.append((name, ok, detail))
        print(f"  {'PASS ✅' if ok else 'FAIL ❌'}  {name:16s} {detail}", flush=True)
    print("\nALL PASS ✅" if all_ok else "\nSOME FAILED ❌")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
