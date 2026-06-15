# Qwen3.5-MoE — modeling notes

The `qwen3_5` hybrid backbone with the dense FFN replaced by a **sparse Mixture-of-
Experts** block. The token mixers (Gated DeltaNet + gated full-attention), partial RoPE,
QK-norm, cache, and MTP head are **identical** to `qwen3_5/` — only the per-layer MLP
changes. Read in order: `config.py` (dims + MoE fields) → `blocks.py` (components +
`SparseMoeBlock`) → `modeling_qwen3_5_moe.py` (architecture + FFN dispatch) → `weights.py`
(loading); `kv.py` holds the cache, `mtp.py` the opt-in speculative head.

## Forward flow (`Qwen3_5MoeModel.forward`)

```
1. EMBED          token ids → vectors                       (B,T) → (B,T,H)
2. CACHE+POS      positions offset by cache.seen_tokens → partial-RoPE cos/sin
3. DECODER STACK  N × DecoderLayer — full-attention every 4th layer, Gated DeltaNet
                  otherwise; FFN is a SparseMoeBlock on MoE layers, dense MLP otherwise
4. FINAL NORM     RMSNorm
5. LM HEAD        hidden → vocab logits                      (B,T,V)
```

Each `DecoderLayer`: `x = x + mixer(norm(x))` then `x = x + ffn(norm(x))`. `mixer` is
`self_attn` (full) or `linear_attn` (GDN); `ffn` is `SparseMoeBlock` or `MLP`. Both FFNs
share the `ffn(x) -> x` signature, so the residual update reads the same either way.

## Changes vs Qwen3.5 (the dense sibling)

Exactly one architectural change — the FFN — plus the config fields that drive it:

| Aspect | Qwen3.5 (dense) | Qwen3.5-MoE | Where |
|---|---|---|---|
| FFN | one SwiGLU MLP per layer | **router + N routed SwiGLU experts + shared expert** | `blocks.py` `SparseMoeBlock` |
| Routing | — | softmax → **top-k** experts, optional renorm (`norm_topk_prob`) | `blocks.py` |
| Shared expert | — | **always-on** SwiGLU scaled by `sigmoid(shared_expert_gate)` | `blocks.py` |
| Which layers | all dense | **MoE when `(i+1) % decoder_sparse_step == 0` and `i ∉ mlp_only_layers`** | `config.py` `is_moe_layer` |
| New config | — | `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `shared_expert_intermediate_size`, `norm_topk_prob`, `decoder_sparse_step`, `mlp_only_layers` | `config.py` |

Everything else — GDN, gated attention, partial RoPE, QK-norm, the per-layer cache,
tied embeddings, MTP — is copied verbatim (CONVENTIONS §2: a family is self-contained;
duplication is intentional).

## The MoE block in one breath

Per token: `p = softmax(gate(x))` over `num_experts`; keep the top-k and (optionally)
renormalize them to sum to 1; `y = Σ_k p_k · expert_k(x)`. The expert loop is masked —
only tokens routed to an expert pay for it. Then add the shared expert:
`y += sigmoid(shared_expert_gate(x)) · shared_expert(x)`. Each expert is a SwiGLU of
width `moe_intermediate_size`; the shared expert uses `shared_expert_intermediate_size`
(0 → no shared expert, branch disabled).

## Notes

- Experts are stored **per-expert** (`…mlp.experts.{e}.{gate,up,down}_proj.weight`),
  router as `…mlp.gate.weight`, shared as `…mlp.shared_expert.*` /
  `…mlp.shared_expert_gate.weight` — all reproduced by name, so the safetensors map
  stays identity. A *fused/stacked*-expert checkpoint trips the strict shape check;
  `weights.py` marks where to un-stack.
- `MODEL_TYPES = ["qwen3_5_moe"]` must match the checkpoint's `config.json["model_type"]`
  — adjust if it reports something else (e.g. `"qwen3_next"`).
- Multimodal text-LM prefix (`model.language_model.*`) and **HF-only** loading are
  inherited from `qwen3_5/`.
- Smoke test: `sanity_moe.py` (random-init tiny model — routing, FFN dispatch, prefill,
  cached decode). For real parity, run `compare_logits.py` against a checkpoint.
