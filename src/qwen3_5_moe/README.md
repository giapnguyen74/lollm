# Qwen3.5-MoE — modeling notes

> **Status:** loads and generates — verified on **Qwen/Qwen3.6-35B-A3B** (256 experts,
> top-k routed + 1 shared expert) on **CUDA** (bf16). The base LM ignores the checkpoint's
> `mtp.*` head, which is not required for correct next-token decoding.

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
| FFN | one SwiGLU MLP per layer | **router + N routed SwiGLU experts (fused) + shared expert** | `blocks.py` `SparseMoeBlock` / `FusedExperts` |
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
`y += sigmoid(shared_expert_gate(x)) · shared_expert(x)`. Each routed expert is a SwiGLU of
width `moe_intermediate_size`; the shared expert uses `shared_expert_intermediate_size`
(0 → no shared expert, branch disabled).

The routed experts are stored **fused** — `FusedExperts` keeps the checkpoint's two stacked
Parameters as-is rather than `num_experts` separate `Linear` modules, so the weight map stays
identity. Each expert slice is in standard `(out, in)` orientation, so `F.linear` applies it
with no transpose:

```
mlp.experts.gate_up_proj   (E, 2·moe_inter, hidden)   gate & up fused on the out axis
mlp.experts.down_proj      (E, hidden, moe_inter)
mlp.gate.weight            (E, hidden)                 router
mlp.shared_expert.*        per-tensor SwiGLU           always-on expert
mlp.shared_expert_gate.weight  (1, hidden)             its sigmoid gate
```
(Qwen3.6-35B-A3B: E=256, hidden=2048, moe_inter=512 → `gate_up_proj` (256, 1024, 2048),
`down_proj` (256, 2048, 512).)

## Notes

- Experts are **fused** (see above), reproduced by name so the safetensors map stays
  identity. The strict shape check in `weights.py` is what verifies the layout — a
  transposed or per-expert checkpoint fails loud there rather than mis-loading.
- `MODEL_TYPES = ["qwen3_5_moe"]` must match the checkpoint's `config.json["model_type"]`
  — confirmed for Qwen3.6-35B-A3B; adjust if another checkpoint reports a different string.
- A fully-sparse checkpoint has no dense `intermediate_size` (every layer is MoE);
  `config.py` falls back to `moe_intermediate_size`, used only if a layer is forced dense.
- Multimodal text-LM prefix (`model.language_model.*`) and **HF-only** loading are
  inherited from `qwen3_5/`.
- Smoke test: `sanity_moe.py` (random-init tiny model — routing, FFN dispatch, prefill,
  cached decode). End-to-end generation verified on Qwen3.6-35B-A3B (CUDA, bf16). MTP-based
  speculative decoding is not yet wired for this family (`run.py` gates `--mtp` to `qwen3_5`).
