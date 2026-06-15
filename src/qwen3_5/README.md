# Qwen3.5 — modeling notes

A **hybrid** decoder: most layers are a linear-attention mixer (**Gated DeltaNet**),
every 4th is **gated full-attention**, and the FFN stays a dense SwiGLU. One
`model_type: "qwen3_5"` covers Qwen3.5 and Qwen3.6 (same arch, different sizes).
Read in order: `config.py` (dims) → `blocks.py` (components) → `modeling_qwen3_5.py`
(architecture + forward) → `weights.py` (loading); `kv.py` holds the cache and
`mtp.py` the opt-in speculative head.

## Forward flow (`Qwen3_5Model.forward`)

```
1. EMBED          token ids → vectors                       (B,T) → (B,T,H)
2. CACHE+POS      positions offset by cache.seen_tokens → partial-RoPE cos/sin
3. DECODER STACK  N × DecoderLayer — full-attention every 4th layer, Gated
                  DeltaNet otherwise; each reads/writes its own cache slot
4. FINAL NORM     RMSNorm
5. LM HEAD        hidden → vocab logits                      (B,T,V)
```

Each `DecoderLayer` is two pre-norm residual sub-blocks: `x = x + mixer(norm(x))`
then `x = x + mlp(norm(x))`, where `mixer` is `self_attn` (full) or `linear_attn` (GDN).

## Changes vs Qwen3 (the dense predecessor)

Qwen3 = uniform GQA full-attention every layer, RoPE over the full head_dim, plain
attention output, a single growing KV cache. Qwen3.5 keeps RMSNorm, QK-norm, SwiGLU,
tied embeddings, and adds:

| Aspect | Qwen3 | Qwen3.5 | Where |
|---|---|---|---|
| Layer schedule | every layer full-attention | **hybrid**: 1 full-attention per 4, rest linear | `config.py` `layer_types` |
| Linear mixer | — | **Gated DeltaNet** — causal conv + gated delta-rule recurrence | `blocks.py` `GatedDeltaNet` |
| Attention output | `o_proj(o)` | **output gate**: `o * sigmoid(gate)`, gate packed in `q_proj` (×2) | `blocks.py` `GatedAttention` |
| RoPE coverage | full head_dim | **partial** — only `head_dim · partial_rotary_factor` rotated | `blocks.py` `RoPE` |
| Cache | one growing KV | **per-layer**: (K,V) for full, fixed `(conv,recurrent)` for GDN | `kv.py` `Qwen3_5Cache` |
| Position source | KV length | **`cache.seen_tokens`** counter (layer 0 has no KV to probe) | `modeling` |
| Speculative head | — | **opt-in MTP/Eagle** head drafting token t+2 | `mtp.py` |

## Gated DeltaNet in one breath

Per head, a recurrent state `S` (head_k × head_v) updated by the gated delta rule:
`S ← S·exp(g); S ← S + kᵀ(v − ⟨S,k⟩)β; out = ⟨S,q⟩`. Two equivalent kernels: a
**chunked** form (one matmul per chunk, sequential only at chunk boundaries) used for
prefill, and a **recurrent** form (clean step-by-step) used for cached single-token
decode. q/k are L2-normalized; `g` is a per-head decay from `A_log`/`dt_bias`, `β` a
sigmoid gate. A depthwise causal Conv1d over cat(q,k,v) runs first; its last `kernel−1`
columns are the conv-state carried across steps.

## Notes

- The checkpoint is the multimodal VL model, so the text LM lives under
  `model.language_model.*`; `weights.py` rewrites that prefix (vision tower ignored).
- **HF only** — GGUF for Gated DeltaNet isn't standardized, so the loader hard-fails
  rather than guess.
- `output_gate_type` is parsed but unused: the reference hardcodes sigmoid and ignores
  the field (the 27B sets it "swish"); we match the reference (transformers 4.57.1).
- Parity: `compare_logits.py` on Qwen3.5-4B — cosine ≈ 1, argmax match (fp32 CPU).
  MTP has no transformers reference, so it's validated structurally only.
- **MoE variant** lives in the sibling `qwen3_5_moe/` family (sparse FFN, same backbone).
