# Qwen3 — modeling notes

A decoder-only transformer, the Qwen2 design plus **QK-norm** and **no QKV bias**.
Read in order: `config.py` → `blocks.py` → `modeling_qwen3.py` → `weights.py`.

## Forward flow

Same as Qwen2: EMBED → POSITIONS+RoPE → DECODER STACK (attention + MLP) → FINAL
NORM → LM HEAD. Each `DecoderLayer` is pre-norm residual: `x + attn(norm(x))`,
then `x + mlp(norm(x))`.

## Changes vs the original GPT decoder

Baseline = GPT-2-style: learned absolute positions, LayerNorm, MHA, GELU MLP.

| Aspect | GPT-2 | Qwen3 | Where |
|---|---|---|---|
| Positions | learned absolute | RoPE | `blocks.py` |
| Norm | LayerNorm | RMSNorm | `blocks.py` |
| Attention | MHA | **GQA** | `blocks.py` |
| **QK-norm** | none | **RMSNorm on Q and K per head, before RoPE** | `blocks.py` `Attention` |
| Attn bias | yes | **none** (Qwen2 had q/k/v bias; Qwen3 drops it) | `blocks.py` |
| MLP | GELU | SwiGLU | `blocks.py` |
| Output head | separate | tied (small models) | `weights.py` |

## Changes vs Qwen2 (the immediate predecessor)

Just two: **+ QK-norm** (`q_norm`/`k_norm`, RMSNorm over `head_dim` applied to each
head's Q/K before RoPE — Qwen3's training stabilizer) and **− QKV bias** (Qwen2's
q/k/v projections had a bias; Qwen3 has none). Everything else — GQA, SwiGLU, RoPE,
plain RMSNorm, tied embeddings — is identical, which is why this family is mostly a
copy of `qwen2/` with a tweaked `Attention`.

## Notes

- New tensors vs Qwen2: `self_attn.q_norm.weight`, `self_attn.k_norm.weight`
  (GGUF: `attn_q_norm` / `attn_k_norm`). Missing them at load fails loud (strict
  set + shape checks in the streaming loader).
- Qwen3-MoE (fine-grained experts, no shared expert) shares this base + an MoE FFN —
  a follow-on, not in this dense family.
- Validate with `compare_logits.py` (same argmax, cosine ≈ 1).
