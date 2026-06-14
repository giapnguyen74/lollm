# Qwen2 — modeling notes

A decoder-only transformer in the Llama lineage. This package is self-contained;
read it in order: `config.py` (dims) → `blocks.py` (components) → `modeling_qwen2.py`
(architecture + forward) → `weights.py` (checkpoint loading).

## Forward flow (`Qwen2Model.forward`)

```
1. EMBED          token ids → vectors                 (B,T) → (B,T,H)
2. POSITIONS      abs positions (offset by KV cache) → RoPE cos/sin
3. DECODER STACK  N × DecoderLayer (attention + MLP), each growing its KV cache
4. FINAL NORM     RMSNorm
5. LM HEAD        hidden → vocab logits                (B,T,V)
```

Each `DecoderLayer` is two pre-norm residual sub-blocks:
`x = x + attn(norm(x))` then `x = x + mlp(norm(x))`.

## Changes vs the original GPT decoder

Baseline = GPT-2-style decoder: learned **absolute** position embeddings,
**LayerNorm**, full **multi-head attention**, a plain **GELU** MLP (two matrices).

| Aspect | GPT-2 baseline | Qwen2 | Where |
|---|---|---|---|
| Positions | learned absolute embeddings (added at input) | **RoPE** — rotate Q/K by position, every layer | `blocks.py` `RoPE` |
| Norm | LayerNorm (mean-center + scale + bias) | **RMSNorm** (scale only, no mean, no bias) | `blocks.py` `RMSNorm` |
| Attention | full MHA (n_kv = n_head) | **GQA** — fewer KV heads, shared across query groups | `blocks.py` `Attention` |
| MLP | GELU: `down(gelu(up(x)))` (2 matrices) | **SwiGLU**: `down(silu(gate(x))·up(x))` (3 matrices) | `blocks.py` `MLP` |
| Attn bias | yes (q/k/v/o) | **bias on q/k/v only**, none on o_proj (a Qwen2 quirk) | `blocks.py` `Attention` |
| Output head | separate (or tied) | **tied** to embeddings on small models | `weights.py` `load` |
| Inference | — | **KV cache** (store K/V per layer, decode one token at a time) | `blocks.py` `Attention` |

Norm *placement* is the same as GPT-2 (pre-norm). Everything else above is the
modern-LLM delta: RoPE + RMSNorm + GQA + SwiGLU.

## Notes

- Module names mirror HF, so the safetensors weight map is identity; the GGUF map
  (`blk.N.attn_q…`) lives in `weights.py`.
- Validated against transformers with `compare_logits.py` (same argmax, cosine ≈ 1).
