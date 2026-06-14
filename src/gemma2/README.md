# Gemma2 — modeling notes

A decoder-only transformer that diverges from the Llama/Qwen line in several ways —
the most architecturally distinctive family here. Read in order: `config.py` →
`blocks.py` → `modeling_gemma2.py` → `weights.py`.

## Forward flow (`Gemma2Model.forward`)

```
1. EMBED + SCALE   token ids → vectors, then × √hidden_size      (B,T) → (B,T,H)
2. POSITIONS       abs positions (offset by KV cache) → RoPE cos/sin
3. DECODER STACK   N × DecoderLayer, ALTERNATING local (sliding) / global attention
4. FINAL NORM      (1+w) RMSNorm
5. LM HEAD + CAP   hidden → logits, then final-logit soft-cap     (B,T,V)
```

Each `DecoderLayer` is **sandwich-normed** — a norm before *and* after each sublayer:
`x = x + post_norm(attn(pre_norm(x)))` then `x = x + post_norm(mlp(pre_norm(x)))`
(4 norms per layer).

## Changes vs the original GPT decoder

Baseline = GPT-2-style decoder: learned **absolute** positions, **LayerNorm** (one
per sublayer), full **MHA**, plain **GELU** MLP, no soft-capping, global attention.

| Aspect | GPT-2 baseline | Gemma2 | Where |
|---|---|---|---|
| Positions | learned absolute | **RoPE** | `blocks.py` `RoPE` |
| Norm | LayerNorm | **RMSNorm scaled by `(1 + weight)`** (zero-init) | `blocks.py` `GemmaRMSNorm` |
| Norm placement | one norm / sublayer | **sandwich**: pre *and* post each sublayer (4/layer) | `modeling` `DecoderLayer` |
| Embeddings | as-is | **scaled by √hidden_size** before layer 0 | `modeling` `Gemma2Model.forward` |
| Attention | full MHA | **GQA** | `blocks.py` `GemmaAttention` |
| Attn scale | `1/√head_dim` | **`query_pre_attn_scalar^-0.5`** | `blocks.py` `GemmaAttention` |
| Attn logits | raw | **soft-capped**: `cap·tanh(scores/cap)` | `blocks.py` `GemmaAttention` |
| Attn span | global, all layers | **5… alternating local (sliding window) / global** (even=local, odd=global) | `modeling` `DecoderLayer` |
| MLP | GELU `down(gelu(up))` | **GeGLU** `down(gelu_tanh(gate)·up)` (3 matrices) | `blocks.py` `GemmaMLP` |
| Final logits | raw | **soft-capped** | `modeling` `Gemma2Model.forward` |
| Biases | yes | **none** (no attention/MLP bias) | `blocks.py` |
| Output head | separate (or tied) | **tied** to embeddings | `weights.py` `load` |

The two soft-cappings and the sandwich norm are the signature Gemma2 traits.
(Gemma**3** later dropped soft-capping in favour of QK-norm — that's a different family.)

## Notes

- **Attention is done manually**, not via `scaled_dot_product_attention`, because
  SDPA can't apply the attention logit soft-cap or the sliding-window mask cleanly.
- **Parity:** transformers' default SDPA path *skips* the attention soft-cap, so to
  match closely load the reference with `attn_implementation="eager"`. (The final
  soft-cap is model-level and applied by both.) `compare_logits.py` builds the
  reference with the **default** (SDPA) path, so a gemma2 run there will show a
  **lowered cosine / failing gate** — that's expected for this family, not a real
  regression. `compare_logits.py` is a **diagnostic** we use during real test work
  to spot and fix issues, not a hard pass/fail requirement: for a true gemma2 parity
  check, reload the reference with eager attention and confirm same argmax + cosine ≈ 1.
- **GGUF norm `+1` quirk (important).** Gemma's RMSNorm is `(1 + weight)·x̂`.
  llama.cpp **bakes the `+1` into the stored norm weights** (so they work with a
  plain `w·x̂` norm), while HF safetensors store the raw weight. Since our
  `GemmaRMSNorm` re-adds 1, `weights.py` **subtracts 1 from every `*norm.weight`
  on the GGUF path** — without this, every norm is off by 1 and the model emits
  garbage. (Symptom we hit: F32 `input_layernorm.weight` had cosine ≈ 0.70 vs the
  safetensors weight.)
- **GGUF norm names** (`post_attention_norm` / `ffn_norm` = pre-FFN / `post_ffw_norm`)
  match llama.cpp's scheme; strict load fails loudly if a name is wrong.
