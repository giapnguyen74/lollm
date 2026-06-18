# Roadmap — untimed ideas & known problems

Bigger directions and standing problems we **don't have a timeline for** yet. Near-term,
scoped work lives in [TODOS.md](./TODOS.md); resolved gotchas become lessons in
[LESSONS.md](./LESSONS.md).

**IDs are stable** — `R-#` never renumbers or gets reused. Status: 💡 idea · 🔭 later ·
🟡 in progress · ✅ done. No commitment implied by order.

| ID  | status | area        | title |
|-----|--------|-------------|-------|
| R-1 | 🔭 later | gemma4 / multimodal | Vision + audio towers for Gemma 4 (full multimodal) |
| R-2 | 🔭 later | new family  | `llama` family |
| R-3 | 🔭 later | gguf        | GGUF MoE (stacked experts) + validate gemma GGUF metadata to lift the hard-fail |
| R-4 | 🔭 later | tokenizer   | Drop the `transformers.AutoTokenizer` dependency on the safetensors path |
| R-5 | 💡 idea  | perf        | KV-cache & decode perf cliffs (O(T²) cat, full-KV on sliding layers, batch-1, eager GGUF dequant) |
| R-6 | 💡 idea  | perf        | Gemma attention → SDPA/flash (gemma3/4) or a fused Gemma kernel |
| R-7 | 💡 idea  | inference   | Chat-template provenance edge cases (README-only conventions, config-vs-jinja, GGUF metadata) |

---

## R-1 · Gemma 4 vision + audio towers
Text decoder is done and parity-verified. Full multimodal adds the vision tower
(~150M) + audio USM tower (~300M) + projectors, plus the processor/preprocessing seam
(see `multimodal-processors.md`). Large, self-contained phase.

## R-2 · `llama` family
Copy `qwen2/` → `llama/`; the main quirk is the GGUF Q/K permute (see LESSONS.md L-1).

## R-3 · GGUF MoE + lift the gemma hard-fail
Stacked-expert GGUF, and validate gemma2/gemma3/gemma4 GGUF metadata keys (norm `+1`
fold, attn scale, sliding window, QK-norm, dual/proportional RoPE, PLE) against
llama.cpp so `from_gguf` can stop hard-failing.

## R-4 · Drop `transformers.AutoTokenizer` (safetensors path)
The known "PyTorch-only" wart: the safetensors path still uses `AutoTokenizer`. The GGUF
path already tokenizes on its own; port that to safetensors.

## R-5 · KV-cache & decode perf cliffs
`*/kv.py` grows via `torch.cat` each step (O(T²) copy); Gemma local layers cache the
**full** K/V despite only needing the last `sliding_window`; `generate.py` is batch-1;
`loader._load_gguf` dequantizes eagerly (so "streaming peak ≈ steady" holds for
safetensors, not GGUF). Fine at study lengths; bites long context.

## R-6 · Faster Gemma attention
`attention.torch_attention_with_scale` is eager (materializes the full scores matrix, no
FlashAttention). gemma2 *must* stay manual (tanh attn soft-cap, which SDPA can't express);
**gemma3/gemma4 could** route through `scaled_dot_product_attention(scale=, attn_mask=)`,
or a fused Gemma Triton kernel (scale + QK-norm + band [+ softcap]). Validate any optimized
path against the `*_selftest.py` oracles.

## R-7 · Chat-template provenance edge cases
Templates ship inconsistently: inlined in `tokenizer_config.json`, as a standalone
`chat_template.jinja` (now downloaded — see LESSONS.md L-4), or **only described in the repo
README** (can't auto-apply). Also: config-vs-jinja disagreements, and the GGUF
`tokenizer.chat_template` metadata path. No general fix; may need per-model overrides.
