# Roadmap — untimed ideas & known problems

Bigger directions and standing problems we **don't have a timeline for** yet. Near-term,
scoped work lives in [TODOS.md](./TODOS.md); resolved gotchas become lessons in
[LESSONS.md](./LESSONS.md).

**IDs are stable** — `R-#` never renumbers or gets reused. Status: 💡 idea · 🔭 later ·
🟡 in progress · ✅ done. No commitment implied by order.

| ID  | status | area        | title |
|-----|--------|-------------|-------|
| R-1 | 🟡 in progress | gemma4 / multimodal | Vision + audio towers for Gemma 4 (full multimodal) — scoped as TODOs T-6/T-7/T-8 |
| R-2 | 🔭 later | new family  | `llama` family |
| R-3 | 🟡 in progress | gguf        | GGUF MoE (stacked experts) + validate gemma GGUF metadata to lift the hard-fail — scoped as TODOs T-3/T-4 |
| R-4 | ✅ done | tokenizer | Dropped `AutoTokenizer` — BPE (Qwen) + SPM (Gemma) parity-verified |
| R-5 | 💡 idea  | perf        | KV-cache & decode perf cliffs (O(T²) cat, full-KV on sliding layers, batch-1, eager GGUF dequant) |
| R-6 | 💡 idea  | perf        | Gemma attention → SDPA/flash (gemma3/4) or a fused Gemma kernel |
| R-7 | 💡 idea  | inference   | Chat-template provenance edge cases (README-only conventions, config-vs-jinja, GGUF metadata) |
| R-8 | 💡 idea  | quant       | NVFP4 (+ MXFP4) — 4-bit FP microscaling: dequant-to-fp16 (in vision) + opt-in Blackwell kernel (stretch) |

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

## R-4 · Drop `transformers.AutoTokenizer` (safetensors path) 🟡

**Done:** the safetensors path now tokenizes on its own (`tokenization.py`, no
`transformers`). `HFTokenizer(path)` sniffs the files and returns a `BPETokenizer`
(byte-level BPE, from `tokenizer.json` / `vocab.json`+`merges.txt`) or `SPMTokenizer`
(SentencePiece, from `tokenizer.model` — a minimal protobuf parser), reading specials /
eos / bos / chat-template from `tokenizer_config.json` (+ `chat_template.jinja`). Reuses
the GGUF engines (now with `from_gguf` / `from_hf` constructors).

**Status: ✅ done — both engines parity-verified vs `AutoTokenizer`** (5/5 raw incl. CJK,
emoji, code, multi-space, multiline, **and** the chat-templated path), via
`debug/tok_parity.py`.

Two Gemma bugs were found and fixed along the way: (1) **missing BOS** (now prepended when
`add_bos_token`; `apply_chat` doesn't double-add since the template emits `bos_token`), and
(2) **`add_dummy_prefix`** — model-dependent (Llama-2 True, **Gemma False**), so the leading
`▁` must be parsed from the SP `normalizer_spec`, not hardcoded. The greedy-merge engine
matched every non-prefix token, confirming Gemma's SP model is **BPE-style** (no Viterbi
needed). `compare_logits` can't catch tokenizer bugs (same ids fed to both models) — the
dedicated `debug/tok_parity.py` did.

**Verify:** compare `HFTokenizer(path).encode(text)` to `AutoTokenizer(path)(text)` on a
batch of strings (incl. the chat-templated prompt) for each family. Note: the
`compare_logits` gate does **not** catch tokenizer bugs (it encodes once and feeds the
*same* ids to both models), so this needs its own check.

## R-5 · KV-cache & decode perf cliffs
`*/kv.py` grows via `torch.cat` each step (O(T²) copy); Gemma local layers cache the
**full** K/V despite only needing the last `sliding_window`; `generate.py` is batch-1;
`loader._load_gguf` dequantizes eagerly (so "streaming peak ≈ steady" holds for
safetensors, not GGUF). Fine at study lengths; bites long context. (The eager-GGUF-dequant
piece is now scoped as **TODO T-5**.)

## R-6 · Faster Gemma attention
`attention.torch_attention_with_scale` is eager (materializes the full scores matrix, no
FlashAttention). gemma2 *must* stay manual (tanh attn soft-cap, which SDPA can't express);
**gemma3/gemma4 could** route through `scaled_dot_product_attention(scale=, attn_mask=)`,
or a fused Gemma Triton kernel (scale + QK-norm + band [+ softcap]). Validate any optimized
path against the parity gate.

## R-7 · Chat-template provenance edge cases
Templates ship inconsistently: inlined in `tokenizer_config.json`, as a standalone
`chat_template.jinja` (now downloaded — see LESSONS.md L-4), or **only described in the repo
README** (can't auto-apply). Also: config-vs-jinja disagreements, and the GGUF
`tokenizer.chat_template` metadata path. No general fix; may need per-model overrides.
Current behavior: a model with **no** template hard-fails in chat mode (`run.py`), and
`--no-chat` is the explicit bypass to raw prompting — so the README-only-template case
surfaces loudly instead of silently degenerating.

## R-8 · NVFP4 (+ MXFP4) — 4-bit floating-point microscaling

NVIDIA's hardware-accelerated 4-bit float format for LLMs (Blackwell). Worth splitting
into two layers that land very differently against our vision — do the first, gate the
second.

**The format (one sentence).** Weights stored as **E2M1** FP4 (1 sign · 2 exp · 1
mantissa), in **blocks of 16** elements that each carry an **FP8 E4M3** scale, plus a
per-tensor FP32 scale → ~4.5 effective bits/element. Sibling **MXFP4** (what gpt-oss
ships) is the same idea with block-32 and a **power-of-2 (E8M0)** scale; NVFP4's smaller
block + FP8 scale is the more accurate of the two.

### Layer 1 — dequantize it ourselves (IN VISION — the real item)
A new quant format alongside the GGUF K-quants, in the existing seam: extend `dequant.py`
to decode the E2M1 nibbles × per-block E4M3 scale × per-tensor scale → fp16, then run
through the unchanged forward (same "dequant → run fp16" path as GGUF). High study value:
our K-quants are **integer superblock**; this is **floating-point microblock** — a
genuinely different design we don't teach yet. Sets up a clean three-way contrast in
`quantization.md`: K-quant (int / superblock / fp16 scale) vs MXFP4 (FP4 / block-32 / E8M0)
vs NVFP4 (FP4 / block-16 / E4M3). Doing the micro-block machinery once covers **both** FP4
formats.
- **Provenance / loader seam:** NVFP4 checkpoints ship as **safetensors + a quant config**
  (NVIDIA ModelOpt / compressed-tensors; `nvidia/` namespace — Nemotron, DeepSeek-V3.2),
  **not** GGUF. The loader currently branches plain-safetensors vs GGUF; this adds a third
  "safetensors-but-quantized" path: read the quant config, dequant on load. Hard-fail on an
  unrecognized quant scheme (never guess), per house rule.
- **Gate:** dequant correctness — compare our dequantized weights / logits against the
  reference (ModelOpt or `transformers`), same parity bar as everything else.

### Layer 2 — run NVFP4 on the tensor cores (OPT-IN STRETCH — against the default grain)
The actual NVIDIA selling point is the **Blackwell tensor-core matmul** (~2× FP8, ~4×
BF16). That needs Blackwell silicon + a custom CUDA/Triton kernel and fights our
"readable, PyTorch-only default" rule — so it belongs exactly where the Triton flash kernel
already lives: an **opt-in, hardware-gated** path (cf. `LOLLM_ATTN`), never the default,
validated against the parity gate. **Caveat that decides the investment:** unlike GGUF
(where dequant-to-fp16 still buys a smaller download + running GGUF-only models on any
device), NVFP4's entire payoff is the speedup — Layer 1 alone gives accuracy-study value
and a smaller checkpoint but **no speed** (we upcast to fp16 to run). Only pursue Layer 2
when actually targeting Blackwell hardware.
