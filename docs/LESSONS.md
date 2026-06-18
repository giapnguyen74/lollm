# Lessons — gotchas, explicitly recorded

Hard-won gotchas worth not re-learning. These are **not** the structural pattern for
adding a family (that's [../CONVENTIONS.md](../CONVENTIONS.md)) — they're the surprises
that cost us debugging time. Record new ones here as `L-#` (stable, never renumbered).

| ID  | area            | gotcha (one line) |
|-----|-----------------|-------------------|
| L-1 | gguf / weights  | Converters fold runtime conventions into the weights — undo them in `weights.py` |
| L-2 | loading / memory| `.to(device)` makes a second buffer — load lean and free the CPU source |
| L-3 | loading / debug | Load **buffers**, not just parameters; cosine is scale-invariant and hides magnitude bugs |
| L-4 | inference       | Chat templates ship inconsistently; the parity gate never exercises them |

---

## L-1 · Converted weights aren't in the original convention

**Do not assume "GGUF weights == HF weights, just quantized."** Converters *fold
runtime conventions into the weights* (constant folding), so a faithful loader must
know each format's per-arch quirks and undo them in `weights.py`. This is real and
has bitten us:

| Quirk | What the converter did | What the loader must do |
|---|---|---|
| **Gemma RMSNorm `+1`** | Gemma's norm is `(1+w)·x̂`; llama.cpp bakes `+1` into the stored norm weights so its plain `w·x̂` kernel works | on GGUF, **subtract 1** from every `*norm.weight` (else every norm is off by 1 → garbage) |
| **Llama Q/K permute** | llama.cpp permutes Q/K weights for its RoPE layout | un-permute Q/K on the GGUF path (Qwen2 isn't permuted → nothing to do) |
| **Tied embeddings** | `output.weight` omitted when tied | fill `lm_head` from `embed_tokens` |

Why converters do this: the runtime keeps **one generic kernel** (e.g. a single
RMSNorm, a single RoPE) and pushes the architectural constant into the weights at
conversion. The technique is everywhere (BatchNorm folding, scale folding, …) — but
it means on-disk weights encode runtime assumptions you have to reverse.

**How to catch it:** when a GGUF model outputs garbage but encode/decode are fine,
compare the GGUF-dequantized weights to the safetensors weights **per tensor by
cosine**. An *unquantized* (F32) tensor with cosine ≪ 1 (e.g. a norm at 0.70) is the
tell — that's a folded convention, not quantization error. Fix it in `weights.py`,
not in `modeling`.

---

## L-2 · Loading doubles memory unless you free the source

On Apple **MPS** (and CUDA), `model.to(device)` makes a **separate allocation** — a
PyTorch MPS/Metal tensor is a distinct buffer from the CPU tensor it came from, even
though Apple's memory is "unified." So you transiently hold **both** the CPU source
and the device copy. Two further traps inflate it:

- **Default fp32 params.** `nn.Linear`/`nn.Embedding` create fp32 params, so
  `load_state_dict` (copy) *casts the source up to fp32* — a full fp32 CPU duplicate
  before you ever reach `.to(fp16)`.
- **The source dict stays alive.** `Loaded.weights` (safetensors mmap refs, or the
  GGUF-dequantized fp32 tensors) is held for the whole run unless you drop it.

Mitigations we apply:

- **`load_state_dict(sd, strict=True, assign=True)`** — params *become* the loaded
  tensors (source dtype) instead of being copied into fresh fp32 params. No fp32
  inflation on load.
- **Free the source after moving to device** — `run.py` calls `L.weights.clear()`
  right after `model.to(device, dtype)`, so steady-state ≈ one copy (on device);
  only a brief transient during `.to` holds two.
- **Dequantize GGUF straight to fp16** (not fp32) — the model is fp16 on GPU anyway,
  so it's lossless vs the final weights and halves the CPU dequant buffer. (e.g.
  gemma-2-2b: the dequant drops ~10 GB → ~5 GB, so the load peak ~17 GB → ~10 GB.)
- **Stream weights onto the device** (done): build the model on `meta` (no
  allocation), then per parameter `pop` the source → move to device → assign (CPU
  source freed). CPU shrinks as the device grows, so the two never both hold a full
  copy → **load peak ≈ steady state**. (`weights.py` `load` takes `device, dtype`;
  `run.py` no longer does a bulk `.to`.)

Concrete numbers (gemma-2-2b, ~2.6B): fp32 dequant ≈10 GB + fp16 device copy ≈5 GB →
~17 GB; free source → ~6 GB steady (peak still ~17); fp16 dequant → ~10 GB peak;
**streaming → ~6 GB peak**.

### Streaming's tradeoffs (it isn't free)

- **Slower load** — many small host→device transfers + per-tensor Python/sync
  overhead instead of one bulk `.to`. You trade load latency for lower peak memory.
- **More complex, less readable** — builds on `meta` and assigns params by hand
  (touches `module._parameters`), bypassing `load_state_dict`. This cuts against the
  "readable study code" goal; it's the one spot where we accept extra machinery for
  a real benefit (fitting bigger models in limited RAM).
- **You re-implement the safeguards** `load_state_dict` gave for free — we manually
  check **missing** tensors and **shape per param** (a wrong name-map would otherwise
  silently assign a mismatched tensor). Don't drop those checks.
- **Tied weights need manual care** — after the move, share the Parameter explicitly
  (`lm_head.weight = embed_tokens.weight`); otherwise the embedding is duplicated.

Rule of thumb: **"unified memory" still means a second buffer for the device copy** —
load lean (assign, target dtype) and release the CPU source once it's on the device.

---

## L-3 · Load buffers, not just parameters (and cosine hides magnitude)

**A weight loader that only iterates `named_parameters()` silently skips persistent
`buffers`** — and some architectures ship non-trivial buffers in the checkpoint.
Gemma4's `layer_scalar` (a per-layer LayerScale-style residual scale, `register_buffer`,
saved in the safetensors, value ≠ 1.0) is the case that bit us: it was left at its
init default while the checkpoint's value went unloaded.

Why it was nasty to diagnose: a wrong *scalar* changes the residual's **magnitude but
not its direction**, so a per-layer / per-submodule **cosine** check (cosine is
scale-invariant) reads ~1.0 right up to the broken layer — yet the wrong magnitude
corrupts the residual-stream *proportions* of every later layer. Symptom: parity
perfect at layer 0, then a steady collapse to cosine ≈ 0, with the final logits
uncorrelated. (A layer-by-layer hidden-state comparison against the reference found it.)

Rules:
- After loading params, **materialize and load buffers too** (from the checkpoint when
  the name is present, else the module default) — especially required when building on
  `meta`, where an unloaded buffer stays a meta tensor and the forward fails.
- When debugging parity, **don't trust cosine alone** for a "this layer is fine" call —
  a scale error hides from it. Compare magnitudes (norms) when a sum-of-correct-vectors
  still drifts.

---

## L-4 · Chat templates ship inconsistently; the parity gate never tests them

A model's chat template can live in **three** different places, and there's no standard:
inlined in `tokenizer_config.json` (auto-loaded), as a standalone **`chat_template.jinja`**
file (only loaded if downloaded), or **only described in the repo README** (can't
auto-apply). The GGUF path carries its own `tokenizer.chat_template` metadata too.

This bit us: the loader's download allow-list didn't include `chat_template.jinja`, so
gemma-4-e2b-it had no template, `run.py` silently fell back to a **raw prompt**, and the
instruction-tuned model degenerated (`What is coffee` → `coffee coffee coffee…`).

The trap: **`compare_logits` uses raw prompts, so a green parity gate never exercises the
chat template** — only end-to-end generation surfaces this. Correctness has two surfaces:
the forward pass *and* the inference harness (tokenization, template, sampling, decode).

Fixes/guards in place: `loader._HF_PATTERNS` now pulls `*.jinja`; `run.py` warns loudly
when chat formatting was expected but no template exists. Remaining edge cases (README-only
templates, config-vs-jinja disagreement) are tracked as ROADMAP `R-7`.
