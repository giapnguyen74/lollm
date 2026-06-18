# Lessons — gotchas, explicitly recorded

Hard-won gotchas worth not re-learning. These are **not** the structural pattern for
adding a family (that's [../CONVENTIONS.md](../CONVENTIONS.md)) — they're the surprises
that cost us debugging time. Record new ones here as `L-#` (stable, never renumbered).

| ID  | area            | gotcha (one line) |
|-----|-----------------|-------------------|
| L-1 | gguf / weights  | Converters fold runtime conventions into the weights — undo them in `weights.py` |
| L-2 | loading / memory| `.to(device)` makes a second buffer — load lean and free the CPU source |
| L-3 | loading / debug | Load **buffers**, not just parameters; cosine is scale-invariant and hides magnitude bugs |
| L-4 | inference       | "No chat template" has 4 causes (base / not-downloaded / forgotten / clone) — don't guess the fix |
| L-5 | mps / dtype     | fp16 is the MPS default; bf16-on-MPS rejected (~3× slower); fp16≈fp32 because reductions upcast |
| L-6 | gguf / tokenizer| One deliberate "guess": GGUF `tokenizer.ggml.model` defaults to `gpt2` BPE — see why |
| L-7 | parity / testing| Don't seed-match two stochastic loops & compare tokens — drive ONE trajectory, compare logits |
| L-8 | parity / testing| Test sequences **longer than the sliding window** — short prompts hide window bugs (cf. T-2) |
| L-9 | parity / testing| Self-consistency (ours==ours) ≠ ref-parity; know exactly what a green check proves |

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

## L-4 · "No chat template" is ambiguous — four different causes, don't guess the fix

A model's chat template can live in **several** places, and there's no standard: inlined in
`tokenizer_config.json` (auto-loaded), as a standalone **`chat_template.jinja`** file (only
loaded if it downloaded), or **only described in the repo README** (can't auto-apply). The
GGUF path carries its own `tokenizer.ggml.chat_template` metadata too.

This bit us: the loader's download allow-list didn't include `chat_template.jinja`, so
gemma-4-e2b-it had no template, `run.py` silently fell back to a **raw prompt**, and the
instruction-tuned model degenerated (`What is coffee` → `coffee coffee coffee…`).

**The core insight: "the tokenizer has no `chat_template`" has four distinct causes, and the
*right* response differs for each — which is exactly why we must not silently guess one.**

1. **It's genuinely a base / pretrained model.** No template by design; the model is a raw
   next-token predictor. Raw prompting (autocomplete / few-shot) is the **correct** mode, so
   `--no-chat` is right and you should test it that way (prompt it with a prefix the answer
   continues, not a question).
2. **We failed to download the template.** The `chat_template.jinja` exists in the repo but
   our allow-list / cache missed it. The model *is* instruct; the fix is to fetch the file,
   not to prompt it raw. (This is the gemma-4 case — fixed by adding `*.jinja` to
   `_HF_PATTERNS`.)
3. **The author forgot to ship one.** An instruction-tuned model whose repo simply has no
   template anywhere (or only describes the format in prose in the README). Nothing to
   download; the model still needs *a* template to behave.
4. **A clone/finetune of a known base.** Many repos are Qwen- or Gemma-lineage finetunes that
   inherit the parent's chat format but never re-ship the template. Here a sensible **default
   template borrowed from the parent family** (the Qwen `<|im_start|>…` / Gemma
   `<start_of_turn>…` form) usually Just Works — `BPETokenizer.apply_chat` /
   `SPMTokenizer.apply_chat` already carry exactly these family fallbacks for when
   `chat_template` is absent.

The trap underneath all four: **`compare_logits` uses raw prompts, so a green parity gate
never exercises the chat template** — only end-to-end generation surfaces a template problem.
Correctness has two surfaces: the forward pass *and* the inference harness (tokenization,
template, sampling, decode).

**Current behavior (and why):** `loader._HF_PATTERNS` now pulls `*.jinja` (fixes cause 2);
`run.py` **hard-fails** in chat mode when no template exists rather than silently raw-prompting
(per "never guess" — causes 3 and 4 produce confident garbage if treated as raw, and cause 1
should be an explicit choice). `--no-chat` is the deliberate opt-in to raw prompting for cause
1 (base models). For causes 3/4, the family-default template in `apply_chat` is the escape
hatch — but applying it **automatically** would be a guess, so today it's only reached when a
template is present.

**Not implemented (opinion, not a plan):** one could imagine a `--chat-template qwen|gemma`
flag, or auto-applying the router-known family default, to serve causes 3/4. We deliberately
have **not** built this — it trades the project's "never guess" clarity for convenience, and a
wrong family default still emits fluent-but-off output. Left as an open question; the current
answer is the explicit hard-fail. Remaining provenance edge cases (README-only templates,
config-vs-jinja disagreement, per-family default opt-in) are parked under ROADMAP `R-7`.

---

## L-5 · fp16 is the MPS default; bf16-on-MPS was rejected

Real inference on Apple **MPS** runs **fp16**, while the parity gate runs **fp32 on CPU**.
We keep fp16 as the MPS default and it stays near-identical to fp32 in practice because the
overflow-prone reductions — attention softmax and every RMSNorm — already **upcast to fp32**
internally, so the fp16 storage dtype doesn't lose the precision that matters. **bf16-on-MPS
was tried and rejected**: ~3× slower and memory-heavy on Metal, with no accuracy win over the
fp16+fp32-reduction path. Dtype is chosen by `pick_dtype(device)` in `run.py`
(cuda→bf16, mps→fp16, cpu→fp32). The intended `--dtype` override (an fp32 escape hatch for a
model that overflows in fp16) is **not yet implemented** — tracked as `TODOS.md` T-1.

---

## L-6 · The one deliberate guess: GGUF tokenizer defaults to `gpt2` BPE

"Hard-fail, never guess" is the rule everywhere **except one consciously-accepted spot**:
`loader._load_gguf` reads `meta.get("tokenizer.ggml.model", "gpt2")` and `BPETokenizer.from_gguf`
does the same. If a GGUF omits `tokenizer.ggml.model` we **assume GPT-2 byte-level BPE**
rather than raising.

Why this default (and not a hard-fail like everything else):
- The GGUF spec's own default for that key is `"gpt2"`, and in practice the field is reliably
  present for the SentencePiece/`llama` family (Gemma, Llama-2) — i.e. the case where guessing
  wrong would matter is the case where the key is actually written. A genuinely missing key
  almost always means a GPT-2-BPE model (Qwen, Llama-3).
- A wrong guess here is **loud, not silent**: the only other supported engine is SPM, and an
  SPM vocab fed to the BPE merge path produces immediately-garbled decode (or a `from_gguf`
  mismatch), so it surfaces on the first run rather than emitting plausible-but-wrong text.
  That's the property that makes "never guess" matter — and it doesn't hold here, so the
  default is safe to keep.
- Any tokenizer type we *don't* support still hard-fails: `_load_gguf` raises
  `NotImplementedError` for `tokenizer.ggml.model` values other than `gpt2`/`llama`/`gemma`.

So this is a scoped exception, documented on purpose. If it ever bites (a non-GPT-2 model with
the key omitted), flip the default to a hard-fail — the cost is a clearer error, the benefit is
consistency with the rest of the loader.

---

## L-7 · Don't seed-match two stochastic loops and compare tokens — compare logits on one trajectory

When parity-checking a **sampling** loop (diffusion denoise, AR sampling, anything with
`multinomial`/`randint`), the tempting approach is: run our loop and the reference loop with the
same seed, compare the output tokens. **This is fragile and gives false failures.** Any
boundary-sensitive *discrete* decision — a stopping threshold, an `argmax` near a tie, an accept
cutoff — can flip on a **~1e-6 numerical difference** between two correct implementations. One flip
changes how many RNG draws a step consumes (e.g. an extra `renoise`), which **desyncs the RNG
streams**, and from there the trajectories diverge completely even though nothing is wrong.

This bit us in diffusion_gemma rung 5: a 2-block token comparison matched block 0 exactly but
diverged at block 1, purely because the adaptive stop (`mean entropy < 0.005`) triggered one step
later on one side (same argmax block, but one extra random draw).

**Do instead:** drive a **single** trajectory and compare the **continuous** quantity — the
per-step **logits** (cosine / max|Δ|), with both implementations fed the *identical* intermediate
state. That's apples-to-apples and tolerant of the inevitable 1e-6 float drift. Reserve exact
token equality for fully-deterministic settings (greedy + seed-exact `randint` only, no
threshold-flips), and even then keep the number of steps small.

---

## L-8 · Exercise sequences longer than the sliding window — short prompts hide window bugs

A sliding-window attention bug is **invisible** on short prompts: if every test sequence is shorter
than the window, the band mask never clips anything and a *broken* sliding implementation behaves
identically to a correct one. (This is exactly what `TODOS.md` **T-2** warns about.)

diffusion_gemma rungs 1–4 all used ≤6 tokens with `sliding_window=8`, so the window never bit and
everything passed. Rung 5's second block pushed the context to 14 tokens (> window) and **immediately**
exposed a real divergence: the reference clips its sliding-layer KV cache to the last `window`
entries while our decoder read the **full** cache — cosine dropped from 1.0 to ~0.5 on the first
step over the window. Tracked as a diffusion_gemma rung-6 correctness item (and a live instance of T-2).

**Rule:** any parity gate touching sliding-window layers must include a prompt **longer than the
window**, and ideally assert a prefill-vs-incremental-decode equivalence at that length. A short
green test proves nothing about the window.

---

## L-9 · Self-consistency (ours == ours) is not the same as ref-parity

A check can be **true and useful yet prove less than it appears.** In diffusion_gemma rung 5 an
`incremental-encode == full-prefill` check passed cleanly — and it *is* a real property (re-encoding
a committed block equals encoding the whole sequence from scratch). But it compared **our**
incremental path to **our** full path (self-consistency), not to the reference, and it used the same
short sequences as everything else. So a green check gave false confidence that the cache was
"verified" while a long-context divergence from the reference (L-8) sat underneath it.

**Rule:** be explicit about what each check covers. Self-consistency (ours==ours) catches internal
contradictions but **inherits every blind spot** of the inputs you feed it; only an *ours-vs-reference*
check at the regime that actually matters (here, context > window) proves parity there.
