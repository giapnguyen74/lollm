# Family Conventions — how we write arch code to study it

The rules we settled on so every architecture reads the same way. **`qwen2/` is
the reference implementation — copy its shape.** Optimize for *reading one
architecture top to bottom*, not for reuse or speed.

---

## 1. One package per family, five files, one job each

```
<family>/
├── __init__.py          manifest: MODEL_TYPES · DEFAULTS · register(load)
├── config.py            parse config.json (hf) / GGUF metadata → <Family>Config
├── blocks.py            small components: RMSNorm · RoPE · Attention · MLP · (norm variants…)
├── modeling_<family>.py architecture: DecoderLayer + <Family>Model + forward
└── weights.py           the weight-name seam (maps) + load (checkpoint → model)
```

Each file answers one question:

- **config.py** — *what are the dimensions?* (a dataclass + `from_hf` / `from_gguf` + `build_config(raw, fmt)`)
- **blocks.py** — *what are the small pieces?* (detail implementations; nothing about how many layers or how they stack)
- **modeling_\<family\>.py** — *what is the model and how does it run forward?* (`DecoderLayer` shows the per-layer architecture; `<Family>Model` stacks it)
- **weights.py** — *how do I fill this model from a checkpoint?* (`to_raw(canonical, fmt)` name map + `load(raw_config, weights, fmt)`)
- **\_\_init\_\_.py** — *the manifest:* declares model types + sampling defaults, wires `load` into the registry. Importing the package registers the family.

> **Architecture vs detail:** the *layer* and the *stack* live in `modeling`
> (that's the architecture you study); the *components* live in `blocks`. Moving
> `DecoderLayer` into `modeling` is deliberate — you read the shape there and only
> drop into `blocks` for implementation detail.

---

## 2. Self-contained — duplication is intentional

A family imports **only its own siblings and the shared registry — never another
family.** If `qwen2/blocks.py` and `llama/blocks.py` both hold a near-identical
attention, that's fine. One package = one architecture's complete story. (Cost:
no fix propagation across families. Accepted — we're studying, not scaling.)

Shared infra a family *may* use (never duplicate): `loader`, `router`, `generate`
(the loop + sampler), `gguf_reader`, `dequant`, `tokenization`, `models` (registry).

---

## 3. Narrate `forward` with numbered steps

Every `forward` gets numbered comments that trace the flow, so the path from token
ids to logits is readable without running it. Three levels:

**Model** (`<Family>Model.forward`) — the top-level story:
```python
# 1. EMBED — token ids → vectors
# 2. POSITIONS — absolute positions (offset by KV-cache length) + RoPE cos/sin
# 3. DECODER STACK — N layers of (attention + MLP), each growing its KV cache
# 4. FINAL NORM
# 5. LM HEAD — hidden → vocab logits
```

**Layer** (`DecoderLayer.forward`) — one block:
```python
# 1. ATTENTION sub-block (mixes ACROSS tokens):  x = x + attn(norm(x))
# 2. MLP sub-block (transforms EACH token):      x = x + mlp(norm(x))
```

**Attention** (`Attention.forward`) — the heart:
```python
# 1. PROJECT to Q/K/V, split into heads (GQA: fewer KV heads)
# 2. ROPE — rotate Q,K by position
# 3. KV CACHE — append this step's K,V to the past
# 4. GQA EXPAND — repeat KV heads to match query heads
# 5. ATTENTION — scaled dot-product (causal during prefill)
# 6. MERGE heads + output projection
```

Keep shape annotations inline where they help (`(B, T) -> (B, T, H)`).

---

## 4. Conventions that keep loading simple

- **Mirror HF module names** (`model.embed_tokens`, `model.layers.N.self_attn.q_proj`,
  `lm_head`). Then the HF name map is identity and only GGUF needs a real map.
- **The seam lives in `weights.py`**, not `modeling` — it's a *loading* concern.
  Every format quirk (GGUF permute, stacked MoE, extra norms) goes here.
- **`load` is strict.** Build the state dict via `to_raw`, then
  `load_state_dict(strict=True)` — exact tensor set + shapes. A missing tensor =
  wrong arch/map, caught at load.
- **Tie when there's no separate output** (`lm_head` ← `embed_tokens`) — covers HF
  tied models and GGUF (which doesn't record the tie flag).
- **The model exposes `forward(input_ids, past) -> (logits, past)`** — the shared
  `generate` loop drives it; the family never writes its own loop.

---

## 4b. Lesson: converted weights aren't in the original convention

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

## 4c. Lesson: loading doubles memory unless you free the source

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

## 5. Adding a family — checklist

```
[ ] copy qwen2/ → <family>/ ; rename modeling_qwen2.py → modeling_<family>.py
[ ] config.py:   parse this family's config.json + GGUF metadata
[ ] blocks.py:   adjust/add the small components it actually differs in
[ ] modeling:    DecoderLayer + <Family>Model + numbered-step forward
[ ] weights.py:  the hf/gguf name map (+ any quirk) + load
[ ] __init__.py: MODEL_TYPES, DEFAULTS, register(load)
[ ] models.py:   add `import <family>`
[ ] parity:      python src/compare_logits.py --model <repo>   → PASS ✅   (do not skip)
```

The parity gate is the only proof of correctness. Self-contained code makes bugs
easy to *see*; matching logits proves they're *absent*.

---

## 6. Why this is the way

We chose readability over reuse on purpose (see `docs/inference-engine-vision.md`):
a study engine is worth most when each architecture is a complete, self-contained
story you can open and follow — `config` (dims) → `blocks` (pieces) → `modeling`
(shape + flow) → `weights` (load) — with the `forward` narrated step by step.
Duplication is the price; clarity per architecture is the payoff.
