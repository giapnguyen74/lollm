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

## 4b. Gotchas & lessons → see [docs/LESSONS.md](docs/LESSONS.md)

Hard-won surprises (GGUF weight constant-folding, load-time memory doubling, buffers
vs parameters, chat-template provenance, …) are recorded explicitly in
**[docs/LESSONS.md](docs/LESSONS.md)** as `L-#` entries — kept out of this file so
CONVENTIONS stays the *structural* pattern for adding a family. When a new gotcha bites,
record it there.

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
[ ] real run:    python src/run.py --model <repo> --prompt "…"  on your ACTUAL device
                 → coherent text (the fp32 CPU gate never exercises the bf16/fp16 runtime)
```

Two checks — they catch **different classes of bug**:

- **Parity gate** (`compare_logits`, fp32/CPU) catches **silent implementation errors** —
  a wrong RoPE, norm, scale, or weight-map. These barely show in generation: a subtly
  wrong model still emits **fluent, confident text**, so you'd never spot the bug by
  reading output. Parity compares token-for-token against the reference, so it flags the
  mismatch *immediately* (and a layer-by-layer comparison localizes it). It's the fast debugger
  for *math* bugs — but it runs fp32 on CPU, so it never exercises the deployed dtype or
  the harness.
- **Real run** (`run.py` on your ACTUAL device) catches **visible runtime/harness
  failures** — fp16 overflow → NaN, missing chat template → loops, a cache/decode bug →
  garbage. These live entirely outside the fp32 parity path.

The trap: **fluent generation is not evidence of a correct implementation — only parity
is.** Pass *both* before trusting a family (see LESSONS.md L-4, L-5).

**Dev-time only — throwaway random-weight tests.** While *building* a complex family,
the fastest way to localize a *math* bug is a temporary **module-parity** check: build
your block and the `transformers` block with the *same random weights* and compare
outputs — no download, runs in seconds, points straight at the broken component. But it
**only tests layer math**: random weights bypass the real loader (so it can't catch a
name-map / skipped-buffer bug) and it never touches the runtime dtype or harness. So it
**never replaces** the two checks above — write it to debug, **delete it once parity
passes**. We keep these out of the committed suite; the durable tests are
`compare_logits` + `sanity_test` only.

---

## 6. Why this is the way

We chose readability over reuse on purpose (see `docs/inference-engine-vision.md`):
a study engine is worth most when each architecture is a complete, self-contained
story you can open and follow — `config` (dims) → `blocks` (pieces) → `modeling`
(shape + flow) → `weights` (load) — with the `forward` narrated step by step.
Duplication is the price; clarity per architecture is the payoff.
