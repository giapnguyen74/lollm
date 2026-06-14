# Inference Study — Vision & Plan

> A from-scratch inference engine whose purpose is **to study how models run**, not
> to be fast or DRY. The shape is deliberately simple: **load a model, probe its
> architecture, route to a self-contained `modeling_<family>.py`, run end-to-end.**
>
> **Non-goal:** performance and code reuse. For production use llama.cpp / vLLM /
> transformers. This exists so each architecture can be read as a complete story.

---

## The vision (user story)

> Point it at a model — an HF repo, a local cache dir, or a `.gguf` — and a prompt.
> It loads the model, figures out which architecture it is, routes to that family's
> implementation, and generates text. One command; one readable file per family.

```
infer "Qwen/Qwen2.5-0.5B-Instruct"   "..."   # downloads / uses HF cache
infer "./local/model/dir"            "..."   # local safetensors
infer "model.Q4_K_M.gguf"            "..."   # gguf
```

![Loader → probe → route → modeling_<family>](./inference-engine-vision.svg)

---

## The shape: three parts

```
1. LOADER     fetch from HF Hub / local cache / .gguf
              → read config (or GGUF metadata), weights, tokenizer
                       │
2. PROBE+ROUTE read model_type / general.architecture  →  pick the family package
                       │
3. <family>/  package — entry modeling_<family>.py, may import its OWN siblings
   families/                       (qwen2/ · llama/ · gemma3/ ...)
   ├── qwen2/
   │   ├── modeling_qwen2.py        ← entry: end-to-end inference
   │   ├── config.py                ← (optional) its own config reader
   │   └── blocks.py                ← (optional) its own attention/norm/mlp
   ├── llama/   modeling_llama.py + siblings
   └── gemma3/  modeling_gemma3.py + siblings
                       │
                       ▼
                     text
```

### Part 1 — Loader (the first milestone)

Get the model onto disk and into memory:

- **HF / cache:** resolve a repo id (or local dir) via the HF cache; read
  `config.json`, load the `*.safetensors`, load the tokenizer.
- **GGUF:** locate/parse a `.gguf`; read metadata; dequantize weights.
- Output a small handoff: **(raw config/metadata, weights keyed by the file's own
  raw names, a `fmt` tag, tokenizer)**. The loader does **not** rename or normalize
  tensors — names stay exactly as the file has them. The family maps them (below).

### Part 2 — Probe + route

Read the architecture id and dispatch:

- **safetensors:** `config.json` → `architectures` / `model_type` (e.g. `"qwen2"`).
- **GGUF:** metadata → `general.architecture`.
- A tiny registry maps that id → a `<family>/` package. **Unknown → fail loud**
  (don't guess; don't run one arch's weights through another's code).
- The router **only probes and routes** — it never touches tensor names or weights.
  All weight handling (including format quirks) is the family's job.

### Part 3 — the `<family>/` package (end-to-end, self-contained)

Each family is a **package** `families/<family>/` whose entry point is
`modeling_<family>.py`. The entry spells out the architecture end-to-end — config
reading, the **per-format weight-name maps**, embeddings, attention, norms, RoPE,
MLP/MoE, the decoder layer, the KV-cache representation, and the **forward pass
(token ids → logits + state)**. The shared loop (next section) drives that forward;
the family does **not** own the loop.

It is **free to import its own sibling modules** within the package (e.g.
`qwen2/config.py`, `qwen2/blocks.py`, `qwen2/names.py`) when that aids clarity — a
family need not be one giant file. The one rule:

> **A family imports only from itself and the thin shared infra — never from
> another family.**

So the **reuse boundary is the family package**: inside it, write normal modular
code; *across* families, duplicate. If `qwen2/blocks.py` and `llama/blocks.py` both
contain a near-identical attention, **that's fine** — each family's story stays
complete and readable on its own. (The cost is **no fix propagation** — a bug fixed
in one family's blocks stays in another's. Accepted, because we're not scaling past
a handful of families.)

### The weight-name seam — the family owns it

This is the part that's actually hard, so it's stated plainly: safetensors names
(`model.layers.N.self_attn.q_proj`) and GGUF names (`blk.N.attn_q`) differ, and some
families add tensors a generic map wouldn't know (Gemma's `q_norm`/`k_norm`, the
extra sandwich norms). The loader hands over weights by the **file's own raw names**;
**each family maps those into its own modules, per format** (e.g. a small
`names.py`). This is where every format quirk lives — and it lives *in the family*,
not in shared code:

- GGUF Q/K **permute** for Llama-style RoPE,
- GGUF **stacked MoE** experts → per-expert split,
- a family's **extra tensors** (Gemma norms),
- tied embeddings.

Because the map is the family's, a wrong/incomplete map is caught at load (a tensor
the forward needs has no source) — fail loud, per family.

### The shared generate loop

Prefill → decode → stop, plus the sampler, is the **most architecture-agnostic code
in the system**, so it lives **once** in shared infra — *not* per family. It calls
the family's `forward(ids, state) → (logits, state)`, samples, threads the opaque
`state` back in, and stops on the family's eos ids. A family provides only the
forward and its pipeline defaults (chat template, stop ids) — never its own loop.

If a future family genuinely needs a different loop (Mamba's recurrent step,
speculative decoding), we **update the shared loop then** — not preemptively, and not
by copying it into families.

---

## Why duplication is a *feature* here

The goal is to **read one architecture end-to-end as a complete story.** A shared
"primitives toolbox" optimizes for reuse, but it fragments that story across files
— to understand Gemma you'd jump between `attention.py`, `norm.py`, `rope.py`, and
a thin assembly. For *studying*, the opposite is right: open `modeling_gemma3.py`
and the entire model is there, top to bottom, including exactly how its norm,
QK-norm, sliding window, and dual RoPE differ — visible in one place, not diffed
against a generic base.

This is also a real, deliberate policy in transformers: the **single-model-file**
philosophy intentionally repeats code so each `modeling_*.py` is self-contained and
readable in isolation. We take it further (study-first): the unit of
self-containment is the **family package** — split a family into siblings for
clarity, but duplicate *across* families freely; extract shared primitives *only
if* we later choose to, and never at the cost of a family's readability.

So the trade is explicit: **a little shared infra (loader, router, sampler) +
thick, self-contained family packages that duplicate across each other.** Reuse is
a non-goal; clarity per architecture is the goal.

---

## What's shared vs. duplicated

| Shared (thin infra) | Owned per `<family>/` package |
|---|---|
| loader (fetch + dequant → raw tensors by file names + `fmt` + tokenizer) | config parsing |
| probe + route (model_type → family) | **per-format weight-name maps** (+ format quirks) |
| **generate loop** (prefill → decode → stop) | embeddings, attention, norms, RoPE |
| sampler (greedy/temp/top-k/p) | decoder layer + **forward** (ids → logits + state) |
| parity-gate harness | KV-cache representation · pipeline defaults (chat template, stops) |

The loop and sampler are firmly **shared** — a family provides a `forward`, not its
own loop. (If a family ever needs a different loop, we change the shared one then.)
The weight-name maps are firmly **per family** — that's where format quirks belong.

---

## Correctness — the one real check

Self-containment makes bugs easy to *see* but doesn't prove correctness. The single
gate that does: **compare next-token logits against a reference** (transformers for
safetensors, llama.cpp for GGUF). Every new `modeling_<family>.py` ships with a
parity check; a PASS is what "done" means.

---

## Roadmap (milestones)

- [ ] **M1 — Loader.** HF repo / local cache → config + safetensors + tokenizer. Print the probed `model_type`. *(the "first part")*
- [ ] **M2 — Router.** `model_type` → `modeling_<family>` registry; unknown fails loud.
- [ ] **M3 — `qwen2/` package.** `modeling_qwen2.py` (+ siblings) end-to-end from safetensors; PASS `compare_logits` vs transformers.
- [ ] **M4 — GGUF loading.** Parse + dequantize; run the `qwen2/` package from a `.gguf` too.
- [ ] **M5 — `gemma3/` package.** A *different* family, fully self-contained — study the diffs (sandwich norm, QK-norm, dual RoPE, sliding window). PASS parity.
- [ ] **M6 — `llama/`, `mixtral/`, …** each as its own readable package.

---

## Principles

1. **Study first.** Readability and self-containment beat reuse and speed.
2. **Thin infra, thick family packages.** Loader + router + sampler are shared; each architecture is a self-contained `<family>/` package.
3. **A family imports only itself + shared infra — never another family.** The reuse boundary is the package; duplicate across families freely.
4. **The router only probes and routes; fail loud on unknown.** It never touches tensor names or weights.
5. **The family owns the per-format weight-name maps** — that's where format quirks (GGUF permute, stacked MoE, extra norms) live. The loader stays dumb (fetch + dequant).
6. **The generate loop + sampler are shared** — a family provides a `forward(ids, state)`, not a loop. Change the shared loop only when a family truly needs it.
7. **A parity gate is the only proof of correctness** — one per family.
