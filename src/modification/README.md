# modification — studying behavior modification in the residual stream

A study module for **directional activation methods**: derive a direction in the
residual stream, then either **add** it (induce/amplify a behavior) or **project it
out** (suppress one). Cheap, fast to iterate, often reversible, and — when they work —
interpretable.

This is *not* a model family. It is a cross-cutting study package that hooks the
existing families (`qwen2`, `qwen3`, `gemma*`, …) through their `nn.Module` seams and
studies what happens when we perturb their activations. The companion study writeup is
[`docs/Steering and Ablating/steering-and-ablation-research-v2.md`](../../docs/Steering%20and%20Ablating/steering-and-ablation-research-v2.md);
this README is the *runnable plan* that turns that note into code, section by section.

> **Honest framing (from the note).** These methods are **empirically effective but not
> guaranteed**. Reliability is a property we *measure* per model and behavior — not
> something the math grants. The whole module is built around measuring it.

---

## Why this fits lollm cleanly

The families mirror HF module names, so the residual stream is directly hookable — no
family-specific plumbing:

```
model.model.embed_tokens          # embeddings
model.model.layers[i]             # DecoderLayer i — its OUTPUT is the residual stream after i+1 blocks
model.model.norm                  # final norm
model.lm_head                     # logits
```

A `register_forward_hook` on `model.model.layers[i]` gives us both roles we need:

- **read** the residual stream at layer `i` (extraction, §5 of the note), and
- **rewrite** it (intervention, §4) — add a vector, or project a direction out.

Because *every* family exposes this same shape (per CONVENTIONS §4), the module can be
**model-agnostic**: it hooks by module path, never by family. We wire an actual model
in later (see "Deferred").

---

## First try — step by step

Scope of the first cut is the **core pair** from the note's difficulty ordering
(§2): *diff-of-means extraction → activation steering → directional ablation*. Steering
and ablation share the same extracted direction, so one extraction path feeds both.

### Step 1 — load & prepare data (`data.py`) — the true first step

No model, no seam needed yet — this is pure data. Everything downstream depends on it,
and the note's split discipline (§6) must be built in from the start, not bolted on.

1. **Two matched sets.** **A** elicits the behavior, **B** does not. Match on
   length/format/topic so the eventual difference isolates the behavior, not a confound
   (§5). Prefer CAA-style **paired** prompts (identical stem, contrasting completion) —
   we keep the pairing metadata even though diff-of-means doesn't require it, because the
   subspace work (§8) later does.
2. **Split each set — extraction ⟂ sweep (⟂ test).** The direction is computed on the
   **extract split** only; the layer×α sweep is scored on a disjoint **held-out sweep
   split**; and a small **test split** is kept untouched to report the finally-chosen
   config once, unbiased by the sweep. *"Always evaluate on held-out prompts, never the
   extraction set."* (§6). Fix and record the split seed.
3. **Emit a stable schema** — e.g. `{behavior, split, label ∈ {A,B}, pair_id, prompt}` —
   so extract/sweep just filter by `split` and `label`. Load from small on-disk files
   under `behaviors/<name>/` so a behavior is reproducible and reviewable.

Output: for one behavior, `A_extract, B_extract, A_sweep, B_sweep (, A_test, B_test)` —
matched, split, seeded.

### Step 2 — extract the direction (`extract.py`)

Difference-of-means (§5), on the **extract split only**:

1. **Capture** the residual at the chosen site (`L{i}.out`, per D1) and position (last
   token is the common default) at **all** layers in one forward pass per prompt. This is
   the *read* side of the seam; the *write* side (intervention) is decided in D2, so
   extraction moves first.
2. `r = mean(A_extract) − mean(B_extract)` per layer, in float32; normalize
   `v̂ = r / ‖r‖`; keep `‖r‖` as a scale reference.

Output: one candidate direction per layer. (Sign matters for addition; ablation ignores
it.)

### Step 3 — steer (`steer.py`) — affine translation

```
h' = h + α·v̂
```

Fixed scaled shift applied to every token. `+α` amplifies, `−α` suppresses. Applied at
the **same single layer** it was extracted at (classic steering; avoids the cross-layer
mismatch discussed in §7). `α` is tunable and — because residual norm grows with depth
(§3) — is **not** uniform across layers, so we tune it per layer rather than assume one
value transfers.

### Step 4 — ablate (`ablate.py`) — projection

```
h' = h − (h·v̂) v̂          # P = I − v̂v̂ᵀ, idempotent
```

Removes whatever component of the behavior each token carried — adaptive, lands on
exactly zero for every token. Unlike a fixed subtraction, it can't under-remove strong
tokens or over-push neutral ones past zero. Typically **broadcast** to all layers/
positions (an ablation practice, §7), which is why it needs the global re-validation in
Step 5.

> **The one-line distinction (§4):** steering *moves along* the axis; ablation *deletes*
> the axis.

### Step 5 — validate (`sweep.py`) — the part that makes it science

Quality can't be read off the vector; we **apply and measure on the held-out sweep
split** from Step 1 (§6). Three metrics, always together:

1. **Target effect** — did the behavior move as intended?
2. **Collateral** — is the model still coherent/capable (perplexity or KL on off-target
   text)?
3. **Norm-matched random-direction control** — apply a random unit vector at the same
   magnitude. If it moves the behavior or breaks fluency *comparably* to `v̂`, our
   "effect" is non-specific perturbation, not evidence the direction encodes the
   behavior. The note calls this the most important check.

Sweep **layer × α** for steering (ablation sweeps layers only), read the grid for the
**fluency cliff**, and take the strongest effect just below it — a Pareto choice, not a
raw max. Fix and report seed, temperature, and hook site (§6 brittleness caution).

### Step 6 — tie it together (`run_modification.py`)

A small CLI/entry that loads a model (via the shared `loader`/`router`), runs
data → extract → sweep → apply on one behavior end to end, reports the chosen config on
the untouched **test split**, and prints a before/after comparison plus the control.
This is the "does the whole loop actually work" check, analogous to the families'
`run.py`.

---

## Planned file layout

Following CONVENTIONS (small files, one job each, numbered-step narration in the flow):

```
modification/
├── README.md              this plan
├── __init__.py            exports; no model imports at top level
├── data.py                load two matched sets; split extract ⟂ sweep ⟂ test (seeded)
├── behaviors/<name>/      the on-disk prompt sets per behavior (A/B, reviewable)
├── hooks.py               the seam — capture (read) now; intervene (write) per D2
├── extract.py             difference-of-means on the extract split → per-layer direction
├── steer.py               h' = h + α·v̂
├── ablate.py              h' = h − (h·v̂)v̂   (+ broadcast)
├── sweep.py               held-out eval: target · collateral · random control
└── run_modification.py    end-to-end entry on a chosen model
```

Deferred to later cuts (bigger, built incrementally):

- **Subspace / concept erasure** (§8): multi-axis basis, LEACE/INLP, choosing `k`.
  Note the correction — a two-class mean difference is rank-1, so a real subspace must
  come from multiple contrast axes, covariance-based erasure, or genuinely *paired*
  differences.
- **Weight-baking** (§4): orthogonalize writing matrices against `v̂` for a permanent,
  hook-free change — only after inference-time validation passes.
- **Per-layer / partial ablation** and **conditional steering** (§7–8) as remedies when
  a single global direction drifts or entangles.

---

## Deferred: the model

Per the plan, no model is wired in yet. The module is written to hook **by module
path**, so it stays model-agnostic; we choose and import a concrete model (likely a
small instruct model for fast iteration, per §5) when we start running Step 1. Until
then, the code targets the seam described under "Why this fits lollm cleanly," not any
one family.

---

## Design decisions

Decisions and open questions recorded as we go, so the rationale isn't lost.

### D1 — the residual stream is written *multiple times per layer* (settled)

"The residual at layer `L`" is not a single instant. The residual stream is one
continuous running sum threaded through the network, but each block **reads and writes
it more than once**, so a layer has several distinct residual snapshots — and the extra
Gemma writes mean we must name *which* one we mean.

Plain pre-norm block (`qwen2`, `qwen3`):

```
x = x + attn(norm(x))    # write #1 → post-attention residual  (mid-layer site)
x = x + mlp(norm(x))     # write #2 → post-MLP residual         (= layer output = hidden_states[L])
```

Gemma writes more per layer:

- `gemma2` / `gemma3`: sandwich norm — the sub-block *output* is normed before the add,
  but still **two** residual adds (post-attn, post-mlp).
- `gemma4`: **three** adds — attention, MLP, and a per-layer-embedding (PLE) injection —
  then the whole layer output is scaled by `layer_scalar`:

  ```
  x = x + post_norm(attn(...))   # write #1
  x = x + post_norm(mlp(...))    # write #2
  x = x + post_norm(PLE(...))    # write #3
  return x * layer_scalar
  ```

Keep straight: what *flows along* the stream (the running sum `x`) vs. what is *written
into* it. `attn(...)` / `mlp(...)` are the **contributions** (deltas) added to the
stream — they are **not** the residual stream itself.

**Named residual sites** (the vocabulary the tap/hook layer keys on):

| site | qwen2/qwen3 | gemma2/gemma3 | gemma4 |
|---|---|---|---|
| `L{i}.in`        | layer input `x`               | same | same |
| `L{i}.post_attn` | after attention add (mid)     | after sandwich attn add | after attn add |
| `L{i}.post_mlp`  | after MLP add (= layer output)| after sandwich mlp add  | after mlp add (not yet layer output) |
| `L{i}.post_ple`  | —                             | —    | after PLE add |
| `L{i}.out`       | = `post_mlp`                  | = `post_mlp` | `post_ple × layer_scalar` |

**Implemented (qwen2):** the seam taps **both residual write-backs** per block, named
`post_attn` (after `x = x + attn`) and `out` (after `x = x + mlp`, = the layer output =
HF `hidden_states[L+1]`). A capture/steer fn selects via `ctx.site`; `out` is the primary
site for extraction (matches the note's indexing and the "same tensor/position" discipline,
§3/§5). We dropped the speculative `in`/`post_mlp`/`post_ple` names — qwen2 has exactly
these two write-backs. Reaching the internal `post_attn` for **writing** is precisely why
we don't use `register_forward_hook` (see D2).

### D2 — intervention mechanism: per-layer `hook_fn` slot at the write-backs (RESOLVED)

We do **not** use `register_forward_hook`. A forward hook only sees a module's boundaries
(layer input/output), so it can read but **cannot write** the internal post-attention
residual (a local inside `forward`). Since we own the code, the family's `modeling` calls
`self.hook_fn(x, site)` right after each residual write-back, and
`<family>/hook.py::attach` sets that per-layer slot on a *populated* model:

```python
x = x + h
if self.hook_fn: x = self.hook_fn(x, "post_attn")   # write-back #1 (internal)
x = x + self.mlp(self.post_attention_layernorm(x))
if self.hook_fn: x = self.hook_fn(x, "out")         # write-back #2 (= layer output)
```

The user fn is `fn(act, ctx) -> Tensor | None`: return `None` to observe (capture) or
skip; return a tensor to replace the residual (steer/ablate). `ctx` — built in `hook.py`
from a per-layer closure — carries `layer_idx`, `n_layers`, `site`, `seqlen`; the fn
**self-filters** on these, so `attach(model, fn)` takes no site/layer arguments. Lifecycle
is explicit via a `Handle` context manager (slots cleared on exit — no leaked state).
`hook.py` lives in the family like `weights.py` (it owns the write-back map) and hard-fails
on a non-qwen2 model.

Why this over the earlier `taps`-param idea: identical reach (both write-backs, read+write)
and identical visibility in `forward`, but no extra `forward` parameter to thread through
each family's differing signature — the call is just `self.hook_fn(x, site)` and `hook.py`
supplies the rest. **Validated:** `src/hook_test.py` shows the seam fires (24 layers × 2
sites = 48 triggers) and `compare_logits` passes unchanged (max|Δ| = 0.0000, cosine 1.0,
top-5 match) — the guarded lines are a true no-op when nothing is attached.

### D3 — chat-template wrapping is an encode-time concern (settled)

Instruct models represent behaviors in the chat format they were trained on, so
extraction prompts must go through the template (position = last token under
`add_generation_prompt`, the generation point; note §5). But in lollm that wrapping is
the tokenizer's job — `tok.apply_chat(text)` already builds
`[{"role":"user","content":text}]` + `add_generation_prompt=True` internally — so we
never hand-build that list. It therefore lives at **encode time** (the hooking model /
Step 2), *not* in `data.py`, which stays model-agnostic and stores raw text
(`{behavior, split, label, pair_id, prompt}`). Rules: wrap A and B **identically** so
template tokens cancel in `μ_A − μ_B` (no confound); make chat-vs-raw a run toggle
(base models prompt raw, mirroring the CLI's `--no-chat`). Known gap: `apply_chat`
renders a **single user turn** only, so CAA-style *contrasting-completion* pairing isn't
expressible yet — only prompt-contrast (flag for the §8 paired/subspace work).

### D4 — capture strategy: record-all-once, filter at read time (settled)

The forward pass is the only expensive step; all downstream analysis is cheap linear
algebra over the same activations. So `extract.Extractor` **records every site in one
forward pass** and selects a subset only at read time (`direction(site)` /
`subspace(site)`) — restricting the capture set up front would force a re-run to look
elsewhere, and the layer sweep (§6) / cross-layer check (§7) need all layers anyway.
Position: **last token only** (`position=-1`); all-positions capture is a deliberate
future opt-in (it multiplies the cache by sequence length). Persistence: **in memory for
now** — a run-once, save-to-disk activation cache (making `extract` model-free and
reusable across experiments) is a later refactor, not built yet.

### D5 — finding: residual norm grows ~20× with depth, back-loaded (measured)

`hook_test.py` on Qwen2.5-0.5B-Instruct (prompt *"Explain RoPE in one line."*, last-token
‖residual‖ at the `out` site): layer 0 ≈ 3.3 → layer 23 ≈ 67 — about **20×**, and heavily
**back-loaded**: roughly flat through the middle band, then exploding in the last third
(layer 20 ≈ 45 → layer 23 ≈ 67, peaking ≈ 79 at layer 23 `post_attn`). Growth is
non-monotonic (layers 8, 11–13 dip), and the **final layer inverts** (`out` drops 79 → 67,
absmax 65 → 26 — the usual pre-final-norm cleanup).

Consequence for steering (note §3, now measured on this model): a fixed-magnitude `α·v̂` is
a large perturbation early and negligible late, so **α must scale to the local residual
norm**, not be one global constant. The sweep (Step 5) should capture these per-layer `out`
norms and set α relative to them; and the predictable mid-band — not the last layer — is
where steering should behave, corroborating the "middle third" heuristic (§6). The `out`
norms `attach` already sees make this measurement free.

## Dual-use note

These are neutral mechanisms with opposed applications (note §9). The register that
keeps the work defensible — and the level this module stays at — is **understanding,
evaluation, and hardening**: characterizing how a behavior is represented and how
robustly it can be moved, not stripping safety from a shipped model.
