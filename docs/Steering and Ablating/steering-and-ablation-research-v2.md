# Steering and Ablating Behaviors in Open-Weight LLMs (v2)

*A technical research note on directional methods for adding and suppressing behaviors in the residual stream.*

---


**Confidence markers used below:**

- `[established]` — standard, load-bearing, widely reproduced.
- `[study-specific]` — reported in particular papers; may not generalize; attribute before relying on it.
- `[heuristic]` — a rule of thumb from practice, not a measured constant; treat the numbers as starting points to tune, not findings.
- `[unverified-cite]` — reference recalled from memory without network access to check title/year/authors.

A dual-use caveat applies throughout and is stated in §9.

---

## 1. Scope and motivation

Having the actual weights unlocks intervention methods that API-only access forbids. This note focuses on **directional activation methods**: derive a direction in the residual stream, then either *add* it (to induce/amplify a behavior) or *project it out* (to suppress one). These are cheap, fast to iterate on, often reversible, and — when they work — interpretable.

The note builds from the landscape down to a runnable workflow: extract by difference-of-means, validate with a held-out sweep *and controls*, apply, and generalize to a behavior *subspace* when a single direction is insufficient. Throughout, the honest framing is that these methods are **empirically effective but not guaranteed** — their reliability is a property you measure per model and behavior, not something the math grants.

---

## 2. The landscape of behavior-modification methods

Two families: **weight-modifying** (permanent) and **inference-time** (frozen weights).

### Weight-modifying

- **Fine-tuning (SFT).** General and effective; data-hungry and prone to collateral drift. LoRA/QLoRA/adapters/(IA)³ capture most of the benefit at a fraction of trainable parameters; a detachable LoRA gives crude reversibility.
- **Preference optimization.** RLHF (PPO) and lighter successors — DPO, IPO, KTO, ORPO — steer via contrastive pairs. DPO is the common default.
- **Model / knowledge editing.** ROME, MEMIT, MEND locate and rewrite an association. Surgical; natural for facts, awkward for broad behaviors.
- **Task arithmetic.** Add/subtract a task vector (fine-tuned minus base). Composable, somewhat brittle.
- **Unlearning.** Explicit capability/content removal; hard without collateral damage.

### Inference-time / activation

- **Steering vectors / activation addition.** RepE, CAA, ITI. `[established]`
- **Directional ablation ("abliteration").** Project a direction out; canonical case is the refusal direction. `[established]`
- **SAE feature steering.** Clamp an interpretable sparse-autoencoder feature (e.g. the "Golden Gate" demonstration). Precise; costs training an SAE. `[study-specific]`
- **Decoding-time.** Logit biasing, contrastive decoding, DExperts. No internals touched.

### A hands-on difficulty ordering `[heuristic]`

Logit biasing (trivial, crude) → steering vectors (recommended first project) → directional ablation (reuses the same extraction) → LoRA (when a permanent or diffuse change is needed). SAE steering, full RLHF, and ROME/MEMIT are heavier starts.

---

## 3. Why directional methods work — and the limits of that story

The premise is the **linear representation hypothesis**: many high-level behaviors correlate with a direction (or low-dimensional subspace) in the residual stream, so moving along it changes the behavior. `[established]` as an empirical regularity for *many* behaviors; `[study-specific]` for any *given* behavior — some are not cleanly linear, and assuming linearity is itself a hypothesis to test (see §8).

Two structural facts are often cited as licensing cross-layer application:

- **Shared residual stream.** Every block reads from and writes to the same `d_model` space, so a direction is expressible in one coordinate system at every layer. `[established]`
- **High dimensionality.** Removing one direction from a 2000–4000-dim space deletes a single degree of freedom.

**Important correction to the naive version of this argument.** A shared coordinate system is *necessary but not sufficient* for broadcasting one direction across layers. It guarantees the vector is *representable* at every layer, not that it *means the same behavior* at every layer. Whether the behavior occupies a consistent direction across depth is an empirical question — measured, not assumed (§7). v1 overstated this; the corrected claim is that the shared stream makes broadcasting *possible*, and the held-out sweep plus a cross-layer consistency check are what make it *safe*.

### Normalization changes the picture (the key mechanism v1 omitted)

Modern transformers apply LayerNorm/RMSNorm to the input of each block (pre-norm). Two consequences that directly affect steering:

1. **Residual norm grows with depth.** The residual stream accumulates contributions, so its magnitude generally increases through the network. `[established]` A *fixed-magnitude* added vector therefore has a **layer-dependent relative effect** — the same `α` is strong early and weak late (or vice versa depending on where you hook). This is a concrete reason a single global coefficient is not uniform across layers, and why per-layer scaling (e.g. scaling the steer to a fraction of the local residual norm) is common practice. `[heuristic]`
2. **Downstream norms partially rescale the intervention.** After you add a vector, the next block's norm re-normalizes its input, absorbing part of the change. Where you hook relative to the norm (residual stream vs. post-norm block input) determines how much of your intervention survives.

Practical rule: **extract and intervene at the same, explicitly-chosen point in the residual stream, and expect to tune `α` per layer rather than assume one value transfers.** `[established]` for the "same point" discipline; `[heuristic]` for the per-layer tuning.

---

## 4. Steering vs. ablation: the core mechanical distinction

Both start from the **same ingredient** — a direction `v`, usually from a difference of class means (§5). They differ in the *operation*.

### Activation addition (steering) — affine translation

```
h' = h + α·v̂
```

A fixed, scaled shift; `α` positive amplifies, negative suppresses; the **same shift applies to every token**. Suppression by steering is `h − α·v̂`.

### Directional ablation — projection

```
h' = h − (h·v̂) v̂        with  v̂ = v / ‖v‖
```

Removes the component of `h` along `v̂`; the amount removed is **adaptive** (whatever was present), landing on exactly zero. As an operator, `P = I − v̂v̂ᵀ`, idempotent.

### Why "subtract a vector" ≠ "ablate that direction"

A fixed subtraction ignores where each token started: a strongly-expressing activation may stay positive (under-removed); an already-neutral one is pushed *past* zero into the opposite behavior (over-corrected). Ablation zeros the component for **every** token, matched to how much was present. On a number line of the behavior component (0 in the middle):

- **Steering (fixed αv̂):** arrows the same length, landing in different places relative to 0 — inconsistent.
- **Ablation (projection):** arrows of different length, all landing exactly on 0 — consistent.

**Steering moves along the axis; ablation deletes the axis.**

### Consequences

| Property | Steering (add/subtract) | Ablation (project out) |
|---|---|---|
| Operation | Fixed affine shift | Adaptive linear projection |
| Coefficient | Tunable `α` | None (basic form) |
| Sign matters? | Yes | No — acts on the 1-D line, not its orientation |
| Suppression robustness | Soft; strong prompts can overcome; can over-push past zero | Removes the mechanism; harder to overcome `[study-specific]` |
| Typical scope | Same single layer extracted at | Broadcast to all layers & positions |
| Permanence | Inference-time additive bias | Can be baked into weights (orthogonalize writing matrices against `v̂`) `[established]` |
| Amplification | Natural (`+v̂`) | No natural "add" counterpart |

The weight-baking route is why "abliterated" models ship with refusal removed and no runtime hook.

---

## 5. Extracting the direction: difference-of-means

Dataset A elicits the behavior; dataset B does not. The mean shift between their activations is the direction. `[established]`

### Steps

1. **Match the datasets.** Keep A and B alike in everything except the target behavior (format, length, topic) so the difference isolates the behavior, not a confound. CAA-style paired prompts (identical stem, contrasting completion) are the gold standard. `[established]`
2. **Pick a hook point** — a layer and a token position (last token is a common default). Extract at all layers at once; use the same position for both sets.
3. **Cache activations** — one forward pass per prompt; grab the residual vector at the chosen point.
4. **Average each set, subtract** — `r = μ_A − μ_B`, computed in float32.
5. **Normalize** — `v̂ = r / ‖r‖`; keep `‖r‖` as a scale reference.

Both sides are averaged so per-prompt idiosyncrasies cancel, leaving mainly the axis the sets *systematically* differ on.

### Hook-site precision (tightened from v1)

In Hugging Face, `output_hidden_states=True` returns a tuple of length `num_layers + 1`: index 0 is the embedding output, and index `i ≥ 1` is the residual stream **after** block `i`. So "layer `L`" here means `hidden_states[L]` = residual stream after `L` blocks. `[established]`

To intervene at the *same* point, register a forward hook that modifies the output residual of block `L` (or, equivalently, the input residual of block `L+1`). Off-by-one between "layer index in `hidden_states`" and "module you hook" is a common, silent bug. **Extraction and intervention must use the identical tensor and position.** `[established]`

### Code (illustrative — `float16`, last token, all layers)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "meta-llama/Llama-3.2-1B-Instruct"      # small instruct model = fast iteration
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(
    name, torch_dtype=torch.float16, device_map="auto"
).eval()

def last_token_acts(prompt):
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = torch.stack(out.hidden_states, dim=0)   # [L+1, 1, seq, d]
    return hs[:, 0, -1, :].float().cpu()         # [L+1, d]  last token, every layer

pos_prompts = [...]   # behavior present  (A)
neg_prompts = [...]   # behavior absent   (B)

pos = torch.stack([last_token_acts(p) for p in pos_prompts])   # [N_A, L+1, d]
neg = torch.stack([last_token_acts(p) for p in neg_prompts])   # [N_B, L+1, d]

dirs = pos.mean(0) - neg.mean(0)                 # [L+1, d]  one raw direction per layer
dirs = dirs / dirs.norm(dim=-1, keepdim=True)    # normalize per layer
```

`add_hook` / `generate` / scoring helpers are referenced later as **pseudocode** — they are not defined here and the snippets above/below will not run end-to-end as-is.

### Pitfalls

- **Confounds.** Length or topic imbalance yields a length/topic direction. Match aggressively.
- **Position consistency.** Same token-position choice on both sides.
- **Sample size.** Diff-of-means is fairly stable with modest data, but the safe count is model- and behavior-dependent — sweep it rather than trusting a fixed number. (v1 gave "a few dozen … a few hundred" as if established; it is `[heuristic]` at best.)
- **Sign only matters for addition;** ablation ignores it.

### Alternatives

PCA top component, or a logistic-regression probe (weights = direction). The claim that diff-of-means often *steers better* than a probe is `[study-specific]` — reported in parts of the CAA/RepE line of work, plausibly because a probe can exploit a linearly-separable but not causally load-bearing direction — but it is not a universal law; validate both if it matters.

---

## 6. Selecting the best direction

Quality cannot be read off the vector. Cosine and norm only *rank candidates*; the reliable selector is **apply-and-measure on held-out prompts**, with controls.

### Metrics (use all three)

1. **Target effect** — did the behavior move as intended?
2. **Collateral** — is the model still coherent and generally capable (perplexity on unrelated text, KL from base on off-target prompts, a few capability items)?
3. **Control: a norm-matched random direction.** *(New in v2, and arguably the most important check.)* Apply a random unit vector scaled to the same magnitude and measure the same two metrics. If a random direction moves the behavior — or degrades fluency — comparably to your extracted `v̂`, then your "effect" is largely non-specific activation perturbation, not evidence the direction encodes the behavior. `[established]` as a methodological norm in the interpretability literature.

The best `v̂` maximizes target effect **subject to** collateral under threshold **and** a clear margin over the random control — a Pareto choice, not a raw max.

### Evaluation validity (v2 caution)

The target-effect metric is usually the weakest link. String/keyword refusal detectors are gameable in both directions (a model that emits "I cannot…" then complies, or that complies without the trigger phrase); LLM-judge scoring carries its own biases (length, style, self-preference). Report which metric was used and treat a single automated score with suspicion. `[established]` as a known problem; the specific failure rates are `[study-specific]`.

### The sweep

Effect depends on **layer and coefficient together** (and normalization, §3). Grid over both; read the grid for the **fluency cliff** (where output breaks) and take the strongest effect just below it. Ablation has no `α`, so its sweep is over layers only.

```python
import itertools

held_out = [...]                       # prompts NOT used for extraction
layers   = range(6, 20)
alphas   = [2, 4, 8, 12, 16]           # tune to the local residual norm (see §3)

results = {}
for L, a in itertools.product(layers, alphas):
    add_hook(model, layer=L, vec=dirs[L], coeff=a)   # pseudocode hook
    gen = [generate(model, p) for p in held_out]
    remove_hook(model)
    results[(L, a)] = {
        "effect":  behavior_score(gen),
        "fluency": fluency_score(gen),
    }

FLOOR = your_threshold
best = max((k for k, v in results.items() if v["fluency"] > FLOOR),
           key=lambda k: results[k]["effect"])
```

### Priors, clearly labeled as heuristics

Pre-filter cheaply by class separation (largest `‖μ_A − μ_B‖`, or peak linear-probe accuracy), then confirm the top few causally — separability ≠ causal effect. Winners *tend* to sit in the **middle third** of layers `[heuristic]`; the cross-layer consistency cutoff (v1's "~0.8+ cosine") is a `[heuristic]` threshold, not a measured constant. **Always evaluate on held-out prompts, never the extraction set.** `[established]`

### Brittleness / reproducibility (v2 addition)

Steering effects frequently fail to transfer across prompt distributions, and results move with decoding temperature, random seed, and hook site. Fix and report these; a clean single-config result overstates robustness. `[established]` as a general caution; magnitudes are `[study-specific]`.

---

## 7. Application scope: single point vs. broadcast

A direction is extracted at one point but ablation typically applies at *all* layers/positions. Is that safe, and can a "max-diff" direction damage other layers?

### What licenses broadcasting — precisely

The shared residual stream makes the vector *representable* everywhere (§3); it does **not** guarantee the direction *means the behavior* everywhere. That guarantee is empirical.

### How the risk is controlled

Max-separation is only a heuristic for *which candidates to audition*, never the final selector. The held-out sweep scores each vector **as actually applied** (broadcast), so cross-layer damage surfaces as collateral and the candidate loses. You never ship a direction whose global application wasn't measured globally.

### When broadcasting bites

- **Drift.** The behavior sits on a slightly different direction at other layers, so one vector is off-axis there → under-removes (leaks) or clips neighboring content.
- **Entanglement.** The direction also carries useful signal elsewhere; projecting it out everywhere deletes that too. The refusal direction, for example, is reported to partly encode genuine "this is dangerous," so removing it broadly can erode legitimate caution. `[study-specific]`

### Diagnostic and remedies

Check cosine similarity of the per-layer directions. High alignment across the middle band → "linearly consistent," broadcasting one direction is safer. Scatter → risky (and a hint of subspace structure, §8). Remedies: select on global effect + collateral (the sweep already rejects destructive directions); extract/ablate **per-layer** directions (more faithful to drift, more compute); or restrict scope — a middle-to-late band only, or **partial ablation** `h − c·(h·v̂)v̂`, `0 < c < 1`, trading completeness for less collateral.

Note: broadcast-to-all-layers is fundamentally an *ablation* practice. Classic steering (addition) is usually extracted and applied at the *same* single layer, so the cross-layer mismatch mostly doesn't arise there.

---

## 8. When the behavior is a subspace, not a single direction

### Correcting v1's rank confusion

A **two-class mean difference is inherently a single vector** — `μ_A − μ_B` is rank-1. You therefore *cannot* obtain a multi-dimensional behavior subspace from two class means alone. v1's "SVD of `D = pos − neg`" was wrong on two counts: it implied paired data (element-wise subtraction of unaligned sets is meaningless) and it conflated the mean-difference direction with the *spread* of differences. A genuine subspace must come from one of the following sources:

1. **Multiple contrastive axes (most practical).** Define several related contrasts — e.g. distinct refusal categories, or the behavior across different prompt templates — each yielding its own mean-difference vector. Stack and orthonormalize them into a basis.
2. **Concept-erasure methods that use full covariance structure**, not just two means: INLP (iterative) and LEACE (closed-form). These account for within-class covariance that mean-difference ignores.
3. **PCA/SVD on genuinely paired per-example differences** — valid only when row `i` of A is truly matched to row `i` of B (e.g. same stem, toggled completion). Then the spread of the differences is meaningful.

### Detecting whether you need a subspace

If you have paired data, inspect the singular spectrum of the *centered, paired* differences; if you have multiple contrast axes, inspect how aligned their mean-difference vectors are (near-parallel → effectively rank-1; spread out → multi-dimensional). Corroborating tell: per-layer directions from §7 that scatter rather than align.

```python
# Option A: multiple contrast axes -> basis by orthonormalization
#   R has one normalized mean-difference vector per contrast, shape [m, d] at a chosen layer L.
R = torch.stack([ (A_c.mean(0) - B_c.mean(0)) for (A_c, B_c) in contrasts ])  # [m, d]
R = R / R.norm(dim=-1, keepdim=True)
# how many independent axes? singular values of R reveal effective rank:
U, S, Vt = torch.linalg.svd(R, full_matrices=False)   # inspect S; Vt[:k] ~ basis
Vk = Vt[:k]                                            # [k, d]

# Option B: paired per-example differences at layer L (rows aligned!)
#   Dp = pos[:, L, :] - neg[:, L, :]     # [N, d], requires true pairing
#   mean(Dp) is the diff-of-means direction; SVD of (Dp - Dp.mean(0)) describes its spread.
```

### The add-vs-ablate asymmetry sharpens

**Ablation generalizes cleanly.** Remove the whole subspace with `P = I − VₖVₖᵀ` (project onto the orthogonal complement). This is the concept-erasure setting; **LEACE** (closed-form, minimally-perturbing) is the principled default and **INLP** (iterative, with a decodability stopping rule) a natural alternative that also suggests `k`. `[study-specific]` for their exact guarantees.

```python
Vk = Vk                          # [k, d] orthonormal basis
P  = torch.eye(Vk.shape[1]) - Vk.T @ Vk
# hook: h_new = h @ P     (broadcast across layers/positions as in §7)
```

**Steering does not generalize cleanly.** Adding a fixed vector is inherently 1-D; "add the subspace" isn't well-defined. Options are compromises: steer along the dominant axis only, sum several component-vectors, or go **adaptive/conditional** (pick the in-subspace vector relevant to the current activation, or gate the steer on where the activation already sits — conditional activation steering). So: to *suppress* a subspace, project it out; to *induce* one, a single vector usually isn't enough.

### Choosing k

Same Pareto logic as choosing `α`, over dimensionality. Too small → the behavior **leaks** (re-encoded on the axes left in — a plausible reason single-direction refusal ablation is sometimes bypassable). Too large → useful computation erased, capability drops. Pick `k` near the spectrum elbow, then raise it until the concept is no longer linearly decodable on held-out data while collateral stays under threshold. `[heuristic]` for the elbow; `[established]` for the decodability-plus-collateral stopping criterion.

---

## 9. Dual-use note

These are neutral mechanisms with opposed applications. The refusal-ablation procedure that lets a safety researcher characterize guardrails is the standard way practitioners strip safety training from open weights; concept erasure that removes a harmful capability is mechanically identical to erasure that removes a protective one. The register that keeps this work defensible is understanding, evaluation, and hardening — which is the level this note stays at.

---

## 10. Recommended workflow (summary)

1. Small instruct model; extract at all layers via `output_hidden_states`.
2. Matched paired datasets, controlling for length/topic confounds.
3. Diff-of-means, last token, all layers; normalize; **fix and record the exact hook site** (§5).
4. Decide single-direction vs. subspace: check alignment of multiple contrast axes and/or paired-difference spectrum (§8) — remembering two class means give only one vector.
5. Sweep layer × coefficient on held-out prompts with **target, collateral, and random-control** metrics; account for normalization when interpreting `α` (§3); find the fluency cliff.
6. Match operation to goal: addition (same layer) to induce/amplify; projection (broadcast, or per-layer if directions drift) to suppress thoroughly; a subspace projector (LEACE/INLP) when rank-1 leaks.
7. Re-validate globally whenever broadcasting; report seed/temperature/hook-site.
8. Bake into weights only after inference-time validation, if permanence is required.

---

## References (recalled without network access — verify before citing)

All entries are `[unverified-cite]`: names and results are recalled from training, but exact titles, years, and author lists were **not** checked in this environment. Confirm against the primary sources.

- **Representation Engineering (RepE)** — Zou et al. Reading/controlling representations; umbrella framing.
- **Contrastive Activation Addition (CAA)** — Rimsky et al. Paired-prompt steering vectors.
- **Inference-Time Intervention (ITI)** — Li et al. Truthfulness steering.
- **Refusal in LLMs is mediated by a single direction** — Arditi et al. Canonical directional-ablation result. *Note the tension with §8:* subsequent work argues refusal is more distributed/multi-dimensional than a single direction; treat the "single direction" framing as contested, not settled.
- **INLP** — Ravfogel et al. Iterative nullspace projection.
- **LEACE** — Belrose et al. Closed-form, minimally-perturbing concept erasure.
- **Task arithmetic** — Ilharco et al. Editing via task vectors.
- **SAE feature steering** — sparse-autoencoder feature control (e.g. the "Golden Gate" demonstration).

---

## Appendix: changelog from v1

- **Fixed** the subspace section: a two-class mean difference is rank-1; a real subspace must come from multiple contrast axes, covariance-based erasure (LEACE/INLP), or genuinely paired differences. Rewrote the code accordingly and removed the mis-shaped `D = pos − neg`.
- **Added** normalization (LayerNorm/RMSNorm) treatment in §3 — residual-norm growth with depth, downstream re-normalization, and the consequence that one `α` is not uniform across layers.
- **Added** the norm-matched random-direction control as a required metric (§6), plus evaluation-validity and reproducibility cautions.
- **Reconciled** §3 and §7: the shared stream makes broadcasting *possible*, not *safe*; safety is empirical.
- **Tightened** hook-site precision (§5): HF `hidden_states` indexing and the off-by-one between extraction index and hooked module.
- **Hedged** invented numbers (cosine cutoff, "middle third," sample sizes, diff-of-means-vs-probe) with explicit `[heuristic]` / `[study-specific]` markers.
- **Flagged** all references as `[unverified-cite]` and noted the contested status of the "single refusal direction" claim.
- **Labeled** code as illustrative pseudocode where helper functions are undefined.
