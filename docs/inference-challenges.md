# Inference Pain Points & Open Issues

> A running catalogue of what makes LLM inference hard — collected from building
> `hf-model-inference/`. Each item notes *why it bites* and our *status*:
> ✅ handled · ⚠️ partial · ⬜ open / not yet addressed.
>
> The meta-lesson: inference is "easy" for one model and hard *in general*,
> because every model is a small variation and the variations hide in a dozen
> different places.

---

## 0. The meta-problem: architecture proliferation

Every flagship model is a *small variant* of a few base shapes, and the
differences hide in config flags, tensor names, tokenizers, and stop rules — not
in the headline architecture.

- **Self-declaration is the dispatch key, not the code.** `config.json` declares `model_type` / `architectures`, which the runtime maps to an implementation in a registry. But the file carries **no code** — the executor must already implement that arch. This is why new models are day-0 on `transformers` (remote code) but lag weeks in llama.cpp/MLX (someone hand-ports the forward pass). ⚠️
- **Two levels of variation.** Level 1 = *structural* (Mamba vs transformer vs MoE) → needs a new builder in a registry. Level 2 = *parametric* (bias on/off, tied embeddings, head_dim, rope scaling) → should be absorbed as config **fields**, not new classes. Mixing these up is how codebases explode. ⚠️
- **`model_type` is occasionally missing or mislabeled.** Robust loaders fall back to sniffing tensor names. ⬜
- **Our status:** single-family, field-parameterized (Llama/Qwen dense + Mixtral MoE). No level-1 registry yet — adding Mamba would currently mean a rewrite rather than a registered entry. ⬜

---

## 1. Weight loading & formats

- **Name matching.** `load_state_dict` only works if module names exactly equal checkpoint keys. The single biggest "load it yourself" failure. We mirror HF names to avoid remapping. ✅
- **Weight permutation.** HF Llama stores Q/K in a layout that must be *permuted* for some RoPE implementations (the HF→GGUF converter does this). Silent wrong outputs if missed. ✅ (we match HF's layout directly, so no permute needed)
- **Tied embeddings.** Many small models omit `lm_head.weight` and reuse `embed_tokens`. Must detect and re-tie. ✅
- **Expert naming differs by family.** Mixtral: `block_sparse_moe.experts.{e}.w1/w2/w3`. Qwen-MoE: `mlp.experts.{e}.gate_proj/up_proj/down_proj`. Same math, different keys → loader needs per-family mapping. ⚠️ (Mixtral only)
- **Sharded safetensors** + `model.safetensors.index.json` must be merged. ✅
- **dtype / device choice.** bf16 on CUDA, fp16 on MPS, fp32 on CPU; mismatches cause errors or silent precision loss. ✅
- **Pickle vs safetensors.** `.bin` weights are pickle (code-execution risk); prefer safetensors. ✅ (downloader skips `.bin`)

---

## 2. Tokenization & chat templates

- **Chat template must match training format.** Feed a model the wrong role markers and quality drops — it's seeing a distribution it wasn't trained on. Each family ships its own Jinja template. ✅ (we apply the model's own template)
- **`apply_chat_template` return type varies by transformers version** — bare tensor, dict, or `BatchEncoding`. Caused a real crash; needs normalization. ✅
- **Legacy vs modern tokenizer files.** `vocab.json` + `merges.txt` (old BPE) vs `tokenizer.json` (fast). Same vocab, different format. ✅
- **The model is stateless.** Multi-turn "memory" = resending the full message history each turn; the template re-renders it. ✅
- **Tokenizer correctness.** GGUF's embedded tokenizer can be lower-fidelity than the original; tokenization mismatches shift everything downstream. ⬜

---

## 3. Positional encoding

- **RoPE vs absolute.** Modern models rotate Q/K (RoPE); older ones add a learned position table (which caps context). Visible tell: no `pos_embd` tensor. ✅
- **Position offset with KV cache.** During decode you feed 1 token but must give it its *true* absolute position (`past_len`), or RoPE rotates it as if it were token 0. ✅
- **RoPE scaling variants** for context extension: `linear`, `yarn`, `llama3`. Each computes frequencies differently; wrong one → wrong long-context behavior. ⚠️ (llama3 + vanilla handled; YaRN not)
- **`rope_theta` differences** across models (10k vs 1M). Read from config. ✅

---

## 4. Attention variants

- **MHA / GQA / MQA.** `num_key_value_heads` < query heads = GQA; =1 = MQA. Wrong handling breaks shapes and the KV cache size. ✅
- **Causal masking: prefill vs decode.** Causal when q_len == kv_len (prefill); full attention to cache during single-token decode. Subtle off-by-one source. ✅
- **`head_dim` not always `hidden / heads`.** Newer configs set it explicitly. ✅
- **Attention bias is sometimes architectural, not a flag.** Qwen2 biases Q/K/V but doesn't always expose a config key; inferred from family. ⚠️
- **Newer attention features:** QK-norm, sliding-window attention, attention sinks, logit soft-capping (Gemma). Not yet supported. ⬜

---

## 5. KV cache

- **Memory grows with context.** `2 × layers × kv_heads × seq_len × head_dim × dtype`. The cache, not the weights, dominates memory at long context (the reason GQA exists). ✅ (correct, unoptimized)
- **Naive `torch.cat` reallocates every step** (O(n) copies). Production pre-allocates a fixed buffer (static / paged KV cache). ⬜
- **Per-layer, independent caches.** One `(K,V)` per layer, looked up/updated every step. ✅
- **Device residency.** Cache stays on GPU/MPS; only the sampled token id returns to host. (On Apple unified memory the CPU/GPU split blurs.) ✅

---

## 6. Mixture-of-Experts

- **All experts in memory, few active.** Memory-heavy to *serve* even though cheap to *run* — the central MoE trade. ✅ (structure)
- **Coarse vs fine-grained.** Mixtral (8 experts, top-2) vs Qwen3/DeepSeek (128 experts, top-8, smaller experts). Our routing handles both via config; only names/sizes differ. ⚠️
- **Shared experts.** DeepSeek-V2/V3 add always-on experts; Qwen3-MoE drops them. A structural variant. ⬜
- **Routing stability / load balancing** is a *training* concern; inference just routes top-k. (Good to know, not our problem at inference.) ✅
- **Dispatch efficiency.** Loop over experts (gather/scatter), never every token through every expert. ✅

---

## 7. Generation & decoding correctness

- **Multiple stop tokens.** `tokenizer.eos_token_id` is *one* id; `generation_config.json` may list several (Qwen: `<|im_end|>` **and** `<|endoftext|>`). Miss one → runaway generation. ✅
- **Template break at decode.** Model may fail to stop, repeat, or bleed into a fake next turn. Defenses: all stop ids + `max_new_tokens` cap + `skip_special_tokens`. ✅
- **Repetition / loops.** Repetition penalty (presence-based) implemented; frequency-based and no-repeat-ngram variants not. ⚠️
- **`generation_config.json` is split-authority.** Sampling params (temp/top-k/p) are *suggestions*; token-id fields (eos/pad/bos) are *authoritative*. Treat them differently. ✅
- **Sampling correctness.** top-k/top-p edge cases (keep-at-least-one, the HF nucleus shift), penalty divide-vs-multiply on sign. ✅

---

## 8. Quantization & memory

- **Full precision is expensive.** Weights alone ≈ `params × 2 bytes` (fp16). A 7B model ≈ 14 GB before cache/activations — the reason GGUF/quantization exists. ⬜ (we run full precision only)
- **Quant formats.** Legacy block quants (Q4_0…) vs K-quants (256-superblocks) vs IQ (importance-matrix) vs MXFP4. Each unpacks differently in-kernel. (Documented in `gguf-format.md`.) ⬜
- **KV-cache quantization** (q8_0 cache) — second big memory lever after weight quant. ⬜

---

## 9. Performance (beyond correctness)

Our implementation is correct but **slow** — these are the optimizations real engines add, none of which change the math:

- Fused kernels / **FlashAttention** (memory-efficient attention). ⬜
- **Paged / static KV cache** (no per-step reallocation). ⬜
- **Continuous batching** (serve many requests, fill gaps). ⬜
- **Speculative decoding** (draft model proposes, target verifies). ⬜
- Prefill vs decode have very different compute profiles (compute-bound vs memory-bound). ⬜
- Device differences: CUDA discrete VRAM vs Apple unified memory change the tradeoffs. (aware)

---

## 10. Numerical parity & correctness traps

- **RMSNorm in fp32** then cast back — needed to match the reference. ✅
- **RoPE exact layout** (half-split / NeoX vs interleaved) — wrong layout silently corrupts. ✅
- **Attention scale** (`1/sqrt(head_dim)`) and any custom scaling. ✅
- **Endianness.** GGUF is little-endian by default; big-endian models exist. ⬜
- **The only real correctness gate is matching reference logits** (`compare_logits.py`). Everything above can be subtly wrong and still "produce text." ✅ (harness exists)

---

## How to use this doc

When adding a model and it misbehaves, walk the list by *symptom*:

- **Garbage / wrong tokens from step 1** → §1 weights (names, permute, tie), §10 parity.
- **Coherent but off-format** → §2 chat template.
- **Won't stop / loops** → §7 stop ids, repetition.
- **Wrong past ~trained length** → §3 RoPE scaling.
- **Shape errors** → §4 attention heads / head_dim, §6 MoE config.
- **OOM** → §5 KV cache, §8 quantization.
- **"Unknown architecture"** → §0 dispatch / missing implementation.

Most "it doesn't work" bugs are one specific item here, not a deep mystery — which is the whole reason to keep the list.
