# Known issues / review backlog

Tracked-but-not-yet-fixed items from a project self-review. Each has enough context
to pick up cold. Ordered roughly by importance. (Resolved/decided items at the bottom.)

---

## 1. fp16 on MPS is a latent Gemma overflow risk the parity gate can't see

**Severity:** medium–high (correctness on a "supported" path).

`run.py::pick_dtype` returns **float16 on MPS**, but the parity gate
(`compare_logits.py`) runs **fp32 on CPU** — so any fp16-specific overflow is
invisible to it. Gemma is the worst case: large activations from the ×√hidden
embedding scale and the wide GeGLU intermediate, plus Gemma's known historical fp16
`inf` problems. Softmax and RMSNorm are upcast to fp32 (good), but the q·kᵀ matmul
and GeGLU run in fp16 on MPS.

So "validated on MPS" + "parity verified" can both be true while a Gemma MPS run
still produces NaNs/garbage that no automated check would catch.

**Options to consider**
- Default Gemma (or all models) to **bfloat16 on MPS** where the torch build supports
  it; fall back to fp16 only if bf16 is unavailable.
- Or: add an optional fp16 path to the parity gate (run our model in fp16 on the
  accelerator, reference in fp32) and assert cosine still ≈ 1.
- At minimum: document that fp16 + Gemma on MPS is unvalidated.

**Where:** `src/run.py::pick_dtype`; verification in `src/compare_logits.py`.

---

## 2. Performance cliffs in the KV cache and decode loop

**Severity:** medium (fine at study/short lengths; bites exactly the long-context
cases Gemma3/4 target).

- **O(T²) cache growth.** `*/kv.py` appends with `torch.cat` every decode step, so the
  whole K/V is reallocated and copied each token → quadratic total copying over a
  generation. A pre-allocated ring/grown buffer would fix it but costs readability.
- **Sliding layers cache the full K/V.** Gemma local layers only need the last
  `sliding_window` (512) keys, but the cache keeps everything and the window is applied
  in the mask. Wasted memory that grows with context (a production engine would shrink
  the local-layer cache).
- **Batch size 1 only.** `generate.py` wraps a single sequence (`[ids]`); no batched
  decoding.
- **GGUF dequant is eager, not streamed.** `loader.py::_load_gguf` dequantizes *every*
  tensor into one CPU dict up front, so the "streaming load, peak ≈ steady" property
  (true for safetensors) does **not** hold for GGUF — peak CPU ≈ full fp16 model.

**Where:** `src/*/kv.py`, `src/generate.py`, `src/loader.py::_load_gguf`.

---

## 3. Stale doc reference: `qwen3_5_selftest.py` does not exist

`README`/docs referenced `src/qwen3_5_selftest.py` (conv causality, recurrent==chunked
delta-rule, prefill==incremental, MTP shapes), but the file isn't in the repo. Either
restore it or scrub the reference. (The new `src/gemma3_selftest.py` is the pattern to
follow if recreating it.) The verification blurb now lives in `docs/architecture.md`.

**Where:** `docs/architecture.md` (verification section), `src/`.

---

## 4. Gemma attention is unoptimized (no FlashAttention / SDPA)

**Severity:** medium (perf only; correct today). Related to #2.

`attention.torch_attention_with_scale` (used by gemma2/gemma3) does **not** call
`F.scaled_dot_product_attention`, so it gets **no FlashAttention / memory-efficient
kernel**. It materializes the full `(B,H,Tq,Tk)` scores tensor (O(T²) memory), runs
eager (matmul → softcap → mask → fp32 softmax → matmul = several kernel launches), and
rebuilds the `arange`/mask tensors every layer and every decode step. Fine for
short-prompt study; a real cliff at long context.

Why it bypasses SDPA differs by family:

- **gemma2 must** stay manual — it needs tanh **attention-logit soft-capping**
  mid-kernel, which torch's built-in SDPA can't express (same reason HF forces
  `attn_implementation="eager"` for Gemma2).
- **gemma3 / gemma4 could be optimized** — they dropped soft-caps, and both things
  they need are expressible in SDPA: the custom scale via `scaled_dot_product_attention(
  ..., scale=query_pre_attn_scalar**-0.5)`, and the sliding window via an additive
  `attn_mask`. Caveat: passing an explicit mask can disable the fastest causal-only
  flash fast-path (may route to the memory-efficient backend instead).

**Options to consider**
- Route gemma3/gemma4 through SDPA (`scale=`, `attn_mask=` for the band) to get the
  fused kernels; keep gemma2 on the manual path.
- Or write a Gemma-specific Triton kernel fusing scale + QK-norm + band (+ optional
  softcap) — the broader "custom kernel" investigation.
- Either way, keep `torch_attention_with_scale` as the readable reference + the
  `gemma3_selftest.py` oracle to validate any optimized path against.

**Where:** `src/attention.py`, `src/gemma2/blocks.py`, `src/gemma3/blocks.py`.

