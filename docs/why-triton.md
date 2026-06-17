# Why Triton — Hitting the Memory-Bandwidth Wall

> The reasoning behind the `triton/` study folder. We didn't pick up Triton because
> kernels are fun; we picked it up because once a working engine exists, the next
> question is *tok/s*, and the answer to *tok/s* is almost never "more math." It's
> "move fewer bytes." This doc is the chain of reasoning from "inference is simple"
> to "so we learned to write kernels."

---

## 1. The math is simple — that was never the hard part

A forward pass is plain arithmetic: matmuls, a softmax, some norms, repeated for each
layer (Chapters 1–2). There are no gradients, no training machinery; every number is
frozen and we just use it. You can write the whole thing in a few hundred readable lines,
and this repo does.

So if the math is easy, where does the difficulty go? Into **scale** and **engineering**.
A model is billions of these trivial operations over tens of gigabytes of weights, and the
moment you care how *fast* it runs, you stop fighting the math and start fighting the
hardware.

## 2. The wall is memory bandwidth, not compute

Here is the surprise that reorganizes everything. Generating one token, at batch size 1,
looks like this: stream **every active weight** plus the **entire KV cache** from VRAM into
the GPU's cores, use each value for a tiny handful of multiply-adds, and throw it away. Then
do it all again for the next token.

The compute is cheap and finishes almost instantly. The *byte movement* is what costs the
latency. Modern GPUs can do tens of TFLOP/s but "only" a few TB/s of memory bandwidth, and
in decode you read far more bytes than you do useful FLOPs per byte. The cores spend most of
their time **waiting on memory**.

A concrete ceiling makes it undeniable. A 35B-parameter dense model in bf16 is ~70 GB. At
~2 TB/s of memory bandwidth you can read the weights at most ~28 times per second — so you
cannot exceed **~28 tok/s**, no matter how fast the arithmetic units are. The weights-read
per token *is* the speed limit. Faster cores buy you nothing here.

### Prefill vs decode — the same wall, two sides

This is why the two phases of inference behave so differently:

- **Prefill** (process the whole prompt at once) reuses each loaded weight across *many*
  query rows. High FLOPs-per-byte → **compute-bound**. The cores are the bottleneck.
- **Decode** (one token at a time) uses each loaded weight for a *single* row. Low
  FLOPs-per-byte → **memory-bound**. The bus is the bottleneck.

The formal name for "FLOPs per byte" is **arithmetic intensity**, and plotting it against the
hardware's compute/bandwidth limits is the **roofline model**. Decode lives far to the left of
the roofline — pinned under the memory roof with the compute roof unused overhead.

## 3. Two ways past the wall

Everything that improves single-stream tok/s is one of two moves.

### Lever 1 — move fewer bytes (model / representation)

Change *what* has to be read each token.

- **MoE** cuts the **weight** bytes. Qwen3.5-35B-A3B activates ~3B of 35B params per token —
  ~10× less weight traffic. (Caveat: it's 3B *somewhere* in the 35B, so routing and locality
  matter.)
- **Linear attention / GQA / MLA** cut the **KV** bytes. The KV cache grows with context and
  is re-read every token, so at long context it can dominate even the weights. Linear attention
  swaps the growing cache for a fixed-size state; GQA shares KV heads; MLA compresses them.
- **Quantization** cuts the **weight** bytes directly: int4 weights are 4× smaller than fp16,
  so the bandwidth ceiling rises ~4× — paid for in a little accuracy and some dequant compute.
  (See `quantization.md`; the *fused dequant-matmul* idea in `optimization.md` is this lever.)

### Lever 2 — trade compute for bytes (systems)

Keep the model fixed; spend the idle FLOPs to avoid touching HBM. Since decode leaves the
compute roof unused, **extra arithmetic is effectively free** if it deletes memory traffic.

- **FlashAttention / kernel fusion.** Recompute softmax tiles in on-chip SRAM and keep
  intermediates in registers instead of writing the big attention matrix out to HBM and reading
  it back. Spend FLOPs to delete round-trips.
- **Speculative decoding** (our MTP path). Verify several draft tokens in a *single* forward
  pass, amortizing one expensive weight-read across multiple committed tokens. It literally
  converts memory-bound decode into more compute-bound work.

## 4. Why this forces you into Triton

Here is the link to kernels. In plain PyTorch, every operation is its own kernel launch: it
reads its inputs from HBM, computes, and writes its output back to HBM. A chain of ops is a
chain of HBM round-trips. That is exactly the traffic Lever 2 wants to eliminate — but you
**cannot** eliminate it from the Python level, because the round-trips happen *between* the
ops you're calling.

To fuse operations, to keep an intermediate in SRAM, to decide what stays in registers and how
tiles march through memory — you have to write the kernel yourself. The classic options are:

- **CUDA C++** — total control, maximum effort, easy to get wrong.
- **Triton** — Python-like kernels where you control the *memory-relevant* decisions (block
  sizes, what's loaded into SRAM, masking) while the compiler handles register allocation and
  instruction scheduling. The approachable way to *act on* the bandwidth insight.

So Triton isn't a detour from the inference story — it's the tool that lets you implement
Lever 2 at all. Learning it is how you turn "decode is memory-bound" from a diagnosis into a
fix. FlashAttention is the canonical worked example, which is why the study folder ports it.

## 5. What we actually found

We productionized the `05` flash-attention KV kernel (`src/_triton_attn.py`) behind an opt-in
dispatcher (`src/attention.py`, `LOLLM_ATTN`), with autotune-once + a startup warmup so the
compile/tune cost doesn't land on the first token. Mechanics live in `triton/README.md`.

The result, honestly read, *confirms the theory*:

- In **normal decode**, our kernel is a **tie** with PyTorch SDPA — because torch already
  dispatches to FlashAttention-2, which is *already* doing the memory-optimal thing. You can't
  beat memory-optimal with a second memory-optimal kernel.
- The real win shows up in the **MTP / chunked-prefill case** (`q_len > 1` with a populated
  cache), the one path torch doesn't serve with its fast causal kernel. (Quantifying that is
  future work — left for the MTP pass.)

The lesson is the point of the whole exercise: a tie isn't a failure, it's evidence that the
bottleneck is real and that the baseline is already on the roof. The durable wins come from
**Lever 1** (fewer bytes by design — MoE, linear attention, quantization) and from applying
**Lever 2** where the baseline *isn't* already optimal. Triton is what makes both inspectable
and, when it's worth it, improvable.

---

*See also: `optimization.md` (deferred perf ideas), `quantization.md` (Lever 1 in bytes),
`triton/README.md` (kernel mechanics and the `LOLLM_ATTN` integration).*
