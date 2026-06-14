# Optimization notes

> A parking lot for performance ideas we've *considered but deliberately deferred*.
> Per the vision (study-first, readable over optimized), the default code stays the
> clear implementation; this doc records the faster alternatives, what they'd cost in
> clarity, and when they'd be worth it. Status: ✅ done · 🚧 planned · ⬜ idea only.

---

## GGUF dequantization: where to turn quantized bytes into weights

**Current state (✅ shipped, the readable reference).** GGUF weights are dequantized
**on the CPU in numpy** (`dequant.py`), producing fp16 tensors, and only then moved to
the device (`loader._load_gguf` → `weights.py`). The flow is:

```
mmap quantized bytes ─► numpy dequant → fp16 (CPU/RAM) ─► torch.from_numpy ─► .to(device)
```

Consequences worth being explicit about:

- Dequant math runs on CPU; the **full fp16 model is materialized in RAM** before the
  device transfer begins.
- We pay quantization's **disk** savings but get **zero runtime memory savings** — once
  loaded, the model occupies full fp16 size in RAM/VRAM. Quantization here is
  *load-time decompression*, not *memory-resident quantization*.

This is the right default for a study engine: `dequant.py` is the single clearest place
to learn how Q4_K / Q5_K / Q6_K actually unpack. We keep it.

There are two distinct ways to do better, with very different payoff and cost.

### Idea A — dequantize on the device, still materialize fp16 ⬜

Send the **quantized bytes** to the device and run the bit-unpacking there; keep the
resulting fp16 tensor in VRAM.

- **Wins:** ~4× less host→device transfer (move Q4 bytes, not fp16); dequant is
  embarrassingly parallel, so the GPU eats it. Faster *load*.
- **Stays PyTorch-only:** the numpy ops (`& 0x0F`, `>> 4`, reshape, fp16 reinterpret via
  `.view`) all have torch equivalents (`bitwise_and`, `bitwise_right_shift`,
  `view(torch.float16)`), so no custom CUDA needed.
- **Costs:** the torch port reads worse than the numpy reference; and **MPS
  integer/bitwise op coverage is spotty**, so on our validated Apple-silicon path it
  risks silent CPU fallbacks or unsupported-op errors — each K-quant kernel would need
  to be verified on `mps` first.
- **Memory:** does **not** reduce resident model size (still full fp16 in VRAM). Only
  helps load time / transfer.

### Idea B — fused dequant-matmul: keep weights quantized in VRAM ⬜ (the real prize)

Never materialize fp16 weights at all. Keep the weights **quantized in VRAM** and
dequantize on the fly *inside the matmul* (a fused dequant-matmul kernel) — what
llama.cpp / MLX / production engines do.

- **Wins:** this is the whole point of quantization — a quantized model actually **fits
  in less VRAM**, and inference (memory-bandwidth-bound) is often *faster* because you
  move fewer bytes per matmul. Trades a little compute (unpack each block) for a lot of
  VRAM.
- **Costs:** a big complexity jump — custom fused kernels per quant type, hard to keep
  readable and effectively impossible in pure PyTorch on MPS. Really a separate study
  topic of its own.

### Decision / priority

**Load time is not a current concern**, so Idea A's payoff (faster load, smaller
transfer) doesn't buy us much, and it adds MPS risk for no resident-memory benefit. We
**skip A**. When we do reach for optimization here, **go straight to B** — it's the only
one that delivers the actual goal of quantization (less VRAM), even though it's the
harder build. Until then, the CPU-numpy reference stands.

(Cheap, unrelated tidy available any time: *pipeline* the existing dequant — dequant one
tensor, move it to the device, free it, next — to drop CPU peak from "whole fp16 model"
to "one tensor," with no readability loss and no MPS risk. Not an A/B step; just better
housekeeping on the current path.)
