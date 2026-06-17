# Triton study

> **Why this folder exists:** see `docs/why-triton.md` — the reasoning from "inference math
> is simple" to "decode is memory-bandwidth-bound, so we learned to write kernels." This
> README is the *how* (the run loop and the lollm integration); that doc is the *why*.

Workflow: **edit on macOS, run on the CUDA workstation.**
Triton has no macOS build (Linux wheels only), so nothing here runs locally on the
Mac — including `TRITON_INTERPRET=1`, which still needs to `import triton`. The Mac
is the editor; the workstation is the runtime.

## On the workstation

Two run modes, same code:

```bash
# Fast logic/correctness loop — pure-Python interpreter on CPU, no GPU.
# Real print() / pdb / breakpoints work. Slow, but great for indexing & masking bugs.
./run_interpret.sh 01_vector_add.py

# Real run — compiles to the GPU, validates performance.
./run_gpu.sh 01_vector_add.py
```

`run_interpret.sh` sets `TRITON_INTERPRET=1`; `run_gpu.sh` leaves it unset.

## Files

- `common.py`      — device/interpret helpers shared by the exercises
- `01_vector_add.py`   — hello-world: blocks, offsets, masks
- `02_fused_softmax.py` — row-wise stable softmax (the seed of online softmax / FA)
- `05_flash_attention_kvcache.py` — offset-causal flash attention with a KV cache
  (prefill / chunked prefill / decode in one kernel); the kernel wired into lollm below
- `requirements.txt`   — standalone deps for these exercises (`pip install -r requirements.txt`)

To run the kernel **inside lollm**, install the optional extra on the workstation instead:
`pip install -e ".[triton]"`. Triton is not a core dependency (Linux-only wheels), so the
default install stays torch-only and the kernel is opt-in via `LOLLM_ATTN` (see below).

## Typical loop

1. Edit a kernel on the Mac.
2. Sync to the workstation.
3. `./run_interpret.sh <file>` — nail correctness with loose tolerances + prints.
4. `./run_gpu.sh <file>` — confirm it compiles on the GPU and benchmark vs PyTorch.

## Using the kernel in lollm

The `05` kernel is productionized in `src/_triton_attn.py` and wired into the model
attention blocks through a single dispatcher, `src/attention.py`. **Torch is the default
everywhere; Triton is opt-in.** Every `Attention.forward` calls `attention(q, k, v)` instead
of `F.scaled_dot_product_attention` directly; the dispatcher picks the path:

```
LOLLM_ATTN unset / =torch   torch SDPA always (default)
LOLLM_ATTN=triton           Triton kernel; raises if unavailable (needs CUDA tensor + triton)
LOLLM_ATTN=auto             Triton when available (CUDA + installed), else torch
```

The `import triton` is lazy and guarded, so on the Mac (no Triton wheel) the import fails
once, caches `False`, and we stay on torch — nothing to install locally. `src/_triton_attn.py`
is the productionized copy; this `triton/` folder stays the edit-on-Mac / run-on-workstation
study sandbox.

### Two production changes vs the study kernel

- **Autotune keys on `HEAD_DIM` only**, not on sequence length. `total_k` grows by one every
  decode step, so keying on it re-ran the whole 16-config sweep *per token*. Head dim is
  constant across a run → tune once, reuse.
- **Stride-aware loads.** The kernel takes per-dim strides and reads Q/K/V from their
  `(B,H,S,D)` views, so there's no `.reshape(...).contiguous()` copy.

### Warmup (important)

Even after the autotune fix, the one-time sweep + JIT compile lands on the *first* attention
call — layer 0's prefill — inflating time-to-first-token. `run.py` calls `attention.warmup(...)`
right after model load (with the model's real `head_dim`/`n_heads`) to run one throwaway
prefill-shaped call, moving that cost off the critical path. It's a no-op unless `LOLLM_ATTN`
enables Triton on CUDA. You'll see `[triton attention: warming up (...)]` on stderr when it fires.

## Debug notes

- Most bugs are: wrong `mask` (OOB reads/writes → NaNs/garbage), wrong stride math,
  or a forgotten `tl.constexpr`.
- In a real GPU kernel use `tl.device_print("label", var)`.
- `out of resource: shared memory` means your tile is too big — shrink BLOCK sizes.
- Benchmark with `triton.testing.do_bench`; ignore the first call (JIT compile).
