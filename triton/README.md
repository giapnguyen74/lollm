# Triton study

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
- `requirements.txt`   — install on the workstation: `pip install -r requirements.txt`

## Typical loop

1. Edit a kernel on the Mac.
2. Sync to the workstation.
3. `./run_interpret.sh <file>` — nail correctness with loose tolerances + prints.
4. `./run_gpu.sh <file>` — confirm it compiles on the GPU and benchmark vs PyTorch.

## Debug notes

- Most bugs are: wrong `mask` (OOB reads/writes → NaNs/garbage), wrong stride math,
  or a forgotten `tl.constexpr`.
- In a real GPU kernel use `tl.device_print("label", var)`.
- `out of resource: shared memory` means your tile is too big — shrink BLOCK sizes.
- Benchmark with `triton.testing.do_bench`; ignore the first call (JIT compile).
