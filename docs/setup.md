# Setup — hardware, backends, and tested models

Companion to the README's **Install** section: the per-backend torch wheel details
and the list of models proven end-to-end. The README has the short path; this is
the full matrix.

## Hardware / backend

> ✅ **Validated on Apple Silicon (Mac M-series, MPS) and NVIDIA (CUDA).** Parity
> also runs on **CPU** (the `compare_logits` gate is fp32 on CPU). **ROCm** is
> written against the generic `torch`/`nn` API and *should* work (it presents as
> `cuda`) but isn't verified yet.

The PyTorch **version** (`torch>=2.2`) and the **backend** (CPU / CUDA / ROCm / MPS)
are separate: the version is pinned in `pyproject.toml`, but the backend is chosen at
*install time* by which wheel index you pull from — `pyproject.toml` can't pick it
for you (there's no way to detect a GPU from a dependency spec). So install `torch`
first for your hardware, then `pip install -e .` (which leaves your chosen build in
place):

```bash
# macOS / Apple Silicon — the default wheel already includes CPU + MPS (the tested path)
pip install torch

# NVIDIA (CUDA) — pick the tag matching your driver; see the selector linked below
pip install torch --index-url https://download.pytorch.org/whl/cu126

# AMD (ROCm, Linux) — exposes the device as "cuda" in code
pip install torch --index-url https://download.pytorch.org/whl/rocm7.1

# CPU-only (any OS)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e .                  # then the rest of the deps
```

Compute tags (`cu126`, `cu128`, `rocm7.1`, …) change over time — the official
[PyTorch Get Started selector](https://pytorch.org/get-started/locally/) always
generates the current command for your OS + CUDA version. (For a locked, per-backend
setup, [uv](https://docs.astral.sh/uv/guides/integration/pytorch/) can declare the
torch index in `pyproject.toml` with platform markers.)

**Verify what you got.** `run.py` auto-detects the device (cuda → mps → cpu) and the
matching dtype, so check it landed where you expect:

```python
import torch
print(torch.__version__)                  # the +cuXYZ / +cpu suffix tells you the backend
print(torch.backends.mps.is_available())  # Apple Metal (the validated path)
print(torch.cuda.is_available())          # NVIDIA / ROCm
```

`run.py` already maps each backend to its fast dtype — **bfloat16** on CUDA,
**float16** on MPS, **float32** on CPU — so once the right wheel is installed the
engine adapts automatically.

## Tested models

End-to-end (load → generate → `compare_logits` parity vs `transformers`) on both Mac
(MPS) and CUDA:

| family | model | notes |
|---|---|---|
| `qwen2`   | `Qwen/Qwen2.5-0.5B-Instruct` | dense |
| `qwen3`   | `Qwen/Qwen3-0.6B`            | QK-norm, no bias |
| `gemma2`  | `google/gemma-2-2b-it`      | sandwich norm, sliding window, soft-caps (gated repo) |
| `gemma3`  | `google/gemma-3-1b-it`      | QK-norm (replaces soft-caps), 5:1 local/global dual RoPE, sandwich norm (gated repo) |
| `qwen3_5` | `Qwen/Qwen3.5-4B`           | hybrid GDN + gated attention; parity cosine ≈ 1 |
| `qwen3_5` | `Qwen/Qwen3.6-27B`          | same family (untied `lm_head`, larger); runs on CUDA |

`sanity_test.py` runs the parity gate across the small models in one shot:
`python src/sanity_test.py` (or `--only qwen3_5`).

> **gemma2 parity:** Gemma2 uses attention logit soft-capping, which standard SDPA
> skips. To compare, load the reference with eager attention:
> `AutoModelForCausalLM.from_pretrained(..., attn_implementation="eager")`. Gemma3
> dropped soft-caps (QK-norm instead), so it doesn't require eager — but eager is
> still the safest apples-to-apples comparison.
