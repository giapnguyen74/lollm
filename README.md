<img src="docs/lollm-logo.svg" alt="lollm — learn only LLM" height="72">

# lollm — a study-first LLM inference engine

A small, readable inference engine for **studying how LLMs (and related models) run
inference**. A thin **loader + router**, a shared **generate loop**, and
self-contained **family packages** — currently **qwen2 · qwen3 · gemma2 · gemma3 ·
qwen3_5** (the last covering **Qwen3.5 / Qwen3.6**: a hybrid Gated-DeltaNet +
gated-attention decoder, with an optional MTP speculative head), from HF safetensors
**and** GGUF (qwen3_5 is safetensors-only). PyTorch is the only dependency for the
actual model math.

## Vision

1. **Study-first.** The point is to *read and understand* LLM inference end to end —
   load, tokenize, the forward pass, sampling, quantized weights. Clarity for a
   learner beats every other concern.
2. **PyTorch-only for the model.** All LLM-related math (the architecture, attention,
   norms, RoPE, the generate loop) depends on **PyTorch alone** — no `transformers`
   modeling, no `llama.cpp`. We parse GGUF and dequantize ourselves (numpy) and build
   each architecture from `nn.Module` primitives. (`huggingface_hub` only *downloads*
   files — it never touches the forward pass.) **Known wart:** the safetensors path
   still leans on `transformers.AutoTokenizer` for tokenization; we plan to drop that
   later (the GGUF path already tokenizes on its own).
3. **Hard-fail, never guess.** When something is unknown or unverified — a missing
   `model_type`, an unmapped weight name, an unconfirmed GGUF key — we **raise loudly**
   instead of falling back to a plausible default. A crash tells us exactly what to
   fix; a silent guess emits confident garbage.
4. **Readable over optimized.** Prefer the clear implementation to the fast one.
   Duplication across families is intentional; each architecture reads on its own.
   We optimize only when it doesn't cost clarity (e.g. streaming weight load).
5. **Validate against transformers.** Prove each implementation against `transformers`
   as the reference: `compare_logits.py` runs our model and the reference on the same
   prompt and checks they predict the same next token (same argmax + cosine ≈ 1).
   **Run it before trusting any model's output** — it's how we catch a wrong RoPE,
   norm, or weight map.

## Approach

```
spec ─► loader (raw config + weights by file names + tokenizer)
     ─► router (model_type → family)
     ─► <family>.load(raw_config, weights, fmt)   ← family builds the model + maps its own weight names
     ─► shared generate loop (calls model.forward(ids, past)) ─► text
```

- **Loader is dumb** — never renames tensors; the **family owns the name map**.
- **Router only routes** — `model_type` → family; unknown → raises.
- **The loop is shared** — families provide `forward(ids, past) → (logits, past)`.
- **Each family is self-contained** — its own RoPE / norm / attention / MLP; imports
  only its siblings + the shared registry, never another family.

Full layout, the diagram, and the verification setup live in
**[docs/architecture.md](docs/architecture.md)**; the family coding pattern (file
roles, numbered-step `forward`) is in **[CONVENTIONS.md](CONVENTIONS.md)**. `qwen2/`
is the reference to copy.

## Install

Dependencies live in `pyproject.toml`. Use a **virtual environment**.

```bash
# 1. create + activate a venv (Python ≥ 3.10)
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. install torch for YOUR backend first (see docs/setup.md), then the project
pip install torch                 # macOS/MPS default; CUDA/ROCm/CPU → docs/setup.md
pip install -e .                  # editable: src/ changes take effect immediately
```

`pip install -e .` pulls in torch, numpy, safetensors, transformers, huggingface_hub,
regex, and jinja2 (floors pinned in `pyproject.toml`) and exposes two console scripts
— `lollm` (run) and `compare` (parity gate). For an exact, reproducible environment,
freeze: `pip freeze > requirements.lock`.

> **Backend, tested models, and the per-GPU torch wheel matrix:**
> see **[docs/setup.md](docs/setup.md)**. Validated on Apple Silicon (MPS) and NVIDIA
> (CUDA); CPU runs the parity gate; ROCm should work but isn't verified.

## Running

```bash
# safetensors (HF repo / local dir)
python src/run.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Explain RoPE in one line."
python src/run.py --model ./local/qwen2/dir --prompt "Hi" --temperature 0   # greedy
python src/run.py --model google/gemma-3-1b-it --prompt "Explain RoPE in one line."  # gemma3 (gated)

# Qwen3.5 / Qwen3.6 (hybrid GDN family)
python src/run.py --model Qwen/Qwen3.5-4B  --prompt "What is coffee"          # direct answer
python src/run.py --model Qwen/Qwen3.5-4B  --prompt "What is coffee" --think  # show <think>…</think>
python src/run.py --model Qwen/Qwen3.5-4B  --prompt "What is coffee" --mtp    # self-speculative (MTP)

# GGUF (local .gguf or repo:QUANT — downloaded + dequantized)
python src/run.py --model Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M --prompt "Hi"
python src/run.py --model ./qwen2.5-0.5b-instruct-q4_k_m.gguf   --prompt "Hi"

# parity gate — one model, or all families at once
python src/compare_logits.py --model Qwen/Qwen2.5-0.5B-Instruct
python src/sanity_test.py                                  # qwen2/qwen3/qwen3_5/gemma2/gemma3
```

## Status

| family | safetensors | GGUF | notes |
|---|---|---|---|
| `qwen2`        | ✅ | ✅ | dense |
| `qwen3`        | ✅ | ✅ | QK-norm, no bias |
| `gemma2`       | ✅ | 🚧 hard-fail | sandwich norm, sliding window, soft-caps |
| `gemma3`       | ✅ | 🚧 hard-fail | QK-norm (replaces soft-caps), 5:1 local/global dual RoPE |
| `gemma4`       | ✅ text-only | 🚧 hard-fail | PLE + shared-KV + proportional global RoPE + double-wide MLP + per-layer residual scale; **parity ✅** on `google/gemma-4-e2b-it` (cosine ≈ 1); vision/audio towers not built |
| `qwen3_5`      | ✅ | — | hybrid GDN + gated attention; MTP head, `--think` toggle |
| `qwen3_5_moe`  | ✅ | — | same backbone + sparse MoE FFN (fused experts) |

✅ = parity-verified vs `transformers` (same top token, cosine ≈ 1). 🚧 **GGUF
hard-fails by design** for the gemma families: their arch-specific metadata keys
aren't validated against llama.cpp yet, so per "hard-fail, never guess" we raise
rather than default (see each family's `config.py::from_gguf`).

Shared infra: loader/router/streaming generate loop · GGUF parse + dequant
(Q4_K/Q5_K/Q6_K/Q5_0/…) · uniform tokenizer (BPE + SentencePiece) · streaming weight
load (peak ≈ steady) with a progress bar · cache-aware download skip · per-family
`kv.py` cache · an optional Triton flash-attention kernel (CUDA, opt-in via
`LOLLM_ATTN`; torch SDPA is the default/validated path) · the `compare_logits` /
`sanity_test` parity gate plus offline family self-tests (e.g. `gemma3_selftest.py`).

## TODO

- ✅ **gemma4** text decoder (`src/gemma4/`: PLE, shared-KV, proportional global RoPE,
  double-wide MLP, per-layer residual scale, final soft-cap) — **parity ✅** on
  `google/gemma-4-e2b-it`. Remaining: vision/audio towers (see
  `docs/multimodal-processors.md`).
- ⬜ **llama** family.
- ⬜ **GGUF MoE** (stacked experts) — and validate gemma2/gemma3 GGUF metadata keys
  against llama.cpp to lift the hard-fail.
- ⬜ Drop the `transformers.AutoTokenizer` dependency on the safetensors path.

See **[docs/ISSUES.md](docs/ISSUES.md)** for the reviewed backlog (fp16-on-MPS Gemma
risk, KV-cache/perf cliffs, etc.).
