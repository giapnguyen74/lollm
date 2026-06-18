# Project architecture — how lollm is wired

How the engine is laid out and why. For the *per-model* architecture write-ups see
the other `docs/*-architecture.md` files; for the family coding pattern see
[CONVENTIONS.md](../CONVENTIONS.md).

## Design at a glance

![Inference engine vision: spec + prompt → loader (config · weights · tokenizer) → probe model_type → route to family → shared generate loop → text](inference-engine-vision.svg)

## The flow

```
spec ─► loader (raw config + weights by file names + tokenizer)
     ─► router (model_type → qwen2 family)
     ─► qwen2.load(raw_config, weights, fmt)   ← family builds the model + maps its own weight names
     ─► shared generate loop (calls model.forward(ids, past)) ─► text
```

## Layout

```
src/
├── loader.py         # SHARED: fetch (HF/cache/gguf) → raw config + weights (file names) + tokenizer
├── router.py         # SHARED: probe model_type → route to model (fail loud)
├── generate.py       # SHARED: the one loop (prefill → decode → stop) + sampler
├── gguf_reader.py    # SHARED: parse GGUF (metadata + tensor table + raw bytes)
├── dequant.py        # SHARED: dequantize GGUF blocks (Q4_K, Q6_K, Q5_0, …)
├── tokenization.py   # SHARED: uniform tokenizer — HFTokenizer + GGUFTokenizer (embedded BPE)
├── progress.py       # SHARED: load-phase progress bar (caller-driven; model code stays UI-free)
├── models.py         # registry: Family record + register/get; imports each family (e.g. `import qwen2`)
├── run.py            # CLI: loader → router → model.load → generate (+ --think, --mtp)
├── compare_logits.py # parity gate vs transformers (reusable compare())
├── sanity_test.py    # run the parity gate across all families in one go
└── qwen2/                       # one self-contained family package
    ├── __init__.py              # manifest: MODEL_TYPES · DEFAULTS · register(load)
    ├── config.py                # Qwen2Config — parse config.json (hf) / metadata (gguf)
    ├── blocks.py                # small components: RMSNorm · RoPE · attention · MLP
    ├── modeling_qwen2.py        # architecture: DecoderLayer + Qwen2Model + forward
    ├── kv.py                    # the family KV cache (methods, not inline) — swappable later
    └── weights.py               # the weight-name seam (maps) + load (checkpoint → model)
```

The `qwen3_5/` package adds `blocks.py` (Gated DeltaNet + gated attention), `mtp.py`
(the optional speculative head), and a hybrid `kv.py` (growing KV for full layers +
fixed `(conv, recurrent)` slots for linear layers). `gemma3/` is `gemma2/` plus
QK-norm, GeGLU, and dual RoPE (its diff is in `docs/gemma3-architecture.md`).

A family package imports only its own siblings (`config`, `blocks`) + the shared
registry — never another family. Adding a model = drop a `<family>/` package + add
`import <family>` to `models.py`. See [CONVENTIONS.md](../CONVENTIONS.md) for the
file roles, the self-containment rule, and the numbered-step `forward` narration.
`qwen2/` is the reference to copy.

## Design seams (per the vision)

- **Loader is dumb.** It never renames tensors — weights come keyed by the file's
  own names + a `fmt` tag. The **family owns the name map** (`qwen2/weights.py`),
  which is where format quirks live.
- **Router only routes.** `model_type` → family; unknown → raises.
- **The loop is shared.** Families provide `forward(ids, past) → (logits, past)`,
  never their own loop.
- **The family is self-contained.** `modeling_qwen2.py` has its *own* RoPE / RMSNorm
  / attention / MLP — it imports only its siblings (`config`, `blocks`) and the
  shared registry, never another family. Duplication across families is intentional
  (study clarity).

## Verification

Each model is proven **end-to-end** (load → generate → `compare_logits` parity vs
`transformers`) on **Mac (MPS)** and **CUDA** — e.g. Qwen3.5-4B matches the
reference at cosine ≈ 1 (same top token), and Qwen3.6-27B runs on CUDA. Plus:
registry/router dispatch (unknown `model_type` raises), `weights.to_raw` maps, and
sampling precedence. `compare_logits.py` / `sanity_test.py` are the gate to re-run for
any new model.
