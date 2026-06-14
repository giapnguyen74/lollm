# lollm — a study-first LLM inference engine

A small, readable inference engine for **studying how LLMs (and related models) run
inference**. A thin **loader + router**, a shared **generate loop**, and
self-contained **family packages** — currently **qwen2 · qwen3 · gemma2**, from HF
safetensors **and** GGUF. PyTorch is the only dependency for the actual model math.

## Vision

1. **Study-first.** The point is to *read and understand* LLM inference end to end —
   load, tokenize, the forward pass, sampling, quantized weights. Clarity for a
   learner beats every other concern.
2. **PyTorch-only for the model.** All LLM-related math (the architecture, attention,
   norms, RoPE, the generate loop) depends on **PyTorch alone** — no `transformers`
   modeling, no `llama.cpp`. We parse GGUF and dequantize ourselves (numpy) and build
   each architecture from `nn.Module` primitives. (`huggingface_hub` only *downloads*
   files — that's fine, it never touches the forward pass.) **Known wart:** the
   safetensors path still leans on `transformers.AutoTokenizer` for tokenization; we
   plan to drop that dependency later (the GGUF path already tokenizes on its own).
3. **Hard-fail, never guess.** When something is unknown or unverified — a missing
   `model_type`, an unmapped weight name, an unconfirmed GGUF key — we **raise loudly**
   instead of falling back to a plausible default. A crash tells us exactly what to
   fix; a silent guess emits confident garbage.
4. **Readable over optimized.** Prefer the clear implementation to the fast one.
   Duplication across families is intentional; each architecture reads on its own.
   We optimize only when it doesn't cost clarity (e.g. streaming weight load).
5. **Validate against transformers.** We prove each implementation against
   `transformers` `AutoModelForCausalLM` as the reference: `compare_logits.py` runs
   our model and the reference on the same prompt and checks they predict the same
   next token (same argmax + cosine ≈ 1). **If you use this repo, run
   `compare_logits.py` to validate any model/architecture before trusting its
   output** — it's how we catch a wrong RoPE, norm, or weight map.

```bash
# safetensors (HF repo / local dir)
python src/run.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Explain RoPE in one line."
python src/run.py --model ./local/qwen2/dir --prompt "Hi" --temperature 0   # greedy

# GGUF (local .gguf or repo:QUANT — downloaded + dequantized)
python src/run.py --model Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M --prompt "Hi"
python src/run.py --model ./qwen2.5-0.5b-instruct-q4_k_m.gguf   --prompt "Hi"

python src/compare_logits.py --model Qwen/Qwen2.5-0.5B-Instruct             # parity gate (safetensors)
```

## Design at a glance

![Inference engine vision: spec + prompt → loader (config · weights · tokenizer) → probe model_type → route to family → shared generate loop → text](docs/inference-engine-vision.svg)

## Layout

```
src/
├── loader.py         # SHARED: fetch (HF/cache/gguf) → raw config + weights (file names) + tokenizer
├── router.py         # SHARED: probe model_type → route to model (fail loud)
├── generate.py       # SHARED: the one loop (prefill → decode → stop) + sampler
├── gguf_reader.py    # SHARED: parse GGUF (metadata + tensor table + raw bytes)
├── dequant.py        # SHARED: dequantize GGUF blocks (Q4_K, Q6_K, Q5_0, …)
├── tokenization.py   # SHARED: uniform tokenizer — HFTokenizer + GGUFTokenizer (embedded BPE)
├── models.py         # registry: Family record + register/get; imports each family (e.g. `import qwen2`)
├── run.py            # CLI: loader → router → model.load → generate
├── compare_logits.py # parity gate vs transformers
└── qwen2/                       # one self-contained family package
    ├── __init__.py              # manifest: MODEL_TYPES · DEFAULTS · register(load)
    ├── config.py                # Qwen2Config — parse config.json (hf) / metadata (gguf)
    ├── blocks.py                # small components: RMSNorm · RoPE · attention · MLP
    ├── modeling_qwen2.py        # architecture: DecoderLayer + Qwen2Model + forward
    └── weights.py               # the weight-name seam (maps) + load (checkpoint → model)
```

A family package imports only its own siblings (`config`, `blocks`) + the shared
registry — never another family. Adding a model = drop a `<family>/` package + add
`import <family>` to `models.py`.

See **[CONVENTIONS.md](./CONVENTIONS.md)** for the family pattern we follow (file
roles, the self-containment rule, and the numbered-step `forward` narration) so
every architecture reads the same way. `qwen2/` is the reference to copy.

## The flow

```
spec ─► loader (raw config + weights by file names + tokenizer)
     ─► router (model_type → qwen2 family)
     ─► qwen2.load(raw_config, weights, fmt)   ← family builds the model + maps its own weight names
     ─► shared generate loop (calls model.forward(ids, past)) ─► text
```

## Design (per the vision)

- **Loader is dumb.** It never renames tensors — weights come keyed by the file's
  own names + a `fmt` tag. The **family owns the name map** (`qwen2/weights.py`), which
  is where format quirks live.
- **Router only routes.** `model_type` → family; unknown → raises.
- **The loop is shared.** Families provide `forward(ids, past) → (logits, past)`,
  never their own loop.
- **The family is self-contained.** `modeling_qwen2.py` has its *own* RoPE / RMSNorm
  / attention / MLP — it imports only its siblings (`config`, `blocks`) and the
  shared registry, never another family. Duplication across families is intentional
  (study clarity).

## Status

✅ **qwen2** · ✅ **qwen3** (+ QK-norm, no bias) — both from safetensors **and** GGUF.
✅ **gemma2** from safetensors; 🚧 **gemma2 GGUF** *hard-fails by design* — its
Gemma2-specific metadata keys (attn scale, soft-caps, sliding window) aren't yet
validated against llama.cpp, so per the vision we raise rather than guess (see
`gemma2/config.py::from_gguf`).
Shared loader/router/streaming-loop · GGUF parse + dequant (Q4_K/Q5_K/Q6_K/Q5_0/…) ·
uniform tokenizer (BPE + SentencePiece) · streaming weight load (peak ≈ steady) · parity gate.
⬜ next families (`llama/`, `gemma3/`, `mixtral/`, qwen3-MoE). ⬜ GGUF MoE (stacked experts).

> **gemma2 parity:** Gemma2 uses attention logit soft-capping, which standard SDPA
> skips. To compare, load the reference with eager attention:
> `AutoModelForCausalLM.from_pretrained(..., attn_implementation="eager")`.

## Verified

Syntax across all modules; registry/router dispatch (qwen2 registers, unknown
raises); `weights.to_raw` maps (hf identity + gguf); sampling precedence. **Not** yet
run end-to-end — that's `compare_logits.py` on a real model in your venv (the real
gate).
