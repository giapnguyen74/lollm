# inference — study-first engine (v1: qwen2)

First build of `docs/inference-engine-vision.md`. A thin **loader + router**, a
shared **generate loop**, and self-contained **family packages**. v1 ships one
family — **qwen2** — from HF safetensors.

```bash
# safetensors (HF repo / local dir)
python src/run.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Explain RoPE in one line."
python src/run.py --model ./local/qwen2/dir --prompt "Hi" --temperature 0   # greedy

# GGUF (local .gguf or repo:QUANT — downloaded + dequantized)
python src/run.py --model Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M --prompt "Hi"
python src/run.py --model ./qwen2.5-0.5b-instruct-q4_k_m.gguf   --prompt "Hi"

python src/compare_logits.py --model Qwen/Qwen2.5-0.5B-Instruct             # parity gate (safetensors)
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

✅ **qwen2** · ✅ **qwen3** (+ QK-norm, no bias) · ✅ **gemma2** — all from safetensors
**and** GGUF · shared loader/router/streaming-loop · GGUF parse + dequant
(Q4_K/Q5_K/Q6_K/Q5_0/…) · uniform tokenizer (BPE + SentencePiece) · streaming weight
load (peak ≈ steady) · parity gate.
⬜ next families (`llama/`, `gemma3/`, `mixtral/`, qwen3-MoE). ⬜ GGUF MoE (stacked experts).

> **gemma2 parity:** Gemma2 uses attention logit soft-capping, which standard SDPA
> skips. To compare, load the reference with eager attention:
> `AutoModelForCausalLM.from_pretrained(..., attn_implementation="eager")`.

## Verified

Syntax across all modules; registry/router dispatch (qwen2 registers, unknown
raises); `weights.to_raw` maps (hf identity + gguf); sampling precedence. **Not** yet
run end-to-end — that's `compare_logits.py` on a real model in your venv (the real
gate).
