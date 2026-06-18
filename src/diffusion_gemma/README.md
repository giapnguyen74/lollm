# diffusion_gemma — text-diffusion family (TEMPORARY / work-in-progress)

> **Status: experimental, not yet registered.** This family is being brought up rung by rung and
> is **not wired into the engine** (`models.py` / `run.py`) yet — run it via `compare_logits.py`
> below, not `python src/run.py`. Architecture write-up:
> **[docs/diffusion_gemma-architecture.md](../../docs/diffusion_gemma-architecture.md)**.
> This README is a placeholder until the family is finished and promoted.

A study implementation of **`google/diffusiongemma-26B-A4B-it`** — Google DeepMind's open
**text-diffusion** model (released 2026-06-10). It keeps the **Gemma 4 26B-A4B MoE backbone** but
replaces autoregressive decoding with **discrete diffusion**: denoise a fixed **canvas** of tokens
in parallel, commit the confident ones, re-randomise the rest, and chain canvases
block-autoregressively.

## The shape in one breath

One Gemma-lineage backbone, run in two roles (weights tied):

- **Encoder** = causal prefill — encodes the prompt (and each committed block) into the **KV cache**.
- **Decoder** = the denoiser — **bidirectional** attention over the canvas + cross-attention to the
  read-only encoder cache, with a **self-conditioning** step (feeds the previous step's prediction back).

Per-layer FFN is **dense MLP + routed MoE in parallel** (8 of 128 experts + the always-on dense MLP).
Generation = inner **denoise loop** (entropy-bound accept / renoise + temperature schedule + adaptive
stop) wrapped in an outer **block-AR loop** (commit → re-encode → next canvas).

## Files

```
config.py                   real config.json (text_config) → flat dataclass
blocks.py                   RMSNorm · dual+proportional RoPE · attention (encoder causal /
                            decoder bidirectional, global k_eq_v) · dense MLP · MoE router/experts
modeling_diffusion_gemma.py EncoderTextModel (causal, incremental cache) · DecoderTextModel
                            (bidirectional + self-conditioning + tied lm-head/softcap)
sampler.py                  EntropyBoundSampler · linear temperature · adaptive stop · denoise_block
generate.py                 generate_diffusion — the block-AR outer loop
compare_logits.py           parity gate vs the real transformers checkpoint (see below)
```

## Verification status (vs `transformers`, tiny random-weight module-parity)

| rung | what | result |
|---|---|---|
| 1 | encoder backbone (causal forward) | ✅ cosine ≈ 1 (1.8e-6) |
| 2 | decoder denoise pass (bidirectional + cross-attn + self-cond) | ✅ cosine 1.0 (~5e-6) |
| 3 | one denoise step (logits, temp, entropy-bound accept, renoise) | ✅ exact / ~1e-6 |
| 4 | inner loop + adaptive stop | ✅ block bit-exact |
| 5 | outer block-AR loop (commit → incremental re-encode → next canvas) | ✅ per-step logits cosine 1.0 |
| 6 | peripherals (below) | 🔴 open |

The rung-by-rung throwaway parity harnesses live in the workshop copy (`debug/diffusion_gemma/`),
not here. Here, the real check is `compare_logits.py`.

## Run the parity gate (real checkpoint)

```bash
python src/diffusion_gemma/compare_logits.py --model google/diffusiongemma-26B-A4B-it \
    --prompt "Why is the sky blue?"
python src/diffusion_gemma/compare_logits.py --generate        # + a real generation
```

It loads the real model, builds our encoder/decoder from the real config, **shares** the reference
weights (no extra memory), and compares **denoiser logits** on a fixed canvas (per-position argmax
agreement + cosine). `--generate` runs `generate_diffusion` and prints text. **PASS** =
`cosine > 0.999 and argmax agreement > 0.99` (looser because the real run is bf16; `--dtype float32`
tightens it). ~26B params — see the header of `compare_logits.py` for memory/dtype notes.

## Known limitations (rung 6)

- **Long-context sliding-window cache** — sliding layers must clip the KV cache to `sliding_window`
  (1024) once context exceeds it; we currently keep the full cache, so generations **longer than the
  window** diverge from the reference (within-window is verified). The one real correctness gap. See
  `LESSONS.md` L-8 and `debug/diffusion_gemma/PLANNING.md`.
- **Vision tower** not built (text-only) — the `gemma4_vision` SigLIP encoder + soft-token merge.
- **GGUF** unsupported (hard-fail) until the `diffusion_gemma` metadata is verified vs llama.cpp.
- **Not integrated** — no `run.py` / `models.py` wiring; the shared AR generate loop can't drive
  diffusion, so it needs `generate_diffusion` selected explicitly (a deferred, explicit step).

## Config sources (two files)

DiffusionGemma splits config the standard HF way:

- **`config.json`** — architecture. `config.py::from_hf` reads its `text_config`; `canvas_length`
  (the block size) is top-level and read separately.
- **`generation_config.json`** — the diffusion **sampler** knobs: `max_denoising_steps`,
  `confidence_threshold`, `stability_threshold`, `t_min`/`t_max`, `sampler_config.entropy_bound`,
  `eos_token_id`. These are generation policy (like `temperature`/`top_p` for AR models), **not** in
  `config.json`.

> **Current behaviour:** we **reuse these generation params from the loaded reference**
> (`ref.generation_config` in `compare_logits.py`) rather than parsing `generation_config.json`
> ourselves. That's fine for the parity gate. The standalone run path (no `transformers`) will need a
> small `generation_config.json` parser feeding the loader's `gen_meta` — **deferred to integration**.
