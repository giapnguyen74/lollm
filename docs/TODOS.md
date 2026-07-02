# TODOs — near-term, actionable

Things we intend to fix **soon** (concrete, scoped). Longer-term / untimed ideas live in
[ROADMAP.md](./ROADMAP.md); resolved gotchas become lessons in
[LESSONS.md](./LESSONS.md).

**IDs are stable** — `T-#` never renumbers or gets reused. When an item is done, flip its
status to ✅ (leave it in place for one cycle, then move the lesson to LESSONS.md and
delete the row). Status: 🔴 open · 🟡 in progress · ✅ done · 📌 acknowledged (accepted, no fix planned).

**Priority:** `P0` do first (unblocks a most-wanted capability) · `P1` next · `P2` larger / later.

| ID  | pri | status | area    | title |
|-----|-----|--------|---------|-------|
| T-1 | —  | 📌 acknowledged | mps / dtype | fp16-on-MPS: fp16 default kept (bf16 rejected, L-5); `--dtype` override **not shipped** |
| T-2 | P1 | 🔴 open | parity      | Add a >sliding_window prompt to the parity gate (window is untested) |
| T-3 | **P0** | 🔴 open | gguf | Lift the Gemma GGUF hard-fail — validate metadata vs llama.cpp (gemma2/3/4) |
| T-4 | P1 | 🔴 open | gguf | GGUF MoE — stacked/fused expert tensors (qwen3_5_moe, diffusion_gemma) |
| T-5 | P2 | 🔴 open | gguf / perf | Stream GGUF dequant onto device (kill the eager full-dequant peak) |
| T-6 | P1 | 🔴 open | multimodal | Gemma 4 processor seam — `processor.py` (image patchify + audio log-mel) |
| T-7 | P1 | 🔴 open | multimodal | Gemma 4 vision tower + projector + soft-token merge |
| T-8 | P2 | 🔴 open | multimodal | Gemma 4 audio (USM) tower + projector + soft-token merge |

> Sequencing: **T-3 → T-4 → T-5** is the GGUF (smaller-models) track; **T-6 → T-7 → T-8**
> is the multimodal track. The two are independent and can run in parallel. GGUF is
> ranked first overall: most reduce-size / quantized checkpoints ship as GGUF, and T-3 is
> the one blocker stopping quantized Gemma from running at all today.

---

## T-1 · fp16-on-MPS dtype 📌

Real inference runs **fp16 on MPS**, but the parity gate runs **fp32 on CPU**, so a
fp16-only overflow would be invisible to the gate. Gemma has the largest activations
(×√hidden embedding, wide GeGLU).

**Decision (made):** keep **fp16 as the MPS default** — in practice it's near-identical
to fp32 because the overflow-prone reductions (attention softmax, every RMSNorm) already
upcast to fp32. bf16-on-MPS was tried and rejected (~3× slower, memory-heavy — see L-5).


## T-2 · Parity gate must exercise the sliding window

`compare_logits` uses a ~6-token prompt; Gemma's window is 512, so local layers behave
like full causal and the band mask is **never tested** by the gate. A correct and a
broken sliding implementation pass identically.

**Do:** add a long-prompt (>512 token) case to `compare_logits` / `sanity_test` for
gemma3 + gemma4, and ideally a prefill==incremental-decode assertion in the gate itself.
**Where:** `src/compare_logits.py`, `src/sanity_test.py`.


# ── GGUF track (smaller / quantized models) — promotes ROADMAP R-3, R-5 ──

## T-3 · Lift the Gemma GGUF hard-fail — P0  🔴

**Why P0:** most reduce-size checkpoints ship as GGUF, but every gemma family currently
**hard-fails** in `config.from_gguf` (`gemma2/config.py`, `gemma3/config.py`,
`gemma4/config.py`) because the arch-specific metadata keys aren't validated against
llama.cpp yet. So quantized Gemma — the most useful small model to run on MPS — can't
load at all. qwen2/qwen3 GGUF already works; this brings Gemma up to parity.

**Do:** for each gemma family, map the GGUF metadata keys → `<Family>Config` and replace
the `NotImplementedError` with real parsing, one key at a time, each confirmed against a
known checkpoint written by llama.cpp:
- norm `+1` fold — already de-baked in `gemma2/weights.py` (`name.endswith("norm.weight")
  → t - 1.0`); currently dead code behind the config hard-fail. Confirm it fires once
  config parses, and extend to gemma3/4 (which also carry q/k_norm — scope the `endswith`
  so it doesn't catch those).
- attn scale (`query_pre_attn_scalar`), sliding window + the local/global pattern,
- gemma2 dual soft-caps (attn-logit + final-logit), gemma3/4 QK-norm,
- dual / proportional RoPE thetas (local + global), and gemma4 PLE keys.

**Gate:** `compare_logits` a `Q4_K`/`Q6_K` gemma GGUF against the same model's safetensors
(or `transformers`) — same top token + cosine ≈ 1. **Where:** the three `*/config.py`
`from_gguf`, `gemma2/weights.py` (de-bake), `gemma{3,4}/weights.py`.

## T-4 · GGUF MoE — stacked / fused expert tensors — P1  🔴

GGUF packs MoE experts as fused tensors (e.g. `blk.N.ffn_gate_exps.weight`, shape
`[E, …]`) rather than per-expert modules. Map them onto the fused `Experts` param layout
already used by `qwen3_5_moe` (`gate_up_proj`/`down_proj` as `(E, …)` parameters) — and
the same layout `diffusion_gemma` uses. The strict per-tensor shape check is the
validator (see `qwen3_5_moe/weights.py` notes). **Gate:** parity on a GGUF MoE checkpoint.
**Where:** `gguf_reader.py` (if any key reshaping is needed), `qwen3_5_moe/weights.py`,
`dequant.py` (confirm the K-quant blocks dequant per-expert correctly).

## T-5 · Stream GGUF dequant onto device — P2  🔴

`loader._load_gguf` **eagerly** dequantizes every tensor into a CPU dict up front, so the
"streaming peak ≈ steady" property (which holds for the safetensors path) does **not**
hold for GGUF — peak RAM spikes to the full fp16 model during load. Make GGUF dequant
lazy / per-tensor so the family loader can dequant→`.to(device)`→free one tensor at a
time, matching the safetensors streaming loader. Watch the L-10 / streaming-loader
refcount footgun. **Where:** `loader._load_gguf`, `dequant.py`, the family `load`.


# ── Multimodal track (Gemma 4 vision + audio) — promotes ROADMAP R-1 ──

> Design + the processor/weights/forward seams are written up in
> [multimodal-processors.md](./multimodal-processors.md) and
> [gemma4-architecture.md](./gemma4-architecture.md). The gemma4 **text** decoder is done
> and parity-verified; these add the towers. `gemma4/weights.py` already rewrites
> `model.` → `model.language_model.` with a text-only fallback, so the tower tensors slot
> in beside the existing language model without disturbing it.

## T-6 · Gemma 4 processor seam — `processor.py` — P1  🔴

A processor seam **parallel to the config seam**: just as `config.py` reads `config.json`,
a `processor.py` reads the checkpoint's `preprocessor_config.json` (resolutions, mean/std,
patch size, mel params, token budget) — no hardcoded global preprocessing path, fail loud
on unknown keys. Implement the **model-specific** normalization ourselves in torch (the
instructive part): image patchify; audio log-mel (`torch.stft` + mel filterbank). Treat
generic decode/resample (libjpeg/PIL, ffmpeg → mono 16 kHz) as allowed external tools, the
same stance as `huggingface_hub` only *downloading*. **Gate:** a "processor parity" check —
same pixel/mel tensor as `transformers` `AutoProcessor` for the same input (analogous to
`compare_logits` for weights). **Where:** new `src/gemma4/processor.py`.

## T-7 · Gemma 4 vision tower + soft-token merge — P1  🔴

Load `vision_tower.*` + the projector (conditional / skippable for text-only — Gemma's
conditional parameter loading), run the SigLIP-style tower over patches, project to
**soft-tokens**, and splice them into the embedding sequence at `image_token_id`
placeholders (wrapped by `boi`/`eoi`). Mind the gemma4 detail that **PLE is computed from
the IDs before the overwrite**, so media slots get the pad token's neutral per-layer
signal. The decoder stack itself is unchanged. Add the hard-fail assertion that produced
soft-token count == reserved placeholder count (variable token budget → dynamic shapes).
**Gate:** parity vs `transformers` on an image prompt. **Where:** `gemma4/` (new
`vision.py` or fold into `blocks.py`), `weights.py`, the merge step near the shared loop.

## T-8 · Gemma 4 audio (USM) tower + soft-token merge — P2  🔴

Same pattern as T-7 for audio: USM-style conformer tower over log-mel frames (~1 token /
160 ms, 30 s max clip), project to soft-tokens, splice at `audio_token_id` (wrapped by
`boa`/`eoa`; convention is audio **after** text). Larger tower (~300M) and lower-traffic
than vision, hence P2. Also needs the typed-message input schema (`text`/`image`/`audio`
parts) plumbed through `run.py` / the loop so the CLI can carry mixed media — shared with
T-7. **Gate:** parity vs `transformers` on an audio prompt. **Where:** `gemma4/` (new
`audio.py`), `weights.py`, `run.py` input plumbing.
