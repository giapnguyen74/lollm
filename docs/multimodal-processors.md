# Multimodal processors — turning image / audio / text into model inputs

> Background + design notes for when lollm grows past text (Gemma4 vision/audio,
> Qwen-VL, etc.). The model **weights** are only half the story for a multimodal
> model; the other half is the **processor** that turns raw files into the tensors the
> weights expect. This doc captures how that works, what's standardized and what
> isn't, and how it would map onto our existing seams. Nothing here is implemented yet.

---

## The core idea: weights never see a file format

A multimodal checkpoint contains encoder tensors for each modality (see
`gemma4-architecture.md`: `vision_tower.*`, `audio_tower.*`, the text `embed_tokens`).
But **no weight ever touches a `.jpg` or a `.mp3`.** Every input is first converted to
a *fixed numeric tensor* by a preprocessing step that runs **outside** the model. All
the variety of input formats lives in that step, not in the weights.

There are really three normalizers — one per modality — each producing something the
model can embed:

| Modality | Raw input | Normalizer | Canonical tensor the model sees |
|---|---|---|---|
| Text  | unicode string (any encoding) | **tokenizer** | integer token IDs → embedding lookup |
| Image | jpg/png/webp/… | **image processor** | normalized pixel patches `(N_patches, C·P·P)` or `(C,H,W)` |
| Audio | wav/mp3/flac/… | **feature extractor** | log-mel spectrogram frames `(n_mels, T)` |

Text already works this way in lollm: the tokenizer maps UTF-8 bytes → IDs from the
fixed vocab. Image/audio just need their own equivalents. (Text has "formats" too —
UTF-8 vs UTF-16, NFC/NFD normalization — and the tokenizer is exactly what flattens
them. The image/audio processors are the same idea, one level more involved.)

---

## What's standardized — and what isn't (three layers)

**1. Decode — fully standard.** Container/codec formats are standardized and have
universal decoders: JPEG/PNG/WebP → raw RGB via libjpeg/PIL; WAV/FLAC/MP3/Opus → PCM
via ffmpeg. Getting to "raw pixels" or "raw waveform" is a solved, shared step.

**2. Raw intermediate — de facto conventions.** After decode there's a common lingua
franca, widely shared but not guaranteed:
- images: an RGB `uint8 H×W×3` array;
- speech: **mono PCM at 16 kHz** (USM, Whisper, wav2vec2 all converged on 16 kHz);
- the **ViT patch** idea (split into P×P patches, linear-embed) is a shared *pattern*;
- the **log-mel spectrogram** is the near-universal audio feature representation.

**3. Model-specific normalization — NOT standard.** This is where it fragments. Each
model defines its own exact recipe: patch size, input resolution(s), pixel mean/std,
aspect-ratio handling, how many tokens an image becomes; for audio, the mel parameters
(`n_mels`, window, hop). Gemma4's normalization, resolutions and token budgets are
*its* spec; Qwen-VL, LLaVA, Whisper each differ. **There is no data format where one
model's image tensor would feed another model's tower.**

> So: "the checkpoint has tensors for all three modalities" and "inputs come in many
> formats" aren't in tension — the format handling is in the processor, which is
> *separate from the weights* and *specific to the model*.

---

## The unifying thing is an *interface*, not a format

Since there's no universal input tensor, what lets one codebase drive many models is
two interface-level conventions:

- **The per-model processor + its config.** In `transformers`, every model ships a
  `preprocessor_config.json` and an `AutoProcessor` / `AutoImageProcessor` /
  `FeatureExtractor` reads it. You call one uniform API
  (`processor(images=…, audio=…, text=…)`) and it dispatches to *that model's* recipe.
  It's a unified interface over per-model specs — exactly analogous to how `AutoConfig`
  reads each model's `config.json`.
- **The typed-message schema.** The OpenAI-style content list —
  `[{"type":"image","url":…}, {"type":"audio",…}, {"type":"text",…}]` — is the de
  facto interchange for *how you hand mixed media to a model*; chat templates adopted
  it too. It standardizes the envelope, not the tensors.

Neither is a tensor format. Both are "read the model's own spec and apply it."

---

## The soft-token pattern (how the modalities merge)

One *architectural* pattern has converged even though the numbers differ (Gemma,
Qwen-VL, LLaVA, …): each non-text input becomes a sequence of **soft-tokens** —
embedding vectors produced by the encoder — that **replace placeholder token IDs** in
the text stream.

```
text:   "describe <image> please"
ids:    [ ..., <boi>, <image>×N, <eoi>, ... ]      # N placeholder ids reserve slots
embeds: [ ..., e_boi,  V1..VN ,  e_eoi, ... ]      # encoder soft-tokens overwrite them
                         ▲ vision_tower(pixels) → project → N vectors
```

From there it's one shared decoder stream — text and media live in the same hidden
space. (Gemma4 detail: PLE is computed from the *IDs* before this overwrite, so
media positions get the pad-token's neutral per-layer signal — see
`gemma4-architecture.md`.)

---

## Concrete: Gemma 4 E2B

- **Image** — Gemma-3-lineage SigLIP-style processor: a few resolutions (256/512/768),
  variable aspect ratio, and a **configurable token budget** (70 / 140 / 280 / 560 /
  1120 tokens per image). The processor decides how many patches; the tower's learned
  2D positions + RoPE handle the variable grid. Output soft-tokens replace
  `image_token_id`, wrapped by `boi`/`eoi`. Convention: image **before** text.
- **Audio** — USM-style: decode → mono **16 kHz** → log-mel frames → conformer tower
  emits ~**1 token / 160 ms** (~6/sec), 30 s max clip. Soft-tokens replace
  `audio_token_id`, wrapped by `boa`/`eoa`. Convention: audio **after** text.

The placeholder/wrapper token IDs (`boi/eoi/image`, `boa/eoa/audio`, `video`) are in
the top-level `Gemma4Config`.

---

## What this means for lollm

The good news: it extends our existing seams rather than breaking them. The "family
owns its quirks" rule already covers config parsing and the weight-name map; a
multimodal family would also **own its processor spec**.

- **A processor seam, parallel to the config seam.** Just as `config.py` reads
  `config.json`, a `processor.py` would read the checkpoint's `preprocessor_config.json`
  (resolutions, mean/std, patch size, mel params, token budget) — no hardcoded global
  preprocessing path. Read each model's own spec; fail loud on unknown keys.
- **Weights:** load `vision_tower.*` / `audio_tower.*` + projectors (separable — Gemma's
  "conditional parameter loading" means they can be skipped for text-only).
- **Forward:** run the encoder(s), project to soft-tokens, and splice them into the
  embedding sequence at placeholder positions before the decoder stack — the only new
  step in the shared loop's vicinity (the decoder itself is unchanged).
- **Input plumbing:** accept the typed-message schema (`text`/`image`/`audio` parts) so
  the CLI/loop can carry mixed media.

**The PyTorch-only tension gets bigger.** Text already leans on
`transformers.AutoTokenizer` (the known wart). Multimodal adds a dependency surface we'd
otherwise have to reimplement:
- image: decode (libjpeg/PIL), resize, normalize, patchify;
- audio: decode + **resample** to 16 kHz (ffmpeg/`torchaudio`), and the **log-mel**
  transform (`torch.stft` + a mel filterbank — doable in pure torch, unlike resampling).

A reasonable stance for a study engine: implement the *model-specific* normalization
(patchify, mel, token budget) ourselves in torch — that's the instructive part — and
treat the generic decode/resample (libjpeg, ffmpeg) as allowed external tools, the same
way `huggingface_hub` only *downloads* files. Worth deciding explicitly when we get
there.

### Open questions for later

- How much of the processor to reimplement vs. borrow (mirrors the tokenizer wart).
- Whether to validate the processor against `transformers`' `AutoProcessor` the way
  `compare_logits.py` validates the model (a "processor parity" check: same pixel/mel
  tensor for the same input).
- Variable image token budget + variable aspect ratio make output shapes dynamic —
  the placeholder-count must match the produced soft-token count exactly, or the merge
  mis-aligns (a good hard-fail assertion).

---

See also: `gemma4-architecture.md` (the towers + soft-token merge), `qwen3_5-architecture.md`
(another multimodal checkpoint we use text-only), and the tokenizer notes in `README.md`.
