# The Raw (HuggingFace) Model Format → GGUF

> Companion to `gguf-format.md`. This doc covers the "raw" model — the
> HuggingFace repo layout that GGUF is converted *from* — and then maps it,
> field by field, into GGUF.

A raw model is a **directory of separate files**. GGUF's whole job is to fold
that directory into one self-describing binary (and usually quantize it on the
way). Understanding the source format makes the conversion obvious.

---

## 1. What's in a raw model repo

Four logical pieces:

**Tokenizer**
- `tokenizer.json` — modern "fast" tokenizer: vocab + merges + normalization + pre-tokenization, all in one JSON.
- `tokenizer_config.json` — tokenizer class, special-token config, and the **chat template** (Jinja).
- `special_tokens_map.json` — BOS / EOS / PAD / UNK.
- Sometimes `tokenizer.model` (raw SentencePiece) or the older `vocab.json` + `merges.txt` pair.

**Architecture definition**
- `config.json` — **data, not code**: `architectures`, `model_type`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `rope_theta`, `rms_norm_eps`, `vocab_size`, `max_position_embeddings`, …
- The actual model **code** usually lives in the `transformers` library (`modeling_llama.py`), selected by name. Only `trust_remote_code` models ship their own `modeling_*.py` in the repo via `auto_map`.

**Weights**
- `model.safetensors`, or sharded `model-0000N-of-0000M.safetensors` + an index `model.safetensors.index.json` mapping tensor → shard.

**Generation defaults**
- `generation_config.json` — default temperature, top_p, eos handling.

---

## 2. config.json parameterizes; it does not define

The key mental model: `config.json` does **not** define the architecture's
computation. The `architectures` / `model_type` field is a *name the runtime
looks up* in its registry of already-implemented architectures. The config just
fills in the numbers.

This splits all models into two regimes:

- **Known architecture** → `config.json` alone is enough. `"model_type": "llama"` tells transformers to use `modeling_llama.py`, tells llama.cpp to use its `LLM_ARCH_LLAMA` graph, tells mlx-lm to use `models/llama.py`. No new code — you're instantiating an existing implementation with different hyperparameters.
- **Custom / novel architecture** → you need the actual code, and where it comes from depends on the runtime:

| Runtime | How it gets a custom arch | Implication |
|---|---|---|
| HF transformers | `trust_remote_code=True` + `auto_map` → loads the provider's `modeling_*.py` | Works day 0, but runs the provider's **Python** |
| llama.cpp / GGUF | Someone implements the graph in **C++** + a conversion script | New arch **can't run until llama.cpp adds support** |
| mlx-lm / vLLM / SGLang | Architecture registered in their Python codebase | Same — needs an upstream implementation |

This is why a hot new model runs in `transformers` the day it drops but takes
days/weeks to appear in llama.cpp / Ollama / MLX: someone has to hand-port the
forward pass into the executor's own language. **GGUF deliberately cannot carry
code** — it's pure data, which is a security/portability win but means the
executor must already know the architecture. (The spec lists an embedded
computation graph as a *future* extension to escape this.)

---

## 3. safetensors byte layout

Deliberately simple, and `mmap`-able — the same philosophy as GGUF's tensor
region:

```
┌────────────┬──────────────────────────────┬───────────────────────┐
│ 8 bytes    │ JSON header (N bytes)        │ raw tensor bytes       │
│ uint64 N   │ name → { dtype, shape,       │ (contiguous blob)      │
│ = header   │          data_offsets:[a,b] }│                        │
│   length   │ + optional __metadata__      │                        │
└────────────┴──────────────────────────────┴───────────────────────┘
```

The header is a JSON dict mapping each tensor name to its dtype, shape, and
`[start, end]` byte offsets into the blob. No pickle, no executable content
(the security win over `.bin`). Weights are typically `bf16` or `f16`.

---

## 4. How a raw model maps into GGUF

Conversion (e.g. llama.cpp's `convert_hf_to_gguf.py`) reads the directory and
writes one GGUF. Three things get mapped: **config → metadata**, **tokenizer →
metadata**, **safetensors → tensor data**.

### 4a. config.json → GGUF metadata

| `config.json` field | GGUF metadata key |
|---|---|
| `architectures` / `model_type` | `general.architecture` |
| `hidden_size` | `[arch].embedding_length` |
| `num_hidden_layers` | `[arch].block_count` |
| `intermediate_size` | `[arch].feed_forward_length` |
| `num_attention_heads` | `[arch].attention.head_count` |
| `num_key_value_heads` | `[arch].attention.head_count_kv` (GQA) |
| `max_position_embeddings` | `[arch].context_length` |
| `rms_norm_eps` | `[arch].attention.layer_norm_rms_epsilon` |
| `rope_theta` | `[arch].rope.freq_base` |
| `rope_scaling.*` | `[arch].rope.scaling.*` |
| `vocab_size` | (implied by length of `tokenizer.ggml.tokens`) |
| `num_local_experts` / `num_experts_per_tok` | `[arch].expert_count` / `expert_used_count` |

(`[arch]` is `llama`, `qwen2`, etc. — same prefix as `general.architecture`.)

### 4b. Tokenizer files → GGUF metadata

| Source | GGUF metadata key |
|---|---|
| `tokenizer.json` vocab | `tokenizer.ggml.tokens` (array[string]) |
| BPE merges | `tokenizer.ggml.merges` (array[string]) |
| SentencePiece scores | `tokenizer.ggml.scores` (array[float32]) |
| token types | `tokenizer.ggml.token_type` (array[int32]) |
| tokenizer class | `tokenizer.ggml.model` (`llama` / `gpt2` / …) |
| `special_tokens_map.json` | `tokenizer.ggml.bos_token_id`, `eos_token_id`, … |
| `tokenizer_config.json` chat template | `tokenizer.chat_template` |

The entire vocabulary is embedded as a giant string array — this is why the
converted file needs no external tokenizer.

### 4c. safetensors tensors → GGUF tensor data

Each weight is **renamed** from the HF convention to the GGML convention, then
written into the aligned tensor-data blob:

| HF tensor name | GGML / GGUF tensor name |
|---|---|
| `model.embed_tokens.weight` | `token_embd.weight` |
| `model.norm.weight` | `output_norm.weight` |
| `lm_head.weight` | `output.weight` (or tied to `token_embd`) |
| `model.layers.N.input_layernorm.weight` | `blk.N.attn_norm.weight` |
| `model.layers.N.self_attn.q_proj.weight` | `blk.N.attn_q.weight` |
| `model.layers.N.self_attn.k_proj.weight` | `blk.N.attn_k.weight` |
| `model.layers.N.self_attn.v_proj.weight` | `blk.N.attn_v.weight` |
| `model.layers.N.self_attn.o_proj.weight` | `blk.N.attn_output.weight` |
| `model.layers.N.post_attention_layernorm.weight` | `blk.N.ffn_norm.weight` |
| `model.layers.N.mlp.gate_proj.weight` | `blk.N.ffn_gate.weight` |
| `model.layers.N.mlp.up_proj.weight` | `blk.N.ffn_up.weight` |
| `model.layers.N.mlp.down_proj.weight` | `blk.N.ffn_down.weight` |

**Gotcha — weight permutation.** For Llama-style models, HF stores `q_proj` /
`k_proj` in a layout that doesn't match llama.cpp's RoPE implementation, so the
converter **permutes** those tensors during conversion (reshape → swap axes →
reshape). This is exactly the kind of rearrangement the spec's
`[llm].tensor_data_layout` key exists to record. Values aren't just copied
verbatim; some are transposed, permuted, or merged (e.g. fused QKV split apart).

### 4d. Optional quantization

The converter can write weights as-is (`F16` / `BF16`), or quantize them. In
practice you often do it in two steps: convert to a `F16` GGUF first, then run
`llama-quantize` to produce `Q4_K_M`, `Q5_K_M`, etc. Quantization rewrites each
tensor's bytes into block format (see `gguf-format.md` §6) and sets
`general.file_type` + `general.quantization_version`.

### 4e. The full pipeline

```
raw model directory
├── config.json ───────────► general.* + [arch].* metadata
├── tokenizer.json ────────► tokenizer.ggml.* metadata
├── special_tokens_map.json ┘
├── *.safetensors ─────────► rename + permute → aligned tensor-data blob
└── (optional) quantize fp16 → Q4_K/Q5_K/... blocks
                              │
                              ▼
                    single self-describing .gguf
```

---

## 5. Side-by-side summary

| Concern | HuggingFace (raw) | GGUF (converted) |
|---|---|---|
| Packaging | directory of files | one binary file |
| Tokenizer | `tokenizer.json` + configs | `tokenizer.ggml.*` metadata |
| Hyperparameters | `config.json` | `general.*` + `[arch].*` metadata |
| Architecture code | transformers / remote `.py` | must be built into the executor |
| Weights | `*.safetensors` (bf16/fp16) | aligned tensor blob (often quantized) |
| Tensor names | `model.layers.N.self_attn.q_proj` | `blk.N.attn_q` |
| Carries executable code? | yes (remote code) | **never** (data only) |
| Loading | framework + Python | `mmap`, any language |

The one-line takeaway: **converting to GGUF = read the separate HF files, fold
config + tokenizer into metadata, rename/permute the safetensors weights into an
aligned blob, and optionally quantize.** Nothing about the math changes — only
how the model is packaged and (optionally) the numeric precision of the weights.

---

### Sources

- safetensors format — https://github.com/huggingface/safetensors
- GGUF spec (metadata keys, tensor naming) — https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- Tensor renaming / permutation reflects llama.cpp's `convert_hf_to_gguf.py` conventions.
