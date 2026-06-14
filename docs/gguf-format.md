# GGUF File Format — A Technical Study

> Goal of this doc: understand exactly what bytes are in a `.gguf` file and how a
> loader turns them into something you can run on a GPU. This is the prerequisite
> for writing your own loader / inference loop.

GGUF ("GGML Universal File") is the single-file model format used by `llama.cpp`
and other GGML-based executors. It is the successor to GGML/GGMF/GGJT. The whole
design is oriented around one thing: **load a model fast, with no external files
and no extra dependencies, ideally via `mmap`.**

Five design goals, straight from the spec:

- **Single-file deployment** — weights + tokenizer + all hyperparameters in one file.
- **Extensible** — new metadata can be added without breaking old readers (this is the key win over GGJT).
- **`mmap`-friendly** — tensor data is aligned so it can be memory-mapped and paged in on demand.
- **Easy to parse** — readable in a few hundred lines in any language, no libraries needed.
- **Self-describing** — the file alone tells you the architecture, shapes, quant types, and tokenizer.

---

## 1. The big picture: four regions

A GGUF file is laid out sequentially as four regions:

```
┌──────────────────────────────────────────────┐
│ 1. HEADER                                      │  magic, version, counts
│    + metadata key-value pairs                  │  hyperparameters, tokenizer, etc.
├──────────────────────────────────────────────┤
│ 2. TENSOR INFO TABLE                           │  name, shape, type, offset per tensor
│    (one entry per tensor)                      │
├──────────────────────────────────────────────┤
│ 3. PADDING                                     │  0x00 bytes to reach ALIGNMENT
├──────────────────────────────────────────────┤
│ 4. TENSOR DATA                                 │  the actual (quantized) weights
│    (aligned, mmap-able blob)                   │
└──────────────────────────────────────────────┘
```

Regions 1–2 are the "metadata" you parse up front. Region 4 is the bulk of the
file (gigabytes) and is the part you `mmap` and hand to the GPU. Region 3 exists
purely so that region 4 starts on an aligned boundary.

Everything is **little-endian by default**. Version 3 of the format added
big-endian support, but if nothing says otherwise, assume little-endian.

---

## 2. Primitive data types

Two enums drive everything. First, the metadata value types (how a single value
is encoded):

| Enum value | Type | Bytes |
|---|---|---|
| 0 | `UINT8` | 1 |
| 1 | `INT8` | 1 |
| 2 | `UINT16` | 2 |
| 3 | `INT16` | 2 |
| 4 | `UINT32` | 4 |
| 5 | `INT32` | 4 |
| 6 | `FLOAT32` | 4 |
| 7 | `BOOL` | 1 (0=false, 1=true) |
| 8 | `STRING` | 8-byte length + UTF-8 bytes |
| 9 | `ARRAY` | type + length + elements |
| 10 | `UINT64` | 8 |
| 11 | `INT64` | 8 |
| 12 | `FLOAT64` | 8 |

A **GGUF string** is never null-terminated. It is a `uint64` length followed by
exactly that many UTF-8 bytes:

```c
struct gguf_string_t {
    uint64_t len;        // length in bytes
    char     string[len];// UTF-8, NOT null-terminated
};
```

An **array** is a type byte + a `uint64` count + the elements packed back to
back. Arrays can nest, and the length counts elements, not bytes. This is how
the token vocabulary is stored (`array[string]` of ~32k–128k entries).

Second, the tensor element types (`ggml_type`), which tell you how the raw
weight bytes are encoded. The important ones:

```
F32=0  F16=1  Q4_0=2  Q4_1=3  Q5_0=6  Q5_1=7  Q8_0=8  Q8_1=9
Q2_K=10  Q3_K=11  Q4_K=12  Q5_K=13  Q6_K=14  Q8_K=15
IQ2_XXS=16 ... IQ4_XS=23   (the "IQ" importance-matrix quants)
I8=24 I16=25 I32=26 I64=27  F64=28  BF16=30  MXFP4=39
```

(Several early types — Q4_2, Q4_3, and the `Q4_0_4_4` repacking variants — have
been removed and you'll never see them in current files.)

---

## 3. The header

```c
struct gguf_header_t {
    uint32_t magic;             // 'G','G','U','F' = 0x47 0x47 0x55 0x46
    uint32_t version;           // 3 for the current spec
    uint64_t tensor_count;      // number of tensors
    uint64_t metadata_kv_count; // number of metadata key-value pairs
    gguf_metadata_kv_t metadata_kv[metadata_kv_count];
};
```

Parsing order matters: you read the 4-byte magic, check it equals `GGUF`, read
the version, then the two counts. `tensor_count` is stored explicitly (not in
the metadata) precisely so a reader always knows how many tensor-info entries to
expect, regardless of what metadata is present.

Then you read `metadata_kv_count` key-value pairs. Each one is:

```c
struct gguf_metadata_kv_t {
    gguf_string_t key;               // hierarchical, lower_snake_case, dot-separated
    gguf_metadata_value_type value_type;   // one of the enum values above
    gguf_metadata_value_t value;     // interpreted per value_type
};
```

Keys are namespaced with dots, e.g. `llama.attention.head_count`. The `general.`
prefix is shared across all architectures; the architecture prefix (`llama.`,
`qwen2.`, `mamba.`, …) carries the model-specific hyperparameters.

### Metadata you actually need to run the model

These are the fields a loader reads to build the compute graph:

- **`general.architecture`** (string, required) — e.g. `llama`. Selects which model implementation to use.
- **`general.alignment`** (uint32) — the `ALIGNMENT` constant. Defaults to **32** if absent; must be a multiple of 8.
- **`general.quantization_version`** (uint32) — required if any tensor is quantized.
- `general.name`, `general.file_type` — descriptive / the dominant quant type.

Architecture hyperparameters (shown for `[llm]` = `llama`, etc.):

- `[llm].context_length` (`n_ctx`) — training context window.
- `[llm].embedding_length` (`n_embd`) — hidden size.
- `[llm].block_count` — number of transformer layers.
- `[llm].feed_forward_length` (`n_ff`) — MLP inner size.
- `[llm].attention.head_count` (`n_head`).
- `[llm].attention.head_count_kv` — KV heads; if less than `head_count`, the model uses **Grouped-Query Attention**.
- `[llm].attention.layer_norm_rms_epsilon` — RMSNorm epsilon.
- `[llm].rope.dimension_count`, `[llm].rope.freq_base`, `[llm].rope.scaling.*` — RoPE positional encoding params.
- MoE: `[llm].expert_count`, `[llm].expert_used_count`.
- SSM (Mamba): `[llm].ssm.conv_kernel`, `ssm.inner_size`, `ssm.state_size`, `ssm.time_step_rank`.

### The embedded tokenizer

The tokenizer lives entirely in metadata — this is why GGUF needs no external
files:

- `tokenizer.ggml.model` — `llama` (SentencePiece), `gpt2` (BPE), `rwkv`, etc.
- `tokenizer.ggml.tokens` — `array[string]`, the vocabulary indexed by token ID.
- `tokenizer.ggml.scores` — `array[float32]`, per-token scores (SentencePiece).
- `tokenizer.ggml.token_type` — `array[int32]` (1=normal, 2=unknown, 3=control, 4=user-defined, 5=unused, 6=byte).
- `tokenizer.ggml.merges` — `array[string]`, BPE merge rules.
- `tokenizer.ggml.bos_token_id` / `eos_token_id` / `unknown_token_id` / `padding_token_id`.
- `tokenizer.chat_template` — a Jinja template describing the prompt format.

---

## 4. The tensor info table

After the metadata comes one entry per tensor (`tensor_count` of them):

```c
struct gguf_tensor_info_t {
    gguf_string_t name;          // <= 64 bytes
    uint32_t n_dimensions;       // <= 4
    uint64_t dimensions[n_dimensions];
    ggml_type type;              // element encoding (F16, Q4_K, ...)
    uint64_t offset;             // offset INTO the tensor_data blob, multiple of ALIGNMENT
};
```

Key subtlety: `offset` is **relative to the start of the tensor-data region**,
not the start of the file. A reader computes the absolute file position as
`tensor_data_start + offset`. Each tensor's offset is a multiple of `ALIGNMENT`,
and tensors are padded to `ALIGNMENT` between each other — which is what makes
the whole data blob `mmap`-able and lets the GPU DMA aligned chunks.

### Standardized tensor names

GGML uses a fixed naming scheme so a generic loader can find each weight:

```
token_embd.weight          # input embedding table
output_norm.weight         # final norm
output.weight              # LM head (may be tied to token_embd)

blk.{N}.attn_norm.weight   # per-layer, N = 0 .. block_count-1
blk.{N}.attn_q.weight
blk.{N}.attn_k.weight
blk.{N}.attn_v.weight
blk.{N}.attn_output.weight
blk.{N}.ffn_norm.weight
blk.{N}.ffn_gate.weight    # SwiGLU gate
blk.{N}.ffn_up.weight
blk.{N}.ffn_down.weight
```

MoE adds `ffn_gate_inp` (router) and `ffn_{gate,up,down}_exp` per expert. Mamba
adds `ssm_in`, `ssm_conv1d`, `ssm_x`, `ssm_a`, `ssm_d`, `ssm_dt`, `ssm_out`.

---

## 5. Padding and tensor data

After the tensor-info table, the file is padded with `0x00` to the next multiple
of `ALIGNMENT`. That aligned position is `tensor_data_start`. Everything after it
is raw weight bytes, located via the `offset` fields above.

```c
uint64_t align_offset(uint64_t offset) {
    return offset + (ALIGNMENT - (offset % ALIGNMENT)) % ALIGNMENT;
}
```

A loader does **not** copy this region. It `mmap`s the file, and the OS pages in
weight data lazily as the forward pass touches it. On GPU, those pages are
uploaded to VRAM (this is the `n_gpu_layers` offload decision — how many layers'
worth of this blob to copy to the device).

---

## 6. Quantization formats — where the bytes go

This is the part that matters most for inference, and the part the spec leaves to
the implementation. Quantization is **block-based**: weights are grouped into
fixed-size blocks that share scale factors, so you store a few low-bit integers
plus a scale instead of full floats.

### Legacy "type-0/type-1" quants — block of 32 weights

| Type | Layout per 32-weight block | Bytes/block | Bits/weight |
|---|---|---|---|
| `Q4_0` | 1× f16 scale + 32× 4-bit | 18 | 4.5 |
| `Q4_1` | f16 scale + f16 min + 32× 4-bit | 20 | 5.0 |
| `Q5_0` | f16 scale + 32× 5-bit (4 hi-bits packed) | 22 | 5.5 |
| `Q5_1` | f16 scale + f16 min + 32× 5-bit | 24 | 6.0 |
| `Q8_0` | f16 scale + 32× 8-bit | 34 | 8.5 |

Reconstruction for `Q4_0`: `w[i] = scale * (q[i] - 8)`. For the `_1` variants
that carry a min: `w[i] = scale * q[i] + min`.

### K-quants — superblock of 256 weights

The "K" family (introduced for better quality at the same size) groups **256**
weights into a superblock, subdivided into sub-blocks each with its own quantized
scale/min, plus a super-scale. This two-level scaling is why K-quants beat the
legacy quants at equal bit budgets.

| Type | Bits/weight | Typical use |
|---|---|---|
| `Q2_K` | ~2.56 | smallest, lossy |
| `Q3_K` | ~3.44 | small |
| `Q4_K` | ~4.5 | **the default sweet spot** (Q4_K_M) |
| `Q5_K` | ~5.5 | higher quality |
| `Q6_K` | ~6.56 | near-lossless |
| `Q8_K` | ~8 | intermediate / activations |

The `_S` / `_M` / `_L` suffixes you see in filenames (e.g. `Q4_K_M`) are not
distinct `ggml_type`s — they are *mixes*. A "medium" file uses a higher-bit quant
for the more sensitive tensors (e.g. `attn_v`, `ffn_down`) and a lower-bit quant
for the rest. That mix is recorded in `general.file_type`.

### IQ-quants and others

The `IQ*` types (IQ2_XXS … IQ4_XS) use an **importance matrix** computed from
calibration data to decide where to spend bits, achieving usable quality at 2–3
bits/weight. `BF16` and `F16` store full half-precision; `MXFP4` is a newer
4-bit microscaling float block format.

The practical takeaway: a tensor's `ggml_type` tells the kernel exactly how to
unpack each block back into floats on the fly during matmul. The GPU never sees
"a Q4_K tensor" as floats in memory — it dequantizes block-by-block inside the
kernel.

---

## 7. How a loader reads a GGUF file (end to end)

```
1. open + mmap the file
2. read magic  -> assert == "GGUF"
3. read version, tensor_count, metadata_kv_count
4. loop metadata_kv_count times:
       read key (string)
       read value_type
       read value (dispatch on value_type; arrays loop)
   -> now you have architecture, hyperparameters, tokenizer
5. ALIGNMENT = general.alignment (default 32)
6. loop tensor_count times:
       read name, n_dimensions, dimensions[], type, offset
   -> now you have the shape/type/location of every weight
7. pos = align_offset(current position)   # skip padding
   tensor_data_start = pos
8. for each tensor: data ptr = tensor_data_start + info.offset
       (no copy — just a pointer into the mmap)
9. build the compute graph from general.architecture + hyperparameters
10. for inference: decide n_gpu_layers, upload those tensors' blocks to VRAM
```

Steps 1–8 are cheap and identical across architectures — that's the genius of a
self-describing format. Step 9 is the only architecture-specific code.

---

## 8. Filename convention

GGUF files follow `<BaseName>-<SizeLabel>-<FineTune>-<Version>-<Encoding>-<Type>-<Shard>.gguf`:

- `Mixtral-8x7B-v0.1-Q4_0.gguf` → Mixtral, 8 experts × 7B, Q4_0.
- `Hermes-2-Pro-Llama-3-8B-v1.0-F16.gguf` → 8B, full F16.
- `Grok-100B-v1.0-Q4_0-00003-of-00009.gguf` → shard 3 of 9 of a 100B model.

Large models are **sharded** into multiple files (`NNNNN-of-NNNNN`); the loader
stitches them by reading each shard's tensor table.

---

## 9. What to do next (study path)

1. **Inspect a real file.** `gguf-dump` (in llama.cpp) or `gguf` Python package prints the header, every metadata KV, and the tensor table. Do this on a small model (e.g. a 1–3B Q4_K_M) and match each field to the structs above.
2. **Write a minimal parser** that prints architecture + hyperparameters + tensor list, without any inference. ~150 lines in Python.
3. **Dequantize one tensor by hand** — pick a `Q4_0` tensor, read a block, apply `scale * (q - 8)`, and confirm against `gguf` library output.
4. **Then** move to the forward pass: build attention/MLP from the hyperparameters and stream tokens.

---

### Source

GGML GGUF specification — https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
(structs, enums, metadata keys, and naming convention quoted/derived from the spec).
Quant block sizes are from the GGML implementation conventions (block size 32 for
legacy quants, 256-weight superblocks for K-quants).
