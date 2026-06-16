"""
loader.py — shared infra. Fetch a model and hand it over *raw*.

Resolves the spec (local dir/.gguf, HF repo, or `repo:QUANT` for gguf), reads the
raw config/metadata, loads weights **keyed by the file's own names**, builds a
uniform tokenizer, and reads generation defaults. It does NOT rename tensors and
knows no architecture — the family maps names. Supports HF safetensors and GGUF.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch


from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

                                                      
from safetensors.torch import load_file

from dequant import dequantize
from gguf_reader import GGUFReader
from progress import bar
from tokenization import GGUFTokenizer, HFTokenizer, SPMTokenizer

# what we pull for a safetensors repo
_HF_PATTERNS = [
    "config.json", "generation_config.json", "*.safetensors",
    "*.safetensors.index.json", "tokenizer*", "*.model", "special_tokens_map.json",
]


@dataclass
class Loaded:
    model_type: str        # probed → router
    fmt: str               # "hf" | "gguf"
    raw_config: dict       # config.json (hf) or GGUF metadata, untouched
    weights: dict          # raw tensor name → tensor (gguf: dequantized eagerly)
    tokenizer: object      # uniform: encode/decode/apply_chat/chat_template
    gen_meta: dict         # {"stop_ids": [...], "sampling": {...}}


def resolve(spec: str, token: str | None = None) -> str:
    """Local path, HF repo, or 'repo:QUANT' (gguf) → a local path."""
    if os.path.exists(spec):
        return spec
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if ":" in spec:                                  # gguf 'repo:QUANT'
        repo, quant = spec.split(":", 1)
        files = [f for f in list_repo_files(repo, token=token) if f.lower().endswith(".gguf")]
        matches = sorted(f for f in files if quant.lower() in f.lower())
        if not matches:
            raise FileNotFoundError(f"no .gguf matching '{quant}' in {repo}. Available: {files}")
        shard0 = [m for m in matches if "00001-of" in m]
        chosen = shard0[0] if shard0 else matches[0]
        print(f"[downloading {repo} :: {chosen}]", flush=True)
        return hf_hub_download(repo, chosen, token=token)

    # safetensors repo. One snapshot_download fetches whatever's missing and reuses the cache
    # for the rest — the hub no-ops cached files. Bars are silenced via HF_HUB_DISABLE_PROGRESS_BARS
    # (set above), so this stays quiet; we print one concise line ourselves.
    kw = dict(token=token, allow_patterns=_HF_PATTERNS)
    print(f"[resolving {spec} (downloading if needed)]", file=sys.stderr, flush=True)
    return snapshot_download(spec, **kw)


def _load_hf(path: str) -> Loaded:
    raw_config = json.load(open(os.path.join(path, "config.json")))
    weights: dict = {}
    shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    report = bar("reading shards")
    for i, shard in enumerate(shards):
        weights.update(load_file(shard))
        report(i + 1, len(shards))

    tok = HFTokenizer(path)
    stop_ids, sampling = list(tok.eos_ids), {}
    gc_path = os.path.join(path, "generation_config.json")
    if os.path.exists(gc_path):
        gc = json.load(open(gc_path))
        eos = gc.get("eos_token_id")
        if eos is not None:
            stop_ids = eos if isinstance(eos, list) else [eos]
        sampling = {k: gc[k] for k in
                    ("temperature", "top_p", "top_k", "repetition_penalty") if k in gc}

    model_type = raw_config.get("model_type")
    if model_type is None:                         # fail loud — never guess the family
        raise ValueError("config.json has no 'model_type'; cannot route to a family")
    return Loaded(model_type, "hf", raw_config, weights, tok,
                  {"stop_ids": stop_ids, "sampling": sampling})


def _load_gguf(path: str) -> Loaded:
    reader = GGUFReader(path)
    meta = reader.metadata

    # eager dequant → plain tensors by the file's own raw names.
    # Dequant to fp16 (not fp32): the model runs fp16 on GPU anyway, so this is
    # lossless vs the final weights and HALVES the CPU dequant buffer (the big peak).
    weights = {}
    for name, info in reader.tensors.items():
        n = int(np.prod(info.torch_shape))
        flat = dequantize(info.ggml_type, reader.raw_tensor(name), n)
        weights[name] = torch.from_numpy(flat.reshape(info.torch_shape).astype(np.float16))

    # Tokenizer by type (both hand-written from the GGUF vocab, no external deps):
    #   gpt2  → byte-level BPE   (Qwen, Llama-3)
    #   llama → SentencePiece    (Gemma, Llama-2, Mistral-SPM)
    tmodel = meta.get("tokenizer.ggml.model", "gpt2")
    if tmodel == "gpt2":
        tok = GGUFTokenizer(meta)
    elif tmodel in ("llama", "gemma"):
        tok = SPMTokenizer(meta)
    else:
        raise NotImplementedError(f"tokenizer.ggml.model='{tmodel}' not supported "
                                  f"(have: gpt2 BPE, llama SentencePiece)")

    return Loaded(meta["general.architecture"], "gguf", meta, weights, tok,
                  {"stop_ids": list(tok.eos_ids), "sampling": {}})   # gguf carries no sampling


def load(spec: str, token: str | None = None) -> Loaded:
    path = resolve(spec, token)
    if os.path.isfile(path) and path.lower().endswith(".gguf"):
        return _load_gguf(path)
    return _load_hf(path)
