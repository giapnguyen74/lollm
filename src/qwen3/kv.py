"""
qwen3/kv.py — the family KV cache (lightweight).

One array indexed by layer, each slot the growing (K, V) for that layer, plus a
`seen_tokens` position counter. The model goes through these methods instead of poking the
list, which leaves room to swap in an advanced KV policy (paged/quantized) later without
touching the model. (Every layer is full-attention here, so every slot is a KV tuple —
simpler than the hybrid qwen3_5.)
"""

from __future__ import annotations

import torch


class Qwen3Cache:
    def __init__(self, n_layers: int):
        self.layers = [None] * n_layers      # per-layer (K, V)
        self.seen_tokens = 0

    def append_kv(self, layer, k, v):
        """Append this step's k,v (post-RoPE, pre-GQA-expansion); return the full (K, V)."""
        s = self.layers[layer]
        if s is None:
            self.layers[layer] = (k, v)
        else:
            self.layers[layer] = (torch.cat([s[0], k], dim=2), torch.cat([s[1], v], dim=2))
        return self.layers[layer]

    def read_kv(self, layer):
        return self.layers[layer]

    def advance(self, n):
        self.seen_tokens += n

    def clone(self):
        new = Qwen3Cache(len(self.layers))
        new.seen_tokens = self.seen_tokens
        new.layers = [None if s is None else (s[0].clone(), s[1].clone()) for s in self.layers]
        return new

    def nbytes(self):
        return sum(t.numel() * t.element_size() for s in self.layers if s for t in s)
