"""
gemma4/kv.py — the family KV cache.

Same growing-(K,V)-per-layer shape as gemma2/gemma3, with one Gemma4 wrinkle: only
**non-shared** attention layers append here. The top `num_kv_shared_layers` layers
compute no K/V at all — within a forward they reuse the donor layer's K/V via the
model's transient `shared_kv` dict, so their slots here stay empty. Local (sliding)
layers still cache the FULL K/V; the window lives in the attention mask.
"""

from __future__ import annotations

import torch


class Gemma4Cache:
    def __init__(self, n_layers: int):
        self.layers = [None] * n_layers
        self.seen_tokens = 0

    def append_kv(self, layer, k, v):
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
        new = Gemma4Cache(len(self.layers))
        new.seen_tokens = self.seen_tokens
        new.layers = [None if s is None else (s[0].clone(), s[1].clone()) for s in self.layers]
        return new

    def nbytes(self):
        return sum(t.numel() * t.element_size() for s in self.layers if s for t in s)
