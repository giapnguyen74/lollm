"""
qwen3_5/kv.py — the family cache (lightweight).

One array, indexed by layer, holding that layer's per-step state:
  • full-attention layers → a growing (K, V) tuple (the standard KV cache),
  • linear (GDN) layers   → a fixed-size (conv_state, recurrent_state) tuple,
  • seen_tokens           → the RoPE position offset.

Each layer is exclusively one kind, so a single `self.layers` slot per layer is enough. The
only thing that matters is that the model goes through these methods rather than poking
`self.layers` directly — that leaves room to swap the KV storage for something fancier
(paged/quantized) later without touching the model.
"""

from __future__ import annotations

import torch


class Qwen3_5Cache:
    def __init__(self, n_layers: int):
        self.layers = [None] * n_layers      # per-layer: (K,V) for full, (conv,rec) for linear
        self.seen_tokens = 0

    # ── full-attention layers ──
    def append_kv(self, layer, k, v):
        """Append this step's k,v (post-RoPE, pre-GQA-expansion); return the full (K, V)."""
        s = self.layers[layer]
        if s is None:
            self.layers[layer] = (k, v)
        else:
            self.layers[layer] = (torch.cat([s[0], k], dim=2), torch.cat([s[1], v], dim=2))
        return self.layers[layer]

    def read_kv(self, layer):
        return self.layers[layer]            # (K, V) or None

    # ── linear (GDN) layers ──
    def linear_state(self, layer):
        return self.layers[layer] or (None, None)

    def set_linear_state(self, layer, conv, rec):
        self.layers[layer] = (conv, rec)

    # ── bookkeeping ──
    def advance(self, n):
        self.seen_tokens += n

    def clone(self):
        """Snapshot for speculative decoding (linear state isn't invertible, so we copy)."""
        new = Qwen3_5Cache(len(self.layers))
        new.seen_tokens = self.seen_tokens
        new.layers = [None if s is None else (s[0].clone(), s[1].clone()) for s in self.layers]
        return new

    def nbytes(self):
        """Logical bytes held (metadata only — no device sync). For monitoring growth."""
        return sum(t.numel() * t.element_size() for s in self.layers if s for t in s)
