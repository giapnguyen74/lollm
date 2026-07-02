"""
hook.py — shared infra: inject a user function into a *populated* decoder model.

Generic across every lollm decoder family. It is the read/write seam for the
modification study (see src/modification/): a family's `modeling` calls
`self.hook_fn(x, site)` right after each residual WRITE-BACK (every `x = x + …`), and
`attach` sets that per-layer `hook_fn` slot on a loaded model. That reaches every point
where x re-enters the stream — for capture AND intervention — which a
`register_forward_hook` cannot do (it can't WRITE the internal post-attention residual,
a local inside `forward`).

The family contract (what a model must implement to be hookable):
  • each `DecoderLayer.__init__` sets `self.hook_fn = None`;
  • each `DecoderLayer.forward` calls `x = self.hook_fn(x, site)` (guarded by
    `if self.hook_fn:`) after each residual write-back, passing the site name;
  • the family declares its `SITES` tuple (the write-back names) next to its modeling.
`attach` only requires the slot to exist (a `hasattr` guard). Whether `forward` actually
calls it is confirmed by a family's real-run test (e.g. src/hook_test.py: a "fired N times"
count — 0 would mean the slot exists but forward never invokes it).

Usage — build the model the normal way, THEN inject (nothing here touches loading):

    model = <family>.load(raw_config, weights, fmt, device, dtype)   # as usual
    with attach(model, my_fn):
        logits, _ = model(input_ids)          # my_fn fires at every write-back of every layer
    # slots cleared on exit (or call handle.remove())

`attach` injects on every layer; the function FILTERS ITSELF via `ctx` (return None to
no-op) — simpler than passing site/layer lists. `my_fn(act, ctx) -> Tensor | None`:
  • return None  → OBSERVE only (capture), or skip this trigger. x is kept unchanged.
    (Copy what you stash — `act` aliases the live tensor; e.g. act[:, -1, :].detach().clone().)
  • return a Tensor → REPLACE the residual at that write-back (steer / ablate).
`ctx` (HookContext) says WHEN/WHERE it fired, so one fn can scale α per layer (note §3) or
act on a chosen site/layer (`if ctx.site == "out" and ctx.layer_idx == 12`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:                      # hook.py calls no torch functions — it only stores a
    import torch                       # slot and passes tensors through — so torch is a
    HookFn = Callable[["torch.Tensor", "HookContext"], "Optional[torch.Tensor]"]


# ─────────────────────────────── the trigger context ──────────────────────────────

@dataclass
class HookContext:
    """Handed to the user function on every trigger — "when/where am I firing?". Built
    here from a per-layer closure (layer_idx/n_layers) plus the live residual.

    Deliberately family-agnostic: only fields every decoder shares. A family that varies
    per layer (gemma: sliding/global; qwen3_5: attn/gdn; moe: dense/moe) can carry that in
    the `site` string it passes, or extend this later."""
    layer_idx: int            # which decoder block (0-based)
    n_layers: int             # total blocks — lets the fn reason about relative depth (§3)
    site: str                 # residual write-back that fired (family-defined, e.g. "out")
    seqlen: int               # number of token positions in THIS forward call (x.shape[1])


# ─────────────────────────────────── attach ───────────────────────────────────────

def attach(model, fn) -> "Handle":
    """Inject `fn` at every residual write-back of every layer of a populated decoder
    model. No site/layer arguments — the function self-filters via `ctx` (return None to
    no-op). Returns a removable, context-managed Handle."""
    _require_hookable(model)
    blocks = model.model.layers
    n = len(blocks)
    for i, block in enumerate(blocks):
        block.hook_fn = _wrap(fn, i, n)     # per-layer: carries layer_idx / n_layers
    return Handle(blocks)


def _wrap(fn, layer_idx: int, n_layers: int):
    """Adapt the user fn to modeling's minimal `hook_fn(x, site)` call: supply the layer
    metadata (closed over here), read seqlen off the live residual, build the context, run
    the fn, and let a returned tensor replace the residual (None keeps it)."""
    def hook_fn(x, site):
        ctx = HookContext(layer_idx=layer_idx, n_layers=n_layers, site=site,
                          seqlen=x.shape[1])
        out = fn(x, ctx)
        return x if out is None else out
    return hook_fn


# ─────────────────────────────────── handle ───────────────────────────────────────

class Handle:
    """Owns the attached slots so their lifecycle is explicit — no leaked state. Use as a
    context manager, or call .remove() when done."""

    def __init__(self, blocks):
        self._blocks = list(blocks)

    def remove(self) -> None:
        for b in self._blocks:
            b.hook_fn = None
        self._blocks = []

    def __enter__(self) -> "Handle":
        return self

    def __exit__(self, *exc) -> bool:
        self.remove()
        return False                    # never swallow exceptions


# ─────────────────────────────────── internals ────────────────────────────────────

def _require_hookable(model) -> None:
    """hasattr guard: the model must expose `model.model.layers`, and those layers must
    implement the `hook_fn` slot. (Note: this confirms the slot EXISTS — `self.hook_fn =
    None` still makes hasattr True — not that `forward` calls it; a family's real-run test
    confirms the seam actually fires.)"""
    if not (hasattr(model, "model") and hasattr(model.model, "layers")):
        raise TypeError(
            "attach expects a decoder model exposing model.model.layers. Got "
            f"{type(model).__name__}. Build it with <family>.load(...) first.")
    blocks = model.model.layers
    if len(blocks) == 0:
        raise ValueError("model has no layers")
    if not hasattr(blocks[0], "hook_fn"):
        raise TypeError(
            f"{type(blocks[0]).__name__} has no `hook_fn` slot — this family does not "
            "implement the hook seam. Add `self.hook_fn = None` in DecoderLayer.__init__ "
            "and call `x = self.hook_fn(x, site)` after each residual write-back in forward.")
