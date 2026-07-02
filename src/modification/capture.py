"""
capture.py — the read-side bridge: turn a loaded lollm model + its hook seam into the
`HookedModel` protocol that extract.py consumes.

extract.py is pure math (difference-of-means, subspace) and knows nothing about models.
Its one dependency is the `HookedModel` protocol: `.sites` + `.capture(prompts, sites,
position) -> {site: [B, d]}`. `HookCapture` here IS that implementation, built on the
shared hook seam (src/hook.py::attach): it attaches a capture fn, runs one forward per
prompt, and harvests the last-token residual at every (layer, write-back) site.

Family-agnostic by injection — the caller passes the shared `attach` and the family's
per-layer site names, so this file never imports a specific family:

    from hook import attach                 # shared seam (src/hook.py)
    from qwen2.modeling_qwen2 import SITES   # family-declared write-back names
    cap = HookCapture(model, tokenizer, attach, SITES, device)
    ex  = Extractor(cap).collect(loader)     # extract.py, unchanged

Site keys are "L{layer_idx}.{site}" (e.g. "L12.out") — the composite name extract.py's
`direction("L12.out")` selects on. Records EVERY site in the one forward pass (README D4:
record-all, filter at read time); `capture(sites=...)` just returns the requested subset.
Position defaults to -1 (last token, the generation point). Batch size is 1 per forward
(no padding); batching is a later optimization.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch


class HookCapture:
    """Implements extract.HookedModel over a family hook. `attach(model, fn)` must set a
    per-layer slot that calls `fn(act, ctx)` at each residual write-back, where `ctx` has
    `.layer_idx` and `.site` (the src/hook.py contract)."""

    def __init__(self, model, tokenizer, attach, site_names: Sequence[str],
                 device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.attach = attach
        self.device = device
        n_layers = len(model.model.layers)
        # every (layer, write-back) pair this model exposes — the full record-all set.
        self.sites = [f"L{i}.{s}" for i in range(n_layers) for s in site_names]

    @torch.no_grad()
    def capture(self, prompts: Sequence[str], sites: Sequence[str] | None = None,
                position: int = -1) -> Mapping[str, torch.Tensor]:
        """Run each prompt through one prefill forward with a capture fn attached; return
        {site: [N, d]} of last-token residuals for the requested `sites` (default: all)."""
        want = set(sites) if sites is not None else set(self.sites)
        acc: dict[str, list[torch.Tensor]] = {s: [] for s in want}

        for prompt in prompts:
            store: dict[str, torch.Tensor] = {}

            def cap(act, ctx):
                # act: (B=1, T, d) residual at this write-back. Keep the last-token vector.
                key = f"L{ctx.layer_idx}.{ctx.site}"
                if key in want:
                    store[key] = act[:, position, :].detach().float().to("cpu")[0]   # [d]
                return None                      # observe only — never modify the stream

            ids = self.tokenizer.apply_chat(prompt)          # chat template (README D3)
            input_ids = torch.tensor([ids], device=self.device)
            with self.attach(self.model, cap):
                self.model(input_ids)

            missing = want - store.keys()
            if missing:
                raise KeyError(f"capture missed sites {sorted(missing)} for a prompt "
                               f"(hook did not fire there)")
            for k in want:
                acc[k].append(store[k])

        # stack per site → [N, d]; return in the requested order.
        order = list(sites) if sites is not None else self.sites
        return {s: torch.stack(acc[s]) for s in order}
