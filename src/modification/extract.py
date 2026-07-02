"""
extract.py — derive a behavior DIRECTION (vector) or SUBSPACE (space) from activations.

Study module — see modification/README.md, Step 2. Generic extraction with three
injected concerns kept apart:

  • a HOOKING MODEL — anything that can run prompts and return the residual activation
    at named sites (the READ side of the seam; the intervention/WRITE side is still
    open, see README D2). Extraction needs only reads, so it moves first.
  • a DATALOADER — yields batches of labelled prompts (A = behavior present,
    B = behavior absent). Extraction never tokenizes or wraps templates itself; that
    is an encode-time concern the hooking model owns (README D3).
  • the MATH — difference-of-means for a vector (note §5), SVD of paired differences
    or a stack of contrast axes for a space (note §8).

The rank correction (note §8) is load-bearing here: a two-class mean difference is
INHERENTLY rank-1 — you cannot conjure a multi-dimensional subspace from two class
means alone. So `direction()` needs no pairing, but `subspace()` REQUIRES paired data
(row i of A matched to row i of B) and hard-fails without it — we never fabricate rank
we don't have (repo rule: hard-fail, never guess). The multi-axis route (note §8
Option A) is `space_from_axes`, which stacks several already-extracted directions.

Capture strategy (decided).
  • Record ALL sites in ONE forward pass, then filter at READ time. The forward pass is
    the only expensive step; every downstream analysis (a direction at each layer, the
    layer×α sweep §6, the cross-layer consistency check §7, swapping diff-of-means for a
    probe §5) is cheap linear algebra over the SAME activations. So `collect` captures
    everything the model exposes; `direction(site)` / `subspace(site)` select a subset
    afterward. Restricting capture up front would force a re-run to look elsewhere.
  • Position: LAST TOKEN only (position=-1) — the generation point under
    add_generation_prompt (note §5). All-positions capture is a deliberate future opt-in
    (it multiplies the cache by sequence length); not recorded here.
  • Persistence: kept IN MEMORY for now. A run-once, save-to-disk activation cache
    (so extract becomes model-free and reused across experiments) is a later refactor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, Sequence, runtime_checkable

import torch


# ───────────────────────────── the read-side contract ─────────────────────────────
# extract.py depends on this Protocol, not on any concrete hooking implementation, so it
# is testable with a stub and stays model-free. The real implementation is
# modification/capture.py::HookCapture, which bridges a loaded lollm model + the shared
# hook seam (src/hook.py::attach) into this interface. Sites are composite names
# "L{layer_idx}.{site}" (e.g. "L12.out"); direction()/subspace() select on them.

@runtime_checkable
class HookedModel(Protocol):
    sites: Sequence[str]                     # residual sites this model can read (README D1)

    def capture(
        self, prompts: Sequence[str], sites: Sequence[str], position: int = -1
    ) -> Mapping[str, torch.Tensor]:
        """Run `prompts`; return {site: activations [N, d]} taken at token `position`
        (default -1 = last token, the generation point under add_generation_prompt)."""
        ...


# ───────────────────────────────── result types ──────────────────────────────────

@dataclass
class Direction:
    """A single behavior axis at one site (note §5). `v_hat` is unit-norm; `raw_norm`
    is ‖μ_A − μ_B‖, kept as the scale reference for tuning α (note §3)."""
    site: str
    v_hat: torch.Tensor          # [d], ‖·‖ = 1
    raw_norm: float              # ‖μ_A − μ_B‖
    n_a: int
    n_b: int


@dataclass
class Subspace:
    """A behavior subspace at one site from PAIRED differences (note §8).

    `mean_diff` is the rank-1 diff-of-means direction (the primary axis). `spread` are
    the singular vectors of the *centered* paired differences — the axes the behavior
    varies along around that mean. `singulars` is the full spectrum, so k can be chosen
    at the elbow downstream (note §8, "choosing k"). For an erasure basis you generally
    want BOTH the mean direction and the leading spread axes: use `basis(k)`.
    """
    site: str
    mean_diff: torch.Tensor      # [d], the rank-1 primary direction (not unit)
    spread: torch.Tensor         # [r, d], orthonormal spread axes (rows)
    singulars: torch.Tensor      # [r], singular values of the centered paired diffs
    n_pairs: int

    def basis(self, k: int) -> torch.Tensor:
        """Orthonormal [k+1, d] basis = the mean direction plus the top-`k` spread axes,
        orthonormalized together (QR). This is what you project out to erase the behavior
        (note §8, P = I − VₖVₖᵀ). k=0 recovers just the mean direction."""
        if k < 0:
            raise ValueError(f"k must be ≥ 0, got {k}")
        if k > self.spread.shape[0]:
            raise ValueError(
                f"k={k} exceeds available spread rank {self.spread.shape[0]} at {self.site}")
        rows = torch.cat([self.mean_diff.unsqueeze(0), self.spread[:k]], dim=0)   # [k+1, d]
        q, _ = torch.linalg.qr(rows.T)                                            # [d, k+1]
        return q.T                                                                # [k+1, d]


# ─────────────────────────────────── extractor ────────────────────────────────────

class Extractor:
    """Collect labelled activations across a dataloader, then compute a vector or a space.

    Usage:
        ex = Extractor(hooking_model, sites=["L12.out"]).collect(loader)
        v  = ex.direction("L12.out")          # diff-of-means vector (note §5)
        S  = ex.subspace(site="L12.out")      # paired-difference subspace (note §8)
    """

    def __init__(
        self,
        model: HookedModel,
        sites: Sequence[str] | None = None,     # capture set; default = ALL the model exposes
        position: int = -1,                     # last token (note §5); all-positions is future
        pos_label: str = "A",
        neg_label: str = "B",
        dtype: torch.dtype = torch.float32,     # note §5: average in float32
    ):
        # By default we RECORD ALL sites in one pass and filter at read time (see the
        # module docstring). Passing `sites` restricts the capture set — rarely needed,
        # and it forfeits looking at other sites without re-running the model.
        self.model = model
        self.sites = list(sites) if sites is not None else list(model.sites)
        if not self.sites:
            raise ValueError("no sites to extract at (model exposes none, and none given)")
        self.position = position
        self.pos_label, self.neg_label = pos_label, neg_label
        self.dtype = dtype
        self._reset()

    def _reset(self) -> None:
        # Per-example activations kept on CPU so both the vector (means only) and the
        # space (per-example paired diffs) can be computed from the same collection.
        self._acts: dict[str, list[torch.Tensor]] = {s: [] for s in self.sites}
        self._labels: list[str] = []       # label per example, shared across sites
        self._pairs: list[object] = []      # pair_id per example (or None)

    # ── collection ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def collect(self, dataloader: Iterable) -> "Extractor":
        """Populate the activation buffers from `dataloader`. Chainable (returns self).

        Each batch is a mapping with `prompt` (list[str]) and `label` (list[str]), plus
        optional `pair_id` (list) — the schema from README Step 1. A list of per-example
        dicts is also accepted.
        """
        for batch in dataloader:
            # 1. UNPACK this batch into aligned prompts / labels / pair_ids.
            prompts, labels, pairs = _unpack(batch)
            self._validate_labels(labels)
            # 2. CAPTURE the residual at every site in ONE forward pass (record-all,
            #    filter-at-read-time). Re-selecting sites later needs no re-run.
            caps = self.model.capture(prompts, self.sites, self.position)
            _require_sites(caps, self.sites)
            # 3. ACCUMULATE per example, cast to float32 on CPU, grouped later by label.
            for s in self.sites:
                a = caps[s].to(self.dtype).to("cpu")            # [B, d]
                if a.shape[0] != len(prompts):
                    raise ValueError(
                        f"site {s!r} returned {a.shape[0]} rows for {len(prompts)} prompts")
                self._acts[s].extend(a[i] for i in range(a.shape[0]))
            self._labels.extend(labels)
            self._pairs.extend(pairs)
        if not self._labels:
            raise ValueError("dataloader yielded no examples")
        return self

    # ── the vector (note §5) ──────────────────────────────────────────────────────
    def direction(self, site: str | None = None):
        """Difference-of-means direction. `site=None` → dict over all sites."""
        if site is None:
            return {s: self._direction_at(s) for s in self.sites}
        return self._direction_at(site)

    def _direction_at(self, site: str) -> Direction:
        A, B = self._split_by_label(site)
        # r = μ_A − μ_B ; normalize to v̂ ; keep ‖r‖ as the scale reference (note §3, §5).
        r = A.mean(0) - B.mean(0)
        norm = r.norm()
        if float(norm) == 0.0:
            raise ValueError(
                f"zero mean-difference at {site!r}: class means coincide — the sets do "
                f"not separate here (wrong site, or A/B not actually contrasting)")
        return Direction(site, r / norm, float(norm), A.shape[0], B.shape[0])

    # ── the space (note §8) ─────────────────────────────────────────────────────
    def subspace(self, site: str | None = None, k: int | None = None):
        """Subspace from PAIRED per-example differences (note §8). Requires pair_ids on
        every example — a two-class mean difference alone is rank-1. `site=None` → dict."""
        if site is None:
            return {s: self._subspace_at(s, k) for s in self.sites}
        return self._subspace_at(site, k)

    def _subspace_at(self, site: str, k: int | None) -> Subspace:
        if any(p is None for p in self._pairs):
            raise ValueError(
                "subspace() requires paired data (a pair_id on every example). A two-class "
                "mean difference is inherently rank-1 (note §8); a genuine subspace must come "
                "from paired per-example differences, or stack several directions via "
                "space_from_axes(). Provide pair_id, or use direction().")
        # 1. Build matched A−B differences over complete pairs (rows aligned by pair_id).
        a_by, b_by = {}, {}
        for act, lab, pid in zip(self._acts[site], self._labels, self._pairs):
            (a_by if lab == self.pos_label else b_by).setdefault(pid, act)
        pids = sorted(set(a_by) & set(b_by), key=str)
        if not pids:
            raise ValueError(f"no complete A/B pairs at {site!r}")
        D = torch.stack([a_by[p] - b_by[p] for p in pids])       # [P, d]
        # 2. mean(D) is the diff-of-means direction; SVD of the CENTERED diffs describes
        #    the spread AROUND it — this is the part that carries extra rank (note §8).
        mean_diff = D.mean(0)                                     # [d]
        _, S, Vt = torch.linalg.svd(D - mean_diff, full_matrices=False)
        r = Vt.shape[0] if k is None else min(k, Vt.shape[0])
        return Subspace(site, mean_diff, Vt[:r], S, len(pids))

    # ── internals ────────────────────────────────────────────────────────────────
    def _split_by_label(self, site: str):
        if site not in self._acts:
            raise KeyError(f"site {site!r} not collected; have {list(self._acts)}")
        acts = self._acts[site]
        if not acts:
            raise ValueError("nothing collected — call collect(dataloader) first")
        stacked = torch.stack(acts)                              # [N, d]
        idx_a = [i for i, l in enumerate(self._labels) if l == self.pos_label]
        idx_b = [i for i, l in enumerate(self._labels) if l == self.neg_label]
        if not idx_a or not idx_b:
            raise ValueError(
                f"need both classes: got {len(idx_a)} '{self.pos_label}' and "
                f"{len(idx_b)} '{self.neg_label}' examples")
        return stacked[idx_a], stacked[idx_b]

    def _validate_labels(self, labels: Sequence[str]) -> None:
        allowed = {self.pos_label, self.neg_label}
        bad = {l for l in labels if l not in allowed}
        if bad:
            raise ValueError(f"unexpected labels {bad}; expected only {allowed}")


# ───────────────────────── multi-axis space (note §8 Option A) ─────────────────────

def space_from_axes(directions: Sequence[Direction | torch.Tensor], k: int | None = None
                     ) -> torch.Tensor:
    """Build an orthonormal basis from several already-extracted contrast axes — the
    "most practical" subspace route (note §8 Option A): one direction per related
    contrast (e.g. distinct refusal categories, or the behavior across templates),
    stacked and orthonormalized. Returns [k, d]; inspect against the count of axes to
    judge effective rank (near-parallel axes ⇒ effectively rank-1).
    """
    vecs = [d.v_hat if isinstance(d, Direction) else d for d in directions]
    if len(vecs) < 2:
        raise ValueError("need ≥ 2 axes to form a space; one axis is just a direction")
    R = torch.stack(vecs).float()                                # [m, d]
    R = R / R.norm(dim=-1, keepdim=True)
    _, _, Vt = torch.linalg.svd(R, full_matrices=False)          # right vectors = basis
    r = Vt.shape[0] if k is None else min(k, Vt.shape[0])
    return Vt[:r]


# ─────────────────────────────── batch unpacking ──────────────────────────────────

def _unpack(batch):
    """Accept a mapping of lists ({'prompt':[...], 'label':[...], 'pair_id':[...]}) or a
    list of per-example dicts; return (prompts, labels, pair_ids) with pair_ids defaulted
    to None when absent."""
    if isinstance(batch, Mapping):
        prompts, labels = list(batch["prompt"]), list(batch["label"])
        pairs = list(batch["pair_id"]) if "pair_id" in batch else [None] * len(prompts)
    else:  # iterable of per-example records
        recs = list(batch)
        prompts = [r["prompt"] for r in recs]
        labels = [r["label"] for r in recs]
        pairs = [r.get("pair_id") for r in recs]
    if not (len(prompts) == len(labels) == len(pairs)):
        raise ValueError("prompt / label / pair_id lengths disagree in a batch")
    return prompts, labels, pairs


def _require_sites(caps: Mapping[str, torch.Tensor], sites: Sequence[str]) -> None:
    missing = [s for s in sites if s not in caps]
    if missing:
        raise KeyError(f"hooking model did not return sites {missing}; returned {list(caps)}")
